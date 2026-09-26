"""Hugging Face Hub as the persistent experiment store.

Guarantees
----------
* **Verified commits.** After every commit the remote file list is re-read and each
  file is checked against the local bytes: LFS/Xet files by sha256, small files by
  their git blob sha1, both by size. A commit that does not verify raises.
* **Atomic groups.** A checkpoint and its pointer (LATEST.json) land in ONE commit, so
  a reader never sees a pointer to a half-uploaded checkpoint.
* **Retries with backoff** on network errors, 5xx and 429.
* **Optimistic concurrency** for shared JSON (registry, claims): read-modify-write
  pinned to the parent commit; a concurrent writer causes a clean retry, not a lost
  update. Several Kaggle accounts can therefore share one repo.
* **Commit budget.** HF allows roughly 128 commits/hour/repo; a trailing-hour counter
  throttles before the server does.
* The token is resolved from Kaggle Secrets or the environment and is never printed.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable, Optional

from src.pipeline.env import log_event

_TOKEN_HELP = (
    "No Hugging Face token found.\n"
    "  On Kaggle: Add-ons -> Secrets -> add a secret named HF_TOKEN (a WRITE token from\n"
    "  https://huggingface.co/settings/tokens), tick 'Attached' for this notebook, then\n"
    "  re-run. Note: pushing a notebook with `kaggle kernels push` detaches secrets --\n"
    "  re-attach it in the notebook editor afterwards.\n"
    "  Elsewhere: export HF_TOKEN=... in the environment.")


class HubError(RuntimeError):
    pass


def resolve_hf_token() -> str:
    """Kaggle Secret HF_TOKEN -> env HF_TOKEN -> cached `hf auth login` token."""
    token = None
    try:
        from kaggle_secrets import UserSecretsClient  # only exists on Kaggle
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001 - not on Kaggle, or secret not attached
        token = None
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:  # noqa: BLE001
            token = None
    if not token:
        raise HubError(_TOKEN_HELP)
    os.environ["HF_TOKEN"] = token        # child processes (DataLoader workers) inherit it
    return token


def validate_token(token: str, repo_id: str) -> dict:
    """whoami + a write-permission sanity check against the target namespace."""
    from huggingface_hub import HfApi
    try:
        who = HfApi(token=token).whoami()
    except Exception as e:  # noqa: BLE001
        raise HubError(f"HF token rejected by whoami(): {type(e).__name__}. "
                       "Create a new WRITE token and update the HF_TOKEN secret.") from None
    name = who.get("name")
    orgs = [o.get("name") for o in who.get("orgs", [])]
    namespace = repo_id.split("/")[0]
    access = (who.get("auth") or {}).get("accessToken") or {}
    role = access.get("role")
    if namespace not in [name, *orgs]:
        raise HubError(f"token belongs to '{name}' but the experiment repo is in "
                       f"namespace '{namespace}'. Use a token of that account or change "
                       "CONFIG['project']['hf_repo'].")
    if role == "read":
        raise HubError("HF token is READ-only; checkpoints cannot be uploaded. "
                       "Create a WRITE (or fine-grained repo.write) token.")
    return {"user": name, "role": role, "namespace_ok": True}


# ----------------------------------------------------------------- hashing
def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git_blob_sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------- store
class HubStore:
    def __init__(self, repo_id: str, token: str, repo_type: str = "model",
                 private: bool = True, max_commits_per_hour: int = 100,
                 retries: int = 6):
        from huggingface_hub import HfApi
        from src.pipeline.env import quiet_third_party
        quiet_third_party()
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.private = private
        self.api = HfApi(token=token)
        self.retries = retries
        self.max_commits_per_hour = max_commits_per_hour
        self._commit_times: deque = deque()

    # ---------------------------------------------------------- plumbing
    def ensure_repo(self) -> str:
        url = self._retry(lambda: self.api.create_repo(
            self.repo_id, repo_type=self.repo_type, private=self.private, exist_ok=True),
            "create_repo")
        return str(url)

    def head(self) -> Optional[str]:
        info = self._retry(lambda: self.api.repo_info(self.repo_id, repo_type=self.repo_type),
                           "repo_info")
        return info.sha

    def web_url(self, path: str = "") -> str:
        kind = {"model": "", "dataset": "datasets/", "space": "spaces/"}[self.repo_type]
        base = f"https://huggingface.co/{kind}{self.repo_id}"
        return f"{base}/tree/main/{path}" if path else base

    def _retry(self, fn: Callable, what: str, retries: Optional[int] = None,
               retry_on_conflict: bool = False):
        from huggingface_hub.errors import HfHubHTTPError
        n = self.retries if retries is None else retries
        for attempt in range(n):
            try:
                return fn()
            except HfHubHTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (409, 412) and not retry_on_conflict:
                    raise
                if status is not None and 400 <= status < 500 and status not in (408, 409, 412, 429):
                    raise                                   # auth / not-found: do not retry
                retry_after = None
                if status == 429:
                    try:
                        retry_after = float(e.response.headers.get("Retry-After", ""))
                    except (TypeError, ValueError, AttributeError):
                        retry_after = None
                self._backoff(attempt, what, e, status, retry_after)
            except (ConnectionError, TimeoutError, OSError) as e:
                self._backoff(attempt, what, e)
            except Exception as e:  # noqa: BLE001 - httpx/requests transport errors
                if "timed out" in str(e).lower() or "connect" in str(e).lower():
                    self._backoff(attempt, what, e)
                else:
                    raise
        raise HubError(f"{what} failed after {n} attempts")

    @staticmethod
    def _backoff(attempt: int, what: str, err: Exception, status=None, retry_after=None):
        if status == 429:        # rate limit (commit budget is per hour): wait much longer
            delay = retry_after if retry_after else min(900.0, 60.0 * (attempt + 1))
        else:
            delay = min(120.0, 2.0 * (2 ** attempt)) * (0.75 + 0.5 * random.random())
        log_event("hub_retry", f"{what}: {type(err).__name__} (HTTP {status}); retry in {delay:.0f}s",
                  logging.WARNING, attempt=attempt + 1, status=status)
        time.sleep(delay)

    def _throttle(self):
        now = time.time()
        while self._commit_times and now - self._commit_times[0] > 3600:
            self._commit_times.popleft()
        if len(self._commit_times) >= self.max_commits_per_hour:
            wait = 3600 - (now - self._commit_times[0]) + 1
            log_event("hub_throttle", f"commit budget reached; waiting {wait:.0f}s",
                      logging.WARNING)
            time.sleep(wait)
        self._commit_times.append(time.time())

    # ---------------------------------------------------------- reading
    def exists(self, path: str) -> bool:
        return bool(self.paths_info([path]))

    def paths_info(self, paths: list[str]) -> dict:
        if not paths:
            return {}
        infos = self._retry(lambda: self.api.get_paths_info(
            self.repo_id, paths, repo_type=self.repo_type, expand=False), "get_paths_info")
        return {i.path: i for i in infos}

    def list_files(self, prefix: str, recursive: bool = True) -> list:
        from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
        try:
            items = self._retry(lambda: list(self.api.list_repo_tree(
                self.repo_id, path_in_repo=prefix, recursive=recursive,
                repo_type=self.repo_type)), "list_repo_tree")
        except EntryNotFoundError:
            return []
        except HfHubHTTPError as e:
            if getattr(getattr(e, "response", None), "status_code", None) == 404:
                return []
            raise
        return items

    def list_dirs(self, prefix: str) -> list[str]:
        return sorted(i.path for i in self.list_files(prefix, recursive=False)
                      if not hasattr(i, "size") or getattr(i, "size", None) is None)

    def download(self, path: str, local_dir: str | Path, revision: Optional[str] = None) -> Path:
        local = self._retry(lambda: self.api.hf_hub_download(
            self.repo_id, path, repo_type=self.repo_type, revision=revision,
            local_dir=str(local_dir)), f"download {path}")
        return Path(local)

    def read_bytes(self, path: str, revision: Optional[str] = None) -> Optional[bytes]:
        from huggingface_hub.errors import EntryNotFoundError
        import tempfile
        try:
            with tempfile.TemporaryDirectory() as td:
                p = self.download(path, td, revision=revision)
                return p.read_bytes()
        except EntryNotFoundError:
            return None
        except HubError:
            raise
        except Exception as e:  # noqa: BLE001
            if "404" in str(e) or "Entry Not Found" in str(e):
                return None
            raise

    def read_json(self, path: str, revision: Optional[str] = None):
        b = self.read_bytes(path, revision=revision)
        return None if b is None else json.loads(b.decode("utf-8"))

    # ---------------------------------------------------------- writing
    def commit(self, adds: dict[str, str | Path | bytes] | None = None,
               deletes: Iterable[str] = (), message: str = "update",
               parent_commit: Optional[str] = None, verify: bool = True,
               delete_folders: Iterable[str] = ()) -> str:
        """One atomic commit. `adds` maps repo path -> local file path or raw bytes."""
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        adds = adds or {}
        ops = []
        for rp, src in adds.items():
            ops.append(CommitOperationAdd(path_in_repo=rp, path_or_fileobj=(
                io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src))))
        for rp in deletes:
            ops.append(CommitOperationDelete(path_in_repo=rp))
        for rp in delete_folders:
            ops.append(CommitOperationDelete(path_in_repo=rp.rstrip("/") + "/", is_folder=True))
        if not ops:
            return self.head() or ""
        self._throttle()
        info = self._retry(lambda: self.api.create_commit(
            self.repo_id, operations=ops, commit_message=message[:200],
            repo_type=self.repo_type, parent_commit=parent_commit), f"commit '{message[:40]}'",
            retry_on_conflict=parent_commit is None)
        sha = getattr(info, "oid", None) or ""
        if verify and adds:
            self.verify(adds)
        return sha

    def verify(self, adds: dict[str, str | Path | bytes]):
        """Re-read remote metadata; compare sha256 (LFS/Xet) or git blob sha1 + size."""
        remote = self.paths_info(list(adds))
        bad = []
        for rp, src in adds.items():
            info = remote.get(rp)
            if info is None:
                bad.append(f"{rp}: missing after commit")
                continue
            data = src if isinstance(src, (bytes, bytearray)) else None
            size = len(data) if data is not None else Path(src).stat().st_size
            if getattr(info, "size", size) != size:
                bad.append(f"{rp}: size {info.size} != local {size}")
                continue
            lfs = getattr(info, "lfs", None)
            lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
            if lfs_sha:
                local = hashlib.sha256(data).hexdigest() if data is not None else sha256_file(src)
                if local != lfs_sha:
                    bad.append(f"{rp}: sha256 mismatch")
            elif getattr(info, "blob_id", None):
                local = git_blob_sha1(bytes(data)) if data is not None else _git_blob_sha1_file(Path(src))
                if local != info.blob_id:
                    bad.append(f"{rp}: blob sha1 mismatch")
        if bad:
            raise HubError("upload verification failed:\n  " + "\n  ".join(bad))

    def write_json(self, path: str, obj, message: Optional[str] = None) -> str:
        data = json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8")
        return self.commit({path: data}, message=message or f"update {path}")

    def update_json(self, path: str, fn: Callable, message: str, max_attempts: int = 8):
        """Read-modify-write with the parent commit pinned (optimistic concurrency).

        `fn(current_or_None) -> new_obj` must be pure; it is re-run on conflict.
        Returns the object that was written.
        """
        from huggingface_hub.errors import HfHubHTTPError
        for attempt in range(max_attempts):
            head = self.head()
            cur = self.read_json(path, revision=head)
            new = fn(cur)
            data = json.dumps(new, indent=2, sort_keys=True, default=str).encode("utf-8")
            try:
                self.commit({path: data}, message=message, parent_commit=head)
                return new
            except HfHubHTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (409, 412):
                    time.sleep(1.0 + random.random() * 2 * (attempt + 1))
                    continue
                raise
        raise HubError(f"update_json({path}) lost {max_attempts} races in a row")
