"""Publish the train/val/test splits once, then have every worker fetch the same bytes.

Why this exists (this is a correctness issue, not a convenience):
`scripts/build_manifest.py` is deterministic given the same files on disk — it sorts
`rglob` results and shuffles identities with `random.Random(seed)`. But "the same files on
disk" is exactly what stops being true across ten Kaggle accounts: a mounted Kaggle dataset
resolves to its LATEST version, so if a dataset owner publishes a new version mid-campaign,
some accounts build their splits from a different file set. Seeds trained on different
subject partitions are not comparable, and a clip that is test on one account can be train
on another — silent leakage that would invalidate the results table.

So: build once, hash, publish, and make every worker verify the hash before training.
The published `splits/SPLITS_SHA256.json` is also what you cite in the paper so reviewers
can reproduce the exact partition.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

REPO_ID = "MoshinAli/david-net-av-backup"
SPLITS_PREFIX = "splits"
MANIFEST_NAME = "SPLITS_SHA256.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _token(token: Optional[str] = None) -> Optional[str]:
    return token or os.environ.get("HF_TOKEN") or os.environ.get("hf")


def build_index(splits_root: Path) -> dict:
    """{relative_path: {sha256, bytes, lines}} for every .jsonl under splits_root."""
    index = {}
    for p in sorted(splits_root.rglob("*.jsonl")):
        rel = p.relative_to(splits_root).as_posix()
        with open(p, "rb") as f:
            n_lines = sum(1 for line in f if line.strip())
        index[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size, "lines": n_lines}
    return index


def publish(splits_root: str, repo_id: str = REPO_ID, token: Optional[str] = None,
            extra_provenance: Optional[dict] = None) -> dict:
    """Upload every split file plus the hash index. Run ONCE, from one account."""
    from huggingface_hub import HfApi
    root = Path(splits_root)
    index = build_index(root)
    if not index:
        raise SystemExit(f"no .jsonl files under {root}")

    # ONE commit for all ~33 split files plus the index. Uploading them individually
    # cost 34 of the 128 commits/hour this repo is allowed, for no benefit.
    from huggingface_hub import CommitOperationAdd
    api = HfApi(token=_token(token))
    doc = {"files": index, "provenance": extra_provenance or {}}
    ops = [CommitOperationAdd(path_in_repo=f"{SPLITS_PREFIX}/{rel}",
                              path_or_fileobj=str(root / rel)) for rel in index]
    ops.append(CommitOperationAdd(path_in_repo=f"{SPLITS_PREFIX}/{MANIFEST_NAME}",
                                  path_or_fileobj=json.dumps(doc, indent=1).encode()))
    api.create_commit(repo_id=repo_id, repo_type="model", operations=ops,
                      commit_message=f"splits: publish campaign partition ({len(index)} files)")
    print(f"published {len(index)} split files to {repo_id}/{SPLITS_PREFIX}")
    for rel, meta in index.items():
        print(f"  {meta['lines']:>7} lines  {meta['sha256'][:16]}  {rel}")
    return doc


def fetch_and_verify(dest: str, repo_id: str = REPO_ID, token: Optional[str] = None) -> Path:
    """Download the published splits and fail loudly on any hash mismatch.

    Every worker calls this instead of rebuilding manifests. It is also ~7 minutes faster
    per session than re-walking 340k files across seven mounted datasets.
    """
    from huggingface_hub import hf_hub_download
    tok = _token(token)
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    idx_local = hf_hub_download(repo_id, f"{SPLITS_PREFIX}/{MANIFEST_NAME}", repo_type="model",
                                token=tok, force_download=True)
    doc = json.load(open(idx_local, encoding="utf-8"))

    bad = []
    for rel, meta in doc["files"].items():
        local = hf_hub_download(repo_id, f"{SPLITS_PREFIX}/{rel}", repo_type="model", token=tok)
        target = dest_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != Path(local).resolve():
            target.write_bytes(Path(local).read_bytes())
        got = sha256_file(target)
        if got != meta["sha256"]:
            bad.append(f"{rel}: expected {meta['sha256'][:16]} got {got[:16]}")

    if bad:
        raise SystemExit("SPLIT HASH MISMATCH — refusing to train on a different partition:\n  "
                         + "\n  ".join(bad))
    print(f"splits verified: {len(doc['files'])} files match {MANIFEST_NAME} -> {dest_path}")
    return dest_path


def published_index(repo_id: str = REPO_ID, token: Optional[str] = None) -> Optional[dict]:
    """The hash index if splits have been published, else None."""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo_id, f"{SPLITS_PREFIX}/{MANIFEST_NAME}", repo_type="model",
                            token=_token(token), force_download=True)
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001 - not published yet
        return None
