"""Keep the shared HF repo's storage from running away during a multi-account campaign.

The problem: HF keeps every revision of every LFS file. `epoch_latest.pt` is ~2.2 GB and is
overwritten once per epoch, so a single 10-epoch seed leaves ~22 GB of *superseded* blobs
behind even though only one copy is ever live. Ten workers make that arithmetic hostile —
measured 2026-09-24: this repo was billed 545.08 GB to back a 10.71 GB live tree, i.e.
534.37 GB of retained history.

`super_squash_history` collapses the branch to a single commit and releases the orphaned
blobs. It is irreversible and it destroys the ability to check out an earlier revision, but
what lives in those revisions is only superseded checkpoints: the current `best.pt`,
`epoch_latest.pt`, every log and every metrics file are files in the tree, so they survive.

Squash is serialised through a lease in `coord/maintenance.json` so that ten workers do not
all rewrite history at once, and it is deliberately run right after a checkpoint push
rather than during one: a concurrent upload may get rejected while history is rewritten,
which HFBackup's retry/backoff absorbs, but there is no reason to invite it.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

REPO_ID = "MoshinAli/david-net-av-backup"
MAINTENANCE_PATH = "coord/maintenance.json"
DEFAULT_INTERVAL_H = 6


def _token(token: Optional[str] = None) -> Optional[str]:
    return token or os.environ.get("HF_TOKEN") or os.environ.get("hf")


def storage_report(repo_id: str = REPO_ID, token: Optional[str] = None) -> dict:
    """Live bytes in the tree vs. bytes the account is actually billed for."""
    from huggingface_hub import HfApi
    api = HfApi(token=_token(token))
    info = api.model_info(repo_id, files_metadata=True)
    live = sum((s.size or 0) for s in (info.siblings or []))
    # `used_storage` is what the repo actually costs, i.e. live blobs PLUS every retained
    # revision. The gap against the live tree is the history we are here to reclaim.
    used = getattr(info, "used_storage", None)
    return {"live_bytes": live, "repo_used_bytes": used,
            "n_files": len(info.siblings or []),
            "history_overhead_bytes": (used - live) if (used and used > live) else None}


def print_storage(repo_id: str = REPO_ID, token: Optional[str] = None):
    r = storage_report(repo_id, token)
    gb = lambda b: "n/a" if b is None else f"{b / 1e9:.2f} GB"  # noqa: E731
    print(f"[storage] live tree: {gb(r['live_bytes'])} across {r['n_files']} files")
    print(f"[storage] repo billed: {gb(r['repo_used_bytes'])}"
          f"  (retained history {gb(r['history_overhead_bytes'])})")
    return r


def reclaim_history(repo_id: str = REPO_ID, token: Optional[str] = None,
                    branch: str = "main") -> bool:
    """Collapse branch history to one commit and release orphaned blobs. Irreversible."""
    from huggingface_hub import HfApi
    api = HfApi(token=_token(token))
    try:
        api.super_squash_history(repo_id=repo_id, repo_type="model", branch=branch)
        print(f"[storage] squashed {repo_id}@{branch} to a single commit")
        return True
    except Exception as e:  # noqa: BLE001 - never take training down over housekeeping
        print(f"[storage] squash failed (continuing): {str(e)[:160]}")
        return False


def maybe_reclaim(coord, interval_h: float = DEFAULT_INTERVAL_H, repo_id: str = REPO_ID,
                  token: Optional[str] = None) -> bool:
    """Squash at most once per `interval_h` across the whole fleet.

    `coord` is a Coordinator; its _get/_put give us a tiny lease record. The check is
    advisory, not a hard mutex: the worst case is two workers squashing within the same
    minute, which is harmless (the second is a no-op on an already-squashed branch).
    """
    now = time.time()
    rec = coord._get(MAINTENANCE_PATH) or {}
    last = float(rec.get("last_squash", 0))
    if now - last < interval_h * 3600:
        return False
    coord._put({"last_squash": now, "worker_id": coord.worker_id,
                "previous": rec.get("last_squash")}, MAINTENANCE_PATH)
    return reclaim_history(repo_id=repo_id, token=token)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Report or reclaim HF repo storage.")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--squash", action="store_true", help="collapse history (IRREVERSIBLE)")
    args = ap.parse_args()

    before = print_storage(args.repo_id)
    if not args.squash:
        return
    names_before = _file_names(args.repo_id)
    if not reclaim_history(args.repo_id):
        raise SystemExit(1)
    names_after = _file_names(args.repo_id)
    missing = sorted(names_before - names_after)
    print(f"[storage] live files before={len(names_before)} after={len(names_after)}")
    if missing:
        raise SystemExit(f"files disappeared during squash: {missing}")
    print("[storage] every live file survived")
    after = print_storage(args.repo_id)
    if before["repo_used_bytes"] and after["repo_used_bytes"]:
        freed = before["repo_used_bytes"] - after["repo_used_bytes"]
        print(f"[storage] reclaimed {freed / 1e9:.1f} GB")


def _file_names(repo_id: str) -> set:
    from huggingface_hub import HfApi
    info = HfApi(token=_token()).model_info(repo_id, files_metadata=True)
    return {s.rfilename for s in (info.siblings or [])}


if __name__ == "__main__":
    main()
