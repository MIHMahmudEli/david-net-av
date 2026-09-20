"""Read the watchdog telemetry a dead Kaggle session left on HF.

    python scripts/read_watchdog.py                 # every session + run watchdog, last lines
    python scripts/read_watchdog.py --run stage1_v2_seed42 --tail 30

Interpretation:
  * session watchdog stops ticking + train watchdog stops at the same minute  -> the whole
    session was killed (platform: quota / idle / manual). Look at `run_type` (Interactive
    vs Batch) and `uptime_s` of the last line.
  * train watchdog stops, session keeps ticking with `cell` still on Cell 11  -> the
    training PROCESS died. `event: signal` tells which signal (if catchable); otherwise
    the last `clips` are the batch being processed (a decoder crash is data-dependent).
  * host_avail_gb / shm_free_gb / disk_free_gb trending to ~0 before death        -> resource kill.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ID = "MoshinAli/david-net-av-backup"


def _token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("hf")
    if not t and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith("hf="):
                t = line.split("=", 1)[1].strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run_id filter (default: all)")
    ap.add_argument("--tail", type=int, default=12)
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=_token())
    files = [f.path for f in api.list_repo_tree(REPO_ID, repo_type="model", recursive=True)
             if hasattr(f, "path") and "watchdog_" in f.path]
    if args.run:
        files = [f for f in files if args.run in f]
    if not files:
        print("no watchdog logs on HF"); return
    for fp in sorted(files):
        local = hf_hub_download(REPO_ID, fp, repo_type="model", token=_token())
        lines = [json.loads(l) for l in open(local, encoding="utf-8") if l.strip()]
        print(f"\n===== {fp}  ({len(lines)} lines; first {lines[0]['t']} -> last {lines[-1]['t']})")
        for l in lines[-args.tail:]:
            keys = ("t", "event", "cell", "epoch", "micro", "opt", "rss_gb", "host_avail_gb", "shm_free_gb",
                    "disk_free_gb", "vram_alloc_gb", "gpu_util", "n_ffmpeg", "signal", "run_type", "uptime_s")
            print("  " + " ".join(f"{k}={l[k]}" for k in keys if k in l and l[k] is not None))
            if l.get("event") in ("exception", "signal"):
                print("     ->", str(l.get("error", l.get("signal")))[-600:])
        if lines[-1].get("clips"):
            print("  last clips:", lines[-1]["clips"])


if __name__ == "__main__":
    main()
