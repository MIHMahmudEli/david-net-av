"""Set up a multi-account training campaign on the shared HF repo. Run ONCE, from anywhere.

    python scripts/publish_campaign.py splits --splits-root /kaggle/working/splits
    python scripts/publish_campaign.py jobs   --seeds 42,123,456 --epochs 10
    python scripts/publish_campaign.py status

`splits` freezes the data partition every worker must use (see src/utils/splits_sync.py).
`jobs`   writes the queue that workers claim from (see src/utils/coordinator.py).
`status` prints who is running what right now, across all accounts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.coordinator import REPO_ID as COORD_REPO, Coordinator  # noqa: E402
from src.utils.splits_sync import REPO_ID as MODEL_REPO, publish, published_index  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def cmd_splits(args):
    if published_index(args.repo_id) and not args.force:
        raise SystemExit("splits already published — pass --force to overwrite.\n"
                         "Overwriting mid-campaign changes the partition under runs that are "
                         "already training; prefer starting a new campaign instead.")
    publish(args.splits_root, repo_id=args.repo_id,
            extra_provenance={"git_commit": _git_commit(),
                              "built_by": os.environ.get("KAGGLE_USER_NAME", "local"),
                              "note": "subject-disjoint FakeAVCeleb splits, seed 42"})


def cmd_jobs(args):
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    jobs = []
    for i, seed in enumerate(seeds):
        jobs.append({
            "job_id": f"stage1_seed{seed}",
            "kind": "stage1",
            "run_id": f"{args.run_prefix}_seed{seed}",
            "epochs": args.epochs,
            "priority": i,                       # seeds are independent: any order is fine
            "config": {"seed": seed},
        })
    # Evaluation waits for every seed, so whichever worker is free last picks it up.
    jobs.append({
        "job_id": "evaluate_all",
        "kind": "eval",
        "run_id": f"{args.run_prefix}_eval",
        "epochs": 0,
        "priority": 900,
        "depends_on": [j["job_id"] for j in jobs],
        "config": {"seeds": seeds},
    })

    doc = {"campaign": args.run_prefix, "created_by": _git_commit(), "jobs": jobs}
    coord = Coordinator(repo_id=args.repo_id)
    coord._put(doc, "coord/jobs.json")
    print(f"published {len(jobs)} jobs to {args.repo_id}/coord/jobs.json")
    for j in jobs:
        dep = f"  after {j['depends_on']}" if j.get("depends_on") else ""
        print(f"  [{j['priority']:>3}] {j['job_id']:<22} {j['kind']:<8} {j['epochs']:>3} ep{dep}")


def cmd_status(args):
    rows = Coordinator(repo_id=args.repo_id).status()
    if not rows:
        print("no jobs published yet — run: publish_campaign.py jobs")
        return
    width = max(len(r["job_id"]) for r in rows)
    for r in rows:
        mark = {"done": "[x]", "running": "[~]", "free": "[ ]", "blocked": "[-]"}[r["state"]]
        print(f"{mark} {r['job_id']:<{width}}  {r['state']:<8} {r['worker']}")
    n_done = sum(1 for r in rows if r["state"] == "done")
    print(f"\n{n_done}/{len(rows)} complete")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("splits", help="freeze and publish the data partition")
    s.add_argument("--splits-root", required=True, help="dir holding <dataset>/{train,val,test}.jsonl")
    s.add_argument("--force", action="store_true")
    s.add_argument("--repo-id", default=MODEL_REPO, help="splits live with the artifacts")
    s.set_defaults(func=cmd_splits)

    j = sub.add_parser("jobs", help="author the job queue")
    j.add_argument("--seeds", default="42,123,456")
    j.add_argument("--epochs", type=int, default=10)
    j.add_argument("--run-prefix", default="stage1_v3")
    j.add_argument("--repo-id", default=COORD_REPO, help="control plane repo")
    j.set_defaults(func=cmd_jobs)

    st = sub.add_parser("status", help="who is running what, across all accounts")
    st.add_argument("--repo-id", default=COORD_REPO)
    st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
