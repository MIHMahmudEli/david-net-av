"""CLI entry point for executing individual experiments via Session driver."""
from __future__ import annotations

import argparse
from src.pipeline.config import build_config
from src.pipeline.driver import Session


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment via Session driver")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--only", required=True, help="Experiment name to run")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--worker", default="kaggle-worker", help="Worker identifier")
    args = parser.parse_args()

    cfg = build_config()
    cfg["mode"] = args.mode
    cfg["session"]["worker_name"] = args.worker
    cfg["seeds"] = [args.seed]
    cfg["plan"] = {"only": [args.only]}

    print(f"=== [DRIVER] Starting experiment {args.only} (seed {args.seed}) on {args.worker} ===")
    s = Session(cfg, repo_dir=".")
    s.connect()
    s.prepare_data()
    s.prepare_features()
    res = s.run_experiments()
    print("=== [DRIVER] Execution finished successfully ===", res)


if __name__ == "__main__":
    main()
