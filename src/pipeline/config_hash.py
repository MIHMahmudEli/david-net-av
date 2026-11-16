"""CLI and utility for validating config hashes against registered/expected values."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.pipeline.config import build_config, config_hash
from src.pipeline.plan import full_plan, plan_table


def check_config_hashes(cache_path: str | Path | None = None) -> bool:
    """Verifies that the current codebase produces identical config hashes to the completed runs."""
    cfg = build_config()
    plan = full_plan()
    rows = plan_table(plan, [42], cfg)

    if cache_path is None:
        cache_path = Path("report/.cache/audit/registry__experiments.json")
    else:
        cache_path = Path(cache_path)

    if not cache_path.exists():
        print(f"[ERROR] Registry cache not found at {cache_path}")
        return False

    reg = json.loads(cache_path.read_text(encoding="utf-8"))
    reg_exps = {
        e["name"]: e.get("config_hash")
        for e in reg.get("experiments", {}).values()
        if e.get("seed") == 42
    }

    all_matched = True
    print(f"{'Experiment Name':32s} | {'Current Hash':14s} | {'Registry Hash':14s} | Status")
    print("-" * 75)
    for r in rows:
        name = r["name"]
        cur_hash = r["config_hash"]
        reg_hash = reg_exps.get(name)
        matched = (cur_hash == reg_hash)
        status = "MATCH" if matched else "MISMATCH"
        if not matched:
            all_matched = False
        print(f"{name:32s} | {cur_hash:14s} | {str(reg_hash):14s} | {status}")

    print("-" * 75)
    if all_matched:
        print("[SUCCESS] All 27 experiment configurations match the registered registry config hashes 100%.")
    else:
        print("[FAILURE] Some config hashes differed! Treat mismatched runs as different protocol versions.")
    return all_matched


def main():
    parser = argparse.ArgumentParser(description="Check config hashes")
    parser.add_argument("command", choices=["check"], help="Subcommand to run")
    parser.add_argument("--cache", default=None, help="Path to registry__experiments.json")
    args = parser.parse_args()

    if args.command == "check":
        ok = check_config_hashes(args.cache)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
