"""Experiment orchestrator: the full paper matrix from one command.

Generates ablation configs from the base config (docs/04_experiments.md §4),
runs Stage 0 (QACP) + Stage 1 (train) + evaluate for each x each seed, collects
results JSONs into results/, and regenerates the thesis figures.

    python scripts/run_experiments.py --base configs/david_net.yaml \
        --test-manifest src/data/splits/fakeavceleb/test.jsonl \
        --seeds 42 43 44 --out results/
    python scripts/run_experiments.py ... --only full no_qacp --dry-run

Each run writes results/<ablation>_seed<k>.json. Aggregation (mean ± std,
bootstrap CIs) is scripts/aggregate_results.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

# ablation grid — keys map onto config fields (docs/04_experiments.md §4)
ABLATIONS: dict[str, dict] = {
    "full":            {},
    "no_sync":         {"use_sync": False},
    "no_disentangle":  {"use_disentangle": False, "loss_weights.disentangle": 0.0},
    "no_qacp":         {"_skip_qacp": True},
    "no_loc":          {"loss_weights.loc": 0.0},
    "single_task":     {"loss_weights.quad": 0.0, "loss_weights.loc": 0.0,
                        "loss_weights.sync": 0.0, "loss_weights.disentangle": 0.0},
    "compose_quad":    {"compose_quadrant": True},
    "no_moddrop":      {"modality_dropout": 0.0},
}


def make_config(base: dict, overrides: dict, seed: int, out_dir: Path,
                name: str) -> tuple[Path, bool]:
    cfg = copy.deepcopy(base)
    skip_qacp = overrides.get("_skip_qacp", False)
    for key, val in overrides.items():
        if key.startswith("_"):
            continue
        if "." in key:                      # nested, e.g. loss_weights.loc
            head, tail = key.split(".", 1)
            cfg.setdefault(head, {})[tail] = val
        else:
            cfg[key] = val
    cfg["seed"] = seed
    cfg["out_dir"] = str(out_dir / f"runs_{name}_seed{seed}")
    path = out_dir / f"cfg_{name}_seed{seed}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path, skip_qacp


def run_one(name: str, cfg_path: Path, skip_qacp: bool, seed: int,
            test_manifest: str, out_dir: Path, dry: bool) -> None:
    py = sys.executable
    extra = ["--dry-run"] if dry else []

    if not skip_qacp:
        subprocess.run([py, "-m", "src.training.pretrain_qacp",
                        "--config", str(cfg_path), *extra], check=True)
        if not dry:
            # wire the last QACP checkpoint into stage 1
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            ckpts = sorted(Path(cfg["out_dir"]).glob("qacp_epoch*.pt"))
            if ckpts:
                cfg["init_from"] = str(ckpts[-1])
                cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    subprocess.run([py, "-m", "src.training.train",
                    "--config", str(cfg_path), *extra], check=True)

    if not dry:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        ckpts = sorted(Path(cfg["out_dir"]).glob("david_net_epoch*.pt"))
        result = out_dir / f"{name}_seed{seed}.json"
        subprocess.run([py, "-m", "src.eval.evaluate",
                        "--config", str(cfg_path),
                        "--checkpoint", str(ckpts[-1]) if ckpts else "",
                        "--manifest", test_manifest,
                        "--out", str(result)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/david_net.yaml")
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--only", nargs="+", default=None,
                    help=f"subset of: {', '.join(ABLATIONS)}")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    names = args.only or list(ABLATIONS)

    plan = [(n, s) for n in names for s in args.seeds]
    print(f"experiment plan: {len(plan)} runs "
          f"({len(names)} configs x {len(args.seeds)} seeds)")
    for name, seed in plan:
        print(f"\n=== {name} / seed {seed} ===")
        cfg_path, skip_qacp = make_config(base, ABLATIONS[name], seed, out_dir, name)
        run_one(name, cfg_path, skip_qacp, seed, args.test_manifest,
                out_dir, args.dry_run)

    if not args.dry_run:
        subprocess.run([sys.executable, "-m", "src.eval.figures",
                        "--results", str(out_dir),
                        "--out", "report/figures/generated"], check=True)
        print("\nall runs complete; figures regenerated.")


if __name__ == "__main__":
    main()
