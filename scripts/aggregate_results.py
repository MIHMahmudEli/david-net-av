"""Aggregate multi-seed results into paper-ready numbers.

Reads results/<name>_seed<k>.json (from run_experiments.py), reports
mean +/- std per configuration, bootstrap 95% CIs on AUC, and emits both a
LaTeX table body (paste into the report) and results/ablation.json for the
figure generator.

    python scripts/aggregate_results.py --results results/ --metric video.auc
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN_RE = re.compile(r"^(?P<name>.+)_seed(?P<seed>\d+)\.json$")


def bootstrap_auc(y_true, y_score, n_boot: int = 1000, seed: int = 0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def get_path(d: dict, dotted: str):
    for part in dotted.split("."):
        d = d[part]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--metric", default="video.auc",
                    help="dotted path into the report, e.g. video.auc / audio.eer")
    ap.add_argument("--boot", action="store_true", help="also bootstrap CIs (slow)")
    args = ap.parse_args()

    rdir = Path(args.results)
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(rdir.glob("*.json")):
        m = RUN_RE.match(p.name)
        if not m or p.name.startswith(("robustness", "ablation", "baseline")):
            continue
        groups[m.group("name")].append(json.loads(p.read_text(encoding="utf-8")))

    if not groups:
        raise SystemExit(f"no <name>_seed<k>.json files under {rdir}")

    print(f"metric: {args.metric}\n")
    print(f"{'config':<18}{'mean':>8}{'std':>8}{'n':>4}   95% CI (seed 1)")
    rows = []
    for name, reps in sorted(groups.items()):
        vals = [get_path(r, args.metric) for r in reps]
        mean, std = float(np.mean(vals)), float(np.std(vals))
        ci = ""
        if args.boot and "preds" in reps[0]:
            mod = args.metric.split(".")[0]
            p = reps[0]["preds"][mod]
            lo, hi = bootstrap_auc(p["y_true"], p["y_score"])
            ci = f"[{lo:.3f}, {hi:.3f}]"
        print(f"{name:<18}{mean:>8.4f}{std:>8.4f}{len(vals):>4}   {ci}")
        rows.append({"name": name, "mean": mean, "std": std, "n": len(vals)})

    # LaTeX table body
    print("\n% ---- LaTeX table body (paste into the report) ----")
    for r in rows:
        print(f"{r['name'].replace('_', ' ')} & "
              f"${r['mean']:.3f} \\pm {r['std']:.3f}$ \\\\")

    # ablation.json feeds fig_ablation once cross-dataset runs exist:
    # merge in cross_dataset_auc per config, then re-run src.eval.figures.
    out = [{"name": r["name"], "in_domain_auc": round(r["mean"], 4),
            "cross_dataset_auc": None} for r in rows]
    with open(rdir / "ablation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {rdir / 'ablation.json'} (fill cross_dataset_auc after "
          f"cross-dataset evaluation, then re-run src.eval.figures)")


if __name__ == "__main__":
    main()
