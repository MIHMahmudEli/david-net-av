"""Publication figure generation for the thesis (Chapter 3 / paper).

Reads the results JSONs written by evaluate.py / train_baseline.py / robustness.py
and regenerates every results figure as PDF (for LaTeX) + PNG (for preview) into
report/figures/generated/. Re-run after every experiment — figures are always
derived from committed result files, never hand-edited.

    python -m src.eval.figures --results results/ --out report/figures/generated
    python -m src.eval.figures --demo --out report/figures/generated   # plumbing test

Design rules (dataviz method, validated palette):
  * categorical hues in FIXED slot order (never cycled); ≤4 series direct-labeled
  * sequential = one blue ramp (confusion heatmap); never rainbow
  * one axis per chart; thin marks; recessive hairline grid; muted axis ink
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- palette
# Validated reference palette (light surface #fcfcfb; adjacent CVD dE 24.2).
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
       "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]   # fixed slot order
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#1c5cab", "#104281", "#0d366b"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    })
    return plt


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    print(f"  wrote {name}.pdf/.png")


# ---------------------------------------------------------------- figures
def fig_roc(reports: list[dict], out_dir: Path):
    """ROC curves per modality for every method (form: line; slots by method)."""
    from sklearn.metrics import roc_curve
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, mod in zip(axes, ("video", "audio")):
        ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.0, linestyle="--", zorder=1)
        si = 0
        for rep in reports:
            preds = rep.get("preds", {})
            p = preds.get(mod) if isinstance(preds.get(mod), dict) else (
                preds if rep.get("modality") == mod else None)
            if not p or len(set(p["y_true"])) < 2:
                continue
            fpr, tpr, _ = roc_curve(p["y_true"], p["y_score"])
            label = rep.get("method", "?")
            ax.plot(fpr, tpr, color=CAT[si % len(CAT)], label=label, zorder=2)
            si += 1
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{mod.capitalize()} stream")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        if si:
            ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "results_roc")
    plt.close(fig)


def fig_reliability(report: dict, out_dir: Path, n_bins: int = 10):
    """Reliability diagram per modality (calibration; RQ5)."""
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, mod, color in zip(axes, ("video", "audio"), CAT[:2]):
        p = report["preds"][mod]
        y = np.asarray(p["y_true"]); s = np.asarray(p["y_score"])
        bins = np.linspace(0, 1, n_bins + 1)
        centers, accs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (s > lo) & (s <= hi)
            if m.sum() == 0:
                continue
            centers.append((lo + hi) / 2)
            accs.append(float((y[m] == (s[m] >= 0.5)).mean()))
        ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.0, linestyle="--", zorder=1)
        ax.bar(centers, accs, width=0.8 / n_bins, color=color,
               edgecolor="#fcfcfb", linewidth=1.0, zorder=2)
        ece = report.get("calibration", {}).get(f"{mod}_ece")
        title = f"{mod.capitalize()}" + (f"  (ECE = {ece:.3f})" if ece is not None else "")
        ax.set_title(title)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, out_dir, "results_reliability")
    plt.close(fig)


def fig_confusion(report: dict, out_dir: Path):
    """Quadrant confusion matrix (heatmap; sequential blue ramp)."""
    from matplotlib.colors import LinearSegmentedColormap
    plt = _style()
    cm = np.asarray(report["quadrant"]["confusion"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = ["RVRA", "RVFA", "FVRA", "FVFA"]
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#fcfcfb"] + SEQ_BLUE)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.grid(False)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(4), labels); ax.set_yticks(range(4), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(4):
        for j in range(4):
            ink = "#ffffff" if norm[i, j] > 0.55 else INK2
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color=ink, fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "results_confusion")
    plt.close(fig)


def fig_robustness(rob: dict, out_dir: Path):
    """AUC vs degradation level (one small panel per sweep; single series)."""
    plt = _style()
    sweeps = []
    for mod, entries in rob["sweeps"].items():
        for name, pts in entries.items():
            sweeps.append((mod, name, pts))
    n = len(sweeps)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6), squeeze=False)
    x_labels = {"blur_sigma": "Blur σ", "downscale": "Downscale ×",
                "quantize_levels": "Quantization levels", "snr_db": "SNR (dB)"}
    for ax, (mod, name, pts) in zip(axes[0], sweeps):
        xs = [str(p["level"]) for p in pts]
        ys = [p["auc"] for p in pts]
        color = CAT[0] if mod == "video" else CAT[1]
        ax.plot(xs, ys, marker="o", markersize=4, color=color)
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel(x_labels.get(name, name))
        ax.set_ylabel("AUC" if ax is axes[0][0] else "")
        ax.set_title(f"{mod}: {name.replace('_', ' ')}", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "results_robustness")
    plt.close(fig)


def fig_ablation(rows: list[dict], out_dir: Path):
    """Ablation horizontal bars: in-domain vs cross-dataset AUC (2 fixed slots)."""
    plt = _style()
    names = [r["name"] for r in rows]
    indom = [r.get("in_domain_auc") for r in rows]
    cross = [r.get("cross_dataset_auc") for r in rows]
    y = np.arange(len(names))
    h = 0.38

    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(names) + 1.2))
    ax.barh(y - h / 2, indom, height=h, color=CAT[0], label="In-domain",
            edgecolor="#fcfcfb", linewidth=1.0)
    ax.barh(y + h / 2, cross, height=h, color=CAT[1], label="Cross-dataset",
            edgecolor="#fcfcfb", linewidth=1.0)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("AUC")
    ax.set_xlim(0.5, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    for yy, v in zip(y - h / 2, indom):
        if v:
            ax.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=7, color=INK2)
    for yy, v in zip(y + h / 2, cross):
        if v:
            ax.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=7, color=INK2)
    fig.tight_layout()
    _save(fig, out_dir, "results_ablation")
    plt.close(fig)


def fig_localization_example(report: dict, out_dir: Path):
    """One qualitative timeline: per-frame manipulation prob vs ground truth."""
    ex = report.get("localization_example")
    if not ex:
        return
    plt = _style()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 2.8), sharex=True)
    for ax, mod, color in zip(axes, ("video", "audio"), CAT[:2]):
        t = np.linspace(0, ex["duration_sec"], len(ex[mod]["prob"]))
        ax.plot(t, ex[mod]["prob"], color=color, label=f"{mod} P(manipulated)")
        for s, e in ex[mod].get("gt_segments", []):
            ax.axvspan(s, e, color=GRID, zorder=0)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(mod)
        ax.legend(loc="upper right", fontsize=7)
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    _save(fig, out_dir, "results_localization")
    plt.close(fig)


# ---------------------------------------------------------------- demo data
def _demo_reports(seed: int = 7):
    """Synthetic-but-plausible results so the figure plumbing runs today."""
    rng = np.random.default_rng(seed)

    def preds(n, auc_ish):
        y = rng.integers(0, 2, n).tolist()
        s = [float(np.clip(rng.normal(0.28 + 0.44 * t * auc_ish, 0.18), 0.01, 0.99))
             for t in y]
        return y, s

    yv, sv = preds(400, 1.0)
    ya, sa = preds(400, 0.95)
    yq = rng.integers(0, 4, 400)
    pq = np.where(rng.random(400) < 0.72, yq, rng.integers(0, 4, 400))
    cm = np.zeros((4, 4), int)
    for t, p in zip(yq, pq):
        cm[t, p] += 1

    main = {
        "method": "DAVID-Net (ours)",
        "quadrant": {"confusion": cm.tolist()},
        "calibration": {"video_ece": 0.041, "audio_ece": 0.056},
        "preds": {"video": {"y_true": yv, "y_score": sv},
                  "audio": {"y_true": ya, "y_score": sa}},
        "localization_example": {
            "duration_sec": 8.0,
            "video": {"prob": np.clip(np.concatenate([
                rng.normal(0.1, 0.05, 30), rng.normal(0.85, 0.08, 25),
                rng.normal(0.12, 0.05, 45)]), 0, 1).tolist(),
                "gt_segments": [[2.4, 4.4]]},
            "audio": {"prob": np.clip(rng.normal(0.12, 0.06, 100), 0, 1).tolist(),
                      "gt_segments": []},
        },
    }
    yb, sb = preds(400, 0.7)
    baseline = {"method": "Baseline (frame CNN)",
                "modality": "video",
                "preds": {"video": {"y_true": yb, "y_score": sb}}}
    rob = {"sweeps": {
        "video": {"blur_sigma": [{"level": l, "auc": a} for l, a in
                                 zip([0, 1, 2, 4], [0.94, 0.91, 0.86, 0.78])],
                  "downscale": [{"level": l, "auc": a} for l, a in
                                zip([1, 2, 4, 8], [0.94, 0.92, 0.85, 0.74])]},
        "audio": {"snr_db": [{"level": l, "auc": a} for l, a in
                             zip([100, 20, 10, 0], [0.96, 0.93, 0.88, 0.79])]},
    }}
    ablation = [
        {"name": "Full (+ QACP)", "in_domain_auc": 0.962, "cross_dataset_auc": 0.874},
        {"name": "− QACP", "in_domain_auc": 0.955, "cross_dataset_auc": 0.801},
        {"name": "− MISMATCH class", "in_domain_auc": 0.958, "cross_dataset_auc": 0.842},
        {"name": "− sync module", "in_domain_auc": 0.941, "cross_dataset_auc": 0.823},
        {"name": "− disentanglement", "in_domain_auc": 0.949, "cross_dataset_auc": 0.816},
        {"name": "late fusion", "in_domain_auc": 0.938, "cross_dataset_auc": 0.782},
    ]
    return main, [main, baseline], rob, ablation


# ---------------------------------------------------------------- CLI
def generate_all(results_dir: str | None, out: str, demo: bool):
    out_dir = Path(out)
    if demo:
        print("generating DEMO figures (synthetic data — replace with real runs)")
        main, reports, rob, ablation = _demo_reports()
    else:
        rdir = Path(results_dir)
        reports = [json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(rdir.glob("*.json"))
                   if "robustness" not in p.name and "ablation" not in p.name]
        main = next((r for r in reports if r.get("method", "").startswith("david")), None)
        rob_p = next(iter(sorted(rdir.glob("robustness*.json"))), None)
        rob = json.loads(rob_p.read_text(encoding="utf-8")) if rob_p else None
        abl_p = next(iter(sorted(rdir.glob("ablation*.json"))), None)
        ablation = json.loads(abl_p.read_text(encoding="utf-8")) if abl_p else None

    if reports:
        fig_roc(reports, out_dir)
    if main:
        fig_reliability(main, out_dir)
        if "quadrant" in main:
            fig_confusion(main, out_dir)
        fig_localization_example(main, out_dir)
    if rob:
        fig_robustness(rob, out_dir)
    if ablation:
        fig_ablation(ablation, out_dir)
    print(f"figures -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures/generated")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    generate_all(args.results, args.out, args.demo)


if __name__ == "__main__":
    main()
