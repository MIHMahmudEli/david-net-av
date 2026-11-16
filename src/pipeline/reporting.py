"""Manuscript figures (PNG 300 dpi + vector PDF) and tables (CSV + LaTeX booktabs).

Everything here reads ONLY saved artifacts (training_history.csv, train_steps.jsonl,
predictions/*.csv, metrics/*.json), so every figure and table can be regenerated
without retraining: `python -m src.pipeline.reporting --results <dir>`.

Style: IEEE column widths (3.5 in single / 7.16 in double), serif 8 pt, one y-axis,
thin marks, recessive grid, legends whenever >= 2 series. Categorical colours are the
first four slots of a CVD-validated palette (blue, orange, aqua, yellow), assigned to
the same entity in every figure; magnitude uses a single-hue blue ramp.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.pipeline.evaluation import mean_std_ci

SERIES = {"video": "#2a78d6", "audio": "#eb6834", "clip": "#1baf7a", "extra": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SINGLE, DOUBLE = 3.5, 7.16


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    for p in paths:
        fig.savefig(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


# ============================================================ per-experiment figures
def fig_training(history: pd.DataFrame, out: Path, stage: str) -> list[Path]:
    plt = style()
    paths = []
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    if "train/total" in history:
        ax.plot(history["epoch"], history["train/total"], color=SERIES["video"], label="Training")
    vcol = "val/loss" if "val/loss" in history else "val/qacp_loss"
    if vcol in history:
        ax.plot(history["epoch"], history[vcol], color=SERIES["audio"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    paths += save(fig, out, "training_loss")
    aucs = [c for c in ("val/video_auc", "val/audio_auc", "val/clip_auc") if c in history]
    if aucs:
        fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
        for c in aucs:
            task = c.split("/")[1].split("_")[0]
            ax.plot(history["epoch"], history[c], color=SERIES[task], marker="o",
                    markersize=3, label=f"{task.capitalize()} AUC")
        if "val/quad_macro_f1" in history:
            ax.plot(history["epoch"], history["val/quad_macro_f1"], color=SERIES["extra"],
                    marker="s", markersize=3, label="Quadrant macro-F1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation score")
        ax.legend()
        paths += save(fig, out, "validation_metrics")
    return paths


def fig_lr(steps: pd.DataFrame, out: Path) -> list[Path]:
    plt = style()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
    ax.plot(steps["step"], steps["lr"], color=SERIES["video"])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    return save(fig, out, "learning_rate")


def _roc_pr(df: pd.DataFrame, out: Path, kind: str, suffix: str = "") -> list[Path]:
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    from src.pipeline.evaluation import task_frame
    plt = style()
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.85))
    drawn = 0
    for task in ("video", "audio", "clip"):
        y, p = task_frame(df, task)
        if len(np.unique(y)) < 2:
            continue
        if kind == "roc":
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=SERIES[task], label=f"{task.capitalize()} (AUC {auc(fpr, tpr):.3f})")
        else:
            pr, rc, _ = precision_recall_curve(y, p)
            from sklearn.metrics import average_precision_score
            ax.plot(rc, pr, color=SERIES[task],
                    label=f"{task.capitalize()} (AP {average_precision_score(y, p):.3f})")
        drawn += 1
    if not drawn:
        plt.close(fig)
        return []
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.6, linestyle=":")
        ax.set_xlabel("False positive rate (real flagged as fake)")
        ax.set_ylabel("True positive rate (fake detected)")
    else:
        ax.set_xlabel("Recall (fake)")
        ax.set_ylabel("Precision (fake)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right" if kind == "roc" else "lower left")
    return save(fig, out, f"{'roc_curve' if kind == 'roc' else 'precision_recall_curve'}{suffix}")


def fig_calibration(df: pd.DataFrame, out: Path, n_bins: int = 10, suffix: str = "") -> list[Path]:
    from src.pipeline.evaluation import task_frame
    plt = style()
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.85))
    bins = np.linspace(0, 1, n_bins + 1)
    for task in ("video", "audio"):
        y, p = task_frame(df, task)
        if len(y) < 10:
            continue
        idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() >= 5:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, color=SERIES[task], marker="o", markersize=3, label=task.capitalize())
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.6, linestyle=":", label="Perfect calibration")
    ax.set_xlabel("Predicted probability of fake")
    ax.set_ylabel("Observed fraction fake")
    ax.legend(loc="upper left")
    return save(fig, out, f"calibration_curve{suffix}")


def fig_score_distribution(df: pd.DataFrame, thresholds: dict, out: Path, suffix: str = "") -> list[Path]:
    """Error distribution for a classifier: score histograms of real vs fake per head,
    with the validation-fitted threshold marked."""
    from src.pipeline.evaluation import task_frame
    plt = style()
    tasks = [t for t in ("video", "audio") if len(task_frame(df, t)[0])]
    if not tasks:
        return []
    fig, axes = plt.subplots(1, len(tasks), figsize=(DOUBLE if len(tasks) > 1 else SINGLE, 2.1),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        y, p = task_frame(df, task)
        b = np.linspace(0, 1, 26)
        ax.hist(p[y == 0], bins=b, color=SERIES["video"], alpha=0.75, label="Real", density=True)
        ax.hist(p[y == 1], bins=b, color=SERIES["audio"], alpha=0.75, label="Fake", density=True)
        ax.axvline(thresholds.get(task, 0.5), color=INK, linewidth=0.8, linestyle="--",
                   label="Val. threshold")
        ax.set_xlabel(f"{task.capitalize()} score")
        ax.set_ylabel("Density")
        ax.legend()
    return save(fig, out, f"score_distribution{suffix}")


def fig_confusion(q: dict, out: Path, suffix: str = "") -> list[Path]:
    plt = style()
    from src.pipeline.features import QUAD_NAMES
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.9))
    for ax, key, title in ((axes[0], "confusion", "Counts"),
                           (axes[1], "confusion_normalized", "Row-normalized (recall)")):
        m = np.array(q[key], dtype=float)
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() if key == "confusion" else 1)
        for i in range(4):
            for j in range(4):
                v = m[i, j]
                txt = f"{int(v)}" if key == "confusion" else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 * (m.max() if key == "confusion" else 1) else INK)
        ax.set_xticks(range(4), QUAD_NAMES)
        ax.set_yticks(range(4), QUAD_NAMES)
        ax.set_xlabel("Predicted quadrant")
        ax.set_ylabel("True quadrant")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return save(fig, out, f"confusion_matrix{suffix}")


def fig_per_class(q: dict, out: Path, suffix: str = "") -> list[Path]:
    plt = style()
    from src.pipeline.features import QUAD_NAMES
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    w = 0.26
    x = np.arange(4)
    for k, (metric, color) in enumerate((("precision", SERIES["video"]), ("recall", SERIES["audio"]),
                                         ("f1", SERIES["clip"]))):
        vals = [q["per_class"][n][metric] for n in QUAD_NAMES]
        ax.bar(x + (k - 1) * w, vals, width=w - 0.02, color=color, label=metric.capitalize())
    ax.set_xticks(x, QUAD_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    return save(fig, out, f"per_class_performance{suffix}")


def fig_per_generator(rows: list[dict], out: Path, suffix: str = "") -> list[Path]:
    rows = [r for r in rows if r["group"] != "real" and r.get("n_fake", 0)]
    if not rows:
        return []
    plt = style()
    rows = sorted(rows, key=lambda r: r["detection_rate"])
    fig, ax = plt.subplots(figsize=(SINGLE, 0.28 * len(rows) + 0.8))
    ax.barh([r["group"] for r in rows], [r["detection_rate"] for r in rows],
            color=SERIES["video"], height=0.6)
    for i, r in enumerate(rows):
        ax.text(r["detection_rate"] + 0.01, i, f"{r['detection_rate']:.2f} (n={r['n_fake']})",
                va="center", fontsize=6.5, color=INK2)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Detection rate at validation threshold")
    return save(fig, out, f"per_generator{suffix}")


def experiment_figures(exp_dir: Path) -> list[Path]:
    """All per-experiment figures from the experiment's saved files."""
    figs = exp_dir / "figures"
    out = []
    hist = exp_dir / "metrics" / "training_history.csv"
    stage = json.loads((exp_dir / "configs" / "config.json").read_text())["experiment"]["stage"] \
        if (exp_dir / "configs" / "config.json").exists() else ""
    if hist.exists():
        out += fig_training(pd.read_csv(hist), figs, stage)
    steps = exp_dir / "logs" / "train_steps.jsonl"
    if steps.exists():
        s = pd.read_json(steps, lines=True)
        if len(s):
            out += fig_lr(s, figs)
    pred = exp_dir / "predictions" / "test.csv"
    met = exp_dir / "metrics" / "test_metrics.json"
    thr = exp_dir / "metrics" / "thresholds.json"
    if pred.exists():
        df = pd.read_csv(pred)
        th = json.loads(thr.read_text()) if thr.exists() else {}
        out += _roc_pr(df, figs, "roc") + _roc_pr(df, figs, "pr")
        out += fig_calibration(df, figs) + fig_score_distribution(df, th, figs)
    if met.exists():
        m = json.loads(met.read_text())
        if m.get("quadrant"):
            out += fig_confusion(m["quadrant"], figs) + fig_per_class(m["quadrant"], figs)
    gen = exp_dir / "metrics" / "per_generator.csv"
    if gen.exists():
        out += fig_per_generator(pd.read_csv(gen).to_dict("records"), figs)
    return out


# ============================================================ tables
def _fmt(ms: dict, digits: int = 3) -> str:
    if ms["n"] == 0 or math.isnan(ms["mean"]):
        return "--"
    if ms["n"] == 1 or math.isnan(ms["std"]):
        return f"{ms['mean']:.{digits}f}"
    return f"{ms['mean']:.{digits}f} $\\pm$ {ms['std']:.{digits}f}"


def to_latex(df: pd.DataFrame, caption: str, label: str, path: Path, col_fmt: Optional[str] = None):
    cols = list(df.columns)
    col_fmt = col_fmt or "l" + "c" * (len(cols) - 1)
    esc = lambda s: str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{col_fmt}}}", "\\toprule",
             " & ".join(esc(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(r[c]) if "$" in str(r[c]) else esc(r[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, out: Path, name: str, caption: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"{name}.csv"
    tex = out / f"{name}.tex"
    df.to_csv(csv, index=False)
    to_latex(df, caption, f"tab:{name}", tex)
    return [csv, tex]


MAIN_METRICS = [("video", "auc", "Video AUC"), ("audio", "auc", "Audio AUC"),
                ("clip", "auc", "Clip AUC"), ("video", "eer", "Video EER"),
                ("audio", "eer", "Audio EER"), ("video", "f1", "Video F1"),
                ("audio", "f1", "Audio F1"), ("clip", "balanced_accuracy", "Clip bal. acc."),
                ("quadrant", "macro_f1", "Quad. macro-F1")]


def _metric(m: dict, task: str, key: str):
    return (m.get(task) or {}).get(key, math.nan)


def aggregate(results: list[dict], cfg_eval: dict) -> dict:
    """results: one dict per completed experiment with keys name, seed, group,
    components, test (metrics), cross {corpus: metrics}, preds_test (DataFrame)."""
    by = {}
    for r in results:
        by.setdefault(r["name"], []).append(r)
    return by


def main_table(by: dict, names: list[str], level: float) -> pd.DataFrame:
    rows = []
    for n in names:
        if n not in by:
            continue
        row = {"Model": n, "Seeds": len(by[n])}
        for task, key, label in MAIN_METRICS:
            row[label] = _fmt(mean_std_ci([_metric(r["test"], task, key) for r in by[n]], level))
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_table(by: dict, name: str, level: float) -> pd.DataFrame:
    from src.pipeline.features import QUAD_NAMES
    rows = []
    reps = [r for r in by.get(name, []) if r["test"].get("quadrant")]
    for q in QUAD_NAMES:
        row = {"Class": q}
        for k in ("precision", "recall", "f1"):
            row[k.capitalize() if k != "f1" else "F1"] = _fmt(mean_std_ci(
                [r["test"]["quadrant"]["per_class"][q][k] for r in reps], level))
        row["Support"] = reps[0]["test"]["quadrant"]["per_class"][q]["support"] if reps else 0
        rows.append(row)
    return pd.DataFrame(rows)


def cross_table(by: dict, names: list[str], corpora: list[str], level: float) -> pd.DataFrame:
    """AUC where both classes exist; detection rate (marked DR) for fakes-only corpora."""
    rows = []
    for n in names:
        if n not in by:
            continue
        row = {"Model": n}
        for c in corpora:
            vals, kind = [], "AUC"
            for r in by[n]:
                m = (r.get("cross") or {}).get(c)
                if not m:
                    continue
                task = "audio" if "audio" in m and "video" not in m else ("video" if "video" in m else "clip")
                tm = m.get(task, {})
                if tm.get("auc_defined"):
                    vals.append(tm["auc"])
                else:
                    kind = "DR"
                    vals.append(tm.get("detection_rate", math.nan))
            row[f"{c} ({kind})"] = _fmt(mean_std_ci(vals, level))
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_table(by: dict, ref: str, names: list[str], level: float,
                   delong: Optional[dict] = None) -> pd.DataFrame:
    from src.pipeline.plan import COMPONENTS
    short = {"qacp": "QACP", "sync": "Sync", "disentangle": "Dis.", "localization": "Loc.",
             "multitask": "MT", "modality_dropout": "MD", "mismatch_class": "MM",
             "copy_synthesis": "CS", "self_blending": "SB"}
    rows = []
    ref_auc = mean_std_ci([_metric(r["test"], "clip", "auc") for r in by.get(ref, [])], level)
    for n in [ref] + [x for x in names if x != ref]:
        if n not in by:
            continue
        comps = by[n][0]["components"]
        row = {"Experiment": n}
        for c in COMPONENTS:
            row[short[c]] = "\\checkmark" if comps.get(c) else "--"
        v = mean_std_ci([_metric(r["test"], "video", "auc") for r in by[n]], level)
        a = mean_std_ci([_metric(r["test"], "audio", "auc") for r in by[n]], level)
        cl = mean_std_ci([_metric(r["test"], "clip", "auc") for r in by[n]], level)
        row["Video AUC"], row["Audio AUC"], row["Clip AUC"] = _fmt(v), _fmt(a), _fmt(cl)
        crossv = [np.nanmean([m.get("video", m.get("audio", {})).get("auc", math.nan)
                              for m in (r.get("cross") or {}).values()
                              if m.get("video", m.get("audio", {})).get("auc_defined")] or [math.nan])
                  for r in by[n]]
        row["Cross-dataset AUC"] = _fmt(mean_std_ci(crossv, level))
        row["$\\Delta$ clip AUC"] = ("--" if n == ref or math.isnan(cl["mean"]) or math.isnan(ref_auc["mean"])
                                     else f"{cl['mean'] - ref_auc['mean']:+.3f}")
        if delong is not None and n != ref:
            row["Sig. seeds (Holm $p<0.05$)"] = delong.get(n, "--")
        rows.append(row)
    return pd.DataFrame(rows)


def hyperparameter_table(cfg: dict) -> pd.DataFrame:
    from src.pipeline.config import flatten
    keep = ("model.", "train.qacp.", "train.stage1.", "train.phase_b.", "features.", "data.")
    f = flatten({k: cfg[k] for k in ("model", "train", "features", "data") if k in cfg})
    rows = [{"Parameter": k, "Value": json.dumps(v) if isinstance(v, (list, dict)) else v}
            for k, v in sorted(f.items()) if k.startswith(keep) and "cross_datasets" not in k]
    return pd.DataFrame(rows)


def fig_ablation(by: dict, ref: str, names: list[str], out: Path, level: float) -> list[Path]:
    plt = style()
    rows = []
    for n in [ref] + [x for x in names if x != ref]:
        if n in by:
            ms = mean_std_ci([_metric(r["test"], "clip", "auc") for r in by[n]], level)
            if not math.isnan(ms["mean"]):
                rows.append((n.replace("davidnet-", "").replace("davidnet", "full"), ms))
    if len(rows) < 2:
        return []
    fig, ax = plt.subplots(figsize=(SINGLE, 0.3 * len(rows) + 0.8))
    y = np.arange(len(rows))[::-1]
    for yi, (n, ms) in zip(y, rows):
        err = None if math.isnan(ms["ci_low"]) else [[ms["mean"] - ms["ci_low"]], [ms["ci_high"] - ms["mean"]]]
        ax.errorbar(ms["mean"], yi, xerr=err, fmt="o", markersize=4,
                    color=SERIES["video"] if n != "full" else SERIES["audio"], capsize=2, linewidth=1)
    ax.axvline(rows[0][1]["mean"], color=INK2, linewidth=0.6, linestyle=":")
    ax.set_yticks(y, [n for n, _ in rows])
    ax.set_xlabel(f"Test clip-level AUC (mean, {int(level * 100)}% t-CI over seeds)")
    return save(fig, out, "ablation_results")


def fig_cross(by: dict, names: list[str], corpora: list[str], out: Path, level: float) -> list[Path]:
    plt = style()
    names = [n for n in names if n in by][:4]
    if not names:
        return []
    colors = [SERIES["video"], SERIES["audio"], SERIES["clip"], SERIES["extra"]]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
    w = 0.8 / len(names)
    x = np.arange(len(corpora))
    for k, n in enumerate(names):
        means, errs = [], []
        for c in corpora:
            vals = []
            for r in by[n]:
                m = (r.get("cross") or {}).get(c) or {}
                t = m.get("video") if "video" in m else m.get("audio", {})
                vals.append(t.get("auc") if t.get("auc_defined") else t.get("detection_rate", math.nan))
            ms = mean_std_ci(vals, level)
            means.append(ms["mean"])
            errs.append(0 if math.isnan(ms["std"]) else ms["std"])
        ax.bar(x + (k - (len(names) - 1) / 2) * w, means, width=w - 0.02,
               yerr=errs if any(errs) else None,
               color=colors[k], capsize=2, label=n, error_kw={"linewidth": 0.8})
    ax.set_xticks(x, corpora)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color=INK2, linewidth=0.6, linestyle=":")
    ax.set_ylabel("AUC (fakes-only corpora: detection rate)")
    ax.legend(ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, 1.2))
    return save(fig, out, "cross_dataset_results")


def fig_pooled_roc(by: dict, names: list[str], out: Path) -> list[Path]:
    """Test ROC of the main models, one line per seed (thin) in the model's colour."""
    from sklearn.metrics import roc_curve
    from src.pipeline.evaluation import task_frame
    plt = style()
    colors = [SERIES["video"], SERIES["audio"], SERIES["clip"], SERIES["extra"]]
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.85))
    drawn = 0
    for k, n in enumerate([n for n in names if n in by][:4]):
        for j, r in enumerate(by[n]):
            if r.get("preds_test") is None:
                continue
            y, p = task_frame(r["preds_test"], "clip")
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=colors[k], linewidth=0.9, alpha=0.9,
                    label=n if j == 0 else None)
            drawn += 1
    if not drawn:
        plt.close(fig)
        return []
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.6, linestyle=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    return save(fig, out, "roc_main_models")
