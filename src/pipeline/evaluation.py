"""Predictions, validation-fitted thresholds, metrics, bootstrap CIs and DeLong tests.

Tasks evaluated (each on the rows where its label is known and its input present)
  video    video-authenticity head      y = video_label, score = p_video (v_avail = 1)
  audio    audio-authenticity head      y = audio_label, score = p_audio (a_avail = 1)
  clip     "anything fake in the clip"  y = clip_label,  score = max of available heads
  quadrant 4-class RVRA/RVFA/FVRA/FVFA  (both streams present, quadrant known)

Positive class = FAKE throughout. Threshold-dependent metrics (accuracy, precision,
recall/sensitivity, specificity, F1, confusion matrix) use thresholds fitted on the
VALIDATION split (policy: EER point by default) and frozen for test and every
cross-dataset corpus -- the test set never influences a threshold.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.pipeline.features import QUAD_NAMES

TASKS = ("video", "audio", "clip")


# ====================================================================== predictions
@torch.no_grad()
def predict(model, loader, device: str, amp_dtype=None, exp_id: str = "",
            model_version: str = "", split: str = "") -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        v_av = batch["v_avail"].to(device)
        a_av = batch["a_avail"].to(device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            dtype=amp_dtype or torch.float16,
                            enabled=device == "cuda" and amp_dtype is not None):
            out = model(video, audio, v_avail=v_av, a_avail=a_av)
        pv = torch.sigmoid(out["logit_v"].float()).cpu().numpy()
        pa = torch.sigmoid(out["logit_a"].float()).cpu().numpy()
        pq = (torch.softmax(out["logit_quad"].float(), -1).cpu().numpy()
              if out.get("logit_quad") is not None else np.full((len(pv), 4), np.nan))
        va, aa = batch["v_avail"].numpy(), batch["a_avail"].numpy()
        for i in range(len(pv)):
            p_video = float(pv[i]) if va[i] > 0 else np.nan
            p_audio = float(pa[i]) if aa[i] > 0 else np.nan
            avail = [p for p in (p_video, p_audio) if not np.isnan(p)]
            q = int(batch["quadrant"][i])
            row = {
                "sample_id": batch["clip_id"][i], "experiment_id": exp_id,
                "model_version": model_version, "split": split,
                "dataset": batch["dataset"][i], "generator": batch["generator"][i],
                "race": batch["race"][i], "gender": batch["gender"][i],
                "v_avail": int(va[i] > 0), "a_avail": int(aa[i] > 0),
                "video_label": int(batch["video_label"][i]),
                "audio_label": int(batch["audio_label"][i]),
                "clip_label": int(batch["clip_label"][i]),
                "quadrant_label": QUAD_NAMES[q] if q >= 0 else "",
                "p_video": p_video, "p_audio": p_audio,
                "p_clip": max(avail) if avail else np.nan,
            }
            for j, name in enumerate(QUAD_NAMES):
                row[f"p_quad_{name}"] = float(pq[i, j])
            rows.append(row)
    return pd.DataFrame(rows)


def task_frame(df: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "video":
        m = df["video_label"].isin([0, 1]) & (df["v_avail"] == 1) & df["p_video"].notna()
        return df.loc[m, "video_label"].to_numpy(), df.loc[m, "p_video"].to_numpy()
    if task == "audio":
        m = df["audio_label"].isin([0, 1]) & (df["a_avail"] == 1) & df["p_audio"].notna()
        return df.loc[m, "audio_label"].to_numpy(), df.loc[m, "p_audio"].to_numpy()
    if task == "clip":
        m = df["clip_label"].isin([0, 1]) & df["p_clip"].notna()
        return df.loc[m, "clip_label"].to_numpy(), df.loc[m, "p_clip"].to_numpy()
    raise KeyError(task)


# ====================================================================== thresholds
def eer_point(y, p) -> tuple[float, float]:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def fit_threshold(y, p, policy: str) -> float:
    y, p = np.asarray(y), np.asarray(p)
    if policy == "fixed_0.5" or len(np.unique(y)) < 2:
        return 0.5
    if policy == "val_eer":
        return float(min(1.0, eer_point(y, p)[1]))
    if policy == "val_youden":
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y, p)
        return float(min(1.0, thr[int(np.argmax(tpr - fpr))]))
    raise KeyError(policy)


def fit_thresholds(val_df: pd.DataFrame, policy: str) -> dict:
    out = {}
    for t in TASKS:
        y, p = task_frame(val_df, t)
        out[t] = fit_threshold(y, p, policy) if len(y) else 0.5
    return out


# ====================================================================== metrics
def stratified_bootstrap_auc(y, p, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats.append(roc_auc_score(y[idx], p[idx]))
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def ece(y, p, n_bins: int = 15) -> float:
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def binary_metrics(y, p, thr: float, n_boot: int = 1000, n_bins: int = 15,
                   seed: int = 0) -> dict:
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 roc_auc_score)
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    pred = (p >= thr).astype(int)
    two = n_pos > 0 and n_neg > 0
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if tp + fn else math.nan
    tnr = tn / (tn + fp) if tn + fp else math.nan
    prec = tp / (tp + fp) if tp + fp else math.nan
    out = {
        "n": int(len(y)), "n_fake": n_pos, "n_real": n_neg, "threshold": float(thr),
        "auc": float(roc_auc_score(y, p)) if two else math.nan,
        "auc_ci_low": math.nan, "auc_ci_high": math.nan,
        "pr_auc": float(average_precision_score(y, p)) if two else math.nan,
        "eer": eer_point(y, p)[0] if two else math.nan,
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.nanmean([tpr, tnr])) if two else math.nan,
        "precision": float(prec), "recall": float(tpr), "sensitivity": float(tpr),
        "specificity": float(tnr),
        "f1": float(f1_score(y, pred, zero_division=0)) if n_pos else math.nan,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)) if two else math.nan,
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece(y, p, n_bins),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "detection_rate": float(tpr) if n_pos else math.nan,   # the only valid number
        "auc_defined": bool(two),                               # for fakes-only corpora
    }
    if two and n_boot:
        out["auc_ci_low"], out["auc_ci_high"] = stratified_bootstrap_auc(y, p, n_boot, seed)
    return out


def multiclass_metrics(df: pd.DataFrame) -> Optional[dict]:
    from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                                 roc_auc_score, balanced_accuracy_score)
    m = (df["quadrant_label"] != "") & (df["v_avail"] == 1) & (df["a_avail"] == 1)
    cols = [f"p_quad_{n}" for n in QUAD_NAMES]
    sub = df.loc[m]
    if sub.empty or sub[cols].isna().all().all():
        return None
    y = sub["quadrant_label"].map({n: i for i, n in enumerate(QUAD_NAMES)}).to_numpy()
    P = sub[cols].to_numpy()
    pred = P.argmax(1)
    labels = list(range(4))
    rep = classification_report(y, pred, labels=labels, target_names=QUAD_NAMES,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / cm.sum(1, keepdims=True)
    try:
        auc = float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        auc = math.nan
    return {
        "n": int(len(y)), "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=labels, zero_division=0)),
        "macro_auc_ovr": auc,
        "per_class": {n: {"precision": rep[n]["precision"], "recall": rep[n]["recall"],
                          "f1": rep[n]["f1-score"], "support": int(rep[n]["support"])}
                      for n in QUAD_NAMES},
        "confusion": cm.tolist(), "confusion_normalized": np.nan_to_num(cm_norm).tolist(),
    }


def group_metrics(df: pd.DataFrame, task: str, by: str, thr: float, min_n: int = 20) -> list[dict]:
    """Per-generator / per-subgroup detection rates and AUC (against the split's reals)."""
    from sklearn.metrics import roc_auc_score
    y_all, p_all = task_frame(df, task)
    label = {"video": "video_label", "audio": "audio_label", "clip": "clip_label"}[task]
    rows = []
    reals = df[df[label] == 0]
    for g, sub in df.groupby(by):
        yy, pp = task_frame(sub, task)
        if len(yy) < 1:
            continue
        row = {"group": g, "n": int(len(yy)), "n_fake": int((yy == 1).sum()),
               "detection_rate": float(((pp >= thr) & (yy == 1)).sum() / max(1, (yy == 1).sum()))
               if (yy == 1).any() else math.nan,
               "mean_score": float(np.mean(pp))}
        fakes = sub[sub[label] == 1]
        if len(fakes) and len(reals) and by == "generator":
            yf, pf = task_frame(fakes, task)
            yr, pr = task_frame(reals, task)
            if len(yf) and len(yr):
                row["auc_vs_reals"] = float(roc_auc_score(np.r_[yf, yr], np.r_[pf, pr]))
        elif (yy == 0).any() and (yy == 1).any() and len(yy) >= min_n:
            row["auc"] = float(roc_auc_score(yy, pp))
        rows.append(row)
    return rows


def evaluate_frame(df: pd.DataFrame, thresholds: dict, cfg_eval: dict, seed: int = 0) -> dict:
    out = {"n": int(len(df))}
    for t in TASKS:
        y, p = task_frame(df, t)
        if len(y):
            out[t] = binary_metrics(y, p, thresholds.get(t, 0.5), cfg_eval["bootstrap"],
                                    cfg_eval["ece_bins"], seed)
    q = multiclass_metrics(df)
    if q is not None:
        out["quadrant"] = q
    aucs = [out[t]["auc"] for t in ("video", "audio") if t in out and out[t]["auc_defined"]]
    out["mean_auc"] = float(np.mean(aucs)) if aucs else math.nan
    return out


# ====================================================================== DeLong
def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y, p1, p2) -> dict:
    """Two correlated ROC AUCs on the same samples (DeLong 1988; Sun & Xu 2014)."""
    from scipy.stats import norm
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    if m == 0 or n == 0:
        return {"auc1": math.nan, "auc2": math.nan, "z": math.nan, "p_value": math.nan}
    preds = np.vstack([p1, p2])
    v01, v10, aucs = [], [], []
    for s in preds:
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(s)
        auc = (tz[pos].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v01.append((tz[pos] - tx) / n)
        v10.append(1.0 - (tz[neg] - ty) / m)
    s01, s10 = np.cov(np.vstack(v01)), np.cov(np.vstack(v10))
    S = s01 / m + s10 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / math.sqrt(var) if var > 0 else math.nan
    p = 2 * (1 - norm.cdf(abs(z))) if var > 0 else math.nan
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff),
            "z": float(z), "p_value": float(p)}


def holm_bonferroni(pvals: dict) -> dict:
    """Family-wise error control across the ablation family."""
    items = sorted((p, k) for k, p in pvals.items() if not math.isnan(p))
    m = len(items)
    adj, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    for k, p in pvals.items():
        adj.setdefault(k, math.nan)
    return adj


# ====================================================================== aggregation
def mean_std_ci(values: list[float], level: float = 0.95) -> dict:
    """Across seeds: mean, sample std, and a Student-t CI (n is small: 3 seeds)."""
    from scipy.stats import t
    v = np.array([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return {"mean": math.nan, "std": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    mean = float(v.mean())
    if len(v) == 1:
        return {"mean": mean, "std": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 1}
    std = float(v.std(ddof=1))
    h = float(t.ppf(0.5 + level / 2, len(v) - 1) * std / math.sqrt(len(v)))
    return {"mean": mean, "std": std, "ci_low": mean - h, "ci_high": mean + h, "n": int(len(v))}
