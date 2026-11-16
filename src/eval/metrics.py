"""Evaluation metrics: AUC, EER, per-modality F1, quadrant accuracy, ECE, localization AP.

Kept numpy/sklearn based so it runs standalone on saved prediction dumps.
"""
from __future__ import annotations

import numpy as np


def _safe_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def eer(y_true, y_score):
    """Equal Error Rate."""
    from sklearn.metrics import roc_curve
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def per_modality(y_true, y_score, thr: float = 0.5):
    from sklearn.metrics import f1_score, accuracy_score
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= thr).astype(int)
    return {
        "auc": _safe_auc(y_true, y_score),
        "eer": eer(y_true, y_score),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }


def quadrant_metrics(y_true, y_pred, n_classes: int = 4):
    from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
    labels = list(range(n_classes))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def per_group(y_true, y_score, groups, thr: float = 0.5, min_n: int = 5):
    """Per-generator / per-dataset breakdown of the binary metrics.

    Groups with a single class report AUC/EER as NaN but still get acc/n so the
    manuscript can state detection rate on all-fake generators."""
    from sklearn.metrics import accuracy_score
    y_true = np.asarray(y_true); y_score = np.asarray(y_score); groups = np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if m.sum() < min_n:
            continue
        yt, ys = y_true[m], y_score[m]
        out[str(g)] = {
            "n": int(m.sum()),
            "auc": _safe_auc(yt, ys),
            "eer": eer(yt, ys),
            "acc": float(accuracy_score(yt, (ys >= thr).astype(int))),
            "pos_rate": float(yt.mean()),
        }
    return out


def expected_calibration_error(y_true, y_prob, n_bins: int = 15):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        acc = (y_true[m] == (y_prob[m] >= 0.5)).mean()
        conf = y_prob[m].mean()
        ece += (m.sum() / len(y_prob)) * abs(acc - conf)
    return float(ece)


def _iou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def localization_ap(preds, gts, iou_thr: float = 0.5):
    """preds: list of (start, end, score). gts: list of (start, end). Simple AP@IoU."""
    if not gts:
        return float("nan")
    preds = sorted(preds, key=lambda x: -x[2])
    matched = set()
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    for i, (s, e, _) in enumerate(preds):
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched:
                continue
            iou = _iou((s, e), g)
            if iou > best:
                best, best_j = iou, j
        if best >= iou_thr and best_j >= 0:
            tp[i] = 1; matched.add(best_j)
        else:
            fp[i] = 1
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    recall = tp_c / len(gts)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if (recall >= t).any() else 0.0
        ap += p / 11
    return float(ap)
