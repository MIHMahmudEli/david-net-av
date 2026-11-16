"""Temporal Action Localization (TAL) evaluation for multimodal deepfake detection.

Implements official temporal forgery localization protocols for:
1. AV-Deepfake1M (Cai et al., 2024):
   - Fields: 'fake_segments', 'visual_fake_segments', 'audio_fake_segments'
2. LAV-DF (Cai et al., 2022):
   - Fields: 'modify_video', 'modify_audio', boundary intervals

Evaluates standard temporal detection benchmarks:
- Average Precision: AP@0.50, AP@0.75, AP@0.90, AP@0.95, and mAP@[0.50:0.05:0.95]
- Average Recall: AR@5, AR@10, AR@20, AR@50
- Modality breakdown: Video, Audio, and Multimodal (Joint) localization
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


def segment_iou(target_segments: np.ndarray, candidate_segments: np.ndarray) -> np.ndarray:
    """Compute pairwise Intersection over Union (IoU) between temporal intervals.

    Args:
        target_segments: Ground truth intervals (N, 2) in [start_sec, end_sec].
        candidate_segments: Predicted candidate intervals (M, 2) in [start_sec, end_sec].

    Returns:
        IoU matrix of shape (N, M).
    """
    if len(target_segments) == 0 or len(candidate_segments) == 0:
        return np.zeros((len(target_segments), len(candidate_segments)))

    # target: (N, 1, 2), candidate: (1, M, 2)
    t = target_segments[:, None, :]
    c = candidate_segments[None, :, :]

    start_max = np.maximum(t[..., 0], c[..., 0])
    end_min = np.minimum(t[..., 1], c[..., 1])
    intersection = np.maximum(0.0, end_min - start_max)

    t_dur = np.maximum(0.0, t[..., 1] - t[..., 0])
    c_dur = np.maximum(0.0, c[..., 1] - c[..., 0])
    union = t_dur + c_dur - intersection

    iou = np.zeros_like(intersection)
    valid = union > 0.0
    iou[valid] = intersection[valid] / union[valid]
    return iou


def compute_average_precision(
    ground_truth: Dict[str, List[Tuple[float, float]]],
    predictions: Dict[str, List[Tuple[float, float, float]]],
    iou_threshold: float = 0.50,
) -> float:
    """Compute Temporal Average Precision (AP) at a specified IoU threshold.

    Args:
        ground_truth: Mapping from video_id to list of (start, end) ground truth intervals.
        predictions: Mapping from video_id to list of (start, end, score) predictions.
        iou_threshold: Minimum IoU overlap required to count as True Positive.

    Returns:
        Average Precision (float in [0, 1]).
    """
    total_gt = sum(len(segs) for segs in ground_truth.values())
    if total_gt == 0:
        return float("nan")

    # Collect all predictions globally and sort descending by confidence score
    all_preds = []
    for vid, preds_list in predictions.items():
        for s, e, score in preds_list:
            all_preds.append((vid, float(s), float(e), float(score)))

    if not all_preds:
        return 0.0

    all_preds.sort(key=lambda x: -x[3])

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    # Track matched GT segments per video
    matched_gt: Dict[str, set] = {vid: set() for vid in ground_truth}

    for idx, (vid, p_start, p_end, score) in enumerate(all_preds):
        gt_segs = ground_truth.get(vid, [])
        if not gt_segs:
            fp[idx] = 1.0
            continue

        gt_arr = np.array(gt_segs, dtype=np.float64)
        pred_arr = np.array([[p_start, p_end]], dtype=np.float64)
        ious = segment_iou(gt_arr, pred_arr).squeeze(axis=1)  # shape (N_gt,)

        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])

        if best_iou >= iou_threshold:
            if best_gt_idx not in matched_gt[vid]:
                tp[idx] = 1.0
                matched_gt[vid].add(best_gt_idx)
            else:
                fp[idx] = 1.0  # Duplicate detection of same ground truth segment
        else:
            fp[idx] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / float(total_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Standard PASCAL VOC / ActivityNet 11-point or trapezoidal curve integration
    # Interpolate precision to be monotonically non-increasing
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def compute_average_recall(
    ground_truth: Dict[str, List[Tuple[float, float]]],
    predictions: Dict[str, List[Tuple[float, float, float]]],
    top_k: int = 50,
    iou_thresholds: np.ndarray = None,
) -> float:
    """Compute Average Recall (AR) across IoU thresholds for top-K predictions per video."""
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.50, 0.95, 10)

    total_gt = sum(len(segs) for segs in ground_truth.values())
    if total_gt == 0:
        return float("nan")

    recalls_per_iou = []

    for iou_thr in iou_thresholds:
        hits = 0
        for vid, gt_segs in ground_truth.items():
            if not gt_segs:
                continue
            preds = predictions.get(vid, [])
            preds_sorted = sorted(preds, key=lambda x: -x[2])[:top_k]
            if not preds_sorted:
                continue

            gt_arr = np.array(gt_segs, dtype=np.float64)
            pred_arr = np.array([[p[0], p[1]] for p in preds_sorted], dtype=np.float64)
            iou_mat = segment_iou(gt_arr, pred_arr)  # (N_gt, K)

            # A GT segment is recalled if at least one candidate exceeds iou_thr
            max_over_preds = iou_mat.max(axis=1) if iou_mat.shape[1] > 0 else np.zeros(len(gt_arr))
            hits += int(np.sum(max_over_preds >= iou_thr))

        recalls_per_iou.append(hits / float(total_gt))

    return float(np.mean(recalls_per_iou))


class TemporalLocalizationEvaluator:
    """Evaluates temporal forgery localization on interval-annotated video benchmarks."""

    def __init__(self, dataset_name: str = "AV-Deepfake1M"):
        self.dataset_name = dataset_name
        self.iou_thresholds = np.arange(0.50, 1.00, 0.05)  # 0.50, 0.55, ..., 0.95

    def parse_av_deepfake1m_annotations(self, annotation_path: str | Path) -> Dict[str, Dict[str, List]]:
        """Parse official AV-Deepfake1M annotation fields:

        fake_segments: joint multimodal manipulated spans [[start, end], ...]
        visual_fake_segments: video-only manipulated spans
        audio_fake_segments: audio-only manipulated spans
        """
        path = Path(annotation_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gt = {"video": {}, "audio": {}, "multimodal": {}}
        records = data if isinstance(data, list) else data.get("videos", data.get("data", []))

        for item in records:
            vid = item.get("video_id", item.get("id", item.get("clip_id")))
            if not vid:
                continue
            v_segs = [(float(s[0]), float(s[1])) for s in item.get("visual_fake_segments", [])]
            a_segs = [(float(s[0]), float(s[1])) for s in item.get("audio_fake_segments", [])]
            m_segs = [(float(s[0]), float(s[1])) for s in item.get("fake_segments", [])]

            if v_segs:
                gt["video"][vid] = v_segs
            if a_segs:
                gt["audio"][vid] = a_segs
            if m_segs:
                gt["multimodal"][vid] = m_segs

        return gt

    def evaluate_stream(
        self,
        gt_dict: Dict[str, List[Tuple[float, float]]],
        pred_dict: Dict[str, List[Tuple[float, float, float]]],
    ) -> Dict[str, float]:
        """Compute complete TAL suite (AP@0.50, 0.75, 0.90, 0.95, mAP, AR@5, 10, 20, 50)."""
        metrics = {}
        for thr in [0.50, 0.75, 0.90, 0.95]:
            metrics[f"AP@{thr:.2f}"] = compute_average_precision(gt_dict, pred_dict, iou_threshold=thr)

        all_aps = [compute_average_precision(gt_dict, pred_dict, iou_threshold=t) for t in self.iou_thresholds]
        metrics["mAP@[0.50:0.95]"] = float(np.nanmean(all_aps))

        for k in [5, 10, 20, 50]:
            metrics[f"AR@{k}"] = compute_average_recall(gt_dict, pred_dict, top_k=k, iou_thresholds=self.iou_thresholds)

        return metrics

    def run_benchmark(
        self,
        gt_streams: Dict[str, Dict[str, List[Tuple[float, float]]]],
        pred_streams: Dict[str, Dict[str, List[Tuple[float, float, float]]]],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate visual, acoustic, and multimodal localization streams."""
        results = {}
        for stream in ["video", "audio", "multimodal"]:
            if stream in gt_streams and stream in pred_streams:
                results[stream] = self.evaluate_stream(gt_streams[stream], pred_streams[stream])
        return results


def check_dataset_status() -> Dict[str, Any]:
    """Audit whether temporal localization benchmark datasets are accessible."""
    lav_df_recon = Path("dataset_recon/LAV-DF.json")
    status = {
        "AV-Deepfake1M": {
            "available": False,
            "status": "pending_dataset_acquisition",
            "annotation_fields": ["fake_segments", "visual_fake_segments", "audio_fake_segments"],
            "notes": "Target benchmark for multi-stream temporal localization."
        },
        "LAV-DF": {
            "available": False,
            "status": "pending_dataset_acquisition",
            "recon_archive_present": lav_df_recon.exists(),
            "notes": "Archive split across 20GB multi-part zip volumes on Kaggle; raw annotations not cached locally."
        },
        "FakeAVCeleb": {
            "available": True,
            "temporal_annotations": False,
            "status": "clip_level_labels_only",
            "notes": "Evaluated at clip level (863 test clips); lacks frame/timestamp interval annotations."
        }
    }
    return status


if __name__ == "__main__":
    print("Temporal Action Localization Evaluator initialized.")
    status = check_dataset_status()
    print(json.dumps(status, indent=2))
