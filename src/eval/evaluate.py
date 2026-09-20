"""Evaluate a trained DAVID-Net on a manifest and dump the metric report.

Supports crash-proof evaluation via HuggingFace backup. Each dataset eval
is uploaded immediately so completed evals survive session death.

Usage:
    python -m src.eval.evaluate --config configs/david_net.yaml \
        --checkpoint runs/david_net_epoch0.pt --manifest src/data/splits/test.jsonl \
        --run-id stage1_seed42
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import AVDeepfakeDataset, collate, preflight_check
from src.eval.metrics import per_modality, quadrant_metrics, expected_calibration_error, per_group
from src.training.train import build_model, move, availability_masks
from src.utils.config import load_config


@torch.no_grad()
def evaluate(cfg, checkpoint: str, manifest: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    model.eval()

    root_dir = getattr(cfg, "root_dir", None)
    ds = AVDeepfakeDataset(manifest, cfg.shard_root, cfg.n_frames, cfg.audio_len,
                           root_dir=root_dir, train=False)
    preflight_check(ds, name="eval")
    max_clips = int(getattr(cfg, "eval_max_clips", 0) or 0)
    if max_clips and len(ds) > max_clips:
        # stratified subsample (per quadrant) so huge audio corpora (WaveFake: 134k
        # clips) evaluate in minutes instead of days; deterministic for the paper
        import random as _r
        by_q = {}
        for i, r in enumerate(ds.records):
            by_q.setdefault(r["quadrant"], []).append(i)
        rng = _r.Random(0)
        keep = []
        for q, idx in by_q.items():
            rng.shuffle(idx)
            keep += idx[: max(1, int(max_clips * len(idx) / len(ds)))]
        ds.records = [ds.records[i] for i in sorted(keep)]
        print(f"[eval] subsampled to {len(ds)} clips (eval_max_clips={max_clips})")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, collate_fn=collate)

    pv, pa, pq, yv, ya, yq, ids, gens = [], [], [], [], [], [], [], []
    loc_example = None
    for batch in dl:
        batch = move(batch, device)
        v_av, a_av = availability_masks(batch, 0.0)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(batch["video"], batch["audio"], v_avail=v_av, a_avail=a_av)
        pv += torch.sigmoid(out["logit_v"].float()).cpu().tolist()
        pa += torch.sigmoid(out["logit_a"].float()).cpu().tolist()
        pq += out["logit_quad"].argmax(-1).cpu().tolist()
        yv += batch["video_label"].cpu().tolist()
        ya += batch["audio_label"].cpu().tolist()
        yq += batch["quadrant"].cpu().tolist()
        ids += batch["clip_id"]
        gens += batch["generator"]
        # one qualitative localization timeline (first clip with a manipulated stream)
        if loc_example is None:
            for i in range(batch["video"].size(0)):
                if batch["video_label"][i] > 0 or batch["audio_label"][i] > 0:
                    dur = float(cfg.audio_len) / 16000.0
                    loc_example = {
                        "clip_id": batch["clip_id"][i], "duration_sec": dur,
                        "video": {"prob": torch.sigmoid(out["loc_v"][i].float()).cpu().tolist(),
                                  "gt_segments": [[0.0, dur]] if batch["video_label"][i] > 0 else []},
                        "audio": {"prob": torch.sigmoid(out["loc_a"][i].float()).cpu().tolist(),
                                  "gt_segments": [[0.0, dur]] if batch["audio_label"][i] > 0 else []},
                    }
                    break

    report = {
        "method": "david-net",
        "checkpoint": checkpoint,
        "test_manifest": manifest,
        "video": per_modality(yv, pv),
        "audio": per_modality(ya, pa),
        "quadrant": quadrant_metrics(yq, pq),
        "calibration": {
            "video_ece": expected_calibration_error(yv, pv),
            "audio_ece": expected_calibration_error(ya, pa),
        },
        "n": len(yv),
        "per_generator": {"video": per_group(yv, pv, gens), "audio": per_group(ya, pa, gens)},
        "per_quadrant": {"video": per_group(yv, pv, yq), "audio": per_group(ya, pa, yq)},
        "localization_example": loc_example,
        "preds": {
            "clip_id": ids, "generator": gens,
            "video": {"y_true": yv, "y_score": pv},
            "audio": {"y_true": ya, "y_score": pa},
            "quadrant": {"y_true": yq, "y_pred": pq},
        },
    }
    return report


def _upload_eval(run_id: str, ds_name: str, report: dict, local_dir: str):
    """Upload eval report to HF immediately after each dataset completes."""
    try:
        from src.utils.hf_backup import HFBackup
        backup = HFBackup(run_id=run_id, local_dir=local_dir)

        # Save report locally first
        out_dir = Path(local_dir) / "eval_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        local_path = out_dir / f"eval_{ds_name}.json"
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Upload to HF: runs/<run_id>/eval/<ds_name>.json
        api = backup._get_api()
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"{backup.base_path}/eval/{ds_name}.json",
            repo_id=backup.repo_id,
            repo_type=backup.repo_type,
        )
        print(f"  -> Uploaded eval_{ds_name}.json to HF")
    except Exception as e:
        print(f"  -> HF upload failed for {ds_name}: {e}")


def _check_completed_evals(run_id: str, local_dir: str) -> set:
    """Check HF repo for already-completed evals to skip."""
    try:
        from src.utils.hf_backup import HFBackup
        backup = HFBackup(run_id=run_id, local_dir=local_dir)
        api = backup._get_api()
        eval_path = f"{backup.base_path}/eval"
        files = api.list_repo_tree(
            backup.repo_id, path_in_repo=eval_path,
            repo_type=backup.repo_type, recursive=True
        )
        return {f.name.replace("eval_", "").replace(".json", "")
                for f in files if hasattr(f, "path") and f.path.endswith(".json")}
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="eval_report.json")
    ap.add_argument("--run-id", default=None, help="Run ID for HF backup")
    ap.add_argument("--ds-name", default=None, help="Dataset name for HF eval path")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="Skip if eval already uploaded to HF")
    ap.add_argument("--root-dir", default=None,
                    help="media root for this manifest (overrides cfg.root_dir)")
    ap.add_argument("--max-clips", type=int, default=0,
                    help="stratified subsample for very large corpora (0 = all)")
    args = ap.parse_args()

    # Check if already done
    if args.skip_if_done and args.run_id and args.ds_name:
        completed = _check_completed_evals(args.run_id, ".")
        if args.ds_name in completed:
            print(f"Eval for {args.ds_name} already exists on HF — skipping")
            return

    cfg = load_config(args.config)
    if args.root_dir:
        cfg.root_dir = args.root_dir
    if args.max_clips:
        cfg.eval_max_clips = args.max_clips
    report = evaluate(cfg, args.checkpoint, args.manifest)

    # Save locally
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f)

    # Upload to HF immediately
    if args.run_id and args.ds_name:
        _upload_eval(args.run_id, args.ds_name, report, ".")

    printable = {k: v for k, v in report.items() if k != "preds"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
