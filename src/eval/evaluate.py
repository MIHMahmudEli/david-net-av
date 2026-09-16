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

from src.data.datasets import AVDeepfakeDataset, collate
from src.eval.metrics import per_modality, quadrant_metrics, expected_calibration_error
from src.training.train import build_model, move
from src.utils.config import load_config


@torch.no_grad()
def evaluate(cfg, checkpoint: str, manifest: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
    model.eval()

    root_dir = getattr(cfg, "root_dir", None)
    ds = AVDeepfakeDataset(manifest, cfg.shard_root, cfg.n_frames, cfg.audio_len,
                           root_dir=root_dir)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, collate_fn=collate)

    pv, pa, pq, yv, ya, yq, ids, gens = [], [], [], [], [], [], [], []
    for batch in dl:
        batch = move(batch, device)
        out = model(batch["video"], batch["audio"])
        pv += torch.sigmoid(out["logit_v"]).cpu().tolist()
        pa += torch.sigmoid(out["logit_a"]).cpu().tolist()
        pq += out["logit_quad"].argmax(-1).cpu().tolist()
        yv += batch["video_label"].cpu().tolist()
        ya += batch["audio_label"].cpu().tolist()
        yq += batch["quadrant"].cpu().tolist()
        ids += batch["clip_id"]
        gens += batch["generator"]

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
    args = ap.parse_args()

    # Check if already done
    if args.skip_if_done and args.run_id and args.ds_name:
        completed = _check_completed_evals(args.run_id, ".")
        if args.ds_name in completed:
            print(f"Eval for {args.ds_name} already exists on HF — skipping")
            return

    cfg = load_config(args.config)
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
