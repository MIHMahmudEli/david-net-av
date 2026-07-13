"""Evaluate a trained DAVID-Net on a manifest and dump the metric report.

Usage:
    python -m src.eval.evaluate --config configs/david_net.yaml \
        --checkpoint runs/david_net_epoch0.pt --manifest src/data/splits/test.jsonl
"""
from __future__ import annotations

import argparse
import json

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

    ds = AVDeepfakeDataset(manifest, cfg.shard_root, cfg.n_frames, cfg.audio_len)
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
        # raw predictions: the figure generator (src/eval/figures.py) builds
        # ROC / reliability / per-generator breakdowns from these
        "preds": {
            "clip_id": ids, "generator": gens,
            "video": {"y_true": yv, "y_score": pv},
            "audio": {"y_true": ya, "y_score": pa},
            "quadrant": {"y_true": yq, "y_pred": pq},
        },
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    report = evaluate(cfg, args.checkpoint, args.manifest)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f)
    printable = {k: v for k, v in report.items() if k != "preds"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
