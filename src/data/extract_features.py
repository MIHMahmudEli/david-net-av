"""One-time SSL feature extraction (Phase A of the DGX Spark compute plan).

Runs the frozen video/audio encoders over every clip in a manifest ONCE and caches
the token sequences to disk. Training then reads the cache and never touches the
backbones again — this is what makes the 15-20 ablation runs cheap on bandwidth-
limited hardware (docs/07_compute_and_hardware.md §2).

Usage:
    python -m src.data.extract_features --config configs/david_net.yaml \
        --manifest src/data/splits/train.jsonl --out data/feats/demo

Cache layout:  <out>/<clip_id>_vfeat.pt   (L_v, d)  video token sequence
               <out>/<clip_id>_afeat.pt   (L_a, d)  audio token sequence
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.datasets import AVDeepfakeDataset
from src.models.video_encoder import build_video_encoder
from src.models.audio_encoder import build_audio_encoder
from src.utils.config import load_config


@torch.no_grad()
def extract(cfg, manifest: str, out_dir: str, batch_size: int = 4, overwrite: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    venc = build_video_encoder(cfg).to(device).eval()
    aenc = build_audio_encoder(cfg).to(device).eval()

    ds = AVDeepfakeDataset(manifest, cfg.shard_root, cfg.n_frames, cfg.audio_len)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=lambda b: b,  # keep list of dicts; we batch manually below
    )

    n_done, n_skip = 0, 0
    for batch in dl:
        todo = [s for s in batch if overwrite or not (
            (out / f"{s['clip_id']}_vfeat.pt").exists()
            and (out / f"{s['clip_id']}_afeat.pt").exists())]
        n_skip += len(batch) - len(todo)
        if not todo:
            continue
        video = torch.stack([s["video"] for s in todo]).to(device)
        audio = torch.stack([s["audio"] for s in todo]).to(device)
        vfeat = venc(video)   # (B, L_v, d)
        afeat = aenc(audio)   # (B, L_a, d)
        for i, s in enumerate(todo):
            torch.save(vfeat[i].cpu().clone(), out / f"{s['clip_id']}_vfeat.pt")
            torch.save(afeat[i].cpu().clone(), out / f"{s['clip_id']}_afeat.pt")
        n_done += len(todo)
    print(f"extracted {n_done} clips, skipped {n_skip} already-cached -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    extract(cfg, args.manifest, args.out, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
