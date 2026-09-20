"""Robustness evaluation: AUC under controlled degradations (RQ5, Fig. robustness).

Tensor-space degradations (no ffmpeg needed, applied on preprocessed shards):
  video : gaussian blur, downscale-upscale (resolution loss), jpeg-like quantization
  audio : additive white noise at target SNR (dB)

Compression sweeps that need a real codec (H.264 CRF) are done by re-encoding the
raw clips with ffmpeg and re-running preprocess+evaluate — see HANDOFF.md §Robustness.

Usage:
    python -m src.eval.robustness --config configs/david_net.yaml \
        --checkpoint runs/best.pt --manifest src/data/splits/fakeavceleb/test.jsonl \
        --out results/robustness_david-net.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.datasets import AVDeepfakeDataset, collate
from src.eval.metrics import per_modality
from src.training.train import build_model, move
from src.utils.config import load_config


# ------------------------------------------------------------- degradations
def video_blur(frames: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur via separable conv. frames: (B, T, 3, H, W)."""
    if sigma <= 0:
        return frames
    k = max(3, int(sigma * 4) | 1)
    x = torch.arange(k, dtype=torch.float32, device=frames.device) - k // 2
    g = torch.exp(-x.pow(2) / (2 * sigma * sigma))
    g = (g / g.sum()).view(1, 1, 1, k)
    b, t, c, h, w = frames.shape
    f = frames.flatten(0, 1).reshape(b * t * c, 1, h, w)
    f = F.conv2d(f, g, padding=(0, k // 2))
    f = F.conv2d(f, g.transpose(2, 3), padding=(k // 2, 0))
    return f.reshape(b, t, c, h, w)


def video_downscale(frames: torch.Tensor, factor: int) -> torch.Tensor:
    """Resolution loss: downscale by `factor` then upscale back."""
    if factor <= 1:
        return frames
    b, t, c, h, w = frames.shape
    f = frames.flatten(0, 1)
    f = F.interpolate(f, scale_factor=1 / factor, mode="bilinear", align_corners=False)
    f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
    return f.reshape(b, t, c, h, w)


def video_quantize(frames: torch.Tensor, levels: int) -> torch.Tensor:
    """JPEG-like intensity quantization (blockiness proxy)."""
    if levels >= 256:
        return frames
    return (frames * levels).round() / levels


def audio_noise(wave: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Additive white noise at a target SNR. wave: (B, N)."""
    if snr_db >= 100:
        return wave
    sig_pow = wave.pow(2).mean(dim=1, keepdim=True).clamp(min=1e-10)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    return wave + torch.randn_like(wave) * noise_pow.sqrt()


# level grids reported in the paper (Fig. robustness)
VIDEO_LEVELS = {
    "blur_sigma": [0, 1, 2, 4],
    "downscale": [1, 2, 4, 8],
    "quantize_levels": [256, 32, 16, 8],
}
AUDIO_LEVELS = {"snr_db": [100, 20, 10, 0]}


@torch.no_grad()
def sweep(cfg, checkpoint: str, manifest: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    model.eval()

    ds = AVDeepfakeDataset(manifest, cfg.shard_root, cfg.n_frames, cfg.audio_len,
                           root_dir=getattr(cfg, "root_dir", None), train=False)
    max_clips = int(getattr(cfg, "eval_max_clips", 0) or 0)
    if max_clips and len(ds) > max_clips:
        from src.training.train import _stratified_subsample
        ds.records = _stratified_subsample(ds.records, max_clips)
        print(f"[robustness] subsampled to {len(ds)} clips")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, collate_fn=collate)

    def run_once(v_fn=None, a_fn=None):
        pv, pa, yv, ya = [], [], [], []
        for batch in dl:
            batch = move(batch, device)
            video = v_fn(batch["video"]) if v_fn else batch["video"]
            audio = a_fn(batch["audio"]) if a_fn else batch["audio"]
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(video, audio)
            pv += torch.sigmoid(out["logit_v"].float()).cpu().tolist()
            pa += torch.sigmoid(out["logit_a"].float()).cpu().tolist()
            yv += batch["video_label"].cpu().tolist()
            ya += batch["audio_label"].cpu().tolist()
        return per_modality(yv, pv)["auc"], per_modality(ya, pa)["auc"]

    results = {"video": {}, "audio": {}}
    for sigma in VIDEO_LEVELS["blur_sigma"]:
        v_auc, _ = run_once(v_fn=lambda x, s=sigma: video_blur(x, s))
        results["video"].setdefault("blur_sigma", []).append({"level": sigma, "auc": v_auc})
    for f in VIDEO_LEVELS["downscale"]:
        v_auc, _ = run_once(v_fn=lambda x, k=f: video_downscale(x, k))
        results["video"].setdefault("downscale", []).append({"level": f, "auc": v_auc})
    for q in VIDEO_LEVELS["quantize_levels"]:
        v_auc, _ = run_once(v_fn=lambda x, k=q: video_quantize(x, k))
        results["video"].setdefault("quantize_levels", []).append({"level": q, "auc": v_auc})
    for snr in AUDIO_LEVELS["snr_db"]:
        _, a_auc = run_once(a_fn=lambda x, s=snr: audio_noise(x, s))
        results["audio"].setdefault("snr_db", []).append({"level": snr, "auc": a_auc})
    return {"method": "david-net", "checkpoint": checkpoint,
            "manifest": manifest, "sweeps": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="results/robustness.json")
    ap.add_argument("--root-dir", default=None)
    ap.add_argument("--max-clips", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.root_dir:
        cfg.root_dir = args.root_dir
    if args.max_clips:
        cfg.eval_max_clips = args.max_clips
    report = sweep(cfg, args.checkpoint, args.manifest)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["sweeps"], indent=2))


if __name__ == "__main__":
    main()
