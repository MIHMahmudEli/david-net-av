"""DAVID-Net training entry point. Config-driven; runnable end-to-end on dummy data.

Usage:
    python -m src.training.train --config configs/david_net.yaml
    python -m src.training.train --config configs/david_net.yaml --dry-run
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from src.data.datasets import AVDeepfakeDataset, collate
from src.models.david_net import DavidNet, DavidNetConfig
from src.models.video_encoder import build_video_encoder
from src.models.audio_encoder import build_audio_encoder
from src.training.losses import LossWeights, total_loss
from src.utils.config import load_config
from src.utils.seed import set_seed


def build_model(cfg) -> DavidNet:
    mcfg = DavidNetConfig(
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_fusion_layers=cfg.n_fusion_layers,
        dropout=cfg.dropout, use_sync=cfg.use_sync, use_disentangle=cfg.use_disentangle,
        compose_quadrant=cfg.compose_quadrant,
    )
    if getattr(cfg, "feature_cache", None):
        # Phase-A cached-feature regime: inputs are already (L, d) token sequences,
        # so the encoders collapse to identity (docs/07_compute_and_hardware.md §2).
        import torch.nn as nn
        venc, aenc = nn.Identity(), nn.Identity()
    else:
        venc = build_video_encoder(cfg)
        aenc = build_audio_encoder(cfg)
    return DavidNet(mcfg, venc, aenc)


def move(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def train(cfg):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if getattr(cfg, "feature_cache", None):
        from src.data.datasets import CachedFeatureDataset
        train_ds = CachedFeatureDataset(cfg.train_manifest, cfg.feature_cache,
                                        cfg.n_frames, cfg.audio_len)
    else:
        train_ds = AVDeepfakeDataset(cfg.train_manifest, cfg.shard_root,
                                     cfg.n_frames, cfg.audio_len)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate)

    model = build_model(cfg).to(device)
    if getattr(cfg, "init_from", None):  # Stage 1: warm-start from a QACP checkpoint
        state = torch.load(cfg.init_from, map_location=device)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        print(f"init_from {cfg.init_from}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    weights = LossWeights(**cfg.loss_weights)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    steps = 0
    model.train()
    for epoch in range(cfg.epochs):
        for batch in train_dl:
            batch = move(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(batch["video"], batch["audio"])
                loss, parts = total_loss(out, batch, weights, model=model)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            steps += 1
            if steps % cfg.log_every == 0:
                print(f"epoch {epoch} step {steps} " +
                      " ".join(f"{k}={v:.4f}" for k, v in parts.items()))
            if cfg.dry_run and steps >= 2:
                print("[dry-run] forward/backward OK, stopping.")
                return model
        _save(model, cfg, epoch)
    return model


def _save(model, cfg, epoch):
    import os
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = f"{cfg.out_dir}/david_net_epoch{epoch}.pt"
    torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, path)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True
    train(cfg)


if __name__ == "__main__":
    main()
