"""QACP Stage-0 pretraining: factorized SupCon on synthetic quadrants.

Usage:
    python -m src.training.pretrain_qacp --config configs/qacp.yaml
    python -m src.training.pretrain_qacp --config configs/qacp.yaml --dry-run

Output checkpoint feeds Stage-1 supervised training:
    python -m src.training.train --config configs/david_net.yaml \
        (set init_from: runs/qacp/qacp_epochN.pt in the config)

See docs/02_architecture.md §8b.
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.data.datasets import AVDeepfakeDataset
from src.data.synthetic_quadrants import QACPDataset, collate_qacp
from src.training.losses import qacp_loss
from src.training.train import build_model, move
from src.utils.config import load_config
from src.utils.seed import set_seed


def pretrain(cfg):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # base manifest should contain REAL (RVRA) clips only
    base = AVDeepfakeDataset(cfg.train_manifest, cfg.shard_root, cfg.n_frames, cfg.audio_len,
                             filt=lambda r: r["video_label"] == 0 and r["audio_label"] == 0)
    ds = QACPDataset(base)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, collate_fn=collate_qacp)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    steps = 0
    model.train()
    for epoch in range(cfg.epochs):
        for batch in dl:
            batch = move(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(batch["video"], batch["audio"])
                loss, parts = qacp_loss(out, batch, temperature=cfg.qacp_temperature)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            steps += 1
            if steps % cfg.log_every == 0:
                print(f"[qacp] epoch {epoch} step {steps} " +
                      " ".join(f"{k}={v:.4f}" for k, v in parts.items()))
            if cfg.dry_run and steps >= 2:
                print("[dry-run] QACP forward/backward OK, stopping.")
                return model
        os.makedirs(cfg.out_dir, exist_ok=True)
        path = f"{cfg.out_dir}/qacp_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, path)
        print(f"saved {path}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True
    pretrain(cfg)


if __name__ == "__main__":
    main()
