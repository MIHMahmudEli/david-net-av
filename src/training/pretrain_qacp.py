"""QACP Stage-0 pretraining: factorized SupCon on synthetic quadrants.

Supports crash-proof training via HuggingFace backup (hf_backup.py).

Usage:
    python -m src.training.pretrain_qacp --config configs/qacp.yaml
    python -m src.training.pretrain_qacp --config configs/qacp.yaml --dry-run
    python -m src.training.pretrain_qacp --config configs/qacp.yaml --run-id qacp_001

Output checkpoint feeds Stage-1 supervised training:
    python -m src.training.train --config configs/david_net.yaml \
        (set init_from: runs/qacp/qacp_epochN.pt in the config)

See docs/02_architecture.md §8b.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import traceback

import torch
from torch.utils.data import DataLoader

from src.data.datasets import AVDeepfakeDataset
from src.data.synthetic_quadrants import QACPDataset, collate_qacp
from src.training.losses import qacp_loss
from src.training.train import build_model, move
from src.utils.config import load_config
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def pretrain(cfg):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # base manifest should contain REAL (RVRA) clips only
    root_dir = getattr(cfg, "root_dir", None)
    base = AVDeepfakeDataset(cfg.train_manifest, cfg.shard_root, cfg.n_frames, cfg.audio_len,
                             filt=lambda r: r["video_label"] == 0 and r["audio_label"] == 0,
                             root_dir=root_dir)
    ds = QACPDataset(base)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, collate_fn=collate_qacp)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Gradient accumulation: simulate larger effective batch on limited VRAM
    grad_accum = getattr(cfg, "grad_accum_steps", 1)
    temperature = getattr(cfg, "qacp_temperature", 0.1)

    # ─── HF Backup setup ──────────────────────────────────────────────
    run_id = getattr(cfg, "run_id", None)
    backup = None
    start_epoch = 0

    if run_id:
        from src.utils.hf_backup import HFBackup
        backup = HFBackup(run_id=run_id, local_dir=getattr(cfg, "local_dir", "/kaggle/working"))
        backup.setup()

        resume = backup.load_resume_state()
        if resume is not None:
            start_epoch = resume.get("epoch", -1) + 1
            try:
                model.load_state_dict(resume["model"])
                opt.load_state_dict(resume["optimizer"])
                print(f"QACP resumed from HF: epoch {start_epoch}")
            except Exception as e:
                print(f"Resume load warning: {e} — starting from scratch")
                start_epoch = 0
        else:
            print("No QACP resume state — starting fresh")

    effective_batch = cfg.batch_size * grad_accum
    print(f"[qacp] Effective batch size: {cfg.batch_size} × {grad_accum} = {effective_batch}")

    # ─── Training loop ────────────────────────────────────────────────
    steps = 0
    model.train()

    try:
        for epoch in range(start_epoch, cfg.epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            opt.zero_grad(set_to_none=True)

            for batch in dl:
                batch = move(batch, device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    out = model(batch["video"], batch["audio"])
                    loss, parts = qacp_loss(out, batch, temperature=temperature)
                scaled_loss = loss / grad_accum
                scaler.scale(scaled_loss).backward()
                steps += 1
                epoch_loss += loss.item()
                epoch_steps += 1

                if steps % grad_accum == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                if steps % cfg.log_every == 0:
                    print(f"[qacp] epoch {epoch} step {steps} " +
                          " ".join(f"{k}={v:.4f}" for k, v in parts.items()))

                if cfg.dry_run and steps >= 2:
                    print("[dry-run] QACP forward/backward OK, stopping.")
                    if backup:
                        backup.emergency_push(model, epoch)
                    return model

            # ─── End of epoch ──────────────────────────────────────────
            # Flush remaining accumulated gradients
            if steps % grad_accum != 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            avg_loss = epoch_loss / max(epoch_steps, 1)

            # Push to HF (saves locally, uploads, then cleans up local to save disk)
            if backup:
                backup.push_checkpoint(model, opt, epoch, vars(cfg), {"avg_loss": avg_loss})
                backup.push_log({
                    "epoch": epoch, "avg_loss": avg_loss, "steps": epoch_steps,
                    "phase": "qacp", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            else:
                os.makedirs(cfg.out_dir, exist_ok=True)
                path = f"{cfg.out_dir}/qacp_epoch{epoch}.pt"
                torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, path)
                print(f"saved {path}")

        # ─── Run complete ─────────────────────────────────────────────
        if backup:
            backup.push_final({"epochs": cfg.epochs, "total_steps": steps, "phase": "qacp"})
            print(f"QACP training complete. Artifacts pushed to HF.")

    except KeyboardInterrupt:
        print("\nInterrupted — pushing emergency checkpoint...")
        if backup:
            backup.emergency_push(model, epoch)
        raise
    except Exception:
        logger.error(f"QACP crashed: {traceback.format_exc()}")
        if backup:
            backup.emergency_push(model, epoch if 'epoch' in dir() else 0)
        raise

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-id", default=None, help="Unique run ID for HF backup")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True
    if args.run_id:
        cfg.run_id = args.run_id
    pretrain(cfg)


if __name__ == "__main__":
    main()
