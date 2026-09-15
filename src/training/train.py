"""DAVID-Net training entry point. Config-driven; runnable end-to-end on dummy data.

Supports crash-proof training via HuggingFace backup (hf_backup.py).
Checkpoints are pushed to HF after every epoch; resume is automatic.

Usage:
    python -m src.training.train --config configs/david_net.yaml
    python -m src.training.train --config configs/david_net.yaml --dry-run
    python -m src.training.train --config configs/david_net.yaml --run-id run_001
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
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

logger = logging.getLogger(__name__)


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


def sample_modality_masks(batch_size: int, p: float, device):
    """Modality dropout: with prob p a sample loses ONE stream (never both).

    Trains the network to handle audio-only inputs and silent (video-only)
    clips — see docs/02_architecture.md §7b.
    """
    if p <= 0:
        return None, None
    drop = torch.rand(batch_size, device=device) < p
    drop_video = torch.rand(batch_size, device=device) < 0.5
    v_avail = torch.where(drop & drop_video, 0.0, 1.0)
    a_avail = torch.where(drop & ~drop_video, 0.0, 1.0)
    return v_avail, a_avail


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

    # ─── HF Backup setup ──────────────────────────────────────────────
    run_id = getattr(cfg, "run_id", None)
    backup = None
    start_epoch = 0

    if run_id:
        from src.utils.hf_backup import HFBackup, crash_guard
        backup = HFBackup(run_id=run_id, local_dir=getattr(cfg, "local_dir", "/kaggle/working"))
        backup.setup()

        # Resume check
        resume = backup.load_resume_state()
        if resume is not None:
            start_epoch = resume.get("epoch", -1) + 1
            try:
                model.load_state_dict(resume["model"])
                opt.load_state_dict(resume["optimizer"])
                print(f"Resumed from HF: epoch {start_epoch}")
            except Exception as e:
                print(f"Resume load warning: {e} — starting from scratch")
                start_epoch = 0
        else:
            print("No resume state found — starting fresh")

    # ─── Training loop ────────────────────────────────────────────────
    steps = 0
    model.train()
    crash_guard_ctx = (lambda: __import__("src.utils.hf_backup", fromlist=["crash_guard"]).crash_guard(backup, model, lambda: epoch)) if backup else None

    try:
        if crash_guard_ctx:
            crash_guard_ctx().__enter__()

        for epoch in range(start_epoch, cfg.epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            for batch in train_dl:
                batch = move(batch, device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    v_av, a_av = sample_modality_masks(
                        batch["video"].size(0), cfg.modality_dropout, batch["video"].device)
                    out = model(batch["video"], batch["audio"], v_avail=v_av, a_avail=a_av)
                    loss, parts = total_loss(out, batch, weights, model=model)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                steps += 1
                epoch_loss += loss.item()
                epoch_steps += 1
                if steps % cfg.log_every == 0:
                    print(f"epoch {epoch} step {steps} " +
                          " ".join(f"{k}={v:.4f}" for k, v in parts.items()))

                if cfg.dry_run and steps >= 2:
                    print("[dry-run] forward/backward OK, stopping.")
                    if backup:
                        backup.emergency_push(model, epoch)
                    return model

            # ─── End of epoch ──────────────────────────────────────────
            avg_loss = epoch_loss / max(epoch_steps, 1)
            _save(model, cfg, epoch)

            # Push to HF
            if backup:
                backup.push_checkpoint(model, opt, epoch, vars(cfg), {"avg_loss": avg_loss})
                backup.push_log({
                    "epoch": epoch, "avg_loss": avg_loss, "steps": epoch_steps,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

            print(f"epoch {epoch} done — avg_loss={avg_loss:.4f}")

        # ─── Run complete ─────────────────────────────────────────────
        if backup:
            backup.push_final({"epochs": cfg.epochs, "total_steps": steps})
            print(f"Training complete. All artifacts pushed to HF.")

    except KeyboardInterrupt:
        print("\nInterrupted — pushing emergency checkpoint...")
        if backup:
            backup.emergency_push(model, epoch)
        raise
    except Exception:
        logger.error(f"Training crashed: {traceback.format_exc()}")
        if backup:
            backup.emergency_push(model, epoch if 'epoch' in dir() else 0)
        raise
    finally:
        if crash_guard_ctx:
            crash_guard_ctx().__exit__(None, None, None)

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
    ap.add_argument("--run-id", default=None, help="Unique run ID for HF backup (e.g. run_001)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True
    if args.run_id:
        cfg.run_id = args.run_id
    train(cfg)


if __name__ == "__main__":
    main()
