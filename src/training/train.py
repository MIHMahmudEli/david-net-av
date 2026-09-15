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
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import AVDeepfakeDataset, BalancedGeneratorSampler, collate
from src.data.augment import VideoAugmentor, AudioAugmentor, augment_batch
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
    if p <= 0:
        return None, None
    drop = torch.rand(batch_size, device=device) < p
    drop_video = torch.rand(batch_size, device=device) < 0.5
    v_avail = torch.where(drop & drop_video, 0.0, 1.0)
    a_avail = torch.where(drop & ~drop_video, 0.0, 1.0)
    return v_avail, a_avail


@torch.no_grad()
def validate(model, val_dl, device, weights):
    """Run validation and return metrics dict."""
    model.eval()
    all_v_pred, all_a_pred, all_v_true, all_a_true, all_quad_pred, all_quad_true = [], [], [], [], [], []
    total_loss_val = 0.0
    n_batches = 0

    for batch in val_dl:
        batch = move(batch, device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(batch["video"], batch["audio"])
            loss, _ = total_loss(out, batch, weights, model=model)
        total_loss_val += loss.item()
        n_batches += 1

        all_v_pred += torch.sigmoid(out["logit_v"]).cpu().tolist()
        all_a_pred += torch.sigmoid(out["logit_a"]).cpu().tolist()
        all_quad_pred += out["logit_quad"].argmax(-1).cpu().tolist()
        all_v_true += batch["video_label"].cpu().tolist()
        all_a_true += batch["audio_label"].cpu().tolist()
        all_quad_true += batch["quadrant"].cpu().tolist()

    # Compute AUC
    from src.eval.metrics import per_modality, quadrant_metrics
    v_metrics = per_modality(all_v_true, all_v_pred)
    a_metrics = per_modality(all_a_true, all_a_pred)
    q_metrics = quadrant_metrics(all_quad_true, all_quad_pred)

    model.train()
    return {
        "val_loss": total_loss_val / max(n_batches, 1),
        "video_auc": v_metrics.get("auc", 0.0),
        "audio_auc": a_metrics.get("auc", 0.0),
        "quadrant_acc": q_metrics.get("accuracy", 0.0),
    }


def train(cfg):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─── Data ─────────────────────────────────────────────────────────
    if getattr(cfg, "feature_cache", None):
        from src.data.datasets import CachedFeatureDataset
        train_ds = CachedFeatureDataset(cfg.train_manifest, cfg.feature_cache,
                                        cfg.n_frames, cfg.audio_len)
    else:
        train_ds = AVDeepfakeDataset(cfg.train_manifest, cfg.shard_root,
                                     cfg.n_frames, cfg.audio_len)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          sampler=BalancedGeneratorSampler(train_ds.records, cfg.batch_size),
                          num_workers=cfg.num_workers, collate_fn=collate)

    # Validation set (optional)
    val_manifest = getattr(cfg, "val_manifest", None)
    val_dl = None
    if val_manifest and os.path.exists(val_manifest):
        val_ds = AVDeepfakeDataset(val_manifest, cfg.shard_root, cfg.n_frames, cfg.audio_len)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate)
        print(f"Validation: {len(val_ds)} clips")

    # ─── Model ────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    if getattr(cfg, "init_from", None):
        state = torch.load(cfg.init_from, map_location=device)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        print(f"init_from {cfg.init_from}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    weights = LossWeights(**cfg.loss_weights)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ─── HF Backup ────────────────────────────────────────────────────
    run_id = getattr(cfg, "run_id", None)
    backup = None
    start_epoch = 0
    best_auc = 0.0

    if run_id:
        from src.utils.hf_backup import HFBackup
        backup = HFBackup(run_id=run_id, local_dir=getattr(cfg, "local_dir", "/kaggle/working"))
        backup.setup()

        resume = backup.load_resume_state()
        if resume is not None:
            start_epoch = resume.get("epoch", -1) + 1
            best_auc = resume.get("best_auc", 0.0)
            try:
                model.load_state_dict(resume["model"])
                opt.load_state_dict(resume["optimizer"])
                print(f"Resumed from HF: epoch {start_epoch}, best_auc={best_auc:.4f}")
            except Exception as e:
                print(f"Resume load warning: {e} — starting from scratch")
                start_epoch = 0
        else:
            print("No resume state found — starting fresh")

    # ─── Augmentation ──────────────────────────────────────────────────
    use_aug = getattr(cfg, "augment", True)
    v_aug = VideoAugmentor(p=0.5) if use_aug else None
    a_aug = AudioAugmentor(p=0.5) if use_aug else None

    # ─── Training loop ────────────────────────────────────────────────
    steps = 0
    model.train()

    try:
        for epoch in range(start_epoch, cfg.epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for batch in train_dl:
                batch = move(batch, device)
                batch = augment_batch(batch, v_aug, a_aug)
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

            # Validation
            metrics = {"avg_loss": avg_loss}
            if val_dl:
                val_metrics = validate(model, val_dl, device, weights)
                metrics.update(val_metrics)
                val_auc = (val_metrics["video_auc"] + val_metrics["audio_auc"]) / 2
                print(f"epoch {epoch} — loss={avg_loss:.4f} val_video_auc={val_metrics['video_auc']:.4f} "
                      f"val_audio_auc={val_metrics['audio_auc']:.4f} val_quad_acc={val_metrics['quadrant_acc']:.4f}")

                # Push best model
                if val_auc > best_auc:
                    best_auc = val_auc
                    if backup:
                        backup.push_best(model, epoch, val_auc)
                    print(f"  ** new best AUC: {val_auc:.4f}")
            else:
                print(f"epoch {epoch} — loss={avg_loss:.4f}")

            # Push to HF
            if backup:
                metrics["best_auc"] = best_auc
                backup.push_checkpoint(model, opt, epoch, vars(cfg), metrics)
                backup.push_log({
                    "epoch": epoch, "avg_loss": avg_loss, "steps": epoch_steps,
                    **{k: v for k, v in metrics.items() if isinstance(v, float)},
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

        # ─── Run complete ─────────────────────────────────────────────
        final_metrics = {"epochs": cfg.epochs, "total_steps": steps, "best_auc": best_auc}
        if backup:
            backup.push_final(final_metrics)
            print(f"Training complete. Best AUC: {best_auc:.4f}")

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

    return model


def _save(model, cfg, epoch):
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = f"{cfg.out_dir}/david_net_epoch{epoch}.pt"
    torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, path)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True
    if args.run_id:
        cfg.run_id = args.run_id
    train(cfg)


if __name__ == "__main__":
    main()
