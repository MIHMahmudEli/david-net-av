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
import math
import os
import time
import traceback

import torch
from torch.utils.data import DataLoader

from src.data.datasets import AVDeepfakeDataset, preflight_check
from src.data.synthetic_quadrants import QACPDataset, QACPBalancedSampler, collate_qacp
from src.training.losses import qacp_loss
from src.training.train import build_model, move, enable_gradient_checkpointing, _lr_lambda
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
                             root_dir=root_dir, train=True)
    print(f"[qacp] {len(base)} pristine (RVRA) clips in {cfg.train_manifest}")
    if len(base) < 2:
        raise RuntimeError("[qacp] need at least 2 RVRA clips (MISMATCH needs a donor)")
    preflight_check(base, name="qacp-real")
    views = int(getattr(cfg, "qacp_views_per_clip", 4))
    ds = QACPDataset(base, views_per_clip=views, seed=cfg.seed)
    sampler = QACPBalancedSampler(ds, cfg.batch_size, seed=cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler,
                    num_workers=cfg.num_workers, collate_fn=collate_qacp,
                    drop_last=True, pin_memory=(device == "cuda"))
    print(f"[qacp] {len(ds)} pseudo-samples/epoch ({views} views per clip), "
          f"{len(dl)} micro-batches of {cfg.batch_size}")

    model = build_model(cfg).to(device)
    if getattr(cfg, "gradient_checkpointing", True):
        enable_gradient_checkpointing(model)

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[qacp] Parameters: {trainable_p:,} trainable / {total_p:,} total "
          f"({100.0 * trainable_p / max(total_p, 1):.1f}%)")
    if trainable_p == 0:
        raise RuntimeError(
            "[qacp] No trainable parameters! Check freeze_blocks / freeze_feature_extractor. "
            "Set freeze_blocks < total_encoder_blocks (VideoMAE-base has 12 blocks; "
            "recommend freeze_blocks=8 to leave last 4 unfrozen for QACP).")

    enc_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if ("video_encoder" in name or "audio_encoder" in name) else other_params).append(p)
    lr_enc = getattr(cfg, "lr_encoder", cfg.lr)
    opt = torch.optim.AdamW([
        {"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": enc_params, "lr": lr_enc, "weight_decay": cfg.weight_decay},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    grad_accum = 1 if cfg.dry_run else max(1, int(getattr(cfg, "grad_accum_steps", 1)))
    temperature = getattr(cfg, "qacp_temperature", 0.1)
    milestone_every = getattr(cfg, "milestone_every", 5)
    keep_milestones = getattr(cfg, "keep_milestones", 3)
    max_norm = float(getattr(cfg, "max_grad_norm", 1.0))
    max_micro = int(getattr(cfg, "max_steps_per_epoch", 0) or 0)   # 0 = full epoch
    n_micro = min(len(dl), max_micro) if max_micro else len(dl)
    steps_per_epoch = max(1, n_micro // grad_accum)
    warmup_epochs = getattr(cfg, "warmup_epochs", 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(int(warmup_epochs * steps_per_epoch), cfg.epochs * steps_per_epoch))

    # Early stopping: halt if avg_loss fails to improve for `patience` epochs
    patience = getattr(cfg, "patience", 8)
    min_delta = getattr(cfg, "min_delta", 1e-4)

    # ─── HF Backup setup ──────────────────────────────────────────────
    run_id = getattr(cfg, "run_id", None)
    backup = None
    start_epoch = 0
    _resume = None

    if run_id:
        from src.utils.hf_backup import HFBackup
        backup = HFBackup(run_id=run_id, local_dir=getattr(cfg, "local_dir", "/kaggle/working"))
        backup.setup()

        _resume = backup.load_resume_state()
        if _resume is not None:
            start_epoch = _resume.get("epoch", -1) + 1
            try:
                model.load_state_dict(_resume["model"])
                opt.load_state_dict(_resume["optimizer"])
                print(f"QACP resumed from HF: epoch {start_epoch}")
            except Exception as e:  # noqa: BLE001
                print(f"Resume load warning: {e} — starting from scratch")
                start_epoch = 0
                _resume = None
        else:
            print("No QACP resume state — starting fresh")
    for _ in range(start_epoch * steps_per_epoch):
        scheduler.step()

    effective_batch = cfg.batch_size * grad_accum
    print(f"[qacp] Effective batch size: {cfg.batch_size} x {grad_accum} = {effective_batch}; "
          f"{steps_per_epoch} optimizer steps/epoch; temperature={temperature}")

    # ─── Training loop ────────────────────────────────────────────────
    micro_steps = 0
    opt_steps = start_epoch * steps_per_epoch
    model.train()
    best_loss = float("inf") if _resume is None else _resume.get("best_loss", float("inf"))
    no_improve = 0 if _resume is None else _resume.get("no_improve", 0)

    if no_improve >= patience:
        print(f"[qacp] Already converged (no_improve={no_improve} >= patience={patience}) — skipping training")
        if backup:
            backup.push_log({
                "epoch": start_epoch, "avg_loss": best_loss, "best_loss": best_loss,
                "steps": 0, "is_best": False,
                "skipped": True, "reason": f"already converged no_improve={no_improve}",
                "phase": "qacp", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        return model

    try:
        for epoch in range(start_epoch, cfg.epochs):
            ds.set_epoch(epoch)
            sampler.set_epoch(epoch)
            epoch_loss, epoch_steps, skipped = 0.0, 0, 0
            parts_sum = {}
            t_epoch = time.time()
            opt.zero_grad(set_to_none=True)
            accum = 0

            for it, batch in enumerate(dl):
                if max_micro and it >= max_micro:
                    break
                batch = move(batch, device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    out = model(batch["video"], batch["audio"])
                loss, parts = qacp_loss(out, batch, temperature=temperature)
                micro_steps += 1
                if not torch.isfinite(loss):
                    skipped += 1
                    print(f"[qacp] epoch {epoch} micro-step {micro_steps} non-finite loss — skipping")
                    opt.zero_grad(set_to_none=True)
                    accum = 0
                    if skipped > 20:
                        raise RuntimeError("[qacp] too many non-finite steps — aborting")
                    continue
                scaler.scale(loss / grad_accum).backward()
                accum += 1
                epoch_loss += loss.item()
                epoch_steps += 1
                for k, v in parts.items():
                    parts_sum[k] = parts_sum.get(k, 0.0) + v

                if accum >= grad_accum:
                    scaler.unscale_(opt)
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=max_norm)
                    scaler.step(opt)
                    scaler.update()
                    if torch.isfinite(gnorm):
                        scheduler.step()
                        opt_steps += 1
                    opt.zero_grad(set_to_none=True)
                    accum = 0
                    if opt_steps % cfg.log_every == 0:
                        print(f"[qacp] epoch {epoch} step {opt_steps} " +
                              " ".join(f"{k}={v:.4f}" for k, v in parts.items()) +
                              f" gnorm={float(gnorm):.2f} lr={opt.param_groups[0]['lr']:.2e}")

                if cfg.dry_run and micro_steps >= 2 * grad_accum:
                    print("[dry-run] QACP forward/backward OK, stopping.")
                    if backup:
                        backup.emergency_push(model, epoch)
                    return model

            # ─── End of epoch ──────────────────────────────────────────
            if epoch_steps == 0:
                raise RuntimeError(f"[qacp] epoch {epoch}: no finite step")
            avg_loss = epoch_loss / epoch_steps
            avg_parts = {k: v / epoch_steps for k, v in parts_sum.items()}
            is_best = avg_loss < (best_loss - min_delta)
            if is_best:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1

            print(f"[qacp] epoch {epoch} ({(time.time() - t_epoch) / 60:.1f} min) avg_loss={avg_loss:.4f} "
                  + " ".join(f"{k}={v:.4f}" for k, v in avg_parts.items() if k != "total")
                  + f" best={best_loss:.4f}{' (NEW BEST)' if is_best else ''} no_improve={no_improve}/{patience}")
            # chance-level reference: SupCon on a balanced batch of B with collapsed
            # embeddings sits at ln(B-1); print it once so plateaus are recognisable
            if epoch == start_epoch:
                print(f"[qacp] reference: collapsed/chance loss per term ~ ln({cfg.batch_size}-1) = "
                      f"{math.log(max(2, cfg.batch_size - 1)):.4f}")

            if not backup:
                os.makedirs(cfg.out_dir, exist_ok=True)
                path = f"{cfg.out_dir}/qacp_epoch{epoch}.pt"
                torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, path)
                print(f"saved {path}")

            if no_improve >= patience:
                print(f"[qacp] Early stopping: no improvement for {patience} epochs")
                if backup:
                    backup.push_checkpoint(
                        model, opt, epoch, vars(cfg), {"avg_loss": avg_loss, "early_stop": True},
                        milestone_every=milestone_every, keep_milestones=keep_milestones,
                        resume_extras={"best_loss": best_loss, "no_improve": no_improve})
                    backup.push_log({
                        "epoch": epoch, "avg_loss": avg_loss, "best_loss": best_loss,
                        "steps": epoch_steps, "is_best": is_best, **avg_parts,
                        "early_stop": True, "reason": f"no_improve={patience}",
                        "phase": "qacp", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                    backup.push_final({"epochs": epoch + 1, "total_steps": opt_steps, "phase": "qacp",
                                       "best_loss": best_loss, "early_stop": True})
                break

            if backup:
                backup.push_checkpoint(
                    model, opt, epoch, vars(cfg), {"avg_loss": avg_loss},
                    milestone_every=milestone_every, keep_milestones=keep_milestones,
                    resume_extras={"best_loss": best_loss, "no_improve": no_improve})
                if is_best:
                    backup.push_best(model, epoch, avg_loss)
                backup.push_log({
                    "epoch": epoch, "avg_loss": avg_loss, "best_loss": best_loss,
                    "steps": epoch_steps, "is_best": is_best, **avg_parts,
                    "phase": "qacp", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

        if backup:
            backup.push_final({"epochs": cfg.epochs, "total_steps": opt_steps, "phase": "qacp",
                               "best_loss": best_loss})
            print("QACP training complete. Artifacts pushed to HF.")

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
