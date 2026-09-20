"""DAVID-Net training entry point. Config-driven; runnable end-to-end on dummy data.

Supports crash-proof training via HuggingFace backup (hf_backup.py).
Checkpoints are pushed to HF after every epoch; resume is automatic.

Usage:
    python -m src.training.train --config configs/david_net.yaml
    python -m src.training.train --config configs/david_net.yaml --dry-run
    python -m src.training.train --config configs/david_net.yaml --run-id run_001

Safety rails added after the first Kaggle run (which trained 50 epochs on random
tensors and then NaN-looped for 29 epochs without stopping):
  * data preflight — media must exist and decode before a single GPU step
  * gradient accumulation actually implemented (`grad_accum_steps`)
  * non-finite loss/grad steps are skipped; after `nan_patience` consecutive skips the
    last good snapshot is restored and the LR halved; after `max_nan_restores` the run
    raises instead of silently finishing with AUC=0.5
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import time
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import AVDeepfakeDataset, BalancedBatchSampler, collate, preflight_check
from src.data.augment import VideoAugmentor, AudioAugmentor, augment_batch
from src.models.david_net import DavidNet, DavidNetConfig
from src.models.video_encoder import build_video_encoder
from src.models.audio_encoder import build_audio_encoder
from src.training.losses import LossWeights, total_loss, distillation_loss
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
            batch[k] = v.to(device, non_blocking=True)
    return batch


def sample_modality_masks(batch_size: int, p: float, device):
    if p <= 0:
        return None, None
    drop = torch.rand(batch_size, device=device) < p
    drop_video = torch.rand(batch_size, device=device) < 0.5
    v_avail = torch.where(drop & drop_video, 0.0, 1.0)
    a_avail = torch.where(drop & ~drop_video, 0.0, 1.0)
    return v_avail, a_avail


def availability_masks(batch, p_dropout: float = 0.0):
    """Combine genuine stream availability (audio-only files, silent clips) with
    modality dropout. A sample never loses both streams."""
    v_av = batch.get("v_avail")
    a_av = batch.get("a_avail")
    B = batch["video"].size(0)
    dev = batch["video"].device
    if v_av is None:
        v_av = torch.ones(B, device=dev)
    if a_av is None:
        a_av = torch.ones(B, device=dev)
    v_av, a_av = v_av.float(), a_av.float()
    if p_dropout > 0:
        dv, da = sample_modality_masks(B, p_dropout, dev)
        # only drop a stream if the other one is genuinely present
        v_av = torch.where((a_av > 0) & (dv == 0), torch.zeros_like(v_av), v_av)
        a_av = torch.where((v_av > 0) & (da == 0), torch.zeros_like(a_av), a_av)
    return v_av, a_av


def enable_gradient_checkpointing(model):
    for enc_name in ("video_encoder", "audio_encoder"):
        enc = getattr(model, enc_name, None)
        bb = getattr(enc, "backbone", None)
        if bb is not None and hasattr(bb, "gradient_checkpointing_enable"):
            try:
                bb.gradient_checkpointing_enable()
                print(f"Gradient checkpointing enabled on {enc_name}")
            except Exception as e:  # noqa: BLE001
                print(f"Gradient checkpointing unavailable on {enc_name}: {e}")


@torch.no_grad()
def validate(model, val_dl, device, weights):
    """Run validation and return metrics dict."""
    model.eval()
    all_v_pred, all_a_pred, all_v_true, all_a_true, all_quad_pred, all_quad_true = [], [], [], [], [], []
    total_loss_val = 0.0
    n_batches = 0

    for batch in val_dl:
        batch = move(batch, device)
        v_av, a_av = availability_masks(batch, 0.0)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(batch["video"], batch["audio"], v_avail=v_av, a_avail=a_av)
        loss, _ = total_loss(out, batch, weights, model=model)
        if torch.isfinite(loss):
            total_loss_val += loss.item()
            n_batches += 1

        all_v_pred += torch.sigmoid(out["logit_v"].float()).cpu().tolist()
        all_a_pred += torch.sigmoid(out["logit_a"].float()).cpu().tolist()
        all_quad_pred += out["logit_quad"].argmax(-1).cpu().tolist()
        all_v_true += batch["video_label"].cpu().tolist()
        all_a_true += batch["audio_label"].cpu().tolist()
        all_quad_true += batch["quadrant"].cpu().tolist()

    from src.eval.metrics import per_modality, quadrant_metrics
    v_metrics = per_modality(all_v_true, all_v_pred)
    a_metrics = per_modality(all_a_true, all_a_pred)
    q_metrics = quadrant_metrics(all_quad_true, all_quad_pred)

    model.train()
    return {
        "val_loss": total_loss_val / max(n_batches, 1),
        "video_auc": _nan_to_zero(v_metrics.get("auc", 0.0)),
        "audio_auc": _nan_to_zero(a_metrics.get("auc", 0.0)),
        "video_eer": _nan_to_zero(v_metrics.get("eer", 0.0)),
        "audio_eer": _nan_to_zero(a_metrics.get("eer", 0.0)),
        "quadrant_acc": q_metrics.get("acc", 0.0),
        "quadrant_macro_f1": q_metrics.get("macro_f1", 0.0),
    }


def _stratified_subsample(records, n_keep: int, seed: int = 0):
    """Deterministic per-quadrant subsample (keeps class balance for quick validation)."""
    import random as _r
    by_q = {}
    for i, r in enumerate(records):
        by_q.setdefault(r.get("quadrant", "?"), []).append(i)
    rng = _r.Random(seed)
    keep = []
    for idx in by_q.values():
        rng.shuffle(idx)
        keep += idx[: max(1, round(n_keep * len(idx) / len(records)))]
    return [records[i] for i in sorted(keep)]


def _nan_to_zero(x: float) -> float:
    return 0.0 if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)


def _lr_lambda(warmup_steps: int, total_steps: int, floor: float = 0.01):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return f


def _host_mem(tag: str):
    """Print host RAM use (Kaggle kills the whole container on RAM OOM, silently)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        rss = psutil.Process(os.getpid()).memory_info().rss
        print(f"[mem:{tag}] process RSS {rss / 1e9:.2f} GB | host used {vm.used / 1e9:.2f} / {vm.total / 1e9:.1f} GB "
              f"(avail {vm.available / 1e9:.2f} GB)")
    except Exception:  # noqa: BLE001
        pass


def _to_cpu(obj):
    if torch.is_tensor(obj):
        t = obj.detach()
        if t.is_floating_point() and t.dtype == torch.float32:
            t = t.half()          # rollback copy; fp16 is plenty for a safety net
        return t.to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return copy.deepcopy(obj)


def _snapshot(model, opt):
    """CPU copy of model + optimizer state (keeps VRAM free on a 15 GB T4)."""
    return {"model": _to_cpu(model.state_dict()), "optimizer": _to_cpu(opt.state_dict())}


def _restore(model, opt, snap):
    model.load_state_dict({k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v)
                           for k, v in snap["model"].items()})
    opt.load_state_dict(snap["optimizer"])   # casts state back to the param devices


def train(cfg):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = getattr(cfg, "run_id", None)

    # ─── Data ─────────────────────────────────────────────────────────
    root_dir = getattr(cfg, "root_dir", None)
    if getattr(cfg, "feature_cache", None):
        from src.data.datasets import CachedFeatureDataset
        train_ds = CachedFeatureDataset(cfg.train_manifest, cfg.feature_cache,
                                        cfg.n_frames, cfg.audio_len)
    else:
        train_ds = AVDeepfakeDataset(cfg.train_manifest, cfg.shard_root,
                                     cfg.n_frames, cfg.audio_len, root_dir=root_dir, train=True)
    preflight_check(train_ds, name="train")
    sampler = BalancedBatchSampler(train_ds.records, cfg.batch_size, seed=cfg.seed)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                          num_workers=cfg.num_workers, collate_fn=collate,
                          pin_memory=(device == "cuda"), drop_last=True,
                          persistent_workers=False,
                          prefetch_factor=(2 if cfg.num_workers > 0 else None))

    # Validation set (optional)
    val_manifest = getattr(cfg, "val_manifest", None)
    val_dl = None
    if val_manifest and os.path.exists(val_manifest):
        val_ds = AVDeepfakeDataset(val_manifest, cfg.shard_root, cfg.n_frames, cfg.audio_len,
                                   root_dir=root_dir, train=False)
        val_max = int(getattr(cfg, "val_max_clips", 0) or 0)
        if val_max and len(val_ds) > val_max:
            val_ds.records = _stratified_subsample(val_ds.records, val_max)
            print(f"Validation subsampled to {len(val_ds)} clips (val_max_clips={val_max})")
        preflight_check(val_ds, name="val")
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate,
                            pin_memory=(device == "cuda"))
        print(f"Validation: {len(val_ds)} clips")

    # ─── Model ────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    if getattr(cfg, "init_from", None):
        if not os.path.exists(cfg.init_from):
            print(
                f"WARNING: init_from '{cfg.init_from}' not found on disk — "
                "skipping weight init and training from random initialisation.\n"
                "To fix: ensure the QACP checkpoint is downloaded from HF before "
                "launching Stage 1 (see Cell 10 in train_kaggle.ipynb)."
            )
        else:
            state = torch.load(cfg.init_from, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state["model"], strict=False)
            print(f"init_from {cfg.init_from}: {len(missing)} missing, {len(unexpected)} unexpected keys")
            if missing:
                print(f"  missing (first 10): {missing[:10]}")
            del state
            import gc
            gc.collect()

    if getattr(cfg, "gradient_checkpointing", True):
        enable_gradient_checkpointing(model)

    # Optional teacher for DAVID-Net-Lite: a full-size checkpoint whose saved cfg
    # rebuilds its own architecture; frozen, eval, forward only.
    teacher = None
    distill_w = float(getattr(cfg, "distill_weight", 0.0) or 0.0)
    distill_T = float(getattr(cfg, "distill_temperature", 2.0) or 2.0)
    if getattr(cfg, "distill_from", None) and distill_w > 0:
        t_state = torch.load(cfg.distill_from, map_location="cpu", weights_only=False)
        if "cfg" not in t_state:
            raise RuntimeError("distill_from checkpoint has no saved cfg (need best.pt/last.pt from train.py)")
        from types import SimpleNamespace
        t_cfg = SimpleNamespace(**t_state["cfg"])
        t_cfg.feature_cache = None
        teacher = build_model(t_cfg)
        teacher.load_state_dict(t_state["model"])
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"Distillation teacher loaded from {cfg.distill_from} "
              f"({sum(p.numel() for p in teacher.parameters()):,} params), weight={distill_w}, T={distill_T}")

    weights = LossWeights(**cfg.loss_weights)

    # Separate param groups: heads/fusion vs encoder adapters
    enc_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if ("video_encoder" in name or "audio_encoder" in name) else other_params).append(p)
    n_train = sum(p.numel() for p in enc_params + other_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_train:,} trainable / {n_total:,} total ({100 * n_train / max(1, n_total):.1f}%)")

    lr_enc = getattr(cfg, "lr_encoder", 1e-5)
    opt = torch.optim.AdamW([
        {"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": enc_params, "lr": lr_enc, "weight_decay": cfg.weight_decay},
    ])

    grad_accum = 1 if cfg.dry_run else max(1, int(getattr(cfg, "grad_accum_steps", 1)))
    max_micro = int(getattr(cfg, "max_steps_per_epoch", 0) or 0)   # 0 = full epoch
    n_micro = min(len(train_dl), max_micro) if max_micro else len(train_dl)
    steps_per_epoch = max(1, n_micro // grad_accum)                # optimizer steps
    warmup_epochs = getattr(cfg, "warmup_epochs", 2)
    total_opt_steps = cfg.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(int(warmup_epochs * steps_per_epoch), total_opt_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    max_norm = float(getattr(cfg, "max_grad_norm", 1.0))
    nan_patience = int(getattr(cfg, "nan_patience", 5))
    max_nan_restores = int(getattr(cfg, "max_nan_restores", 3))
    print(f"Effective batch: {cfg.batch_size} x {grad_accum} = {cfg.batch_size * grad_accum}; "
          f"{steps_per_epoch} optimizer steps/epoch; max_grad_norm={max_norm}")

    # ─── HF Backup ────────────────────────────────────────────────────
    backup = None
    start_epoch = 0
    best_auc = 0.0
    skip_samples = 0                       # mid-epoch resume offset (samples)
    ckpt_every = int(getattr(cfg, "checkpoint_every_steps", 300) or 0)   # optimizer steps; 0 = off

    if run_id:
        from src.utils.hf_backup import HFBackup
        backup = HFBackup(run_id=run_id, local_dir=getattr(cfg, "local_dir", "/kaggle/working"))
        backup.setup()

        if backup.is_complete(cfg.epochs):
            print(f"Run {run_id} already complete on HF ({cfg.epochs} epochs) — nothing to do.")
            return model
        resume = backup.load_resume_state()
        if resume is not None:
            start_epoch = resume.get("epoch", -1) + 1
            best_auc = resume.get("best_auc", 0.0)
            meta = resume.get("_hf_meta", {}) or {}
            if best_auc == 0.0:
                best_auc = meta.get("best_auc", 0.0)
            # mid-epoch checkpoint: resume inside the epoch at the recorded sample offset
            if meta.get("partial_epoch") is not None and int(meta["partial_epoch"]) >= start_epoch:
                start_epoch = int(meta["partial_epoch"])
                skip_samples = int(meta.get("partial_samples", 0))
            try:
                model.load_state_dict(resume["model"])
                opt.load_state_dict(resume["optimizer"])
                print(f"Resumed from HF: epoch {start_epoch}"
                      f"{f' (+{skip_samples} samples into it)' if skip_samples else ''}, best_auc={best_auc:.4f}")
            except Exception as e:  # noqa: BLE001
                print(f"Resume load warning: {e} — starting from scratch")
                start_epoch, skip_samples = 0, 0
        else:
            print("No resume state found — starting fresh")
    # fast-forward the LR schedule on resume (whole epochs + the partial one)
    for _ in range(start_epoch * steps_per_epoch + skip_samples // (cfg.batch_size * grad_accum)):
        scheduler.step()

    # ─── Augmentation ──────────────────────────────────────────────────
    use_aug = getattr(cfg, "augment", True)
    v_aug = VideoAugmentor(p=0.5) if use_aug else None
    a_aug = AudioAugmentor(p=0.5) if use_aug else None

    # ─── Training loop ────────────────────────────────────────────────
    micro_steps = 0
    opt_steps = start_epoch * steps_per_epoch + skip_samples // (cfg.batch_size * grad_accum)
    nan_restores = 0
    _host_mem("before-snapshot")
    snapshot = _snapshot(model, opt)   # last known-good weights (fp16 CPU copy)
    _host_mem("after-snapshot")
    model.train()
    t_run = time.time()

    try:
        for epoch in range(start_epoch, cfg.epochs):
            epoch_skip = skip_samples if epoch == start_epoch else 0
            sampler.set_epoch(epoch, skip=epoch_skip)
            epoch_loss, epoch_steps, nan_streak, nan_total = 0.0, 0, 0, 0
            t_epoch = time.time()
            opt.zero_grad(set_to_none=True)
            accum = 0
            seen = epoch_skip                  # samples consumed in this epoch
            steps_this_epoch = 0

            for it, batch in enumerate(train_dl):
                if max_micro and it >= max_micro:
                    break
                seen += batch["video"].size(0)
                batch = move(batch, device)
                batch = augment_batch(batch, v_aug, a_aug)
                v_av, a_av = availability_masks(batch, cfg.modality_dropout)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    out = model(batch["video"], batch["audio"], v_avail=v_av, a_avail=a_av)
                loss, parts = total_loss(out, batch, weights, model=model)
                if teacher is not None:
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
                        t_out = teacher(batch["video"], batch["audio"], v_avail=v_av, a_avail=a_av)
                    l_kd = distillation_loss(out, t_out, distill_T)
                    loss = loss + distill_w * l_kd
                    parts["kd"] = float(l_kd.detach())
                    parts["total"] = float(loss.detach())
                micro_steps += 1

                if not torch.isfinite(loss):
                    nan_streak += 1
                    nan_total += 1
                    print(f"epoch {epoch} micro-step {micro_steps} non-finite loss — skipping")
                    opt.zero_grad(set_to_none=True)
                    accum = 0
                else:
                    scaler.scale(loss / grad_accum).backward()
                    accum += 1
                    epoch_loss += loss.item()
                    epoch_steps += 1

                if accum >= grad_accum:
                    scaler.unscale_(opt)
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=max_norm)
                    if torch.isfinite(gnorm):
                        scaler.step(opt)      # applies the clipped update
                        scaler.update()
                        scheduler.step()
                        opt_steps += 1
                        nan_streak = 0
                    else:
                        # inf/NaN grads: GradScaler skips the step and lowers its scale.
                        # This is expected occasionally right after scale growth.
                        scaler.step(opt)
                        scaler.update()
                        nan_streak += 1
                        nan_total += 1
                        print(f"epoch {epoch} opt-step {opt_steps} non-finite grad norm — step skipped")
                    opt.zero_grad(set_to_none=True)
                    accum = 0

                    steps_this_epoch += 1
                    if opt_steps == 1:
                        _host_mem("after-first-step")
                        if device == "cuda":
                            print(f"[mem] peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                    # mid-epoch checkpoint: a killed session loses at most ckpt_every steps
                    if backup and ckpt_every and steps_this_epoch % ckpt_every == 0:
                        backup.push_checkpoint(
                            model, opt, epoch - 1, vars(cfg), {"partial": True},
                            milestone_every=10 ** 9, keep_milestones=int(getattr(cfg, "keep_milestones", 3)),
                            resume_extras={"best_auc": best_auc, "partial_epoch": epoch, "partial_samples": seen})
                        print(f"epoch {epoch} step {opt_steps}: mid-epoch checkpoint pushed ({seen} samples into epoch)")
                    if opt_steps % cfg.log_every == 0 and torch.isfinite(loss):
                        lr_now = opt.param_groups[0]["lr"]
                        print(f"epoch {epoch} step {opt_steps} " +
                              " ".join(f"{k}={v:.4f}" for k, v in parts.items()) +
                              f" gnorm={float(gnorm):.2f} lr={lr_now:.2e}")

                if nan_streak >= nan_patience:
                    nan_restores += 1
                    if nan_restores > max_nan_restores:
                        raise RuntimeError(
                            f"Training diverged: {nan_restores} restores from NaN did not help. "
                            "Inspect data (preflight), lower lr, or disable AMP.")
                    _restore(model, opt, snapshot)
                    for pg in opt.param_groups:      # after restore: halve the base LR
                        pg["initial_lr"] = pg.get("initial_lr", pg["lr"]) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(
                        opt, _lr_lambda(int(warmup_epochs * steps_per_epoch), total_opt_steps))
                    for _ in range(opt_steps):
                        scheduler.step()
                    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
                    nan_streak = 0
                    print(f"epoch {epoch}: {nan_patience} consecutive non-finite steps — restored "
                          f"last good snapshot, halved LR (restore #{nan_restores})")

                if cfg.dry_run and micro_steps >= 2 * grad_accum:
                    print("[dry-run] forward/backward OK, stopping.")
                    if backup:
                        backup.emergency_push(model, epoch)
                    return model

            # ─── End of epoch ──────────────────────────────────────────
            avg_loss = epoch_loss / max(epoch_steps, 1)
            dt = time.time() - t_epoch
            print(f"epoch {epoch} done in {dt / 60:.1f} min — loss={avg_loss:.4f} "
                  f"micro-steps={epoch_steps} skipped={nan_total}")
            if epoch_steps == 0:
                raise RuntimeError(f"epoch {epoch}: no finite training step at all — aborting")

            metrics = {"avg_loss": avg_loss, "epoch_minutes": dt / 60, "nan_steps": nan_total}
            if val_dl:
                val_metrics = validate(model, val_dl, device, weights)
                metrics.update(val_metrics)
                val_auc = (val_metrics["video_auc"] + val_metrics["audio_auc"]) / 2
                print(f"epoch {epoch} — val_loss={val_metrics['val_loss']:.4f} "
                      f"video_auc={val_metrics['video_auc']:.4f} audio_auc={val_metrics['audio_auc']:.4f} "
                      f"quad_acc={val_metrics['quadrant_acc']:.4f} quad_f1={val_metrics['quadrant_macro_f1']:.4f}")
                if val_auc > best_auc:
                    best_auc = val_auc
                    _save(model, cfg, epoch, "best")
                    if backup:
                        backup.push_best(model, epoch, val_auc)
                    print(f"  ** new best mean AUC: {val_auc:.4f}")
            _save(model, cfg, epoch, "last")
            snapshot = _snapshot(model, opt)

            if backup:
                metrics["best_auc"] = best_auc
                backup.push_checkpoint(model, opt, epoch, vars(cfg), metrics,
                                       milestone_every=int(getattr(cfg, "milestone_every", 5)),
                                       keep_milestones=int(getattr(cfg, "keep_milestones", 3)),
                                       resume_extras={"best_auc": best_auc})
                backup.push_log({
                    "epoch": epoch, "steps": epoch_steps,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

        final_metrics = {"epochs": cfg.epochs, "total_opt_steps": opt_steps, "best_auc": best_auc,
                         "minutes": (time.time() - t_run) / 60}
        if backup:
            backup.push_final(final_metrics)
        print(f"Training complete. Best mean AUC: {best_auc:.4f}")

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


def _save(model, cfg, epoch, tag: str = "last"):
    """Local checkpoint. Only `last.pt` and `best.pt` per run are kept (a 1 GB file per
    epoch filled Kaggle's 20 GB working disk before)."""
    run_id = getattr(cfg, "run_id", None) or "david_net"
    d = os.path.join(cfg.out_dir, run_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{tag}.pt")
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "epoch": epoch}, path)
    print(f"saved {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume-from", default=None, help="Path to checkpoint to resume from")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True
    if args.run_id:
        cfg.run_id = args.run_id
    if args.resume_from:
        cfg.init_from = args.resume_from
    train(cfg)


if __name__ == "__main__":
    main()
