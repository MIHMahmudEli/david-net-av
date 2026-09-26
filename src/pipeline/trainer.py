"""One resumable training loop for QACP (Stage 0), Stage 1, baselines and Phase B.

Resume semantics
----------------
Checkpoints are taken only at optimizer-step boundaries, so no partially accumulated
gradient ever needs saving. The loop state records `samples_in_epoch`; on resume the
deterministic EpochSampler skips exactly that many samples of the same epoch order, the
RNG streams are restored, and training continues with the next optimizer step it would
have taken without the interruption. (On CPU / deterministic kernels the resumed run
is bitwise identical to an uninterrupted one -- tests/test_pipeline_recovery.py.)

Failure handling
----------------
* CUDA OOM: halve the micro-batch, double gradient accumulation (the EFFECTIVE batch,
  and therefore the optimization problem, is unchanged), roll back to the last step
  boundary, continue. Recorded in the loop state.
* Non-finite loss: the micro-batch is skipped; too many in a row aborts the run
  loudly instead of producing a silent chance-level model.
* Session budget: before Kaggle's 12 h wall, checkpoint + flush and return "paused".
* DAVIDNET_CRASH_AFTER_STEP=N (tests only): hard-exit after the checkpoint at step N,
  exactly like a killed Kaggle container.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pipeline.env import (capture_rng_state, log_event, resolve_micro_batch,
                              restore_rng_state, utcnow)
from src.pipeline.features import EpochSampler, collate


# ====================================================================== job description
@dataclass
class TrainJob:
    exp: dict
    cfg: dict
    stage: str                                   # key of cfg["train"]
    model: nn.Module
    train_ds: torch.utils.data.Dataset
    val_ds: torch.utils.data.Dataset
    objective: Callable                           # (model, batch, cfg, train) -> (loss, parts)
    validate: Callable                            # (model, loader, ctx) -> {"val/...": float}
    sampler_keys: Optional[list] = None
    param_groups: Optional[list] = None
    selection_mode: str = "max"
    init_state: Optional[dict] = None             # warm start (fresh runs only)
    bytes_per_sample_gb: float = 0.02
    early_stop_fn: Optional[Callable] = None      # (history) -> reason | None
    num_workers: Optional[int] = None             # None -> hardware default; 0 for in-memory features


@dataclass
class RunContext:
    hw: dict
    ckpt: object                                  # CheckpointManager
    local_dir: Path
    registry: object = None
    stopwatch: object = None
    heartbeat_minutes: float = 10.0
    extra_files: dict = field(default_factory=dict)


def warmup_cosine(step: int, warmup: int, total: int, floor: float) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))


def _workers(job, hw) -> int:
    return hw["num_workers"] if job.num_workers is None else job.num_workers


def _loader_generator(seed: int) -> torch.Generator:
    """A DataLoader draws its base seed from the GLOBAL torch RNG every time an iterator
    is created unless it owns a generator. Iterators are created at different points of
    the RNG stream in an uninterrupted vs a resumed run, which silently shifts every
    later random draw (modality-dropout masks). A private generator removes that."""
    return torch.Generator().manual_seed(seed)


def _to(batch: dict, device: str) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


# ====================================================================== the loop
def train(job: TrainJob, ctx: RunContext) -> dict:
    cfg, tcfg = job.cfg, job.cfg["train"][job.stage]
    hw = ctx.hw
    device = hw["device"]
    prec = hw["precision"]
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(prec)
    use_scaler = prec == "fp16"
    model = job.model.to(device)
    seed = int(job.exp["seed"])
    epochs = int(tcfg["epochs"])
    eff = int(tcfg["effective_batch"])
    micro, accum = resolve_micro_batch(eff, tcfg["micro_batch"], hw.get("vram_gb", 0.0),
                                       job.bytes_per_sample_gb)
    groups = job.param_groups or [{"params": [p for p in model.parameters() if p.requires_grad]}]
    opt = torch.optim.AdamW(groups, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
                            betas=tuple(tcfg["betas"]))
    steps_per_epoch = max(1, math.ceil(len(job.train_ds) / eff))
    total_steps = steps_per_epoch * epochs
    warmup = int(tcfg["warmup_ratio"] * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, warmup, total_steps, tcfg["min_lr_ratio"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    params = [p for g in opt.param_groups for p in g["params"]]

    st = {"epoch": 0, "samples_in_epoch": 0, "step": 0, "history": [],
          "best": {"metric": None, "epoch": None, "step": None}, "bad_epochs": 0,
          "micro": micro, "accum": accum, "completed": False, "train_seconds": 0.0,
          "oom_events": [], "nonfinite_total": 0, "sessions": [],
          "epoch_accum": {"n": 0, "sums": {}}, "stop_reason": None}

    resumed = ctx.ckpt.latest(map_location="cpu")
    if resumed is not None:
        state, meta = resumed
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        if use_scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        st = state["loop"]
        restore_rng_state(state["rng"])
        log_event("resumed", f"{job.exp['exp_id']} from step {st['step']} "
                  f"(epoch {st['epoch']}, {st['samples_in_epoch']} samples into it)",
                  step=st["step"], epoch=st["epoch"])
    elif job.init_state is not None:
        missing, unexpected = model.load_state_dict(job.init_state, strict=False)
        log_event("warm_start", f"{len(missing)} missing / {len(unexpected)} unexpected keys",
                  missing=len(missing), unexpected=len(unexpected))
    if st["completed"]:
        log_event("experiment_completed", f"{job.exp['exp_id']} already complete")
        return _summary(job, st, "completed")

    st["sessions"].append({"started": utcnow(), "gpu": hw.get("gpu_name"),
                           "precision": prec, "resumed_from_step": st["step"]})
    sampler = EpochSampler(job.sampler_keys or [0] * len(job.train_ds),
                           tcfg.get("sampler", "uniform") if job.sampler_keys else "uniform",
                           seed)
    val_loader = DataLoader(job.val_ds, batch_size=max(64, st["micro"]), shuffle=False,
                            num_workers=_workers(job, hw), collate_fn=collate,
                            pin_memory=hw["pin_memory"], generator=_loader_generator(seed))
    logs = ctx.local_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    step_log = logs / "train_steps.jsonl"
    hist_csv = ctx.local_dir / "metrics" / "training_history.csv"
    extra = {"logs/train_steps.jsonl": step_log, "metrics/training_history.csv": hist_csv,
             **ctx.extra_files}
    crash_after = int(os.environ.get("DAVIDNET_CRASH_AFTER_STEP", "0") or 0)
    if os.environ.get("DAVIDNET_CRASH_EXP") not in (None, "", job.exp.get("name")):
        crash_after = 0                          # the test targets a different experiment
    last_ckpt = {"t": time.time(), "step": st["step"]}
    last_hb = [time.time()]
    ck = cfg["checkpoint"]
    log_event("training_started", f"{job.exp['exp_id']} {job.stage}: {len(job.train_ds)} train "
              f"/ {len(job.val_ds)} val, micro {st['micro']} x accum {st['accum']} = {eff}, "
              f"{total_steps} steps, {prec}", exp_id=job.exp["exp_id"])

    def state_dict() -> dict:
        return {"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict() if use_scaler else None,
                "rng": capture_rng_state(), "loop": st, "stage": job.stage,
                "exp_id": job.exp["exp_id"], "saved_at": utcnow()}

    def save(reason: str):
        ctx.ckpt.save(state_dict(), st["step"], st["epoch"], reason, extra_files=extra)
        last_ckpt.update(t=time.time(), step=st["step"])
        if crash_after and st["step"] >= crash_after:
            ctx.ckpt.wait()
            log_event("simulated_crash", f"hard exit after step {st['step']}", logging.WARNING)
            os._exit(137)

    def heartbeat():
        if ctx.registry is not None and time.time() - last_hb[0] > ctx.heartbeat_minutes * 60:
            try:
                ctx.registry.heartbeat(job.exp["exp_id"], progress_step=st["step"],
                                       progress_epoch=st["epoch"])
            except Exception as e:  # noqa: BLE001
                log_event("heartbeat_failed", str(e)[:120], logging.WARNING)
            last_hb[0] = time.time()

    # -------------------------------------------------------------- epochs
    while st["epoch"] < epochs:
        ep = st["epoch"]
        if hasattr(job.train_ds, "set_epoch"):
            job.train_ds.set_epoch(ep)
        t_epoch = time.time()
        try:
            status = _run_epoch(job, ctx, model, opt, sched, scaler, sampler, st, params,
                                amp_dtype, device, tcfg, step_log, last_ckpt, ck, save,
                                heartbeat)
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if st["micro"] == 1:
                raise
            st["micro"] //= 2
            st["accum"] = max(1, round(eff / st["micro"]))
            st["oom_events"].append({"step": st["step"], "micro": st["micro"], "t": utcnow()})
            log_event("oom_recovered", f"micro-batch -> {st['micro']} x accum {st['accum']} "
                      "(effective batch unchanged)", logging.WARNING)
            continue
        st["train_seconds"] += time.time() - t_epoch
        if status == "paused":
            save("session_budget")
            ctx.ckpt.flush(raise_on_error=False)
            log_event("session_paused", f"{job.exp['exp_id']} paused at step {st['step']}")
            return _summary(job, st, "paused")

        # ---------------- end of epoch: validation, selection, history, checkpoint
        t_val = time.time()
        val = job.validate(model, val_loader, ctx)
        acc = st["epoch_accum"]
        row = {"epoch": ep + 1, "step": st["step"], "lr": opt.param_groups[0]["lr"],
               **{f"train/{k}": v / max(1, acc["n"]) for k, v in acc["sums"].items()},
               **val, "epoch_seconds": round(time.time() - t_epoch, 1),
               "val_seconds": round(time.time() - t_val, 1)}
        st["history"].append(row)
        _write_history(hist_csv, st["history"])
        metric = val.get(tcfg["selection_metric"])
        better = metric is not None and not (isinstance(metric, float) and math.isnan(metric)) and (
            st["best"]["metric"] is None or
            (metric > st["best"]["metric"] if job.selection_mode == "max" else metric < st["best"]["metric"]))
        if better:
            st["best"] = {"metric": float(metric), "epoch": ep + 1, "step": st["step"]}
            st["bad_epochs"] = 0
            ctx.ckpt.mark_best(model.state_dict(), {"epoch": ep + 1, "step": st["step"],
                                                    "metric_name": tcfg["selection_metric"],
                                                    "metric": float(metric),
                                                    "exp_id": job.exp["exp_id"]}, cfg)
        else:
            st["bad_epochs"] += 1
        log_event("epoch_completed", f"{job.exp['exp_id']} epoch {ep + 1}/{epochs}",
                  **{k: v for k, v in row.items() if isinstance(v, (int, float))},
                  best=st["best"]["metric"], improved=better)
        log_event("validation_completed", f"{tcfg['selection_metric']}={metric}")
        st["epoch"] += 1
        st["samples_in_epoch"] = 0
        st["epoch_accum"] = {"n": 0, "sums": {}}
        reason = None
        if st["bad_epochs"] >= tcfg["early_stopping_patience"]:
            reason = f"early stopping: no improvement in {st['bad_epochs']} epochs"
        elif job.early_stop_fn is not None:
            reason = job.early_stop_fn(st["history"])
        if reason:
            st["stop_reason"] = reason
            log_event("early_stop", reason)
            break
        # interval-driven only (HF allows ~128 commits/h/repo): a Phase-A epoch takes
        # seconds, so saving every epoch would exhaust the budget. The final state is
        # saved as "completed" below; a pause saves immediately.
        due = (st["step"] - last_ckpt["step"] >= ck["every_steps"] or
               time.time() - last_ckpt["t"] >= ck["every_minutes"] * 60)
        if st["epoch"] < epochs and due:
            save("epoch_end")
        heartbeat()
        if ctx.stopwatch is not None and ctx.stopwatch.should_stop() and st["epoch"] < epochs:
            ctx.ckpt.flush(raise_on_error=False)
            log_event("session_paused", f"{job.exp['exp_id']} paused after epoch {st['epoch']}")
            return _summary(job, st, "paused")

    st["completed"] = True
    st["stop_reason"] = st["stop_reason"] or "max epochs reached"
    save("completed")
    ctx.ckpt.flush(raise_on_error=True)
    log_event("training_finished", f"{job.exp['exp_id']}: best {tcfg['selection_metric']}="
              f"{st['best']['metric']} at epoch {st['best']['epoch']}")
    return _summary(job, st, "completed")


def _run_epoch(job, ctx, model, opt, sched, scaler, sampler, st, params, amp_dtype, device,
               tcfg, step_log, last_ckpt, ck, save, heartbeat) -> str:
    sampler.set_epoch(st["epoch"], skip=st["samples_in_epoch"])
    loader = DataLoader(job.train_ds, batch_size=st["micro"], sampler=sampler,
                        num_workers=_workers(job, ctx.hw), collate_fn=collate,
                        pin_memory=ctx.hw["pin_memory"], drop_last=False,
                        generator=_loader_generator(int(job.exp["seed"])))
    model.train()
    opt.zero_grad(set_to_none=True)
    micro_i, pending, nonfinite_run = 0, 0, 0
    parts_sum: dict = {}
    t0 = time.time()
    cfg = job.cfg
    use_scaler = scaler.is_enabled()

    def optimizer_step():
        nonlocal micro_i, pending, parts_sum, t0
        if use_scaler:
            scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, tcfg["max_grad_norm"]))
        if use_scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        st["step"] += 1
        st["samples_in_epoch"] += pending
        acc = st["epoch_accum"]
        acc["n"] += 1
        for k, v in parts_sum.items():
            acc["sums"][k] = acc["sums"].get(k, 0.0) + v / max(1, micro_i)
        dt = time.time() - t0
        rec = {"t": utcnow(), "step": st["step"], "epoch": st["epoch"] + 1,
               "lr": opt.param_groups[0]["lr"], "grad_norm": gnorm,
               "samples_per_s": round(pending / max(dt, 1e-6), 1),
               **{k: v / max(1, micro_i) for k, v in parts_sum.items()}}
        with open(step_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        micro_i, pending, parts_sum, t0 = 0, 0, {}, time.time()

    for batch in loader:
        batch = _to(batch, device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            loss, parts = job.objective(model, batch, cfg, True)
        bs = len(batch["video"])
        if not torch.isfinite(loss):
            nonfinite_run += 1
            st["nonfinite_total"] += 1
            log_event("nonfinite_loss", f"step {st['step']}: skipped micro-batch",
                      logging.WARNING, run=nonfinite_run)
            if nonfinite_run >= tcfg.get("nonfinite_patience", 20):
                raise FloatingPointError(f"{nonfinite_run} consecutive non-finite losses -- "
                                         "aborting instead of training a broken model")
            pending += bs          # the samples were consumed; keep resume position exact
            continue
        nonfinite_run = 0
        (scaler.scale(loss / st["accum"]) if use_scaler else loss / st["accum"]).backward()
        micro_i += 1
        pending += bs
        for k, v in parts.items():
            parts_sum[k] = parts_sum.get(k, 0.0) + float(v)
        if micro_i >= st["accum"]:
            optimizer_step()
            heartbeat()
            due = (st["step"] - last_ckpt["step"] >= ck["every_steps"] or
                   time.time() - last_ckpt["t"] >= ck["every_minutes"] * 60)
            if due:
                save("interval")
            if ctx.stopwatch is not None and ctx.stopwatch.should_stop():
                return "paused"
    if micro_i > 0:
        optimizer_step()
    elif pending:
        st["samples_in_epoch"] += pending
    return "finished"


def _write_history(path: Path, history: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in history:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in history:
            w.writerow({k: r.get(k, "") for k in keys})


def _summary(job: TrainJob, st: dict, status: str) -> dict:
    return {"exp_id": job.exp["exp_id"], "status": status, "stage": job.stage,
            "epochs_run": st["epoch"], "steps": st["step"], "best": st["best"],
            "stop_reason": st.get("stop_reason"), "train_hours": round(st["train_seconds"] / 3600, 3),
            "oom_events": st["oom_events"], "nonfinite_total": st["nonfinite_total"],
            "sessions": st["sessions"], "micro_batch": st["micro"], "grad_accum": st["accum"]}


# ====================================================================== objectives
def _masks(batch: dict, p_drop: float, train: bool):
    v_av, a_av = batch["v_avail"].float(), batch["a_avail"].float()
    if train and p_drop > 0:
        B = v_av.shape[0]
        drop = torch.rand(B, device=v_av.device) < p_drop
        which = torch.rand(B, device=v_av.device) < 0.5
        both = (v_av > 0) & (a_av > 0)               # never drop the only stream
        v_av = torch.where(drop & which & both, torch.zeros_like(v_av), v_av)
        a_av = torch.where(drop & ~which & both, torch.zeros_like(a_av), a_av)
    return v_av, a_av


def stage1_objective(model, batch, cfg, train: bool, stage: str = "stage1"):
    from src.training.losses import LossWeights, total_loss
    t = cfg["train"][stage]
    v_av, a_av = _masks(batch, t["modality_dropout"] if train else 0.0, train)
    out = model(batch["video"], batch["audio"], v_avail=v_av, a_avail=a_av)
    b = dict(batch, quadrant=batch["quadrant"].clamp(min=0))
    loss, parts = total_loss(out, b, LossWeights(**t["loss_weights"]), model=model)
    return loss, parts


def qacp_objective(model, batch, cfg, train: bool):
    from src.training.losses import qacp_loss
    out = model(batch["video"], batch["audio"])
    return qacp_loss(out, batch, temperature=cfg["train"]["qacp"]["temperature"])


def baseline_objective(model, batch, cfg, train: bool):
    from src.training.losses import focal_bce
    import torch.nn.functional as F
    out = model(batch["video"], batch["audio"], v_avail=batch["v_avail"].float(),
                a_avail=batch["a_avail"].float())
    loss = batch["video"].new_zeros((), dtype=torch.float32)
    parts = {}
    for key, lab, av in (("logit_v", "video_label", "v_avail"), ("logit_a", "audio_label", "a_avail")):
        logit = out[key]
        if torch.isnan(logit).all():
            continue
        l = focal_bce(logit, batch[lab].float().clamp(min=0),
                      mask=batch[av].float() * (batch[lab] >= 0).float())
        loss = loss + l
        parts[key[-1]] = float(l.detach())
    if out.get("logit_quad") is not None:
        both = batch["v_avail"].float() * batch["a_avail"].float() * (batch["quadrant"] >= 0).float()
        lq = (F.cross_entropy(out["logit_quad"].float(), batch["quadrant"].clamp(min=0),
                              reduction="none") * both).sum() / both.sum().clamp(min=1)
        loss = loss + 0.5 * lq
        parts["quad"] = float(lq.detach())
    parts["total"] = float(loss.detach())
    return loss, parts


# ====================================================================== validators
@torch.no_grad()
def _val_loss(model, loader, objective, cfg, device, amp_dtype) -> dict:
    model.eval()
    sums, n = {}, 0
    for batch in loader:
        batch = _to(batch, device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            _, parts = objective(model, batch, cfg, False)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    return {f"val/{k}_loss" if k != "total" else "val/loss": v / max(1, n) for k, v in sums.items()}


def make_supervised_validator(objective, cfg):
    from src.pipeline.evaluation import predict, task_frame
    from sklearn.metrics import roc_auc_score, f1_score

    def validate(model, loader, ctx) -> dict:
        device = ctx.hw["device"]
        amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(ctx.hw["precision"])
        out = _val_loss(model, loader, objective, cfg, device, amp)
        df = predict(model, loader, device, amp)
        aucs = []
        for task in ("video", "audio", "clip"):
            y, p = task_frame(df, task)
            if len(np.unique(y)) == 2:
                out[f"val/{task}_auc"] = float(roc_auc_score(y, p))
                if task != "clip":
                    aucs.append(out[f"val/{task}_auc"])
                out[f"val/{task}_f1@0.5"] = float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))
        out["val/mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
        cols = [c for c in df.columns if c.startswith("p_quad_")]
        m = df["quadrant_label"] != ""
        if m.any() and not df.loc[m, cols].isna().all().all():
            from src.pipeline.features import QUAD_NAMES
            y = df.loc[m, "quadrant_label"].map({n: i for i, n in enumerate(QUAD_NAMES)})
            out["val/quad_macro_f1"] = float(f1_score(y, df.loc[m, cols].to_numpy().argmax(1),
                                                      average="macro", zero_division=0))
        return out
    return validate


def make_qacp_validator(cfg):
    from src.training.losses import at_loss_floor

    def validate(model, loader, ctx) -> dict:
        device = ctx.hw["device"]
        amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(ctx.hw["precision"])
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(0)          # a fixed validation draw every epoch
        out = _val_loss(model, loader, qacp_objective, cfg, device, amp)
        parts = {k.replace("val/", "").replace("_loss", ""): v for k, v in out.items()}
        out["val/qacp_loss"] = out.pop("val/loss")
        out["val/at_floor"] = float(at_loss_floor(parts, loader.batch_size))
        return out
    return validate


def qacp_floor_stop(history: list[dict]) -> Optional[str]:
    """Stop when the validation SupCon terms sit on their analytic floors 3 epochs running."""
    if len(history) >= 3 and all(h.get("val/at_floor", 0.0) >= 1.0 for h in history[-3:]):
        return "QACP SupCon terms at their loss floor for 3 epochs (nothing left to learn)"
    return None
