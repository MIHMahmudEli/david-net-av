"""HuggingFace backup module for crash-proof training.

Handles checkpoint upload, resume, and emergency push to a HuggingFace repo.
Designed for Kaggle sessions that can die without warning.

Usage:
    from src.utils.hf_backup import HFBackup

    backup = HFBackup(run_id="run_001")
    backup.setup()  # creates repo if needed, checks for resume state

    # At startup - check for resume
    state = backup.load_resume_state()
    if state:
        start_epoch = state["epoch"]
        model.load_state_dict(state["model"])

    # After each epoch
    backup.push_checkpoint(model, optimizer, epoch, config, metrics)

    # When val metric improves
    backup.push_best(model, epoch, val_metric)

    # At end of run
    backup.push_final(metrics, figures_dir)

    # On crash (wrap training loop)
    try:
        train(...)
    except Exception:
        backup.emergency_push(model, optimizer, epoch)
        raise
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _retry(max_retries: int = 3, base_delay: float = 2.0):
    """Decorator: retry with exponential backoff. Never crashes training."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"[HFBackup] {func.__name__} attempt {attempt+1}/{max_retries} "
                                   f"failed: {e}. Retrying in {delay:.0f}s...")
                    time.sleep(delay)
            logger.error(f"[HFBackup] {func.__name__} failed after {max_retries} attempts: {last_err}")
            return None
        return wrapper
    return decorator


class HFBackup:
    """Crash-proof HuggingFace backup for training runs.

    Repo layout:
        runs/<run_id>/
            checkpoints/epoch_NNNN.pt
            best/best.pt
            logs/train_log.jsonl
            metrics/metrics.json
            figures/*.pdf
            state/resume_state.json
    """

    def __init__(
        self,
        run_id: str,
        repo_id: str = "david-net-av/backup",
        repo_type: str = "model",
        local_dir: str = "/kaggle/working",
    ):
        self.run_id = run_id
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.base_path = f"runs/{run_id}"
        self.local_dir = Path(local_dir)
        self._api = None
        self._token = None

    def _get_token(self) -> str:
        """Read HF_TOKEN from environment. Fail loudly if missing."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("hf")
        if not token:
            raise RuntimeError(
                "HF_TOKEN not found. Set it in Kaggle Secrets or .env.\n"
                "Kaggle: add via Settings -> Secrets -> Add -> Name=HF_TOKEN"
            )
        self._token = token
        return token

    def _get_api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self._get_token())
        return self._api

    def setup(self):
        """Create repo if needed. No-op if already exists."""
        api = self._get_api()
        try:
            api.create_repo(self.repo_id, repo_type=self.repo_type, exist_ok=True)
            logger.info(f"[HFBackup] Repo {self.repo_id} ready")
        except Exception as e:
            logger.warning(f"[HFBackup] Repo setup note: {e}")

    # ─── Resume logic ────────────────────────────────────────────────────

    def load_resume_state(self) -> Optional[dict]:
        """Check HF repo for existing resume_state.json for this run_id.

        Returns the state dict (containing epoch, model, optimizer, rng, config)
        or None if no resume point exists.

        This is account-agnostic: the HF repo path is `runs/<run_id>/`,
        so any Kaggle account can resume the same run.
        """
        api = self._get_api()
        state_path = f"{self.base_path}/state/resume_state.json"
        ckpt_path = f"{self.base_path}/checkpoints"

        try:
            # List checkpoint files to find the latest
            files = api.list_repo_tree(
                self.repo_id, path_in_repo=ckpt_path,
                repo_type=self.repo_type, recursive=True
            )
            ckpt_files = sorted(
                [f for f in files if hasattr(f, "path") and f.path.endswith(".pt")],
                key=lambda f: f.path
            )
            if not ckpt_files:
                logger.info("[HFBackup] No existing checkpoints found — starting fresh")
                return None

            latest_ckpt = ckpt_files[-1]
            logger.info(f"[HFBackup] Found checkpoint: {latest_ckpt.path}")

            # Download the checkpoint
            ckpt_local = api.hf_hub_download(
                self.repo_id, latest_ckpt.path, repo_type=self.repo_type
            )

            # Try to download resume_state.json
            try:
                state_local = api.hf_hub_download(
                    self.repo_id, state_path, repo_type=self.repo_type
                )
                with open(state_local) as f:
                    state_meta = json.load(f)
            except Exception:
                state_meta = {}

            # Load checkpoint
            import torch
            ckpt = torch.load(ckpt_local, map_location="cpu")
            ckpt["_hf_meta"] = state_meta
            return ckpt

        except Exception as e:
            logger.warning(f"[HFBackup] Resume check failed: {e}")
            return None

    # ─── Upload functions ────────────────────────────────────────────────

    @_retry(max_retries=3, base_delay=2.0)
    def _upload_file(self, local_path: str, repo_path: str):
        """Upload a single file with retry."""
        api = self._get_api()
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )

    @_retry(max_retries=3, base_delay=2.0)
    def _upload_bytes(self, data: bytes, repo_path: str):
        """Upload bytes with retry."""
        import io
        api = self._get_api()
        api.upload_file(
            path_or_fileobj=io.BytesIO(data),
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )

    def push_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        config: dict,
        extra: Optional[dict] = None,
    ):
        """Push checkpoint + resume_state.json after each epoch."""
        import torch

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            **(extra or {}),
        }

        # Save locally first
        ckpt_dir = self.local_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        local_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        torch.save(state, local_path)

        # Upload checkpoint
        repo_path = f"{self.base_path}/checkpoints/epoch_{epoch:04d}.pt"
        self._upload_file(str(local_path), repo_path)

        # Update resume_state.json
        resume_state = {
            "epoch": epoch,
            "run_id": self.run_id,
            "checkpoint": repo_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        state_bytes = json.dumps(resume_state, indent=2).encode()
        self._upload_bytes(state_bytes, f"{self.base_path}/state/resume_state.json")

        logger.info(f"[HFBackup] Pushed checkpoint epoch {epoch}")

        # Keep only last 3 local checkpoints to save disk
        self._cleanup_local_checkpoints(keep=3)

    def push_best(self, model, epoch: int, metric: float):
        """Push best model when validation metric improves."""
        import torch

        ckpt_dir = self.local_dir / "best"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        local_path = ckpt_dir / "best.pt"
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "metric": metric,
        }, local_path)

        self._upload_file(str(local_path), f"{self.base_path}/best/best.pt")

        # Also save metric value
        meta = json.dumps({"epoch": epoch, "metric": metric}, indent=2).encode()
        self._upload_bytes(meta, f"{self.base_path}/best/best_meta.json")
        logger.info(f"[HFBackup] Pushed best model (epoch {epoch}, metric={metric:.4f})")

    def push_log(self, entry: dict):
        """Append a line to train_log.jsonl (appends locally, re-uploads full file)."""
        log_path = self.local_dir / "train_log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Re-upload full log (small file, acceptable)
        self._upload_file(str(log_path), f"{self.base_path}/logs/train_log.jsonl")

    def push_metrics(self, metrics: dict):
        """Push final/intermediate metrics dict."""
        data = json.dumps(metrics, indent=2, default=str).encode()
        self._upload_bytes(data, f"{self.base_path}/metrics/metrics.json")

    def push_figures(self, figures_dir: str):
        """Push all PDF (and PNG) figures from a directory."""
        fig_dir = Path(figures_dir)
        if not fig_dir.exists():
            return
        for f in fig_dir.glob("*.pdf"):
            self._upload_file(str(f), f"{self.base_path}/figures/{f.name}")
        for f in fig_dir.glob("*.png"):
            self._upload_file(str(f), f"{self.base_path}/figures/{f.name}")
        logger.info(f"[HFBackup] Pushed figures from {figures_dir}")

    def push_final(self, metrics: dict, figures_dir: Optional[str] = None):
        """Push final metrics + figures at end of run."""
        self.push_metrics(metrics)
        if figures_dir:
            self.push_figures(figures_dir)

    def emergency_push(self, model, epoch: int):
        """Last-ditch push on crash. Best-effort, never raises."""
        try:
            logger.warning(f"[HFBackup] Emergency push at epoch {epoch}")
            import torch
            ckpt_dir = self.local_dir / "emergency"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            local_path = ckpt_dir / f"emergency_epoch_{epoch:04d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch}, local_path)
            self._upload_file(str(local_path), f"{self.base_path}/emergency/emergency_epoch_{epoch:04d}.pt")
        except Exception as e:
            logger.error(f"[HFBackup] Emergency push failed: {e}")

    def _cleanup_local_checkpoints(self, keep: int = 3):
        """Delete old local checkpoints, keeping only the latest N."""
        ckpt_dir = self.local_dir / "checkpoints"
        if not ckpt_dir.exists():
            return
        files = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.name)
        for f in files[:-keep]:
            f.unlink(missing_ok=True)


# ─── Convenience: crash wrapper ──────────────────────────────────────────

def crash_guard(backup: HFBackup, model, get_epoch):
    """Context manager that does emergency push on exception.

    Usage:
        with crash_guard(backup, model, lambda: epoch):
            train(...)
    """
    class _Guard:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                backup.emergency_push(model, get_epoch())
            return False  # don't suppress the exception
    return _Guard()
