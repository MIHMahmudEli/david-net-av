"""Logging, seeding, RNG state capture, environment fingerprint, GPU auto-config.

Everything here is side-effect free at import time so the notebook can import it
before deciding anything about the hardware.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

LOG = logging.getLogger("davidnet")

# Events the manuscript's reproducibility appendix relies on. `log_event` accepts any
# name, but these are the ones the pipeline emits; keeping them in one place makes the
# JSONL log greppable.
EVENTS = (
    "experiment_started", "dataset_loaded", "model_initialized", "training_started",
    "epoch_completed", "validation_completed", "checkpoint_saved", "checkpoint_uploaded",
    "checkpoint_upload_failed", "resumed", "evaluation_completed", "figures_generated",
    "results_uploaded", "experiment_completed", "session_paused", "oom_recovered",
    "nonfinite_loss", "error",
)


def quiet_third_party():
    """Silence progress-bar floods BEFORE huggingface_hub/transformers are imported.

    Kaggle reaps an interactive session whose output stream floods with carriage-return
    redraws (learned the hard way upstream); child processes inherit these variables."""
    for k, v in {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_VERBOSITY": "error",
                 "TRANSFORMERS_VERBOSITY": "error", "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
                 "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"}.items():
        os.environ[k] = v
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:  # noqa: BLE001
        pass


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _JsonlHandler(logging.Handler):
    """One JSON object per record: timestamp, level, event, message, fields."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord):
        try:
            row = {"t": utcnow(), "level": record.levelname,
                   "event": getattr(record, "event", None), "msg": record.getMessage()}
            row.update(getattr(record, "fields", {}) or {})
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)


def setup_logging(log_dir: str | Path, level: int = logging.INFO,
                  jsonl_name: str = "pipeline.jsonl") -> Path:
    """Console (human) + JSONL (machine) logging. Idempotent across notebook re-runs."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(level)
    LOG.propagate = False
    for h in list(LOG.handlers):
        LOG.removeHandler(h)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                           "%Y-%m-%d %H:%M:%S"))
    LOG.addHandler(console)
    path = log_dir / jsonl_name
    LOG.addHandler(_JsonlHandler(path))
    return path


def log_event(event: str, msg: str = "", level: int = logging.INFO, **fields):
    """Structured log line: `event` is machine-readable, `fields` become JSON keys."""
    text = f"[{event}] {msg}" if msg else f"[{event}]"
    if fields:
        short = ", ".join(f"{k}={_short(v)}" for k, v in fields.items())
        text = f"{text} ({short})"
    LOG.log(level, text, extra={"event": event, "fields": fields})


def _short(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


# ======================================================================== seeding
def seed_everything(seed: int, deterministic: bool = True):
    """Seed python/numpy/torch. `deterministic` trades a little speed for repeatable
    cuDNN kernels; full bitwise determinism on GPU is not guaranteed by PyTorch for
    every op (scatter/index_add backward), which is why runs are repeated over seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        # cuBLAS needs this for deterministic matmuls (documented PyTorch requirement)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def capture_rng_state() -> dict:
    """Everything needed to continue a random stream exactly after a restart."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except RuntimeError as e:  # different GPU count on the new session
            log_event("rng_partial_restore", f"CUDA RNG not restored: {e}", logging.WARNING)


# ======================================================================== environment
_TRACKED_PACKAGES = ("torch", "torchaudio", "torchvision", "transformers", "huggingface_hub",
                     "safetensors", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib",
                     "opencv-python", "opencv-python-headless", "accelerate")


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return None


def _cmd(args: list[str]) -> Optional[str]:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def git_revision(repo_dir: str | Path) -> dict:
    repo_dir = str(repo_dir)
    return {
        "commit": _cmd(["git", "-C", repo_dir, "rev-parse", "HEAD"]),
        "dirty": bool(_cmd(["git", "-C", repo_dir, "status", "--porcelain"])),
        "remote": _cmd(["git", "-C", repo_dir, "remote", "get-url", "origin"]),
    }


def gpu_info() -> list[dict]:
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append({"index": i, "name": p.name, "total_memory_gb": round(p.total_memory / 1e9, 2),
                    "capability": f"{p.major}.{p.minor}",
                    "multi_processor_count": p.multi_processor_count})
    return out


def environment_report(repo_dir: str | Path | None = None) -> dict:
    """The environment block every experiment stores in configs/environment.json."""
    rep = {
        "timestamp_utc": utcnow(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpus": gpu_info(),
        "packages": {p: _pkg_version(p) for p in _TRACKED_PACKAGES if _pkg_version(p)},
        "nvidia_smi": _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"]),
        "kaggle": {k: os.environ.get(k) for k in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE",
                                                   "KAGGLE_DOCKER_IMAGE", "KAGGLE_CONTAINER_NAME")
                   if os.environ.get(k)},
        "ffmpeg": (_cmd(["ffmpeg", "-version"]) or "").splitlines()[0:1],
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        rep["host_ram_gb"] = round(vm.total / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    if repo_dir is not None:
        rep["git"] = git_revision(repo_dir)
    return rep


def pip_freeze() -> str:
    """Exact package versions -> requirements.lock.txt in the experiment repo."""
    out = _cmd([sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"])
    return out or ""


# ======================================================================== GPU auto-config
def autoconfig_hardware(cfg: dict) -> dict:
    """Resolve 'auto' entries of the hardware block against the actual machine.

    Returns the resolved values; the caller writes them back into the config so the
    decision is recorded with the experiment (it changes nothing scientific: the
    EFFECTIVE batch size is held fixed and only its split into micro-batch x
    accumulation adapts to memory).
    """
    hw = dict(cfg.get("hardware", {}))
    gpus = gpu_info()
    has_gpu = bool(gpus)
    cap = tuple(int(x) for x in gpus[0]["capability"].split(".")) if has_gpu else (0, 0)
    vram = gpus[0]["total_memory_gb"] if has_gpu else 0.0

    # precision: bf16 needs Ampere (8.0+); T4 (7.5) / P100 (6.0) use fp16 + GradScaler.
    if hw.get("precision", "auto") == "auto":
        if not has_gpu or not cfg.get("mixed_precision", True):
            hw["precision"] = "fp32"
        elif cap >= (8, 0):
            hw["precision"] = "bf16"
        elif cap >= (6, 0):
            hw["precision"] = "fp16"
        else:
            hw["precision"] = "fp32"

    if hw.get("num_workers", "auto") == "auto":
        # Kaggle: 4 vCPU. Leave one for the main process; the memory cgroup (32 GB)
        # counts every worker's copy-on-write pages, so do not go wider.
        hw["num_workers"] = max(0, min(3, (os.cpu_count() or 1) - 1))

    hw["device"] = "cuda" if has_gpu else "cpu"
    hw["n_gpus"] = len(gpus)
    hw["gpu_name"] = gpus[0]["name"] if has_gpu else "cpu"
    hw["vram_gb"] = vram
    hw["pin_memory"] = has_gpu
    return hw


def resolve_micro_batch(effective_batch: int, per_device_hint: int | str, vram_gb: float,
                        bytes_per_sample_gb: float) -> tuple[int, int]:
    """Split a fixed effective batch into (micro_batch, grad_accum).

    `per_device_hint` 'auto' -> largest power of two whose activations fit in ~70% of
    VRAM given a measured/estimated per-sample footprint.
    """
    if per_device_hint != "auto":
        mb = int(per_device_hint)
    else:
        budget = max(0.5, 0.7 * vram_gb) if vram_gb else 2.0
        mb = 1
        while mb * 2 <= effective_batch and (mb * 2) * bytes_per_sample_gb <= budget:
            mb *= 2
    mb = max(1, min(mb, effective_batch))
    accum = max(1, round(effective_batch / mb))
    return mb, accum


class Stopwatch:
    """Wall-clock budget guard for the Kaggle 12 h session limit."""

    def __init__(self, budget_hours: float, safety_minutes: float = 20.0):
        self.t0 = time.time()
        self.budget_s = budget_hours * 3600.0
        self.safety_s = safety_minutes * 60.0

    def elapsed_h(self) -> float:
        return (time.time() - self.t0) / 3600.0

    def remaining_s(self) -> float:
        return self.budget_s - (time.time() - self.t0)

    def should_stop(self) -> bool:
        return self.remaining_s() < self.safety_s
