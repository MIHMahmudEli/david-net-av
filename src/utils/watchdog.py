"""Session/process watchdog: periodic system + progress telemetry pushed to HF.

Why: Kaggle can kill the training process (signal) or the whole session (container)
without leaving a traceback, and every local file dies with it. This module writes a
small JSONL line every `interval_s` seconds and uploads it every `push_every_s` seconds,
so after a death the last lines on HF show *when* it happened and what the machine
looked like (RAM, VRAM, disk, /dev/shm, ffmpeg process count, load, GPU util) plus the
training position (epoch / micro-step / last clip ids).

Two entry points:
  * `Watchdog(...)` — inside the training process (train.py / pretrain_qacp.py)
  * `start_session_watchdog(...)` — inside the notebook kernel (survives a training
    process crash; if THIS stops too, the session itself was killed)

Also installs SIGTERM/SIGHUP handlers that log the signal and push immediately
(SIGINT stays on Python's KeyboardInterrupt path so the emergency checkpoint still runs).
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

# Telemetry is high-frequency and small; checkpoints are low-frequency and huge. HF
# caps commits per REPO per hour, so keeping them in one repo makes the watchdog starve
# the checkpoints. The control plane lives in its own repo with its own budget.
REPO_ID = "MoshinAli/david-net-av-coord"
REPO_TYPE = "dataset"


def _cgroup_mem() -> tuple:
    """(used_bytes, limit_bytes) for THIS container, or (None, None).

    psutil reports host-level memory and counts reclaimable page cache as available, so
    it cannot see a cgroup nearing its limit. Kaggle kills on the cgroup number, so that
    is the one worth watching. cgroup v2 first, then v1.
    """
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            used = int(f.read().strip())
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
        return used, (None if raw == "max" else int(raw))
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
            used = int(f.read().strip())
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            lim = int(f.read().strip())
        return used, (None if lim > (1 << 62) else lim)
    except Exception:  # noqa: BLE001
        return None, None


def _sys_snapshot() -> dict:
    snap = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "uptime_s": round(time.monotonic(), 1)}
    try:
        import psutil
        vm = psutil.virtual_memory()
        snap.update(rss_gb=round(psutil.Process(os.getpid()).memory_info().rss / 1e9, 2),
                    host_used_gb=round(vm.used / 1e9, 2), host_avail_gb=round(vm.available / 1e9, 2),
                    load=os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
                    n_ffmpeg=sum(1 for p in psutil.process_iter(["name"]) if (p.info["name"] or "").startswith("ffmpeg")),
                    n_python=sum(1 for p in psutil.process_iter(["name"]) if (p.info["name"] or "").startswith("python")))
    except Exception:  # noqa: BLE001
        pass
    for label, path in (("disk_free_gb", "/kaggle/working"), ("shm_free_gb", "/dev/shm"), ("tmp_free_gb", "/tmp")):
        try:
            u = shutil.disk_usage(path)
            snap[label] = round(u.free / 1e9, 2)
            if label == "shm_free_gb":
                snap["shm_total_gb"] = round(u.total / 1e9, 2)
        except Exception:  # noqa: BLE001
            pass
    cg_used, cg_max = _cgroup_mem()
    if cg_used is not None:
        snap["cgroup_used_gb"] = round(cg_used / 1e9, 2)
        if cg_max:
            snap["cgroup_max_gb"] = round(cg_max / 1e9, 2)
            snap["cgroup_pct"] = round(100.0 * cg_used / cg_max, 1)
    try:
        import torch
        if torch.cuda.is_available():
            snap.update(vram_alloc_gb=round(torch.cuda.memory_allocated() / 1e9, 2),
                        vram_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        if out:
            util, mem, temp = [x.strip() for x in out.splitlines()[0].split(",")]
            snap.update(gpu_util=int(util), gpu_mem_mb=int(mem), gpu_temp=int(temp))
    except Exception:  # noqa: BLE001
        pass
    return snap


class Watchdog:
    """Background telemetry + signal logging for one training process."""

    def __init__(self, run_id: str, local_dir: str = "/kaggle/working", interval_s: int = 60,
                 push_every_s: int = 600, tag: str = "train", token: Optional[str] = None,
                 fast_push_every_s: int = 60, fast_push_window_s: int = 300):
        self.run_id, self.tag = run_id, tag
        self.interval_s, self.push_every_s = interval_s, push_every_s
        # Deaths cluster in the first minutes of a stage, and a push only lands on HF every
        # `push_every_s` — so the fatal window is exactly what we lose. Push at a fine
        # cadence for the first `fast_push_window_s`, then back off to keep commits sane.
        self.fast_push_every_s, self.fast_push_window_s = fast_push_every_s, fast_push_window_s
        self._t0 = 0.0
        self.local = Path(local_dir) / f"watchdog_{tag}_{run_id}.jsonl"
        self.repo_path = f"runs/{run_id}/logs/watchdog_{tag}.jsonl"
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("hf")
        self.progress: dict = {}
        # 2 s samples of the cgroup counter. Shipped with each push, so the last push
        # before a kill carries the run-up rather than a single stale reading.
        from collections import deque
        self._trace = deque(maxlen=90)          # ~3 min of history
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_push = 0.0
        self._lock = threading.Lock()

    # -- progress from the training loop (cheap; called every micro-step)
    def update(self, **kw):
        self.progress.update(kw)

    def note(self, event: str, push: bool = False, **kw):
        self._write({"event": event, **kw})
        if push:
            self.push()

    def _write(self, extra: dict):
        line = {**_sys_snapshot(), **self.progress, **extra}
        if self._trace:
            line["mem_trace"] = list(self._trace)[-45:]   # ~90 s at 2 s resolution
        with self._lock:
            self.local.parent.mkdir(parents=True, exist_ok=True)
            with open(self.local, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, default=str) + "\n")
        return line

    def push(self):
        if not self.token or not self.local.exists():
            return
        try:
            from huggingface_hub import HfApi
            HfApi(token=self.token).upload_file(path_or_fileobj=str(self.local), path_in_repo=self.repo_path,
                                                repo_id=REPO_ID, repo_type=REPO_TYPE,
                                                commit_message=f"watchdog {self.tag} {self.run_id}")
            self._last_push = time.monotonic()
        except Exception as e:  # noqa: BLE001
            print(f"[watchdog] push failed: {str(e)[:120]}")

    def _push_interval(self) -> int:
        if self._t0 and (time.monotonic() - self._t0) < self.fast_push_window_s:
            return self.fast_push_every_s
        return self.push_every_s

    def _sample_loop(self):
        while not self._stop.wait(2.0):
            used, mx = _cgroup_mem()
            if used is not None:
                self._trace.append((time.strftime("%H:%M:%S"), round(used / 1e9, 2),
                                    round(100.0 * used / mx, 1) if mx else None))

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            line = self._write({"event": "tick"})
            print(f"[watchdog] {line.get('t')} ep={line.get('epoch')} micro={line.get('micro')} "
                  f"rss={line.get('rss_gb')}G avail={line.get('host_avail_gb')}G shm_free={line.get('shm_free_gb')}G "
                  f"disk_free={line.get('disk_free_gb')}G vram={line.get('vram_alloc_gb')}G gpu={line.get('gpu_util')}% "
                  f"ffmpeg={line.get('n_ffmpeg')} cgroup={line.get('cgroup_used_gb')}/"
                  f"{line.get('cgroup_max_gb')}GB ({line.get('cgroup_pct')}%)", flush=True)
            if time.monotonic() - self._last_push >= self._push_interval():
                self.push()

    def start(self):
        self._t0 = time.monotonic()
        self._install_signal_handlers()
        self.note("start", push=True, pid=os.getpid(), run_type=os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
                  cpu=os.cpu_count())
        self._thread = threading.Thread(target=self._loop, name="watchdog", daemon=True)
        self._thread.start()
        threading.Thread(target=self._sample_loop, name="watchdog-mem", daemon=True).start()
        return self

    def stop(self, event: str = "stop"):
        self._stop.set()
        self.note(event, push=True)

    def _install_signal_handlers(self):
        def handler(signum, frame):
            name = signal.Signals(signum).name
            self.note("signal", push=True, signal=name)
            print(f"[watchdog] received {name} — logged to HF, re-raising default action", flush=True)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        for sig in [getattr(signal, n) for n in ("SIGTERM", "SIGHUP") if hasattr(signal, n)]:
            try:
                signal.signal(sig, handler)
            except Exception:  # noqa: BLE001
                pass


def start_session_watchdog(session_id: str, local_dir: str = "/kaggle/working", interval_s: int = 60,
                           push_every_s: int = 600, current_cell: Optional[Callable[[], str]] = None,
                           fast_push_every_s: int = 60, fast_push_window_s: int = 300) -> Watchdog:
    """Kernel-level watchdog for the notebook: one line per minute with the same telemetry
    plus which cell is running. Lives in the notebook process, so it keeps reporting after
    a training subprocess dies; its LAST line on HF is the session's time of death."""
    wd = Watchdog(run_id=f"session_{session_id}", local_dir=local_dir, interval_s=interval_s,
                  push_every_s=push_every_s, tag="session", fast_push_every_s=fast_push_every_s,
                  fast_push_window_s=fast_push_window_s)
    if current_cell is not None:
        _orig_write = wd._write

        def _write(extra):
            try:
                extra = {"cell": current_cell(), **extra}
            except Exception:  # noqa: BLE001
                pass
            return _orig_write(extra)
        wd._write = _write
    return wd.start()
