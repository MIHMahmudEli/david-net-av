"""Full training state -> local disk -> Hugging Face, and verified discovery for resume.

Remote layout under an experiment directory (e.g. experiments/EXP_007_davidnet-full_s42/):

    checkpoints/step-000500/training_state.pt   model+optimizer+scheduler+scaler+RNG+loop
    checkpoints/step-000500/meta.json           step, epoch, sha256, config hash, reason
    checkpoints/LATEST.json                     pointer + history of VERIFIED checkpoints
    best_model/model.safetensors                weights of the best-on-validation epoch
    best_model/meta.json                        epoch/step/metric + sha256

Invariants
----------
1. A checkpoint and the LATEST.json that points at it are ONE commit -> a reader never
   follows a pointer to a partial upload.
2. The commit is verified (sha256) before anything older is pruned.
3. Pruning keeps `keep_last` verified checkpoints and never touches the one LATEST
   points to, so the only valid checkpoint can never be deleted.
4. A failed upload never stops training: the local copy stays, the failure is logged,
   and the next save (or finalize) uploads the newest state.
5. On resume, every candidate (remote pointer, remote history, local disk) is checked
   against its recorded sha256; the newest one that verifies wins.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import torch

from src.pipeline.env import log_event, utcnow
from src.pipeline.hub import HubError, sha256_file

STATE_FILE = "training_state.pt"


def _step_dir(step: int) -> str:
    return f"step-{step:07d}"


def _atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


class CheckpointManager:
    def __init__(self, store, exp_dir: str, local_root: str | Path, keep_last: int = 2,
                 background: bool = True, config_hash: str = ""):
        self.store = store                      # HubStore, or None for local-only tests
        self.exp_dir = exp_dir.rstrip("/")
        self.local_root = Path(local_root)
        self.local_ckpt = self.local_root / "checkpoints"
        self.local_best = self.local_root / "best_model"
        self.local_ckpt.mkdir(parents=True, exist_ok=True)
        self.keep_last = max(1, keep_last)
        self.background = background
        self.config_hash = config_hash
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None
        self._uploaded_steps: list[int] = []
        self._pending_step: Optional[int] = None
        self._best_dirty = False
        self._extra_files: dict = {}

    # ================================================================== save
    def save(self, state: dict, step: int, epoch: int, reason: str,
             extra_files: Optional[dict] = None) -> Path:
        """Write the checkpoint locally (atomic), then upload it (background)."""
        self.wait()                                     # one upload in flight at a time
        d = self.local_ckpt / _step_dir(step)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(state, d / STATE_FILE)
        meta = {"step": step, "epoch": epoch, "reason": reason, "created_at": utcnow(),
                "config_hash": self.config_hash,
                "files": {STATE_FILE: {"sha256": sha256_file(d / STATE_FILE),
                                       "bytes": (d / STATE_FILE).stat().st_size}}}
        _atomic_write_json(meta, d / "meta.json")
        log_event("checkpoint_saved", f"step {step} ({reason})", step=step, epoch=epoch,
                  mb=round(meta["files"][STATE_FILE]["bytes"] / 1e6, 1))
        self._pending_step = step
        self._extra_files = dict(extra_files or {})
        if self.store is None:
            self._prune_local()
            return d
        if self.background:
            self._thread = threading.Thread(target=self._upload_pending, daemon=True)
            self._thread.start()
        else:
            self._upload_pending()
        return d

    def mark_best(self, model_state: dict, meta: dict, config: dict):
        """Persist best-on-validation weights locally; uploaded with the next checkpoint."""
        from safetensors.torch import save_file
        self.wait()
        self.local_best.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().to("cpu").contiguous() for k, v in model_state.items()}
        tmp = self.local_best / "model.safetensors.tmp"
        save_file(tensors, str(tmp), metadata={"format": "pt"})
        os.replace(tmp, self.local_best / "model.safetensors")
        meta = dict(meta, saved_at=utcnow(), config_hash=self.config_hash,
                    sha256=sha256_file(self.local_best / "model.safetensors"))
        _atomic_write_json(meta, self.local_best / "meta.json")
        _atomic_write_json(config, self.local_best / "config.json")
        self._best_dirty = True

    def _best_adds(self) -> dict:
        if not self._best_dirty or not (self.local_best / "model.safetensors").exists():
            return {}
        return {f"{self.exp_dir}/best_model/{n}": self.local_best / n
                for n in ("model.safetensors", "meta.json", "config.json")}

    def _upload_pending(self):
        step = self._pending_step
        if step is None:
            return
        d = self.local_ckpt / _step_dir(step)
        rel = f"checkpoints/{_step_dir(step)}"
        history = [s for s in self._remote_history() if s != step] + [step]
        # Prune in the SAME commit (one commit per checkpoint; HF's commit budget is
        # ~128/h/repo). Safety: the previously verified checkpoint is always kept, so if
        # this commit landed but failed verification, a valid one still exists.
        keep_n = max(2, self.keep_last)
        drop = history[:-keep_n]
        history = history[-keep_n:]
        pointer = {"step": step, "dir": rel, "updated_at": utcnow(),
                   "config_hash": self.config_hash,
                   "history": [f"checkpoints/{_step_dir(s)}" for s in history]}
        adds = {f"{self.exp_dir}/{rel}/{STATE_FILE}": d / STATE_FILE,
                f"{self.exp_dir}/{rel}/meta.json": d / "meta.json",
                f"{self.exp_dir}/checkpoints/LATEST.json":
                    json.dumps(pointer, indent=2).encode("utf-8")}
        best = self._best_adds()
        adds.update(best)
        for rp, lp in self._extra_files.items():
            if Path(lp).exists():
                adds[f"{self.exp_dir}/{rp}"] = Path(lp)
        t0 = time.time()
        try:
            self.store.commit(adds, message=f"{self.exp_dir.split('/')[-1]}: checkpoint step {step}",
                              delete_folders=[f"{self.exp_dir}/checkpoints/{_step_dir(x)}" for x in drop])
        except Exception as e:  # noqa: BLE001 - never kill training over an upload
            self._last_error = f"{type(e).__name__}: {e}"
            log_event("checkpoint_upload_failed", f"step {step}: {self._last_error[:200]}",
                      logging.ERROR, step=step)
            return
        if best:
            self._best_dirty = False
        self._uploaded_steps = [int(h.split("-")[-1]) for h in pointer["history"]]
        self._pending_step = None if self._pending_step == step else self._pending_step
        self._last_error = None
        log_event("checkpoint_uploaded", f"step {step} verified on HF",
                  step=step, seconds=round(time.time() - t0, 1), with_best=bool(best))
        self._prune_local()

    def _remote_history(self) -> list[int]:
        if self._uploaded_steps:
            return list(self._uploaded_steps)
        try:
            p = self.store.read_json(f"{self.exp_dir}/checkpoints/LATEST.json")
        except Exception:  # noqa: BLE001
            p = None
        hist = (p or {}).get("history", [])
        return [int(h.split("-")[-1]) for h in hist]

    def _prune_local(self):
        """Keep the newest local checkpoint and anything not yet uploaded."""
        dirs = sorted(self.local_ckpt.glob("step-*"))
        uploaded = set(self._uploaded_steps) if self.store is not None else {
            int(d.name.split("-")[-1]) for d in dirs}
        for d in dirs[:-1]:
            s = int(d.name.split("-")[-1])
            if s in uploaded:
                shutil.rmtree(d, ignore_errors=True)

    def wait(self):
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._thread = None

    def flush(self, raise_on_error: bool = True):
        """Block until the newest local checkpoint is on HF (end of session / run)."""
        self.wait()
        if self.store is None:
            return
        if self._pending_step is not None:
            for attempt in range(3):
                self._upload_pending()
                if self._pending_step is None:
                    break
                time.sleep(15 * (attempt + 1))
        if self._best_dirty:
            try:
                self.store.commit(self._best_adds(),
                                  message=f"{self.exp_dir.split('/')[-1]}: best model")
                self._best_dirty = False
            except Exception as e:  # noqa: BLE001
                self._last_error = f"{type(e).__name__}: {e}"
        if raise_on_error and (self._pending_step is not None or self._best_dirty):
            raise HubError(f"could not upload the final state of {self.exp_dir}: "
                           f"{self._last_error}. Local copy kept at {self.local_root}.")

    # ================================================================== load
    def latest(self, map_location="cpu") -> Optional[tuple[dict, dict]]:
        """Newest checkpoint that passes sha256 verification (remote or local)."""
        cands: list[tuple[int, str, str]] = []           # (step, where, ref)
        for d in self.local_ckpt.glob("step-*"):
            if (d / "meta.json").exists() and (d / STATE_FILE).exists():
                cands.append((int(d.name.split("-")[-1]), "local", str(d)))
        pointer = None
        if self.store is not None:
            pointer = self.store.read_json(f"{self.exp_dir}/checkpoints/LATEST.json")
            if pointer:
                for h in pointer.get("history", []):
                    cands.append((int(h.split("-")[-1]), "remote", h))
        seen = set()
        for step, where, ref in sorted(cands, key=lambda c: (-c[0], c[1] != "local")):
            if (step, where) in seen:
                continue
            seen.add((step, where))
            try:
                d = Path(ref) if where == "local" else self._download(ref)
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                want = meta["files"][STATE_FILE]["sha256"]
                if sha256_file(d / STATE_FILE) != want:
                    raise HubError("sha256 mismatch")
                if self.config_hash and meta.get("config_hash") not in ("", None, self.config_hash):
                    raise HubError(f"checkpoint belongs to config {meta.get('config_hash')}, "
                                   f"not {self.config_hash}")
                state = torch.load(d / STATE_FILE, map_location=map_location, weights_only=False)
                if where == "remote":
                    self._uploaded_steps = [int(h.split("-")[-1]) for h in pointer["history"]]
                log_event("checkpoint_found", f"step {step} ({where}) verified",
                          step=step, source=where)
                return state, meta
            except Exception as e:  # noqa: BLE001 - try the next-newest candidate
                log_event("checkpoint_invalid", f"step {step} ({where}): {e}", logging.WARNING)
        return None

    def _download(self, rel_dir: str) -> Path:
        dst = self.local_ckpt / "_remote" / Path(rel_dir).name
        for n in ("meta.json", STATE_FILE):
            self.store.download(f"{self.exp_dir}/{rel_dir}/{n}", self.local_root / "_dl")
            src = self.local_root / "_dl" / self.exp_dir / rel_dir / n
            dst.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst / n)
        return dst

    def load_best(self, map_location="cpu") -> Optional[tuple[dict, dict]]:
        from safetensors.torch import load_file
        local = self.local_best / "model.safetensors"
        if not local.exists() and self.store is not None:
            try:
                for n in ("model.safetensors", "meta.json", "config.json"):
                    self.store.download(f"{self.exp_dir}/best_model/{n}", self.local_root / "_dl")
                    src = self.local_root / "_dl" / self.exp_dir / "best_model" / n
                    self.local_best.mkdir(parents=True, exist_ok=True)
                    os.replace(src, self.local_best / n)
            except Exception as e:  # noqa: BLE001
                log_event("best_model_missing", str(e)[:200], logging.WARNING)
                return None
        if not local.exists():
            return None
        meta = json.loads((self.local_best / "meta.json").read_text(encoding="utf-8"))
        if meta.get("sha256") and sha256_file(local) != meta["sha256"]:
            raise HubError(f"best_model of {self.exp_dir} fails sha256 verification")
        return load_file(str(local), device=str(map_location)), meta

    # ================================================================== completion
    def resume_state_delete_ops(self) -> list[str]:
        """Folders to delete once an experiment is complete (bundled into its results
        commit). Empty unless best_model is verified on the Hub."""
        if self.store is None:
            return []
        if not self.store.paths_info([f"{self.exp_dir}/best_model/model.safetensors"]):
            log_event("prune_skipped", "best_model not on HF; keeping resume checkpoints",
                      logging.WARNING)
            return []
        return [f"{self.exp_dir}/checkpoints"]

    def prune_resume_state(self):
        """After completion: drop resume checkpoints, keep best_model (verified first)."""
        if self.store is None:
            return
        info = self.store.paths_info([f"{self.exp_dir}/best_model/model.safetensors"])
        if not info:
            log_event("prune_skipped", "best_model not on HF; keeping resume checkpoints",
                      logging.WARNING)
            return
        try:
            self.store.commit(delete_folders=[f"{self.exp_dir}/checkpoints"],
                              message=f"{self.exp_dir.split('/')[-1]}: completed, drop resume state")
        except Exception as e:  # noqa: BLE001
            log_event("prune_failed", str(e)[:200], logging.WARNING)
        shutil.rmtree(self.local_ckpt, ignore_errors=True)
