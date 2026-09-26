"""Experiment registry: stable IDs, no silent overwrites, cross-account claims.

registry/experiments.json on the HF repo:
    {"next_id": 8,
     "experiments": {"EXP_007": {"key": "full:davidnet-full:s42", "config_hash": "...",
                                 "status": "running", "claimed_by": "...", ...}}}

Rules
-----
* An experiment is identified by (mode, name, seed). Registering it again with the
  SAME config hash returns the existing ID (that is how a new Kaggle session resumes).
* Registering it with a DIFFERENT config hash is refused: results are never
  overwritten. Change the experiment name (e.g. add "-v2") to create a new one.
* A worker claims an experiment before training it; the claim carries a heartbeat.
  Another worker may take over only when the heartbeat is older than the lease, so a
  crashed session releases its experiment automatically.
"""
from __future__ import annotations

import os
import platform
import time
import uuid
from typing import Optional

from src.pipeline.env import log_event, utcnow

REGISTRY_PATH = "registry/experiments.json"

_WORKER = None


def worker_id() -> str:
    """Stable when DAVIDNET_WORKER is set (one name per Kaggle account): a restarted
    session of the same account then resumes its own claimed experiments at once,
    while other accounts still wait for the lease to expire. Without a name, each
    process is a distinct worker."""
    global _WORKER
    if _WORKER is None:
        named = os.environ.get("DAVIDNET_WORKER")
        _WORKER = named or f"{platform.node() or 'worker'}-{uuid.uuid4().hex[:6]}"
    return _WORKER


class ConfigConflict(RuntimeError):
    pass


class Registry:
    def __init__(self, store, namespace: str = ""):
        self.store = store
        self.ns = namespace            # "" for the paper, "smoke/" etc. for tests
        self.path = f"{namespace}{REGISTRY_PATH}"

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def key(mode: str, name: str, seed: int) -> str:
        return f"{mode}:{name}:s{seed}"

    def exp_dir(self, entry: dict) -> str:
        return f"{self.ns}experiments/{entry['exp_id']}_{entry['name']}_s{entry['seed']}"

    def load(self) -> dict:
        return self.store.read_json(self.path) or {"next_id": 1, "experiments": {}}

    def find(self, key: str, reg: Optional[dict] = None) -> Optional[dict]:
        reg = reg or self.load()
        for e in reg["experiments"].values():
            if e["key"] == key:
                return e
        return None

    # ------------------------------------------------------------------ register
    def register(self, *, mode: str, name: str, seed: int, config_hash: str,
                 meta: Optional[dict] = None) -> dict:
        key = self.key(mode, name, seed)
        result = {}

        def fn(reg):
            reg = reg or {"next_id": 1, "experiments": {}}
            existing = self.find(key, reg)
            if existing is not None:
                if existing["config_hash"] != config_hash:
                    raise ConfigConflict(
                        f"experiment {existing['exp_id']} ({key}) already exists with config "
                        f"hash {existing['config_hash']}; this run has {config_hash}. "
                        "Existing results are never overwritten -- give the experiment a new "
                        "name (e.g. append '-v2') if the change is intentional.")
                result["entry"] = existing
                return reg
            exp_id = f"EXP_{reg['next_id']:03d}"
            reg["next_id"] += 1
            entry = {"exp_id": exp_id, "key": key, "mode": mode, "name": name, "seed": seed,
                     "config_hash": config_hash, "status": "registered",
                     "created_at": utcnow(), "updated_at": utcnow(),
                     "claimed_by": None, "heartbeat": None, **(meta or {})}
            reg["experiments"][exp_id] = entry
            result["entry"] = entry
            result["new"] = True
            return reg

        self.store.update_json(self.path, fn, message=f"registry: register {key}")
        entry = result["entry"]
        entry["dir"] = self.exp_dir(entry)
        if result.get("new"):
            log_event("experiment_registered", f"{entry['exp_id']} = {key}",
                      config_hash=config_hash)
        return entry

    # ------------------------------------------------------------------ claims
    def claim(self, exp_id: str, lease_minutes: float = 45.0, force: bool = False) -> bool:
        me = worker_id()
        got = {"ok": False}

        def fn(reg):
            e = reg["experiments"][exp_id]
            if e["status"] == "completed":
                got["ok"] = False
                got["why"] = "already completed"
                return reg
            live = (e.get("claimed_by") not in (None, me) and e.get("heartbeat")
                    and time.time() - float(e["heartbeat"]) < lease_minutes * 60)
            if live and not force:
                got["why"] = f"held by {e['claimed_by']}"
                return reg
            e.update(status="running", claimed_by=me, heartbeat=time.time(),
                     updated_at=utcnow())
            got["ok"] = True
            return reg

        self.store.update_json(self.path, fn, message=f"registry: claim {exp_id}")
        if not got["ok"]:
            log_event("claim_skipped", f"{exp_id}: {got.get('why')}")
        return got["ok"]

    def heartbeat(self, exp_id: str, **fields):
        me = worker_id()

        def fn(reg):
            e = reg["experiments"][exp_id]
            if e.get("claimed_by") == me:
                e.update(heartbeat=time.time(), updated_at=utcnow(), **fields)
            return reg

        self.store.update_json(self.path, fn, message=f"registry: heartbeat {exp_id}")

    def set_status(self, exp_id: str, status: str, **fields):
        def fn(reg):
            e = reg["experiments"][exp_id]
            e.update(status=status, updated_at=utcnow(), **fields)
            if status in ("completed", "failed", "paused"):
                # released: a paused run is resumable by the next session immediately
                e["claimed_by"] = None
                e["heartbeat"] = None
            return reg

        self.store.update_json(self.path, fn, message=f"registry: {exp_id} -> {status}")
