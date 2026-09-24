"""Multi-worker job coordination over a shared HuggingFace repo.

Why: Stage 1 plus the manuscript experiments are ~30 GPU-hours, but a Kaggle batch session
caps at 12 h. Several Kaggle accounts can therefore work the same queue in parallel, each
claiming a job, training it, and pushing to the SAME HF repo, so a job interrupted on one
account resumes on the next from `runs/<run_id>/state/resume_state.json`.

Repo layout (alongside the existing `runs/<run_id>/...` tree):

    coord/jobs.json                        the queue, authored once by scripts/publish_jobs.py
    coord/claims/<job_id>.<worker>.json    one lease per worker per job
    coord/done/<job_id>.json               completion receipt (metrics digest + provenance)

Mutual exclusion without transactions
-------------------------------------
HF has no compare-and-swap we can lean on here: with ~10 workers pushing multi-GB
checkpoints, `parent_commit` optimistic locking fails constantly for unrelated reasons.
Instead every worker writes its OWN claim path (so no write ever conflicts), waits out a
settle window until concurrent claims are visible, then applies a deterministic tiebreak:
earliest `claimed_at`, worker_id breaking exact ties. Losers withdraw. Worst case is two
workers briefly starting the same job, and the loser yields inside the settle window, long
before either has written a checkpoint.

A claim is a LEASE: it must be refreshed (`heartbeat`) or another worker reclaims the job
after `lease_minutes`. That is what makes a killed Kaggle session self-healing.
"""
from __future__ import annotations

import json
import os
import random
import socket
import time
from dataclasses import dataclass, field
from typing import Optional

# The control plane (jobs, claims, heartbeats) commits far more often than the
# checkpoints do. HF caps commits per repo per hour, so it gets its own repo and its
# own budget; `runs/` artifacts stay in the model repo.
REPO_ID = "MoshinAli/david-net-av-coord"
REPO_TYPE = "dataset"
SETTLE_S = 45          # longer than one HF list beat, so other claims become visible
LEASE_MINUTES = 60     # a session that stops heartbeating this long is presumed dead


def worker_identity() -> str:
    """Stable, human-readable id for this Kaggle account/session.

    Kaggle exposes the owner under different env vars depending on image, so fall through
    the options; always append a session nonce, because two sessions on the SAME account
    must not be able to impersonate each other's lease.
    """
    # Kaggle does NOT set the owner env vars in a batch session -- a real run identified
    # itself as "5930701ab423", a container hostname, which tells you nothing about which
    # of ten accounts holds a lease. DAVIDNET_WORKER lets each account name itself.
    who = (os.environ.get("DAVIDNET_WORKER") or os.environ.get("KAGGLE_USER_NAME")
           or os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_KERNEL_OWNER")
           or socket.gethostname() or "worker")
    nonce = os.environ.get("DAVIDNET_SESSION_ID") or time.strftime("%m%d%H%M%S")
    return f"{who}-{nonce}"


@dataclass
class Job:
    job_id: str
    kind: str                       # stage1 | baseline | ablation | logo | lite | eval
    run_id: str
    epochs: int
    priority: int = 100             # lower runs first
    config: dict = field(default_factory=dict)
    depends_on: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class Coordinator:
    def __init__(self, worker_id: Optional[str] = None, repo_id: str = REPO_ID,
                 token: Optional[str] = None, lease_minutes: int = LEASE_MINUTES,
                 settle_s: int = SETTLE_S):
        self.repo_id = repo_id
        self.worker_id = worker_id or worker_identity()
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("hf")
        self.lease_s = lease_minutes * 60
        self.settle_s = settle_s
        self._api = None

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api

    # ---------------------------------------------------------------- repo primitives
    def _put(self, obj: dict, path: str):
        payload = json.dumps(obj, indent=1, default=str).encode()
        self.api.upload_file(path_or_fileobj=payload, path_in_repo=path, repo_id=self.repo_id,
                             repo_type=REPO_TYPE, commit_message=f"coord: {path}")

    def _get(self, path: str) -> Optional[dict]:
        # force_download: coordination reads must never come from the local HF cache, or a
        # worker keeps seeing the claim state it saw the first time it looked.
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(self.repo_id, path, repo_type=REPO_TYPE,
                                    token=self.token, force_download=True)
            return json.load(open(local, encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a missing file is a normal state, not an error
            return None

    def _ls(self, prefix: str) -> list:
        try:
            tree = self.api.list_repo_tree(self.repo_id, path_in_repo=prefix,
                                           repo_type=REPO_TYPE, recursive=False)
            return [f.path for f in tree if getattr(f, "size", None) is not None]
        except Exception:  # noqa: BLE001
            return []

    # ---------------------------------------------------------------- queue state
    def load_jobs(self) -> list:
        doc = self._get("coord/jobs.json") or {}
        jobs = [Job.from_dict(j) for j in doc.get("jobs", [])]
        return sorted(jobs, key=lambda j: (j.priority, j.job_id))

    def _live_claims(self) -> dict:
        """job_id -> live (non-expired) claims."""
        now, out = time.time(), {}
        for path in self._ls("coord/claims"):
            c = self._get(path)
            if not c:
                continue
            if now - float(c.get("heartbeat", 0)) > self.lease_s:
                continue                      # expired lease: the job is reclaimable
            out.setdefault(c["job_id"], []).append(c)
        return out

    def _done_ids(self) -> set:
        return {p.split("/")[-1][:-5] for p in self._ls("coord/done") if p.endswith(".json")}

    def status(self) -> list:
        """Queue snapshot for humans: what every worker is doing right now."""
        jobs, claims, done = self.load_jobs(), self._live_claims(), self._done_ids()
        rows = []
        for j in jobs:
            if j.job_id in done:
                state, who = "done", ""
            elif j.job_id in claims:
                state, who = "running", self._winner(claims[j.job_id])["worker_id"]
            elif not all(d in done for d in j.depends_on):
                # `acquire` already refuses these; say so, or the queue looks stuck.
                state, who = "blocked", "waiting on " + ",".join(
                    d for d in j.depends_on if d not in done)
            else:
                state, who = "free", ""
            rows.append({"job_id": j.job_id, "run_id": j.run_id, "state": state, "worker": who})
        return rows

    # ---------------------------------------------------------------- claiming
    @staticmethod
    def _winner(claims: list) -> dict:
        """Deterministic: earliest claim wins, worker_id breaks exact ties."""
        return sorted(claims, key=lambda c: (float(c["claimed_at"]), c["worker_id"]))[0]

    def _claim_path(self, job_id: str) -> str:
        return f"coord/claims/{job_id}.{self.worker_id}.json"

    def _write_claim(self, job_id: str, claimed_at: Optional[float] = None) -> dict:
        now = time.time()
        claim = {"job_id": job_id, "worker_id": self.worker_id,
                 "claimed_at": claimed_at if claimed_at is not None else now,
                 "heartbeat": now,
                 "run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "?")}
        self._put(claim, self._claim_path(job_id))
        return claim

    def withdraw(self, job_id: str, reason: str = "yield"):
        try:
            self.api.delete_file(self._claim_path(job_id), repo_id=self.repo_id, repo_type=REPO_TYPE,
                                 commit_message=f"coord: withdraw {job_id} ({reason})")
        except Exception:  # noqa: BLE001 - already gone is fine
            pass

    def heartbeat(self, job_id: str, claimed_at: Optional[float] = None):
        """Refresh the lease. A few hundred bytes; call every few minutes while training."""
        self._write_claim(job_id, claimed_at=claimed_at)

    def complete(self, job_id: str, metrics: Optional[dict] = None,
                 provenance: Optional[dict] = None):
        self._put({"job_id": job_id, "worker_id": self.worker_id, "finished_at": time.time(),
                   "metrics": metrics or {}, "provenance": provenance or {}},
                  f"coord/done/{job_id}.json")
        self.withdraw(job_id, reason="complete")

    def acquire(self, kinds: Optional[list] = None) -> Optional[Job]:
        """Claim the highest-priority runnable job, or None once the queue is drained.

        Runnable = not done, dependencies done, and neither claimed nor holding a live
        lease. After claiming we wait out the settle window and re-check: if another worker
        got there first we withdraw and fall through to the next candidate.
        """
        for _ in range(20):                    # bounded: the queue is tens of jobs
            done, claims = self._done_ids(), self._live_claims()
            candidates = [j for j in self.load_jobs()
                          if j.job_id not in done
                          and j.job_id not in claims
                          and (not kinds or j.kind in kinds)
                          and all(d in done for d in j.depends_on)]
            if not candidates:
                return None
            job = candidates[0]
            self._write_claim(job.job_id)
            # Jitter so ten workers started by the same schedule do not march in lockstep
            # and collide on every job in queue order.
            time.sleep(self.settle_s + random.uniform(0, 15))
            live = self._live_claims().get(job.job_id, [])
            if live and self._winner(live)["worker_id"] == self.worker_id:
                print(f"[coord] {self.worker_id} acquired {job.job_id} (run_id={job.run_id})")
                return job
            self.withdraw(job.job_id, reason="lost-race")
            print(f"[coord] {self.worker_id} lost {job.job_id}, trying next")
        return None
