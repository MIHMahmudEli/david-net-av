"""The claim protocol must hand each job to exactly one worker, even when ten of them
start at the same instant. These tests run the real logic against an in-memory stand-in
for the HF repo, so no network and no 12-hour feedback loop.
"""
import json
import threading
import time

import pytest

from src.utils.coordinator import Coordinator, Job


class FakeRepo:
    """Minimal stand-in for the HF repo: a path -> json dict store with a lock."""

    def __init__(self):
        self.files = {}
        self.lock = threading.Lock()

    def bind(self, coord: Coordinator):
        coord._put = lambda obj, path: self._put(path, obj)
        coord._get = lambda path: self._get(path)
        coord._ls = lambda prefix: self._ls(prefix)
        coord.withdraw = lambda job_id, reason="yield": self._rm(coord._claim_path(job_id))
        return coord

    def _put(self, path, obj):
        with self.lock:
            self.files[path] = json.loads(json.dumps(obj, default=str))

    def _get(self, path):
        with self.lock:
            return json.loads(json.dumps(self.files[path])) if path in self.files else None

    def _ls(self, prefix):
        with self.lock:
            return [p for p in self.files if p.startswith(prefix.rstrip("/") + "/")]

    def _rm(self, path):
        with self.lock:
            self.files.pop(path, None)


def make_jobs(repo, n=3):
    repo._put("coord/jobs.json", {"jobs": [
        {"job_id": f"stage1_seed{i}", "kind": "stage1", "run_id": f"run_{i}",
         "epochs": 10, "priority": i} for i in range(n)]})


def worker(repo, name, settle=0.05, kinds=None):
    c = Coordinator(worker_id=name, token="x", settle_s=settle)
    return repo.bind(c)


def test_jobs_load_in_priority_order():
    repo = FakeRepo(); make_jobs(repo, 3)
    jobs = worker(repo, "w1").load_jobs()
    assert [j.job_id for j in jobs] == ["stage1_seed0", "stage1_seed1", "stage1_seed2"]
    assert all(isinstance(j, Job) for j in jobs)


def test_unknown_fields_in_jobs_json_are_ignored():
    """The queue must survive being extended by a later version of publish_jobs.py."""
    repo = FakeRepo()
    repo._put("coord/jobs.json", {"jobs": [{"job_id": "a", "kind": "stage1", "run_id": "r",
                                            "epochs": 1, "future_field": "???"}]})
    assert worker(repo, "w1").load_jobs()[0].job_id == "a"


def test_ten_concurrent_workers_each_get_a_distinct_job():
    repo = FakeRepo(); make_jobs(repo, 3)
    got, errors = {}, []

    def run(name):
        try:
            job = worker(repo, name).acquire()
            if job:
                got[name] = job.job_id
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run, args=(f"w{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    claimed = list(got.values())
    assert len(claimed) == len(set(claimed)), f"a job was handed out twice: {got}"
    assert set(claimed) <= {"stage1_seed0", "stage1_seed1", "stage1_seed2"}
    # 3 jobs and 10 workers: every job is taken, the rest correctly find nothing.
    assert set(claimed) == {"stage1_seed0", "stage1_seed1", "stage1_seed2"}, got


def test_done_jobs_are_never_reissued():
    repo = FakeRepo(); make_jobs(repo, 2)
    w = worker(repo, "w1")
    w.complete("stage1_seed0", metrics={"auc": 0.9})
    assert worker(repo, "w2").acquire().job_id == "stage1_seed1"


def test_live_lease_blocks_and_expired_lease_is_reclaimed():
    repo = FakeRepo(); make_jobs(repo, 1)
    holder = worker(repo, "w1")
    assert holder.acquire().job_id == "stage1_seed0"

    assert worker(repo, "w2").acquire() is None, "a live lease must block other workers"

    # Simulate the holder's Kaggle session being reaped: heartbeat goes stale.
    path = holder._claim_path("stage1_seed0")
    stale = repo._get(path)
    stale["heartbeat"] = time.time() - (holder.lease_s + 60)
    repo._put(path, stale)

    assert worker(repo, "w3").acquire().job_id == "stage1_seed0", "expired lease must be reclaimable"


def test_heartbeat_keeps_a_lease_alive_without_resetting_claim_order():
    repo = FakeRepo(); make_jobs(repo, 1)
    w = worker(repo, "w1")
    job = w.acquire()
    claimed_at = repo._get(w._claim_path(job.job_id))["claimed_at"]
    time.sleep(0.05)
    w.heartbeat(job.job_id, claimed_at=claimed_at)
    after = repo._get(w._claim_path(job.job_id))
    assert after["claimed_at"] == claimed_at, "heartbeat must not move claimed_at (tiebreak key)"
    assert after["heartbeat"] > claimed_at


def test_dependencies_gate_release():
    repo = FakeRepo()
    repo._put("coord/jobs.json", {"jobs": [
        {"job_id": "train", "kind": "stage1", "run_id": "r", "epochs": 10, "priority": 1},
        {"job_id": "evaluate", "kind": "eval", "run_id": "r", "epochs": 0, "priority": 0,
         "depends_on": ["train"]}]})
    # `evaluate` sorts first by priority but must wait for `train`.
    assert worker(repo, "w1").acquire().job_id == "train"
    assert worker(repo, "w2").acquire() is None
    worker(repo, "w1").complete("train")
    assert worker(repo, "w3").acquire().job_id == "evaluate"


def test_kind_filter_lets_a_cpu_worker_skip_training_jobs():
    repo = FakeRepo()
    repo._put("coord/jobs.json", {"jobs": [
        {"job_id": "t", "kind": "stage1", "run_id": "r", "epochs": 10, "priority": 0},
        {"job_id": "e", "kind": "eval", "run_id": "r", "epochs": 0, "priority": 1}]})
    assert worker(repo, "w1").acquire(kinds=["eval"]).job_id == "e"


def test_empty_queue_returns_none():
    repo = FakeRepo()
    repo._put("coord/jobs.json", {"jobs": []})
    assert worker(repo, "w1").acquire() is None
