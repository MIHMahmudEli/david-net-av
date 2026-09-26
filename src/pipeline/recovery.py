"""End-to-end Kaggle-session recovery test against the REAL Hugging Face repo.

    reference : a session runs the small synthetic plan uninterrupted
    crash     : a separate process runs the same plan and is hard-killed (exit 137) right
                after the checkpoint at step N of the target experiment was uploaded
    wipe      : its local disk (work + scratch) is deleted -> exactly a fresh Kaggle VM
    resume    : a new process runs the notebook logic again: it must find the experiment
                on HF, download + verify the checkpoint, restore model/optimizer/scheduler/
                scaler/RNG/loop state, finish, evaluate and publish
    compare   : resumed vs reference best-model weights and test metrics

On CPU the comparison is bitwise; on GPU (non-deterministic kernels) it is a tolerance
check on the test metrics. Usage (the notebook does this in recovery_test mode):
    report = run_recovery_test(CONFIG, repo_dir)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.pipeline.config import deep_merge
from src.pipeline.env import log_event


def _session_main(cfg_path: str):
    cfg = json.loads(Path(cfg_path).read_text())
    repo = cfg.pop("_repo_dir")
    sys.path.insert(0, repo)
    from src.pipeline.driver import Session
    s = Session(cfg, repo)
    s.connect()
    s.prepare_data()
    s.prepare_features()
    res = s.run_experiments()
    s.build_reports()
    print("RESULT " + json.dumps(res))


def _spawn(cfg: dict, repo_dir: str, tag: str, env_extra: dict) -> subprocess.CompletedProcess:
    c = deep_merge(cfg, {"mode": "recovery_test", "recovery": {"tag": tag}})
    base = Path(cfg["project"]["scratch_dir"]) / "recovery_runs" / tag
    c["project"] = dict(c["project"], work_dir=str(base / "work"), scratch_dir=str(base / "scratch"))
    c["_repo_dir"] = str(repo_dir)
    base.mkdir(parents=True, exist_ok=True)
    cfg_path = base / "config.json"
    cfg_path.write_text(json.dumps(c))
    env = dict(os.environ, PYTHONHASHSEED="0", **env_extra)
    env.setdefault("DAVIDNET_WORKER", f"recovery-{tag}")
    return subprocess.run([sys.executable, "-m", "src.pipeline.recovery", str(cfg_path)],
                          cwd=str(repo_dir), env=env, capture_output=True, text=True, timeout=7200)


def run_recovery_test(cfg: dict, repo_dir: str, crash_exp: str = "davidnet",
                      crash_after_step: int = 5) -> dict:
    from src.pipeline.hub import HubStore, resolve_hf_token
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ref_tag, run_tag = f"{stamp}_ref", f"{stamp}_crash"
    report = {"tags": [ref_tag, run_tag], "crash_exp": crash_exp,
              "crash_after_step": crash_after_step}

    log_event("recovery_test", f"1/4 reference session ({ref_tag})")
    r = _spawn(cfg, repo_dir, ref_tag, {})
    report["reference_exit"] = r.returncode
    if r.returncode != 0:
        report["verdict"] = "FAIL (reference run failed)"
        report["stderr"] = r.stderr[-3000:]
        return report

    log_event("recovery_test", f"2/4 crash session: kill after step {crash_after_step} of {crash_exp}")
    r = _spawn(cfg, repo_dir, run_tag, {"DAVIDNET_CRASH_AFTER_STEP": str(crash_after_step),
                                        "DAVIDNET_CRASH_EXP": crash_exp})
    report["crash_exit"] = r.returncode
    if r.returncode != 137:
        report["verdict"] = f"FAIL (expected exit 137 from the killed session, got {r.returncode})"
        report["stderr"] = r.stderr[-3000:]
        return report

    base = Path(cfg["project"]["scratch_dir"]) / "recovery_runs" / run_tag
    shutil.rmtree(base / "work", ignore_errors=True)
    shutil.rmtree(base / "scratch", ignore_errors=True)
    log_event("recovery_test", "3/4 local disk wiped (fresh VM); resume session")
    r = _spawn(cfg, repo_dir, run_tag, {})
    report["resume_exit"] = r.returncode
    out = r.stdout + r.stderr
    report["resumed_event_seen"] = "[resumed]" in out
    if r.returncode != 0:
        report["verdict"] = "FAIL (resume session failed)"
        report["stderr"] = r.stderr[-3000:]
        return report

    log_event("recovery_test", "4/4 compare resumed vs reference")
    p = cfg["project"]
    store = HubStore(p["hf_repo"], resolve_hf_token(), p["hf_repo_type"], p["hf_private"])
    cmp = {}
    for tag, key in ((ref_tag, "ref"), (run_tag, "run")):
        reg = store.read_json(f"recovery_test/{tag}/registry/experiments.json")
        for e in reg["experiments"].values():
            d = f"recovery_test/{tag}/experiments/{e['exp_id']}_{e['name']}_s{e['seed']}"
            best = store.read_json(f"{d}/best_model/meta.json") or {}
            test = store.read_json(f"{d}/metrics/test_metrics.json") or {}
            cmp.setdefault(e["name"], {})[key] = {
                "status": e["status"], "best_sha256": best.get("sha256"),
                "test_clip_auc": (test.get("clip") or {}).get("auc")}
    report["comparison"] = cmp
    target = cmp.get(crash_exp, {})
    same_weights = target.get("ref", {}).get("best_sha256") == target.get("run", {}).get("best_sha256")
    auc_r = target.get("ref", {}).get("test_clip_auc")
    auc_c = target.get("run", {}).get("test_clip_auc")
    close = auc_r is not None and auc_c is not None and abs(auc_r - auc_c) <= 0.02
    all_done = all(v.get("run", {}).get("status") == "completed" for v in cmp.values())
    report["bitwise_identical_weights"] = same_weights
    report["verdict"] = ("PASS" if report["resumed_event_seen"] and all_done and (same_weights or close)
                         else "FAIL (see comparison)")
    log_event("recovery_test", f"verdict: {report['verdict']}")
    return report


if __name__ == "__main__":
    _session_main(sys.argv[1])
