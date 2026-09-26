"""Kill-and-resume must be invisible: a run interrupted mid-epoch and resumed from its
checkpoint ends with bitwise-identical weights to an uninterrupted run (CPU, fp32)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

RUNNER = textwrap.dedent('''
    import json, sys, torch
    from pathlib import Path
    sys.path.insert(0, {root!r})
    from src.pipeline.config import build_config
    from src.pipeline.synthetic import synthetic_records, synthetic_table, tiny_config
    from src.pipeline.features import FeatureDataset
    from src.pipeline.checkpoint import CheckpointManager
    from src.pipeline.models import build_davidnet
    from src.pipeline.trainer import (TrainJob, RunContext, train, stage1_objective,
                                      make_supervised_validator)
    from src.pipeline.env import seed_everything, setup_logging
    out = Path(sys.argv[1]); setup_logging(out / "logs")
    cfg = tiny_config(build_config({{"mode": "smoke"}}))
    recs = synthetic_records(96, seed=0)
    tr, va = recs[:64], recs[64:]
    table = synthetic_table(recs, d=32)
    seed_everything(7, deterministic=True)
    model = build_davidnet(cfg)
    exp = {{"exp_id": "EXP_T", "seed": 7, "name": "t", "dir": "x"}}
    job = TrainJob(exp=exp, cfg=cfg, stage="stage1", model=model,
                   train_ds=FeatureDataset(tr, table, True, 7),
                   val_ds=FeatureDataset(va, table, False, 7),
                   objective=stage1_objective,
                   validate=make_supervised_validator(stage1_objective, cfg),
                   sampler_keys=[r["quadrant"] for r in tr])
    hw = {{"device": "cpu", "precision": "fp32", "num_workers": 0, "pin_memory": False,
          "vram_gb": 0.0, "gpu_name": "cpu"}}
    ck = CheckpointManager(None, "x", out, keep_last=2, background=False)
    res = train(job, RunContext(hw=hw, ckpt=ck, local_dir=out))
    torch.save(model.state_dict(), out / "final.pt")
    print(json.dumps({{"status": res["status"], "steps": res["steps"]}}))
''')


def _run(out: Path, crash_after: int = 0):
    script = out.parent / "runner.py"
    script.write_text(RUNNER.format(root=str(ROOT)), encoding="utf-8")
    env = dict(os.environ, DAVIDNET_CRASH_AFTER_STEP=str(crash_after), PYTHONHASHSEED="0")
    return subprocess.run([sys.executable, str(script), str(out)], capture_output=True,
                          text=True, env=env, timeout=600)


def test_kill_and_resume_is_bitwise_identical(tmp_path):
    ref = tmp_path / "ref"
    r = _run(ref)
    assert r.returncode == 0, r.stderr[-2000:]
    ref_steps = json.loads(r.stdout.strip().splitlines()[-1])["steps"]

    run = tmp_path / "crash"
    r1 = _run(run, crash_after=3)                     # dies mid-epoch 1 (4 steps/epoch)
    assert r1.returncode == 137, (r1.returncode, r1.stderr[-2000:])
    assert not (run / "final.pt").exists()
    r2 = _run(run)                                    # a "new session" resumes
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert json.loads(r2.stdout.strip().splitlines()[-1])["steps"] == ref_steps
    log = (run / "logs" / "pipeline.jsonl").read_text(encoding="utf-8")
    assert '"event": "resumed"' in log

    a = torch.load(ref / "final.pt")
    b = torch.load(run / "final.pt")
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), f"weights differ after resume: {k}"
