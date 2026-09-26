"""Generate kaggle/davidnet_q1_pipeline.ipynb (the Kaggle notebook) from source.

    python kaggle/build_notebook.py

Cells are deliberately thin: all logic lives in src/pipeline (tested, versioned), the
notebook holds the configuration and the research narrative.
"""
from pathlib import Path

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""# DAVID-Net-AV — Kaggle training pipeline (Q1 experiments)

Per-modality audio-visual deepfake detection (RVRA / RVFA / FVRA / FVFA) with QACP pretraining.
This notebook is the **only** entry point for training. It is resumable: every run continues where
the previous Kaggle session stopped, using the Hugging Face repo as the persistent source of truth.

**How to run**
1. *Settings*: Accelerator **GPU T4 x2** (or P100), Internet **On**, Persistence optional.
2. *Add Input → Datasets*: `aicontentdetections/fakeavceleb-v1-2`, `reubensuju/celeb-df-v2`,
   `pranay22077/dfdc-10`, `fahimaislam1812/deepfaketimit`, `anishsarkar22/asvpoof-2019-dataset-la`,
   `abdallamohamed312/in-the-wild-audio-deepfake`, `walimuhammadahmad/fakeaudio`.
   Nothing is downloaded — the mounts are read in place.
3. *Add-ons → Secrets*: `HF_TOKEN` (a **write** token of the account owning `CONFIG['project']['hf_repo']`),
   tick **Attached**. (A `kaggle kernels push` detaches secrets — re-attach after pushing.)
4. Set `MODE` below: `"recovery_test"` → `"smoke"` → `"full"` (in that order the first time).
5. **Save Version → Save & Run All (Commit)**. A batch commit survives closing the browser;
   an interactive session dies with the tab.

When a session ends (12 h limit, crash, or quota), just commit the notebook again — on this or
another account (set a different `worker_name`): it detects the latest verified checkpoint on
Hugging Face and continues."""))

cells.append(md("## 1. Research Configuration\nEvery tunable value lives here; the pipeline never hard-codes them. "
                "Values not listed fall back to `src/pipeline/config.py::DEFAULT_CONFIG` and are recorded, "
                "fully resolved, with every experiment (`configs/config.json`)."))
cells.append(code('''MODE = "recovery_test"   # "recovery_test" -> "smoke" -> "full"

CONFIG = {
    "mode": MODE,
    "seeds": [42, 123, 456],
    "project": {
        "hf_repo": "MoshinAli/davidnet-q1-experiments",        # experiments, checkpoints, results
        "hf_data_repo": "MoshinAli/davidnet-q1-data",          # frozen features + clip cache
        "code_repo": "https://github.com/MIHMahmudEli/david-net-av.git",
        "code_revision": "main",           # pin to a commit hash for the paper runs
        "work_dir": "/kaggle/working/davidnet",
        "scratch_dir": "/tmp/davidnet",
    },
    "data": {"split_protocol": "strict_identity", "split_seed": 42,
             "split_fractions": [0.65, 0.15, 0.20]},
    "features": {"video_model": "MCG-NJU/videomae-base",
                 "audio_model": "microsoft/wavlm-base-plus",
                 "video_spatial_grid": 4, "audio_tokens": 50},
    "model": {"d_model": 768, "n_heads": 8, "n_fusion_layers": 4, "dropout": 0.1},
    "train": {
        "qacp":   {"epochs": 40, "effective_batch": 256, "lr": 3e-4, "temperature": 0.1},
        "stage1": {"epochs": 30, "effective_batch": 64, "lr": 1e-4, "weight_decay": 0.05,
                   "selection_metric": "val/mean_auc", "early_stopping_patience": 6},
        "phase_b": {"epochs": 6, "effective_batch": 16, "lr": 3e-5, "lr_encoder": 1e-5,
                    "unfreeze_top_blocks_video": 2, "unfreeze_top_blocks_audio": 2},
    },
    "mixed_precision": True,
    "checkpoint": {"every_steps": 500, "every_minutes": 20, "keep_last": 2},
    "session": {"time_budget_hours": 11.5, "safety_minutes": 25,
                "worker_name": "kaggle-main"},   # UNIQUE per Kaggle account
    "evaluation": {"threshold_policy": "val_eer", "bootstrap": 1000},
    "plan": {"groups": None,          # e.g. ["baseline", "qacp", "proposed"] to restrict
             "only": None, "run_phase_b": True, "stop_on_error": True},
}'''))

cells.append(md("## 2. Environment & Hardware Verification\nFetch the pinned code revision, then report the hardware "
                "and library versions. Precision, micro-batch and workers are resolved automatically "
                "(T4/P100 → fp16 + GradScaler, Ampere+ → bf16; the *effective* batch never changes)."))
cells.append(code('''import os, sys, subprocess, json
REPO_DIR = "/kaggle/working/david-net-av"
rev = CONFIG["project"]["code_revision"]
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "-q", CONFIG["project"]["code_repo"], REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "-q", "origin"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", "-q", rev], check=True)
if rev in ("main", "master"):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "-q", "--ff-only"], check=True)
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
from src.pipeline.env import quiet_third_party, environment_report, autoconfig_hardware
quiet_third_party()
env = environment_report(REPO_DIR)
print(f"code     : {env['git']['commit']} (dirty={env['git']['dirty']})")
print(f"python   : {env['python']}  torch {env['torch']}  CUDA {env['cuda_version']}  cuDNN {env['cudnn_version']}")
for g in env["gpus"] or [{"name": "NO GPU", "total_memory_gb": 0, "capability": "-"}]:
    print(f"gpu      : {g['name']}  {g['total_memory_gb']} GB  sm_{g['capability']}")
print("packages :", ", ".join(f"{k} {v}" for k, v in env["packages"].items()))
print("hardware :", autoconfig_hardware(CONFIG | {"hardware": {"precision": "auto", "num_workers": "auto"}}))
if MODE != "recovery_test" and not env["gpus"]:
    print("WARNING: no GPU - enable the accelerator before a smoke/full run")'''))

cells.append(md("## 3. Dependency Setup\nOnly installs what the Kaggle image lacks (nothing is re-downloaded if present)."))
cells.append(code('''import importlib
need = {"safetensors": "safetensors", "huggingface_hub": "huggingface_hub>=0.30",
        "transformers": "transformers", "soundfile": "soundfile", "sklearn": "scikit-learn"}
missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
print("installed:", missing or "nothing (all present)")'''))

cells.append(md("## 4. Hugging Face Authentication\nThe token is read from Kaggle Secrets (or `HF_TOKEN` env), validated "
                "(`whoami`, namespace, write permission) and **never printed**. The experiment repo and the data "
                "repo are created private if they do not exist."))
cells.append(code('''from src.pipeline.driver import Session
S = Session(CONFIG, REPO_DIR)
who = S.connect()
print(f"authenticated as {who['user']} (token role: {who['role']})")
print("experiments:", S.store.web_url())
print("data       :", S.data_store.web_url())'''))

cells.append(md("### Recovery test (mode `recovery_test`)\nRuns before any long training: a small synthetic plan is trained "
                "(1) uninterrupted, (2) in a process that is **hard-killed** right after a checkpoint upload, then "
                "(3) the local disk is wiped (= a new Kaggle VM) and a new process resumes from Hugging Face. "
                "The resumed model must match the reference. Only after `PASS` switch `MODE` to `smoke`."))
cells.append(code('''if MODE == "recovery_test":
    from src.pipeline.recovery import run_recovery_test
    report = run_recovery_test(S.cfg, REPO_DIR, crash_exp="davidnet", crash_after_step=6)
    print(json.dumps({k: v for k, v in report.items() if k != "stderr"}, indent=1))
    if "stderr" in report:
        print(report["stderr"])
    S.publish_session_record({"recovery_test": report})
    assert report["verdict"] == "PASS", report["verdict"]
    print("\\nRECOVERY TEST PASSED - set MODE = 'smoke' and commit again.")'''))

cells.append(md("## 5. Dataset Access\nDatasets are read in place from the Kaggle mounts (no copies: copying FakeAVCeleb into "
                "`/kaggle/working` once pushed the container past its 32 GB memory cgroup)."))
cells.append(code('''from src.pipeline import manifests as M
if MODE != "recovery_test":
    for name, slug in M.KAGGLE_SLUGS.items():
        root = M.find_kaggle_root(name)
        print(f"{name:16s} {'OK  ' + str(root) if root else 'NOT MOUNTED  (Add Input -> ' + slug + ')'}")'''))

cells.append(md("## 6. Dataset Verification\nThe first session builds the manifests and splits and **freezes** them on the Hub "
                "(`data/SPLITS_SHA256.json`). Every later session downloads and sha256-verifies them instead of "
                "rebuilding, so all seeds, sessions and accounts use byte-identical splits.\n\n"
                "Primary protocol: **strict identity-disjoint** — every identity in a clip (FakeAVCeleb fakes also "
                "name *target* identities in their filenames) is confined to one split; the leakage audit below must "
                "show zero shared identities."))
cells.append(code('''import pandas as pd
idx = S.prepare_data()
print("frozen data:", idx["freeze_sha"][:16], "|", idx.get("protocol"))
audit = S.work / "data" / "split_audit.json"
if audit.exists():
    print(json.dumps(json.loads(audit.read_text()), indent=1))
summ = S.work / "data" / "split_summary.csv"
if summ.exists():
    display(pd.read_csv(summ))'''))

cells.append(md("## 7. Data Preprocessing\nOne decode per clip → identical face-centred crop for **every** corpus → frozen VideoMAE/WavLM "
                "token features (Phase A) + the Phase-B clip cache, written shard by shard to the data repo. "
                "Resumable: finished shards are skipped. A session that runs out of time pauses cleanly here."))
cells.append(code('''features_ready = S.prepare_features()
print("feature set:", S.fsid, "| complete:", features_ready)
if not features_ready:
    S.publish_session_record({"stage": "features (paused)"})
    print("Feature extraction paused at the time budget - commit the notebook again to continue.")'''))

cells.append(md("## 8. Model Initialization\nParameter counts of every architecture in the plan (sanity check of the configuration)."))
cells.append(code('''from src.pipeline.models import build_davidnet, build_baseline, count_parameters
rows = [{"model": "DAVID-Net (Phase A: fusion + heads)", **count_parameters(build_davidnet(S.cfg, "A"))}]
for b in ("video_probe", "audio_probe", "late_fusion"):
    rows.append({"model": b, **count_parameters(build_baseline(b, S.cfg["model"]["d_model"]))})
display(pd.DataFrame(rows))'''))

cells.append(md("## 9. Experiment Configuration\nBaseline → QACP → Proposed → Ablations → LOGO → Legacy → Final, every entry for every seed. "
                "Each experiment has its own ID, config hash, checkpoint and results; `init_from` makes the "
                "relationships explicit."))
cells.append(code('''from src.pipeline.plan import plan_table
plan, seeds = S.plan()
ptab = pd.DataFrame(plan_table(plan, seeds, S.cfg))
display(ptab)
print(len(ptab), "experiment runs planned")'''))

cells.append(md("## 10. Checkpoint Detection\nWhat already exists on the Hub (registry status and latest verified checkpoint per experiment)."))
cells.append(code('''reg = S.registry.load()
rows = []
for e in reg["experiments"].values():
    if e.get("mode") != S.mode:
        continue
    ptr = S.store.read_json(f"{S.registry.exp_dir(e)}/checkpoints/LATEST.json") if e["status"] in ("running", "paused", "failed") else None
    rows.append({"id": e["exp_id"], "name": e["name"], "seed": e["seed"], "status": e["status"],
                 "latest_checkpoint_step": (ptr or {}).get("step"), "claimed_by": e.get("claimed_by")})
display(pd.DataFrame(rows) if rows else "no experiments registered yet - a fresh start")'''))

cells.append(md("## 11. Resume Logic\nFor each experiment the trainer calls `CheckpointManager.latest()`: it lists the verified checkpoints "
                "(`LATEST.json` history + any local copy), downloads only the newest, checks its sha256 and config hash, "
                "and restores model, optimizer, scheduler, AMP scaler, RNG streams, epoch/step, the in-epoch sample "
                "position and the training history. A corrupt or incomplete checkpoint is skipped in favour of the "
                "previous verified one. No checkpoint → a new run. A completed experiment is never retrained, and a "
                "different configuration under an existing name is refused (never overwritten)."))

cells.append(md("## 12. Training\n## 13. Automatic Checkpoint Upload\n## 14. Validation\n## 15. Test Evaluation\n"
                "## 16. Metrics Generation\n## 17. Prediction Export\n"
                "`run_experiments()` executes, per experiment: training with periodic verified checkpoint uploads "
                f"(every `checkpoint.every_steps` steps / `every_minutes` minutes, background thread), per-epoch "
                "validation and best-model selection **on the validation split only**, then — once, on the best "
                "checkpoint — threshold fitting on validation, test evaluation, zero-shot cross-dataset evaluation, "
                "metric files (JSON + CSV) and prediction export (per sample: id, labels, probabilities for every "
                "head and quadrant, experiment id, model version). It stops cleanly before the session limit."))
cells.append(code('''if MODE != "recovery_test" and features_ready:
    run = S.run_experiments()
    print(json.dumps(run, indent=1))'''))
cells.append(code('''# Inspect the most recent completed experiment (all of this is also on the Hub)
exps = sorted((S.work / "experiments").glob("EXP_*")) if (S.work / "experiments").exists() else []
done = [e for e in exps if (e / "metrics" / "test_metrics.json").exists()]
if done:
    e = done[-1]
    print("experiment:", e.name)
    display(pd.read_csv(e / "metrics" / "training_history.csv").tail(5))
    display(pd.read_csv(e / "metrics" / "test_metrics.csv"))
    display(pd.read_csv(e / "predictions" / "test.csv").head())
    print(json.loads((e / "metrics" / "thresholds.json").read_text()))'''))

cells.append(md("## 18. Figure Generation\n## 19. Table Generation\nAggregates every completed experiment (mean ± std and Student-t CIs over seeds, "
                "DeLong tests vs. the proposed model with Holm correction) into manuscript tables (CSV + LaTeX) and "
                "figures (PNG 300 dpi + PDF), all regenerated from saved files."))
cells.append(code('''summary = S.build_reports() if MODE != "recovery_test" else {}
from IPython.display import Image
figdir = S.work / "reports" / "figures"
for f in sorted(figdir.glob("*.png"))[:4] if figdir.exists() else []:
    display(Image(filename=str(f), width=520))
tabdir = S.work / "reports" / "tables"
for t in ("main_results", "ablation_results", "cross_dataset_results"):
    if (tabdir / f"{t}.csv").exists():
        print(t); display(pd.read_csv(tabdir / f"{t}.csv"))'''))

cells.append(md("## 20. Experiment Summary"))
cells.append(code('''reg = S.registry.load()
display(pd.DataFrame([{k: e.get(k) for k in ("exp_id", "name", "seed", "group", "status",
                                             "test_clip_auc", "train_hours")}
                      for e in reg["experiments"].values() if e.get("mode") == S.mode]))'''))

cells.append(md("## 21. Hugging Face Artifact Upload\nExperiment artifacts are uploaded as they are produced; this flushes the "
                "repository README (status of every experiment)."))
cells.append(code('''readme = S.final_report()
print(S.store.web_url())'''))

cells.append(md("## 22. Reproducibility Metadata\n`sessions/<timestamp>/` receives this session's resolved config, environment, "
                "exact `pip freeze` and log; each experiment additionally stores its own "
                "`configs/{config,environment,experiment_metadata}.json` and `requirements.lock.txt` "
                "(git commit, HF revision, feature-set id, data freeze hash, model revisions)."))
cells.append(code('''rec = S.publish_session_record({"stage": "end of session"})
print("session record:", rec)'''))

cells.append(md("## 23. Final Experiment Report"))
cells.append(code('''from IPython.display import Markdown
display(Markdown(readme))
print(f"session time: {S.stopwatch.elapsed_h():.2f} h")'''))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
out = Path(__file__).with_name("davidnet_q1_pipeline.ipynb")
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
