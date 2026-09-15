# DAVID-Net — Fine-Grained Detection of AI-Generated Video & Voice

A Q1-targeted thesis project: detect, **per modality**, whether a clip's **video** and its **audio**
are real or AI-generated, attribute it to one of four quadrants (RVRA / RVFA / FVRA / FVFA),
**localize** manipulated segments in time, and serve it through a public API + web UI.

> **Model:** *DAVID-Net* — Disentangled Audio-Visual Deepfake detector.
> **Target journal:** IEEE TIFS / Information Fusion / Pattern Recognition (all JCR Q1).

---

## Why this is novel
Most audio-visual deepfake detectors output a single "real vs. fake" label. DAVID-Net instead:
0. is pretrained with **QACP** (Quadrant-Aware Contrastive Pretraining) — all four authenticity
   quadrants are *constructed from pristine data alone* (self-blended video, vocoder copy-synthesis
   audio, cross-clip mismatch), so the embedding space learns manipulation semantics instead of
   generator fingerprints — the mechanism behind cross-generator generalization,
1. gives **independent decisions** for the visual and acoustic streams (attribution, not just detection),
2. reasons explicitly about **audio-visual synchronization/consistency**,
3. **disentangles** "is each stream authentic?" from "do the streams agree?" (so it still catches
   internally-consistent double-fakes — the FVFA case that fools sync-only detectors),
4. **localizes** manipulated intervals per modality, and
5. is validated under **cross-generator / cross-dataset** protocols and shipped as a live demo.

Full rationale and gap analysis: [`docs/01_research_proposal.md`](docs/01_research_proposal.md).

---

## Repository layout
```
docs/         Research plan, architecture, datasets, experiments, deployment, related work, compute/hardware
configs/      YAML run configs (source of truth for every reported number)
src/
  data/       Unified manifest schema, dataset loader, preprocessing
  models/     DAVID-Net, video/audio encoders, cross-modal fusion, sync module
  training/   Training loop + multi-task losses
  eval/       Metrics (AUC/EER/quadrant/ECE/localization AP) + evaluate.py
  utils/      Config loader, seeding
api/          FastAPI service + Dockerfile for a Hugging Face Space
ui/           Next.js front-end plan
scripts/      Dataset access links, preprocessing runners
tests/        Smoke tests
```

---

## Quickstart (no data / no downloads — verifies the pipeline runs)
```bash
pip install -r requirements.txt

# 1) model forward/backward smoke test
python -m src.models.david_net

# 2) Stage 0 — QACP pretraining dry run (synthetic quadrants from real clips)
python -m src.training.pretrain_qacp --config configs/qacp.yaml --dry-run

# 3) Stage 1 — supervised multi-task training dry run
#    (set init_from: runs/qacp/qacp_epochN.pt in the config to warm-start from QACP)
python -m src.training.train --config configs/david_net.yaml --dry-run

# 4) evaluate (dummy)
python -m src.eval.evaluate --config configs/david_net.yaml --manifest src/data/splits/train.jsonl

# 5) serve the API locally
uvicorn api.app:app --port 7860
```

To use real backbones, set `video_backbone: videomae` and `audio_backbone: wavlm` in the
config and install the extra deps (`transformers`, `torchaudio`, `opencv-python-headless`).

**On DGX Spark:** run inside an NVIDIA NGC PyTorch container (ARM64), use the cached-feature
training regime, and BF16. See [`docs/07_compute_and_hardware.md`](docs/07_compute_and_hardware.md).

---

## Continuing this project
**Read [`HANDOFF.md`](HANDOFF.md) first** — it is the complete continuation guide:
current state, the exact command playbook to run when datasets arrive, known
pitfalls, how results flow into the thesis report, and the remaining task list.

When the data lands, the whole experiment matrix is:
```bash
python scripts/run_experiments.py --base configs/david_net.yaml \
    --test-manifest src/data/splits/fakeavceleb/test.jsonl --seeds 42 43 44
python scripts/aggregate_results.py --results results/ --metric video.auc --boot
python -m src.eval.figures --results results/ --out report/figures/generated
```
Thesis figures regenerate automatically from the result JSONs and appear in the
report on the next `pdflatex` run.

## Kaggle Training (crash-proof)

Training runs on **Kaggle notebooks** (T4 GPU, 12h/session, ~30h/week quota).
Sessions can die without warning — all progress is backed up to HuggingFace.

### Setup (one-time)
1. Add `HF_TOKEN` to Kaggle Secrets (Settings -> Secrets -> Add)
2. Attach training datasets to `/kaggle/input/`
3. Attach this repo as a Kaggle dataset (or let the notebook clone it)

### Start a new run
```
python -m src.training.train --config configs/david_net_kaggle.yaml --run-id run_001
```

### Resume (automatic)
On a new Kaggle session, the same command automatically resumes from the last
checkpoint pushed to HF. No manual intervention needed.

### Switch accounts (quota exhausted)
The HF repo path is account-agnostic (`runs/<run_id>/`). When your primary
account's weekly quota runs out:
1. Switch to a secondary Kaggle account
2. Add the same `HF_TOKEN` to the new account's Secrets
3. Run the same `--run-id` command — it picks up where the last account left off

### Where things live on HF
```
david-net-av/backup/
  runs/<run_id>/
    checkpoints/epoch_0001.pt, epoch_0002.pt, ...
    best/best.pt
    logs/train_log.jsonl
    metrics/metrics.json
    figures/*.pdf
    state/resume_state.json
```

### Run the notebook
Open `scripts/train_kaggle.ipynb` in Kaggle, select T4 GPU, and hit Run All.
The notebook handles: install deps, clone repo, load secrets, extract datasets,
build manifests, train, and verify backup.

---

## Roadmap
See the 12-month plan in [`docs/01_research_proposal.md`](docs/01_research_proposal.md#9-timeline-12-months).
Dataset access requests should start **first** — approvals are slow (see [`scripts/download.md`](scripts/download.md)).

## Responsible use
This is a **defensive** detector. It outputs probabilities, not proof; it is not legal evidence.
No generation tooling is included. See the deployment guardrails in `docs/05_deployment.md`.
