# HANDOFF — Complete Continuation Guide

> **Audience:** any AI agent or team member picking up this thesis project cold.
> Read this file top-to-bottom before touching anything. It records the project's
> state, its conventions, every remaining step in order, and the pitfalls already
> hit once so you don't hit them twice.

---

## 1. What this project is

**Thesis:** *A Unified Forensic Framework for the Detection of Multimodal Synthetic
Media: Identifying Artifacts in Deepfakes and Generative AI Content* — AIUB BSc CSE
group thesis (4 members, Summer 2025–26), targeting a **Q1 journal** (primary:
IEEE TIFS; alternates: Information Fusion, Pattern Recognition, IEEE TMM).

**The scientific claim:** given a video clip, decide **independently** whether the
*video* stream and the *audio* stream are real or AI-generated (four quadrants:
RVRA/RVFA/FVRA/FVFA), **localize** manipulated segments in time, and **generalize
to unseen generators**. Two named contributions:

1. **QACP** (Quadrant-Aware Contrastive Pretraining) — the headline. Constructs
   all four quadrants *from pristine data only* (vocoder copy-synthesis audio,
   self-blended face video, cross-clip mismatch) and pretrains three embedding
   spaces with a **factorized SupCon** objective. The **MISMATCH class**
   (real+real but desynced) teaches "desync ≠ fake". No generator outputs used →
   the cross-generator generalization mechanism. `docs/02_architecture.md` §8b.
2. **DAVID-Net** — disentangled dual-branch cross-modal transformer: per-modality
   authenticity embeddings (z_v, z_a) kept orthogonal to a consistency embedding
   (z_c); sync module; 4 heads (H_v, H_a, H_quad, H_loc); missing-modality
   support via learnable null tokens + modality dropout (§7b).

**The paper lives or dies on the cross-dataset/LOGO columns of the ablation
table** — in-domain gains alone will be rejected as incremental. Never cut the
generalization experiments to save time.

## 2. Ground rules (learned the hard way — do not violate)

- **Git identity:** commit as the user. Repo-local git config already set to
  `113976745+MIHMahmudEli@users.noreply.github.com`. **Never add a
  `Co-Authored-By: Claude` trailer** — the user explicitly removed it.
- **Repo:** https://github.com/MIHMahmudEli/david-net-av (branch `main`). Push
  after each completed unit of work. Multi-line commit messages via
  `git commit -F <file>` (PowerShell here-strings passed to `-m` break).
- **Report format is the AIUB OBE 2.1 template** (`Thesis Report Template/`).
  Its structure is mandatory and already implemented in `report/`. Exactly 3
  numbered chapters (Introduction / Research Methodology / Results and Analysis)
  + fixed front/back matter. Do NOT restructure. Times 12pt, 1.5 spacing,
  justified, max 50–60 pages, no first/second person in body text.
- **Report builds with MiKTeX on this machine:**
  `cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main`.
- **Architecture/code/docs/figures move together.** Any model change updates:
  `src/`, `docs/02_architecture.md`, `report/figures/fig_davidnet.tex`, and the
  methodology text. (This drifted once; the user caught it.)
- **Every reported number = a committed config + a committed split file.**
  Subject-disjoint splits, seeds {42,43,44}, mean±std. No exceptions.
- **Train on `splits/<ds>/train.jsonl`, validate on `val.jsonl`, report on
  `test.jsonl`.** Run 1 trained on the full manifest (train+val+test) — any number from
  such a run is leakage and must be discarded.
- **Tests must pass before pushing:** `python -m pytest -q` from repo root with
  `PYTHONPATH` set to the repo root.
- **This machine (Windows dev box):** MiKTeX ✓, matplotlib ✓, torch-CPU ✓,
  **no ffmpeg, no CUDA, no face libs** — media tests auto-skip here.
  **Training machine: NVIDIA DGX Spark** (ARM64! see §6 pitfalls).

## 3. Repository map

```
HANDOFF.md            ← you are here
README.md             quickstart
docs/                 the research plan (source of truth for WHY)
  01_research_proposal.md   RQs, contributions, timeline, Q1 strategy
  02_architecture.md        DAVID-Net + QACP full spec (§7b missing-modality, §8b QACP)
  03_datasets.md            dataset catalog, unified manifest schema, split protocol
  04_experiments.md         metrics, baselines, tables/figures ↔ RQ map, ablation grid
  05_deployment.md          HF Space API + Next.js UI design + response contract
  06_related_work.md        literature map + positioning statement
  07_compute_and_hardware.md DGX Spark strategy (cached features, BF16, ARM64)
report/               LaTeX thesis (OBE 2.1) — main.tex + frontmatter/ + chapters/
  figures/fig_*.tex         TikZ architecture diagrams (hand-maintained)
  figures/generated/        auto-generated results figures (never hand-edit)
configs/              david_net.yaml (Stage 1) + qacp.yaml (Stage 0)
src/
  data/               datasets.py (manifest loader + CachedFeatureDataset),
                      preprocess.py (ffmpeg+face pipeline), extract_features.py,
                      synthetic_quadrants.py (QACP transforms)
  models/             david_net.py, video_encoder.py, audio_encoder.py
  training/           train.py (Stage 1), pretrain_qacp.py (Stage 0), losses.py
  eval/               metrics.py, evaluate.py, robustness.py, figures.py
  baselines/          models.py (registry), train_baseline.py
scripts/              build_manifest.py (FakeAVCeleb→manifest+splits),
                      run_experiments.py (ablation matrix), aggregate_results.py,
                      paper_artifacts.py (ALL manuscript metrics/tables/figures → HF paper/),
                      publish_model.py (public HF model repo for end users),
                      download.md (dataset access + the 5-command chain)
api/                  FastAPI service (app.py, inference.py, Dockerfile)
ui/                   Next.js plan (README.md — app not scaffolded yet)
tests/                pytest suite (17 passing, 1 ffmpeg-gated skip)
results/              (created by runs; results JSONs live here)
```

## 4. Current state (what is DONE and verified)

- Full research plan, Q1-positioned, incl. gap analysis and RQ1–RQ6.
- DAVID-Net + QACP + missing-modality support implemented; smoke-tested end to
  end on dummy data (model fwd/bwd, Stage 0 → Stage 1 warm start, eval).
- Cached-feature pipeline (extract → CachedFeatureDataset → identity encoders).
- Real preprocessing pipeline (ffmpeg decode, face track w/ EMA smoothing,
  backend chain insightface→mediapipe→haar→center, mouth crops, audio norm).
- FakeAVCeleb manifest converter + subject-disjoint + LOGO splits (tested).
- Baseline harness (registry; framecnn/speccnn runnable; xception/wavlm behind
  optional deps) sharing manifests+metrics with the main model.
- Robustness sweeps (blur/downscale/quantize/SNR) in tensor space.
- Figure generator (`src/eval/figures.py`) — ROC, reliability, confusion,
  robustness, ablation, localization example — with `--demo` mode; writes
  PDF+PNG to `report/figures/generated/`.
- Experiment orchestrator (`scripts/run_experiments.py`) + multi-seed
  aggregator with bootstrap CIs and LaTeX table body output.
- Thesis report: OBE 2.1 structure complete, 6 TikZ diagrams (DAVID-Net,
  QACP, sync module, pipeline, deployment, Gantt) + 3 pseudocode algorithms
  (QACP Stage-0, Stage-1 step, missing-modality inference), compiles clean
  (~53 pp — inside the journal-target 50–60 pp). Float parameters tuned in
  main.tex so figures/tables share pages with text (no half-empty float
  pages); keep new floats as `[htbp]`. **Full manuscript draft written with placeholder
  results:** every placeholder number is wrapped in `\dummy{…}` (renders
  blue); replace each with the measured value from `results/*.json`, then
  `\renewcommand{\dummy}[1]{#1}` in `main.tex` for camera-ready. The
  generated figures currently come from `--demo` mode — regenerate from real
  results. Red `[TODO: …]` markers remain only for facts nobody can invent
  (external examiner, defense date, editorial acknowledgement, team
  contribution split). Bibliography now ~62 entries incl. 2024–2025 SOTA
  (AVFF, LSDA, LAA-Net, ASVspoof 5, SpoofCeleb, UMMAFormer, DiMoDif,
  Deepfake-Eval-2024, EU AI Act…); before submission add 2–4 citations from
  the target journal's 2026 issues (see `docs/08_journal_shortlist.md`).
- API: /predict (silent-clip aware), /predict-audio, Dockerfile for HF Space.

**NOT yet done:** real datasets (access pending), any real training run,
baseline paper-grade runs, results chapters, HF Space deployment, Next.js app,
Gantt chart dates, camera-ready polish.

## 5. THE PLAYBOOK — run this when the dataset arrives

FakeAVCeleb lands in `data/fakeavceleb/`. Then, in order:

```bash
# 0. environment sanity (on the DGX Spark, inside an NGC pytorch container)
python -m pytest -q                              # all pass (ffmpeg test now runs)

# 1. manifest + splits  (commit the generated split files!)
python scripts/build_manifest.py --root data/fakeavceleb \
    --out src/data/manifests/fakeavceleb.jsonl \
    --splits-dir src/data/splits/fakeavceleb --seed 42

# 2. preprocess (one-time; resumable; check preprocess_meta.jsonl backend column)
python -m src.data.preprocess --manifest src/data/manifests/fakeavceleb.jsonl \
    --raw-root data/fakeavceleb --out data/shards/fakeavceleb

# 3. switch configs to real backbones + shards:
#    david_net.yaml + qacp.yaml:  video_backbone: videomae, audio_backbone: wavlm,
#    shard_root: data/shards/fakeavceleb,
#    train_manifest: src/data/splits/fakeavceleb/train.jsonl
#    (install: transformers, torchaudio; see requirements.txt)

# 4. cache SSL features once (Phase A — this is the expensive pass)
python -m src.data.extract_features --config configs/david_net.yaml \
    --manifest src/data/manifests/fakeavceleb.jsonl --out data/feats/fakeavceleb
#    then set feature_cache: data/feats/fakeavceleb in both configs

# 5. BENCHMARK ONE EPOCH FIRST; set epochs from measured time (docs/07 §3)

# 6. full matrix (QACP + train + eval, all ablations x 3 seeds):
python scripts/run_experiments.py --base configs/david_net.yaml \
    --test-manifest src/data/splits/fakeavceleb/test.jsonl \
    --seeds 42 43 44 --out results/

# 7. baselines under the identical split:
python -m src.baselines.train_baseline --baseline video-framecnn \
    --train-manifest src/data/splits/fakeavceleb/train.jsonl \
    --test-manifest src/data/splits/fakeavceleb/test.jsonl \
    --shard-root data/shards/fakeavceleb --epochs 10 \
    --out results/baseline_video-framecnn.json
#    repeat for audio-speccnn, video-xception (needs timm), audio-wavlm
#    (needs transformers). AASIST/RawNet2/LipForensics: run their official
#    repos on OUR test manifest; save {"method","modality","metrics","preds"}
#    JSONs into results/ — the figures/tables pick them up automatically.

# 8. LOGO + cross-dataset: re-run evaluate with --manifest set to each
#    logo_*_test.jsonl and (once converted) dfdc/kodf test manifests.
#    Write converters for those datasets modeled on scripts/build_manifest.py.

# 9. robustness + aggregation + figures:
python -m src.eval.robustness --config configs/david_net.yaml \
    --checkpoint <best.pt> --manifest src/data/splits/fakeavceleb/test.jsonl \
    --out results/robustness_david-net.json
python scripts/aggregate_results.py --results results/ --metric video.auc --boot
python -m src.eval.figures --results results/ --out report/figures/generated

# 10. fill the report (see §7), recompile, commit, push.
```

## 6. Known pitfalls (each of these already bit once)

- **ARM64 (DGX Spark):** use NGC containers, NOT bare pip. `mediapipe` is
  fragile on aarch64 — prefer insightface, or run face detection once on any
  x86 box and ship the crops. Video decode: DALI or the ffmpeg-pipe path in
  `preprocess.py` (which needs no OpenCV).
- **FakeAVCeleb filenames repeat** (`00000.mp4` in every folder). Metadata
  lookups must be full-path keyed. Already fixed in `build_manifest.py` —
  keep it that way for the DFDC/KoDF converters too.
- **Subject leakage kills the paper.** Always split by identity (converter does
  this). When adding datasets, extract identity or use provided subject IDs.
- **SupCon `-inf × 0 = NaN`:** masked log-probs must use `masked_fill`, not
  multiplication (fixed in `losses.py::supcon_loss`; don't regress it).
- **Class imbalance:** FakeAVCeleb is ~9:1 fake:real and quadrant-skewed —
  never report bare accuracy; AUC/macro-F1; keep focal loss + consider
  generator-balanced sampling when writing the real DataLoader sampler.
- **Whole-clip fakes** use the `WHOLE_CLIP=9999.0` sentinel segment →
  all-ones localization mask via `segments_to_mask` clipping.
- **PowerShell:** no `&&`; multi-line git commit messages via `-F` file; python
  inline `-c` strings with backslash paths break — use forward slashes.
- **THE run-1 disaster (2026-09-20): `root_dir` ≠ manifest root → training on
  `torch.randn`.** `build_manifest.py` was pointed at `<mount>/FakeAVCeleb_v1.2` (so
  `rel_path` is relative to that) while the notebook handed training `<mount>`. No file
  was found, and `AVDeepfakeDataset` silently substituted random tensors for EVERY
  sample. QACP + 3 Stage-1 seeds (≈20 GPU-hours) trained on noise: audio loss frozen at
  0.1733 (= focal loss at the class prior), val AUC 0.5, QACP terms at ln(B−1), then
  NaN loops. Fixes: the loader now RAISES when a configured root has no file, auto-
  resolves one directory level, `preflight_check()` decodes real samples before step 0,
  and the notebook asserts the first clip exists + decodes (Cell 7). **Never re-add a
  random-tensor fallback for real runs.**
- **Diagnosing "flat" losses:** a focal-BCE stuck at exactly 0.1733 (= 0.25·ln 2) means a
  constant logit at a 50/50 prior; a SupCon term stuck at ln(B−1) means collapsed or
  input-independent embeddings. Both mean "the branch sees no information" — check the
  data before touching the model.
- **FakeAVCeleb `meta_data.csv`** puts the directory in an *unnamed trailing column* and the
  bare filename in `path`; the old parser matched nothing, so every generator became
  `unknown`/`faceswap` (LOGO splits and the balanced sampler were meaningless). Fixed in
  `_load_meta_csv`; verify with `python scripts/build_manifest.py` — expect
  wav2lip=9602, fsgan=3964, fsgan-wav2lip=3553, faceswap-wav2lip≈2700, faceswap=730,
  rtvc=500, real=500.
- **Kaggle image (Sept 2026):** torch 2.10, torchaudio 2.10 + torchcodec 0.10,
  transformers 5.0, cv2 4.13, ffmpeg at /usr/bin. `torchaudio.load` on mp4 works; the
  decoder nevertheless uses ffmpeg pipes (single aligned A/V window, no seeking).
- **Cross-dataset manifests:** DFDC labels live in per-part `metadata.json`
  (REAL/FAKE) — not per-clip JSON; ASVspoof protocol label is the *last* token of the
  line (`... - - bonafide`). Both converters were reading the wrong field → single-class
  manifests → AUC = NaN. Audio-only corpora go through the model with `v_avail = 0`
  (null video token), never with random video.

## 7. Filling the thesis report (report/, OBE 2.1)

Every unfinished slot is marked `\todo{...}` (red). Search for them:
`grep -rn "todo{" report/`. The mapping:

| Report slot | Filled from |
|---|---|
| Ch.3 Table 3.1 (in-domain) | `results/full_seed*.json` via `aggregate_results.py` (LaTeX body is printed) + baseline JSONs |
| Ch.3 Table 3.2 (cross-dataset) | evaluate runs on DFDC/KoDF manifests |
| Ch.3 ablation table | `results/ablation.json` (fill `cross_dataset_auc`!) |
| Ch.3 figures | `\includegraphics{generated/results_roc}` etc. — files: results_roc, results_reliability, results_confusion, results_robustness, results_ablation, results_localization (all PDF in `report/figures/generated/`) |
| Abstract + Conclusion TODOs | 2–3 sentences of headline numbers once Table 3.1/3.2 exist |
| Planning WBS dates + Gantt | TikZ Gantt built (`report/figures/fig_gantt.tex`) with draft dates in blue; shift bars + WBS dates to the official semester calendar |
| Approval page | External examiner name + defense date + supervisor rank |
| Author Contributions | adjust drafted division of work with the team |
| Economic Decision TODOs | GPU-hours from W&B / power draw measured on the Spark |

**Where the numbers come from after a Kaggle run:** notebook Cell 13 runs the robustness
sweep and `scripts/paper_artifacts.py`, which pushes to HF `MoshinAli/david-net-av-backup/paper/`:
`metrics/summary.json` (per dataset × seed + mean/std + bootstrap CI), `metrics/tables.tex`
(Table 1/2/per-generator/efficiency bodies), `metrics/per_generator.json`,
`metrics/efficiency.json`, `metrics/eval/*` (slim reports + prediction dumps), and
`figures/results_{roc,roc_cross_dataset,reliability,confusion,localization,robustness,
training_curves,per_generator}.{pdf,png}`; copy `figures/` into `report/figures/generated/`.
Cell 14 (`scripts/publish_model.py`) publishes weights + config + model card + `api/` to the
PUBLIC repo `MoshinAli/david-net-av`; the FastAPI Space pulls `david_net.pt` from there.
Cells 13–17 produce the remaining experiments, each as a resumable HF `run_id`:
Cell 14 explainability (Fig. C, `src/eval/explain.py`: SmoothGrad saliency on frames +
spectrogram, sync curve, localization), Cell 15 baselines (`baseline_*.json`, same split/metrics),
Cell 16 ablations (`abl_<name>_v2_seed42`, Table 5 + `results_ablation`), Cell 17 LOGO
(`logo_<family>_v2_seed42`, Table 3; families = wav2lip incl. hybrids / fsgan / faceswap / rtvc,
subject-disjoint test identities). Fairness (Table 6) needs no extra compute — race/gender
ride along in the eval predictions and `paper_artifacts.py` writes `fairness.json`.

**GPU-hour budget (T4, ~50 min per Stage-1 epoch on the 15k train split — measure it on
your first epoch and rescale):** main run 3 seeds × 10 ep ≈ 25 h · ablations 4 × 4 ep ≈ 13 h ·
LOGO 4 × 4 ep ≈ 10 h · baselines ≈ 4 h · robustness/explain/artifacts ≈ 1 h → **≈ 55 GPU-h**,
i.e. two weekly Kaggle quotas on one account or one week on two accounts (HF resume is
account-agnostic — same `run_id`, same `HF_TOKEN` secret). Suggested session order:
(1) QACP + seed 42 → (2) seeds 123/456 → (3) Cell 12 eval + Cells 14/15/18/19 (first complete
paper package) → (4) ablations → (5) LOGO → re-run Cell 18 so tables/figures include them.
Ablations/LOGO use 4 epochs and one seed: state this in the paper ("reduced-budget protocol").

**Numbers discipline:** every number in a table traces to a JSON in `results/`
which traces to a config in `results/cfg_*.yaml`. If a number can't be traced,
delete it.

## 8. Remaining build tasks (priority order)

1. **DFDC / KoDF / LAV-DF / AV-Deepfake1M converters** — clone the pattern of
   `scripts/build_manifest.py`; each needs: labels→quadrant (most are binary:
   set both labels equal unless per-modality info exists), identity extraction,
   generator field, temporal segments where provided (AV-DF1M/LAV-DF).
2. **Generator-balanced batch sampler** (`torch.utils.data.WeightedRandomSampler`
   over generator × quadrant) — wire into train.py; ablate.
3. **Explainability outputs** — Grad-CAM on the video branch, spectrogram
   saliency, attach real PNGs to the API `explain=true` path and the report's
   qualitative figure.
4. **HF Space deployment** — export DAVID-Net-Lite weights → HF model repo →
   Docker Space from `api/Dockerfile` → smoke-test `/predict` with a real clip.
   See docs/05.
5. **Next.js UI** — scaffold per `ui/README.md` (upload → verdict cards →
   dual timeline → sync curve; server-side proxy to the Space; disclaimers).
6. **Gantt chart** — drafted in TikZ with placeholder dates; align to the
   official semester calendar.
7. **LoRA Phase-B finetune** (only the best config; docs/07 §2) — optional but
   strengthens final numbers.
8. **Paper manuscript** — the report chapters are written journal-style on
   purpose; the TIFS submission is a restructure of the same content
   (standard IEEE two-column, drop OBE-specific sections, expand related work).

## 9. Definition of done (acceptance checklist)

- [ ] Tables 3.1/3.2 + ablation filled with traced numbers (3 seeds, mean±std, CIs)
- [ ] Cross-dataset + LOGO results present (the Q1 make-or-break)
- [ ] All `\todo{}` markers gone from report/; compiles ≤ 60 pages
- [ ] Figures regenerated from final results/ (never stale)
- [ ] `pytest -q` green; every experiment config committed
- [ ] HF Space live + UI deployed; README links them
- [ ] Model card + intended-use statement published with weights
- [ ] Supervisor sign-off; plagiarism check; submission per AIUB calendar
```
