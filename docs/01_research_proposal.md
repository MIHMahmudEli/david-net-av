# Research Proposal

## Fine-Grained, Per-Modality Detection and Localization of AI-Generated Video and Voice via a Disentangled Cross-Modal Transformer

**Working title of the model:** DAVID-Net — *Disentangled Audio-Visual Deepfake* detector.

**Target venues (Q1):** IEEE Transactions on Information Forensics and Security (TIFS, JCR Q1), *Information Fusion* (Q1), *Pattern Recognition* (Q1), IEEE Transactions on Multimedia (TMM, Q1), or *Expert Systems with Applications* (Q1). Primary target: **IEEE TIFS**. Systems/deployment angle also fits *IEEE TMM* or a demo/tools track.

---

## 1. Problem Statement and Motivation

Modern generative pipelines can now synthesize a talking-head video and a matching cloned voice independently. A single clip can therefore fall into one of **four authenticity quadrants**:

| Quadrant | Video | Audio | Typical generator example |
|----------|-------|-------|---------------------------|
| RVRA | Real | Real | Genuine recording |
| RVFA | Real | **Fake** | Voice cloning / TTS dubbed over real footage |
| FVRA | **Fake** | Real | Face swap / reenactment over real speech |
| FVFA | **Fake** | **Fake** | Fully synthetic avatar (e.g., lip-sync + TTS) |

Most published audio-visual (AV) deepfake detectors collapse this into a **single binary** "real vs. fake" decision. This is inadequate for three reasons:

1. **Attribution.** A forensic analyst, journalist, or moderation system needs to know *which* stream was manipulated, not merely that *something* is off. "The face is genuine but the voice was cloned" is a materially different finding than "the face was swapped."
2. **Asymmetric errors.** Binary detectors that assume "fake-if-any-modality-fake" cannot express the RVFA/FVRA distinction and tend to shortcut on whichever modality is easier, hurting the harder modality.
3. **Partial / temporal forgeries.** Real-world manipulations are increasingly *localized* — only a few seconds of audio or a short facial segment are altered (as in the AV-Deepfake1M and LAV-DF benchmarks). A clip-level binary label discards this.

**This thesis reframes AV deepfake detection as a fine-grained, per-modality, temporally-localized attribution problem** and proposes an architecture explicitly designed for it, together with a reproducible pipeline and a deployed public detector.

---

## 2. Research Questions

- **RQ1.** Can a single model produce *independent, well-calibrated* authenticity decisions for the visual and acoustic streams (four-quadrant attribution) while remaining competitive with specialised unimodal detectors on each stream?
- **RQ2.** Does explicit modeling of **audio-visual synchronization/consistency** improve detection of manipulations that are individually subtle but break cross-modal coherence (e.g., lip-sync mismatch, prosody-motion mismatch)?
- **RQ3.** Does **feature disentanglement** (separating modality-specific authenticity evidence from cross-modal consistency evidence) improve **generalization to unseen generators and datasets** compared with early/late fusion baselines?
- **RQ4.** Can the model **localize in time** which segments of each stream were manipulated, and does the localization objective act as useful auxiliary supervision for the clip-level decision?
- **RQ5.** Is the resulting detector **robust** to real-world degradations (video re-compression, audio codec/noise, resolution loss) and **fair** across demographic subgroups?
- **RQ6.** Can an embedding space pretrained on **synthetically constructed quadrants built from pristine data alone** (self-blended video, vocoder copy-synthesis audio, cross-clip mismatch — no deepfake generator outputs) generalize to *real* forgeries from unseen generators better than supervised training on generator outputs?

---

## 3. Gap Analysis (why this is publishable in a Q1 venue)

| Prior line of work | Representative methods | Limitation this thesis addresses |
|--------------------|------------------------|----------------------------------|
| Unimodal video | Xception, EfficientNet-B4, LipForensics, RealForensics | Single modality; blind to audio-only forgeries |
| Unimodal audio (anti-spoofing) | RawNet2, AASIST, Wav2Vec2/WavLM front-ends | Single modality; blind to face manipulation |
| AV binary fusion | AVFakeNet, MDS (Modality Dissonance), "Emotions Don't Lie", AVoiD-DF | Collapse to one label; no per-modality attribution |
| AV consistency / sync | SyncNet-style, AVAD (anomaly), voice-face homogeneity | Detects mismatch but does not attribute which stream is fake; weak on FVFA where *both* are fake yet internally consistent |
| Temporal localization | LAV-DF (BA-TFD/BA-TFD+), AV-Deepfake1M baselines | Localize but usually treat modalities jointly; limited per-modality attribution + weak cross-dataset generalization |

**Identified gap:** No widely-adopted method simultaneously delivers (a) *independent per-modality authenticity*, (b) *cross-modal consistency reasoning*, (c) *temporal localization*, and (d) *validated cross-generator/cross-dataset generalization*, in one reproducible, deployable system. DAVID-Net targets exactly this intersection.

---

## 4. Contributions

1. **Quadrant-Aware Contrastive Pretraining (QACP) — primary contribution.** A self-supervised stage that *constructs* all four authenticity quadrants from pristine data alone (self-blended face video, vocoder copy-synthesis audio, cross-clip real-audio mismatch) and structures three embedding spaces with a **factorized supervised-contrastive objective** — per-stream authenticity spaces invariant to the other stream, and a consistency space invariant to authenticity. The explicit *real-but-mismatched* class forces the model to learn that desynchronization is not forgery, fixing the classic sync-detector failure mode. No generator outputs are used, which is the mechanism behind cross-generator generalization (RQ6). Full recipe in `02_architecture.md` §8b.
2. **Problem reformulation.** AV deepfake detection is cast as a **multi-task problem**: two independent per-modality binary heads + a 4-class quadrant head + a per-modality temporal-localization head, replacing the single binary label. We formalize the label space and the evaluation protocol.
3. **DAVID-Net architecture.** A **disentangled dual-branch cross-modal transformer** with: (i) modality-specific authenticity encoders, (ii) an explicit **AV-synchronization consistency module** trained with a contrastive sync objective, and (iii) a **disentanglement constraint** that separates "is this stream internally authentic?" evidence from "do the streams agree?" evidence. This lets the model correctly handle FVFA clips that are internally consistent yet doubly fake.
4. **Generalization-first training regime.** QACP pretraining (Contribution 1) + self-supervised encoder initialization (VideoMAE / WavLM), cross-generator training splits, curriculum from full-clip to localized forgeries, and degradation augmentation. We report **leave-one-generator-out** and **cross-dataset** results as first-class metrics, not afterthoughts.
5. **Reproducible, deployed system.** A full open pipeline (data prep → training → evaluation → explainability), a **FastAPI inference service on Hugging Face Spaces**, and a **Next.js** front-end that returns per-modality verdicts with temporal heatmaps. This is a genuine systems/tooling contribution that strengthens acceptance at TMM/TIFS.
6. **Comprehensive empirical study.** Cross-dataset generalization, robustness to compression/noise, calibration, fairness across demographic subgroups, ablations of every module, and qualitative explainability (Grad-CAM for video, saliency for audio, sync-attention maps).

---

## 5. Proposed Method (overview — full detail in `02_architecture.md`)

DAVID-Net is trained in two stages — **Stage 0: QACP** (quadrant-aware contrastive pretraining on synthetically constructed quadrants from pristine data; `02_architecture.md` §8b), then **Stage 1: supervised multi-task finetuning**. The network has four components:

1. **Encoding.**
   - *Video branch:* face/lip-crop tubelets → VideoMAE / TimeSformer backbone → spatio-temporal token sequence `V`.
   - *Audio branch:* waveform → WavLM / Wav2Vec2 (or AASIST-style graph front-end) → acoustic token sequence `A`.
2. **Consistency module.** A SyncNet-inspired contrastive head aligns short `V`/`A` windows in time; its agreement score and cross-attention maps form the *cross-modal* evidence stream.
3. **Disentangled fusion.** A cross-modal transformer produces two representation spaces: **modality-specific authenticity** features (`z_v`, `z_a`) and a **cross-modal consistency** feature (`z_c`). A disentanglement loss (orthogonality / mutual-information penalty) discourages leakage between them.
4. **Multi-task heads.**
   - `H_v`: video real/fake (uses `z_v`, `z_c`).
   - `H_a`: audio real/fake (uses `z_a`, `z_c`).
   - `H_quad`: 4-class quadrant (RVRA/RVFA/FVRA/FVFA).
   - `H_loc`: per-modality frame-level manipulation probability (temporal localization).

**Loss:** `L = λ1·BCE_v + λ2·BCE_a + λ3·CE_quad + λ4·Loc + λ5·L_sync(contrastive) + λ6·L_disentangle`.

---

## 6. Datasets

Full catalog and preprocessing in `03_datasets.md`. Primary: **FakeAVCeleb** (native 4-quadrant labels), **AV-Deepfake1M / AV-Deepfake1M++** (large-scale temporal localization), **LAV-DF** (content-driven localized forgeries). Cross-dataset generalization test sets: **DFDC**, **KoDF**, **DeepfakeTIMIT**, plus audio-only **ASVspoof 2019/2021** and **In-the-Wild** for the audio branch. This multi-dataset design directly supports RQ3/RQ5 and is expected by Q1 reviewers.

---

## 7. Experimental Protocol (full detail in `04_experiments.md`)

- **Metrics:** clip-level AUC, EER, Accuracy, macro-F1; per-modality Accuracy/F1; 4-class confusion matrix; temporal localization AP@IoU (0.5/0.75) and AR; **cross-dataset AUC**; Expected Calibration Error (ECE); robustness curves vs. compression (CRF) and audio SNR; fairness gap across gender/skin-tone/language subgroups.
- **Baselines:** unimodal (Xception, EfficientNet-B4, LipForensics; RawNet2, AASIST, Wav2Vec2) and multimodal (MDS, "Emotions Don't Lie", AVFakeNet, AVoiD-DF, BA-TFD+ for localization). Re-implemented or run from official weights under an identical split.
- **Ablations:** remove sync module; remove disentanglement; early vs. late vs. proposed fusion; frozen vs. fine-tuned encoders; single-task vs. multi-task; with/without degradation augmentation.
- **Statistical rigor:** 3-seed runs, mean±std, bootstrap 95% CIs on AUC, DeLong test for AUC differences, McNemar for accuracy differences.

---

## 8. Expected Outcomes / Hypotheses

- H1: DAVID-Net matches or beats the best *unimodal* detector on each stream while additionally solving attribution (RQ1).
- H2: The sync + disentanglement modules give a measurable cross-dataset AUC gain (target: +3–8 AUC points vs. late-fusion baseline) (RQ2/RQ3).
- H3: The localization head improves clip-level decisions on partial forgeries (AV-Deepfake1M) via auxiliary supervision (RQ4).
- H4: The model degrades gracefully under compression/noise and shows small fairness gaps after balanced sampling (RQ5).
- H5: QACP pretraining beats no-pretraining and generic SSL pretraining on **LOGO and cross-dataset** AUC (target: the largest single-module gain in the ablation table), and a linear probe on the frozen QACP space already separates quadrants on unseen generators (RQ6).

Negative/partial results (e.g., disentanglement not helping FVFA) are still reported — Q1 reviewers value honest ablations.

---

## 9. Timeline (12 months)

| Phase | Months | Deliverable |
|-------|--------|-------------|
| P0 — Setup & lit review | 1 | `06_related_work.md`, environment, data access approvals |
| P1 — Data pipeline | 1–2 | Reproducible preprocessing for all datasets, EDA notebook |
| P2 — Baselines | 2–3 | Reproduced unimodal + multimodal baselines under common split |
| P3 — QACP + DAVID-Net v1 | 3–5 | Synthetic-quadrant builder; QACP pretraining (Stage 0); core model finetuned on FakeAVCeleb (Stage 1); internal test results |
| P4 — Generalization & localization | 5–7 | Cross-dataset + localization results; ablations |
| P5 — Robustness, calibration, fairness | 7–8 | Robustness/fairness study; explainability |
| P6 — Deployment | 8–9 | HF Spaces API + Next.js UI live |
| P7 — Writing & submission | 9–11 | Journal manuscript v1, internal review |
| P8 — Revision buffer | 11–12 | Rebuttal experiments, camera-ready |

---

## 10. Threats to Validity, Ethics, and Reproducibility

- **Dataset bias / shortcut learning:** mitigated by cross-dataset evaluation and degradation augmentation; report failure cases.
- **Demographic fairness:** explicit subgroup analysis; balanced sampling.
- **Dual-use / responsible disclosure:** the detector is defensive; we release detection weights and evaluation code but not any generation tooling. We include a model card and intended-use statement.
- **Reproducibility:** fixed seeds, pinned environment (`environment.yml`), config-driven runs, released checkpoints and split files, and a public inference demo.
- **Compute footprint:** target hardware is **NVIDIA DGX Spark** (128 GB unified memory, Blackwell); we use frozen pretrained encoders + cached features to stay bandwidth-efficient and report GPU-hours. Full strategy in `07_compute_and_hardware.md`; a "lite" configuration is provided for the public demo.

---

## 11. Why This Meets the Q1 Bar

- A **clear, novel problem reformulation** (per-modality attribution + localization) rather than an incremental architecture tweak.
- A **principled architecture** with an explicit hypothesis behind every module.
- **Generalization and robustness as first-class results**, which is exactly where reviewers reject incremental deepfake papers.
- A **complete, reproducible, deployed system** — reproducibility and real-world usability are strong differentiators.
- **Rigorous statistics and honest ablations.**

See `02_architecture.md`, `03_datasets.md`, `04_experiments.md`, and `05_deployment.md` for the operational detail.
