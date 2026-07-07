# Experimental Protocol

This is the empirical contract for the paper. Every table/figure planned below maps to a research question.

## 1. Metrics

**Clip-level detection**
- AUC (primary), EER, Accuracy, macro-F1.
- Per-modality: video-AUC / video-F1, audio-AUC / audio-F1.
- 4-class quadrant: confusion matrix, per-quadrant recall, macro-F1.

**Temporal localization** (AV-Deepfake1M, LAV-DF)
- AP@IoU={0.5, 0.75, 0.95}, AR@{10,20,50}, mAP. Per-modality.

**Generalization**
- Cross-dataset AUC (train FakeAVCeleb+AVDeepfake1M → test DFDC/KoDF/DF-TIMIT).
- LOGO AUC (leave-one-generator-out), averaged over held-out generators.

**Calibration & reliability**
- Expected Calibration Error (ECE), reliability diagrams, temperature scaling.

**Robustness**
- AUC vs. video compression (H.264 CRF 23/30/40), vs. audio SNR (20/10/0 dB, MUSAN), vs. resolution downscale, vs. re-encoding. Report degradation curves.

**Fairness**
- AUC / EER gap across gender, apparent skin-tone, and language subgroups.

**Efficiency**
- Params, FLOPs, latency (T4 GPU + CPU), GPU-hours to train.

## 2. Baselines

| Type | Methods |
|------|---------|
| Video unimodal | Xception, EfficientNet-B4, LipForensics, RealForensics |
| Audio unimodal | RawNet2, AASIST, Wav2Vec2/WavLM-linear |
| AV binary fusion | MDS (Modality Dissonance), "Emotions Don't Lie", AVFakeNet, AVoiD-DF |
| AV localization | BA-TFD, BA-TFD+ (LAV-DF baselines), AV-Deepfake1M baseline |

Rules: identical splits, identical preprocessing, official weights where available else faithful re-implementation (documented). Report both "as-published" numbers and "our-split" numbers.

## 3. Planned tables/figures → RQ map

| Artifact | Content | RQ |
|----------|---------|----|
| Table 1 | In-domain per-modality + quadrant results vs. baselines | RQ1 |
| Table 2 | Cross-dataset AUC (DFDC/KoDF/DF-TIMIT) | RQ3 |
| Table 3 | LOGO generalization | RQ3 |
| Table 4 | Temporal localization AP@IoU | RQ4 |
| Table 5 | **Ablations** (see §4) | RQ2/RQ3/RQ4/RQ6 |
| Table 8 | **QACP study**: no-pretrain vs. generic SSL vs. QACP; linear probe on frozen QACP space vs. unseen generators; per-pseudo-class contribution | RQ6 |
| Fig. A | Robustness curves (compression/noise) | RQ5 |
| Fig. B | Reliability diagram + ECE | RQ5 |
| Table 6 | Fairness subgroup gaps | RQ5 |
| Fig. C | Qualitative: Grad-CAM + sync curve + localization timeline | RQ2/RQ4 |
| Table 7 | Efficiency (params/FLOPs/latency) | deploy |

## 4. Ablation grid (Table 5)

- (a) Full DAVID-Net.
- (b) − sync module.
- (c) − disentanglement loss.
- (d) early fusion / (e) late fusion instead of proposed disentangled fusion.
- (f) single-task (binary only) vs. multi-task.
- (g) − localization head.
- (h) frozen vs. fine-tuned encoders.
- (i) − degradation augmentation.
- (j) H_quad direct vs. composed from H_v·H_a.
- (k) − QACP pretraining (train Stage 1 from scratch).
- (l) QACP variants: − MISMATCH class; − copy-synthesis (audio); − self-blending (video); factorized SupCon → single joint SupCon.

Each ablation reports in-domain AUC + cross-dataset AUC so we can attribute *generalization* gains to specific modules. (k)/(l) are the paper's key evidence: the QACP claim stands or falls on the cross-dataset column.

## 5. Statistical rigor

- 3 seeds; report mean ± std.
- 95% bootstrap CIs on AUC.
- DeLong test for AUC differences; McNemar for accuracy; Holm-Bonferroni correction across the ablation family.
- Pre-register the primary metric (cross-dataset AUC) to avoid cherry-picking.

## 6. Compute plan (NVIDIA DGX Spark)

Target hardware: **NVIDIA DGX Spark** (GB10 Grace-Blackwell, 128 GB unified LPDDR5x, Blackwell GPU). Full strategy in [`07_compute_and_hardware.md`](07_compute_and_hardware.md). Summary:

- **Two-phase training.** Phase A (primary): freeze SSL encoders, **precompute features once**, train fusion+heads on cached features — most runs, all ablations, all seeds happen here (fast, bandwidth-light). Phase B (optional): end-to-end **LoRA** finetune of the best config only.
- The 128 GB unified memory removes OOM as a constraint (large batches / long clips OK); the bottleneck is memory *bandwidth*, so we cache features and use BF16 + gradient checkpointing.
- Budget ~15–20 Phase-A runs incl. ablations/seeds + 2–4 Phase-B finetunes on one Spark; a second Spark (ConnectX-7) is the scaling path for full finetunes.
- Log everything with Weights & Biases (or MLflow); configs are the source of truth (`configs/`).

## 7. Reproducibility checklist (attach to submission)

- [ ] Pinned env (`environment.yml`), fixed seeds.
- [ ] Committed split/manifest files.
- [ ] Config for every reported number.
- [ ] Released checkpoints (Base + Lite).
- [ ] Public inference demo (HF Space).
- [ ] Model card + datasheet.
