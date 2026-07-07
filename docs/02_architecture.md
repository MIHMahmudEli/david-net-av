# DAVID-Net Architecture

*Disentangled Audio-Visual Deepfake detector.* This document specifies the proposed model in enough detail to implement and to write the methods section of the paper.

---

## 1. Design principles

1. **Two questions, kept separate.** "Is stream X internally authentic?" and "Do the two streams agree with each other?" are different questions. Fusing them too early lets a model shortcut (e.g., calling everything fake whenever lips desync). DAVID-Net keeps **modality-specific authenticity** evidence disentangled from **cross-modal consistency** evidence, then combines them only at the decision heads.
2. **Independent per-modality decisions.** The video and audio heads must be able to fire independently so the four quadrants (RVRA/RVFA/FVRA/FVFA) are all reachable — including **FVFA**, where the two fake streams may be mutually *consistent* (a lip-synced synthetic avatar) and therefore invisible to a pure sync detector.
3. **Localization as supervision, not just output.** Frame-level manipulation labels (available in AV-Deepfake1M / LAV-DF) are used as an auxiliary task that sharpens the clip-level decision.
4. **Pretrained, mostly-frozen encoders** for data efficiency and generalization; only adapters + fusion + heads are trained heavily.

---

## 2. Notation

- Input clip: `T` seconds. Video sampled at `f_v` fps → `N_v` frames; audio at 16 kHz.
- `V ∈ R^{L_v × d}` : video token sequence (spatio-temporal tubelets over the face/lip crop).
- `A ∈ R^{L_a × d}` : audio token sequence.
- `z_v, z_a` : modality-specific authenticity embeddings.
- `z_c` : cross-modal consistency embedding.

---

## 3. Video branch

- **Face/mouth ROI extraction** (preprocessing): RetinaFace/MediaPipe → aligned face crop + a tighter mouth crop. Two crops because face-swap artifacts live in the whole face, lip-sync artifacts live around the mouth.
- **Backbone:** VideoMAE-Base (self-supervised) or TimeSformer, initialized from public weights, partially frozen (freeze first `k` blocks, fine-tune the rest with LoRA/adapters).
- Output projected to `d=768` tokens → `V`.
- Optional **frequency/artifact cue:** a lightweight DCT/high-pass residual stream concatenated as extra tokens (deepfake spatial artifacts are often high-frequency).

## 4. Audio branch

- **Front-end:** WavLM-Base+ or Wav2Vec2-XLS-R features (self-supervised, strong for anti-spoofing), OR an AASIST-style spectro-temporal graph attention front-end. Default: **WavLM features + a small graph-attention head** (best of both — SSL semantics + anti-spoofing inductive bias).
- Output projected to `d=768` tokens → `A`.

## 5. Cross-modal synchronization / consistency module

- Slice `V` and `A` into short overlapping windows (~0.2 s).
- A **contrastive sync objective** (SyncNet-style / InfoNCE) trains window-level `V`/`A` pairs: positives = temporally aligned real windows; negatives = time-shifted or cross-clip windows.
- Produces: a per-window agreement score `s_t` and cross-attention maps. Aggregated into the **consistency embedding `z_c`** and a per-frame sync-inconsistency signal fed to the localization head.
- Rationale (RQ2): voice-cloning-over-real-video (RVFA) and face-swap-over-real-voice (FVRA) frequently break fine-grained lip/prosody alignment; sync evidence is the strongest cue there.

## 6. Disentangled cross-modal fusion

A cross-modal transformer with **two output pathways**:

- **Authenticity pathway** → `z_v`, `z_a` : self-attention within each modality + gated cross-attention, but regularized so it captures *within-stream* forgery artifacts.
- **Consistency pathway** → `z_c` : cross-attention emphasizing agreement/disagreement between streams, seeded by the sync module.

**Disentanglement loss** keeps them separate, options (ablated in the paper):

- Orthogonality: minimize `|cos(z_v, z_c)| + |cos(z_a, z_c)|`.
- Mutual-information penalty (CLUB / vCLUB estimator) between authenticity and consistency features.
- Adversarial: a discriminator tries to predict consistency from `z_v`/`z_a`; the encoder is trained to prevent it.

Default: **orthogonality + MI penalty** (cheap, stable).

## 7. Multi-task heads

| Head | Input | Output | Loss |
|------|-------|--------|------|
| `H_v` video authenticity | `[z_v ; z_c]` | P(video fake) | BCE |
| `H_a` audio authenticity | `[z_a ; z_c]` | P(audio fake) | BCE |
| `H_quad` quadrant | `[z_v ; z_a ; z_c]` | softmax over {RVRA,RVFA,FVRA,FVFA} | CE (or derived from H_v·H_a with a consistency prior) |
| `H_loc` localization | per-frame features + sync signal | per-frame P(manip) for each modality | BCE / focal, evaluated as AP@IoU |

`H_quad` can be trained directly OR composed from `H_v`, `H_a` outputs; we ablate both. Composition enforces logical consistency (quadrant = outer product of the two binary decisions).

## 8. Total objective

```
L = λ1·BCE_v + λ2·BCE_a + λ3·CE_quad + λ4·Loc + λ5·L_sync + λ6·L_disentangle
```

Recommended starting weights: `λ1=λ2=1.0, λ3=0.5, λ4=0.5, λ5=0.3, λ6=0.1` (tuned on the val split; report sensitivity).

Class imbalance: FakeAVCeleb is heavily skewed toward fake — use class-balanced sampling + focal loss on the binary heads.

## 8b. Quadrant-Aware Contrastive Pretraining (QACP) — headline contribution

**Hypothesis.** Detectors generalize poorly because they learn *generator fingerprints*. If we instead pretrain the embedding space to be structured by *quadrant semantics* using forgeries we **construct ourselves from pristine data only**, the space encodes "what manipulation does to a stream" rather than "what generator X looks like" — and transfers to unseen generators.

### Synthetic quadrant construction (no deepfake generators used)

Start from verified-real (RVRA) clips only and manufacture all four quadrants + one auxiliary class:

| Pseudo-class | Construction | What it teaches |
|---|---|---|
| pseudo-RVRA | untouched real clip | anchor |
| pseudo-RVFA | replace audio with **copy-synthesis** (vocoder-resynthesized same utterance: Griffin-Lim / neural vocoder) — content & timing preserved, acoustics synthetic | audio-authenticity cues without any TTS/VC system |
| pseudo-FVRA | **self-blended video**: re-warp/blend the subject's own face across frames (SBI extended to tubelets) — blending/boundary artifacts, no face-swap model | video-authenticity cues without any face generator |
| pseudo-FVFA | both of the above | double-fake anchor that is *internally consistent* (lip timing untouched) |
| pseudo-MISMATCH | swap in **real** audio from a different clip/speaker | **real-but-desynced** — teaches that mismatch ≠ fake |

The MISMATCH class is the disentanglement forcing function: authenticity embeddings must call both streams *real* while the consistency embedding flags *disagreement*. Without it, models collapse "desync" into "fake" (the classic sync-detector failure).

### Objective — factorized supervised contrastive

Three SupCon losses over the three embedding spaces, each with a *different* label partition of the same batch:

```
L_QACP = SupCon(z_v ; video-authenticity)   # {real-video} vs {blended-video}, ignoring audio
       + SupCon(z_a ; audio-authenticity)   # {real-audio} vs {vocoded-audio}, ignoring video
       + SupCon(z_c ; sync-state)           # {matched} vs {mismatched}, ignoring authenticity
```

Because each space is supervised to be *invariant* to the other factors ("ignoring audio/video/authenticity" = those samples are positives for each other), the factorization itself performs disentanglement — the orthogonality/MI penalty of §6 then only has to maintain it, not create it.

### Protocol

1. **Stage 0 (QACP):** pretrain fusion + sync + projection heads on synthetic quadrants built from real clips of the *training* datasets (encoders frozen → runs entirely on cached features; DGX-Spark-cheap, see `07_compute_and_hardware.md`).
2. **Stage 1 (supervised):** finetune with the multi-task heads (§7) on the real labeled quadrant data.
3. Report Stage-0-only linear-probe results — if QACP alone separates quadrants on *unseen generators*, that is a headline figure.

### Why this is novel (positioning)

Self-blending (video) and vocoder copy-synthesis (audio) each exist as *unimodal* proxy-forgery tricks. QACP is (i) the first to **unify them into a quadrant-structured audio-visual pretraining objective**, (ii) the first to add the **real-but-mismatched class** as an explicit disentanglement signal, and (iii) factorized SupCon over three embedding spaces with different label partitions is itself a new formulation for this problem.

## 9. Training recipe (defaults)

- Optimizer AdamW, lr 1e-4 (heads/fusion) / 1e-5 (encoder adapters), cosine schedule, 30–50 epochs, warmup 2 epochs.
- Clip length 2–4 s, 16–32 frames; batch built with **generator-balanced** sampling for cross-generator robustness.
- Augmentation: video (JPEG/HEVC recompression at random CRF, resize, blur, color jitter), audio (MUSAN noise, RIR reverb, codec simulation, SpecAugment).
- **BF16** mixed precision, gradient checkpointing on the video backbone (see `07_compute_and_hardware.md`).
- On DGX Spark, prefer the **cached-feature** regime: freeze encoders, precompute VideoMAE/WavLM features once, train fusion+heads on the cache; keep an optional LoRA end-to-end finetune for the final model.
- **Leave-one-generator-out** and **cross-dataset** are held out from training entirely.

## 10. Complexity / deployment notes

- Provide two configs: **DAVID-Net-Base** (research, VideoMAE-Base + WavLM-Base+) and **DAVID-Net-Lite** (EfficientNet-frame-aggregator + WavLM-Base, for the public HF demo / CPU-friendly inference).
- Inference on a 4 s clip target: < 2 s on a T4 GPU for Base; the Lite model targets CPU-only Spaces.

## 11. Explainability outputs (for the paper and the UI)

- Video: Grad-CAM / attention rollout over the face crop → spatial heatmap.
- Audio: gradient saliency over the spectrogram / time axis.
- Sync: per-frame agreement curve `s_t`.
- Localization: per-modality manipulation timeline — this is what the Next.js UI renders.

## 12. Module → RQ traceability

| Module | Addresses |
|--------|-----------|
| Per-modality heads + quadrant head | RQ1 |
| Sync/consistency module | RQ2 |
| Disentanglement | RQ3 |
| Localization head | RQ4 |
| Augmentation + cross-dataset regime + fairness sampling | RQ5 |
| **QACP pretraining (§8b)** | **RQ6** (generator-free generalization) |
