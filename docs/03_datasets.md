# Datasets & Preprocessing

> **Access note.** Most of these require signing an EULA / request form. Start access requests in Month 0 — approvals can take weeks. None are redistributed by this repo; `scripts/download.md` will hold the per-dataset request links.

## 1. Primary datasets (train + in-domain test)

| Dataset | Modalities | Labels | Why we use it |
|---------|-----------|--------|---------------|
| **FakeAVCeleb** | A+V | Native **4-quadrant** (RVRA/RVFA/FVRA/FVFA) | The only large set with exactly our label space; primary training set for attribution. |
| **AV-Deepfake1M / 1M++** | A+V | Clip + **temporal localization**, content-driven edits | Largest AV set; drives the localization task and generalization. |
| **LAV-DF** | A+V | **Localized** content-driven forgeries with timestamps | Partial forgery localization; complements AV-Deepfake1M. |

## 2. Cross-dataset generalization test sets (never trained on)

| Dataset | Modalities | Use |
|---------|-----------|-----|
| **DFDC** (Deepfake Detection Challenge) | A+V | Large, diverse generators — headline cross-dataset AUC. |
| **KoDF** | A+V | Korean subjects — cross-demographic/language generalization. |
| **DeepfakeTIMIT** | A+V | Classic face-swap; low-resource sanity check. |
| **DF-1.0 / Celeb-DF v2** | V (audio optional) | Video-branch cross-dataset. |

## 3. Audio-branch auxiliary sets (strengthen the acoustic head)

| Dataset | Use |
|---------|-----|
| **ASVspoof 2019 LA / 2021 DF** | Standard anti-spoofing benchmark; pretrain/validate audio head, report EER. |
| **In-the-Wild (voice spoofing)** | Real-world voice-clone generalization. |
| **WaveFake / LibriSeVoc** | Additional vocoder/TTS diversity. |

## 4. Label schema (unified)

Every sample is mapped to:
```json
{
  "clip_id": "str",
  "video_label": 0,           // 0 real, 1 fake
  "audio_label": 1,           // 0 real, 1 fake
  "quadrant": "RVFA",         // derived
  "video_segments": [[s,e], ...],   // manipulated intervals (sec), [] if none/whole
  "audio_segments": [[s,e], ...],
  "generator": "str",         // e.g. wav2lip, sv2tts, faceswap, real
  "dataset": "str",
  "meta": {"gender": "...", "language": "...", "source_fps": 25}
}
```
A conversion script per dataset produces this unified manifest (`src/data/manifests/*.jsonl`). Cross-dataset splits are just manifest filters — no re-download.

## 5. Preprocessing pipeline (`src/data/preprocess.py`)

**Video**
1. Decode → sample at fixed fps (default 25).
2. Face detect + track (RetinaFace / MediaPipe), align to canonical landmarks.
3. Two crops: full face (224×224) and mouth ROI (96×96).
4. Store as short tensor shards (`.npy`/`webdataset`) to avoid re-decoding each epoch.

**Audio**
1. Extract track, resample to 16 kHz mono.
2. Loudness normalize; keep raw waveform (SSL front-ends take raw audio) + precompute log-mel for the spectro-temporal head.
3. Align audio frames to video frames via timestamps for the sync module.

**Sharding & caching:** WebDataset tar shards keyed by `clip_id`; a manifest maps clip → shard. This makes multi-dataset training I/O-efficient.

## 6. Splitting protocol (critical for Q1)

- **In-domain:** official train/val/test where provided; else subject-disjoint 70/10/20 (never split the same identity across train/test).
- **Leave-one-generator-out (LOGO):** hold out one manipulation method entirely (e.g., train without Wav2Lip, test on Wav2Lip).
- **Cross-dataset:** train on FakeAVCeleb+AV-Deepfake1M, test on DFDC/KoDF untouched.
- All split files are committed under `src/data/splits/` for exact reproducibility.

## 7. Class balance & sampling

FakeAVCeleb is ~9:1 fake:real and quadrant-skewed. Use generator-balanced + quadrant-balanced sampling and report metrics that are robust to imbalance (AUC, macro-F1, per-quadrant recall) rather than raw accuracy.

## 8. Ethics / licensing

- Each dataset's license and consent basis is recorded in `scripts/download.md`.
- No raw faces/voices are redistributed. Only manifests, split files, and model weights are released.
- Demographic metadata is used solely for fairness auditing (Section RQ5), not as model input.
