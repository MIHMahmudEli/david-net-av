# Related Work Map

Organize the literature review around five threads. For each, list what they do and the gap DAVID-Net exploits. (Fill exact citations into `references.bib` as you read — starter keys are given.)

## 1. Unimodal video deepfake detection
- Spatial-artifact CNNs: **Xception** (FaceForensics++), **EfficientNet** ensembles (DFDC winners).
- Frequency / fingerprint cues: DCT/GAN-fingerprint methods.
- Temporal / lip-based: **LipForensics**, **RealForensics**, optical-flow and RNN/transformer temporal models.
- *Gap:* audio-blind; degrade badly cross-dataset and under compression.

## 2. Unimodal audio spoofing / synthetic-voice detection
- **RawNet2**, **AASIST** (spectro-temporal graph attention), SSL front-ends (**Wav2Vec2/WavLM/XLS-R**) → strong on **ASVspoof 2019/2021**, **In-the-Wild**.
- *Gap:* video-blind; miss face manipulation entirely.

## 3. Audio-visual (binary) deepfake detection
- **MDS — Modality Dissonance Score** (Chugh et al.).
- **"Emotions Don't Lie"** (Mittal et al.) — affective cross-modal consistency.
- **AVFakeNet**, **AVoiD-DF**, **MRDF**, **FRADE** — fusion transformers.
- *Gap:* single binary label, no per-modality attribution, weak on FVFA and cross-dataset.

## 4. Audio-visual consistency / synchronization
- **SyncNet** and successors; **AVAD** (audio-visual anomaly detection); voice-face homogeneity / speaker-face matching.
- *Gap:* detect mismatch but cannot attribute which stream is fake; fail when both fakes are mutually consistent.

## 5. Temporal localization of partial forgeries
- **LAV-DF** with **BA-TFD / BA-TFD+**; **AV-Deepfake1M / 1M++** baselines.
- *Gap:* usually joint-modality localization; limited per-modality attribution; generalization under-reported.

---

## 6. Proxy-forgery / generator-free training (QACP's lineage — cite and differentiate)
- **Self-Blended Images (SBI)** (Shiohara & Yamasaki) and Face X-ray — synthesize blending artifacts from pristine *images*; unimodal video, no audio.
- **Vocoder copy-synthesis as proxy spoof** (anti-spoofing literature, e.g., vocoded-speech augmentation) — unimodal audio.
- **Prior multi-label AV work** (Zhou & Lim, joint AV detection, ICCV'21; FakeAVCeleb multi-class baselines) — per-modality labels exist, but trained on generator outputs, no pretraining stage, no mismatch class.
- *Gap QACP fills:* unifies the two unimodal proxy tricks into one **quadrant-structured AV pretraining objective**, adds the **real-but-mismatched** class as an explicit disentanglement signal, and formalizes **factorized SupCon** over three differently-partitioned embedding spaces. No prior work does any of the three, let alone together.

## Positioning statement (for the paper intro)

> Prior AV detectors either (i) answer a single binary question, losing attribution; (ii) reason only about cross-modal sync, failing on internally-consistent double fakes; or (iii) localize forgeries without saying *which modality* was altered or *whether it generalizes* to unseen generators. Moreover, all are trained on *generator outputs* and inherit those generators' fingerprints. DAVID-Net (a) introduces **quadrant-aware contrastive pretraining** that constructs all four authenticity quadrants from pristine data alone — no generator outputs — including an explicit *real-but-mismatched* class that teaches the model desynchronization is not forgery; and (b) unifies **per-modality authenticity**, **cross-modal consistency**, and **temporal localization** in one disentangled architecture, reported under a **cross-generator / cross-dataset** protocol with a deployed public detector.

## Starter BibTeX keys — QACP lineage additions
`shiohara2022sbi`, `li2020facexray`, `zhou2021joint`, `wang2023vocoded`, `khosla2020supcon`.

## Starter BibTeX keys (fill in)
`rossler2019faceforensics`, `dolhansky2020dfdc`, `khalid2021fakeavceleb`, `cai2023avdeepfake1m`, `cai2024avdeepfake1mpp`, `cai2022lavdf`, `haliassos2021lipforensics`, `haliassos2022realforensics`, `jung2022aasist`, `tak2021rawnet2`, `chen2022wavlm`, `wang2020asvspoof`, `chugh2020mds`, `mittal2020emotions`, `ilyas2023avfakenet`, `yang2023avoiddf`, `feng2023avanomaly`, `chung2016syncnet`, `tan2024badtfd`.
