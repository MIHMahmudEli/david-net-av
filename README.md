# DAVID-Net: Disentangled Audio-Visual Forensic Attribution under Identity-Disjoint Evaluation

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange.svg)](#citation)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/MIHMahmudEli/david-net-av)
[![Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Preprocessed%20Shards-red)](https://huggingface.co/datasets/MIHMahmudEli/david-net-av)
[![Experiments](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-160--Run%20Logs-blue)](https://huggingface.co/MIHMahmudEli/davidnet-experiments)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel%20Edge-black)](https://david-net-av.vercel.app/)
[![API Docs](https://img.shields.io/badge/API-FastAPI%20HF%20Spaces-brightgreen)](https://mohsinelis-david-net-av-v2.hf.space/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **DAVID-Net** (*Decoupled Audio-Visual Forensic Network*), an asymmetric cross-modal transformer framework for fine-grained four-quadrant deepfake attribution, missing-modality fault tolerance, and millisecond-level temporal boundary localization.

---

## 📌 Overview

Traditional deepfake detectors output a binary verdict (*"real vs. fake"*), collapsing complex multimodal manipulations into an uninterpretable score that cannot identify which communication channel was altered. When confronted with modern deepfakes—where synthetic video is paired with authentic speech, or cloned audio is dubbed over genuine footage—binary detectors fail to provide actionable diagnostic evidence.

**DAVID-Net** reformulates deepfake detection as a **Four-Quadrant Authenticity Attribution** task:

| Quadrant | Visual Stream | Acoustic Stream | Forensic Diagnosis |
|:---|:---:|:---:|:---|
| **RVRA** | Real ($y_v=0$) | Real ($y_a=0$) | **Authentic Media**: Genuine visual and acoustic recording. |
| **RVFA** | Real ($y_v=0$) | **Fake** ($y_a=1$) | **Voice Cloning / Dubbing**: Pristine footage paired with cloned/synthetic speech. |
| **FVRA** | **Fake** ($y_v=1$) | Real ($y_a=0$) | **Facial Manipulation**: Face swap or reenactment over authentic speech. |
| **FVFA** | **Fake** ($y_v=1$) | **Fake** ($y_a=1$) | **Dual Manipulation**: Spliced or end-to-end synthetic double forgery. |

---

## 🏗️ Architecture & Methodological Highlights

1. **Decoupled Feature Representation**:
   - Visual Stream: VideoMAE ViT-Base ($128 \times 768$ spatio-temporal tokens).
   - Acoustic Stream: WavLM Base+ ($50 \times 768$ acoustic tokens).
   - Native transformer hidden dimension $d = 768$.
2. **Windowed Cross-Modal Synchronization**:
   - Computes localized temporal inner-product agreement $s_t = \tilde{V}_t^\top \tilde{A}_t \in [-1, 1]$ over interpolated $K=50$ windows.
   - Symmetric bidirectional InfoNCE contrastive alignment objective $\mathcal{L}_{\text{sync}}$ ($\tau \ge 0.01$) trained strictly on authentic media.
3. **Geometric Subspace Disentanglement**:
   - Enforces strict subspace orthogonality: $z_v \perp z_c$ and $z_a \perp z_c$ ($\mathcal{L}_{\text{ortho}} = |\cos(z_v, z_c^{(v)})| + |\cos(z_a, z_c^{(a)})|$).
   - Prevents the network from conflating physical synchronization with authentic production.
4. **Learnable Null-Token Routing**:
   - Ingests silent video or isolated speech without crashing or random guessing via learnable tokens $\mathbf{e}_{\text{null}}^{(v)}, \mathbf{e}_{\text{null}}^{(a)} \in \mathbb{R}^{1 \times 768}$.
5. **Multi-Task Attribution Heads**:
   - Decoupled projection heads for video veracity ($H_v$), audio veracity ($H_a$), four-quadrant attribution ($H_{quad}$), and frame-level boundary localization ($H_{loc}$).

---

## 📊 Key Empirical Findings

All reported results are averaged over 5 independent locked seeds $\mathcal{S} \in \{7, 42, 123, 456, 2024\}$ from a 160-run evaluation campaign:

- **In-Domain Attribution (FakeAVCeleb, 0.0% Identity Leakage)**:
  - **Clip AUC**: $0.976 \pm 0.006$ (Phase A) / $0.978 \pm 0.005$ (Phase B).
  - **Video AUC**: $0.979 \pm 0.003$ | **Audio AUC**: $0.999 \pm 0.002$.
  - **Macro-$F_1$**: $91.8 \pm 1.7\%$ ($91.3 \pm 1.0\%$ Balanced Accuracy).
  - **Zero Cross-Modal Confusion**: $0.0\%$ misassignment between RVFA and FVRA.
  - **Statistical Significance**: Paired DeLong tests with Holm–Bonferroni correction confirm gains over Late Fusion, EfficientNet-B4, and AASIST across all 5 seeds ($p < 0.002$).
- **Zero-Shot Cross-Dataset Transfer (6 Unseen External Benchmarks)**:
  - ASVspoof 2019 LA: **$0.839 \pm 0.019$ AUC** (+19.1 points over AASIST).
  - In-the-Wild Audio: **$0.691 \pm 0.021$ AUC** (+12.2 points over AASIST).
  - Celeb-DF v2: **$0.682 \pm 0.029$ AUC** (+8.3 points over Late Fusion).
  - DeepFakeTIMIT: **$0.865 \pm 0.014$ AUC**.
- **Leave-One-Generator-Out (LOGO) Generalization**:
  - Held-out FSGAN: $0.928$ | Held-out RTVC/SV2TTS: $0.941$ | Held-out Wav2Lip: $0.902$.
  - Sustains an out-of-domain resilience floor $\ge 90.0\%$ AUC against unseen manipulation families.
- **Hardware Audits on Tesla T4 GPU**:
  - Full attribution in **$5{,}791$\,ms** (17.2% faster than AVFF at $6{,}995$\,ms), operating on the Pareto-optimal frontier.

---

## 📁 Repository Structure

```
david-net-av/
├── README.md                   # Project documentation and reproduction guide
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment definition
├── predict.py                  # Standalone CLI prediction tool
├── evaluate.py                 # Multi-benchmark evaluation harness
├── configs/
│   └── model_config.json       # Architectural hyperparameters (d=768, 4 layers, 8 heads)
├── manifests/
│   └── test_manifest_summary.json # Cryptographic partition digests and split stats
└── src/
    ├── models/
    │   ├── david_net.py        # Core DAVID-Net multi-task architecture
    │   ├── video_encoder.py    # VideoMAE foundation wrapper
    │   └── audio_encoder.py    # WavLM foundation wrapper
    ├── data/
    │   ├── preprocess.py       # YuNet face/mouth cropping & audio normalization
    │   └── datasets.py         # PyTorch multimodal dataset loader
    ├── training/
    │   ├── losses.py           # Multi-task focal, InfoNCE sync, and orthogonality loss
    │   └── qacp.py             # Quadrant-Aware Contrastive Pretraining transforms
    └── eval/
        ├── metrics.py          # AUC, EER, Macro-F1, and ECE calculation
        └── localization.py     # Millisecond temporal tampering interval decoder
```

---

## ⚡ Installation & Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/MIHMahmudEli/david-net-av.git
cd david-net-av

# Using conda (recommended)
conda env create -f environment.yml
conda activate davidnet

# Or using pip
pip install -r requirements.txt
```

### 2. Download Pretrained Weights

Pretrained model weights are hosted on the Hugging Face Hub:

```bash
# Download official release checkpoint (Phase A, 75.6M active parameters)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='MIHMahmudEli/david-net-av', filename='david_net_best.safetensors', local_dir='checkpoints')
"
```

### 3. Run Inference on a Sample Clip

```bash
# Multimodal audio-visual evaluation
python predict.py --input sample_video.mp4 --checkpoint checkpoints/david_net_best.safetensors

# Muted / silent video (routes through acoustic null token)
python predict.py --input silent_clip.mp4 --checkpoint checkpoints/david_net_best.safetensors

# Audio-only phone call / voice memo (routes through visual null token)
python predict.py --input speech_sample.wav --checkpoint checkpoints/david_net_best.safetensors
```

**Example JSON Output**:
```json
{
  "quadrant_prediction": "FVRA",
  "confidence": 0.9412,
  "video_veracity": {
    "probability_fake": 0.9764,
    "verdict": "FAKE"
  },
  "audio_veracity": {
    "probability_fake": 0.0128,
    "verdict": "REAL"
  },
  "synchronization": {
    "mean_agreement": 0.8421,
    "desynchronization_detected": false
  },
  "calibration": {
    "temperature_applied": 1.18,
    "calibrated": true
  }
}
```

---

## 🔬 Benchmark Evaluation Replication

To evaluate DAVID-Net on the official identity-disjoint test partition of FakeAVCeleb:

```bash
python evaluate.py \
  --checkpoint checkpoints/david_net_best.safetensors \
  --manifest manifests/fakeavceleb_test_disjoint.jsonl \
  --batch_size 16
```

All 160 experimental runs across all 5 seeds, configurations, and baseline comparisons are permanently archived on Hugging Face:
- **Model Checkpoints**: [huggingface.co/MIHMahmudEli/david-net-av](https://huggingface.co/MIHMahmudEli/david-net-av)
- **Preprocessed Datasets**: [huggingface.co/datasets/MIHMahmudEli/david-net-av](https://huggingface.co/datasets/MIHMahmudEli/david-net-av)
- **Experiments Archive**: [huggingface.co/MIHMahmudEli/davidnet-experiments](https://huggingface.co/MIHMahmudEli/davidnet-experiments)

---

## 🌐 Online Interactive Demonstration

- **Web Dashboard**: [https://david-net-av.vercel.app/](https://david-net-av.vercel.app/)
- **FastAPI Backend Microservice**: [https://mohsinelis-david-net-av-v2.hf.space/](https://mohsinelis-david-net-av-v2.hf.space/)
- **Interactive OpenAPI Docs**: [https://mohsinelis-david-net-av-v2.hf.space/docs](https://mohsinelis-david-net-av-v2.hf.space/docs)

*Stateless privacy contract: All user-submitted media is decoded ephemerally in RAM and wiped immediately following inference.*

---

## 📜 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{hossain2026davidnet,
  title={DAVID-Net: Disentangled Audio-Visual Forensic Attribution under Identity-Disjoint Evaluation},
  author={Hossain, Moshin Ibna and Khan, Alif Hasan and Shakib, Shadman and Jam, MD. Yeahyea and Hasan, Mohammad Mahmudul},
  journal={Machine Learning with Applications},
  year={2026},
  note={Under review}
}

@thesis{hossain2026thesis,
  title={Decoupled Multimodal Forensic Attribution and Desynchronization Detection},
  author={Hossain, Moshin Ibna and Khan, Alif Hasan and Shakib, Shadman and Jam, MD. Yeahyea},
  school={American International University-Bangladesh},
  year={2026}
}
```

---

## ⚖️ License & Ethical Notice

This repository is distributed under the [MIT License](LICENSE). 

Forensic outputs provided by this system represent probabilistic calibration evidence and should not be construed as automated legal determinations. Forensic practitioners must evaluate predictive intervals within broader chains of custody and corroborating investigative evidence.
