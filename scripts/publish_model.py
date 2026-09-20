"""Publish the trained DAVID-Net checkpoint as a PUBLIC Hugging Face model repo.

The training backup repo (MoshinAli/david-net-av-backup) is private and holds
optimizer states, logs and every seed; end users need a single clean, public repo
with the weights, the config they were trained with, and a model card.

    python scripts/publish_model.py --checkpoint runs/stage1_v2_seed42/best.pt \
        --config /kaggle/working/stage1_v2_seed42_config.yaml \
        --summary /kaggle/working/paper/metrics/summary.json \
        --repo MoshinAli/david-net-av --public

Repo layout:
    david_net.pt     {"model": state_dict, "cfg": training config, "epoch": int}
    config.yaml      the Stage-1 config (architecture + data settings)
    README.md        model card (results table filled from summary.json)
    api/             inference wrapper + FastAPI app + Dockerfile (for a Space)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def _fmt(d, k):
    a = d.get(k)
    if not a or a.get("mean") != a.get("mean"):  # NaN
        return "--"
    return f"{a['mean']:.3f} ± {a['std']:.3f}"


def model_card(summary: dict | None, cfg: dict, repo: str, checkpoint_epoch) -> str:
    rows = ""
    if summary:
        in_dom = summary.get("in_domain")
        for ds, d in summary.get("datasets", {}).items():
            a = d["aggregate"]
            n = next(iter(d["per_seed"].values()))["n"]
            tag = " (in-domain test)" if ds == in_dom else " (cross-dataset)"
            rows += f"| {ds}{tag} | {n} | {_fmt(a, 'video_auc')} | {_fmt(a, 'audio_auc')} | {_fmt(a, 'quadrant_acc')} |\n"
    seeds = ", ".join(summary["run_ids"]) if summary else "n/a"
    return f"""---
license: cc-by-nc-4.0
language: en
tags:
- deepfake-detection
- audio-visual
- video-classification
- audio-classification
- forensics
library_name: pytorch
pipeline_tag: video-classification
---

# DAVID-Net — Disentangled Audio-Visual Deepfake Detector

Per-modality deepfake detection: given a clip, DAVID-Net decides **independently** whether
the **video** stream and the **audio** stream are real or AI-generated, assigns one of four
quadrants (RVRA / RVFA / FVRA / FVFA), estimates audio-visual **synchrony**, and produces a
per-frame **localization** timeline for each stream. Missing modalities (silent video,
audio-only files) are handled with learnable null tokens.

Pretraining: **QACP** (Quadrant-Aware Contrastive Pretraining) on pristine clips only —
pseudo-quadrants built from self-blended video, vocoder copy-synthesis audio and cross-clip
mismatch — followed by supervised multi-task training on FakeAVCeleb (subject-disjoint split).

Backbones: `{cfg.get('video_model_name', 'MCG-NJU/videomae-base')}` (video) and
`{cfg.get('audio_model_name', 'microsoft/wavlm-base-plus')}` (audio); {cfg.get('n_fusion_layers', 4)}-layer
pre-LN cross-modal fusion, d_model = {cfg.get('d_model', 768)}.

## Results (mean ± std over seeds: {seeds})

| Dataset | n | Video AUC | Audio AUC | Quadrant acc |
|---|---|---|---|---|
{rows if rows else "| (fill from paper/metrics/summary.json) | | | | |"}

AUC is reported as `--` where a corpus contains a single class for that stream (e.g. all-fake
audio corpora); per-generator and per-quadrant breakdowns, EER, ECE, bootstrap CIs, robustness
sweeps and training curves are in the thesis artifacts.

## Usage

```python
# pip install torch transformers huggingface_hub opencv-python-headless  (+ ffmpeg on PATH)
# git clone https://github.com/MIHMahmudEli/david-net-av && cd david-net-av
from api.inference import DavidNetInference
engine = DavidNetInference(config="configs/david_net_kaggle.yaml")   # weights auto-download from {repo}
print(engine.predict("clip.mp4"))          # video + audio verdicts, quadrant, sync, timelines
print(engine.predict_audio("voice.wav"))   # audio-only input
```

Or load the raw checkpoint:

```python
import torch
from huggingface_hub import hf_hub_download
state = torch.load(hf_hub_download("{repo}", "david_net.pt"), map_location="cpu", weights_only=False)
state["model"]   # state_dict of src.models.david_net.DavidNet
state["cfg"]     # training config (architecture + data settings)
```

A FastAPI service (`api/app.py`, Dockerfile for a HF Space) exposes `/predict` and
`/predict-audio`; it pulls this checkpoint at start-up.

## Training data & protocol

- FakeAVCeleb v1.2, subject-disjoint 70/10/20 split (train: {cfg.get('train_manifest', 'splits/fakeavceleb/train.jsonl')}).
- Inputs: 16 frames + 4 s of 16 kHz audio taken from the same temporal window.
- Checkpoint epoch: {checkpoint_epoch}. Full configs and split files are versioned in the GitHub repo.

## Intended use & limitations

This is a **defensive research detector**. It outputs calibrated probabilities, not proof, and
must not be used as sole evidence in legal, employment or moderation decisions. Performance
drops on generators, languages, codecs and recording conditions not represented in
FakeAVCeleb; cross-dataset numbers above quantify that. No generation tooling is included.

## Citation

DAVID-Net / QACP — AIUB BSc thesis (2025–26), *A Unified Forensic Framework for the
Detection of Multimodal Synthetic Media*. Manuscript in preparation.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True, help="Stage-1 config yaml used for training")
    ap.add_argument("--summary", default=None, help="paper/metrics/summary.json (fills the results table)")
    ap.add_argument("--repo", default="MoshinAli/david-net-av")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--repo-root", default=".", help="git checkout root (for api/ + configs/)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("hf")
    if not token:
        raise SystemExit("HF_TOKEN missing")
    import torch
    import yaml
    from huggingface_hub import HfApi

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(state.get("cfg") or yaml.safe_load(open(args.config)))
    cfg = {k: v for k, v in cfg.items() if not isinstance(v, (bytes,))}
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8")) if args.summary and Path(args.summary).exists() else None

    stage = Path(tempfile.mkdtemp(prefix="david_net_pub_"))
    torch.save({"model": state["model"], "cfg": cfg, "epoch": state.get("epoch")}, stage / "david_net.pt")
    with open(stage / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    (stage / "README.md").write_text(model_card(summary, cfg, args.repo, state.get("epoch")), encoding="utf-8")
    root = Path(args.repo_root)
    if (root / "api").exists():
        shutil.copytree(root / "api", stage / "api", ignore=shutil.ignore_patterns("__pycache__"))
    for c in ("david_net_kaggle.yaml", "david_net.yaml"):
        if (root / "configs" / c).exists():
            (stage / "configs").mkdir(exist_ok=True)
            shutil.copy2(root / "configs" / c, stage / "configs" / c)

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=args.repo, repo_type="model",
                      commit_message=f"publish DAVID-Net (epoch {state.get('epoch')})")
    size_gb = (stage / "david_net.pt").stat().st_size / 1e9
    print(f"published to https://huggingface.co/{args.repo} ({'public' if args.public else 'private'}), "
          f"david_net.pt = {size_gb:.2f} GB")
    shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
