"""External (published) baselines, trained on OUR strict split with OUR metrics.

AASIST (Jung et al., ICASSP 2022) -- audio anti-spoofing. The official architecture file
    models/AASIST.py (clovaai/aasist, MIT licence) is fetched at a pinned commit and its
    sha256 recorded; hyperparameters follow the official config/AASIST.conf (Adam,
    lr 1e-4, weight decay 1e-4, cosine schedule). Input: the raw 4 s waveform.
EfficientNet-B4 frame classifier -- the backbone of the DFDC-winning video detectors
    (timm `tf_efficientnet_b4.ns_jft_in1k`, Noisy-Student weights), fine-tuned on the same
    face crops DAVID-Net sees; clip score = mean logit over `frames_per_clip` frames
    (random frames in training, evenly spaced in evaluation).

Both plug into the common trainer as models returning {logit_v, logit_a, logit_quad}.
"""
from __future__ import annotations

import hashlib
import importlib.util
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn

AASIST_COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
AASIST_URL = f"https://raw.githubusercontent.com/clovaai/aasist/{AASIST_COMMIT}/models/AASIST.py"
AASIST_CONFIG = {"nb_samp": 64600, "first_conv": 128,
                 "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                 "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
                 "temperatures": [2.0, 2.0, 100.0, 100.0]}
EFFNET_NAME = "tf_efficientnet_b4.ns_jft_in1k"


def fetch_aasist(dst_dir: str | Path) -> tuple[type, dict]:
    dst = Path(dst_dir) / f"AASIST_{AASIST_COMMIT[:8]}.py"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(AASIST_URL, str(dst))
    spec = importlib.util.spec_from_file_location("aasist_official", dst)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prov = {"source": AASIST_URL, "commit": AASIST_COMMIT,
            "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "config": AASIST_CONFIG}
    return mod.Model, prov


class AASISTBaseline(nn.Module):
    def __init__(self, cache_dir: str | Path):
        super().__init__()
        Model, self.provenance = fetch_aasist(cache_dir)
        self.net = Model(dict(AASIST_CONFIG))

    def forward(self, video, audio, v_avail=None, a_avail=None):
        with torch.autocast(device_type="cuda", enabled=False):       # sinc filters: fp32
            _, out = self.net(audio.float())
        logit = (out[:, 0] - out[:, 1]).float()      # our convention: positive = fake
        nan = torch.full_like(logit, float("nan"))
        return {"logit_v": nan, "logit_a": logit, "logit_quad": None}


class FrameCNNBaseline(nn.Module):
    def __init__(self, frames_per_clip: int = 4, model_name: str = EFFNET_NAME):
        super().__init__()
        import timm
        self.net = timm.create_model(model_name, pretrained=True, num_classes=1)
        cfg = self.net.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))
        self.k = frames_per_clip
        self.provenance = {"timm_model": model_name, "timm": timm.__version__,
                           "pretrained_cfg": {k: str(v) for k, v in cfg.items()
                                              if k in ("hf_hub_id", "tag", "input_size")}}

    def forward(self, video, audio, v_avail=None, a_avail=None):
        B, T = video.shape[:2]
        if self.training:        # random frames (torch RNG -> restored exactly on resume)
            idx = torch.stack([torch.randperm(T, device=video.device)[:self.k] for _ in range(B)])
        else:
            idx = torch.linspace(0, T - 1, self.k, device=video.device).round().long().expand(B, -1)
        frames = torch.gather(video, 1, idx.view(B, self.k, 1, 1, 1).expand(-1, -1, *video.shape[2:]))
        x = (frames.flatten(0, 1) - self.mean) / self.std
        logit = self.net(x).view(B, self.k).mean(1).float()
        nan = torch.full_like(logit, float("nan"))
        return {"logit_v": logit, "logit_a": nan, "logit_quad": None}


def build_external(name: str, cfg: dict, cache_dir: str | Path) -> nn.Module:
    if name == "aasist":
        return AASISTBaseline(cache_dir)
    if name == "effnet_b4":
        return FrameCNNBaseline(cfg["train"]["external"]["frames_per_clip"])
    raise KeyError(name)
