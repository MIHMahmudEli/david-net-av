"""Baseline detectors evaluated under the SAME manifests/splits as DAVID-Net.

Registry pattern: each baseline is a nn.Module taking the standard batch tensors
and returning a single fake-probability logit per clip for ITS modality.

Built-in (dependency-light, runnable today):
    video-framecnn : per-frame CNN + temporal mean-pool  (Xception-style stand-in)
    audio-speccnn  : log-spectrogram CNN                  (LCNN-style stand-in)

Paper-grade baselines plug in through the same registry (uncomment deps in
requirements.txt and the timm/transformers branches below):
    video-xception : timm 'legacy_xception' per-frame + mean-pool
    video-effb4    : timm 'efficientnet_b4' per-frame + mean-pool
    audio-wavlm    : transformers WavLM + linear head
Official AASIST / RawNet2 / LipForensics checkpoints should be run from their
reference repos on OUR split manifests; store their prediction dumps in
results/ with the same JSON schema so figures/tables pick them up identically
(see HANDOFF.md §Baselines).
"""
from __future__ import annotations

import torch
import torch.nn as nn

REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def build_baseline(name: str, **kw) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline '{name}' — available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)


# ============================================================ video baselines
@register("video-framecnn")
class VideoFrameCNN(nn.Module):
    """Per-frame CNN + temporal mean-pool. Modality: video."""
    modality = "video"

    def __init__(self, width: int = 64):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1), nn.GELU(),
            nn.Conv2d(w, 2 * w, 3, 2, 1), nn.GELU(),
            nn.Conv2d(2 * w, 4 * w, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(4 * w, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        video = batch["video"]                    # (B, T, 3, H, W)
        b, t = video.shape[:2]
        feats = self.net(video.flatten(0, 1))     # (B*T, C)
        return self.head(feats.view(b, t, -1).mean(1)).squeeze(-1)


@register("video-xception")
class VideoXception(nn.Module):
    """timm legacy_xception per-frame + mean-pool (the canonical FF++ baseline)."""
    modality = "video"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import timm  # lazy — uncomment timm in requirements.txt
        self.backbone = timm.create_model("legacy_xception", pretrained=pretrained,
                                          num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        video = batch["video"]
        b, t = video.shape[:2]
        feats = self.backbone(video.flatten(0, 1))
        return self.head(feats.view(b, t, -1).mean(1)).squeeze(-1)


# ============================================================ audio baselines
@register("audio-speccnn")
class AudioSpecCNN(nn.Module):
    """Log-STFT spectrogram CNN. Modality: audio."""
    modality = "audio"

    def __init__(self, width: int = 64):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(1, w, 3, 2, 1), nn.GELU(),
            nn.Conv2d(w, 2 * w, 3, 2, 1), nn.GELU(),
            nn.Conv2d(2 * w, 4 * w, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(4 * w, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        wave = batch["audio"]                     # (B, N)
        win = torch.hann_window(400, device=wave.device)
        spec = torch.stft(wave, n_fft=400, hop_length=160, window=win,
                          return_complex=True).abs().clamp(min=1e-6).log()
        return self.head(self.net(spec.unsqueeze(1))).squeeze(-1)


@register("audio-wavlm")
class AudioWavLM(nn.Module):
    """WavLM + linear probe (SSL anti-spoofing baseline)."""
    modality = "audio"

    def __init__(self, model_name: str = "microsoft/wavlm-base-plus"):
        super().__init__()
        from transformers import AutoModel  # lazy
        self.backbone = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        h = self.backbone(batch["audio"]).last_hidden_state.mean(1)
        return self.head(h).squeeze(-1)
