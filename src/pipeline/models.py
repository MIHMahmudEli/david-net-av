"""Model builders: DAVID-Net (Phase A on features, Phase B on pixels/waveforms) and the
feature-space baselines that share its inputs, splits and metrics."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.david_net import DavidNet, DavidNetConfig


def build_davidnet(cfg: dict, phase: str = "A", revisions: dict | None = None,
                   trainable_video: int = 0, trainable_audio: int = 0,
                   gradient_checkpointing: bool = False) -> DavidNet:
    m = cfg["model"]
    mcfg = DavidNetConfig(d_model=m["d_model"], n_heads=m["n_heads"],
                          n_fusion_layers=m["n_fusion_layers"], dropout=m["dropout"],
                          use_sync=m["use_sync"], use_disentangle=m["use_disentangle"],
                          compose_quadrant=m["compose_quadrant"])
    if phase == "A":
        venc, aenc = nn.Identity(), nn.Identity()      # inputs are cached token sequences
    else:
        from src.pipeline.encoders import build_encoders
        venc, aenc = build_encoders(cfg["features"], revisions or {}, trainable_video,
                                    trainable_audio, gradient_checkpointing)
    return DavidNet(mcfg, venc, aenc)


class _Pool(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # (B, L, d) -> (B, d)
        return self.drop(self.norm(x.mean(1)))


class LinearProbe(nn.Module):
    """Unimodal baseline: mean-pooled frozen SSL tokens -> linear authenticity logit."""

    def __init__(self, d: int, modality: str, dropout: float = 0.1):
        super().__init__()
        self.modality = modality
        self.pool = _Pool(d, dropout)
        self.head = nn.Linear(d, 1)

    def forward(self, video, audio, v_avail=None, a_avail=None):
        x = video if self.modality == "video" else audio
        logit = self.head(self.pool(x)).squeeze(-1)
        nan = torch.full_like(logit, float("nan"))
        return {"logit_v": logit if self.modality == "video" else nan,
                "logit_a": logit if self.modality == "audio" else nan,
                "logit_quad": None}


class LateFusionMLP(nn.Module):
    """Multimodal baseline without cross-modal attention, sync or disentanglement."""

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.pv, self.pa = _Pool(d, dropout), _Pool(d, dropout)
        self.null_v = nn.Parameter(torch.zeros(d))
        self.null_a = nn.Parameter(torch.zeros(d))
        self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout))
        self.head_v, self.head_a = nn.Linear(d, 1), nn.Linear(d, 1)
        self.head_q = nn.Linear(d, 4)

    def forward(self, video, audio, v_avail=None, a_avail=None):
        zv, za = self.pv(video), self.pa(audio)
        if v_avail is not None:
            zv = zv * v_avail[:, None] + self.null_v * (1 - v_avail[:, None])
        if a_avail is not None:
            za = za * a_avail[:, None] + self.null_a * (1 - a_avail[:, None])
        h = self.mlp(torch.cat([zv, za], -1))
        return {"logit_v": self.head_v(h).squeeze(-1), "logit_a": self.head_a(h).squeeze(-1),
                "logit_quad": self.head_q(h)}


def build_baseline(name: str, d: int, dropout: float = 0.1) -> nn.Module:
    if name == "video_probe":
        return LinearProbe(d, "video", dropout)
    if name == "audio_probe":
        return LinearProbe(d, "audio", dropout)
    if name == "late_fusion":
        return LateFusionMLP(d, dropout)
    raise KeyError(f"unknown baseline {name!r}")


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
