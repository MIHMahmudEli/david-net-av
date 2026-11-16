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


class CrossAttentionMLP(nn.Module):
    """Strong multimodal baseline: cross-attention fusion without orthogonal disentanglement or quadrant loss."""
    def __init__(self, d: int, dropout: float = 0.1, n_heads: int = 8):
        super().__init__()
        self.d = d
        self.proj_v = nn.Linear(d, d)
        self.proj_a = nn.Linear(d, d)
        self.ca_v = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ca_a = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_v = nn.LayerNorm(d)
        self.ln_a = nn.LayerNorm(d)
        self.pool = _Pool(d, dropout)
        self.head_v = nn.Linear(d, 1)
        self.head_a = nn.Linear(d, 1)
        self.null_v = nn.Parameter(torch.zeros(d))
        self.null_a = nn.Parameter(torch.zeros(d))

    def forward(self, video, audio, v_avail=None, a_avail=None):
        zv = self.proj_v(video)
        za = self.proj_a(audio)
        cv, _ = self.ca_v(self.ln_v(zv), self.ln_a(za), self.ln_a(za), need_weights=False)
        ca, _ = self.ca_a(self.ln_a(za), self.ln_v(zv), self.ln_v(zv), need_weights=False)
        zv = self.pool(zv + cv)
        za = self.pool(za + ca)
        if v_avail is not None:
            zv = zv * v_avail[:, None] + self.null_v * (1 - v_avail[:, None])
        if a_avail is not None:
            za = za * a_avail[:, None] + self.null_a * (1 - a_avail[:, None])
        return {"logit_v": self.head_v(zv).squeeze(-1), "logit_a": self.head_a(za).squeeze(-1),
                "logit_quad": None}


class AVFFModel(nn.Module):
    """Audio-Visual Feature Fusion (AVFF, Zhou et al., ACM MM 2021)."""
    def __init__(self, d: int, d_mid: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj_v = nn.Sequential(nn.Linear(d, d_mid), nn.ReLU())
        self.proj_a = nn.Sequential(nn.Linear(d, d_mid), nn.ReLU())
        self.conv1d_v = nn.Conv1d(d_mid, d_mid, kernel_size=3, padding=1)
        self.conv1d_a = nn.Conv1d(d_mid, d_mid, kernel_size=3, padding=1)
        self.cross_attn = nn.MultiheadAttention(d_mid, num_heads=4, dropout=dropout, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_mid, d_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_mid, 1)
        )
        self.head_v = nn.Linear(d_mid, 1)
        self.head_a = nn.Linear(d_mid, 1)

    def forward(self, video, audio, v_avail=None, a_avail=None):
        bv, ba = self.proj_v(video), self.proj_a(audio)
        cv = self.conv1d_v(bv.transpose(1, 2)).transpose(1, 2)
        ca = self.conv1d_a(ba.transpose(1, 2)).transpose(1, 2)
        fused, _ = self.cross_attn(cv, ca, ca)
        pooled_v = fused.mean(dim=1)
        pooled_a = ca.mean(dim=1)
        feat = torch.cat([pooled_v, pooled_a], dim=-1)
        joint_logit = self.classifier(feat).squeeze(-1)
        return {"logit_v": self.head_v(pooled_v).squeeze(-1), "logit_a": self.head_a(pooled_a).squeeze(-1),
                "logit_clip": joint_logit, "logit_quad": None}


class AVoiDDFModel(nn.Module):
    """AVoiD-DF: Audio-Visual Joint Learning for Detecting Deepfakes (Yang et al., 2023)."""
    def __init__(self, d: int, d_emb: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj_v = nn.Linear(d, d_emb)
        self.proj_a = nn.Linear(d, d_emb)
        self.norm = nn.LayerNorm(d_emb)
        self.sim_head = nn.Sequential(
            nn.Linear(d_emb * 2, d_emb),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, 1)
        )
        self.head_v = nn.Linear(d_emb, 1)
        self.head_a = nn.Linear(d_emb, 1)

    def forward(self, video, audio, v_avail=None, a_avail=None):
        ev = self.norm(self.proj_v(video.mean(1)))
        ea = self.norm(self.proj_a(audio.mean(1)))
        cat_feat = torch.cat([ev, ea], dim=-1)
        clip_logit = self.sim_head(cat_feat).squeeze(-1)
        return {"logit_v": self.head_v(ev).squeeze(-1), "logit_a": self.head_a(ea).squeeze(-1),
                "logit_clip": clip_logit, "logit_quad": None}


def build_baseline(name: str, d: int, dropout: float = 0.1) -> nn.Module:
    if name == "video_probe":
        return LinearProbe(d, "video", dropout)
    if name == "audio_probe":
        return LinearProbe(d, "audio", dropout)
    if name == "late_fusion":
        return LateFusionMLP(d, dropout)
    if name == "cross_attention":
        return CrossAttentionMLP(d, dropout)
    if name == "avff":
        return AVFFModel(d, dropout=dropout)
    if name == "avoid_df":
        return AVoiDDFModel(d, dropout=dropout)
    raise KeyError(f"unknown baseline {name!r}")


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
