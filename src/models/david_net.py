"""DAVID-Net: Disentangled Audio-Visual Deepfake detector.

Reference implementation skeleton. Encoders are pluggable (see video_encoder.py /
audio_encoder.py). This file wires the disentangled fusion + multi-task heads and
returns per-modality authenticity, quadrant, localization, and sync outputs.

The code is intentionally dependency-light so it runs as a shape/logic sanity check
before real backbones are attached. See docs/02_architecture.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DavidNetConfig:
    d_model: int = 768
    n_heads: int = 8
    n_fusion_layers: int = 4
    dropout: float = 0.1
    n_quadrants: int = 4  # RVRA, RVFA, FVRA, FVFA
    use_sync: bool = True
    use_disentangle: bool = True
    compose_quadrant: bool = False  # if True, quadrant derived from H_v * H_a


class CrossModalBlock(nn.Module):
    """One layer of self- + cross-attention over video/audio token streams."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.sa_v = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.sa_a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ca_v = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ca_a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_v = nn.LayerNorm(d_model)
        self.ln_a = nn.LayerNorm(d_model)
        self.ff_v = _ff(d_model, dropout)
        self.ff_a = _ff(d_model, dropout)

    def forward(self, v, a):
        v = v + self.sa_v(v, v, v, need_weights=False)[0]
        a = a + self.sa_a(a, a, a, need_weights=False)[0]
        v_c, attn_v = self.ca_v(v, a, a, need_weights=True)
        a_c, attn_a = self.ca_a(a, v, v, need_weights=True)
        v = self.ln_v(v + v_c)
        a = self.ln_a(a + a_c)
        v = v + self.ff_v(v)
        a = a + self.ff_a(a)
        return v, a, (attn_v, attn_a)


def _ff(d_model: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
    )


class SyncModule(nn.Module):
    """Windowed contrastive AV synchronization → consistency embedding + per-frame agreement."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_a = nn.Linear(d_model, d_model)
        self.temp = nn.Parameter(torch.tensor(0.07))

    def forward(self, v, a):
        # v: (B, Lv, d), a: (B, La, d). Align to common length by interpolation.
        L = min(v.size(1), a.size(1))
        vv = F.normalize(self.proj_v(_resize_seq(v, L)), dim=-1)
        aa = F.normalize(self.proj_a(_resize_seq(a, L)), dim=-1)
        agreement = (vv * aa).sum(-1)          # (B, L) per-window cosine agreement
        z_c = torch.cat([vv.mean(1), aa.mean(1)], dim=-1)  # (B, 2d) pooled consistency
        return z_c, agreement, (vv, aa)

    def contrastive_loss(self, vv, aa):
        """InfoNCE over aligned windows (positives on the diagonal)."""
        B, L, d = vv.shape
        v = vv.reshape(B * L, d)
        a = aa.reshape(B * L, d)
        logits = v @ a.t() / self.temp.clamp(min=1e-3)
        target = torch.arange(B * L, device=v.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def _resize_seq(x, L):
    return F.interpolate(x.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)


class DavidNet(nn.Module):
    def __init__(self, cfg: DavidNetConfig, video_encoder: nn.Module, audio_encoder: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        d = cfg.d_model
        self.fusion = nn.ModuleList(
            [CrossModalBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_fusion_layers)]
        )
        self.sync = SyncModule(d) if cfg.use_sync else None

        # Learnable "missing modality" tokens: substituted for an absent stream so the
        # same network handles audio-only inputs and silent (video-only) clips.
        self.null_v = nn.Parameter(torch.zeros(1, 1, d))
        self.null_a = nn.Parameter(torch.zeros(1, 1, d))

        c = 2 * d if cfg.use_sync else 0  # consistency embedding width
        # Modality-specific authenticity projections
        self.head_v = nn.Sequential(nn.Linear(d + c, d), nn.GELU(), nn.Linear(d, 1))
        self.head_a = nn.Sequential(nn.Linear(d + c, d), nn.GELU(), nn.Linear(d, 1))
        self.head_quad = nn.Sequential(nn.Linear(2 * d + c, d), nn.GELU(), nn.Linear(d, cfg.n_quadrants))
        # Per-frame localization (video + audio)
        self.loc_v = nn.Linear(d, 1)
        self.loc_a = nn.Linear(d, 1)

    def forward(self, video, audio, v_avail=None, a_avail=None) -> dict:
        """v_avail / a_avail: optional (B,) float masks, 1 = modality present.

        Missing streams are replaced by learnable null tokens; the consistency
        embedding z_c is zeroed for samples lacking either modality (no
        cross-modal evidence exists for them).
        """
        v = self.video_encoder(video)   # (B, Lv, d)
        a = self.audio_encoder(audio)   # (B, La, d)
        B = v.size(0)

        if v_avail is None:
            v_avail = v.new_ones(B)
        if a_avail is None:
            a_avail = a.new_ones(B)
        v = v * v_avail.view(-1, 1, 1) + self.null_v.expand_as(v) * (1 - v_avail.view(-1, 1, 1))
        a = a * a_avail.view(-1, 1, 1) + self.null_a.expand_as(a) * (1 - a_avail.view(-1, 1, 1))
        both = (v_avail * a_avail)      # (B,) 1 only when cross-modal evidence exists

        attn_maps = []
        for blk in self.fusion:
            v, a, attn = blk(v, a)
            attn_maps.append(attn)

        z_v, z_a = v.mean(1), a.mean(1)  # pooled modality-specific authenticity
        agreement = None
        sync_pack = None
        if self.sync is not None:
            z_c, agreement, sync_pack = self.sync(v, a)
            z_c = z_c * both.unsqueeze(-1)
            agreement = agreement * both.unsqueeze(-1)
        else:
            z_c = z_v.new_zeros(z_v.size(0), 0)

        feat_v = torch.cat([z_v, z_c], dim=-1)
        feat_a = torch.cat([z_a, z_c], dim=-1)
        logit_v = self.head_v(feat_v).squeeze(-1)
        logit_a = self.head_a(feat_a).squeeze(-1)

        if self.cfg.compose_quadrant:
            pv, pa = torch.sigmoid(logit_v), torch.sigmoid(logit_a)
            # outer product → [RVRA, RVFA, FVRA, FVFA]
            quad = torch.stack([(1 - pv) * (1 - pa), (1 - pv) * pa, pv * (1 - pa), pv * pa], dim=-1)
            logit_quad = torch.log(quad.clamp(min=1e-6))
        else:
            logit_quad = self.head_quad(torch.cat([z_v, z_a, z_c], dim=-1))

        return {
            "logit_v": logit_v,
            "logit_a": logit_a,
            "logit_quad": logit_quad,
            "loc_v": self.loc_v(v).squeeze(-1),   # (B, Lv)
            "loc_a": self.loc_a(a).squeeze(-1),   # (B, La)
            "agreement": agreement,
            "z_v": z_v, "z_a": z_a, "z_c": z_c,
            "sync_pack": sync_pack,
            "attn_maps": attn_maps,
            "v_avail": v_avail, "a_avail": a_avail, "both_avail": both,
        }


if __name__ == "__main__":
    # Smoke test with dummy encoders (no external weights required).
    from types import SimpleNamespace

    class _DummyEnc(nn.Module):
        def __init__(self, d, L):
            super().__init__()
            self.d, self.L = d, L
            self.proj = nn.Linear(d, d)

        def forward(self, x):  # x: (B, L, d)
            return self.proj(x)

    cfg = DavidNetConfig()
    model = DavidNet(cfg, _DummyEnc(cfg.d_model, 16), _DummyEnc(cfg.d_model, 50))
    B = 2
    out = model(torch.randn(B, 16, cfg.d_model), torch.randn(B, 50, cfg.d_model))
    print({k: (v.shape if torch.is_tensor(v) else type(v).__name__) for k, v in out.items()})
    print("OK")
