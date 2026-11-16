"""Pooled SSL encoders shared by Phase A (frozen, cached) and Phase B (partially trained).

Phase A caches exactly what these modules output with every block frozen; Phase B
plugs the same modules into DAVID-Net and unfreezes the top blocks. At initialization
the Phase-B network therefore computes the identical function as the Phase-A network
it is warm-started from, so Phase B is a controlled fine-tune, not a new architecture.

Token layout
------------
video: VideoMAE-base, 16 frames -> 8 tubelets x 14 x 14 patches -> adaptive average
       pooling to 8 x g x g  -> (B, 8*g*g, 768)          (g = features.video_spatial_grid)
audio: WavLM-base-plus, 4 s @ 16 kHz -> 199 frames of 20 ms -> adaptive pooling to
       audio_tokens -> (B, audio_tokens, 768)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_revision(model_name: str, revision: Optional[str] = None) -> str:
    """Pin a Hub model to a commit sha (recorded with every feature set / experiment)."""
    if revision:
        return revision
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_name).sha
    except Exception:  # noqa: BLE001 - offline: record the symbolic name
        return "main"


class PooledVideoMAE(nn.Module):
    def __init__(self, model_name: str, revision: str = "main", grid: int = 4,
                 trainable_top_blocks: int = 0, gradient_checkpointing: bool = False):
        super().__init__()
        from transformers import VideoMAEModel
        self.backbone = VideoMAEModel.from_pretrained(model_name, revision=revision)
        cfg = self.backbone.config
        self.tubelet = cfg.tubelet_size
        self.patch = cfg.patch_size
        self.hidden = cfg.hidden_size
        self.grid = grid
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        set_trainable_top_blocks(self.backbone.encoder.layer, self.backbone.embeddings,
                                 trainable_top_blocks)
        if gradient_checkpointing and trainable_top_blocks:
            self.backbone.gradient_checkpointing_enable()

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, 3, H, W) in [0, 1]
        B, T, _, H, W = frames.shape
        x = (frames - self.img_mean) / self.img_std
        h = self.backbone(pixel_values=x).last_hidden_state          # (B, t*h*w, C)
        t, gh, gw = T // self.tubelet, H // self.patch, W // self.patch
        h = h.view(B, t, gh, gw, self.hidden).permute(0, 4, 1, 2, 3)  # (B, C, t, h, w)
        h = F.adaptive_avg_pool3d(h, (t, self.grid, self.grid))
        return h.flatten(2).transpose(1, 2)                            # (B, t*g*g, C)


class PooledWavLM(nn.Module):
    def __init__(self, model_name: str, revision: str = "main", n_tokens: int = 50,
                 trainable_top_blocks: int = 0, gradient_checkpointing: bool = False):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name, revision=revision)
        self.n_tokens = n_tokens
        self.hidden = self.backbone.config.hidden_size
        if hasattr(self.backbone, "feature_extractor"):
            self.backbone.feature_extractor._freeze_parameters()     # conv front-end: always frozen
        set_trainable_top_blocks(self.backbone.encoder.layers, None, trainable_top_blocks,
                                 extra_frozen=[m for m in (getattr(self.backbone, "feature_projection", None),
                                                           getattr(self.backbone.encoder, "pos_conv_embed", None),
                                                           getattr(self.backbone.encoder, "layer_norm", None))
                                               if m is not None])
        if gradient_checkpointing and trainable_top_blocks:
            self.backbone.gradient_checkpointing_enable()

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # wave: (B, N) raw 16 kHz (wavlm-base-plus: do_normalize=False)
        h = self.backbone(wave.float().clamp(-1.0, 1.0)).last_hidden_state   # (B, L, C)
        h = F.adaptive_avg_pool1d(h.transpose(1, 2), self.n_tokens)
        return h.transpose(1, 2)                                             # (B, n, C)


def set_trainable_top_blocks(layers, embeddings, n_top: int, extra_frozen=()):
    """Freeze everything, then unfreeze the last `n_top` transformer blocks."""
    for p in layers.parameters():
        p.requires_grad = False
    if embeddings is not None:
        for p in embeddings.parameters():
            p.requires_grad = False
    for m in extra_frozen:
        for p in m.parameters():
            p.requires_grad = False
    if n_top > 0:
        for blk in list(layers)[-n_top:]:
            for p in blk.parameters():
                p.requires_grad = True


def build_encoders(fcfg: dict, revisions: dict, trainable_video: int = 0,
                   trainable_audio: int = 0, gradient_checkpointing: bool = False):
    v = PooledVideoMAE(fcfg["video_model"], revisions.get("video", "main"),
                       fcfg["video_spatial_grid"], trainable_video, gradient_checkpointing)
    a = PooledWavLM(fcfg["audio_model"], revisions.get("audio", "main"),
                    fcfg["audio_tokens"], trainable_audio, gradient_checkpointing)
    return v, a
