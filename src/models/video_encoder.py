"""Video encoder wrappers.

Default: VideoMAE / TimeSformer via HuggingFace `transformers`, projected to d_model
and returned as a token sequence (B, L, d). A lightweight fallback (Conv3D) is provided
so the pipeline runs without downloading large weights.

Swap the backbone by editing `build_video_encoder(cfg)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ProjectTo(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Identity() if in_dim == d_model else nn.Linear(in_dim, d_model)

    def forward(self, x):
        return self.proj(x)


class VideoMAEEncoder(nn.Module):
    """Wraps HF VideoMAE and returns patch/token embeddings as a sequence.

    Requires: transformers, and input frames (B, T, C, H, W).
    """

    def __init__(self, d_model: int = 768, model_name: str = "MCG-NJU/videomae-base",
                 freeze_blocks: int = 6):
        super().__init__()
        from transformers import VideoMAEModel  # lazy import
        self.backbone = VideoMAEModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.project = ProjectTo(hidden, d_model)
        self._freeze(freeze_blocks)

    def _freeze(self, n: int):
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        layers = self.backbone.encoder.layer
        for blk in layers[: min(n, len(layers))]:
            for p in blk.parameters():
                p.requires_grad = False

    def forward(self, frames):
        # frames: (B, T, C, H, W) → VideoMAE expects pixel_values (B, T, C, H, W)
        out = self.backbone(pixel_values=frames).last_hidden_state  # (B, L, hidden)
        return self.project(out)


class ConvFallbackVideoEncoder(nn.Module):
    """No-download fallback: small 3D CNN → token sequence."""

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 64, 3, stride=(1, 2, 2), padding=1), nn.GELU(),
            nn.Conv3d(64, 128, 3, stride=(1, 2, 2), padding=1), nn.GELU(),
            nn.Conv3d(128, d_model, 3, stride=(1, 2, 2), padding=1), nn.GELU(),
        )
        # Cap the spatial grid so the token sequence stays small (keeps attention cheap on CPU).
        self.pool = nn.AdaptiveAvgPool3d((None, 4, 4))

    def forward(self, frames):
        # frames: (B, T, C, H, W) → (B, C, T, H, W)
        x = frames.permute(0, 2, 1, 3, 4)
        x = self.net(x)                      # (B, d, T, H', W')
        x = self.pool(x)                     # (B, d, T, 4, 4)
        b, d, t, h, w = x.shape
        return x.flatten(2).transpose(1, 2)  # (B, T*16, d)


def build_video_encoder(cfg) -> nn.Module:
    if getattr(cfg, "video_backbone", "fallback") == "videomae":
        return VideoMAEEncoder(cfg.d_model, cfg.video_model_name, cfg.freeze_blocks)
    return ConvFallbackVideoEncoder(cfg.d_model)
