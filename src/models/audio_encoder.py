"""Audio encoder wrappers.

Default: WavLM / Wav2Vec2 (self-supervised) via HuggingFace `transformers`, projected to
d_model, returned as a token sequence (B, L, d). A small spectro-temporal CNN fallback is
provided so the pipeline runs without downloads. An optional graph-attention head
(AASIST-style) can be layered on top for anti-spoofing inductive bias.
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


class WavLMEncoder(nn.Module):
    """Wraps HF WavLM/Wav2Vec2; input raw waveform (B, num_samples) at 16 kHz."""

    def __init__(self, d_model: int = 768, model_name: str = "microsoft/wavlm-base-plus",
                 freeze_feature_extractor: bool = True, freeze_blocks: int = 8):
        super().__init__()
        from transformers import AutoModel  # lazy import
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.project = ProjectTo(hidden, d_model)
        if freeze_feature_extractor and hasattr(self.backbone, "feature_extractor"):
            self.backbone.feature_extractor._freeze_parameters()
        # Freeze the first freeze_blocks transformer layers (WavLM uses .encoder.layers)
        self._freeze_transformer_blocks(freeze_blocks)

    def _freeze_transformer_blocks(self, n: int):
        """Freeze the first n transformer blocks; leave the rest trainable for QACP."""
        layers = None
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
            layers = self.backbone.encoder.layers
        if layers is None:
            return  # unsupported architecture variant, skip gracefully
        for blk in layers[: min(n, len(layers))]:
            for p in blk.parameters():
                p.requires_grad = False

    def forward(self, waveform):
        out = self.backbone(waveform).last_hidden_state  # (B, L, hidden)
        return self.project(out)


class SpecCNNFallbackAudioEncoder(nn.Module):
    """No-download fallback: log-mel → 2D CNN → token sequence.

    Accepts either a precomputed log-mel (B, n_mels, T) or raw waveform (B, N); if raw,
    a torchaudio MelSpectrogram is applied when available.
    """

    def __init__(self, d_model: int = 768, n_mels: int = 80):
        super().__init__()
        self.n_mels = n_mels
        self.melspec = None
        try:
            import torchaudio
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=400, hop_length=160, n_mels=n_mels
            )
        except Exception:
            pass
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, d_model, 3, stride=(2, 1), padding=1), nn.GELU(),
        )

    def forward(self, audio):
        # Raw waveform (B, num_samples) -> time-frequency map.
        if audio.dim() == 2 and audio.size(1) > self.n_mels * 4:
            if self.melspec is not None:
                x = self.melspec(audio).clamp(min=1e-6).log()      # (B, n_mels, T)
            else:  # no torchaudio: STFT magnitude spectrogram, no extra deps
                win = torch.hann_window(400, device=audio.device)
                spec = torch.stft(audio, n_fft=400, hop_length=160, window=win,
                                  return_complex=True)
                x = spec.abs().clamp(min=1e-6).log()               # (B, F=201, T)
        else:  # already a 2D feature map (B, F, T)
            x = audio
        x = x.unsqueeze(1)                     # (B, 1, F, T)
        x = self.net(x)                        # (B, d, F', T')
        b, d, f, t = x.shape
        return x.mean(2).transpose(1, 2)       # (B, T', d)


def build_audio_encoder(cfg) -> nn.Module:
    if getattr(cfg, "audio_backbone", "fallback") == "wavlm":
        return WavLMEncoder(cfg.d_model, cfg.audio_model_name, cfg.freeze_feature_extractor,
                            getattr(cfg, "freeze_blocks", 8))
    return SpecCNNFallbackAudioEncoder(cfg.d_model)
