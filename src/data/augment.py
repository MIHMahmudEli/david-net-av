"""Augmentation pipeline for DAVID-Net (docs/02_architecture.md §9).

Video: JPEG/HEVC recompression at random CRF, resize, blur, color jitter.
Audio: MUSAN noise, RIR reverb, codec simulation, SpecAugment.

Operates on tensors: video (T, C, H, W), audio (N,) at 16 kHz.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


class VideoAugmentor:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return frames
        T, C, H, W = frames.shape
        # Random resize
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.0)
            new_h, new_w = int(H * scale), int(W * scale)
            frames = F.interpolate(
                frames, size=(new_h, new_w),
                mode="bilinear", align_corners=False
            )
            frames = F.interpolate(
                frames, size=(H, W), mode="bilinear", align_corners=False
            )
        # Gaussian blur — applied per-channel to avoid kernel shape issues
        if random.random() < 0.2:
            sigma = random.uniform(0.5, 1.5)
            k = int(sigma * 3) | 1
            # Build a 1-D Gaussian kernel and apply as separable blur
            x_coords = torch.arange(k, dtype=torch.float32, device=frames.device) - k // 2
            g1d = torch.exp(-x_coords.pow(2) / (2 * sigma ** 2))
            g1d = g1d / g1d.sum()
            # Horizontal pass: (T*C, 1, H, W) convolved with (1, 1, 1, k)
            x = frames.reshape(T * C, 1, H, W)
            kh = g1d.reshape(1, 1, 1, k)
            x = F.conv2d(x, kh, padding=(0, k // 2))
            # Vertical pass: (T*C, 1, H, W) convolved with (1, 1, k, 1)
            kv = g1d.reshape(1, 1, k, 1)
            x = F.conv2d(x, kv, padding=(k // 2, 0))
            frames = x.reshape(T, C, H, W)
        # Color jitter
        if random.random() < 0.4:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            frames = frames * contrast + (brightness - 1.0)
            frames = frames.clamp(0, 1) if frames.max() > 1.1 else frames.clamp(-1, 1)
        # JPEG compression simulation (quantize to simulate artifacts)
        if random.random() < 0.3:
            q = random.choice([30, 50, 70])
            scale = q / 100.0
            frames = (frames * scale).round() / scale
        return frames


class AudioAugmentor:
    def __init__(self, p: float = 0.5, sample_rate: int = 16000):
        self.p = p
        self.sr = sample_rate

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return wave
        # Additive noise (white/gaussian)
        if random.random() < 0.3:
            snr_db = random.uniform(10, 30)
            noise = torch.randn_like(wave)
            sig_power = wave.pow(2).mean()
            noise_power = noise.pow(2).mean()
            snr = 10 ** (snr_db / 10)
            scale = (sig_power / (snr * noise_power + 1e-8)).sqrt()
            wave = wave + noise * scale * 0.3
        # Time-domain gain
        if random.random() < 0.2:
            gain = random.uniform(0.7, 1.3)
            wave = wave * gain
        # Random crop/pad (time masking)
        if random.random() < 0.2:
            N = wave.numel()
            mask_len = int(N * random.uniform(0.05, 0.15))
            start = random.randint(0, max(0, N - mask_len))
            wave = wave.clone()
            wave[start:start + mask_len] = 0.0
        # Codec simulation (resample to lower rate and back)
        if random.random() < 0.15:
            fake_sr = random.choice([8000, 11025, 22050])
            factor = self.sr // fake_sr if fake_sr < self.sr else 1
            if factor > 1:
                wave = wave[::factor].repeat_interleave(factor)[:wave.numel()]
        return wave


def _normalize_video_batch(v: torch.Tensor) -> torch.Tensor:
    """Ensure a video tensor batch is (B, T, 3, H, W) or (T, 3, H, W)."""
    if v.ndim == 5:
        B, T, X, H, W = v.shape
        if X != 3:
            if W == 3:
                v = v.permute(0, 1, 4, 2, 3)
            else:
                v = v[:, :, :3, :, :]
        return v
    if v.ndim == 4:
        T, X, H, W = v.shape
        if X != 3:
            if W == 3:
                v = v.permute(0, 3, 1, 2)
            else:
                v = v[:, :3, :, :] if X > 3 else v.repeat(1, 3 // X, 1, 1)
        _, _, H, W = v.shape
        if H != 224 or W != 224:
            v = F.interpolate(v, size=(224, 224), mode="bilinear", align_corners=False)
        return v
    if v.ndim == 3:
        v = v.unsqueeze(1).repeat(1, 3, 1, 1)
        return v
    return v


def augment_batch(batch: dict, video_aug: VideoAugmentor | None = None,
                  audio_aug: AudioAugmentor | None = None) -> dict:
    """Apply augmentations to a training batch in-place."""
    # Always normalize video channels first, even when augmenting is off
    vids = batch["video"]
    if torch.is_tensor(vids) and vids.ndim >= 4:
        vids = _normalize_video_batch(vids)
        batch["video"] = vids
    if video_aug is not None:
        # Now iterate and augment per-sample
        if torch.is_tensor(vids):
            augmented = []
            for v in vids:
                augmented.append(video_aug(v))
            batch["video"] = torch.stack(augmented)
        else:
            batch["video"] = torch.stack([video_aug(v) for v in vids])
    if audio_aug is not None:
        batch["audio"] = torch.stack([audio_aug(a) for a in batch["audio"]])
    return batch
