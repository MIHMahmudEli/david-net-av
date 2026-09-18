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
                frames.permute(0, 2, 3, 1), size=(new_h, new_w),
                mode="bilinear", align_corners=False
            ).permute(0, 3, 1, 2)
            frames = F.interpolate(
                frames, size=(H, W), mode="bilinear", align_corners=False
            )
        # Gaussian blur
        if random.random() < 0.2:
            sigma = random.uniform(0.5, 1.5)
            k = int(sigma * 3) | 1
            x = frames.reshape(T * C, 1, H, W)
            kernel = torch.randn(k, 1, 1, 1) * 0.1
            kernel = kernel.to(x.device)
            x = F.conv2d(x, kernel.expand(1, -1, -1, -1), padding=k // 2)
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


def augment_batch(batch: dict, video_aug: VideoAugmentor | None = None,
                  audio_aug: AudioAugmentor | None = None) -> dict:
    """Apply augmentations to a training batch in-place."""
    if video_aug is not None:
        vids = batch["video"]
        # If vids is already a stacked tensor, iterate over batch dim
        if torch.is_tensor(vids):
            augmented = []
            for v in vids:
                # Defensive: ensure (T, C, H, W) with C=3
                if v.ndim == 4 and v.shape[1] != 3:
                    if v.shape[-1] == 3:
                        v = v.permute(0, 3, 1, 2)
                    else:
                        v = v[:, :3, :, :]
                augmented.append(video_aug(v))
            batch["video"] = torch.stack(augmented)
        else:
            # list of tensors — normalize each before augmenting
            normalized = []
            for v in vids:
                if v.ndim == 4 and v.shape[1] != 3:
                    if v.shape[-1] == 3:
                        v = v.permute(0, 3, 1, 2)
                    else:
                        v = v[:, :3, :, :]
                normalized.append(video_aug(v))
            batch["video"] = torch.stack(normalized)
    if audio_aug is not None:
        batch["audio"] = torch.stack([audio_aug(a) for a in batch["audio"]])
    return batch
