"""Synthetic quadrant construction for QACP (docs/02_architecture.md §8b).

Builds all four authenticity quadrants + the MISMATCH class from *pristine (RVRA) clips
only* — no deepfake generator outputs:

  pseudo-RVFA    : audio -> vocoder copy-synthesis (synthetic acoustics, same content)
  pseudo-FVRA    : video -> self-blended face (blending/boundary artifacts, own identity)
  pseudo-FVFA    : both transforms
  pseudo-MISMATCH: audio swapped with REAL audio from another clip in the batch
                   -> labels stay (video=real, audio=real) but sync_label = mismatched.
                   This is the disentanglement forcing function: mismatch != fake.

The heavy transforms have two implementations:
  * `_copy_synthesis_griffinlim` — real proxy (STFT -> magnitude -> Griffin-Lim), no deps
    beyond torch. Swap in a neural vocoder (e.g. HiFi-GAN copy-synthesis) for the paper.
  * `_self_blend_video` — tubelet self-blending: re-warped copy of the clip alpha-blended
    back with a soft face-region mask (SBI extended to video). The scaffold uses affine
    jitter + soft-mask blending; landmark-driven warping is a TODO for the real run.

Everything operates on tensors (video: (T,3,H,W), audio: (N,)) so it can run inside the
dataloader OR as an offline pass that also caches SSL features (preferred on DGX Spark).
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

SYNC_MATCHED, SYNC_MISMATCHED = 0, 1


# --------------------------------------------------------------------------- audio
def _copy_synthesis_griffinlim(wave: torch.Tensor, n_fft: int = 512, hop: int = 128,
                               n_iter: int = 16) -> torch.Tensor:
    """Vocoder-proxy: discard phase, re-estimate with Griffin-Lim.

    The output has the same words/timing but synthetic phase/micro-acoustics — exactly
    the class of artifact a neural vocoder leaves. Cheap, dependency-free.
    """
    win = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    mag = spec.abs()
    # Griffin-Lim phase recovery
    angle = torch.rand_like(mag) * 2 * torch.pi
    est = mag * torch.exp(1j * angle)
    for _ in range(n_iter):
        rec = torch.istft(est, n_fft=n_fft, hop_length=hop, window=win, length=wave.numel())
        re_spec = torch.stft(rec, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        est = mag * torch.exp(1j * re_spec.angle())
    return torch.istft(est, n_fft=n_fft, hop_length=hop, window=win, length=wave.numel())


# --------------------------------------------------------------------------- video
def _soft_center_mask(h: int, w: int, device) -> torch.Tensor:
    """Soft elliptical mask approximating the face region of an aligned crop."""
    ys = torch.linspace(-1, 1, h, device=device).view(-1, 1)
    xs = torch.linspace(-1, 1, w, device=device).view(1, -1)
    d = (xs / 0.75) ** 2 + (ys / 0.9) ** 2
    return (1 - d).clamp(0, 1) ** 0.5  # (H, W) in [0,1]


def _self_blend_video(frames: torch.Tensor, max_shift: float = 0.03,
                      max_scale: float = 0.05) -> torch.Tensor:
    """SBI-style tubelet self-blending: blend a slightly re-warped copy of the clip back
    into itself under a soft face mask -> blending-boundary artifacts, identity unchanged.

    frames: (T, 3, H, W) in any float range.
    """
    T, C, H, W = frames.shape
    dev = frames.device
    # one consistent random affine for the whole tubelet (temporally coherent artifact)
    tx = random.uniform(-max_shift, max_shift) * 2
    ty = random.uniform(-max_shift, max_shift) * 2
    s = 1.0 + random.uniform(-max_scale, max_scale)
    theta = torch.tensor([[s, 0.0, tx], [0.0, s, ty]], device=dev).unsqueeze(0).repeat(T, 1, 1)
    grid = F.affine_grid(theta, size=(T, C, H, W), align_corners=False)
    warped = F.grid_sample(frames, grid, align_corners=False, padding_mode="border")
    # mild color jitter on the donor to create a statistics seam
    warped = warped * random.uniform(0.92, 1.08) + random.uniform(-0.03, 0.03)
    mask = _soft_center_mask(H, W, dev).view(1, 1, H, W)
    return frames * (1 - mask) + warped * mask


# --------------------------------------------------------------------------- builder
QACP_CLASSES = ["RVRA", "RVFA", "FVRA", "FVFA", "MISMATCH"]


def build_pseudo_sample(frames: torch.Tensor, wave: torch.Tensor,
                        donor_wave: torch.Tensor | None = None,
                        pseudo_class: str | None = None) -> dict:
    """Turn one pristine clip into a pseudo-quadrant training sample.

    donor_wave: real audio from a different clip (required for MISMATCH).
    Returns dict with tensors + the three QACP label partitions.
    """
    if pseudo_class is None:
        choices = QACP_CLASSES if donor_wave is not None else QACP_CLASSES[:4]
        pseudo_class = random.choice(choices)

    v, a = frames, wave
    video_label, audio_label, sync_label = 0, 0, SYNC_MATCHED

    if pseudo_class in ("RVFA", "FVFA"):
        a = _copy_synthesis_griffinlim(a)
        audio_label = 1
    if pseudo_class in ("FVRA", "FVFA"):
        v = _self_blend_video(v)
        video_label = 1
    if pseudo_class == "MISMATCH":
        assert donor_wave is not None, "MISMATCH needs a donor clip's real audio"
        n = min(a.numel(), donor_wave.numel())
        a = donor_wave[:n]
        if n < wave.numel():
            a = F.pad(a, (0, wave.numel() - n))
        sync_label = SYNC_MISMATCHED  # video/audio labels stay REAL — mismatch != fake

    return {
        "video": v, "audio": a, "pseudo_class": pseudo_class,
        "video_label": torch.tensor(video_label),
        "audio_label": torch.tensor(audio_label),
        "sync_label": torch.tensor(sync_label),
    }


class QACPDataset(torch.utils.data.Dataset):
    """Wraps a manifest of REAL clips; emits pseudo-quadrant samples on the fly.

    For DGX Spark, prefer generating these offline once and caching SSL features
    (docs/07_compute_and_hardware.md §2) — this on-the-fly version is for dry runs
    and for the Phase-B end-to-end finetune.
    """

    def __init__(self, base_dataset):
        # base_dataset must yield dicts with 'video' (T,3,H,W) and 'audio' (N,)
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        rec = self.base[i]
        donor = self.base[random.randrange(len(self.base))]
        return build_pseudo_sample(rec["video"], rec["audio"], donor_wave=donor["audio"])


def collate_qacp(batch):
    out = {}
    for k in ("video", "audio", "video_label", "audio_label", "sync_label"):
        out[k] = torch.stack([b[k] for b in batch])
    out["pseudo_class"] = [b["pseudo_class"] for b in batch]
    return out
