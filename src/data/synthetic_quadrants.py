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
                                n_iter: int = 8) -> torch.Tensor:
    """Vocoder-proxy with codec simulation: Griffin-Lim + low-rate resampling.

    WavLM is phase-invariant (trained for ASR), so Griffin-Lim alone produces
    embeddings indistinguishable from real audio. We stack three artifact sources
    that attack the magnitude spectrum — exactly what neural vocoders and codec
    pipelines leave behind:

      1. Griffin-Lim phase recovery (fewer iterations → worse phase estimate)
      2. Codec simulation: downsample to 8-16 kHz then back (spectral truncation)
      3. Mild additive shaped noise (vocoder quantization residual)

    The output preserves words/timing/speaker but carries detectable spectral
    distortion — the class of artifact a HiFi-GAN / WaveRNN vocoder leaves
    after codec compression.
    """
    # ── Step 1: Griffin-Lim with reduced iterations (more phase error) ──
    win = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    mag = spec.abs()
    angle = torch.rand_like(mag) * 2 * torch.pi
    est = mag * torch.exp(1j * angle)
    for _ in range(n_iter):
        rec = torch.istft(est, n_fft=n_fft, hop_length=hop, window=win, length=wave.numel())
        re_spec = torch.stft(rec, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        est = mag * torch.exp(1j * re_spec.angle())
    gl_wave = torch.istft(est, n_fft=n_fft, hop_length=hop, window=win, length=wave.numel())

    # ── Step 2: Codec simulation (spectral truncation via resampling) ──
    # Simulate lossy codec: downsample to a lower rate then upsample back.
    # This removes high-frequency content and introduces spectral ringing.
    fake_sr = random.choice([8000, 11025, 16000])
    orig_len = gl_wave.numel()
    if fake_sr < 16000:
        factor = 16000 // fake_sr
        down = gl_wave[::factor]
        # Nearest-neighbor upsample (introduces staircase artifacts)
        up = down.repeat_interleave(factor)[:orig_len]
        gl_wave = up

    # ── Step 3: Mild shaped noise (vocoder quantization residual) ──
    # Low-level broadband noise at 20-30 dB SNR mimics quantization artifacts.
    noise = torch.randn_like(gl_wave)
    sig_power = gl_wave.pow(2).mean().clamp(min=1e-10)
    noise_power = noise.pow(2).mean().clamp(min=1e-10)
    snr_db = random.uniform(20, 30)
    snr = 10 ** (snr_db / 10)
    scale = (sig_power / (snr * noise_power + 1e-10)).sqrt()
    gl_wave = gl_wave + noise * scale

    return gl_wave


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

    Pseudo-classes are assigned per epoch by `set_epoch` (a fixed permutation table)
    instead of being drawn inside __getitem__, so `QACPBalancedSampler` can lay out
    every mini-batch with distinct classes -> both values of video_label, audio_label
    and sync_label are present in (almost) every batch and SupCon always has positives
    AND negatives on each axis. `views_per_clip` repeats each pristine clip that many
    times per epoch under different pseudo-classes (FakeAVCeleb has only ~350 train
    reals, far too few for one view per epoch).

    For DGX Spark, prefer generating these offline once and caching SSL features
    (docs/07_compute_and_hardware.md §2).
    """

    def __init__(self, base_dataset, views_per_clip: int = 4, seed: int = 0):
        # base_dataset must yield dicts with 'video' (T,3,H,W) and 'audio' (N,)
        self.base = base_dataset
        self.views = max(1, int(views_per_clip))
        self.seed = seed
        self.n_base = len(base_dataset)
        self.set_epoch(0)

    def __len__(self):
        return self.n_base * self.views

    def set_epoch(self, epoch: int):
        """Assign a pseudo-class to every (clip, view) slot for this epoch."""
        rng = random.Random(self.seed * 1000 + epoch)
        n = len(self)
        # balanced multiset of classes, shuffled -> each class ~n/5 times
        classes = [QACP_CLASSES[i % len(QACP_CLASSES)] for i in range(n)]
        rng.shuffle(classes)
        self.class_of = classes
        self.epoch = epoch

    def __getitem__(self, i):
        rec = self.base[i % self.n_base]
        cls = self.class_of[i]
        donor_wave = None
        if cls == "MISMATCH":
            j = random.randrange(self.n_base - 1)
            j = j + 1 if j >= (i % self.n_base) else j   # never the same clip
            donor_wave = self.base[j]["audio"]
        return build_pseudo_sample(rec["video"], rec["audio"], donor_wave=donor_wave,
                                   pseudo_class=cls)


class QACPBalancedSampler(torch.utils.data.Sampler):
    """Orders indices so that consecutive `batch_size` items have distinct pseudo-classes.

    Must be given the same QACPDataset (it reads `class_of`); call `set_epoch` on both
    before each epoch so the sampler and the dataset agree on the class table.
    """

    def __init__(self, dataset: QACPDataset, batch_size: int, seed: int = 0):
        self.ds = dataset
        self.bs = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed * 7919 + self.epoch)
        buckets = {c: [] for c in QACP_CLASSES}
        for i, c in enumerate(self.ds.class_of):
            buckets[c].append(i)
        for b in buckets.values():
            rng.shuffle(b)
        order = list(QACP_CLASSES)
        out = []
        start = 0
        while any(buckets.values()):
            batch = []
            k = 0
            while len(batch) < self.bs and k < len(order):
                c = order[(start + k) % len(order)]
                k += 1
                if buckets[c]:
                    batch.append(buckets[c].pop())
            start += 1
            if not batch:
                break
            out.extend(batch)
        return iter(out)

    def __len__(self):
        return len(self.ds)


def _ensure_video_shape(v: torch.Tensor) -> torch.Tensor:
    """Ensure video is (T, 3, H, W) — defensive against bad shards/decode."""
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
    elif v.ndim == 3:
        v = v.unsqueeze(1).repeat(1, 3, 1, 1)
    elif v.ndim == 5:
        v = v.view(-1, *v.shape[-3:])
        if v.shape[1] != 3:
            v = v[:, :3, :, :]
    return v


def collate_qacp(batch):
    out = {}
    for k in ("video", "audio", "video_label", "audio_label", "sync_label"):
        items = [b[k] for b in batch]
        if k == "video":
            items = [_ensure_video_shape(v) for v in items]
        out[k] = torch.stack(items)
    out["pseudo_class"] = [b["pseudo_class"] for b in batch]
    return out


# NOTE: the former `collate_qacp_stratified` duplicated a sample and FLIPPED its label
# whenever a batch was label-homogeneous. That injects a contradictory (identical
# content, opposite label) pair into SupCon and was one cause of the flat QACP loss.
# Balanced batches are now guaranteed upstream by QACPBalancedSampler instead.
