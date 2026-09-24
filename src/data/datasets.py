"""Unified AV deepfake dataset over the manifest schema in docs/03_datasets.md.

A manifest is a .jsonl file with one record per clip. This loader reads preprocessed
tensor shards when available, else decodes on the fly. Returns a dict batch consumed by
src/training/train.py and the DAVID-Net forward pass.

Hard rule (post-mortem of the first Kaggle run): when a media root is configured and a
clip cannot be found/decoded, the loader RAISES. It never substitutes random tensors.
Random tensors are only produced in explicit dummy mode (no root, no shards), which is
what the offline unit tests use.
"""
from __future__ import annotations

import json
import logging
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.decode import AUDIO_ONLY_EXTS, DecodeError, decode_clip

logger = logging.getLogger(__name__)

QUADRANT_TO_IDX = {"RVRA": 0, "RVFA": 1, "FVRA": 2, "FVFA": 3}


def load_manifest(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def segments_to_mask(segments, length: int, duration: float) -> torch.Tensor:
    mask = torch.zeros(length)
    if not segments or duration <= 0:
        return mask
    for s, e in segments:
        i0 = max(0, int(length * s / duration))
        i1 = min(length, int(length * e / duration))
        mask[i0:i1] = 1.0
    return mask


def resolve_root_dir(root, records: list[dict], max_probe: int = 5):
    """Return a root under which the manifest's rel_paths actually exist.

    The manifest builder is run against `<mount>/FakeAVCeleb_v1.2` while the notebook
    hands training `<mount>` — this caused a whole run on random tensors. We probe the
    first few records and, if they are missing, try one level of sub-directories.
    """
    root = Path(root)
    probes = [r["rel_path"] for r in records[:max_probe] if "rel_path" in r]
    if not probes:
        return root

    def _hits(base: Path) -> int:
        return sum((base / rp).exists() for rp in probes)

    if _hits(root) == len(probes):
        return root
    candidates = []
    if root.exists():
        candidates += [d for d in sorted(root.iterdir()) if d.is_dir()]
    candidates += [root.parent]
    for c in candidates:
        if _hits(c) == len(probes):
            logger.warning(f"[datasets] root_dir {root} does not contain the manifest paths; "
                           f"auto-resolved to {c}")
            return c
    return root  # leave as is; __getitem__ will raise with a precise message


class AVDeepfakeDataset(Dataset):
    def __init__(self, manifest: str, shard_root: Optional[str] = None,
                 n_frames: int = 16, audio_len: int = 64000, filt=None,
                 root_dir=None, use_faces: bool = False, train: bool = True,
                 allow_dummy: Optional[bool] = None):
        """root_dir: str, Path, or list of str/Path for multi-dataset manifests.

        train=True  -> random temporal window per sample (augmentation)
        train=False -> deterministic center window (validation / test)
        allow_dummy -> emit random tensors when no media source is configured.
                       Defaults to True ONLY when neither root_dir nor shard_root is set.
        """
        self.records = load_manifest(manifest)
        self.undecodable: dict = {}   # clip_id -> reason, reported at end of training
        if filt is not None:
            self.records = [r for r in self.records if filt(r)]
        self.shard_root = Path(shard_root) if shard_root else None
        if root_dir is None:
            roots = []
        elif isinstance(root_dir, (list, tuple)):
            roots = [Path(p) for p in root_dir]
        else:
            roots = [Path(root_dir)]
        self.root_dirs = [resolve_root_dir(r, self.records) for r in roots]
        self.n_frames = n_frames
        self.audio_len = audio_len
        self.use_faces = use_faces
        self.train = train
        self.window = "random" if train else "center"
        if allow_dummy is None:
            allow_dummy = not (self.root_dirs or self.shard_root)
        self.allow_dummy = allow_dummy
        if self.allow_dummy:
            warnings.warn("AVDeepfakeDataset: no media source configured -> DUMMY random "
                          "tensors (dry-run mode). Never train a real model like this.",
                          stacklevel=2)

    def __len__(self):
        return len(self.records)

    # ------------------------------------------------------------------ loading
    def _resolve_media(self, rec) -> Optional[Path]:
        if "rel_path" not in rec:
            return None
        for rd in self.root_dirs:
            p = rd / rec["rel_path"]
            if p.exists():
                return p
        return None

    def _load_tensors(self, rec):
        """Returns (video, audio, has_video, has_audio); video is (T, 3, 224, 224)."""
        # 1. Precomputed tensor shards
        if self.shard_root is not None:
            vp = self.shard_root / f"{rec['clip_id']}_video.pt"
            ap = self.shard_root / f"{rec['clip_id']}_audio.pt"
            if vp.exists() and ap.exists():
                video, audio = torch.load(vp), torch.load(ap)
                return _normalize_video(video), audio.flatten(), True, audio.numel() > 0

        # 2. Decode from the media file
        media = self._resolve_media(rec)
        if media is not None:
            try:
                d = decode_clip(str(media), self.n_frames, self.audio_len, 224, self.window)
            except DecodeError as e:
                raise DecodeError(f"clip_id={rec['clip_id']}: {e}") from e
            return d.video, d.audio, d.has_video, d.has_audio

        # 3. Nothing found
        if not self.allow_dummy:
            tried = [str(rd / rec.get("rel_path", "?")) for rd in self.root_dirs]
            raise FileNotFoundError(
                f"clip_id={rec['clip_id']} not found. Tried: {tried}. "
                "Check root_dir (the manifest's rel_path is relative to the directory "
                "build_manifest.py was pointed at) and shard_root.")
        return (torch.randn(self.n_frames, 3, 224, 224), torch.randn(self.audio_len),
                True, True)

    def get_faces(self, rec):
        """Extract face/mouth ROI crops for a record. Only used when use_faces=True."""
        if not self.use_faces:
            return None, None
        media = self._resolve_media(rec)
        if media is not None:
            from src.data.face_preprocess import extract_face_mouth_from_video
            return extract_face_mouth_from_video(str(media), self.n_frames)
        return (torch.randn(self.n_frames, 3, 224, 224),
                torch.randn(self.n_frames, 3, 96, 96))

    def __getitem__(self, i):
        rec = self.records[i]
        try:
            video, audio, has_v, has_a = self._load_tensors(rec)
        except DecodeError as e:
            # One undecodable clip must not end a 10-hour run. Before ffmpeg had a
            # timeout this blocked forever and Kaggle reclaimed the whole session; now
            # it raises, so substitute a neighbour and record the casualty. The set is
            # reported at the end of training so the exclusions can go in the paper
            # rather than silently skewing the split.
            cid = rec.get("clip_id", f"index:{i}")
            if cid not in self.undecodable:
                self.undecodable[cid] = str(e)[:200]
                print(f"[dataset] UNDECODABLE {cid}: {str(e)[:160]}", flush=True)
            if len(self.undecodable) > max(20, len(self.records) // 100):
                raise DecodeError(
                    f"{len(self.undecodable)} clips failed to decode — this is a broken "
                    f"mount or manifest, not a few bad files") from e
            rec = self.records[(i + 1) % len(self.records)]
            video, audio, has_v, has_a = self._load_tensors(rec)
        dur = float(rec.get("duration_sec", 4.0))
        return {
            "clip_id": rec["clip_id"],
            "video": video,
            "audio": audio,
            "video_label": torch.tensor(rec["video_label"]),
            "audio_label": torch.tensor(rec["audio_label"]),
            "quadrant": torch.tensor(QUADRANT_TO_IDX[rec["quadrant"]]),
            "video_seg_mask": segments_to_mask(rec.get("video_segments"), self.n_frames, dur),
            "audio_seg_mask": segments_to_mask(rec.get("audio_segments"), 100, dur),
            "v_avail": torch.tensor(1.0 if has_v else 0.0),
            "a_avail": torch.tensor(1.0 if has_a else 0.0),
            "generator": rec.get("generator", "unknown"),
            "dataset": rec.get("dataset", "unknown"),
            "race": (rec.get("meta") or {}).get("race", "") or "",
            "gender": (rec.get("meta") or {}).get("gender", "") or "",
        }


class CachedFeatureDataset(AVDeepfakeDataset):
    """Serves precomputed SSL token sequences instead of raw media.

    Produced by src/data/extract_features.py. Batches have the same keys as
    AVDeepfakeDataset, but 'video'/'audio' hold feature tensors (L, d) — the model
    must be built with identity encoders (cfg.feature_cache set; see train.py).
    """

    def __init__(self, manifest: str, feature_cache: str, n_frames: int = 16,
                 audio_len: int = 64000, filt=None):
        super().__init__(manifest, shard_root=None, n_frames=n_frames,
                         audio_len=audio_len, filt=filt, allow_dummy=False)
        self.cache = Path(feature_cache)

    def _load_tensors(self, rec):
        vp = self.cache / f"{rec['clip_id']}_vfeat.pt"
        ap = self.cache / f"{rec['clip_id']}_afeat.pt"
        if not (vp.exists() and ap.exists()):
            raise FileNotFoundError(
                f"missing cached features for {rec['clip_id']} in {self.cache} — "
                "run src/data/extract_features.py first")
        return torch.load(vp), torch.load(ap), True, True


# ====================================================================== sampling
class BalancedBatchSampler(torch.utils.data.Sampler):
    """Quadrant- and generator-balanced batches (docs/02_architecture.md §9).

    Every batch cycles through the quadrants present in the manifest (RVRA, RVFA,
    FVRA, FVFA) and, inside each quadrant, through its generators. Minority strata
    (FakeAVCeleb has 500 real clips vs 20k fakes) are drawn with replacement so every
    batch carries both classes for BOTH modality heads — the previous sampler produced
    single-generator batches (all-fake for ~95% of steps).

    An "epoch" is defined as len(records) samples so LR schedules stay meaningful.
    Yields flat indices; DataLoader(batch_size=...) does the batching.
    """

    def __init__(self, records, batch_size: int, samples_per_epoch: Optional[int] = None,
                 seed: int = 0):
        self.batch_size = batch_size
        self.n = samples_per_epoch or len(records)
        self.seed = seed
        self.epoch = 0
        strata = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(records):
            strata[r.get("quadrant", "?")][r.get("generator", "unknown")].append(i)
        self.quadrants = sorted(strata)
        self.strata = {q: {g: idx for g, idx in gens.items()} for q, gens in strata.items()}

    def set_epoch(self, epoch: int, skip: int = 0):
        """`skip` = number of samples already consumed in this epoch (mid-epoch resume).
        The sequence is a pure function of (seed, epoch), so skipping reproduces exactly
        the batches a killed session would have seen next."""
        self.epoch = epoch
        self.skip = max(0, int(skip))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        # per-stratum shuffled cursors; wrap around (= sampling with replacement across
        # the epoch for small strata)
        pools = {q: {g: rng.sample(idx, len(idx)) for g, idx in gens.items()}
                 for q, gens in self.strata.items()}
        cursors = {q: {g: 0 for g in gens} for q, gens in pools.items()}
        gen_order = {q: sorted(gens) for q, gens in pools.items()}
        gen_cursor = {q: 0 for q in pools}
        q_cursor = rng.randrange(len(self.quadrants))
        skip = getattr(self, "skip", 0)
        for k in range(self.n):
            q = self.quadrants[q_cursor % len(self.quadrants)]
            q_cursor += 1
            gens = gen_order[q]
            g = gens[gen_cursor[q] % len(gens)]
            gen_cursor[q] += 1
            pool = pools[q][g]
            c = cursors[q][g]
            if c >= len(pool):
                rng.shuffle(pool)
                c = 0
            cursors[q][g] = c + 1
            if k >= skip:
                yield pool[c]

    def __len__(self):
        return max(0, self.n - getattr(self, "skip", 0))


# kept for backward compatibility with older configs / scripts
BalancedGeneratorSampler = BalancedBatchSampler


# ====================================================================== collate
def _normalize_video(v: torch.Tensor) -> torch.Tensor:
    """Ensure a video tensor is (T, 3, H, W)."""
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


def collate(batch):
    out = {}
    keys_tensor = ["video", "audio", "video_label", "audio_label", "quadrant",
                   "video_seg_mask", "audio_seg_mask", "v_avail", "a_avail"]
    for k in keys_tensor:
        items = [b[k] for b in batch]
        if k == "video" and items[0].ndim == 4:
            items = [_normalize_video(v) for v in items]
        out[k] = torch.stack(items)
    out["clip_id"] = [b["clip_id"] for b in batch]
    out["generator"] = [b["generator"] for b in batch]
    out["dataset"] = [b["dataset"] for b in batch]
    out["race"] = [b.get("race", "") for b in batch]
    out["gender"] = [b.get("gender", "") for b in batch]
    return out


# ====================================================================== preflight
def preflight_check(ds: AVDeepfakeDataset, n_exist: int = 200, n_decode: int = 2,
                    name: str = "dataset") -> dict:
    """Fail fast if the media is missing or decodes to garbage. Prints a summary.

    * existence: sample `n_exist` records, require >= 98% present on disk
    * decode: load `n_decode` samples, require finite tensors with non-trivial variance
    """
    recs = ds.records
    if not recs:
        raise RuntimeError(f"[preflight:{name}] manifest is empty")
    if ds.allow_dummy:
        print(f"[preflight:{name}] DUMMY MODE — random tensors (no root_dir / shard_root)")
        return {"dummy": True}
    if ds.root_dirs:
        idx = random.Random(0).sample(range(len(recs)), min(n_exist, len(recs)))
        missing = [recs[i]["clip_id"] for i in idx if ds._resolve_media(recs[i]) is None]
        frac = 1 - len(missing) / len(idx)
        print(f"[preflight:{name}] {len(recs)} records, roots={[str(r) for r in ds.root_dirs]}, "
              f"{frac * 100:.1f}% of sampled clips found on disk")
        if frac < 0.98:
            raise FileNotFoundError(
                f"[preflight:{name}] {len(missing)}/{len(idx)} sampled clips missing, e.g. "
                f"{missing[:3]}. Fix root_dir before training.")
    stats = []
    for i in range(min(n_decode, len(recs))):
        s = ds[i]
        v, a = s["video"], s["audio"]
        if not (torch.isfinite(v).all() and torch.isfinite(a).all()):
            raise RuntimeError(f"[preflight:{name}] non-finite tensors in {s['clip_id']}")
        vstd, astd = float(v.std()), float(a.std())
        stats.append((s["clip_id"], tuple(v.shape), vstd, tuple(a.shape), astd,
                      float(s["v_avail"]), float(s["a_avail"])))
        print(f"[preflight:{name}] {s['clip_id']}: video{tuple(v.shape)} std={vstd:.3f} "
              f"range=[{float(v.min()):.2f},{float(v.max()):.2f}] audio{tuple(a.shape)} "
              f"std={astd:.4f} v_avail={float(s['v_avail'])} a_avail={float(s['a_avail'])}")
        if s["v_avail"] > 0 and (vstd < 1e-3 or v.min() < -0.01 or v.max() > 1.01):
            raise RuntimeError(f"[preflight:{name}] video of {s['clip_id']} looks wrong "
                               f"(std={vstd:.4f}, expected [0,1] RGB frames)")
        if s["a_avail"] > 0 and astd < 1e-5:
            raise RuntimeError(f"[preflight:{name}] audio of {s['clip_id']} is silent/constant")
    return {"dummy": False, "samples": stats}
