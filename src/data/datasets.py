"""Unified AV deepfake dataset over the manifest schema in docs/03_datasets.md.

A manifest is a .jsonl file with one record per clip. This loader reads preprocessed
tensor shards when available, else decodes on the fly. Returns a dict batch consumed by
src/training/train.py and the DAVID-Net forward pass.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

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


class AVDeepfakeDataset(Dataset):
    def __init__(self, manifest: str, shard_root: Optional[str] = None,
                 n_frames: int = 16, audio_len: int = 64000, filt=None,
                 root_dir=None, use_faces: bool = False):
        """root_dir: str, Path, or list of str/Path for multi-dataset manifests."""
        self.records = load_manifest(manifest)
        if filt is not None:
            self.records = [r for r in self.records if filt(r)]
        self.shard_root = Path(shard_root) if shard_root else None
        if root_dir is None:
            self.root_dirs = []
        elif isinstance(root_dir, (list, tuple)):
            self.root_dirs = [Path(p) for p in root_dir]
        else:
            self.root_dirs = [Path(root_dir)]
        self.n_frames = n_frames
        self.audio_len = audio_len
        self.use_faces = use_faces

    def __len__(self):
        return len(self.records)

    def _load_tensors(self, rec):
        """Load video/audio tensors from cache, MP4, or dry-run fallback.

        Returns: (video, audio) always. Face crops are handled separately
        via get_faces() when use_faces=True.
        """
        # 1. Try precomputed feature cache
        if self.shard_root is not None:
            vp = self.shard_root / f"{rec['clip_id']}_video.pt"
            ap = self.shard_root / f"{rec['clip_id']}_audio.pt"
            if vp.exists() and ap.exists():
                return torch.load(vp), torch.load(ap)

        # 2. Decode from MP4 using rel_path (try all root dirs)
        if self.root_dirs and "rel_path" in rec:
            for rd in self.root_dirs:
                mp4_path = rd / rec["rel_path"]
                if mp4_path.exists():
                    from src.data.decode import decode_av_from_mp4
                    return decode_av_from_mp4(str(mp4_path), self.n_frames, self.audio_len)

        # 3. Dry-run fallback (random tensors)
        video = torch.randn(self.n_frames, 3, 224, 224)
        audio = torch.randn(self.audio_len)
        return video, audio

    def get_faces(self, rec):
        """Extract face/mouth ROI crops for a record. Only used when use_faces=True."""
        if not self.use_faces:
            return None, None
        if self.root_dirs and "rel_path" in rec:
            for rd in self.root_dirs:
                mp4_path = rd / rec["rel_path"]
                if mp4_path.exists():
                    from src.data.face_preprocess import extract_face_mouth_from_video
                    return extract_face_mouth_from_video(str(mp4_path), self.n_frames)
        return (torch.randn(self.n_frames, 3, 224, 224),
                torch.randn(self.n_frames, 3, 96, 96))

    def __getitem__(self, i):
        rec = self.records[i]
        video, audio = self._load_tensors(rec)
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
            "generator": rec.get("generator", "unknown"),
            "dataset": rec.get("dataset", "unknown"),
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
                         audio_len=audio_len, filt=filt)
        self.cache = Path(feature_cache)

    def _load_tensors(self, rec):
        vp = self.cache / f"{rec['clip_id']}_vfeat.pt"
        ap = self.cache / f"{rec['clip_id']}_afeat.pt"
        if not (vp.exists() and ap.exists()):
            raise FileNotFoundError(
                f"missing cached features for {rec['clip_id']} in {self.cache} — "
                "run src/data/extract_features.py first")
        return torch.load(vp), torch.load(ap)


class BalancedGeneratorSampler(torch.utils.data.Sampler):
    """Sampler that ensures each batch has balanced representation across generators.

    Mitigates generator-dominated bias: the model sees a mix of manipulation
    types per step rather than clusters of one generator (docs/02_architecture.md §9).
    """

    def __init__(self, records, batch_size: int, generator_key: str = "generator"):
        self.batch_size = batch_size
        from collections import defaultdict
        gen_to_idx = defaultdict(list)
        for i, r in enumerate(records):
            gen_to_idx[r.get(generator_key, "unknown")].append(i)
        self.generators = list(gen_to_idx.keys())
        self.gen_indices = dict(gen_to_idx)
        self.n = len(records)

    def __iter__(self):
        import random
        pool = {g: list(idxs) for g, idxs in self.gen_indices.items()}
        for g in pool:
            random.shuffle(pool[g])
        batches = []
        while any(pool.values()):
            batch = []
            # Round-robin across generators
            for g in self.generators:
                while pool[g] and len(batch) < self.batch_size:
                    batch.append(pool[g].pop())
                    if len(batch) >= self.batch_size:
                        break
            if batch:
                random.shuffle(batch)
                batches.append(batch)
        random.shuffle(batches)
        # Yield individual indices (DataLoader handles batching via batch_size)
        for batch in batches:
            yield from batch

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def collate(batch):
    out = {}
    keys_tensor = ["video", "audio", "video_label", "audio_label", "quadrant",
                    "video_seg_mask", "audio_seg_mask"]
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch])
    out["clip_id"] = [b["clip_id"] for b in batch]
    out["generator"] = [b["generator"] for b in batch]
    out["dataset"] = [b["dataset"] for b in batch]
    return out
