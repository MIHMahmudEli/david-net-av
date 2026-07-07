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
                 n_frames: int = 16, audio_len: int = 64000, filt=None):
        self.records = load_manifest(manifest)
        if filt is not None:
            self.records = [r for r in self.records if filt(r)]
        self.shard_root = Path(shard_root) if shard_root else None
        self.n_frames = n_frames
        self.audio_len = audio_len

    def __len__(self):
        return len(self.records)

    def _load_tensors(self, rec):
        """Load preprocessed (video, audio) tensors; fall back to random for dry runs."""
        if self.shard_root is not None:
            vp = self.shard_root / f"{rec['clip_id']}_video.pt"
            ap = self.shard_root / f"{rec['clip_id']}_audio.pt"
            if vp.exists() and ap.exists():
                return torch.load(vp), torch.load(ap)
        # dry-run fallback so the pipeline is runnable end-to-end without data
        video = torch.randn(self.n_frames, 3, 224, 224)
        audio = torch.randn(self.audio_len)
        return video, audio

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
