"""Phase-B (end-to-end) inputs: frames/waveforms instead of cached features.

Video corpora come from the clip cache written by prepare.py (clipcache/<corpus>/
shard_*.{bin,json}: DVC2 blobs of the SAME face-cropped 24-frame / 6 s spans the
features were computed from). Audio-only corpora are decoded directly from the
mounted Kaggle dataset (cheap: no video, no ffmpeg for wav/flac).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.pipeline.features import QUAD_IDX, _h, _seg_mask


class ClipCacheIndex:
    def __init__(self, data_store, corpus: str, scratch: Path):
        self.root = Path(scratch) / "clipcache" / corpus
        prefix = f"clipcache/{corpus}"
        if data_store is not None:
            names = sorted({Path(i.path).stem for i in data_store.list_files(prefix, recursive=False)
                            if i.path.endswith(".json")})
            for nm in names:
                for ext in (".json", ".bin"):
                    if not (self.root / f"{nm}{ext}").exists():
                        data_store.download(f"{prefix}/{nm}{ext}", scratch)
        self.where: dict[str, tuple[str, int, int]] = {}
        for js in sorted(self.root.glob("shard_*.json")):
            for cid, (off, ln) in json.loads(js.read_text()).items():
                self.where[cid] = (str(js.with_suffix(".bin")), off, ln)
        self._fh: dict = {}

    def blob(self, cid: str) -> bytes:
        path, off, ln = self.where[cid]
        key = (os.getpid(), path)
        fh = self._fh.get(key)
        if fh is None:
            fh = self._fh[key] = open(path, "rb", buffering=0)
        if hasattr(os, "pread"):
            return os.pread(fh.fileno(), ln, off)
        fh.seek(off)
        return fh.read(ln)


class PixelDataset(Dataset):
    """Same record/label semantics as FeatureDataset, inputs are frames + waveform."""

    def __init__(self, records: list[dict], cache: Optional[ClipCacheIndex], train: bool,
                 seed: int, n_frames: int = 16, audio_len: int = 64000, eval_offset: int = 4,
                 media_root: Optional[str] = None, dcfg: Optional[dict] = None):
        self.cache, self.train, self.seed = cache, train, seed
        self.records = [r for r in records if r.get("modalities") == "audio"
                        or (cache is not None and r["clip_id"] in cache.where)]
        self.missing = len(records) - len(self.records)
        self.n_frames, self.audio_len, self.eval_offset = n_frames, audio_len, eval_offset
        self.media_root, self.dcfg = media_root, dcfg
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return len(self.records)

    def _load(self, i: int, r: dict):
        from src.data.clipcache import decode_blob
        if r.get("modalities") == "audio":
            from src.pipeline.prepare import decode_record
            d = decode_record((r, self.media_root, self.dcfg))
            if d["error"]:
                raise RuntimeError(d["error"])
            a = torch.from_numpy(d["audio"].astype(np.float32) / 32768.0)
            off = int(round(self.eval_offset * 16000 / 4.0))
            a = a[off:off + self.audio_len]
            a = torch.nn.functional.pad(a, (0, self.audio_len - a.numel()))
            return torch.zeros(self.n_frames, 3, 224, 224), a, False, d["has_audio"]
        rng = random.Random(_h(self.seed, self.epoch, i))
        v, a, hv, ha = decode_blob(self.cache.blob(r["clip_id"]), self.n_frames, self.audio_len,
                                   "random" if self.train else "center", rng)
        return v, a, hv, ha

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        v, a, hv, ha = self._load(i, r)
        vl, al = r.get("video_label", -1), r.get("audio_label", -1)
        q = r.get("quadrant")
        return {"idx": i, "clip_id": r["clip_id"], "video": v, "audio": a,
                "v_avail": torch.tensor(float(hv)), "a_avail": torch.tensor(float(ha)),
                "video_label": torch.tensor(vl), "audio_label": torch.tensor(al),
                "clip_label": torch.tensor(r.get("clip_label", int(vl == 1 or al == 1))),
                "quadrant": torch.tensor(QUAD_IDX[q] if q in QUAD_IDX else -1),
                "video_seg_mask": _seg_mask(r.get("video_segments"), 128),
                "audio_seg_mask": _seg_mask(r.get("audio_segments"), 50),
                "generator": r.get("generator", ""), "dataset": r.get("dataset", ""),
                "race": (r.get("meta") or {}).get("race", ""),
                "gender": (r.get("meta") or {}).get("gender", "")}


def phase_b_param_groups(model, cfg: dict) -> list[dict]:
    t = cfg["train"]["phase_b"]
    enc, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if n.startswith(("video_encoder.", "audio_encoder.")) else rest).append(p)
    return [{"params": rest, "lr": t["lr"]}, {"params": enc, "lr": t["lr_encoder"]}]
