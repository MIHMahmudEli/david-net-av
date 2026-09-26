"""In-memory frozen-feature store and the Phase-A datasets built on it.

Only the rows a run needs are kept in RAM (the full 3-view FakeAVCeleb set is ~17 GB
in fp16; a strict-protocol run needs ~7 GB). Every random choice (which view of a
clip, which pseudo-quadrant, which donor) is a pure function of (seed, epoch, index),
so a resumed run replays exactly the same samples it would have seen uninterrupted.
"""
from __future__ import annotations

import json
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.pipeline.env import log_event

QUAD_IDX = {"RVRA": 0, "RVFA": 1, "FVRA": 2, "FVFA": 3}
QUAD_NAMES = ["RVRA", "RVFA", "FVRA", "FVFA"]


def _h(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


class FeatureTable:
    """Rows of (video, audio, v_avail, a_avail) + clip_id -> {view: row}."""

    def __init__(self):
        self.video = self.audio = self.v_avail = self.a_avail = None
        self.rows: dict[str, dict] = {}
        self.n = 0

    def __len__(self):
        return self.n

    @classmethod
    def load(cls, shard_paths: list[tuple[Path, Path]], keep: Optional[set] = None,
             views: Optional[set] = None) -> "FeatureTable":
        from safetensors.torch import load_file
        t = cls()
        parts = defaultdict(list)
        n = 0
        for st, js in shard_paths:
            index = json.loads(Path(js).read_text(encoding="utf-8"))
            sel = [i for i, e in enumerate(index)
                   if (keep is None or e["clip_id"] in keep)
                   and (views is None or e.get("view") in views or "variant" in e)]
            if not sel:
                continue
            tens = load_file(str(st))
            idx = torch.tensor(sel)
            for k, v in tens.items():
                parts[k].append(v.index_select(0, idx) if v.shape[0] == len(index) else v[idx])
            for j, i in enumerate(sel):
                e = index[i]
                key = e.get("view", e.get("variant"))
                t.rows.setdefault(e["clip_id"], {})[key] = n + j
            n += len(sel)
            del tens
        for k, vs in parts.items():
            setattr(t, k, torch.cat(vs))
        t.n = n
        return t


class FeatureStore:
    """Downloads feature shards from the data repo (once per session) and serves tables."""

    def __init__(self, data_store, fsid: str, scratch: str | Path):
        self.ds = data_store
        self.fsid = fsid
        self.scratch = Path(scratch)

    def shard_paths(self, corpus: str, qacp: bool = False) -> list[tuple[Path, Path]]:
        prefix = f"features/{self.fsid}/{'qacp/' if qacp else ''}{corpus}"
        local = self.scratch / prefix
        if self.ds is not None:
            items = self.ds.list_files(prefix, recursive=False)
            names = sorted({Path(i.path).stem for i in items if i.path.endswith(".json")})
            for nm in names:
                for ext in (".json", ".safetensors"):
                    dst = local / f"{nm}{ext}"
                    if not dst.exists():
                        self.ds.download(f"{prefix}/{nm}{ext}", self.scratch)
        names = sorted(p.stem for p in local.glob("shard_*.json"))
        return [(local / f"{nm}.safetensors", local / f"{nm}.json") for nm in names]

    def table(self, corpus: str, keep: Optional[set] = None, views: Optional[set] = None,
              qacp: bool = False) -> FeatureTable:
        t = FeatureTable.load(self.shard_paths(corpus, qacp), keep, views)
        log_event("dataset_loaded", f"{corpus}{' (qacp)' if qacp else ''}: {len(t)} rows, "
                  f"{len(t.rows)} clips", rows=len(t), clips=len(t.rows))
        return t


def _seg_mask(segments, length: int) -> torch.Tensor:
    m = torch.zeros(length)
    for s, e in segments or []:
        if e >= 9999.0 or s <= 0 and e >= 4.0:
            m[:] = 1.0
        else:
            m[int(length * s / 4.0):max(int(length * s / 4.0) + 1, int(length * e / 4.0))] = 1.0
    return m


class FeatureDataset(Dataset):
    """Supervised Stage-1 / evaluation samples from a FeatureTable.

    train=True: one random view per clip per epoch (deterministic in seed, epoch).
    train=False: the centred evaluation view.
    """

    def __init__(self, records: list[dict], table: FeatureTable, train: bool, seed: int,
                 eval_view: int = 4):
        missing = [r["clip_id"] for r in records if r["clip_id"] not in table.rows]
        self.records = [r for r in records if r["clip_id"] in table.rows]
        self.missing = missing
        if missing:
            log_event("features_missing", f"{len(missing)} clips have no features "
                      f"(decode failures); excluded and reported", n=len(missing))
        self.t = table
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.eval_view = eval_view

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def _row(self, i: int) -> int:
        views = self.t.rows[self.records[i]["clip_id"]]
        if self.train and len(views) > 1:
            keys = sorted(k for k in views if k is not None)
            return views[keys[_h(self.seed, self.epoch, i) % len(keys)]]
        return views.get(self.eval_view, next(iter(views.values())))

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        row = self._row(i)
        v, a = self.t.video[row].float(), self.t.audio[row].float()
        vl, al = r.get("video_label", -1), r.get("audio_label", -1)
        q = r.get("quadrant")
        return {
            "idx": i, "clip_id": r["clip_id"], "video": v, "audio": a,
            "v_avail": self.t.v_avail[row].float(), "a_avail": self.t.a_avail[row].float(),
            "video_label": torch.tensor(vl), "audio_label": torch.tensor(al),
            "clip_label": torch.tensor(r.get("clip_label", int(vl == 1 or al == 1))),
            "quadrant": torch.tensor(QUAD_IDX[q] if q in QUAD_IDX else -1),
            "video_seg_mask": _seg_mask(r.get("video_segments"), v.shape[0]),
            "audio_seg_mask": _seg_mask(r.get("audio_segments"), a.shape[0]),
            "generator": r.get("generator", ""), "dataset": r.get("dataset", ""),
            "race": (r.get("meta") or {}).get("race", ""),
            "gender": (r.get("meta") or {}).get("gender", ""),
        }


class QACPFeatureDataset(Dataset):
    """Pseudo-quadrant samples from REAL clips only (QACP Stage 0, feature space).

    classes: subset of RVRA, RVFA, FVRA, FVFA, MISMATCH (ablations remove some).
    Every item uses the centred view for real AND pseudo-fake streams, so the
    window a feature was cut from cannot serve as a shortcut label.
    """

    SYNC_MATCHED, SYNC_MISMATCHED = 0, 1

    def __init__(self, real_records: list[dict], table: FeatureTable, qtable: FeatureTable,
                 classes: list[str], seed: int, items_per_epoch: int, eval_view: int = 4):
        self.recs = [r for r in real_records if r["clip_id"] in table.rows
                     and r["clip_id"] in qtable.rows]
        if len(self.recs) < 2:
            raise ValueError("QACP needs at least two real clips with pseudo variants")
        self.t, self.q = table, qtable
        self.classes = list(classes)
        self.seed, self.epoch = seed, 0
        self.n = items_per_epoch
        self.eval_view = eval_view

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def _real(self, clip_id: str) -> int:
        views = self.t.rows[clip_id]
        return views.get(self.eval_view, next(iter(views.values())))

    def __getitem__(self, i: int) -> dict:
        h = _h(self.seed, self.epoch, i)
        r = self.recs[h % len(self.recs)]
        cls = self.classes[(h // 7919) % len(self.classes)]
        variants = self.q.rows[r["clip_id"]]
        k = sorted(variants)[(h // 104729) % len(variants)]
        qrow = variants[k]
        row = self._real(r["clip_id"])
        v, a = self.t.video[row], self.t.audio[row]
        vl = al = 0
        sync = self.SYNC_MATCHED
        if cls in ("RVFA", "FVFA"):
            a, al = self.q.gl_audio[qrow], 1
        if cls in ("FVRA", "FVFA"):
            v, vl = self.q.sb_video[qrow], 1
        if cls == "MISMATCH":
            j = (h // 15485863) % (len(self.recs) - 1)
            donor = self.recs[j if self.recs[j]["clip_id"] != r["clip_id"] else -1]
            a = self.t.audio[self._real(donor["clip_id"])]
            sync = self.SYNC_MISMATCHED
        return {"video": v.float(), "audio": a.float(), "video_label": torch.tensor(vl),
                "audio_label": torch.tensor(al), "sync_label": torch.tensor(sync),
                "v_avail": torch.tensor(1.0), "a_avail": torch.tensor(1.0),
                "pseudo_class": cls}


def collate(batch: list[dict]) -> dict:
    out = {}
    for k, v in batch[0].items():
        if torch.is_tensor(v):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out


class EpochSampler(torch.utils.data.Sampler):
    """Deterministic per-epoch index order; optional class re-weighting.

    mode: uniform (permutation) | sqrt_balanced (weights ~ n_c^-1/2) | balanced (~ n_c^-1)
    The epoch length equals the dataset length. `skip` drops the first k indices, which
    is how a resumed run continues mid-epoch at the exact next batch.
    """

    def __init__(self, keys: list, mode: str, seed: int):
        self.keys, self.mode, self.seed = keys, mode, seed
        self.epoch, self.skip = 0, 0
        counts = defaultdict(int)
        for k in keys:
            counts[k] += 1
        power = {"uniform": 0.0, "sqrt_balanced": 0.5, "balanced": 1.0}[mode]
        self.weights = torch.tensor([counts[k] ** -power for k in keys], dtype=torch.double)

    def set_epoch(self, epoch: int, skip: int = 0):
        self.epoch, self.skip = epoch, skip

    def order(self) -> list[int]:
        g = torch.Generator().manual_seed(_h(self.seed, self.epoch, "sampler"))
        n = len(self.keys)
        if self.mode == "uniform":
            idx = torch.randperm(n, generator=g)
        else:
            idx = torch.multinomial(self.weights, n, replacement=True, generator=g)
        return idx.tolist()

    def __iter__(self):
        return iter(self.order()[self.skip:])

    def __len__(self):
        return len(self.keys) - self.skip
