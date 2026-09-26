"""Synthetic feature tables with a known, learnable signal.

Used by the unit tests and by CONFIG['mode'] == 'recovery_test', which exercises the
full save -> upload -> kill -> new process -> discover -> restore path against the
real Hugging Face repo without needing a GPU or the datasets.
"""
from __future__ import annotations

import torch

from src.pipeline.features import FeatureTable, QUAD_NAMES


def synthetic_records(n: int, seed: int = 0, prefix: str = "syn") -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    quads = torch.randint(0, 4, (n,), generator=g).tolist()
    recs = []
    for i, q in enumerate(quads):
        name = QUAD_NAMES[q]
        v, a = int(name[0] == "F"), int(name[2] == "F")
        recs.append({"clip_id": f"{prefix}_{i:05d}", "rel_path": f"{prefix}/{i}.mp4",
                     "video_label": v, "audio_label": a, "clip_label": int(v or a),
                     "quadrant": name, "generator": "real" if not (v or a) else "synthetic",
                     "dataset": "synthetic", "identity": f"id{i:05d}",
                     "meta": {"race": ["A", "B"][i % 2], "gender": ["men", "women"][i % 2]},
                     "video_segments": [[0.0, 9999.0]] if v else [],
                     "audio_segments": [[0.0, 9999.0]] if a else [], "modalities": "av"})
    return recs


def synthetic_table(records: list[dict], Lv: int = 8, La: int = 6, d: int = 32,
                    views=(0, 4, 8), strength: float = 0.8, seed: int = 0) -> FeatureTable:
    g = torch.Generator().manual_seed(seed + 1)
    dir_v = torch.randn(d, generator=g)
    dir_a = torch.randn(d, generator=g)
    rows_v, rows_a, t = [], [], FeatureTable()
    n = 0
    for r in records:
        for view in views:
            v = torch.randn(Lv, d, generator=g)
            a = torch.randn(La, d, generator=g)
            v += strength * r["video_label"] * dir_v
            a += strength * r["audio_label"] * dir_a
            rows_v.append(v)
            rows_a.append(a)
            t.rows.setdefault(r["clip_id"], {})[view] = n
            n += 1
    t.video = torch.stack(rows_v).half()
    t.audio = torch.stack(rows_a).half()
    t.v_avail = torch.ones(n, dtype=torch.float16)
    t.a_avail = torch.ones(n, dtype=torch.float16)
    t.n = n
    return t


def tiny_config(cfg: dict, d: int = 32) -> dict:
    """Shrink the model and schedule so a CPU run takes seconds."""
    from src.pipeline.config import deep_merge
    return deep_merge(cfg, {
        "model": {"d_model": d, "n_heads": 2, "n_fusion_layers": 1, "dropout": 0.0},
        "train": {"stage1": {"epochs": 3, "effective_batch": 16, "micro_batch": 8,
                             "early_stopping_patience": 10}},
        "checkpoint": {"every_steps": 2, "every_minutes": 999.0, "keep_last": 2,
                       "background_upload": False},
        "hardware": {"precision": "fp32", "num_workers": 0},
    })


def synthetic_qacp_table(real_records: list[dict], table: FeatureTable, variants: int = 4,
                         strength: float = 0.8, seed: int = 0) -> FeatureTable:
    """Pseudo-fake variants of real clips: video/audio shifted along a fixed direction."""
    g = torch.Generator().manual_seed(seed + 7)
    d = table.video.shape[-1]
    dir_v, dir_a = torch.randn(d, generator=g), torch.randn(d, generator=g)
    q, sb, gl, n = FeatureTable(), [], [], 0
    for r in real_records:
        row = table.rows[r["clip_id"]][sorted(table.rows[r["clip_id"]])[len(table.rows[r["clip_id"]]) // 2]]
        for k in range(variants):
            sb.append(table.video[row].float() + strength * dir_v + 0.1 * torch.randn(table.video.shape[1:], generator=g))
            gl.append(table.audio[row].float() + strength * dir_a + 0.1 * torch.randn(table.audio.shape[1:], generator=g))
            q.rows.setdefault(r["clip_id"], {})[k] = n
            n += 1
    q.sb_video, q.gl_audio, q.n = torch.stack(sb).half(), torch.stack(gl).half(), n
    return q


def recovery_config(cfg: dict) -> dict:
    """Small model + schedule for mode='recovery_test' (runs on CPU in minutes)."""
    from src.pipeline.config import deep_merge
    tiny = {"epochs": 3, "effective_batch": 16, "micro_batch": 8, "early_stopping_patience": 99}
    return deep_merge(cfg, {
        "seeds": cfg["seeds"][:1],
        "model": {"d_model": 32, "n_heads": 2, "n_fusion_layers": 1, "dropout": 0.0},
        "train": {"stage1": tiny, "baseline": tiny,
                  "qacp": dict(tiny, effective_batch=32, micro_batch=32, items_per_epoch=64)},
        "checkpoint": {"every_steps": 6, "every_minutes": 999.0, "keep_last": 2},
        "evaluation": {"bootstrap": 50},
    })
