"""The research CONFIG: defaults, deep merge, validation, and a stable semantic hash.

The notebook's first cell holds a CONFIG dict that overrides DEFAULT_CONFIG. Every
experiment stores its fully-resolved config (configs/config.json) plus the hash of its
*semantic* part -- the keys that can change a reported number. Hardware choices,
checkpoint cadence and paths are excluded from the hash: they may differ between
Kaggle sessions without making two runs different experiments.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    # -------------------------------------------------------------- project / storage
    "project": {
        "name": "DAVID-Net-AV",
        "hf_repo": "MoshinAli/davidnet-q1-experiments",   # persistent source of truth
        "hf_repo_type": "model",
        "hf_private": True,
        "code_repo": "https://github.com/MIHMahmudEli/david-net-av.git",
        "code_revision": "main",          # pin to a commit hash for the final runs
        "work_dir": "/kaggle/working/davidnet",   # small, persistent Kaggle output
        "scratch_dir": "/tmp/davidnet",           # large, per-session (features, caches)
    },
    # smoke: tiny subset, few steps, separate HF namespace -- proves the pipeline end to end
    # recovery_test: kill-and-resume test against the real HF repo (small, CPU-capable)
    # full: the paper
    "mode": "full",
    "seeds": [42, 123, 456],

    # -------------------------------------------------------------- data
    "data": {
        "primary": "fakeavceleb",
        "split_protocol": "strict_identity",   # strict_identity | legacy_source
        "split_seed": 42,                      # the split is fixed across training seeds
        # identity fractions. Under the strict protocol a fake survives only if all its
        # identities land together, so small splits shrink quadratically; 15% val gives
        # 515 val clips (70 real) instead of 309 at 10%, test (863 clips) is unchanged.
        "split_fractions": [0.65, 0.15, 0.2],
        "n_frames": 16,                        # model window: 16 frames over 4 s (4 fps)
        "window_seconds": 4.0,
        "sample_rate": 16000,
        "frame_size": 224,
        "cache_frames": 24,                    # decoded span: 24 frames over 6 s
        "cache_seconds": 6.0,
        "face_crop": {
            "enabled": True,                   # identical crop policy for EVERY dataset
            "detector": "yunet",               # yunet (OpenCV DNN) -> haar fallback
            "detect_frames": 4,                # frames sampled for detection per clip
            "box_scale": 1.8,                  # square crop side = scale x face box side
            "decode_max_side": 640,
        },
        # test-only corpora (zero-shot). Caps keep a session tractable; subsampling is
        # stratified by label (and generator where known) with a fixed seed and is
        # recorded in the manifest provenance.
        "cross_datasets": {
            "celeb-df-v2": {"use_official_test_list": True},
            "dfdc-10": {"max_real": 2500, "max_fake": 2500},
            "deepfaketimit": {},
            "asvspoof2019-la": {"partition": "eval"},
            "in-the-wild": {},
            "wavefake": {"max_per_generator": 1000},
        },
        "subsample_seed": 0,
    },

    # -------------------------------------------------------------- frozen features (Phase A)
    "features": {
        "video_model": "MCG-NJU/videomae-base",
        "audio_model": "microsoft/wavlm-base-plus",
        "video_spatial_grid": 4,       # 8 tubelets x 4x4 pooled patches = 128 video tokens
        "audio_tokens": 50,            # WavLM 20 ms frames pooled to 80 ms -> 50 tokens / 4 s
        "dtype": "float16",
        "train_view_offsets": [0, 4, 8],   # frame offsets into the 24-frame span (train)
        "eval_view_offset": 4,             # centered 16-frame window (val / test)
        "qacp_variants_per_clip": 4,       # pseudo-fake draws per real training clip
        "shard_clips": 512,              # AV corpora (~0.4 GB features + 0.23 GB clip cache)
        "shard_clips_audio": 4096,       # audio-only corpora (~0.3 GB); keeps commits/h low
        "extract_batch_size": "auto",
    },

    # -------------------------------------------------------------- model (DAVID-Net)
    "model": {
        "d_model": 768, "n_heads": 8, "n_fusion_layers": 4, "dropout": 0.1,
        "use_sync": True, "use_disentangle": True, "compose_quadrant": False,
    },

    # -------------------------------------------------------------- training
    "train": {
        "stage1": {
            "epochs": 30,
            "effective_batch": 64,
            "micro_batch": "auto",
            "optimizer": "adamw",
            "lr": 1.0e-4,
            "weight_decay": 0.05,
            "betas": [0.9, 0.999],
            "scheduler": "warmup_cosine",
            "warmup_ratio": 0.05,
            "min_lr_ratio": 0.01,
            "max_grad_norm": 1.0,
            "early_stopping_patience": 6,
            "selection_metric": "val/mean_auc",    # mean of video- and audio-AUC on VAL
            "sampler": "sqrt_balanced",            # uniform | sqrt_balanced | balanced
            "sampler_key": "quadrant",
            "modality_dropout": 0.15,
            "loss_weights": {"v": 1.0, "a": 1.0, "quad": 0.5, "loc": 0.5,
                             "sync": 0.1, "disentangle": 0.1},
            "init_from_qacp": True,
            "nonfinite_patience": 20,
        },
        "qacp": {
            "epochs": 40,
            "effective_batch": 256,
            "micro_batch": "auto",
            "lr": 3.0e-4,
            "weight_decay": 0.05,
            "betas": [0.9, 0.999],
            "scheduler": "warmup_cosine",
            "warmup_ratio": 0.05,
            "min_lr_ratio": 0.01,
            "max_grad_norm": 1.0,
            "early_stopping_patience": 8,
            "selection_metric": "val/qacp_loss",
            "temperature": 0.1,
            "items_per_epoch": 4096,       # pseudo-samples drawn per epoch (real clips are few)
            "pseudo_classes": ["RVRA", "RVFA", "FVRA", "FVFA", "MISMATCH"],
            "nonfinite_patience": 20,
        },
        "baseline": {
            "epochs": 30,
            "effective_batch": 64,
            "micro_batch": "auto",
            "lr": 1.0e-3,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "scheduler": "warmup_cosine",
            "warmup_ratio": 0.05,
            "min_lr_ratio": 0.01,
            "max_grad_norm": 1.0,
            "early_stopping_patience": 6,
            "selection_metric": "val/mean_auc",
            "sampler": "sqrt_balanced",
            "sampler_key": "quadrant",
            "nonfinite_patience": 20,
        },
        "phase_b": {                     # end-to-end partial fine-tune of the final model
            "epochs": 6,
            "effective_batch": 16,
            "micro_batch": "auto",
            "lr": 3.0e-5,                # fusion + heads (already trained in Phase A)
            "lr_encoder": 1.0e-5,
            "weight_decay": 0.05,
            "betas": [0.9, 0.999],
            "scheduler": "warmup_cosine",
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.01,
            "max_grad_norm": 1.0,
            "early_stopping_patience": 3,
            "selection_metric": "val/mean_auc",
            "sampler": "sqrt_balanced",
            "sampler_key": "quadrant",
            "modality_dropout": 0.15,
            "loss_weights": {"v": 1.0, "a": 1.0, "quad": 0.5, "loc": 0.5,
                             "sync": 0.1, "disentangle": 0.1},
            "unfreeze_top_blocks_video": 2,
            "unfreeze_top_blocks_audio": 2,
            "gradient_checkpointing": True,
            "temporal_augment": True,    # random 16-frame sub-window of the 24-frame span
            "nonfinite_patience": 20,
        },
    },

    # -------------------------------------------------------------- persistence
    "checkpoint": {
        "every_steps": 500,
        "every_minutes": 20.0,
        "keep_last": 2,                  # verified resume checkpoints kept on HF
        "prune_resume_state_on_completion": True,  # best_model/ is always kept
        "upload_retries": 6,
        "background_upload": True,
    },
    "session": {"time_budget_hours": 11.5, "safety_minutes": 25.0, "heartbeat_minutes": 10.0,
                # an experiment claimed by a session whose heartbeat is older than this is
                # considered abandoned (crashed container) and may be taken over
                "lease_minutes": 30.0,
                # UNIQUE per Kaggle account (e.g. "acct-main", "acct-2"): lets a restarted
                # session reclaim its own experiments immediately
                "worker_name": "kaggle-main"},
    "plan": {"groups": None, "only": None, "run_phase_b": True, "stop_on_error": True},
    "evaluation": {
        "threshold_policy": "val_eer",    # thresholds are fitted on VALIDATION only
        "bootstrap": 1000,
        "ece_bins": 15,
        "ci_level": 0.95,
    },
    "hardware": {"precision": "auto", "num_workers": "auto", "deterministic": True},
    "mixed_precision": True,
    "smoke": {"max_clips_per_split": 48, "epochs": 2, "every_steps": 3,
              "datasets": ["fakeavceleb", "in-the-wild"]},
}

# Top-level keys that never change a reported number (excluded from the hash).
_NON_SEMANTIC = {"project", "checkpoint", "session", "hardware", "smoke", "plan", "recovery"}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted: str, value) -> dict:
    """cfg['a']['b']['c'] = value for dotted='a.b.c' (creates intermediate dicts)."""
    cur = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return cfg


def get_dotted(cfg: dict, dotted: str, default=None):
    cur = cfg
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def build_config(user: dict | None = None) -> dict:
    cfg = deep_merge(DEFAULT_CONFIG, user or {})
    validate(cfg)
    return cfg


def validate(cfg: dict):
    errs = []
    if cfg["mode"] not in ("smoke", "recovery_test", "full"):
        errs.append(f"mode must be smoke|recovery_test|full, got {cfg['mode']!r}")
    fr = cfg["data"]["split_fractions"]
    if len(fr) != 3 or abs(sum(fr) - 1.0) > 1e-6 or min(fr) <= 0:
        errs.append(f"data.split_fractions must be 3 positive numbers summing to 1, got {fr}")
    if cfg["data"]["split_protocol"] not in ("strict_identity", "legacy_source"):
        errs.append("data.split_protocol must be strict_identity|legacy_source")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        errs.append("seeds must be a non-empty list of distinct integers")
    nf = cfg["data"]["n_frames"]
    for off in cfg["features"]["train_view_offsets"] + [cfg["features"]["eval_view_offset"]]:
        if off < 0 or off + nf > cfg["data"]["cache_frames"]:
            errs.append(f"view offset {off} + n_frames {nf} exceeds cache_frames")
    if cfg["model"]["d_model"] % cfg["model"]["n_heads"]:
        errs.append("model.d_model must be divisible by model.n_heads")
    for stage, s in cfg["train"].items():
        if s["effective_batch"] < 1 or s["lr"] <= 0:
            errs.append(f"train.{stage}: effective_batch>=1 and lr>0 required")
    if cfg["evaluation"]["threshold_policy"] not in ("val_eer", "val_youden", "fixed_0.5"):
        errs.append("evaluation.threshold_policy must be val_eer|val_youden|fixed_0.5")
    if errs:
        raise ValueError("invalid CONFIG:\n  - " + "\n  - ".join(errs))


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def semantic_part(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in _NON_SEMANTIC}


def config_hash(cfg: dict, n: int = 12) -> str:
    return hashlib.sha256(canonical_json(semantic_part(cfg)).encode()).hexdigest()[:n]


def flatten(d: dict, prefix: str = "") -> dict:
    """{'a': {'b': 1}} -> {'a.b': 1}; used for the hyperparameter table."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out
