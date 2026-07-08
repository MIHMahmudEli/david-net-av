"""Minimal YAML config loader → attribute-access namespace."""
from __future__ import annotations

from types import SimpleNamespace


def load_config(path: str) -> SimpleNamespace:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # sensible defaults so partial configs still run
    defaults = dict(
        seed=42, d_model=768, n_heads=8, n_fusion_layers=4, dropout=0.1,
        use_sync=True, use_disentangle=True, compose_quadrant=False,
        video_backbone="fallback", audio_backbone="fallback",
        video_model_name="MCG-NJU/videomae-base",
        audio_model_name="microsoft/wavlm-base-plus",
        freeze_blocks=6, freeze_feature_extractor=True,
        n_frames=16, audio_len=64000,
        batch_size=4, num_workers=0, epochs=1, lr=1e-4, weight_decay=1e-4,
        log_every=10, out_dir="runs", dry_run=False,
        qacp_temperature=0.1, init_from=None,
        feature_cache=None,   # dir of cached SSL features (Phase A); null = raw inputs
        modality_dropout=0.15,  # prob a training sample loses one stream (never both)
        shard_root=None, train_manifest="src/data/splits/train.jsonl",
        loss_weights=dict(v=1.0, a=1.0, quad=0.5, loc=0.5, sync=0.3, disentangle=0.1),
    )
    defaults.update(data or {})
    return SimpleNamespace(**defaults)
