"""Smoke tests: model shapes + one training step + metrics. Run: pytest -q"""
import torch

from src.models.david_net import DavidNet, DavidNetConfig
from src.models.video_encoder import ConvFallbackVideoEncoder
from src.models.audio_encoder import SpecCNNFallbackAudioEncoder
from src.training.losses import LossWeights, total_loss
from src.eval.metrics import per_modality, quadrant_metrics


def _model():
    cfg = DavidNetConfig()
    return DavidNet(cfg, ConvFallbackVideoEncoder(cfg.d_model),
                    SpecCNNFallbackAudioEncoder(cfg.d_model))


def test_forward_shapes():
    model = _model()
    B = 2
    video = torch.randn(B, 8, 3, 64, 64)      # small for speed
    audio = torch.randn(B, 16000)
    out = model(video, audio)
    assert out["logit_v"].shape == (B,)
    assert out["logit_a"].shape == (B,)
    assert out["logit_quad"].shape == (B, 4)


def test_training_step():
    model = _model()
    B = 2
    batch = {
        "video": torch.randn(B, 8, 3, 64, 64),
        "audio": torch.randn(B, 16000),
        "video_label": torch.tensor([0, 1]),
        "audio_label": torch.tensor([1, 1]),
        "quadrant": torch.tensor([1, 3]),
        "video_seg_mask": torch.zeros(B, 8),
        "audio_seg_mask": torch.zeros(B, 100),
    }
    out = model(batch["video"], batch["audio"])
    loss, parts = total_loss(out, batch, LossWeights(), model=model)
    loss.backward()
    assert torch.isfinite(loss)


def test_supcon_and_qacp():
    from src.training.losses import supcon_loss, qacp_loss
    feats = torch.randn(8, 32, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    loss = supcon_loss(feats, labels)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    # identical features with same labels → lower loss than random
    same = torch.ones(4, 16)
    labels2 = torch.tensor([0, 0, 1, 1])
    assert torch.isfinite(supcon_loss(same, labels2))


def test_synthetic_quadrants():
    from src.data.synthetic_quadrants import build_pseudo_sample, QACP_CLASSES
    frames = torch.rand(8, 3, 64, 64)
    wave = torch.randn(16000)
    donor = torch.randn(16000)
    for cls in QACP_CLASSES:
        s = build_pseudo_sample(frames, wave, donor_wave=donor, pseudo_class=cls)
        assert s["video"].shape == frames.shape
        assert s["audio"].shape == wave.shape
        if cls == "MISMATCH":  # mismatch is NOT fake
            assert s["video_label"].item() == 0 and s["audio_label"].item() == 0
            assert s["sync_label"].item() == 1
        if cls == "FVFA":
            assert s["video_label"].item() == 1 and s["audio_label"].item() == 1
    # transforms actually change the content
    rvfa = build_pseudo_sample(frames, wave, pseudo_class="RVFA")
    assert not torch.allclose(rvfa["audio"], wave)
    fvra = build_pseudo_sample(frames, wave, pseudo_class="FVRA")
    assert not torch.allclose(fvra["video"], frames)


def test_missing_modality():
    """Audio-only and silent-video inputs: z_c zeroed, losses masked, grads flow."""
    from src.training.losses import LossWeights, total_loss

    model = _model()
    B = 4
    video = torch.randn(B, 8, 3, 64, 64)
    audio = torch.randn(B, 16000)

    # audio-only (video absent for all samples)
    out = model(video, audio, v_avail=torch.zeros(B), a_avail=torch.ones(B))
    assert torch.allclose(out["z_c"], torch.zeros_like(out["z_c"]))  # no cross-modal evidence
    assert out["logit_a"].shape == (B,)

    # mixed batch: sample 0 loses video, sample 1 loses audio, rest full
    v_av = torch.tensor([0.0, 1.0, 1.0, 1.0])
    a_av = torch.tensor([1.0, 0.0, 1.0, 1.0])
    out = model(video, audio, v_avail=v_av, a_avail=a_av)
    assert torch.allclose(out["z_c"][0], torch.zeros_like(out["z_c"][0]))
    assert not torch.allclose(out["z_c"][2], torch.zeros_like(out["z_c"][2]))

    batch = {
        "video": video, "audio": audio,
        "video_label": torch.tensor([0, 1, 0, 1]),
        "audio_label": torch.tensor([1, 0, 1, 0]),
        "quadrant": torch.tensor([1, 2, 1, 2]),
        "video_seg_mask": torch.zeros(B, 8),
        "audio_seg_mask": torch.zeros(B, 100),
    }
    loss, parts = total_loss(out, batch, LossWeights(), model=model)
    assert torch.isfinite(loss)
    loss.backward()


def test_modality_dropout_sampler():
    from src.training.train import sample_modality_masks
    v_av, a_av = sample_modality_masks(256, 0.5, "cpu")
    # never both dropped
    assert ((v_av == 0) & (a_av == 0)).sum() == 0
    # some drops actually happened at p=0.5
    assert (v_av == 0).sum() + (a_av == 0).sum() > 0
    assert sample_modality_masks(8, 0.0, "cpu") == (None, None)


def test_feature_cache_roundtrip(tmp_path):
    """extract_features -> CachedFeatureDataset -> identity-encoder model forward."""
    from types import SimpleNamespace
    from src.data.extract_features import extract
    from src.data.datasets import CachedFeatureDataset, collate
    from src.training.train import build_model

    cfg = SimpleNamespace(
        d_model=768, n_heads=8, n_fusion_layers=1, dropout=0.0,
        use_sync=True, use_disentangle=True, compose_quadrant=False,
        video_backbone="fallback", audio_backbone="fallback",
        n_frames=4, audio_len=16000, shard_root=None, num_workers=0,
        feature_cache=None,
    )
    manifest = "src/data/splits/train.jsonl"
    cache = str(tmp_path / "feats")
    extract(cfg, manifest, cache, batch_size=4)

    ds = CachedFeatureDataset(manifest, cache, cfg.n_frames, cfg.audio_len)
    batch = collate([ds[i] for i in range(2)])
    assert batch["video"].dim() == 3  # (B, L_v, d) features, not raw frames

    cfg.feature_cache = cache
    model = build_model(cfg)          # identity encoders
    out = model(batch["video"], batch["audio"])
    assert out["logit_v"].shape == (2,)
    assert out["logit_quad"].shape == (2, 4)


def test_metrics():
    m = per_modality([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert 0.0 <= m["auc"] <= 1.0
    q = quadrant_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    assert q["acc"] == 1.0
