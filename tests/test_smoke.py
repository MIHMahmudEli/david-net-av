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


def test_metrics():
    m = per_modality([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert 0.0 <= m["auc"] <= 1.0
    q = quadrant_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    assert q["acc"] == 1.0
