"""Tests for baselines, robustness degradations, and figure generation."""
import json
from pathlib import Path

import torch

from src.baselines.models import build_baseline, REGISTRY


def test_baseline_registry():
    assert {"video-framecnn", "audio-speccnn"} <= set(REGISTRY)


def test_baselines_forward_backward():
    batch = {
        "video": torch.randn(2, 4, 3, 64, 64),
        "audio": torch.randn(2, 16000),
        "video_label": torch.tensor([0, 1]),
        "audio_label": torch.tensor([1, 0]),
    }
    for name in ("video-framecnn", "audio-speccnn"):
        model = build_baseline(name, width=16)
        logits = model(batch)
        assert logits.shape == (2,)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch[f"{model.modality}_label"].float())
        loss.backward()
        assert torch.isfinite(loss)


def test_degradations_shapes():
    from src.eval.robustness import video_blur, video_downscale, video_quantize, audio_noise
    v = torch.rand(1, 4, 3, 32, 32)
    a = torch.randn(2, 16000)
    assert video_blur(v, 2.0).shape == v.shape
    assert video_downscale(v, 4).shape == v.shape
    assert video_quantize(v, 8).shape == v.shape
    # quantization actually reduces distinct values
    assert len(video_quantize(v, 8).unique()) <= 9
    # SNR: noisier signal has larger deviation from original
    n0 = (audio_noise(a, 20) - a).pow(2).mean()
    n1 = (audio_noise(a, 0) - a).pow(2).mean()
    assert n1 > n0
    assert torch.allclose(audio_noise(a, 100), a)  # 100 dB = passthrough


def test_figures_demo(tmp_path):
    """Demo figure generation produces every expected PDF+PNG pair."""
    from src.eval.figures import generate_all
    out = tmp_path / "figs"
    generate_all(results_dir=None, out=str(out), demo=True)
    expected = ["results_roc", "results_reliability", "results_confusion",
                "results_robustness", "results_ablation", "results_localization"]
    for name in expected:
        assert (out / f"{name}.pdf").exists(), name
        assert (out / f"{name}.png").exists(), name


def test_figures_from_results_dir(tmp_path):
    """Figure generator consumes real evaluate.py-schema JSONs from a dir."""
    from src.eval.figures import generate_all, _demo_reports
    main, _, rob, ablation = _demo_reports()
    rdir = tmp_path / "results"
    rdir.mkdir()
    (rdir / "david-net_seed42.json").write_text(json.dumps(main), encoding="utf-8")
    (rdir / "robustness_david-net.json").write_text(json.dumps(rob), encoding="utf-8")
    (rdir / "ablation.json").write_text(json.dumps(ablation), encoding="utf-8")
    out = tmp_path / "figs"
    generate_all(results_dir=str(rdir), out=str(out), demo=False)
    assert (out / "results_roc.pdf").exists()
    assert (out / "results_robustness.pdf").exists()
    assert (out / "results_ablation.pdf").exists()


def test_ablation_config_generation(tmp_path):
    """run_experiments produces valid override configs for every ablation."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import yaml
    from run_experiments import ABLATIONS, make_config

    base = yaml.safe_load(Path("configs/david_net.yaml").read_text(encoding="utf-8"))
    for name, overrides in ABLATIONS.items():
        path, skip_qacp = make_config(base, overrides, seed=42,
                                      out_dir=tmp_path, name=name)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["seed"] == 42
        if name == "no_sync":
            assert cfg["use_sync"] is False
        if name == "no_loc":
            assert cfg["loss_weights"]["loc"] == 0.0
        if name == "no_qacp":
            assert skip_qacp
        if name == "no_moddrop":
            assert cfg["modality_dropout"] == 0.0
