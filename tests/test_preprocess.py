"""Preprocessing pipeline tests.

Pure-python geometry/audio units run everywhere; the ffmpeg integration test
builds a synthetic clip and auto-skips when ffmpeg is not on PATH.
"""
import shutil
import subprocess

import numpy as np
import pytest
import torch

from src.data.preprocess import (
    FaceCropper, face_and_mouth_crops, rms_normalize, uniform_indices,
    _square_crop, FFMPEG,
)


def test_uniform_indices():
    idx = uniform_indices(100, 16)
    assert len(idx) == 16 and idx[0] == 0 and idx[-1] == 99
    # short clip: repeats indices instead of failing
    idx = uniform_indices(3, 8)
    assert len(idx) == 8 and idx.max() == 2
    with pytest.raises(ValueError):
        uniform_indices(0, 8)


def test_rms_normalize():
    loud = torch.randn(16000) * 0.5
    quiet = torch.randn(16000) * 0.001
    for w in (loud, quiet):
        out = rms_normalize(w, target_rms=0.05)
        assert abs(out.pow(2).mean().sqrt().item() - 0.05) < 5e-3
    assert rms_normalize(torch.zeros(0)).numel() == 0  # empty (no audio track)


def test_square_crop_padding():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    # crop hanging off the top-left corner: padded region must be zeros
    out = _square_crop(frame, cx=0, cy=0, size=50)
    assert out.shape == (50, 50, 3)
    assert out[0, 0].sum() == 0          # padded corner
    assert out[-1, -1].sum() == 765      # real pixels


def test_face_crops_fallback_backend():
    """With no face libs installed, the center-fallback still produces valid crops."""
    cropper = FaceCropper()
    frames = np.random.randint(0, 255, (4, 120, 160, 3), dtype=np.uint8)
    face, mouth = face_and_mouth_crops(frames, cropper, face_size=64, mouth_size=32)
    assert face.shape == (4, 3, 64, 64)
    assert mouth.shape == (4, 3, 32, 32)
    assert face.min() >= 0 and face.max() <= 1
    assert cropper.backend in ("insightface", "mediapipe", "opencv-haar", "center-fallback")


def test_tracking_smoothness():
    """EMA tracking: bbox must not jump even with noisy per-frame detections."""
    cropper = FaceCropper(ema=0.8)
    frames = np.random.randint(0, 255, (6, 100, 100, 3), dtype=np.uint8)
    boxes = [cropper.track(f)[0] for f in frames]
    for b0, b1 in zip(boxes, boxes[1:]):
        jump = max(abs(a - b) for a, b in zip(b0, b1))
        assert jump < 20  # smoothed track cannot teleport


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_full_pipeline_synthetic_clip(tmp_path):
    """End-to-end on a generated test clip (2 s color bars + 440 Hz tone)."""
    from src.data.preprocess import preprocess_clip, probe_media
    clip = str(tmp_path / "test.mp4")
    subprocess.run(
        [FFMPEG, "-v", "error",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-shortest", "-pix_fmt", "yuv420p", clip],
        check=True)

    meta = probe_media(clip)
    assert meta["has_video"] and meta["has_audio"]
    assert 1.5 < meta["duration_sec"] < 2.5

    r = preprocess_clip(clip, n_frames=8, audio_len=32000)
    assert r["video"].shape == (8, 3, 224, 224)
    assert r["mouth"].shape == (8, 3, 96, 96)
    assert r["audio"].shape == (32000,)
    assert r["audio"].abs().sum() > 0        # tone was actually extracted
    assert r["meta"]["face_backend"] == r["meta"]["face_backend"]  # recorded
