"""A wedged ffmpeg must raise, not block forever.

This is the bug that cost three Kaggle sessions: `subprocess.run` had no `timeout`, so one
undecodable clip stopped the training process dead. It stayed alive with 29 GB RAM free,
the GPU fell idle, and Kaggle reclaimed the session as an idle accelerator -- reported as
CANCEL_ACKNOWLEDGED with no traceback, which looks exactly like a manual stop.
"""
import subprocess

import pytest

from src.data import decode
from src.data.decode import DecodeError


def _hang(*args, **kwargs):
    """Stand in for an ffmpeg that never returns."""
    assert "timeout" in kwargs, "call site passes no timeout — it can hang forever"
    raise subprocess.TimeoutExpired(cmd=args[0] if args else "ffmpeg",
                                    timeout=kwargs["timeout"])


def test_video_decode_timeout_raises_and_names_the_clip(monkeypatch):
    monkeypatch.setattr(decode.subprocess, "run", _hang)
    with pytest.raises(DecodeError) as e:
        decode._video_ffmpeg("/data/wedged_clip.mp4", start=0.0, win=4.0,
                             n_frames=16, size=224)
    msg = str(e.value)
    assert "timed out" in msg
    assert "wedged_clip.mp4" in msg, "the message must identify the offending file"


def test_audio_decode_timeout_raises_and_names_the_clip(monkeypatch):
    monkeypatch.setattr(decode.subprocess, "run", _hang)
    with pytest.raises(DecodeError) as e:
        decode._audio_ffmpeg("/data/wedged_clip.mp4", start=0.0, win=4.0)
    msg = str(e.value)
    assert "timed out" in msg
    assert "wedged_clip.mp4" in msg


def test_ffprobe_timeout_is_survivable(monkeypatch):
    """_ffprobe already swallows failures and returns None; it must not hang either."""
    monkeypatch.setattr(decode.subprocess, "run", _hang)
    assert decode._ffprobe("/data/wedged_clip.mp4") is None


def test_every_subprocess_call_in_decode_passes_a_timeout():
    """Guard against a future call site reintroducing an unbounded wait."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(decode))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", None) or getattr(fn, "id", None)
        if name != "run":
            continue
        if not any(kw.arg == "timeout" for kw in node.keywords):
            offenders.append(node.lineno)
    assert not offenders, f"subprocess.run without timeout at decode.py lines {offenders}"


def test_timeouts_are_configurable_and_sane():
    assert 0 < decode.FFPROBE_TIMEOUT_S <= decode.FFMPEG_TIMEOUT_S
    assert decode.FFMPEG_TIMEOUT_S >= 10, "too tight: a slow seek on a big clip is normal"
