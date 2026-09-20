"""Decode video/audio from media files for DAVID-Net training and inference.

Design rules (learned from the first Kaggle run, which silently trained on noise):
  * Decoding NEVER falls back to random tensors. Any failure raises `DecodeError`
    so a mis-configured root path or a broken dependency is caught at step 0,
    not after 20 GPU-hours.
  * Video and audio are taken from the SAME temporal window so the sync module
    sees genuinely aligned streams (the old path sampled 16 frames over the whole
    clip and a random 4 s audio crop — never aligned).
  * ffmpeg (present on Kaggle / DGX / any NGC image) is the primary backend; a
    cv2 + torchaudio path covers machines without it.

Returned tensors:  video (T, 3, H, W) float32 in [0, 1] RGB,  audio (N,) float32
at 16 kHz mono.  Audio-only files (wav/flac/mp3/ogg) return `has_video=False`;
silent videos return `has_audio=False` (the model substitutes learnable null
tokens for absent streams).
"""
from __future__ import annotations

import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

AUDIO_ONLY_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
SAMPLE_RATE = 16000


class DecodeError(RuntimeError):
    """Raised when a clip cannot be decoded. Deliberately NOT swallowed."""


@dataclass
class Decoded:
    video: torch.Tensor          # (T, 3, H, W) in [0,1]; zeros when has_video=False
    audio: torch.Tensor          # (N,) float32 16 kHz; zeros when has_audio=False
    has_video: bool
    has_audio: bool
    window: tuple[float, float]  # (start_sec, length_sec) actually decoded


# ------------------------------------------------------------------ probing
def probe(path: str) -> dict:
    """{duration, fps, n_frames, has_video, has_audio} using cv2 (fast) or ffprobe."""
    p = Path(path)
    if p.suffix.lower() in AUDIO_ONLY_EXTS:
        return {"duration": _audio_duration(path), "fps": 0.0, "n_frames": 0,
                "has_video": False, "has_audio": True}
    info = None
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            cap.release()
            if n > 0 and fps > 0:
                info = {"duration": n / fps, "fps": fps, "n_frames": n, "has_video": True}
    except Exception:
        info = None
    if info is None and FFPROBE is not None:
        info = _ffprobe(path)
    if info is None:
        raise DecodeError(f"cannot probe {path} (cv2 and ffprobe both failed)")
    # has_audio is only knowable through ffprobe; assume present for A/V containers
    # and let the audio decoder downgrade to has_audio=False on an empty stream.
    info.setdefault("has_audio", True)
    return info


def _ffprobe(path: str) -> dict | None:
    import json
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-print_format", "json", "-show_format",
             "-show_streams", str(path)], capture_output=True, text=True, check=True).stdout
        info = json.loads(out)
    except Exception:
        return None
    streams = info.get("streams", [])
    v = [s for s in streams if s.get("codec_type") == "video"]
    a = [s for s in streams if s.get("codec_type") == "audio"]
    dur = float(info.get("format", {}).get("duration", 0.0) or 0.0)
    fps = 0.0
    if v:
        num, _, den = (v[0].get("avg_frame_rate", "0/1")).partition("/")
        try:
            fps = float(num) / float(den) if float(den) else 0.0
        except ValueError:
            fps = 0.0
    return {"duration": dur, "fps": fps, "n_frames": int(dur * fps) if fps else 0,
            "has_video": bool(v), "has_audio": bool(a)}


def _audio_duration(path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        pass
    if FFPROBE is not None:
        info = _ffprobe(path)
        if info:
            return info["duration"]
    return 0.0


# ------------------------------------------------------------------ window
def choose_window(duration: float, win_sec: float, mode: str = "random") -> float:
    """Start time of a `win_sec` window inside a clip of `duration` seconds."""
    if duration <= win_sec or duration <= 0:
        return 0.0
    if mode == "center":
        return (duration - win_sec) / 2.0
    return random.uniform(0.0, duration - win_sec)


# ------------------------------------------------------------------ video
def _video_ffmpeg(path: str, start: float, win: float, n_frames: int, size: int) -> torch.Tensor:
    # `fps=n/win` spreads n frames uniformly over the window; -ss before -i seeks
    # accurately (ffmpeg decodes from the previous keyframe and discards).
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{win:.3f}",
           "-i", str(path), "-vf", f"fps={n_frames / win:.6f},scale={size}:{size}",
           "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise DecodeError(f"ffmpeg video decode failed for {path}: {res.stderr.decode(errors='ignore')[:300]}")
    raw = res.stdout
    frame_bytes = size * size * 3
    n = len(raw) // frame_bytes
    if n == 0:
        raise DecodeError(f"ffmpeg produced no frames for {path} (window {start:.2f}+{win:.2f}s)")
    arr = np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8).reshape(n, size, size, 3)
    frames = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).float() / 255.0
    return _pad_frames(frames, n_frames)


def _video_cv2(path: str, start: float, win: float, n_frames: int, size: int,
               fps: float, total: int) -> torch.Tensor:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DecodeError(f"cv2 cannot open {path}")
    f0 = int(start * fps)
    f1 = min(total - 1, int((start + win) * fps) - 1)
    f1 = max(f0, f1)
    idx = np.linspace(f0, f1, n_frames).round().astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            if frames:
                frames.append(frames[-1].clone())
                continue
            cap.release()
            raise DecodeError(f"cv2 failed to read frame {i} of {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()
    return torch.stack(frames)


def _pad_frames(frames: torch.Tensor, n: int) -> torch.Tensor:
    if frames.size(0) >= n:
        return frames[:n]
    pad = frames[-1:].expand(n - frames.size(0), -1, -1, -1)
    return torch.cat([frames, pad], 0)


# ------------------------------------------------------------------ audio
def _audio_ffmpeg(path: str, start: float, win: float) -> torch.Tensor:
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{win:.3f}",
           "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        err = res.stderr.decode(errors="ignore")
        if "does not contain any stream" in err or "Output file #0 does not contain any stream" in err:
            return torch.zeros(0)
        raise DecodeError(f"ffmpeg audio decode failed for {path}: {err[:300]}")
    return torch.from_numpy(np.frombuffer(res.stdout, dtype=np.float32).copy())


def _audio_torchaudio(path: str, start: float, win: float) -> torch.Tensor:
    import torchaudio
    wave, sr = torchaudio.load(str(path))
    if wave.dim() == 2 and wave.size(0) > 1:
        wave = wave.mean(0, keepdim=True)
    wave = wave.reshape(-1)
    if sr != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
    i0 = int(start * SAMPLE_RATE)
    return wave[i0: i0 + int(win * SAMPLE_RATE)]


def _fit_audio(wave: torch.Tensor, target_len: int) -> torch.Tensor:
    if wave.numel() >= target_len:
        return wave[:target_len]
    return torch.nn.functional.pad(wave, (0, target_len - wave.numel()))


# ------------------------------------------------------------------ public API
def decode_clip(path: str, n_frames: int = 16, audio_len: int = 64000,
                target_size: int = 224, window: str = "random") -> Decoded:
    """Decode an aligned (video, audio) window from one media file.

    window: "random" (training) or "center" (validation / evaluation).
    Raises DecodeError on any failure — never returns fabricated data.
    """
    p = Path(path)
    if not p.exists():
        raise DecodeError(f"media file not found: {path}")
    win_sec = audio_len / SAMPLE_RATE
    info = probe(path)
    start = choose_window(info["duration"], win_sec, window)

    has_video = info["has_video"]
    video = torch.zeros(n_frames, 3, target_size, target_size)
    if has_video:
        if FFMPEG is not None:
            video = _video_ffmpeg(path, start, win_sec, n_frames, target_size)
        else:
            video = _video_cv2(path, start, win_sec, n_frames, target_size,
                               info["fps"], info["n_frames"])

    audio = torch.zeros(0)
    if info.get("has_audio", True):
        if FFMPEG is not None:
            audio = _audio_ffmpeg(path, start, win_sec)
        else:
            try:
                audio = _audio_torchaudio(path, start, win_sec)
            except Exception as e:  # noqa: BLE001 — surfaced as DecodeError below
                raise DecodeError(f"torchaudio decode failed for {path}: {e}") from e
    has_audio = audio.numel() > 0
    audio = _fit_audio(audio, audio_len) if has_audio else torch.zeros(audio_len)
    if not has_video and not has_audio:
        raise DecodeError(f"{path} yielded neither video nor audio")
    return Decoded(video=video, audio=audio, has_video=has_video, has_audio=has_audio,
                   window=(start, win_sec))


def decode_av_from_mp4(mp4_path: str, n_frames: int = 16, audio_len: int = 64000,
                       target_size: int = 224, window: str = "random"
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible wrapper: (video, audio) from the same temporal window."""
    d = decode_clip(mp4_path, n_frames, audio_len, target_size, window)
    return d.video, d.audio


def decode_video(video_path: str, n_frames: int = 16, target_size: int = 224,
                 window: str = "random", win_sec: float = 4.0) -> torch.Tensor:
    return decode_clip(video_path, n_frames, int(win_sec * SAMPLE_RATE), target_size, window).video


def decode_audio(audio_path: str, target_len: int = 64000, sample_rate: int = 16000,
                 window: str = "random") -> torch.Tensor:
    assert sample_rate == SAMPLE_RATE, "only 16 kHz is supported"
    return decode_clip(audio_path, 1, target_len, 32, window).audio


def decode_av_with_faces(mp4_path: str, n_frames: int = 16, audio_len: int = 64000,
                         face_size: int = 224, mouth_size: int = 96
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode video with face/mouth ROI extraction + audio (whole clip, legacy path)."""
    from src.data.face_preprocess import extract_face_mouth_from_video
    faces, mouths = extract_face_mouth_from_video(mp4_path, n_frames, face_size, mouth_size)
    d = decode_clip(mp4_path, n_frames, audio_len, face_size, window="center")
    return faces, mouths, d.audio, d.video
