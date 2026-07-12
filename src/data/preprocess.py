"""Real preprocessing: raw clip -> (face crop, mouth crop, audio waveform) tensors.

Pipeline (docs/03_datasets.md §5):
  video: ffmpeg decode @ fixed fps -> face detect + EMA-smoothed track ->
         224x224 face crop + 96x96 mouth crop -> float32 [0,1] tensors
  audio: ffmpeg extract -> 16 kHz mono -> RMS loudness normalize
  meta:  ffprobe (duration, fps, has_audio) -> drives missing-modality flags

Dependency chain (all lazy; the best available backend wins):
  decode/probe : ffmpeg + ffprobe on PATH (required for real media)
  face detect  : insightface (RetinaFace, best; kps -> true mouth crop)
                 -> mediapipe (fragile on ARM, see docs/07)
                 -> OpenCV Haar cascade
                 -> center-crop heuristic (last resort; flagged in meta)

CLI (manifest -> shards consumed by AVDeepfakeDataset / extract_features):
  python -m src.data.preprocess --manifest src/data/manifests/fakeavceleb.jsonl \
      --raw-root data/fakeavceleb --out data/shards/fakeavceleb
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.datasets import load_manifest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


# ============================================================== media probing
def probe_media(path: str) -> dict:
    """ffprobe -> {duration_sec, fps, has_video, has_audio}. Drives the
    missing-modality flags in the API (silent clips, audio-only files)."""
    if FFPROBE is None:
        raise RuntimeError("ffprobe not found on PATH — install ffmpeg")
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    streams = info.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]
    fps = 0.0
    if vstreams:
        num, _, den = (vstreams[0].get("avg_frame_rate", "0/1")).partition("/")
        fps = float(num) / float(den) if float(den or 1) else 0.0
    return {
        "duration_sec": float(info.get("format", {}).get("duration", 0.0)),
        "fps": fps,
        "has_video": bool(vstreams),
        "has_audio": bool(astreams),
    }


# ============================================================== decode
def decode_frames(path: str, fps: int = 25, max_frames: int = 512) -> np.ndarray:
    """ffmpeg rawvideo pipe -> (T, H, W, 3) uint8 RGB. No OpenCV needed."""
    if FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH — install ffmpeg")
    meta = probe_media(path)
    if not meta["has_video"]:
        raise ValueError(f"{path} has no video stream")
    # probe display size from the first video stream
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-print_format", "json",
         "-show_entries", "stream=width,height", str(path)],
        capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-vf", f"fps={fps}",
         "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[: n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3)


def extract_audio(path: str, sr: int = 16000) -> torch.Tensor:
    """ffmpeg -> mono float32 waveform at `sr`. Returns empty tensor if no track."""
    if FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH — install ffmpeg")
    if not probe_media(path)["has_audio"]:
        return torch.zeros(0)
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    wave = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(wave.copy())


def rms_normalize(wave: torch.Tensor, target_rms: float = 0.05) -> torch.Tensor:
    """Simple loudness normalization to a target RMS."""
    if wave.numel() == 0:
        return wave
    rms = wave.pow(2).mean().sqrt().clamp(min=1e-8)
    return (wave * (target_rms / rms)).clamp(-1.0, 1.0)


def uniform_indices(n_total: int, n_sample: int) -> np.ndarray:
    """Evenly spaced frame indices (with repetition if the clip is short)."""
    if n_total <= 0:
        raise ValueError("empty clip")
    return np.linspace(0, n_total - 1, n_sample).round().astype(int)


# ============================================================== face detection
class FaceCropper:
    """Face detector with a backend fallback chain and EMA bbox smoothing.

    Backends: insightface -> mediapipe -> opencv-haar -> center heuristic.
    The chosen backend is recorded so downstream can audit crop quality.
    """

    def __init__(self, ema: float = 0.7):
        self.ema = ema
        self._bbox = None            # smoothed (x1, y1, x2, y2) floats
        self.backend, self._det = self._init_backend()

    def _init_backend(self):
        try:
            from insightface.app import FaceAnalysis
            det = FaceAnalysis(allowed_modules=["detection"])
            det.prepare(ctx_id=0, det_size=(320, 320))
            return "insightface", det
        except Exception:
            pass
        try:
            import mediapipe as mp
            det = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5)
            return "mediapipe", det
        except Exception:
            pass
        try:
            import cv2
            det = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if not det.empty():
                return "opencv-haar", det
        except Exception:
            pass
        return "center-fallback", None

    def _detect_raw(self, frame: np.ndarray):
        """-> (bbox, mouth_center) in pixels, or (None, None)."""
        h, w = frame.shape[:2]
        if self.backend == "insightface":
            faces = self._det.get(frame)
            if faces:
                f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                x1, y1, x2, y2 = f.bbox
                mouth = None
                if getattr(f, "kps", None) is not None and len(f.kps) >= 5:
                    mouth = tuple(np.mean(f.kps[3:5], axis=0))  # mouth-corner midpoint
                return (x1, y1, x2, y2), mouth
        elif self.backend == "mediapipe":
            res = self._det.process(frame)
            if res.detections:
                d = max(res.detections,
                        key=lambda d: d.location_data.relative_bounding_box.width)
                rb = d.location_data.relative_bounding_box
                return (rb.xmin * w, rb.ymin * h,
                        (rb.xmin + rb.width) * w, (rb.ymin + rb.height) * h), None
        elif self.backend == "opencv-haar":
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self._det.detectMultiScale(gray, 1.1, 4)
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                return (float(x), float(y), float(x + fw), float(y + fh)), None
        return None, None

    def track(self, frame: np.ndarray):
        """Detect + EMA-smooth. Always returns a usable bbox + mouth center."""
        h, w = frame.shape[:2]
        bbox, mouth = self._detect_raw(frame)
        if bbox is None:
            if self._bbox is not None:
                bbox = self._bbox            # hold last track through dropouts
            else:                            # center heuristic (flagged via backend)
                s = min(h, w) * 0.8
                bbox = ((w - s) / 2, (h - s) / 2, (w + s) / 2, (h + s) / 2)
        if self._bbox is None:
            self._bbox = bbox
        else:                                # EMA smoothing kills bbox jitter
            self._bbox = tuple(self.ema * o + (1 - self.ema) * n
                               for o, n in zip(self._bbox, bbox))
        if mouth is None:
            x1, y1, x2, y2 = self._bbox      # lower-middle of the face box
            mouth = ((x1 + x2) / 2, y1 + 0.75 * (y2 - y1))
        return self._bbox, mouth


def _square_crop(frame: np.ndarray, cx: float, cy: float, size: float) -> np.ndarray:
    """Square crop centered at (cx, cy), zero-padded at image borders."""
    h, w = frame.shape[:2]
    half = size / 2
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = x1 + int(round(size)), y1 + int(round(size))
    out = np.zeros((y2 - y1, x2 - x1, 3), dtype=frame.dtype)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1: sy2 - y1, sx1 - x1: sx2 - x1] = frame[sy1:sy2, sx1:sx2]
    return out


def _resize(t: torch.Tensor, size: int) -> torch.Tensor:
    """(T, 3, H, W) -> (T, 3, size, size) via torch (no OpenCV dependency)."""
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)


def face_and_mouth_crops(frames: np.ndarray, cropper: FaceCropper,
                         face_size: int = 224, mouth_size: int = 96):
    """(T, H, W, 3) uint8 -> face (T,3,224,224) + mouth (T,3,96,96) float [0,1]."""
    faces, mouths = [], []
    for frame in frames:
        (x1, y1, x2, y2), (mx, my) = cropper.track(frame)
        fsize = max(x2 - x1, y2 - y1) * 1.3          # margin around the face
        faces.append(_square_crop(frame, (x1 + x2) / 2, (y1 + y2) / 2, fsize))
        mouths.append(_square_crop(frame, mx, my, fsize * 0.45))
    def _to_tensor(crops, size):
        ts = [torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1).float() / 255.0
              for c in crops]
        ts = [_resize(t.unsqueeze(0), size).squeeze(0) for t in ts]
        return torch.stack(ts)
    return _to_tensor(faces, face_size), _to_tensor(mouths, mouth_size)


# ============================================================== clip pipeline
def preprocess_clip(path: str, n_frames: int = 16, fps: int = 25,
                    sr: int = 16000, audio_len: int = 64000,
                    cropper: FaceCropper | None = None) -> dict:
    """Full pipeline for one clip. Returns tensors + meta (incl. has_audio flag)."""
    meta = probe_media(path)
    cropper = cropper or FaceCropper()

    all_frames = decode_frames(path, fps=fps)
    idx = uniform_indices(len(all_frames), n_frames)
    video, mouth = face_and_mouth_crops(all_frames[idx], cropper)

    audio = rms_normalize(extract_audio(path, sr=sr))
    if audio.numel() >= audio_len:
        audio = audio[:audio_len]
    else:                                            # pad short/silent clips
        audio = F.pad(audio, (0, audio_len - audio.numel()))

    meta["face_backend"] = cropper.backend
    return {"video": video, "mouth": mouth, "audio": audio, "meta": meta}


# ============================================================== manifest CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--audio-len", type=int, default=64000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest)
    cropper = FaceCropper()
    print(f"face backend: {cropper.backend}")

    n_ok, n_fail = 0, 0
    meta_log = open(out / "preprocess_meta.jsonl", "a", encoding="utf-8")
    for rec in records:
        vp = out / f"{rec['clip_id']}_video.pt"
        apth = out / f"{rec['clip_id']}_audio.pt"
        if vp.exists() and apth.exists():
            continue                                  # resumable
        raw = Path(args.raw_root) / rec.get("rel_path", rec["clip_id"] + ".mp4")
        try:
            r = preprocess_clip(str(raw), args.n_frames, args.fps, args.sr, args.audio_len,
                                cropper=cropper)
            torch.save(r["video"], vp)
            torch.save(r["mouth"], out / f"{rec['clip_id']}_mouth.pt")
            torch.save(r["audio"], apth)
            meta_log.write(json.dumps({"clip_id": rec["clip_id"], **r["meta"]}) + "\n")
            n_ok += 1
        except Exception as e:  # noqa: BLE001 — log and continue over corrupt files
            print(f"[fail] {rec['clip_id']}: {e}")
            n_fail += 1
    meta_log.close()
    print(f"done: {n_ok} ok, {n_fail} failed -> {out}")


if __name__ == "__main__":
    main()
