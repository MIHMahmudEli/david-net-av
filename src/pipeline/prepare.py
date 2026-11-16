"""Decode once -> face-centred crop -> frozen SSL features (+ Phase-B clip cache).

Preprocessing policy (identical for EVERY corpus -- a cross-dataset number is only
meaningful if train and test clips are framed the same way)
  * a 6 s span (24 frames at 4 fps + 6 s of 16 kHz mono audio) centred in the clip
  * video decoded with its aspect ratio kept (longest side <= decode_max_side)
  * faces detected on `detect_frames` frames with OpenCV YuNet (Haar fallback); one
    static square crop per clip around the median face, side = box_scale x face size,
    resized to 224 x 224. No face found -> centre square crop, flagged face_found=False
    (the rate is reported per corpus).
  * 16-frame / 4 s model windows are cut out of the span at `train_view_offsets`
    (training clips) or `eval_view_offset` (everything else).

Outputs on the data repo (private HF dataset):
  features/<feature_set_id>/<corpus>/shard_00000.safetensors  video (n,Lv,768) fp16,
                                                               audio (n,La,768) fp16,
                                                               v_avail, a_avail (n,)
  features/<feature_set_id>/<corpus>/shard_00000.json          [{clip_id, view, ...}]
  features/<feature_set_id>/qacp/<corpus>/shard_*.{safetensors,json}   pseudo-quadrants
  clipcache/<corpus>/shard_00000.{bin,json}                    DVC2 blobs (Phase B)
Every shard is resumable: a shard whose .json is already on the repo is skipped.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import time
import urllib.request
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.pipeline.config import canonical_json
from src.pipeline.env import log_event

PREPARE_VERSION = "prep-v1"          # bump when decode/crop semantics change
SR = 16000
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


# ============================================================================ identity
def feature_set_id(cfg: dict, revisions: dict) -> str:
    import hashlib
    spec = {"v": PREPARE_VERSION, "features": {k: v for k, v in cfg["features"].items()
                                               if k not in ("shard_clips", "shard_clips_audio", "extract_batch_size")},
            "data": {k: cfg["data"][k] for k in ("n_frames", "window_seconds", "sample_rate",
                                                 "frame_size", "cache_frames", "cache_seconds",
                                                 "face_crop")},
            "revisions": revisions}
    return "fs-" + hashlib.sha256(canonical_json(spec).encode()).hexdigest()[:10]


# ============================================================================ faces
class FaceCropper:
    """Static per-clip square crop around the median detected face."""

    def __init__(self, model_path: Optional[str], box_scale: float = 1.8, size: int = 224):
        import cv2
        self.cv2 = cv2
        self.box_scale = box_scale
        self.size = size
        self.backend = "center"
        self._model_path = model_path
        self._yunet = None
        self._haar = None
        self._detect_calls = 0
        self._init_detector()

    def _init_detector(self):
        if self._model_path and Path(self._model_path).exists() and hasattr(self.cv2, "FaceDetectorYN"):
            try:
                self._yunet = self.cv2.FaceDetectorYN.create(str(self._model_path), "", (320, 320), 0.6, 0.3, 50)
                self.backend = "yunet"
            except Exception:  # noqa: BLE001
                self._yunet = None
        if self._yunet is None:
            try:
                self._haar = self.cv2.CascadeClassifier(self.cv2.data.haarcascades +
                                                        "haarcascade_frontalface_default.xml")
                if not self._haar.empty():
                    self.backend = "haar"
            except Exception:  # noqa: BLE001
                self._haar = None

    def _detect(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        if self._yunet is not None:
            self._detect_calls += 1
            if self._detect_calls >= 16:
                # OpenCV issue #24836: FaceDetectorYN.detect accumulates internal C++ heap buffers
                # across calls. Re-instantiating the detector every 16 calls (~4 clips) drops the C++
                # heap allocations back to the OS before memory pressure builds up.
                self._yunet = None
                self._init_detector()
                self._detect_calls = 0
            # YuNet was trained for 320x320. Resizing rgb to fixed 320x320 avoids calling
            # setInputSize with dynamic aspect ratios, which causes OpenCV DNN to rebuild its
            # execution graph and leak C++ layer buffers on alternating video dimensions.
            det_img = self.cv2.resize(rgb, (320, 320), interpolation=self.cv2.INTER_LINEAR)
            bgr = self.cv2.cvtColor(det_img, self.cv2.COLOR_RGB2BGR)
            _, faces = self._yunet.detect(bgr)
            if faces is not None and len(faces):
                sx, sy = w / 320.0, h / 320.0
                valid = []
                for f in faces:
                    fw, fh = float(f[2]) * sx, float(f[3]) * sy
                    score = float(f[14]) if len(f) > 14 else 1.0
                    if fw > 0 and fh > 0 and score >= 0.5:
                        valid.append((f, fw, fh, score))
                if valid:
                    f, fw, fh, score = max(valid, key=lambda x: x[3] if len(x[0]) > 14 else x[1] * x[2])
                    fx, fy = float(f[0]) * sx, float(f[1]) * sy
                    fw = min(float(w), max(16.0, fw))
                    fh = min(float(h), max(16.0, fh))
                    return float(fx + fw / 2), float(fy + fh / 2), float(max(fw, fh))
        elif self._haar is not None:
            gray = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY)
            faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces):
                f = max(faces, key=lambda f: f[2] * f[3])
                x, y, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                fw = min(float(w), max(16.0, fw))
                fh = min(float(h), max(16.0, fh))
                return x + fw / 2, y + fh / 2, float(max(fw, fh))
        return None

    def crop(self, frames: np.ndarray, detect_frames: int = 4):
        """frames (N,H,W,3) uint8 -> (N,size,size,3) uint8, face_found."""
        n, h, w = frames.shape[:3]
        idx = np.linspace(0, n - 1, min(detect_frames, n)).round().astype(int)
        dets = [d for d in (self._detect(frames[i]) for i in idx) if d is not None]
        if dets:
            cx, cy, s = (float(np.median([d[k] for d in dets])) for k in range(3))
            side = s * self.box_scale
            found = True
        else:
            cx, cy, side = w / 2.0, h / 2.0, float(min(h, w))
            found = False
        side = max(16.0, min(float(max(h, w)), side))
        x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
        x2, y2 = x1 + int(round(side)), y1 + int(round(side))

        sx1, sy1 = max(0, x1), max(0, y1)
        sx2, sy2 = min(w, x2), min(h, y2)

        out = np.zeros((n, self.size, self.size, 3), dtype=np.uint8)
        if sx2 > sx1 and sy2 > sy1 and side > 0:
            dx1 = int(round((sx1 - x1) / side * self.size))
            dx2 = int(round((sx2 - x1) / side * self.size))
            dy1 = int(round((sy1 - y1) / side * self.size))
            dy2 = int(round((sy2 - y1) / side * self.size))
            dx1, dy1 = max(0, dx1), max(0, dy1)
            dx2, dy2 = min(self.size, dx2), min(self.size, dy2)
            tw, th = dx2 - dx1, dy2 - dy1
            if tw > 0 and th > 0:
                for i in range(n):
                    patch = frames[i, sy1:sy2, sx1:sx2]
                    out[i, dy1:dy2, dx1:dx2] = self.cv2.resize(
                        patch, (tw, th), interpolation=self.cv2.INTER_AREA)
        return out, found


def ensure_yunet(dst_dir: str | Path) -> Optional[str]:
    dst = Path(dst_dir) / "face_detection_yunet_2023mar.onnx"
    if dst.exists():
        return str(dst)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, str(dst) + ".tmp")
        os.replace(str(dst) + ".tmp", dst)
        return str(dst)
    except Exception as e:  # noqa: BLE001
        log_event("face_detector_fallback", f"YuNet download failed ({e}); using Haar",
                  logging.WARNING)
        return None


# ============================================================================ decode
def _ffprobe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-print_format", "json", "-show_format",
                          "-show_streams", path], capture_output=True, text=True, timeout=60)
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams", [])
    v = [s for s in streams if s.get("codec_type") == "video"]
    a = [s for s in streams if s.get("codec_type") == "audio"]
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    return {"duration": dur, "has_video": bool(v), "has_audio": bool(a),
            "width": int(v[0]["width"]) if v else 0, "height": int(v[0]["height"]) if v else 0}


def _decode_video(path: str, start: float, span: float, n: int, w: int, h: int,
                  max_side: int) -> np.ndarray:
    scale = min(1.0, max_side / max(w, h))
    W, H = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-ss", f"{start:.3f}", "-t", f"{span:.3f}",
           "-i", path, "-vf", f"fps={n / span:.6f},scale={W}:{H}", "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    res = subprocess.run(cmd, capture_output=True, timeout=180)
    k = len(res.stdout) // (W * H * 3) if res.returncode == 0 else 0
    if k == 0:
        # ffmpeg's seek + fps filter yields nothing for some legacy AVI files
        # (DeepfakeTIMIT); OpenCV reads them frame by frame
        arr = _decode_video_cv2(path, start, span, n, W, H)
        k = len(arr)
    else:
        arr = np.frombuffer(res.stdout[:k * W * H * 3], np.uint8).reshape(k, H, W, 3)
    if k < n:                                     # short clip: hold the last frame
        arr = np.concatenate([arr, np.repeat(arr[-1:], n - k, 0)], 0)
    return arr


def _decode_video_cv2(path: str, start: float, span: float, n: int, W: int, H: int) -> np.ndarray:
    """Fallback decoder: seeks to the n needed frames only and resizes each immediately
    (never holds a whole video in memory -- a 1080p clip is GBs of raw frames)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("no frames decoded (ffmpeg and OpenCV)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    i0 = int(start * fps)
    i1 = min(total - 1, int((start + span) * fps)) if total > 0 else i0 + int(span * fps)
    want = np.linspace(i0, max(i0, i1), n).round().astype(int)
    out, last = [], None
    for i in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            last = cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        if last is not None:
            out.append(last)
    cap.release()
    if not out:
        raise RuntimeError("no frames decoded (ffmpeg and OpenCV)")
    while len(out) < n:
        out.append(out[-1])
    return np.stack(out[:n])


def _decode_audio(path: str, start: float, span: float) -> np.ndarray:
    """-> float32 mono 16 kHz. soundfile for plain audio files, ffmpeg otherwise."""
    ext = Path(path).suffix.lower()
    if ext in (".wav", ".flac"):
        try:
            import soundfile as sf
            info = sf.info(path)
            i0 = int(start * info.samplerate)
            x, sr = sf.read(path, start=i0, frames=int(span * info.samplerate),
                            dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != SR:
                # scipy, NOT torch: torch ops inside a worker process deadlocked the pool
                # on Kaggle (WaveFake, 22.05 kHz) and stalled a session for 8+ hours
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(int(sr), SR)
                x = resample_poly(x, SR // g, int(sr) // g)
            return x.astype(np.float32)
        except Exception:  # noqa: BLE001 - fall through to ffmpeg
            pass
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-ss", f"{start:.3f}", "-t", f"{span:.3f}",
           "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True, timeout=120)
    if res.returncode != 0:
        err = res.stderr.decode(errors="ignore")
        if "does not contain any stream" in err:
            return np.zeros(0, np.float32)
        raise RuntimeError(err[:200])
    return np.frombuffer(res.stdout, np.float32).copy()


_WORKER: dict = {}


def _init_worker(yunet_path, box_scale, size):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _WORKER["cropper"] = FaceCropper(yunet_path, box_scale, size)


DECODE_TIMEOUT_S = 240.0
# stop a session cleanly (everything finished is on the Hub) instead of being killed at 100%
MEMORY_PAUSE_FRACTION = 0.88


class _DecodePool:
    """Worker pool that can never hang a session: every clip has a deadline, and on the
    first missed deadline the pool is torn down and rebuilt; the stuck clip is recorded
    as a decode error. When n=0, runs synchronously in-process with zero worker processes,
    zero IPC, and zero shared memory overhead. When n>0, workers are started with 'forkserver'."""

    def __init__(self, n, initargs):
        self.n = max(0, n)
        self.initargs = initargs
        self.pool = None
        self.cropper = None
        if self.n > 0:
            import multiprocessing as mp
            method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
            self.ctx = get_context(method)
            self._start()
        else:
            yunet_path, box_scale, size = initargs
            self.cropper = FaceCropper(yunet_path, box_scale, size)

    def _start(self):
        # never recycle workers: on Kaggle the container is killed (exit 137, no traceback,
        # flat memory) right when a pool replaces a worker -- measured here at 3 workers x
        # 64 clips and upstream at 2 x 200 (commit b6d713e). RSS is flat, so recycling
        # protects against nothing.
        self.pool = self.ctx.Pool(self.n, initializer=_init_worker, initargs=self.initargs,
                                  maxtasksperchild=None)

    def imap(self, jobs, inflight: int | None = None):
        """Ordered results with at most `inflight` clips decoded-but-unconsumed, so
        decoded frames can never pile up in RAM faster than the GPU consumes them."""
        if self.n == 0:
            global _WORKER
            _WORKER["cropper"] = self.cropper
            for j in jobs:
                yield decode_record(j)
            return

        from multiprocessing import TimeoutError as MPTimeout
        inflight = inflight or 2 * self.n
        i_next, futs = 0, {}
        for i in range(len(jobs)):
            while i_next < len(jobs) and i_next < i + inflight:
                futs[i_next] = self.pool.apply_async(decode_record, (jobs[i_next],))
                i_next += 1
            try:
                res = futs.pop(i).get(timeout=DECODE_TIMEOUT_S)
            except MPTimeout:
                cid = jobs[i][0]["clip_id"]
                log_event("decode_timeout", f"{cid}: worker stuck -> pool restarted",
                          logging.WARNING)
                if self.pool is not None:
                    self.pool.terminate()
                    self.pool.join()
                self._start()
                futs = {j: self.pool.apply_async(decode_record, (jobs[j],)) for j in futs}
                res = {"clip_id": cid, "error": f"decode timeout {DECODE_TIMEOUT_S:.0f}s"}
            except Exception as e:
                cid = jobs[i][0]["clip_id"]
                log_event("decode_worker_error", f"{cid}: {type(e).__name__} ({e}) -> pool restarted",
                          logging.WARNING)
                if self.pool is not None:
                    self.pool.terminate()
                    self.pool.join()
                self._start()
                futs = {j: self.pool.apply_async(decode_record, (jobs[j],)) for j in futs}
                res = {"clip_id": cid, "error": f"worker error: {str(e)[:160]}"}
            yield res

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None
        if self.cropper is not None:
            self.cropper = None
            global _WORKER
            _WORKER.pop("cropper", None)


def decode_record(args) -> dict:
    """Worker: one record -> cropped span. Never raises (errors are returned)."""
    rec, root, dcfg = args
    out = {"clip_id": rec["clip_id"], "error": None}
    try:
        path = str(Path(root) / rec["rel_path"])
        span, n = dcfg["cache_seconds"], dcfg["cache_frames"]
        audio_only = rec.get("modalities") == "audio"
        if audio_only:
            info = {"duration": 0.0, "has_video": False, "has_audio": True}
            try:
                import soundfile as sf
                si = sf.info(path)
                info["duration"] = si.frames / si.samplerate
            except Exception:  # noqa: BLE001
                info = _ffprobe(path)
        else:
            info = _ffprobe(path)
        start = max(0.0, (info["duration"] - span) / 2) if info["duration"] > span else 0.0
        frames, face_found = None, None
        if info["has_video"] and not audio_only:
            raw = _decode_video(path, start, span, n, info["width"], info["height"],
                                dcfg["face_crop"]["decode_max_side"])
            if dcfg["face_crop"]["enabled"]:
                frames, face_found = _WORKER["cropper"].crop(raw, dcfg["face_crop"]["detect_frames"])
            else:
                import cv2
                frames = np.stack([cv2.resize(f, (dcfg["frame_size"],) * 2) for f in raw])
        apath = str(Path(root) / rec["audio_rel_path"]) if rec.get("audio_rel_path") else path
        audio = np.zeros(0, np.float32)
        if info.get("has_audio", True) or rec.get("audio_rel_path"):
            audio = _decode_audio(apath, start, span)
        has_audio = audio.size > int(0.1 * SR) and float(np.abs(audio).max(initial=0)) > 1e-4
        full = int(span * SR)
        audio = np.pad(audio[:full], (0, max(0, full - audio.size)))
        from src.pipeline.env import drop_file_cache
        drop_file_cache(path)                  # the mount's pages count against the cgroup
        if apath != path:
            drop_file_cache(apath)
        out.update(frames=frames, audio=(np.clip(audio, -1, 1) * 32767).astype(np.int16),
                   has_video=frames is not None, has_audio=bool(has_audio),
                   face_found=face_found, duration=info["duration"], span_start=start)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


# ============================================================================ features
class FeatureExtractor:
    def __init__(self, cfg: dict, revisions: dict, device: str, dtype=torch.float16):
        from src.pipeline.encoders import build_encoders
        self.v, self.a = build_encoders(cfg["features"], revisions)
        self.v.to(device).eval()
        self.a.to(device).eval()
        self.device = device
        self.amp_dtype = dtype if device == "cuda" else torch.bfloat16
        self.n_frames = cfg["data"]["n_frames"]
        self.win = int(cfg["data"]["window_seconds"] * SR)
        self.fps = cfg["data"]["cache_frames"] / cfg["data"]["cache_seconds"]

    @torch.no_grad()
    def video(self, clips: torch.Tensor) -> torch.Tensor:      # (B,16,3,224,224) float [0,1]
        with torch.autocast(self.device if self.device == "cuda" else "cpu",
                            dtype=self.amp_dtype, enabled=self.device == "cuda"):
            return self.v(clips.to(self.device, non_blocking=True)).half().cpu()

    @torch.no_grad()
    def audio(self, waves: torch.Tensor) -> torch.Tensor:      # (B, 64000) float
        with torch.autocast(self.device if self.device == "cuda" else "cpu",
                            dtype=self.amp_dtype, enabled=self.device == "cuda"):
            return self.a(waves.to(self.device, non_blocking=True)).half().cpu()

    def windows(self, frames: Optional[np.ndarray], audio_i16: np.ndarray, offset: int):
        """Cut the 16-frame / 4 s window starting at frame `offset` (aligned A/V)."""
        a0 = int(round(offset * SR / self.fps))
        wav = torch.from_numpy(audio_i16[a0:a0 + self.win].astype(np.float32) / 32768.0)
        if wav.numel() < self.win:
            wav = torch.nn.functional.pad(wav, (0, self.win - wav.numel()))
        vid = None
        if frames is not None:
            v = frames[offset:offset + self.n_frames]
            vid = torch.from_numpy(np.ascontiguousarray(v)).permute(0, 3, 1, 2).float() / 255.0
        return vid, wav


def _save_shard(path_base: Path, tensors: dict, index: list[dict]):
    from safetensors.torch import save_file
    path_base.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path_base) + ".safetensors")
    Path(str(path_base) + ".json").write_text(json.dumps(index), encoding="utf-8")


class _StreamEncoder:
    """Streams (video, audio) windows through the encoders in fixed-size batches so a
    shard never holds more than `bs` float windows (a 512-clip shard of float views
    would otherwise need ~15 GB of host RAM -- the cgroup kill upstream hit)."""

    def __init__(self, extractor, bs: int):
        self.x, self.bs = extractor, bs
        self.buf_v, self.buf_a = [], []
        self.out_v, self.out_a = [], []
        self._d = extractor.v.hidden

    def add(self, v: Optional[torch.Tensor], a: torch.Tensor):
        self.buf_v.append(v)
        self.buf_a.append(a)
        if len(self.buf_a) >= self.bs:
            self.flush()

    def flush(self):
        if not self.buf_a:
            return
        if os.environ.get("DAVIDNET_DEBUG_EXTRACT", "1") == "1" and len(self.out_a) == 0:
            mem = (torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9) \
                if torch.cuda.is_available() else (0, 0)
            print(f"[dbg] gpu first batch {len(self.buf_a)} windows (video {sum(v is not None for v in self.buf_v)}) "
                  f"cuda alloc/reserved {mem[0]:.2f}/{mem[1]:.2f} GB", flush=True)
        have = [i for i, v in enumerate(self.buf_v) if v is not None]
        a = self.x.audio(torch.stack(self.buf_a))
        v = None
        if have:
            vv = self.x.video(torch.stack([self.buf_v[i] for i in have]))
            v = torch.zeros(len(self.buf_a), vv.shape[1], vv.shape[2], dtype=torch.float16)
            v[have] = vv
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.out_v.append(v)
        self.out_a.append(a)
        self.buf_v, self.buf_a = [], []

    def result(self, Lv: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.flush()
        vs = [v if v is not None else torch.zeros(len(aa), Lv, self._d, dtype=torch.float16)
              for v, aa in zip(self.out_v, self.out_a)]
        v = torch.cat(vs) if vs else torch.zeros(0, Lv, self._d, dtype=torch.float16)
        a = torch.cat(self.out_a) if self.out_a else torch.zeros(0, 1, self._d)
        self.out_v.clear()
        self.out_a.clear()
        return v, a.to(torch.float16)


def _stable_seed(*parts) -> int:
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def extract_corpus(*, corpus: str, records: list[dict], root: str, cfg: dict,
                   extractor: FeatureExtractor, data_store, fsid: str, scratch: Path,
                   multiview_ids: set, qacp_ids: set, num_workers: int, yunet: Optional[str],
                   write_clipcache: bool, batch_size: int = 32, stopwatch=None) -> dict:
    """Resumable extraction of one corpus (all records, fixed shard boundaries).

    multiview_ids: clips that get every `train_view_offsets` window (FakeAVCeleb: all
                   clips, so any split protocol can train on them); others get the
                   centred `eval_view_offset` window only.
    qacp_ids:      real clips that also get `qacp_variants_per_clip` pseudo-fake draws
                   (self-blended video, Griffin-Lim copy-synthesised audio).
    """
    import random as _random
    import cv2
    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(2)
    from src.data.clipcache import encode_clip
    from src.data.synthetic_quadrants import _copy_synthesis_griffinlim, _self_blend_video
    fc = cfg["features"]
    Lv = 8 * fc["video_spatial_grid"] ** 2
    audio_only = all(r.get("modalities") == "audio" for r in records)
    if audio_only:
        Lv = 1          # audio-only corpus: one (null) video token, not 128 zero tokens
    recs = sorted(records, key=lambda r: r["clip_id"])
    if multiview_ids:
        # Multiview corpora (FakeAVCeleb: 3 windows/clip + QACP on real clips) produce ~3-5x
        # more data and take longer per clip. Use 128 clips per shard (~384 windows) so each
        # shard finishes and commits to Hugging Face every ~40s, persisting progress incrementally.
        shard_n = min(128, fc["shard_clips"])
    elif audio_only:
        shard_n = fc["shard_clips_audio"]
    else:
        shard_n = fc["shard_clips"]
    n_shards = math.ceil(len(recs) / shard_n)
    prefix = f"features/{fsid}/{corpus}"
    done = {Path(i.path).name for i in data_store.list_files(prefix, recursive=False)
            if i.path.endswith(".json")} if data_store is not None else set()
    todo_shards = [s for s in range(n_shards) if f"shard_{s:05d}.json" not in done]
    if not todo_shards:
        log_event("features_corpus", f"{corpus}: all {n_shards} shards already extracted",
                  corpus=corpus, skipped_shards=n_shards)
        return {"corpus": corpus, "clips": len(recs), "shards": n_shards,
                "skipped_shards": n_shards, "errors": 0, "face_found": 0,
                "face_checked": 0, "no_audio": 0, "error_examples": []}
    stats = {"corpus": corpus, "clips": len(recs), "shards": n_shards, "skipped_shards": 0,
             "errors": 0, "face_found": 0, "face_checked": 0, "no_audio": 0,
             "error_examples": []}
    dcfg = cfg["data"]
    for s in range(n_shards):
        name = f"shard_{s:05d}"
        if f"{name}.json" in done:
            stats["skipped_shards"] += 1
            continue
        if stopwatch is not None and stopwatch.should_stop():
            log_event("session_paused", f"time budget reached during {corpus} extraction")
            stats["paused"] = True
            break
        chunk = recs[s * shard_n:(s + 1) * shard_n]
        t0 = time.time()
        enc = _StreamEncoder(extractor, batch_size)
        qenc = _StreamEncoder(extractor, batch_size)
        vav, aav, index, q_index, blobs = [], [], [], [], []
        jobs = [(r, root, dcfg) for r in chunk]
        pool = _DecodePool(num_workers, (yunet, dcfg["face_crop"]["box_scale"], dcfg["frame_size"]))
        try:
            for i_clip, (rec, d) in enumerate(zip(chunk, pool.imap(jobs))):
                if os.environ.get("DAVIDNET_DEBUG_EXTRACT", "1") == "1" and (i_clip == 0 or (i_clip + 1) % 16 == 0 or (i_clip + 1) == len(chunk)):
                    print(f"[dbg] {corpus} {name}: decoded {i_clip + 1}/{len(chunk)} clips (err={d['error']})", flush=True)
                if d["error"]:
                    stats["errors"] += 1
                    if len(stats["error_examples"]) < 5:
                        stats["error_examples"].append({"clip_id": rec["clip_id"],
                                                        "error": d["error"]})
                        log_event("decode_error", f"{corpus} {rec['clip_id']}: {d['error']}",
                                  logging.WARNING)
                    continue
                if d["face_found"] is not None:
                    stats["face_checked"] += 1
                    stats["face_found"] += int(d["face_found"])
                stats["no_audio"] += int(not d["has_audio"])
                offsets = (fc["train_view_offsets"] if rec["clip_id"] in multiview_ids
                           else [fc["eval_view_offset"]])
                for off in offsets:
                    v, a = extractor.windows(d["frames"], d["audio"], off)
                    enc.add(v, a)
                    vav.append(float(d["has_video"]))
                    aav.append(float(d["has_audio"]))
                    index.append({"clip_id": rec["clip_id"], "view": off,
                                  "face_found": d["face_found"], "has_video": d["has_video"],
                                  "has_audio": d["has_audio"]})
                if rec["clip_id"] in qacp_ids and d["has_video"]:
                    v, a = extractor.windows(d["frames"], d["audio"], fc["eval_view_offset"])
                    for k in range(fc["qacp_variants_per_clip"]):
                        seed = _stable_seed(rec["clip_id"], k, "qacp")
                        torch.manual_seed(seed)
                        _random.seed(seed)
                        qenc.add(_self_blend_video(v.clone()).clamp(0, 1),
                                 _copy_synthesis_griffinlim(a.clone()).clamp(-1, 1))
                        q_index.append({"clip_id": rec["clip_id"], "variant": k, "seed": seed})
                if write_clipcache and d["frames"] is not None:
                    blobs.append((rec["clip_id"], encode_clip(
                        d["frames"], d["audio"], has_video=True, has_audio=d["has_audio"],
                        duration=float(d["duration"]), span_start=float(d["span_start"]))))
                d["frames"] = None
                d["audio"] = None
        finally:
            pool.close()
        if os.environ.get("DAVIDNET_DEBUG_EXTRACT", "1") == "1":
            print(f"[dbg] {corpus} {name}: saving shard to disk...", flush=True)
        video, audio = enc.result(Lv)
        base = scratch / prefix / name
        _save_shard(base, {"video": video, "audio": audio,
                           "v_avail": torch.tensor(vav, dtype=torch.float16),
                           "a_avail": torch.tensor(aav, dtype=torch.float16)}, index)
        adds = {f"{prefix}/{name}.safetensors": str(base) + ".safetensors",
                f"{prefix}/{name}.json": str(base) + ".json"}
        if q_index:
            sb, gl = qenc.result(Lv)
            qbase = scratch / f"features/{fsid}/qacp/{corpus}" / name
            _save_shard(qbase, {"sb_video": sb, "gl_audio": gl}, q_index)
            adds[f"features/{fsid}/qacp/{corpus}/{name}.safetensors"] = str(qbase) + ".safetensors"
            adds[f"features/{fsid}/qacp/{corpus}/{name}.json"] = str(qbase) + ".json"
        if blobs:
            cbase = scratch / f"clipcache/{corpus}" / name
            cbase.parent.mkdir(parents=True, exist_ok=True)
            idx, off = {}, 0
            with open(str(cbase) + ".bin", "wb") as f:
                for cid, b in blobs:
                    f.write(b)
                    idx[cid] = [off, len(b)]
                    off += len(b)
            Path(str(cbase) + ".json").write_text(json.dumps(idx), encoding="utf-8")
            # same commit as the feature shard: the feature .json is the "done" marker,
            # so the clip cache can never be half-present for a finished shard
            adds[f"clipcache/{corpus}/{name}.bin"] = str(cbase) + ".bin"
            adds[f"clipcache/{corpus}/{name}.json"] = str(cbase) + ".json"
            blobs.clear()
        del enc
        del qenc
        if data_store is not None:
            t_commit = time.time()
            if os.environ.get("DAVIDNET_DEBUG_EXTRACT", "1") == "1":
                print(f"[dbg] {corpus} {name}: committing {len(adds)} files to Hub...", flush=True)
            data_store.commit(adds, message=f"{fsid}/{corpus}: {name} ({len(chunk)} clips)")
            if os.environ.get("DAVIDNET_DEBUG_EXTRACT", "1") == "1":
                print(f"[dbg] {corpus} {name}: committed to Hub in {time.time() - t_commit:.1f}s", flush=True)
            from src.pipeline.env import drop_file_cache
            for p in adds.values():
                drop_file_cache(p)
                Path(p).unlink(missing_ok=True)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from src.pipeline.env import relieve_memory_pressure
        frac = relieve_memory_pressure()
        if frac > MEMORY_PAUSE_FRACTION:
            log_event("session_paused", f"container memory at {frac:.0%} of its limit after "
                      f"{corpus} {name}; stopping cleanly (next session resumes here)",
                      logging.WARNING)
            stats["paused"] = True
            break
        log_event("features_shard", f"{corpus} {name}: {len(chunk)} clips, "
                  f"{len(index)} windows in {time.time() - t0:.0f}s",
                  corpus=corpus, shard=s + 1, of=n_shards)
    if stats["errors"] > max(5, 0.05 * len(recs)):
        log_event("decode_error_rate", f"{corpus}: {stats['errors']}/{len(recs)} clips failed "
                  f"to decode, e.g. {stats['error_examples'][:2]}", logging.ERROR)
    return stats


def write_feature_meta(data_store, fsid: str, cfg: dict, revisions: dict, stats: list[dict],
                       env: dict):
    meta = {"feature_set_id": fsid, "prepare_version": PREPARE_VERSION,
            "features": cfg["features"], "data": {k: cfg["data"][k] for k in (
                "n_frames", "window_seconds", "sample_rate", "frame_size", "cache_frames",
                "cache_seconds", "face_crop")},
            "model_revisions": revisions, "corpus_stats": stats, "environment": env}
    data_store.write_json(f"features/{fsid}/FEATURES_META.json", meta,
                          message=f"{fsid}: metadata")
    return meta
