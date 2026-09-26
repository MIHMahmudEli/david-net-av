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
        self._yunet = None
        self._haar = None
        if model_path and Path(model_path).exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yunet = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.6, 0.3, 50)
                self.backend = "yunet"
            except Exception:  # noqa: BLE001
                self._yunet = None
        if self._yunet is None:
            try:
                self._haar = cv2.CascadeClassifier(cv2.data.haarcascades +
                                                   "haarcascade_frontalface_default.xml")
                if not self._haar.empty():
                    self.backend = "haar"
            except Exception:  # noqa: BLE001
                self._haar = None

    def _detect(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        if self._yunet is not None:
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))
            if faces is not None and len(faces):
                f = max(faces, key=lambda f: f[2] * f[3])
                return float(f[0] + f[2] / 2), float(f[1] + f[3] / 2), float(max(f[2], f[3]))
        elif self._haar is not None:
            gray = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY)
            faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
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
            cx, cy, side = w / 2, h / 2, float(min(h, w))
            found = False
        side = max(16.0, side)
        x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
        x2, y2 = x1 + int(round(side)), y1 + int(round(side))
        out = np.empty((n, self.size, self.size, 3), dtype=np.uint8)
        for i in range(n):
            canvas = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
            sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if sx2 > sx1 and sy2 > sy1:
                canvas[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = frames[i, sy1:sy2, sx1:sx2]
            out[i] = self.cv2.resize(canvas, (self.size, self.size),
                                     interpolation=self.cv2.INTER_AREA)
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
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
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
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{span:.3f}",
           "-i", path, "-vf", f"fps={n / span:.6f},scale={W}:{H}", "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    res = subprocess.run(cmd, capture_output=True, timeout=180)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode(errors="ignore")[:200])
    k = len(res.stdout) // (W * H * 3)
    if k == 0:
        raise RuntimeError("no frames decoded")
    arr = np.frombuffer(res.stdout[:k * W * H * 3], np.uint8).reshape(k, H, W, 3)
    if k < n:                                     # short clip: hold the last frame
        arr = np.concatenate([arr, np.repeat(arr[-1:], n - k, 0)], 0)
    return arr


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
                import torchaudio.functional as AF
                x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
            return x.astype(np.float32)
        except Exception:  # noqa: BLE001 - fall through to ffmpeg
            pass
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{span:.3f}",
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
    _WORKER["cropper"] = FaceCropper(yunet_path, box_scale, size)


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
        have = [i for i, v in enumerate(self.buf_v) if v is not None]
        a = self.x.audio(torch.stack(self.buf_a))
        v = None
        if have:
            vv = self.x.video(torch.stack([self.buf_v[i] for i in have]))
            v = torch.zeros(len(self.buf_a), vv.shape[1], vv.shape[2], dtype=torch.float16)
            v[have] = vv
        self.out_v.append(v)
        self.out_a.append(a)
        self.buf_v, self.buf_a = [], []

    def result(self, Lv: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.flush()
        vs = [v if v is not None else torch.zeros(len(aa), Lv, self._d, dtype=torch.float16)
              for v, aa in zip(self.out_v, self.out_a)]
        v = torch.cat(vs) if vs else torch.zeros(0, Lv, self._d, dtype=torch.float16)
        a = torch.cat(self.out_a) if self.out_a else torch.zeros(0, 1, self._d)
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
    from src.data.clipcache import encode_clip
    from src.data.synthetic_quadrants import _copy_synthesis_griffinlim, _self_blend_video
    fc = cfg["features"]
    Lv = 8 * fc["video_spatial_grid"] ** 2
    audio_only = all(r.get("modalities") == "audio" for r in records)
    if audio_only:
        Lv = 1          # audio-only corpus: one (null) video token, not 128 zero tokens
    recs = sorted(records, key=lambda r: r["clip_id"])
    shard_n = fc["shard_clips_audio"] if audio_only else fc["shard_clips"]
    n_shards = math.ceil(len(recs) / shard_n)
    prefix = f"features/{fsid}/{corpus}"
    done = {Path(i.path).name for i in data_store.list_files(prefix, recursive=False)
            if i.path.endswith(".json")} if data_store is not None else set()
    stats = {"corpus": corpus, "clips": len(recs), "shards": n_shards, "skipped_shards": 0,
             "errors": 0, "face_found": 0, "face_checked": 0, "no_audio": 0,
             "error_examples": []}
    ctx = get_context("fork" if os.name != "nt" else "spawn")
    dcfg = cfg["data"]
    pool = ctx.Pool(max(1, num_workers), initializer=_init_worker,
                    initargs=(yunet, dcfg["face_crop"]["box_scale"], dcfg["frame_size"]),
                    maxtasksperchild=64)
    try:
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
            for rec, d in zip(chunk, pool.imap(decode_record, jobs, chunksize=2)):
                if d["error"]:
                    stats["errors"] += 1
                    if len(stats["error_examples"]) < 5:
                        stats["error_examples"].append({"clip_id": rec["clip_id"],
                                                        "error": d["error"]})
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
            if data_store is not None:
                data_store.commit(adds, message=f"{fsid}/{corpus}: {name} ({len(chunk)} clips)")
                for p in adds.values():
                    Path(p).unlink(missing_ok=True)
            log_event("features_shard", f"{corpus} {name}: {len(chunk)} clips, "
                      f"{len(index)} windows in {time.time() - t0:.0f}s",
                      corpus=corpus, shard=s + 1, of=n_shards)
    finally:
        pool.terminate()
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
