"""Packed, pre-decoded clip cache -- the media path Kaggle can actually survive.

Why this exists
---------------
Stage 1 used to run ffmpeg inside every DataLoader worker: 8 workers x 2 prefetched
batches x 8 clips, each forking a subprocess and holding decoded float32 frames. On
Kaggle that combination killed the container with SIGKILL (exit 137) roughly five
minutes into every run, because the cgroup's 32 GB ceiling counts page cache, worker
copy-on-write pages and pinned buffers together -- and psutil's `available`, which
treats reclaimable cache as free, reported everything was fine right up to the kill.

So decoding moves out of the training loop. Every clip is decoded ONCE into a compact
blob, blobs are concatenated into a handful of large shards, and training does one
seek + read + JPEG decode per sample. No subprocesses, no ffmpeg, no decode timeouts,
and a flat memory profile, because the only thing the kernel caches is the few hundred
KB actually read -- clean, file-backed, reclaimable pages.

Blob layout (one per clip)
--------------------------
    b"DVC2"                     magic
    uint32 little-endian        header length
    header                      JSON, utf-8
    JPEG grid                   all frames tiled into ONE image
    audio                       int16 little-endian, mono @ 16 kHz

Frames are tiled into a single JPEG rather than stored one per frame: it is a little
smaller (one set of Huffman tables) and, more importantly, one `imdecode` call instead
of twenty-four.

Temporal augmentation is preserved, not lost. The cache holds a span WIDER than the
model's window (24 frames / 6 s against the model's 16 frames / 4 s), so `read` still
draws a random contiguous sub-window per epoch. Frames sit at exactly
`CACHE_FRAMES / CACHE_SECONDS` = 4 fps, so a sub-window starting at frame `o` aligns
with audio sample `o * 4000`: video and audio always describe the same instant, which
the sync head depends on.
"""
from __future__ import annotations

import json
import os
import random
import struct
from pathlib import Path

import numpy as np
import torch

MAGIC = b"DVC2"
SAMPLE_RATE = 16000

# The cached span. Wider than the model's 16-frame / 4-second window so that a random
# sub-window is still a real augmentation; 6 s at 4 fps keeps whole-sample alignment.
CACHE_FRAMES = 24
CACHE_SECONDS = 6.0
CACHE_AUDIO_LEN = int(CACHE_SECONDS * SAMPLE_RATE)   # 96000
CACHE_FPS = CACHE_FRAMES / CACHE_SECONDS             # 4.0
FRAME_SIZE = 224
GRID_COLS = 6                                        # 6 x 4 tiles for 24 frames
JPEG_QUALITY = 87

# ~450 KB per clip, so 21.5k clips is ~9.7 GB spread over shards of this size.
SHARD_BYTES = 256 * 1024 * 1024


class CacheError(RuntimeError):
    pass


# ----------------------------------------------------------------- encode / decode
def _jpeg_params() -> list:
    """Quality, and 4:4:4 chroma -- the subsampling must be off.

    Tiling puts unrelated frames side by side, and 4:2:0 shares one chroma sample
    between neighbouring pixels, so colour bleeds across every tile seam. Measured on
    adjacent flat tiles of very different colours: the outer 2-pixel ring of each frame
    was off by up to 0.455 (of 1.0) at 4:2:0 versus 0.004 at 4:4:4, while the interior
    was 0.004 either way. Border artefacts are precisely what a deepfake detector looks
    at, so the ~23% extra size (8.5 GB -> 10.5 GB over the corpus) is worth paying.
    """
    import cv2
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    factor = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
    mode = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
    if factor is not None and mode is not None:
        params += [int(factor), int(mode)]
    return params


def _tile(frames: np.ndarray, cols: int) -> np.ndarray:
    """(N, H, W, 3) -> one (rows*H, cols*W, 3) image, row-major, zero-padded."""
    n, h, w, c = frames.shape
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, c), dtype=frames.dtype)
    for i in range(n):
        r, q = divmod(i, cols)
        grid[r * h:(r + 1) * h, q * w:(q + 1) * w] = frames[i]
    return grid


def _untile(grid: np.ndarray, n: int, cols: int, size: int) -> np.ndarray:
    out = np.empty((n, size, size, 3), dtype=grid.dtype)
    for i in range(n):
        r, q = divmod(i, cols)
        out[i] = grid[r * size:(r + 1) * size, q * size:(q + 1) * size]
    return out


def encode_clip(frames: np.ndarray, audio: np.ndarray, *, has_video: bool,
                has_audio: bool, duration: float, span_start: float) -> bytes:
    """Pack one clip. `frames` is (N, 224, 224, 3) uint8 RGB, `audio` int16 mono @16k."""
    import cv2
    if frames.dtype != np.uint8:
        raise CacheError(f"frames must be uint8, got {frames.dtype}")
    if audio.dtype != np.int16:
        raise CacheError(f"audio must be int16, got {audio.dtype}")
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    if w != h:
        raise CacheError(f"frames must be square, got {h}x{w}")
    grid = _tile(frames[:, :, :, ::-1], GRID_COLS)          # RGB -> BGR for cv2
    ok, buf = cv2.imencode(".jpg", grid, _jpeg_params())
    if not ok:
        raise CacheError("cv2.imencode failed")
    jpeg = buf.tobytes()
    pcm = audio.astype("<i2", copy=False).tobytes()
    header = json.dumps({
        "v": 2, "n": int(n), "size": int(h), "cols": GRID_COLS,
        "sr": SAMPLE_RATE, "fps": CACHE_FPS, "na": len(pcm) // 2,
        "jpeg": len(jpeg), "dur": round(float(duration), 3),
        "start": round(float(span_start), 3),
        "hv": bool(has_video), "ha": bool(has_audio),
    }, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack("<I", len(header)) + header + jpeg + pcm


def decode_blob(blob: bytes, n_frames: int = 16, audio_len: int = 64000,
                window: str = "random", rng: random.Random | None = None):
    """Unpack a blob into (video, audio, has_video, has_audio).

    video is (n_frames, 3, size, size) float32 in [0, 1]; audio is (audio_len,) float32.
    A contiguous sub-window is drawn so video and audio describe the same instant.
    """
    import cv2
    if blob[:4] != MAGIC:
        raise CacheError(f"bad magic {blob[:4]!r} -- not a DVC2 blob")
    hlen = struct.unpack("<I", blob[4:8])[0]
    off = 8 + hlen
    hdr = json.loads(blob[8:off].decode("utf-8"))
    n, size, cols = hdr["n"], hdr["size"], hdr["cols"]

    grid = cv2.imdecode(np.frombuffer(blob[off:off + hdr["jpeg"]], dtype=np.uint8),
                        cv2.IMREAD_COLOR)
    if grid is None:
        raise CacheError("cv2.imdecode failed -- truncated or corrupt shard")
    frames = _untile(grid, n, cols, size)[:, :, :, ::-1]      # BGR -> RGB
    pcm = np.frombuffer(blob[off + hdr["jpeg"]:], dtype="<i2")

    # One offset drives both streams, so they stay aligned.
    span = max(0, n - n_frames)
    if span == 0 or window == "center":
        o = span // 2
    else:
        o = (rng or random).randint(0, span)
    a0 = int(round(o * hdr["sr"] / hdr["fps"]))

    vid = frames[o:o + n_frames]
    if vid.shape[0] < n_frames:                               # short clip: hold last frame
        vid = np.concatenate([vid, np.repeat(vid[-1:], n_frames - vid.shape[0], 0)], 0)
    video = torch.from_numpy(np.ascontiguousarray(vid)).permute(0, 3, 1, 2).float().div_(255.0)

    aud = pcm[a0:a0 + audio_len]
    audio = torch.from_numpy(aud.astype(np.float32) / 32768.0)
    if audio.numel() < audio_len:
        audio = torch.nn.functional.pad(audio, (0, audio_len - audio.numel()))
    return video, audio, bool(hdr["hv"]), bool(hdr["ha"])


# ----------------------------------------------------------------- writing
class CacheWriter:
    """Append blobs to large shards and record where each one landed.

    Shards are dropped from page cache as they are completed (`POSIX_FADV_DONTNEED`).
    Building the cache writes ~10 GB; without this the dirty pages accumulate against
    the same cgroup ceiling that killed training, which is exactly the mistake that
    produced a 20 GB spike when the dataset was copied with `cp`.
    """

    def __init__(self, out_dir, shard_bytes: int = SHARD_BYTES):
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_bytes = int(shard_bytes)
        self.index: dict[str, list] = {}
        self.failures: dict[str, str] = {}
        self._fh = None
        self._shard = -1
        self._off = 0
        self._load_existing()

    def _load_existing(self):
        p = self.root / "index.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            self.index = data.get("clips", {})
            self.failures = data.get("failures", {})
            self._shard = max((v[0] for v in self.index.values()), default=-1)
            if self._shard >= 0:
                sp = self.root / f"shard_{self._shard:04d}.bin"
                self._off = sp.stat().st_size if sp.exists() else 0

    def __contains__(self, clip_id: str) -> bool:
        return clip_id in self.index or clip_id in self.failures

    def _roll(self):
        self._close_shard()
        self._shard += 1
        self._off = 0
        self._fh = open(self.root / f"shard_{self._shard:04d}.bin", "ab")

    def _close_shard(self):
        if self._fh is None:
            return
        self._fh.flush()
        fd = self._fh.fileno()
        os.fsync(fd)
        if hasattr(os, "posix_fadvise"):        # release the page cache we just dirtied
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        self._fh.close()
        self._fh = None

    def add(self, clip_id: str, blob: bytes):
        if self._fh is None or self._off + len(blob) > self.shard_bytes:
            self._roll()
        self._fh.write(blob)
        self.index[clip_id] = [self._shard, self._off, len(blob)]
        self._off += len(blob)

    def fail(self, clip_id: str, reason: str):
        self.failures[clip_id] = str(reason)[:300]

    def flush_index(self):
        """Write the index so an interrupted build resumes instead of restarting."""
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        tmp = self.root / "index.json.tmp"
        tmp.write_text(json.dumps({
            "format": "DVC2", "frames": CACHE_FRAMES, "seconds": CACHE_SECONDS,
            "size": FRAME_SIZE, "sample_rate": SAMPLE_RATE,
            "clips": self.index, "failures": self.failures,
        }, separators=(",", ":")), encoding="utf-8")
        tmp.replace(self.root / "index.json")

    def close(self):
        self._close_shard()
        self.flush_index()


# ----------------------------------------------------------------- reading
class ClipCache:
    """Random access into the packed shards. Safe to share across DataLoader workers.

    File handles are opened lazily per process and keyed by pid, because a handle
    inherited across `fork` shares its file offset with the parent -- two workers
    seeking the same fd would read each other's bytes. `os.pread` is used for the same
    reason: it takes the offset as an argument and never mutates shared state.
    """

    def __init__(self, root):
        self.root = Path(root)
        p = self.root / "index.json"
        if not p.exists():
            raise CacheError(
                f"no clip cache at {self.root} (index.json missing). Build it with "
                "`python -m scripts.build_clip_cache --manifest ... --out ...`")
        data = json.loads(p.read_text(encoding="utf-8"))
        self.clips: dict[str, list] = data["clips"]
        self.failures: dict[str, str] = data.get("failures", {})
        self.meta = {k: data[k] for k in ("format", "frames", "seconds", "size") if k in data}
        self._fh: dict[tuple, object] = {}

    def __len__(self):
        return len(self.clips)

    def __contains__(self, clip_id: str) -> bool:
        return clip_id in self.clips

    def _handle(self, shard: int):
        key = (os.getpid(), shard)
        fh = self._fh.get(key)
        if fh is None:
            fh = open(self.root / f"shard_{shard:04d}.bin", "rb", buffering=0)
            self._fh[key] = fh
        return fh

    def blob(self, clip_id: str) -> bytes:
        try:
            shard, off, ln = self.clips[clip_id]
        except KeyError:
            raise CacheError(f"{clip_id} is not in the cache at {self.root}") from None
        fh = self._handle(shard)
        if hasattr(os, "pread"):
            data = os.pread(fh.fileno(), ln, off)
        else:                                    # Windows: no pread, but no fork either
            fh.seek(off)
            data = fh.read(ln)
        if len(data) != ln:
            raise CacheError(f"short read for {clip_id}: {len(data)}/{ln} bytes")
        return data

    def read(self, clip_id: str, n_frames: int = 16, audio_len: int = 64000,
             window: str = "random", rng: random.Random | None = None):
        return decode_blob(self.blob(clip_id), n_frames, audio_len, window, rng)

    def coverage(self, records) -> float:
        if not records:
            return 0.0
        return sum(r.get("clip_id") in self.clips for r in records) / len(records)
