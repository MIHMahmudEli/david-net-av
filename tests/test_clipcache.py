"""The packed clip cache must return what was put in, and keep A/V on one window.

Two failure modes here are silent and expensive. A colour-channel swap in the JPEG
round trip trains the video encoder on BGR, which costs accuracy without ever raising.
A misaligned audio offset feeds the sync head streams from different instants, which
quietly teaches it that real clips are out of sync. Both are pinned below.
"""
import json
import random
import struct

import numpy as np
import pytest
import torch

pytest.importorskip("cv2")

from src.data.clipcache import (CACHE_AUDIO_LEN, CACHE_FPS, CACHE_FRAMES, SAMPLE_RATE,
                                CacheError, CacheWriter, ClipCache, decode_blob,
                                encode_clip)


def _frames(n=CACHE_FRAMES, size=224):
    """Smooth gradients, not noise: JPEG is lossy and noise would fail any tolerance."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    out = np.empty((n, size, size, 3), dtype=np.uint8)
    for i in range(n):
        t = i / max(n - 1, 1)
        out[i, :, :, 0] = (255 * (0.2 + 0.6 * x)) .astype(np.uint8)
        out[i, :, :, 1] = (255 * (0.2 + 0.6 * y)) .astype(np.uint8)
        out[i, :, :, 2] = np.uint8(255 * (0.1 + 0.8 * t))
    return out


def _audio(n=CACHE_AUDIO_LEN):
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * 220.0 * t) * 32767).astype(np.int16)


def _blob(**kw):
    kw.setdefault("frames", _frames())
    kw.setdefault("audio", _audio())
    return encode_clip(kw["frames"], kw["audio"], has_video=kw.get("has_video", True),
                       has_audio=kw.get("has_audio", True),
                       duration=kw.get("duration", 6.0), span_start=kw.get("span_start", 0.0))


# ------------------------------------------------------------------ encode / decode
def test_blob_starts_with_the_magic_and_a_readable_header():
    blob = _blob()
    assert blob[:4] == b"DVC2"
    hlen = struct.unpack("<I", blob[4:8])[0]
    hdr = json.loads(blob[8:8 + hlen])
    assert hdr["n"] == CACHE_FRAMES and hdr["size"] == 224 and hdr["sr"] == SAMPLE_RATE


def test_decode_rejects_a_foreign_blob():
    with pytest.raises(CacheError, match="bad magic"):
        decode_blob(b"NOPE" + b"\x00" * 64)


def test_video_round_trips_within_jpeg_tolerance():
    frames = _frames()
    video, _, _, _ = decode_blob(_blob(frames=frames), n_frames=16, window="center")
    assert video.shape == (16, 3, 224, 224)
    assert video.dtype == torch.float32
    assert 0.0 <= float(video.min()) and float(video.max()) <= 1.0
    # centre window of 24 frames taking 16 starts at frame 4
    want = torch.from_numpy(frames[4:20]).permute(0, 3, 1, 2).float() / 255.0
    assert torch.allclose(video, want, atol=0.05), float((video - want).abs().max())


def test_tile_seams_do_not_bleed_between_frames():
    """Chroma subsampling must stay off: neighbours in the grid are unrelated frames.

    Adjacent tiles get wildly different flat colours -- the worst case for cross-tile
    bleed. With 4:2:0 the outer ring of each frame was wrong by up to 0.455; the frame
    border is exactly where blending artefacts live, so it has to be clean.
    """
    rng = np.random.default_rng(0)
    colours = rng.integers(0, 256, size=(CACHE_FRAMES, 3)).astype(np.uint8)
    frames = np.repeat(colours[:, None, None, :], 224, axis=1).repeat(224, axis=2)
    video, _, _, _ = decode_blob(_blob(frames=frames), n_frames=CACHE_FRAMES,
                                 window="center")
    want = torch.from_numpy(colours.astype(np.float32) / 255.0)       # (N, 3)
    ring = torch.cat([video[:, :, :2, :].reshape(CACHE_FRAMES, 3, -1),
                      video[:, :, -2:, :].reshape(CACHE_FRAMES, 3, -1),
                      video[:, :, :, :2].reshape(CACHE_FRAMES, 3, -1),
                      video[:, :, :, -2:].reshape(CACHE_FRAMES, 3, -1)], dim=2)
    err = (ring - want[:, :, None]).abs().max()
    assert err < 0.02, f"tile seams bleed: worst border error {float(err):.3f}"


def test_channels_are_rgb_not_bgr():
    """A red-dominant frame must come back red. Guards the cv2 BGR round trip."""
    f = np.zeros((CACHE_FRAMES, 224, 224, 3), dtype=np.uint8)
    f[:, :, :, 0] = 240                      # pure red in RGB
    video, _, _, _ = decode_blob(_blob(frames=f), n_frames=16, window="center")
    r, g, b = (float(video[:, c].mean()) for c in range(3))
    assert r > 0.8 and g < 0.2 and b < 0.2, (r, g, b)


def test_audio_round_trips_within_int16_quantisation():
    audio_i16 = _audio()
    _, audio, _, _ = decode_blob(_blob(audio=audio_i16), audio_len=64000, window="center")
    assert audio.shape == (64000,)
    o = (CACHE_FRAMES - 16) // 2
    a0 = int(round(o * SAMPLE_RATE / CACHE_FPS))
    want = torch.from_numpy(audio_i16[a0:a0 + 64000].astype(np.float32) / 32768.0)
    assert torch.allclose(audio, want, atol=1e-4)


def test_availability_flags_survive_the_round_trip():
    _, _, hv, ha = decode_blob(_blob(has_video=True, has_audio=False))
    assert hv is True and ha is False


# ------------------------------------------------------------------ alignment
def test_video_and_audio_come_from_the_same_offset():
    """The sync head is trained on this invariant, so it gets its own test.

    Frame k is stamped with its own index in the blue channel and the audio carries an
    impulse at the sample that frame k starts on; after decoding, the first frame's
    stamp must match the position of the first impulse.
    """
    frames = np.zeros((CACHE_FRAMES, 224, 224, 3), dtype=np.uint8)
    for k in range(CACHE_FRAMES):
        frames[k, :, :, 2] = 10 * k          # recoverable after JPEG at this spacing
    audio = np.zeros(CACHE_AUDIO_LEN, dtype=np.int16)
    for k in range(CACHE_FRAMES):
        audio[int(round(k * SAMPLE_RATE / CACHE_FPS))] = 32767

    blob = _blob(frames=frames, audio=audio)
    seen = set()
    for seed in range(12):
        video, aud, _, _ = decode_blob(blob, n_frames=16, audio_len=64000,
                                       window="random", rng=random.Random(seed))
        # Whatever frame the window opened on, that frame's impulse must be the FIRST
        # audio sample returned -- that is what "the same window" means to the sync head.
        assert int(np.argmax(np.abs(aud.numpy()))) == 0, seed
        seen.add(round(float(video[0, 2].mean()) * 255.0 / 10.0))
    assert len(seen) > 1, "every draw opened on the same frame - offsets are not moving"


def test_center_window_is_deterministic_and_random_is_not():
    blob = _blob()
    a = decode_blob(blob, window="center")[0]
    b = decode_blob(blob, window="center")[0]
    assert torch.equal(a, b)
    offsets = {round(float(decode_blob(blob, window="random",
                                       rng=random.Random(s))[0][0, 2].mean()), 6)
               for s in range(40)}
    assert len(offsets) > 1, "random window never moved - augmentation is dead"


def test_a_short_clip_pads_by_holding_the_last_frame():
    video, audio, _, _ = decode_blob(_blob(frames=_frames(n=6), audio=_audio(n=24000)),
                                     n_frames=16, audio_len=64000)
    assert video.shape[0] == 16 and audio.shape == (64000,)
    assert torch.equal(video[-1], video[5]), "padding should repeat the final frame"


# ------------------------------------------------------------------ writer / reader
def test_writer_and_reader_round_trip_many_clips(tmp_path):
    w = CacheWriter(tmp_path, shard_bytes=200_000)     # tiny, to force several shards
    ids = [f"clip_{i:03d}" for i in range(8)]
    for i, cid in enumerate(ids):
        f = _frames()
        f[:, :, :, 2] = i * 20
        w.add(cid, _blob(frames=f))
    w.close()

    assert len(list(tmp_path.glob("shard_*.bin"))) > 1, "shard rolling never happened"
    cache = ClipCache(tmp_path)
    assert len(cache) == len(ids)
    for i, cid in enumerate(ids):
        assert cid in cache
        video, _, _, _ = cache.read(cid, n_frames=16, window="center")
        assert abs(float(video[0, 2].mean()) * 255.0 - i * 20) < 8, cid


def test_reader_names_the_missing_clip(tmp_path):
    w = CacheWriter(tmp_path)
    w.add("present", _blob())
    w.close()
    with pytest.raises(CacheError, match="absent"):
        ClipCache(tmp_path).read("absent")


def test_missing_index_explains_how_to_build_one(tmp_path):
    with pytest.raises(CacheError, match="build_clip_cache"):
        ClipCache(tmp_path)


def test_rebuilding_resumes_instead_of_duplicating(tmp_path):
    w = CacheWriter(tmp_path)
    w.add("a", _blob())
    w.fail("b", "ffmpeg said no")
    w.close()

    w2 = CacheWriter(tmp_path)
    assert "a" in w2 and "b" in w2, "resume must skip both successes and known failures"
    assert "c" not in w2
    w2.add("c", _blob())
    w2.close()

    cache = ClipCache(tmp_path)
    assert set(cache.clips) == {"a", "c"}
    assert cache.failures == {"b": "ffmpeg said no"}


def test_coverage_reports_the_fraction_a_manifest_can_train_on(tmp_path):
    w = CacheWriter(tmp_path)
    w.add("a", _blob())
    w.add("b", _blob())
    w.close()
    cache = ClipCache(tmp_path)
    assert cache.coverage([{"clip_id": "a"}, {"clip_id": "b"}]) == 1.0
    assert cache.coverage([{"clip_id": "a"}, {"clip_id": "zz"}]) == 0.5
    assert cache.coverage([]) == 0.0


# ------------------------------------------------------------------ dataset wiring
def _manifest(tmp_path, ids):
    p = tmp_path / "m.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            f.write(json.dumps({
                "clip_id": cid, "rel_path": f"{cid}.mp4",
                "video_label": i % 2, "audio_label": 0,
                "quadrant": ["RVRA", "FVRA"][i % 2], "duration_sec": 6.0,
                "generator": "real", "dataset": "unit",
            }) + "\n")
    return str(p)


def _built_cache(tmp_path, ids):
    w = CacheWriter(tmp_path / "cache")
    for cid in ids:
        w.add(cid, _blob())
    w.close()
    return str(tmp_path / "cache")


def test_dataset_serves_from_the_cache_without_any_media_root(tmp_path):
    """The whole point: training runs with no root_dir, so ffmpeg is never invoked."""
    from src.data.datasets import AVDeepfakeDataset
    ids = [f"c{i}" for i in range(4)]
    ds = AVDeepfakeDataset(_manifest(tmp_path, ids), cache_root=_built_cache(tmp_path, ids),
                           root_dir=None, train=True)
    assert ds.allow_dummy is False, "a cache is a real media source, not dummy mode"
    s = ds[0]
    assert s["video"].shape == (16, 3, 224, 224)
    assert s["audio"].shape == (64000,)
    assert s["clip_id"] == "c0"
    assert float(s["video"].std()) > 1e-3
    assert int(s["quadrant"]) == 0 and int(s["video_label"]) == 0


def test_preflight_passes_on_a_complete_cache(tmp_path, capsys):
    from src.data.datasets import AVDeepfakeDataset, preflight_check
    ids = [f"c{i}" for i in range(4)]
    ds = AVDeepfakeDataset(_manifest(tmp_path, ids), cache_root=_built_cache(tmp_path, ids),
                           root_dir=None, train=False)
    preflight_check(ds, name="unit")
    assert "coverage 100.00%" in capsys.readouterr().out


def test_preflight_refuses_an_incomplete_cache_with_no_fallback(tmp_path):
    from src.data.datasets import AVDeepfakeDataset, preflight_check
    ids = [f"c{i}" for i in range(10)]
    ds = AVDeepfakeDataset(_manifest(tmp_path, ids), cache_root=_built_cache(tmp_path, ids[:5]),
                           root_dir=None, train=False)
    with pytest.raises(FileNotFoundError, match="build_clip_cache"):
        preflight_check(ds, name="unit")


def test_validation_window_is_deterministic_through_the_dataset(tmp_path):
    """train=False must give evaluation the same frames every epoch."""
    from src.data.datasets import AVDeepfakeDataset
    ids = ["c0"]
    root = _built_cache(tmp_path, ids)
    m = _manifest(tmp_path, ids)
    a = AVDeepfakeDataset(m, cache_root=root, root_dir=None, train=False)[0]["video"]
    b = AVDeepfakeDataset(m, cache_root=root, root_dir=None, train=False)[0]["video"]
    assert torch.equal(a, b)
