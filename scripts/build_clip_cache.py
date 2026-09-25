"""Decode every clip in a manifest ONCE into the packed cache src/data/clipcache.py reads.

Run this on a CPU session (no GPU quota spent). It is resumable: re-running skips clips
already in the index, so a session that hits the 12 h wall can be continued by simply
launching it again.

    python -m scripts.build_clip_cache \
        --manifest /kaggle/working/splits_verified/fakeavceleb/train.jsonl \
                   /kaggle/working/splits_verified/fakeavceleb/val.jsonl \
                   /kaggle/working/splits_verified/fakeavceleb/test.jsonl \
        --root /kaggle/input/.../FakeAVCeleb_v1.2 \
        --out /kaggle/working/av_cache --workers 4

Memory is the whole point of this file, so it is bounded on purpose:
  * decoding happens in a pool of separate processes that are recycled every
    `--maxtasksperchild` clips, so a leak in ffmpeg or cv2 cannot accumulate;
  * results stream back through `imap_unordered` with a small chunksize, so at most a
    handful of ~450 KB blobs are in flight;
  * the writer fsyncs and drops each finished shard from page cache.
The build therefore holds a roughly constant footprint no matter how many clips it has
processed, which is what the training run failed to do.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clipcache import (CACHE_AUDIO_LEN, CACHE_FRAMES, CACHE_SECONDS,  # noqa: E402
                                FRAME_SIZE, CacheWriter, encode_clip)
from src.data.datasets import load_manifest, resolve_root_dir  # noqa: E402

_ROOTS: list[Path] = []


def _init(roots):
    global _ROOTS
    _ROOTS = [Path(r) for r in roots]
    # One thread per worker process. cv2/ffmpeg both default to using every core, which
    # on a 4-worker pool means 16 threads fighting over 4 CPUs and a much larger RSS.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001 - cv2 is optional in the ffmpeg path
        pass


def _find(rel: str):
    for r in _ROOTS:
        p = r / rel
        if p.exists():
            return p
    return None


def _one(rec):
    """Decode one record into (clip_id, blob, None) or (clip_id, None, reason)."""
    from src.data.decode import DecodeError, decode_clip
    cid = rec.get("clip_id", "?")
    rel = rec.get("rel_path")
    if not rel:
        return cid, None, "record has no rel_path"
    media = _find(rel)
    if media is None:
        return cid, None, f"not found under any root: {rel}"
    try:
        # A center window of the full cached span; `decode_blob` draws the model's
        # narrower window out of it at training time, which is where the randomness
        # belongs -- baking it in here would freeze one crop for the whole campaign.
        d = decode_clip(str(media), CACHE_FRAMES, CACHE_AUDIO_LEN, FRAME_SIZE,
                        window="center")
    except DecodeError as e:
        return cid, None, str(e)
    except Exception as e:  # noqa: BLE001 - a surprise here must not kill the pool
        return cid, None, f"{type(e).__name__}: {e}"

    frames = (d.video.clamp(0, 1) * 255.0).round().byte().numpy()   # (N, 3, H, W)
    frames = np.ascontiguousarray(frames.transpose(0, 2, 3, 1))     # (N, H, W, 3) RGB
    audio = (d.audio.clamp(-1, 1).numpy() * 32767.0).astype(np.int16)
    try:
        blob = encode_clip(frames, audio, has_video=d.has_video, has_audio=d.has_audio,
                           duration=float(rec.get("duration_sec", CACHE_SECONDS)),
                           span_start=float(d.window[0]))
    except Exception as e:  # noqa: BLE001
        return cid, None, f"encode failed: {type(e).__name__}: {e}"
    return cid, blob, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--root", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--maxtasksperchild", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="stop after N new clips (smoke test)")
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="stop cleanly after this many minutes so the index is never lost")
    args = ap.parse_args()

    records, seen = [], set()
    for m in args.manifest:
        for r in load_manifest(m):
            cid = r.get("clip_id")
            if cid and cid not in seen:
                seen.add(cid)
                records.append(r)
    roots = [str(resolve_root_dir(r, records)) for r in args.root]
    print(f"{len(records)} unique clips from {len(args.manifest)} manifest(s)")
    print(f"roots: {roots}")

    writer = CacheWriter(args.out)
    todo = [r for r in records if r.get("clip_id") not in writer]
    if len(todo) < len(records):
        print(f"resuming: {len(records) - len(todo)} already cached, {len(todo)} to go")
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("cache is already complete")
        writer.close()
        return _report(writer, records, args.out)

    t0 = time.time()
    deadline = t0 + args.time_budget_min * 60 if args.time_budget_min else None
    done = failed = 0
    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    pool = ctx.Pool(args.workers, initializer=_init, initargs=(roots,),
                    maxtasksperchild=args.maxtasksperchild)
    try:
        for cid, blob, err in pool.imap_unordered(_one, todo, chunksize=4):
            if err is None:
                writer.add(cid, blob)
                done += 1
            else:
                writer.fail(cid, err)
                failed += 1
                if failed <= 10:
                    print(f"  FAILED {cid}: {err[:160]}", flush=True)
            n = done + failed
            if n % 250 == 0:
                el = time.time() - t0
                rate = n / el
                writer.flush_index()          # cheap insurance against losing the run
                print(f"  {n}/{len(todo)}  {rate:.1f} clips/s  "
                      f"eta {(len(todo) - n) / max(rate, 1e-6) / 60:.0f} min  "
                      f"failed={failed}", flush=True)
            if deadline and time.time() > deadline:
                print(f"time budget reached after {n} clips -- stopping cleanly; "
                      "re-run to continue", flush=True)
                pool.terminate()
                break
    finally:
        try:
            pool.close()
        except ValueError:      # already terminated by the time budget
            pass
        pool.join()
        writer.close()

    print(f"\nwrote {done} clips, {failed} failed, in {(time.time() - t0) / 60:.1f} min")
    return _report(writer, records, args.out)


def _report(writer, records, out):
    root = Path(out)
    shards = sorted(root.glob("shard_*.bin"))
    total = sum(p.stat().st_size for p in shards)
    have = sum(r.get("clip_id") in writer.index for r in records)
    print(f"cache: {len(writer.index)} clips, {len(shards)} shards, {total / 1e9:.2f} GB")
    print(f"manifest coverage: {have}/{len(records)} = {100.0 * have / max(len(records), 1):.2f}%")
    if writer.failures:
        p = root / "failures.json"
        p.write_text(json.dumps(writer.failures, indent=1), encoding="utf-8")
        print(f"{len(writer.failures)} failures listed in {p}")
    if have < len(records):
        print("INCOMPLETE -- re-run this script to fill the gaps before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
