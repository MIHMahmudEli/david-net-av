"""Preprocessing: raw clip -> (video frames tensor, audio waveform tensor) shards.

This is a scaffold. The real implementation should:
  video: ffmpeg decode @ fixed fps -> RetinaFace/MediaPipe face detect+track ->
         align to canonical landmarks -> 224x224 face crop + 96x96 mouth crop ->
         save as .pt shards keyed by clip_id.
  audio: extract track -> resample 16 kHz mono -> loudness normalize ->
         save raw waveform (+ optional log-mel) as .pt.

Run:
  python -m src.data.preprocess --manifest src/data/manifests/fakeavceleb.jsonl \
      --raw-root data/fakeavceleb --out data/shards/fakeavceleb
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.datasets import load_manifest


def preprocess_clip(raw_path: str, n_frames: int, sr: int, audio_len: int):
    """Return (video_tensor[T,3,224,224], audio_tensor[audio_len]).

    Placeholder — wire ffmpeg + face detector + torchaudio here.
    """
    import torch  # lazy
    # TODO: real decode/crop/resample. Emitting shapes for now.
    return torch.zeros(n_frames, 3, 224, 224), torch.zeros(audio_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--audio-len", type=int, default=64000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest)
    import torch
    for rec in records:
        raw = Path(args.raw_root) / rec.get("rel_path", rec["clip_id"] + ".mp4")
        v, a = preprocess_clip(str(raw), args.n_frames, args.sr, args.audio_len)
        torch.save(v, out / f"{rec['clip_id']}_video.pt")
        torch.save(a, out / f"{rec['clip_id']}_audio.pt")
    print(f"wrote {len(records)} clips to {out}")


if __name__ == "__main__":
    main()
