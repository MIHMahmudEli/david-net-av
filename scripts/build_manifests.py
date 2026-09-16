"""Dataset converters -> unified JSONL manifest (docs/03_datasets.md §4).

Each converter reads a raw dataset directory and emits records matching the
unified schema. Run from repo root:
    python scripts/build_manifests.py --dataset dfdc-10 --root /kaggle/input/... --out manifests/dfdc-10.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


WHOLE_CLIP = 9999.0


def _write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _subject_disjoint_splits(records: list[dict], seed: int = 42,
                              fractions=(0.7, 0.1, 0.2)) -> dict[str, list[dict]]:
    by_id = defaultdict(list)
    for r in records:
        by_id[r.get("identity", f"anon_{abs(hash(r['clip_id'])) % 10**8}")].append(r)
    idents = sorted(by_id)
    import random
    random.Random(seed).shuffle(idents)
    total = len(records)
    targets = [f * total for f in fractions]
    splits = {"train": [], "val": [], "test": []}
    names = list(splits)
    k, filled = 0, 0.0
    for ident in idents:
        clips = by_id[ident]
        if k < 2 and filled + len(clips) > targets[k] and splits[names[k]]:
            k += 1
            filled = 0.0
        splits[names[k]].extend(clips)
        filled += len(clips)
    return splits


def _find_files(root: Path, ext: str) -> list[Path]:
    """Recursively find all files with given extension."""
    return sorted(root.rglob(f"*{ext}"))


# ═══════════════════════════════════════════════════════════════════════
# DFDC-10
# ═══════════════════════════════════════════════════════════════════════

def build_dfdc(root: str) -> list[dict]:
    """DFDC: <clip>.mp4 + <clip>.json in nested dfdc_train_part_XX dirs."""
    root = Path(root)
    records = []
    for mp4 in _find_files(root, ".mp4"):
        clip_id = mp4.stem
        json_path = mp4.with_suffix(".json")
        label = 1
        if json_path.exists():
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                label = meta.get("label", 1)
            except Exception:
                pass
        v_label, a_label = (0, 0) if label == 0 else (1, 1)
        quadrant = "RVRA" if label == 0 else "FVFA"
        rel = mp4.relative_to(root).as_posix()
        records.append({
            "clip_id": rel.replace("/", "__").rsplit(".", 1)[0],
            "rel_path": rel,
            "video_label": v_label, "audio_label": a_label,
            "quadrant": quadrant,
            "video_segments": [[0.0, WHOLE_CLIP]] if v_label else [],
            "audio_segments": [[0.0, WHOLE_CLIP]] if a_label else [],
            "generator": "dfdc" if label else "real",
            "dataset": "dfdc-10",
            "identity": clip_id, "meta": {},
        })
    return records


# ═══════════════════════════════════════════════════════════════════════
# DeepFakeTIMIT
# ═══════════════════════════════════════════════════════════════════════

def build_deepfaketimit(root: str) -> list[dict]:
    """DeepFakeTIMIT: paired video+audio, all fake (face-swap).

    Searches for .avi and .mp4 video files recursively.
    All clips are FVFA (face-swap = both modalities fake).
    """
    root = Path(root)
    records = []

    # Find video files (avi or mp4)
    video_files = _find_files(root, ".avi") + _find_files(root, ".mp4")
    for vid in video_files:
        # Extract identity from parent directory
        identity = vid.parent.name if vid.parent != root else "unknown"
        rel = vid.relative_to(root).as_posix()
        records.append({
            "clip_id": rel.replace("/", "__").rsplit(".", 1)[0],
            "rel_path": rel,
            "video_label": 1, "audio_label": 1,
            "quadrant": "FVFA",
            "video_segments": [[0.0, WHOLE_CLIP]],
            "audio_segments": [[0.0, WHOLE_CLIP]],
            "generator": "deepfaketimit",
            "dataset": "deepfaketimit",
            "identity": identity, "meta": {},
        })

    return records


# ═══════════════════════════════════════════════════════════════════════
# Celeb-DF v2
# ═══════════════════════════════════════════════════════════════════════

def build_celebdf(root: str) -> list[dict]:
    """Celeb-DF v2: video-only. Real=RVRA, Fake=FVRA (face-swap, real audio)."""
    root = Path(root)
    records = []
    for mp4 in _find_files(root, ".mp4"):
        rel = mp4.relative_to(root).as_posix()
        identity = mp4.stem.split("_")[0] if "_" in mp4.stem else mp4.stem
        # Determine real vs fake from path
        path_lower = str(mp4).lower()
        is_fake = "synthesis" in path_lower or "fake" in path_lower
        records.append({
            "clip_id": rel.replace("/", "__").rsplit(".", 1)[0],
            "rel_path": rel,
            "video_label": 1 if is_fake else 0,
            "audio_label": 0,  # audio is always real in Celeb-DF
            "quadrant": "FVRA" if is_fake else "RVRA",
            "video_segments": [[0.0, WHOLE_CLIP]] if is_fake else [],
            "audio_segments": [],
            "generator": "celeb-df" if is_fake else "real",
            "dataset": "celeb-df-v2",
            "identity": identity, "meta": {},
        })
    return records


# ═══════════════════════════════════════════════════════════════════════
# ASVspoof 2019 LA
# ═══════════════════════════════════════════════════════════════════════

def build_asvpoof2019(root: str) -> list[dict]:
    """ASVspoof 2019 LA: protocol files + FLAC audio. Audio-only.

    Searches for protocol .txt files and .flac audio recursively.
    """
    root = Path(root)
    records = []

    # Find all .flac files
    flac_files = _find_files(root, ".flac")

    # Try to find protocol files for labels
    proto_labels = {}
    for txt in _find_files(root, ".txt"):
        try:
            with open(txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        _, utterance, _, label = parts[0], parts[1], parts[2], parts[3]
                        proto_labels[utterance] = label
        except Exception:
            continue

    for flac in flac_files:
        utterance = flac.stem
        rel = flac.relative_to(root).as_posix()

        # Get label from protocol or infer from path
        label = proto_labels.get(utterance, "")
        if not label:
            path_lower = str(flac).lower()
            if "bonafide" in path_lower or "real" in path_lower:
                label = "bonafide"
            elif "spoof" in path_lower or "fake" in path_lower:
                label = "spoof"
            else:
                label = "spoof"  # default: most ASVspoof clips are spoof

        is_real = label == "bonafide"
        # Extract speaker from path or filename
        speaker = flac.parent.name if flac.parent != root else utterance.split("_")[0]

        records.append({
            "clip_id": f"asvpoof_{utterance}",
            "rel_path": rel,
            "video_label": 0,  # no video
            "audio_label": 0 if is_real else 1,
            "quadrant": "RVRA" if is_real else "RVFA",
            "video_segments": [],
            "audio_segments": [[0.0, WHOLE_CLIP]] if not is_real else [],
            "generator": "real" if is_real else "asvpoof",
            "dataset": "asvpoof-2019",
            "identity": speaker,
            "meta": {"label": label},
        })

    return records


# ═══════════════════════════════════════════════════════════════════════
# In-the-Wild Audio Deepfake
# ═══════════════════════════════════════════════════════════════════════

def build_inthewild(root: str) -> list[dict]:
    """In-the-Wild: .wav files in real/ and fake/ subdirs. Audio-only."""
    root = Path(root)
    records = []

    wav_files = _find_files(root, ".wav")
    for wav in wav_files:
        rel = wav.relative_to(root).as_posix()
        # Determine label from path
        path_lower = str(wav).lower()
        if "real" in path_lower:
            is_real = True
        elif "fake" in path_lower:
            is_real = False
        else:
            # Try parent dir name
            is_real = "real" in wav.parent.name.lower()

        records.append({
            "clip_id": f"inthewild_{wav.stem}",
            "rel_path": rel,
            "video_label": 0,
            "audio_label": 0 if is_real else 1,
            "quadrant": "RVRA" if is_real else "RVFA",
            "video_segments": [],
            "audio_segments": [[0.0, WHOLE_CLIP]] if not is_real else [],
            "generator": "real" if is_real else "in-the-wild",
            "dataset": "in-the-wild",
            "identity": wav.stem, "meta": {},
        })

    return records


# ═══════════════════════════════════════════════════════════════════════
# WaveFake
# ═══════════════════════════════════════════════════════════════════════

def build_wavefake(root: str) -> list[dict]:
    """WaveFake: .flac files. All fake (generated audio). Audio-only."""
    root = Path(root)
    records = []

    for flac in _find_files(root, ".flac"):
        rel = flac.relative_to(root).as_posix()
        parts = flac.relative_to(root).parts
        model = parts[0] if len(parts) > 0 else "wavefake"

        records.append({
            "clip_id": f"wavefake_{flac.stem}",
            "rel_path": rel,
            "video_label": 0,  # no video — pair with real
            "audio_label": 1,
            "quadrant": "RVFA",
            "video_segments": [],
            "audio_segments": [[0.0, WHOLE_CLIP]],
            "generator": model, "dataset": "wavefake",
            "identity": flac.stem, "meta": {"model": model},
        })

    return records


# ═══════════════════════════════════════════════════════════════════════
# LAV-DF
# ═══════════════════════════════════════════════════════════════════════

def build_lavdf(root: str) -> list[dict]:
    """LAV-DF: .mp4 files with temporal localization labels.

    Searches for mp4s and any CSV metadata. If no metadata found,
    marks entire clip as manipulated.
    """
    root = Path(root)
    records = []

    mp4s = _find_files(root, ".mp4")
    if not mp4s:
        print("WARNING: No .mp4 files found in LAV-DF root")
        return records

    # Try to load metadata CSV
    meta_data = {}
    for csv_file in _find_files(root, ".csv"):
        try:
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try common column names
                    fname = row.get("video", row.get("filename", row.get("file", "")))
                    if fname:
                        meta_data[fname] = row
        except Exception:
            continue

    for mp4 in mp4s:
        rel = mp4.relative_to(root).as_posix()
        fname = mp4.name

        # Check metadata for temporal labels
        m = meta_data.get(fname, {})
        v_segs = [[0.0, WHOLE_CLIP]]
        a_segs = [[0.0, WHOLE_CLIP]]

        # If we have start/end times, use them
        if m:
            try:
                vs = float(m.get("video_start", 0))
                ve = float(m.get("video_end", WHOLE_CLIP))
                als = float(m.get("audio_start", 0))
                ale = float(m.get("audio_end", WHOLE_CLIP))
                if ve > vs:
                    v_segs = [[vs, ve]]
                if ale > als:
                    a_segs = [[als, ale]]
            except (ValueError, TypeError):
                pass

        records.append({
            "clip_id": rel.replace("/", "__").rsplit(".", 1)[0],
            "rel_path": rel,
            "video_label": 1, "audio_label": 1,
            "quadrant": "FVFA",
            "video_segments": v_segs,
            "audio_segments": a_segs,
            "generator": "lavdf",
            "dataset": "lav-df",
            "identity": mp4.stem, "meta": {},
        })

    return records


# ═══════════════════════════════════════════════════════════════════════
# AV-Deepfake1M
# ═══════════════════════════════════════════════════════════════════════

def build_avdeepfake1m(root: str) -> list[dict]:
    """AV-Deepfake1M: .mp4 files with temporal localization labels.

    Searches for mp4s, CSV metadata, and JSON annotations.
    All clips are AV deepfakes with both modalities manipulated.
    """
    root = Path(root)
    records = []

    mp4s = _find_files(root, ".mp4")
    if not mp4s:
        print("WARNING: No .mp4 files found in AV-Deepfake1M root")
        return records

    # Try to load metadata from CSV or JSON
    meta_data = {}
    for csv_file in _find_files(root, ".csv"):
        try:
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("video", row.get("filename", row.get("file", "")))
                    if fname:
                        meta_data[fname] = row
        except Exception:
            continue

    for mp4 in mp4s:
        rel = mp4.relative_to(root).as_posix()
        fname = mp4.name

        m = meta_data.get(fname, {})
        v_segs = [[0.0, WHOLE_CLIP]]
        a_segs = [[0.0, WHOLE_CLIP]]

        if m:
            try:
                vs = float(m.get("video_start", 0))
                ve = float(m.get("video_end", WHOLE_CLIP))
                als = float(m.get("audio_start", 0))
                ale = float(m.get("audio_end", WHOLE_CLIP))
                if ve > vs:
                    v_segs = [[vs, ve]]
                if ale > als:
                    a_segs = [[als, ale]]
            except (ValueError, TypeError):
                pass

        records.append({
            "clip_id": rel.replace("/", "__").rsplit(".", 1)[0],
            "rel_path": rel,
            "video_label": 1, "audio_label": 1,
            "quadrant": "FVFA",
            "video_segments": v_segs,
            "audio_segments": a_segs,
            "generator": "av-deepfake1m",
            "dataset": "av-deepfake1m",
            "identity": mp4.stem, "meta": {},
        })

    return records


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

BUILDERS = {
    "fakeavceleb": None,  # use existing build_manifest.py
    "dfdc-10": build_dfdc,
    "deepfaketimit": build_deepfaketimit,
    "celeb-df-v2": build_celebdf,
    "asvpoof-2019": build_asvpoof2019,
    "in-the-wild": build_inthewild,
    "wavefake": build_wavefake,
    "lav-df": build_lavdf,
}


def main():
    ap = argparse.ArgumentParser(description="Build unified manifests from raw datasets")
    ap.add_argument("--dataset", required=True, choices=list(BUILDERS.keys()))
    ap.add_argument("--root", required=True, help="Raw dataset root directory")
    ap.add_argument("--out", required=True, help="Output manifest .jsonl path")
    ap.add_argument("--splits-dir", default=None, help="Also write train/val/test splits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    builder = BUILDERS[args.dataset]
    if builder is None:
        print(f"ERROR: Use scripts/build_manifest.py for {args.dataset}")
        return

    print(f"Building {args.dataset} manifest from {args.root}...")
    records = builder(args.root)

    if not records:
        print("No records found. Check dataset structure.")
        return

    _write_jsonl(Path(args.out), records)

    counts = Counter(r["quadrant"] for r in records)
    print(f"Manifest: {len(records)} clips -> {args.out}")
    print(f"  {dict(counts)}")

    if args.splits_dir:
        splits = _subject_disjoint_splits(records, args.seed)
        d = Path(args.splits_dir)
        for name, recs in splits.items():
            _write_jsonl(d / f"{name}.jsonl", recs)
            print(f"  {name}: {len(recs)} clips")


if __name__ == "__main__":
    main()
