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
QUADRANT_MAP = {
    "real_video_real_audio": ("RVRA", 0, 0),
    "real_video_fake_audio": ("RVFA", 0, 1),
    "fake_video_real_audio": ("FVRA", 1, 0),
    "fake_video_fake_audio": ("FVFA", 1, 1),
}


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


# ═══════════════════════════════════════════════════════════════════════
# DFDC-10
# ═══════════════════════════════════════════════════════════════════════

def build_dfdc(root: str) -> list[dict]:
    """DFDC: dfdc_train_part_XX/dfdc_train_part_X/<clip>.mp4 + <clip>.json.

    JSON contains: original (path to real), label (0=real, 1=fake), split.
    No per-modality labels — both video and audio are fake together (FVFA for fakes).
    """
    root = Path(root)
    records = []

    # Find the actual data dir (may be nested)
    data_dir = root
    for candidate in [root / "dfdc_train_part_00", root / "dfdc_train_part_0"]:
        if candidate.exists():
            data_dir = candidate
            break

    # Also check for a parent wrapper
    for d in root.rglob("dfdc_train_part_*"):
        if d.is_dir() and any(d.glob("*.mp4")):
            data_dir = d.parent
            break

    for mp4 in sorted(data_dir.rglob("*.mp4")):
        clip_id = mp4.stem
        json_path = mp4.with_suffix(".json")

        label = 1  # default: fake
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
            label = meta.get("label", 1)

        if label == 0:
            quadrant = "RVRA"
            v_label, a_label = 0, 0
        else:
            quadrant = "FVFA"
            v_label, a_label = 1, 1

        rel = mp4.relative_to(root).as_posix()
        records.append({
            "clip_id": rel.replace("/", "__").rsplit(".", 1)[0],
            "rel_path": rel,
            "video_label": v_label,
            "audio_label": a_label,
            "quadrant": quadrant,
            "video_segments": [[0.0, WHOLE_CLIP]] if v_label else [],
            "audio_segments": [[0.0, WHOLE_CLIP]] if a_label else [],
            "generator": "dfdc" if label else "real",
            "dataset": "dfdc-10",
            "identity": clip_id,
            "meta": {},
        })

    return records


# ═══════════════════════════════════════════════════════════════════════
# DeepFakeTIMIT
# ═══════════════════════════════════════════════════════════════════════

def build_deepfaketimit(root: str) -> list[dict]:
    """DeepFakeTIMIT: higher_quality/<id>/<clip>-video-fram1.avi + <clip>.wav.

    Paired video+audio. All clips are fake (face-swap). Real originals are
    from VCTK corpus (not included). Treat all as FVFA for cross-dataset eval.
    """
    root = Path(root)
    records = []

    # Find higher_quality dir
    hq_dir = root
    for candidate in [root / "DeepfakeTIMIT" / "higher_quality", root / "higher_quality"]:
        if candidate.exists():
            hq_dir = candidate
            break

    # Group by identity folder
    for identity_dir in sorted(hq_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        identity = identity_dir.name

        # Find video files
        for avi in sorted(identity_dir.glob("*.avi")):
            clip_name = avi.stem.replace("-video-fram1", "")
            wav = identity_dir / f"{clip_name}.wav"

            rel = avi.relative_to(root).as_posix()
            records.append({
                "clip_id": rel.replace("/", "__").rsplit(".", 1)[0],
                "rel_path": rel,
                "video_label": 1,
                "audio_label": 1,
                "quadrant": "FVFA",
                "video_segments": [[0.0, WHOLE_CLIP]],
                "audio_segments": [[0.0, WHOLE_CLIP]],
                "generator": "deepfaketimit",
                "dataset": "deepfaketimit",
                "identity": identity,
                "meta": {"wav_path": str(wav.relative_to(root)) if wav.exists() else ""},
            })

    return records


# ═══════════════════════════════════════════════════════════════════════
# Celeb-DF v2
# ═══════════════════════════════════════════════════════════════════════

def build_celebdf(root: str) -> list[dict]:
    """Celeb-DF v2: Celeb-real/<id>_<clip>.mp4 + Celeb-synthesis/<id>_<clip>.mp4.

    Video-only dataset. Audio is from the original video (not synthesized).
    Real = RVRA, Fake = FVRA (fake video, real audio).
    """
    root = Path(root)
    records = []

    # Find dirs
    real_dir = None
    fake_dir = None
    for d in root.iterdir():
        if not d.is_dir():
            continue
        name = d.name.lower()
        if "real" in name:
            real_dir = d
        elif "synthesis" in name or "fake" in name:
            fake_dir = d

    # Real clips
    if real_dir:
        for mp4 in sorted(real_dir.glob("*.mp4")):
            rel = mp4.relative_to(root).as_posix()
            identity = mp4.stem.split("_")[0] if "_" in mp4.stem else mp4.stem
            records.append({
                "clip_id": rel.replace("/", "__").rsplit(".", 1)[0],
                "rel_path": rel,
                "video_label": 0,
                "audio_label": 0,
                "quadrant": "RVRA",
                "video_segments": [],
                "audio_segments": [],
                "generator": "real",
                "dataset": "celeb-df-v2",
                "identity": identity,
                "meta": {},
            })

    # Fake clips
    if fake_dir:
        for mp4 in sorted(fake_dir.glob("*.mp4")):
            rel = mp4.relative_to(root).as_posix()
            identity = mp4.stem.split("_")[0] if "_" in mp4.stem else mp4.stem
            records.append({
                "clip_id": rel.replace("/", "__").rsplit(".", 1)[0],
                "rel_path": rel,
                "video_label": 1,
                "audio_label": 0,
                "quadrant": "FVRA",
                "video_segments": [[0.0, WHOLE_CLIP]],
                "audio_segments": [],
                "generator": "celeb-df",
                "dataset": "celeb-df-v2",
                "identity": identity,
                "meta": {},
            })

    return records


# ═══════════════════════════════════════════════════════════════════════
# ASVspoof 2019 LA
# ═══════════════════════════════════════════════════════════════════════

def build_asvpoof2019(root: str) -> list[dict]:
    """ASVspoof 2019 LA: protocol files + FLAC audio.

    Protocol format: <speaker> <utterance> <attack> <label>
    Labels: bonafide (real) or spoof (fake).
    Audio-only — no video. Map to: RVRA (bonafide) or RVFA (spoof).
    """
    root = Path(root)
    records = []

    # Find protocol files
    proto_dir = root
    for candidate in [root / "LA" / "ASVspoof2019_LA_asv_protocols",
                      root / "ASVspoof2019_LA_asv_protocols"]:
        if candidate.exists():
            proto_dir = candidate
            break

    # Find audio dirs
    audio_dirs = {}
    for d in root.rglob("flac"):
        if d.is_dir():
            audio_dirs[d.parent.name] = d.parent
    if not audio_dirs:
        for d in root.rglob("*"):
            if d.is_dir() and any(d.glob("*.flac")):
                audio_dirs[d.name] = d

    # Parse train/dev/eval protocols
    for proto_file in sorted(proto_dir.glob("*.txt")):
        if "trn" in proto_file.name:
            continue  # skip train-only lists
        with open(proto_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                speaker, utterance, attack, label = parts[0], parts[1], parts[2], parts[3]

                # Find the flac file
                flac_name = f"{utterance}.flac"
                flac_path = None
                for ad in audio_dirs.values():
                    candidate = ad / flac_name
                    if candidate.exists():
                        flac_path = candidate
                        break

                if not flac_path:
                    continue

                rel = flac_path.relative_to(root).as_posix()
                is_real = label == "bonafide"

                records.append({
                    "clip_id": f"asvpoof_{utterance}",
                    "rel_path": rel,
                    "video_label": 0,  # no video
                    "audio_label": 0 if is_real else 1,
                    "quadrant": "RVRA" if is_real else "RVFA",
                    "video_segments": [],
                    "audio_segments": [[0.0, WHOLE_CLIP]] if not is_real else [],
                    "generator": "real" if is_real else f"asvpoof_{attack}",
                    "dataset": "asvpoof-2019",
                    "identity": speaker,
                    "meta": {"attack": attack, "label": label},
                })

    return records


# ═══════════════════════════════════════════════════════════════════════
# In-the-Wild Audio Deepfake
# ═══════════════════════════════════════════════════════════════════════

def build_inthewild(root: str) -> list[dict]:
    """In-the-Wild: release_in_the_wild/fake/<N>.wav + release_in_the_wild/real/<N>.wav.

    Audio-only. meta.csv has columns: file, label (real/fake).
    """
    root = Path(root)
    records = []

    # Find release dir
    release_dir = root
    for candidate in [root / "release_in_the_wild", root]:
        if candidate.exists() and any((candidate / d).exists() for d in ["fake", "real"]):
            release_dir = candidate
            break

    # Load meta.csv if available
    meta_csv = root / "meta.csv"
    if not meta_csv.exists():
        meta_csv = root / "modified_meta.csv"

    labels = {}
    if meta_csv.exists():
        with open(meta_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("file", row.get("filename", ""))
                label = row.get("label", row.get("class", "")).lower()
                labels[fname] = label

    # Scan wav files
    for wav in sorted(release_dir.rglob("*.wav")):
        fname = wav.name
        rel = wav.relative_to(root).as_posix()

        # Determine label from folder name or meta
        label = labels.get(fname, "")
        if not label:
            if "fake" in str(wav.parent).lower():
                label = "fake"
            elif "real" in str(wav.parent).lower():
                label = "real"
            else:
                label = "unknown"

        is_real = label == "real"

        records.append({
            "clip_id": f"inthewild_{wav.stem}",
            "rel_path": rel,
            "video_label": 0,  # no video
            "audio_label": 0 if is_real else 1,
            "quadrant": "RVRA" if is_real else "RVFA",
            "video_segments": [],
            "audio_segments": [[0.0, WHOLE_CLIP]] if not is_real else [],
            "generator": "real" if is_real else "in-the-wild",
            "dataset": "in-the-wild",
            "identity": wav.stem,
            "meta": {},
        })

    return records


# ═══════════════════════════════════════════════════════════════════════
# WaveFake
# ═══════════════════════════════════════════════════════════════════════

def build_wavefake(root: str) -> list[dict]:
    """WaveFake: <model>/<model>/<clip>.flac. All clips are fake (generated audio).

    Audio-only. No real clips — use for audio-branch augmentation only.
    Map to RVFA (real video + fake audio) since we pair with real video.
    """
    root = Path(root)
    records = []

    for flac in sorted(root.rglob("*.flac")):
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
            "generator": model,
            "dataset": "wavefake",
            "identity": flac.stem,
            "meta": {"model": model},
        })

    return records


# ═══════════════════════════════════════════════════════════════════════
# LAV-DF (placeholder — needs extraction first)
# ═══════════════════════════════════════════════════════════════════════

def build_lavdf(root: str) -> list[dict]:
    """LAV-DF: multi-part zips. Extract first, then convert.

    This is a placeholder. After extraction, the structure should be:
    <root>/<clip>.mp4 with metadata CSV for temporal localization.
    """
    root = Path(root)
    records = []

    # Check if extracted
    mp4s = list(root.rglob("*.mp4"))
    if not mp4s:
        print("WARNING: LAV-DF not extracted yet. Run extraction first.")
        return records

    # Try to find metadata
    meta_csv = None
    for csv_file in root.rglob("*.csv"):
        if "meta" in csv_file.name.lower():
            meta_csv = csv_file
            break

    for mp4 in sorted(mp4s):
        rel = mp4.relative_to(root).as_posix()
        records.append({
            "clip_id": rel.replace("/", "__").rsplit(".", 1)[0],
            "rel_path": rel,
            "video_label": 1,  # assume fake
            "audio_label": 1,
            "quadrant": "FVFA",
            "video_segments": [[0.0, WHOLE_CLIP]],
            "audio_segments": [[0.0, WHOLE_CLIP]],
            "generator": "lavdf",
            "dataset": "lav-df",
            "identity": mp4.stem,
            "meta": {},
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
