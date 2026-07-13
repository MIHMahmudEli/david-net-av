"""FakeAVCeleb -> unified manifest converter + split generator.

Converts the FakeAVCeleb directory tree (and its meta_data.csv when present) into
the unified JSONL schema of docs/03_datasets.md §4, then writes reproducible
subject-disjoint and leave-one-generator-out (LOGO) splits.

Expected FakeAVCeleb layout (v1.2):
    <root>/RealVideo-RealAudio/<race>/<gender>/<idXXXXX>/<clip>.mp4
    <root>/RealVideo-FakeAudio/...
    <root>/FakeVideo-RealAudio/...
    <root>/FakeVideo-FakeAudio/...
    <root>/meta_data.csv          (optional: method/race/gender enrichment)

Usage:
    python scripts/build_manifest.py --root data/fakeavceleb \
        --out src/data/manifests/fakeavceleb.jsonl \
        --splits-dir src/data/splits/fakeavceleb --seed 42

Conventions:
  * clip_id  = relative path with separators flattened to "__" (unique, stable).
  * identity = first "idXXXXX" segment in the path (drives subject-disjoint splits;
    the same identity never appears in two different splits).
  * FakeAVCeleb manipulations are whole-clip: fake streams get segments
    [[0.0, WHOLE_CLIP]] where WHOLE_CLIP is a large sentinel that
    segments_to_mask() clips to the actual length -> an all-ones frame mask.
  * LOGO split for generator g:  test  = clips made with g + real (RVRA) clips of
    test identities (AUC needs both classes);  train = everything else with g and
    those test reals removed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

WHOLE_CLIP = 9999.0   # sentinel end-time meaning "the entire clip is manipulated"

QUAD_DIRS = {
    "RealVideo-RealAudio": ("RVRA", 0, 0),
    "RealVideo-FakeAudio": ("RVFA", 0, 1),
    "FakeVideo-RealAudio": ("FVRA", 1, 0),
    "FakeVideo-FakeAudio": ("FVFA", 1, 1),
}

# generator inference from path fragments when meta_data.csv is absent
_PATH_GENERATORS = ("faceswap-wav2lip", "fsgan-wav2lip", "wav2lip", "faceswap",
                    "fsgan", "rtvc")

_ID_RE = re.compile(r"(id\d+)")


def _load_meta_csv(root: Path) -> dict:
    """meta_data.csv (if present) -> lookup table for method/race/gender.

    Keys: the row's full relative path (authoritative), plus bare filename ONLY
    when that filename is unique in the csv — FakeAVCeleb clip names like
    00000.mp4 repeat across quadrant folders, so ambiguous filenames are dropped
    rather than risking cross-quadrant metadata pollution.
    """
    path = root / "meta_data.csv"
    if not path.exists():
        return {}
    by_path, by_name, name_collisions = {}, {}, set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            meta = {"method": row.get("method", ""),
                    "race": row.get("race", ""),
                    "gender": row.get("gender", "")}
            name = row.get("filename") or row.get("path", "").replace("\\", "/").split("/")[-1]
            rel = row.get("path", "").replace("\\", "/").strip("/")
            if rel:
                if name and not rel.endswith(name):
                    rel = f"{rel}/{name}"
                by_path[rel] = meta
            if name:
                if name in by_name:
                    name_collisions.add(name)
                by_name[name] = meta
    for name in name_collisions:
        by_name.pop(name, None)
    return {"by_path": by_path, "by_name": by_name}


def _lookup_meta(meta_csv: dict, rel_path: str, filename: str):
    if not meta_csv:
        return None
    if meta_csv["by_path"]:
        # csv provides paths -> exact path match only; a filename fallback here
        # would pollute same-named clips in other quadrant folders
        return meta_csv["by_path"].get(rel_path)
    return meta_csv["by_name"].get(filename)


def _infer_generator(rel_path: str, quadrant: str, meta: dict | None) -> str:
    if quadrant == "RVRA":
        return "real"
    if meta and meta.get("method"):
        return meta["method"].lower()
    low = rel_path.lower()
    for g in _PATH_GENERATORS:
        if g in low:
            return g
    return "unknown"


def build_manifest(root: str) -> list[dict]:
    root = Path(root)
    meta_csv = _load_meta_csv(root)
    records = []
    for quad_dir, (quadrant, v_label, a_label) in QUAD_DIRS.items():
        base = root / quad_dir
        if not base.exists():
            continue
        for f in sorted(base.rglob("*.mp4")):
            rel = f.relative_to(root).as_posix()
            m = _ID_RE.search(rel)
            meta = _lookup_meta(meta_csv, rel, f.name)
            records.append({
                "clip_id": rel.replace("/", "__").rsplit(".", 1)[0],
                "rel_path": rel,
                "video_label": v_label,
                "audio_label": a_label,
                "quadrant": quadrant,
                "video_segments": [[0.0, WHOLE_CLIP]] if v_label else [],
                "audio_segments": [[0.0, WHOLE_CLIP]] if a_label else [],
                "generator": _infer_generator(rel, quadrant, meta),
                "dataset": "fakeavceleb",
                "identity": m.group(1) if m else f"anon_{abs(hash(rel)) % 10**8}",
                "meta": {"race": (meta or {}).get("race", ""),
                         "gender": (meta or {}).get("gender", "")},
            })
    if not records:
        raise SystemExit(f"no clips found under {root} — check --root; expected "
                         f"subdirs: {', '.join(QUAD_DIRS)}")
    return records


# ============================================================== splits
def subject_disjoint_splits(records: list[dict], seed: int,
                            fractions=(0.7, 0.1, 0.2)) -> dict[str, list[dict]]:
    """Split by identity so no subject leaks across train/val/test."""
    by_id = defaultdict(list)
    for r in records:
        by_id[r["identity"]].append(r)
    idents = sorted(by_id)
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


def logo_splits(records: list[dict], test_reals: list[dict]) -> dict[str, dict]:
    """Leave-one-generator-out: per generator g, hold out ALL g clips for test
    (paired with test-split real clips so AUC is computable)."""
    gens = sorted({r["generator"] for r in records} - {"real", "unknown"})
    test_real_ids = {r["clip_id"] for r in test_reals}
    out = {}
    for g in gens:
        test = [r for r in records if r["generator"] == g] + test_reals
        train = [r for r in records
                 if r["generator"] != g and r["clip_id"] not in test_real_ids]
        out[g] = {"train": train, "test": test}
    return out


def _write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="FakeAVCeleb root directory")
    ap.add_argument("--out", required=True, help="output manifest .jsonl")
    ap.add_argument("--splits-dir", default=None,
                    help="also write train/val/test + LOGO splits here")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = build_manifest(args.root)
    _write_jsonl(Path(args.out), records)

    counts = defaultdict(int)
    for r in records:
        counts[r["quadrant"]] += 1
    print(f"manifest: {len(records)} clips -> {args.out}")
    print("  " + "  ".join(f"{q}={n}" for q, n in sorted(counts.items())))

    if args.splits_dir:
        d = Path(args.splits_dir)
        splits = subject_disjoint_splits(records, args.seed)
        for name, recs in splits.items():
            _write_jsonl(d / f"{name}.jsonl", recs)
            print(f"  {name}: {len(recs)} clips, "
                  f"{len({r['identity'] for r in recs})} identities")
        test_reals = [r for r in splits["test"] if r["quadrant"] == "RVRA"]
        for g, parts in logo_splits(records, test_reals).items():
            _write_jsonl(d / f"logo_{g}_train.jsonl", parts["train"])
            _write_jsonl(d / f"logo_{g}_test.jsonl", parts["test"])
            print(f"  logo[{g}]: train={len(parts['train'])} test={len(parts['test'])}")


if __name__ == "__main__":
    main()
