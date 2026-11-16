"""Unified manifests for the primary corpus and the zero-shot (test-only) corpora.

Record schema (superset of docs/03_datasets.md §4)
--------------------------------------------------
clip_id, rel_path, [audio_rel_path], dataset, generator, identity, meta,
video_label, audio_label   0 real / 1 fake / -1 UNKNOWN (never guessed)
clip_label                 1 if anything in the clip is fake
quadrant                   RVRA|RVFA|FVRA|FVFA or null when a label is unknown/absent
modalities                 "av" | "audio" (audio-only corpora go through the model
                           with the null video token, v_avail = 0)

Label decisions (each changes what may be reported, so each is explicit here)
-----------------------------------------------------------------------------
celeb-df-v2   video 1 for Celeb-synthesis; audio 0 (face swaps keep the target
              video's own audio). Official List_of_testing_videos.txt (518 clips).
dfdc-10       only a clip-level REAL/FAKE label exists. Fakes are treated as visually
              manipulated (video 1); audio is -1 because DFDC does not say which of its
              fakes also had the voice swapped. Audio metrics are NOT reported on DFDC.
deepfaketimit fakes only (this mirror ships no VidTIMIT originals): video 1, audio 0
              (the original TIMIT audio). AUC is undefined -> detection rate only.
asvspoof2019  EVAL partition + official CM protocol; audio-only; generator = attack id.
in-the-wild   real/fake from meta.csv (or folder); audio-only.
wavefake      fakes only (generated_audio/<vocoder>/); generator = vocoder;
              detection rate only.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from src.pipeline.env import log_event

from src.pipeline.dataset_config import KAGGLE_SLUGS

# a directory that proves we found the right root for each corpus
_ROOT_MARKERS = {
    "fakeavceleb": "RealVideo-RealAudio",
    "celeb-df-v2": "Celeb-synthesis",
    "dfdc-10": None,
    "deepfaketimit": "higher_quality",
    "asvspoof2019-la": "ASVspoof2019_LA_cm_protocols",
    "in-the-wild": "release_in_the_wild",
    "wavefake": "generated_audio",
}


def find_kaggle_root(name: str, input_dir: str | None = None) -> Optional[Path]:
    """Locate a mounted dataset on Kaggle, Google Drive / Colab, or local storage."""
    import os
    owner, slug = KAGGLE_SLUGS[name].split("/")
    search_dirs = [input_dir, os.environ.get("DATASET_DIR"), "/kaggle/input",
                   "/content/drive/MyDrive", "/content/drive/MyDrive/datasets",
                   "/content/drive/MyDrive/Thesis", "/content/drive/MyDrive/Thesis/datasets",
                   "/content/datasets", "/content"]
    bases = [Path(d) for d in search_dirs if d and Path(d).exists()]
    marker = _ROOT_MARKERS[name]
    for base in bases:
        cands = [base / slug, base / "datasets" / owner / slug, base / owner / slug, base / name]
        cands += sorted(base.glob(f"*{slug}*"))
        cands += sorted(base.glob(f"*{name}*"))
        for c in cands:
            if not c.exists():
                continue
            if marker is None:
                return c
            # breadth-first, one is_dir() probe per directory (a glob 3 levels deep enumerated
            # all 134k WaveFake files on the network mount just to find a folder name)
            frontier = [c]
            for _ in range(4):
                for d in frontier:
                    if (d / marker).is_dir():
                        return d          # the root is the directory CONTAINING the marker
                frontier = [e for d in frontier for e in _subdirs(d)][:200]
    return None


def _subdirs(d: Path) -> list[Path]:
    import os
    try:
        with os.scandir(d) as it:
            return sorted(Path(e.path) for e in it if e.is_dir(follow_symlinks=False))
    except OSError:
        return []


_LISTINGS: dict = {}


def listing(root: Path, workers: int = 32) -> dict[str, int]:
    """{relative path: size} for every file under root, walked ONCE with a thread pool.

    /kaggle/input is a network-backed mount: each directory listing / stat is a round
    trip, so a serial rglob + exists() over the ~290k files of the seven corpora took
    ~1.9 h. The cost is latency, not bandwidth, so directories are listed concurrently.
    The result is cached and shared by the converters and the fingerprint."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root)
    key = str(root)
    if key in _LISTINGS:
        return _LISTINGS[key]
    out: dict[str, int] = {}

    def scan(d: str):
        subdirs, files = [], []
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        subdirs.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        try:
                            files.append((e.path, e.stat().st_size))
                        except OSError:
                            files.append((e.path, -1))
        except OSError:
            pass
        return subdirs, files

    base = len(str(root)) + 1
    with ThreadPoolExecutor(workers) as ex:
        frontier = [str(root)]
        while frontier:
            nxt = []
            for subdirs, files in ex.map(scan, frontier):
                nxt += subdirs
                for path, size in files:
                    out[path[base:].replace(os.sep, "/")] = size
            frontier = nxt
    _LISTINGS[key] = out
    return out


def _files(root: Path, suffixes: tuple, under: str = "") -> list[Path]:
    under = under.rstrip("/") + "/" if under else ""
    return [root / rel for rel in sorted(listing(root))
            if rel.startswith(under) and rel.lower().endswith(suffixes)]


def dataset_fingerprint(root: Path, pattern: str = "*") -> dict:
    """sha256 over the sorted (relative path, size) listing: identifies the exact version
    of a mounted dataset even though a Kaggle mount does not expose its version number."""
    import hashlib
    h = hashlib.sha256()
    ls = listing(root)
    for rel in sorted(ls):
        h.update(f"{rel}\t{ls[rel]}\n".encode())
    return {"sha256": h.hexdigest(), "n_files": len(ls),
            "bytes": sum(max(0, v) for v in ls.values())}


def _rec(**kw) -> dict:
    r = {"video_segments": [], "audio_segments": [], "meta": {}, "modalities": "av"}
    r.update(kw)
    v, a = r.get("video_label", -1), r.get("audio_label", -1)
    r.setdefault("clip_label", int(v == 1 or a == 1))
    if r["modalities"] == "av" and v in (0, 1) and a in (0, 1):
        r.setdefault("quadrant", {(0, 0): "RVRA", (0, 1): "RVFA",
                                  (1, 0): "FVRA", (1, 1): "FVFA"}[(v, a)])
    else:
        r.setdefault("quadrant", None)
    return r


# ====================================================================== FakeAVCeleb
def build_fakeavceleb(root: Path) -> list[dict]:
    """Delegates to the audited upstream converter (meta_data.csv column-shift fix)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.build_manifest import build_manifest
    recs = build_manifest(str(root))
    for r in recs:
        r["clip_label"] = int(r["video_label"] or r["audio_label"])
        r["modalities"] = "av"
    gens = Counter(r["generator"] for r in recs)
    if gens.get("unknown", 0) or gens.get("real") != 500:
        raise RuntimeError(f"FakeAVCeleb generator labels look wrong: {dict(gens)} -- "
                           "meta_data.csv was not parsed (see HANDOFF pitfalls)")
    return recs


# ====================================================================== video corpora
def build_celebdf(root: Path, use_official_test_list: bool = True) -> list[dict]:
    listed = None
    lst = next(iter(_files(root, ("list_of_testing_videos.txt",))), None)
    if use_official_test_list:
        if lst is None:
            raise FileNotFoundError("Celeb-DF List_of_testing_videos.txt not found")
        listed = {ln.split()[1].strip() for ln in lst.read_text().splitlines() if ln.strip()}
    recs = []
    for sub, v in (("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)):
        for p in _files(root, (".mp4",), sub):
            rel = p.relative_to(root).as_posix()
            if listed is not None and rel not in listed:
                continue
            ids = re.findall(r"id\d+", p.stem)
            recs.append(_rec(clip_id=f"celebdf__{sub}__{p.stem}", rel_path=rel,
                             dataset="celeb-df-v2", video_label=v, audio_label=0,
                             generator="celeb-df" if v else "real",
                             identity="|".join(ids) or p.stem, meta={"subset": sub},
                             video_segments=[[0.0, 9999.0]] if v else []))
    if listed is not None and len(recs) != len(listed):
        log_event("manifest_warning", f"Celeb-DF: {len(listed)} listed, {len(recs)} found")
    return recs


def build_dfdc(root: Path, max_real: int = 2500, max_fake: int = 2500, seed: int = 0) -> list[dict]:
    recs = []
    have = listing(root)
    for meta_path in _files(root, ("metadata.json",)):
        meta = json.loads(meta_path.read_text())
        part = meta_path.parent
        for fname, m in sorted(meta.items()):
            p = part / fname
            if p.relative_to(root).as_posix() not in have:
                continue
            fake = int(str(m.get("label", "")).upper() == "FAKE")
            rel = p.relative_to(root).as_posix()
            recs.append(_rec(clip_id=f"dfdc__{part.name}__{p.stem}", rel_path=rel,
                             dataset="dfdc-10", video_label=fake, audio_label=-1,
                             clip_label=fake, generator="dfdc" if fake else "real",
                             identity=p.stem, meta={"part": part.name,
                                                    "original": m.get("original")}))
    rng = random.Random(seed)
    real = [r for r in recs if r["clip_label"] == 0]
    fake = [r for r in recs if r["clip_label"] == 1]
    rng.shuffle(real)
    # stratify fakes over DFDC parts so no single part dominates the sample
    by_part = defaultdict(list)
    for r in fake:
        by_part[r["meta"]["part"]].append(r)
    for v in by_part.values():
        rng.shuffle(v)
    picked, parts = [], sorted(by_part)
    while len(picked) < min(max_fake, len(fake)):
        for pt in parts:
            if by_part[pt] and len(picked) < max_fake:
                picked.append(by_part[pt].pop())
    out = sorted(real[:max_real] + picked, key=lambda r: r["clip_id"])
    log_event("manifest_subsample", f"DFDC: {len(real)} real / {len(fake)} fake available, "
              f"kept {min(max_real, len(real))} / {len(picked)} (seed {seed})")
    return out


def build_deepfaketimit(root: Path) -> list[dict]:
    recs = []
    have = listing(root)
    for p in _files(root, (".avi",)):
        q = p.parent.parent.name                       # higher_quality | lower_quality
        # the audio of "sa1-video-fram1.avi" is "sa1.wav" (original TIMIT sentence)
        wav = p.with_name(p.stem.split("-video-")[0] + ".wav")
        rel = p.relative_to(root).as_posix()
        recs.append(_rec(clip_id=f"dftimit__{q}__{p.parent.name}__{p.stem}", rel_path=rel,
                         audio_rel_path=wav.relative_to(root).as_posix()
                         if wav.relative_to(root).as_posix() in have else None,
                         dataset="deepfaketimit", video_label=1, audio_label=0,
                         generator=f"deepfaketimit-{q.split('_')[0]}", identity=p.parent.name,
                         meta={"quality": q}, video_segments=[[0.0, 9999.0]]))
    return recs


# ====================================================================== audio corpora
def build_asvspoof2019(root: Path, partition: str = "eval") -> list[dict]:
    protos = [p for p in _files(root, (".txt",))
              if p.name.startswith(f"ASVspoof2019.LA.cm.{partition}.tr")]
    if not protos:
        raise FileNotFoundError(f"ASVspoof2019 LA CM {partition} protocol not found under {root}")
    proto = protos[0]
    flacs = {p.stem: p for p in _files(root, (".flac",))
             if f"ASVspoof2019_LA_{partition}/" in p.relative_to(root).as_posix()}
    if not flacs:
        raise FileNotFoundError(f"ASVspoof2019_LA_{partition} audio not found")
    recs, missing = [], 0
    for ln in proto.read_text().splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        spk, utt, attack, label = parts[0], parts[1], parts[3], parts[-1]
        p = flacs.get(utt)
        if p is None:
            missing += 1
            continue
        fake = int(label == "spoof")
        recs.append(_rec(clip_id=f"asvspoof19__{utt}", rel_path=p.relative_to(root).as_posix(),
                         dataset="asvspoof2019-la", modalities="audio", video_label=-1,
                         audio_label=fake, generator=attack if fake else "real",
                         identity=spk, meta={"partition": partition},
                         audio_segments=[[0.0, 9999.0]] if fake else []))
    if missing:
        log_event("manifest_warning", f"ASVspoof: {missing} protocol entries without audio")
    return recs


def build_inthewild(root: Path) -> list[dict]:
    labels = {}
    meta = root / "meta.csv"
    if meta.exists():
        with open(meta, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels[row.get("file", "").strip()] = (row.get("label", "").strip().lower(),
                                                       row.get("speaker", "").strip())
    recs = []
    for p in _files(root, (".wav",)):
        lab, spk = labels.get(p.name, (p.parent.name.lower(), ""))
        if lab not in ("bona-fide", "bonafide", "real", "spoof", "fake"):
            continue
        fake = int(lab in ("spoof", "fake"))
        recs.append(_rec(clip_id=f"itw__{p.parent.name}__{p.stem}",
                         rel_path=p.relative_to(root).as_posix(), dataset="in-the-wild",
                         modalities="audio", video_label=-1, audio_label=fake,
                         generator="in-the-wild-fake" if fake else "real", identity=spk or p.stem,
                         meta={"speaker": spk}, audio_segments=[[0.0, 9999.0]] if fake else []))
    return recs


def build_wavefake(root: Path, max_per_generator: int = 1000, seed: int = 0) -> list[dict]:
    by_gen = defaultdict(list)
    for p in _files(root, (".wav", ".flac")):
        gen = p.parent.name
        by_gen[gen].append(p)
    rng = random.Random(seed)
    recs = []
    for gen in sorted(by_gen):
        files = by_gen[gen]
        rng.shuffle(files)
        for p in sorted(files[:max_per_generator]):
            recs.append(_rec(clip_id=f"wavefake__{gen}__{p.stem}",
                             rel_path=p.relative_to(root).as_posix(), dataset="wavefake",
                             modalities="audio", video_label=-1, audio_label=1,
                             generator=gen, identity=p.stem, audio_segments=[[0.0, 9999.0]]))
    log_event("manifest_subsample", f"WaveFake: {len(by_gen)} generators, "
              f"<= {max_per_generator} per generator (seed {seed}) -> {len(recs)} clips")
    return recs


def build_cross_dataset(name: str, root: Path, opts: dict, seed: int = 0) -> list[dict]:
    if name == "celeb-df-v2":
        return build_celebdf(root, opts.get("use_official_test_list", True))
    if name == "dfdc-10":
        return build_dfdc(root, opts.get("max_real", 2500), opts.get("max_fake", 2500), seed)
    if name == "deepfaketimit":
        return build_deepfaketimit(root)
    if name == "asvspoof2019-la":
        return build_asvspoof2019(root, opts.get("partition", "eval"))
    if name == "in-the-wild":
        return build_inthewild(root)
    if name == "wavefake":
        return build_wavefake(root, opts.get("max_per_generator", 1000), seed)
    raise KeyError(name)


def manifest_summary(recs: list[dict]) -> dict:
    return {"n": len(recs),
            "clip_label": dict(Counter(r["clip_label"] for r in recs)),
            "video_label": dict(Counter(r["video_label"] for r in recs)),
            "audio_label": dict(Counter(r["audio_label"] for r in recs)),
            "generators": dict(Counter(r["generator"] for r in recs).most_common(20)),
            "modalities": dict(Counter(r["modalities"] for r in recs))}


def write_jsonl(path: Path, recs: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]
