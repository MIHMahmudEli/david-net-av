"""FakeAVCeleb splits: strict identity-disjoint (primary), legacy (supplementary), LOGO.

Why "strict"
------------
A FakeAVCeleb fake is filed under its SOURCE identity's folder, but its filename names
the TARGET identities whose face/voice were used, e.g.
    FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.mp4
A split that separates folder identities only (the "legacy" protocol, and most of the
literature) puts id00476's face in test clips while id00476's own real clip and other
fakes are in training. On the published legacy split this affected 3,104 of 4,375
test clips (71%). The strict protocol assigns IDENTITIES to splits (stratified by
race x gender, the demographic strata of the corpus) and keeps a clip only if EVERY
identity it contains is in the same split; clips that straddle splits are dropped
and counted. `leakage_audit` measures the overlap for any split so both protocols can
be reported side by side.

LOGO
----
Leave-one-generator-family-out over the VIDEO manipulation families on top of the
strict split (generator-disjoint AND identity-disjoint). There is no audio-generator
LOGO: every fake audio track in FakeAVCeleb (RVFA and FVFA) comes from one synthesizer
(SV2TTS/RTVC), so holding it out would still train on it through FVFA clips. Audio
generalization is measured cross-corpus instead (ASVspoof 2019 LA, In-the-Wild,
WaveFake).
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

_ID_RE = re.compile(r"id\d+")

VIDEO_LOGO_FAMILIES = {
    "wav2lip": lambda g: "wav2lip" in g,        # incl. faceswap-wav2lip / fsgan-wav2lip
    "fsgan": lambda g: "fsgan" in g,
    "faceswap": lambda g: "faceswap" in g,
}


def clip_identities(rec: dict) -> set[str]:
    ids = set(_ID_RE.findall(rec.get("rel_path", "")))
    if not ids and rec.get("identity"):
        ids = {rec["identity"]}
    return ids


def _stratum(rec: dict) -> str:
    m = rec.get("meta") or {}
    return f"{m.get('race', '?')}|{m.get('gender', '?')}"


def _allocate(n: int, fractions) -> list[int]:
    """Largest-remainder allocation of n items to len(fractions) bins."""
    raw = [f * n for f in fractions]
    base = [int(x) for x in raw]
    rem = n - sum(base)
    order = sorted(range(len(raw)), key=lambda i: -(raw[i] - base[i]))
    for i in order[:rem]:
        base[i] += 1
    return base


def strict_identity_splits(records: list[dict], seed: int = 42,
                           fractions=(0.7, 0.1, 0.2)) -> dict:
    """Identity-level assignment stratified by race x gender; straddling clips dropped."""
    names = ("train", "val", "test")
    # an identity's stratum is taken from its own (folder) clips
    stratum_of: dict[str, str] = {}
    for r in records:
        if r.get("identity"):
            stratum_of.setdefault(r["identity"], _stratum(r))
    all_ids = set().union(*(clip_identities(r) for r in records))
    for i in all_ids:
        stratum_of.setdefault(i, "?|?")
    by_stratum = defaultdict(list)
    for i, s in stratum_of.items():
        by_stratum[s].append(i)

    rng = random.Random(seed)
    assign: dict[str, str] = {}
    for s in sorted(by_stratum):
        ids = sorted(by_stratum[s])
        rng.shuffle(ids)
        counts = _allocate(len(ids), fractions)
        k = 0
        for name, c in zip(names, counts):
            for i in ids[k:k + c]:
                assign[i] = name
            k += c

    splits = {n: [] for n in names}
    dropped = []
    for r in records:
        where = {assign[i] for i in clip_identities(r)}
        if len(where) == 1:
            splits[where.pop()].append(r)
        else:
            dropped.append(r)
    return {"splits": splits, "dropped": dropped, "identity_assignment": assign,
            "protocol": "strict_identity", "seed": seed, "fractions": list(fractions)}


def legacy_source_splits(records: list[dict], seed: int = 42,
                         fractions=(0.7, 0.1, 0.2)) -> dict:
    """The published protocol: disjoint by FOLDER (source) identity only. Kept for a
    supplementary, literature-comparable table -- it leaks target identities."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.build_manifest import subject_disjoint_splits
    return {"splits": subject_disjoint_splits(records, seed, fractions), "dropped": [],
            "protocol": "legacy_source", "seed": seed, "fractions": list(fractions)}


def leakage_audit(splits: dict[str, list[dict]]) -> dict:
    """Identity overlap between splits, counting EVERY identity in a clip."""
    ids = {n: set().union(*(clip_identities(r) for r in recs)) if recs else set()
           for n, recs in splits.items()}
    out = {}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        shared = ids[a] & ids[b]
        leaking = [r for r in splits[b] if clip_identities(r) & ids[a]]
        out[f"{a}|{b}"] = {"shared_identities": len(shared), "leaking_clips": len(leaking),
                           "clips": len(splits[b]),
                           "leaking_fraction": round(len(leaking) / max(1, len(splits[b])), 4)}
    return out


def assert_no_leakage(splits: dict[str, list[dict]]):
    audit = leakage_audit(splits)
    bad = {k: v for k, v in audit.items() if v["shared_identities"]}
    if bad:
        raise AssertionError(f"identity leakage between splits: {bad}")


def video_logo_splits(splits: dict[str, list[dict]]) -> dict[str, dict]:
    """Per video family f: train/val without f, test = f's test clips + test reals."""
    out = {}
    for fam, is_fam in VIDEO_LOGO_FAMILIES.items():
        test_f = [r for r in splits["test"] if is_fam(r["generator"])]
        if not test_f:
            continue
        out[fam] = {
            "train": [r for r in splits["train"] if not is_fam(r["generator"])],
            "val": [r for r in splits["val"] if not is_fam(r["generator"])],
            "test": test_f + [r for r in splits["test"] if r["quadrant"] == "RVRA"],
        }
    return out


def split_summary(splits: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for name, recs in splits.items():
        q = Counter(r["quadrant"] for r in recs)
        g = Counter(r["generator"] for r in recs)
        rows.append({"split": name, "clips": len(recs),
                     "identities": len(set().union(*(clip_identities(r) for r in recs))) if recs else 0,
                     **{f"n_{k}": q.get(k, 0) for k in ("RVRA", "RVFA", "FVRA", "FVFA")},
                     "generators": dict(sorted(g.items()))})
    return rows


def stratified_subsample(records: list[dict], n: int, seed: int = 0,
                         key=lambda r: r["quadrant"]) -> list[dict]:
    """Up to n records, spread evenly over key groups (smoke tests / capped test sets)."""
    groups = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    out, i = [], 0
    keys = sorted(groups)
    while len(out) < min(n, len(records)):
        k = keys[i % len(keys)]
        if groups[k]:
            out.append(groups[k].pop())
        i += 1
        if all(not groups[k] for k in keys):
            break
    return sorted(out, key=lambda r: r["clip_id"])
