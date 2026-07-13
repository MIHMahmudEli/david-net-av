"""Tests for the FakeAVCeleb manifest converter + split generator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_manifest import (  # noqa: E402
    build_manifest, subject_disjoint_splits, logo_splits, WHOLE_CLIP,
)


def _make_tree(root: Path):
    """Synthetic FakeAVCeleb v1.2 layout: 4 quadrants x identities x clips."""
    layout = {
        "RealVideo-RealAudio": [("id00001", 3), ("id00002", 2), ("id00003", 2)],
        "RealVideo-FakeAudio/rtvc": [("id00001", 2), ("id00002", 1)],
        "FakeVideo-RealAudio/faceswap": [("id00001", 2), ("id00003", 2)],
        "FakeVideo-FakeAudio/wav2lip": [("id00002", 2), ("id00003", 1)],
    }
    for sub, idents in layout.items():
        for ident, n in idents:
            d = root / sub / "African" / "men" / ident
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{i:05d}.mp4").touch()
    # meta csv (realistic: has a path column) enriches exactly one clip;
    # the bare filename 00000.mp4 exists in MANY folders and must not leak
    (root / "meta_data.csv").write_text(
        "path,filename,method,race,gender\n"
        "FakeVideo-RealAudio/faceswap/African/men/id00001,00000.mp4,"
        "faceswap,African,men\n", encoding="utf-8")
    return sum(n for idents in layout.values() for _, n in idents)


def test_build_manifest(tmp_path):
    n_total = _make_tree(tmp_path)
    records = build_manifest(str(tmp_path))
    assert len(records) == n_total

    by_quad = {}
    for r in records:
        by_quad.setdefault(r["quadrant"], []).append(r)
    assert set(by_quad) == {"RVRA", "RVFA", "FVRA", "FVFA"}

    # labels consistent with quadrant
    for r in records:
        assert r["video_label"] == int(r["quadrant"][0] == "F")
        assert r["audio_label"] == int(r["quadrant"][2] == "F")

    # whole-clip fakes carry the sentinel segment; reals carry none
    for r in by_quad["FVFA"]:
        assert r["video_segments"] == [[0.0, WHOLE_CLIP]]
        assert r["audio_segments"] == [[0.0, WHOLE_CLIP]]
    for r in by_quad["RVRA"]:
        assert r["video_segments"] == [] and r["audio_segments"] == []

    # generator: from path fragments (rtvc/faceswap/wav2lip), real for RVRA
    assert all(r["generator"] == "real" for r in by_quad["RVRA"])
    assert all(r["generator"] == "rtvc" for r in by_quad["RVFA"])
    assert all(r["generator"] == "wav2lip" for r in by_quad["FVFA"])

    # csv metadata lands on exactly the referenced clip — the shared filename
    # 00000.mp4 must NOT pollute clips in other folders
    enriched = [r for r in records if r["meta"]["gender"] == "men"]
    assert len(enriched) == 1
    assert enriched[0]["rel_path"] == \
        "FakeVideo-RealAudio/faceswap/African/men/id00001/00000.mp4"

    # identity extracted for subject-disjoint splitting
    assert all(r["identity"].startswith("id") for r in records)
    # clip_ids unique
    assert len({r["clip_id"] for r in records}) == n_total


def test_sentinel_segment_yields_full_mask():
    """The WHOLE_CLIP sentinel must produce an all-ones localization mask."""
    from src.data.datasets import segments_to_mask
    mask = segments_to_mask([[0.0, WHOLE_CLIP]], length=16, duration=4.0)
    assert mask.sum() == 16
    assert segments_to_mask([], length=16, duration=4.0).sum() == 0


def test_subject_disjoint_splits(tmp_path):
    _make_tree(tmp_path)
    records = build_manifest(str(tmp_path))
    splits = subject_disjoint_splits(records, seed=42)

    # every clip lands in exactly one split
    assert sum(len(v) for v in splits.values()) == len(records)
    # identities never straddle splits
    ids = {k: {r["identity"] for r in v} for k, v in splits.items()}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])
    # deterministic under the same seed
    again = subject_disjoint_splits(records, seed=42)
    assert [r["clip_id"] for r in again["test"]] == [r["clip_id"] for r in splits["test"]]


def test_logo_splits(tmp_path):
    _make_tree(tmp_path)
    records = build_manifest(str(tmp_path))
    splits = subject_disjoint_splits(records, seed=42)
    test_reals = [r for r in splits["test"] if r["quadrant"] == "RVRA"]
    logo = logo_splits(records, test_reals)

    assert set(logo) == {"rtvc", "faceswap", "wav2lip"}
    for g, parts in logo.items():
        # held-out generator never leaks into train
        assert all(r["generator"] != g for r in parts["train"])
        # test contains the generator's fakes AND real clips (AUC computable)
        gens_in_test = {r["generator"] for r in parts["test"]}
        assert g in gens_in_test
        labels = {(r["video_label"], r["audio_label"]) for r in parts["test"]}
        assert (0, 0) in labels or not test_reals
        # test reals removed from train (no leakage)
        test_ids = {r["clip_id"] for r in parts["test"]}
        assert not test_ids & {r["clip_id"] for r in parts["train"]}
