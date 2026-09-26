"""Unit tests for the Q1 pipeline's scientifically load-bearing pieces."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline import config as C
from src.pipeline import evaluation as E
from src.pipeline import splits as S


def _fav_like(n_ids=60, seed=0):
    """FakeAVCeleb-like records: one real per identity + fakes naming target ids."""
    rng = np.random.default_rng(seed)
    recs = []
    races = ["A", "B"]
    for i in range(n_ids):
        ident = f"id{i:05d}"
        meta = {"race": races[i % 2], "gender": "men" if i % 3 else "women"}
        base = f"RealVideo-RealAudio/{meta['race']}/{meta['gender']}/{ident}"
        recs.append({"clip_id": f"r{i}", "rel_path": f"{base}/00001.mp4", "identity": ident,
                     "quadrant": "RVRA", "generator": "real", "meta": meta})
        for k in range(4):
            tgt = f"id{int(rng.integers(0, n_ids)):05d}"
            recs.append({"clip_id": f"f{i}_{k}", "identity": ident, "quadrant": "FVFA",
                         "rel_path": f"FakeVideo-FakeAudio/{meta['race']}/{meta['gender']}/{ident}/00001_{k}_{tgt}_wavtolip.mp4",
                         "generator": ["wav2lip", "fsgan", "faceswap-wav2lip"][k % 3], "meta": meta})
    return recs


def test_strict_split_has_no_identity_leakage_and_legacy_does():
    recs = _fav_like()
    strict = S.strict_identity_splits(recs, 42, (0.65, 0.15, 0.2))
    S.assert_no_leakage(strict["splits"])
    kept = sum(len(v) for v in strict["splits"].values())
    assert kept + len(strict["dropped"]) == len(recs)
    # every real clip is kept (a single identity never straddles)
    assert sum(r["quadrant"] == "RVRA" for v in strict["splits"].values() for r in v) == 60
    legacy = S.legacy_source_splits(recs, 42, (0.65, 0.15, 0.2))
    assert S.leakage_audit(legacy["splits"])["train|test"]["shared_identities"] > 0


def test_strict_split_is_deterministic_and_stratified():
    recs = _fav_like()
    a = S.strict_identity_splits(recs, 7)["identity_assignment"]
    b = S.strict_identity_splits(recs, 7)["identity_assignment"]
    assert a == b
    test_reals = [r for r in S.strict_identity_splits(recs, 7)["splits"]["test"] if r["quadrant"] == "RVRA"]
    races = {r["meta"]["race"] for r in test_reals}
    assert races == {"A", "B"}


def test_logo_excludes_family_from_train_and_val():
    recs = _fav_like()
    sp = S.strict_identity_splits(recs, 42)["splits"]
    for fam, parts in S.video_logo_splits(sp).items():
        fn = S.VIDEO_LOGO_FAMILIES[fam]
        assert not any(fn(r["generator"]) for r in parts["train"] + parts["val"])
        assert any(fn(r["generator"]) for r in parts["test"])


def test_config_hash_ignores_non_semantic_keys():
    a = C.build_config({})
    b = C.build_config({"checkpoint": {"every_steps": 7}, "session": {"worker_name": "x"},
                        "hardware": {"precision": "fp32"}, "plan": {"only": ["davidnet"]}})
    c = C.build_config({"train": {"stage1": {"lr": 5e-4}}})
    assert C.config_hash(a) == C.config_hash(b)
    assert C.config_hash(a) != C.config_hash(c)


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        C.build_config({"data": {"split_fractions": [0.5, 0.5, 0.5]}})


def _frame(y, p):
    return pd.DataFrame({"video_label": y, "audio_label": y, "clip_label": y, "v_avail": 1,
                         "a_avail": 1, "p_video": p, "p_audio": p, "p_clip": p,
                         "quadrant_label": "", **{f"p_quad_{q}": np.nan for q in ("RVRA", "RVFA", "FVRA", "FVFA")}})


def test_thresholds_come_from_validation_only():
    rng = np.random.default_rng(0)
    yv = rng.integers(0, 2, 400)
    pv = np.clip(yv * 0.3 + rng.normal(0.35, 0.15, 400), 0, 1)
    thr = E.fit_thresholds(_frame(yv, pv), "val_eer")
    yt = rng.integers(0, 2, 400)
    pt = np.clip(yt * 0.3 + rng.normal(0.35, 0.15, 400), 0, 1)
    m = E.evaluate_frame(_frame(yt, pt), thr, {"bootstrap": 100, "ece_bins": 10})
    assert m["video"]["threshold"] == pytest.approx(thr["video"])
    assert 0.5 < m["video"]["auc"] < 1.0
    assert m["video"]["auc_ci_low"] <= m["video"]["auc"] <= m["video"]["auc_ci_high"]


def test_single_class_corpus_reports_detection_rate_not_auc():
    y = np.ones(50, dtype=int)
    p = np.linspace(0.2, 0.9, 50)
    m = E.binary_metrics(y, p, 0.5, n_boot=0)
    assert not m["auc_defined"] and np.isnan(m["auc"])
    assert m["detection_rate"] == pytest.approx((p >= 0.5).mean())


def test_delong_identical_and_different_scorers():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    good = y * 1.0 + rng.normal(0, 0.5, 300)
    bad = rng.normal(0, 1, 300)
    same = E.delong_test(y, good, good)
    assert same["diff"] == 0
    diff = E.delong_test(y, good, bad)
    assert diff["p_value"] < 1e-6
    from sklearn.metrics import roc_auc_score
    assert diff["auc1"] == pytest.approx(roc_auc_score(y, good))


def test_holm_and_seed_aggregation():
    adj = E.holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.08) and adj["c"] == 0.5
    ms = E.mean_std_ci([0.9, 0.92, 0.94])
    assert ms["mean"] == pytest.approx(0.92) and ms["ci_low"] < 0.92 < ms["ci_high"]


def test_epoch_sampler_is_deterministic_and_skips_exactly():
    from src.pipeline.features import EpochSampler
    keys = ["a"] * 90 + ["b"] * 10
    s = EpochSampler(keys, "sqrt_balanced", seed=3)
    s.set_epoch(2)
    full = list(s)
    s.set_epoch(2, skip=37)
    assert list(s) == full[37:]
    frac_b = sum(keys[i] == "b" for i in full) / len(full)
    assert 0.15 < frac_b < 0.4          # up-weighted from 0.10, not fully balanced
