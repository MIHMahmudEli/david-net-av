"""Real-data self-test on Kaggle (no Hugging Face writes, no secret needed).

Exercises every code path that cannot run on a dev box:
  1. manifest converters on the real mounts (+ label sanity per corpus)
  2. strict split on the real FakeAVCeleb manifest (+ leakage audit)
  3. ffmpeg decode + YuNet face crop + feature extraction (VideoMAE/WavLM, fp16 on GPU)
     for a few clips of EVERY corpus, incl. QACP pseudo variants and the clip cache
  4. one Phase-A training step (DAVID-Net on the extracted features)
  5. one Phase-B forward/backward step on real pixels/waveforms (partial unfreeze)
  6. throughput numbers used to size the real run

    python -m src.pipeline.selftest --out /kaggle/working/selftest
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch


def main(out: str, n_per_corpus: int = 6):
    from src.pipeline import manifests as M
    from src.pipeline import splits as S
    from src.pipeline.config import build_config
    from src.pipeline.encoders import resolve_revision
    from src.pipeline.env import autoconfig_hardware, environment_report, quiet_third_party, setup_logging
    from src.pipeline.prepare import FeatureExtractor, ensure_yunet, extract_corpus
    quiet_third_party()
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    setup_logging(outp / "logs")
    cfg = build_config({"mode": "smoke"})
    hw = autoconfig_hardware(cfg)
    rep = {"env": environment_report("."), "hw": hw, "steps": {}}

    def step(name, fn):
        t0 = time.time()
        try:
            res = fn()
            rep["steps"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), "result": res}
        except Exception as e:  # noqa: BLE001
            rep["steps"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                  "trace": traceback.format_exc()[-2500:]}
        print(f"[selftest] {name}: {'OK' if rep['steps'][name]['ok'] else 'FAIL'} "
              f"({rep['steps'][name].get('seconds', '-')}s)", flush=True)
        if not rep["steps"][name]["ok"]:
            print(rep["steps"][name]["trace"], flush=True)
        return rep["steps"][name].get("result")

    records = {}

    def manifests():
        res = {}
        for name in M.KAGGLE_SLUGS:
            root = M.find_kaggle_root(name)
            if root is None:
                res[name] = "NOT MOUNTED"
                continue
            recs = (M.build_fakeavceleb(root) if name == "fakeavceleb" else
                    M.build_cross_dataset(name, root, cfg["data"]["cross_datasets"][name], 0))
            records[name] = (root, recs)
            res[name] = {"root": str(root), **M.manifest_summary(recs)}
        return res
    step("1_manifests", manifests)

    def split():
        fav = records["fakeavceleb"][1]
        st = S.strict_identity_splits(fav, 42, tuple(cfg["data"]["split_fractions"]))
        S.assert_no_leakage(st["splits"])
        lg = S.legacy_source_splits(fav, 42, tuple(cfg["data"]["split_fractions"]))
        return {"strict": S.split_summary(st["splits"]), "dropped": len(st["dropped"]),
                "legacy_audit": S.leakage_audit(lg["splits"])}
    step("2_splits", split)

    revs = {}
    feats = {}

    def extract():
        revs.update(video=resolve_revision(cfg["features"]["video_model"]),
                    audio=resolve_revision(cfg["features"]["audio_model"]))
        ex = FeatureExtractor(cfg, revs, hw["device"])
        yunet = ensure_yunet(outp / "models")
        res = {}
        for name, (root, recs) in records.items():
            sample = S.stratified_subsample(recs, n_per_corpus, key=lambda r: r["clip_label"])
            t0 = time.time()
            st = extract_corpus(corpus=name, records=sample, root=str(root), cfg=cfg, extractor=ex,
                                data_store=None, fsid="selftest", scratch=outp / "scratch",
                                multiview_ids={r["clip_id"] for r in sample} if name == "fakeavceleb" else set(),
                                qacp_ids={r["clip_id"] for r in sample if r.get("quadrant") == "RVRA"}
                                if name == "fakeavceleb" else set(),
                                num_workers=hw["num_workers"], yunet=yunet,
                                write_clipcache=name in ("fakeavceleb", "celeb-df-v2", "dfdc-10", "deepfaketimit"),
                                batch_size=16)
            st["seconds_per_clip"] = round((time.time() - t0) / max(1, len(sample)), 2)
            res[name] = st
            feats[name] = sample
        res["revisions"] = revs
        return res
    step("3_decode_crop_extract", extract)

    def phase_a_step():
        from src.pipeline.features import FeatureDataset, FeatureStore, collate
        from src.pipeline.models import build_davidnet
        from src.pipeline.trainer import stage1_objective
        fs = FeatureStore(None, "selftest", outp / "scratch")
        t = fs.table("fakeavceleb")
        ds = FeatureDataset(feats["fakeavceleb"], t, True, 42)
        b = collate([ds[i] for i in range(len(ds))])
        b = {k: v.to(hw["device"]) if torch.is_tensor(v) else v for k, v in b.items()}
        m = build_davidnet(cfg, "A").to(hw["device"])
        with torch.autocast("cuda", dtype=torch.float16, enabled=hw["device"] == "cuda"):
            loss, parts = stage1_objective(m, b, cfg, True)
        loss.backward()
        return {"loss": float(loss), "parts": parts, "video_shape": list(b["video"].shape),
                "audio_shape": list(b["audio"].shape)}
    step("4_phase_a_step", phase_a_step)

    def phase_b_step():
        from src.pipeline.features import collate
        from src.pipeline.models import build_davidnet
        from src.pipeline.phase_b import ClipCacheIndex, PixelDataset
        from src.pipeline.trainer import stage1_objective
        cache = ClipCacheIndex(None, "fakeavceleb", outp / "scratch")
        ds = PixelDataset(feats["fakeavceleb"], cache, True, 42)
        b = collate([ds[i] for i in range(min(2, len(ds)))])
        b = {k: v.to(hw["device"]) if torch.is_tensor(v) else v for k, v in b.items()}
        m = build_davidnet(cfg, "B", revs, 2, 2, True).to(hw["device"])
        torch.cuda.reset_peak_memory_stats() if hw["device"] == "cuda" else None
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=hw["device"] == "cuda"):
            loss, parts = stage1_objective(m, b, cfg, True, stage="phase_b")
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9 if hw["device"] == "cuda" else 0
        return {"loss": float(loss), "seconds_fwd_bwd_2_clips": round(time.time() - t0, 2),
                "peak_vram_gb_2_clips": round(peak, 2)}
    step("5_phase_b_step", phase_b_step)

    rep["passed"] = all(s["ok"] for s in rep["steps"].values())
    (outp / "selftest_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("SELFTEST", "PASSED" if rep["passed"] else "FAILED", flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/selftest")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    main(a.out, a.n)
