"""Robustness sweeps: test-set AUC under controlled degradations (docs/04 §1).

For every perturbation the SAME test clips (the strict-split test set, from the clip
cache written by prepare.py) are degraded, their frozen VideoMAE/WavLM features are
recomputed once and shared by every Phase-A model and seed; Phase-B models receive the
degraded pixels/waveforms directly. Thresholds are the ones each experiment fitted on
its (clean) validation split -- nothing is refitted on degraded data.

Perturbations (modality, severity)
  video  h264_crf{23,30,40}   re-encode the 16-frame window with libx264 (JPEG fallback,
                              recorded if ffmpeg/libx264 is unavailable)
         downscale_x{2,4}     down- then up-sample (resolution loss)
         blur_s{1,2}          Gaussian blur, sigma in pixels
  audio  noise_snr{20,10,0}   additive white Gaussian noise at the given SNR (dB)
         telephone_8k         resample to 8 kHz and back (band limit)
Results: <exp>/metrics/robustness.json + .csv; reports/tables/robustness_results.*,
reports/figures/robustness_*.{png,pdf}.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from src.pipeline.env import log_event

SR = 16000


# ====================================================================== perturbations
def _h264(frames: np.ndarray, crf: int) -> tuple[np.ndarray, str]:
    """(T,H,W,3) uint8 -> re-encoded with libx264 at `crf`; JPEG proxy if unavailable."""
    T, H, W, _ = frames.shape
    try:
        with tempfile.TemporaryDirectory() as td:
            mp4 = os.path.join(td, "x.mp4")
            enc = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{W}x{H}", "-r", "4", "-i", "-", "-c:v", "libx264", "-crf", str(crf),
                 "-pix_fmt", "yuv420p", mp4], input=frames.tobytes(), capture_output=True,
                timeout=60)
            if enc.returncode != 0:
                raise RuntimeError(enc.stderr.decode(errors="ignore")[:120])
            dec = subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-f", "rawvideo",
                                  "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=60)
            out = np.frombuffer(dec.stdout, np.uint8)
            n = out.size // (H * W * 3)
            if n == 0:
                raise RuntimeError("no frames after re-encode")
            out = out[:n * H * W * 3].reshape(n, H, W, 3)
            if n < T:
                out = np.concatenate([out, np.repeat(out[-1:], T - n, 0)])
            return out[:T].copy(), "h264"
    except Exception:  # noqa: BLE001
        import cv2
        q = int(np.interp(crf, [18, 45], [90, 10]))
        res = []
        for f in frames:
            ok, buf = cv2.imencode(".jpg", f[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
            res.append(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])
        return np.stack(res), f"jpeg_q{q}"


def _downscale(frames: np.ndarray, factor: int) -> np.ndarray:
    import cv2
    T, H, W, _ = frames.shape
    return np.stack([cv2.resize(cv2.resize(f, (W // factor, H // factor), interpolation=cv2.INTER_AREA),
                                (W, H), interpolation=cv2.INTER_LINEAR) for f in frames])


def _blur(frames: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    return np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in frames])


def _noise(wave: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = float(np.mean(wave ** 2))
    if p <= 1e-12:
        return wave
    n = rng.normal(0.0, math.sqrt(p / (10 ** (snr_db / 10))), wave.shape).astype(np.float32)
    return np.clip(wave + n, -1.0, 1.0)


def _telephone(wave: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly
    return resample_poly(resample_poly(wave, 1, 2), 2, 1)[: wave.size].astype(np.float32)


def perturbations() -> dict[str, dict]:
    """name -> {modality, severity (for plotting), fn}. 'clean' is the reference."""
    P = {"clean": {"modality": "none", "family": "clean", "severity": 0, "fn": None}}
    for i, crf in enumerate((23, 30, 40)):
        P[f"h264_crf{crf}"] = {"modality": "video", "family": "H.264 compression",
                               "severity": i + 1, "fn": lambda f, c=crf: _h264(f, c)}
    for i, k in enumerate((2, 4)):
        P[f"downscale_x{k}"] = {"modality": "video", "family": "Downscaling",
                                "severity": i + 1, "fn": lambda f, k=k: (_downscale(f, k), "resize")}
    for i, s in enumerate((1.0, 2.0)):
        P[f"blur_s{int(s)}"] = {"modality": "video", "family": "Gaussian blur",
                                "severity": i + 1, "fn": lambda f, s=s: (_blur(f, s), "blur")}
    for i, snr in enumerate((20, 10, 0)):
        P[f"noise_snr{snr}"] = {"modality": "audio", "family": "Additive noise",
                                "severity": i + 1, "fn": lambda w, r, s=snr: _noise(w, s, r)}
    P["telephone_8k"] = {"modality": "audio", "family": "Telephone band (8 kHz)",
                         "severity": 1, "fn": lambda w, r: _telephone(w)}
    return P


# ====================================================================== perturbed inputs
def perturbed_windows(records: list[dict], cache, name: str, spec: dict, n_frames: int = 16,
                      audio_len: int = 64000, eval_offset: int = 4, seed: int = 0):
    """Yield (record, video uint8 (T,H,W,3), audio float32 (N,), has_v, has_a, codec)."""
    import cv2
    from src.data.clipcache import MAGIC  # noqa: F401 - format check lives in decode
    import struct
    rng = np.random.default_rng(seed)
    for r in records:
        if r["clip_id"] not in cache.where:
            continue
        blob = cache.blob(r["clip_id"])
        hlen = struct.unpack("<I", blob[4:8])[0]
        off = 8 + hlen
        hdr = json.loads(blob[8:off].decode())
        grid = cv2.imdecode(np.frombuffer(blob[off:off + hdr["jpeg"]], np.uint8), cv2.IMREAD_COLOR)
        size, cols, n = hdr["size"], hdr["cols"], hdr["n"]
        frames = np.stack([grid[(i // cols) * size:(i // cols + 1) * size,
                                (i % cols) * size:(i % cols + 1) * size] for i in range(n)])[:, :, :, ::-1]
        pcm = np.frombuffer(blob[off + hdr["jpeg"]:], "<i2").astype(np.float32) / 32768.0
        v = np.ascontiguousarray(frames[eval_offset:eval_offset + n_frames])
        a0 = int(round(eval_offset * hdr["sr"] / hdr["fps"]))
        a = pcm[a0:a0 + audio_len]
        a = np.pad(a, (0, max(0, audio_len - a.size)))
        codec = ""
        if spec["modality"] == "video":
            v, codec = spec["fn"](v)
        elif spec["modality"] == "audio" and hdr["ha"]:
            a = spec["fn"](a, rng)
        yield r, v, a.astype(np.float32), bool(hdr["hv"]), bool(hdr["ha"]), codec


def perturbed_feature_table(records, cache, name, spec, extractor, batch_size: int = 32):
    """Frozen features of the degraded test windows (shared by every Phase-A model)."""
    from src.pipeline.features import FeatureTable
    rows_v, rows_a, vav, aav, ids, codecs = [], [], [], [], [], set()
    bv, ba = [], []

    def flush():
        if ba:
            rows_v.append(extractor.video(torch.stack(bv)))
            rows_a.append(extractor.audio(torch.stack(ba)))
            bv.clear()
            ba.clear()

    for r, v, a, hv, ha, codec in perturbed_windows(records, cache, name, spec):
        bv.append(torch.from_numpy(v).permute(0, 3, 1, 2).float() / 255.0)
        ba.append(torch.from_numpy(a))
        vav.append(float(hv))
        aav.append(float(ha))
        ids.append(r["clip_id"])
        if codec:
            codecs.add(codec)
        if len(ba) >= batch_size:
            flush()
    flush()
    t = FeatureTable()
    t.video, t.audio = torch.cat(rows_v), torch.cat(rows_a)
    t.v_avail = torch.tensor(vav, dtype=torch.float16)
    t.a_avail = torch.tensor(aav, dtype=torch.float16)
    t.rows = {cid: {4: i} for i, cid in enumerate(ids)}
    t.n = len(ids)
    return t, sorted(codecs)


class PerturbedPixelDataset(torch.utils.data.Dataset):
    """Phase-B input: degraded frames/waveforms with FeatureDataset-compatible fields."""

    def __init__(self, records, cache, name, spec):
        from src.pipeline.features import QUAD_IDX, _seg_mask
        self.items = []
        for r, v, a, hv, ha, _ in perturbed_windows(records, cache, name, spec):
            q = r.get("quadrant")
            self.items.append({
                "clip_id": r["clip_id"],
                "video": torch.from_numpy(v).permute(0, 3, 1, 2).float() / 255.0,
                "audio": torch.from_numpy(a), "v_avail": torch.tensor(float(hv)),
                "a_avail": torch.tensor(float(ha)),
                "video_label": torch.tensor(r["video_label"]), "audio_label": torch.tensor(r["audio_label"]),
                "clip_label": torch.tensor(r.get("clip_label", 0)),
                "quadrant": torch.tensor(QUAD_IDX.get(q, -1)),
                "video_seg_mask": _seg_mask(r.get("video_segments"), 128),
                "audio_seg_mask": _seg_mask(r.get("audio_segments"), 50),
                "generator": r.get("generator", ""), "dataset": r.get("dataset", ""),
                "race": (r.get("meta") or {}).get("race", ""),
                "gender": (r.get("meta") or {}).get("gender", "")})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# ====================================================================== aggregation
def robustness_rows(results: list[dict]) -> list[dict]:
    """results: [{name, seed, per_perturbation: {pert: {video_auc, audio_auc, clip_auc}}}]."""
    from src.pipeline.evaluation import mean_std_ci
    P = perturbations()
    by = {}
    for r in results:
        for pert, m in r["per_perturbation"].items():
            by.setdefault((r["name"], pert), []).append(m)
    rows = []
    for (name, pert), ms in sorted(by.items()):
        row = {"Model": name, "Perturbation": pert, "Family": P.get(pert, {}).get("family", pert),
               "Seeds": len(ms)}
        for k, lab in (("video_auc", "Video AUC"), ("audio_auc", "Audio AUC"), ("clip_auc", "Clip AUC")):
            s = mean_std_ci([m.get(k, math.nan) for m in ms])
            row[lab] = s["mean"]
            row[f"{lab} std"] = s["std"]
        rows.append(row)
    return rows


def fig_robustness(rows: list[dict], out: Path, models: list[str]) -> list[Path]:
    """One panel per perturbation family; x = severity, y = relevant AUC, line per model."""
    from src.pipeline.reporting import DOUBLE, SERIES, save, style
    plt = style()
    P = perturbations()
    fams = [f for f in dict.fromkeys(p["family"] for k, p in P.items() if k != "clean")]
    colors = [SERIES["video"], SERIES["audio"], SERIES["clip"], SERIES["extra"]]
    fig, axes = plt.subplots(1, len(fams), figsize=(DOUBLE, 2.2), sharey=True)
    for ax, fam in zip(axes, fams):
        perts = [k for k, p in P.items() if p["family"] == fam]
        metric = "Video AUC" if P[perts[0]]["modality"] == "video" else "Audio AUC"
        for k, m in enumerate([m for m in models if any(r["Model"] == m for r in rows)][:4]):
            xs, ys, es = [0], [], []
            clean = [r for r in rows if r["Model"] == m and r["Perturbation"] == "clean"]
            ys.append(clean[0][metric] if clean else math.nan)
            es.append(clean[0][f"{metric} std"] if clean else math.nan)
            for p in perts:
                rr = [r for r in rows if r["Model"] == m and r["Perturbation"] == p]
                xs.append(P[p]["severity"])
                ys.append(rr[0][metric] if rr else math.nan)
                es.append(rr[0][f"{metric} std"] if rr else math.nan)
            es = [0 if (e is None or (isinstance(e, float) and math.isnan(e))) else e for e in es]
            ax.errorbar(xs, ys, yerr=es if any(es) else None, color=colors[k], marker="o",
                        markersize=3, capsize=2, linewidth=1.1, label=m)
        ax.set_title(fam, fontsize=7)
        ax.set_xticks(range(len(perts) + 1), ["clean"] + [p.split("_", 1)[1] for p in perts],
                      fontsize=6)
        ax.axhline(0.5, color="#52514e", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("Test AUC (modality under attack)")
    axes[0].legend(loc="lower left", fontsize=6)
    fig.tight_layout()
    return save(fig, out, "robustness")
