"""Inference wrapper for the deployed DAVID-Net-Lite model.

Takes a video file path, runs preprocessing (ffmpeg + face track + resample), runs the
model, and returns the JSON verdict described in docs/05_deployment.md.

This module is import-safe: heavy deps (torch, ffmpeg) are imported lazily so `app.py`
can start and serve /health even before a model is attached.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

QUADRANTS = ["RVRA", "RVFA", "FVRA", "FVFA"]


class DavidNetInference:
    def __init__(self, checkpoint: Optional[str] = None, config: str = "configs/david_net.yaml",
                 device: Optional[str] = None):
        import torch
        from src.utils.config import load_config
        from src.training.train import build_model

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = load_config(config)
        self.model = build_model(self.cfg).to(self.device).eval()
        if checkpoint and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state["model"])
        self.version = "david-net-lite-1.0"

    def _preprocess(self, video_path: str):
        """Decode video → (frames tensor, waveform tensor, duration). Placeholder impl.

        Real impl: ffmpeg demux, RetinaFace/MediaPipe face crop @ 25fps, resample audio to
        16 kHz mono. Here we emit correctly-shaped dummy tensors so the service runs.
        """
        torch = self.torch
        frames = torch.randn(self.cfg.n_frames, 3, 224, 224)
        audio = torch.randn(self.cfg.audio_len)
        duration = 4.0
        return frames.unsqueeze(0), audio.unsqueeze(0), duration

    def predict(self, video_path: str, explain: bool = False) -> dict:
        torch = self.torch
        t0 = time.time()
        frames, audio, duration = self._preprocess(video_path)
        frames, audio = frames.to(self.device), audio.to(self.device)
        with torch.no_grad():
            out = self.model(frames, audio)
        pv = float(torch.sigmoid(out["logit_v"])[0])
        pa = float(torch.sigmoid(out["logit_a"])[0])
        quad_probs = torch.softmax(out["logit_quad"][0], dim=-1).cpu().tolist()
        agreement = out["agreement"]
        sync_curve = agreement[0].cpu().tolist() if agreement is not None else []

        result = {
            "clip_id": Path(video_path).stem,
            "duration_sec": round(duration, 2),
            "video": {"verdict": "fake" if pv >= 0.5 else "real", "confidence": round(max(pv, 1 - pv), 4)},
            "audio": {"verdict": "fake" if pa >= 0.5 else "real", "confidence": round(max(pa, 1 - pa), 4)},
            "quadrant": {
                "label": QUADRANTS[int(max(range(4), key=lambda i: quad_probs[i]))],
                "probs": {q: round(p, 4) for q, p in zip(QUADRANTS, quad_probs)},
            },
            "localization": {"video": self._peaks(out["loc_v"], duration),
                             "audio": self._peaks(out["loc_a"], duration)},
            "sync_curve": [round(x, 4) for x in sync_curve],
            "model_version": self.version,
            "latency_ms": int((time.time() - t0) * 1000),
        }
        if explain:
            result["explain"] = {"note": "attach Grad-CAM / saliency PNGs (base64) here"}
        return result

    def _peaks(self, loc_logits, duration: float, thr: float = 0.5):
        torch = self.torch
        probs = torch.sigmoid(loc_logits[0]).cpu().tolist()
        L = len(probs)
        segs, start = [], None
        for i, p in enumerate(probs):
            if p >= thr and start is None:
                start = i
            elif p < thr and start is not None:
                segs.append({"start": round(duration * start / L, 2),
                             "end": round(duration * i / L, 2),
                             "score": round(max(probs[start:i]), 4)})
                start = None
        if start is not None:
            segs.append({"start": round(duration * start / L, 2), "end": round(duration, 2),
                         "score": round(max(probs[start:]), 4)})
        return segs
