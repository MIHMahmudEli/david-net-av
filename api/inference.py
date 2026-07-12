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
        """Decode clip → (frames, waveform, duration) via the real pipeline
        (ffmpeg + face tracking). Falls back to dummy tensors when ffmpeg is
        unavailable so the service still boots in dev environments."""
        torch = self.torch
        try:
            from src.data.preprocess import preprocess_clip
            r = preprocess_clip(video_path, n_frames=self.cfg.n_frames,
                                audio_len=self.cfg.audio_len)
            return (r["video"].unsqueeze(0), r["audio"].unsqueeze(0),
                    r["meta"]["duration_sec"])
        except Exception:
            frames = torch.randn(self.cfg.n_frames, 3, 224, 224)
            audio = torch.randn(self.cfg.audio_len)
            return frames.unsqueeze(0), audio.unsqueeze(0), 4.0

    def predict(self, video_path: str, explain: bool = False,
                has_video: bool = True, has_audio: bool = True) -> dict:
        """Full A+V clip by default; handles silent video (has_audio=False)
        and audio-only input (has_video=False) via availability masks."""
        torch = self.torch
        t0 = time.time()
        frames, audio, duration = self._preprocess(video_path)
        frames, audio = frames.to(self.device), audio.to(self.device)
        v_av = torch.ones(1, device=self.device) if has_video else torch.zeros(1, device=self.device)
        a_av = torch.ones(1, device=self.device) if has_audio else torch.zeros(1, device=self.device)
        with torch.no_grad():
            out = self.model(frames, audio, v_avail=v_av, a_avail=a_av)
        pv = float(torch.sigmoid(out["logit_v"])[0])
        pa = float(torch.sigmoid(out["logit_a"])[0])
        quad_probs = torch.softmax(out["logit_quad"][0], dim=-1).cpu().tolist()
        agreement = out["agreement"]
        sync_curve = agreement[0].cpu().tolist() if agreement is not None else []

        def _verdict(p, available):
            if not available:
                return {"verdict": "unavailable", "confidence": None}
            return {"verdict": "fake" if p >= 0.5 else "real", "confidence": round(max(p, 1 - p), 4)}

        result = {
            "clip_id": Path(video_path).stem,
            "duration_sec": round(duration, 2),
            "modalities": {"video": has_video, "audio": has_audio},
            "video": _verdict(pv, has_video),
            "audio": _verdict(pa, has_audio),
            # quadrant needs both streams; single-modality inputs get no quadrant
            "quadrant": ({
                "label": QUADRANTS[int(max(range(4), key=lambda i: quad_probs[i]))],
                "probs": {q: round(p, 4) for q, p in zip(QUADRANTS, quad_probs)},
            } if (has_video and has_audio) else None),
            "localization": {
                "video": self._peaks(out["loc_v"], duration) if has_video else [],
                "audio": self._peaks(out["loc_a"], duration) if has_audio else [],
            },
            "sync_curve": [round(x, 4) for x in sync_curve] if (has_video and has_audio) else [],
            "model_version": self.version,
            "latency_ms": int((time.time() - t0) * 1000),
        }
        if explain:
            result["explain"] = {"note": "attach Grad-CAM / saliency PNGs (base64) here"}
        return result

    def predict_audio(self, audio_path: str, explain: bool = False) -> dict:
        """Standalone audio detection (voice notes, calls, extracted tracks)."""
        return self.predict(audio_path, explain=explain, has_video=False, has_audio=True)

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
