# Deployment: Hugging Face API + Next.js UI

Goal: end users upload a video; the system returns **four things** — video verdict, audio verdict, quadrant, and per-modality temporal timelines with confidence + explainability overlays.

## 1. Architecture

```
┌────────────┐    multipart upload      ┌──────────────────────────┐
│  Next.js   │ ───────────────────────► │  FastAPI on HF Spaces     │
│  (Vercel)  │ ◄─────────────────────── │  (DAVID-Net-Lite, GPU/CPU)│
└────────────┘   JSON verdict + PNGs     └──────────────────────────┘
        │                                          │
        │ renders timelines/heatmaps               │ ffmpeg → preprocess → model → explain
        ▼                                          ▼
   User sees per-modality result            weights pulled from HF Hub
```

- **Model hosting:** two options — (a) FastAPI in a HF **Space** (Docker SDK) exposing `/predict`; or (b) a HF **Inference Endpoint** with a custom handler. Default: Docker Space (free tier friendly, full control over ffmpeg).
- **Weights:** stored as a HF Model repo (`your-org/david-net-lite`), pulled at container start.
- **Serve the Lite model** for latency/cost; the Base model can run behind a paid GPU endpoint if needed.

## 2. API contract

`POST /predict` (multipart: `file=<video>`), optional `?explain=true`.

Response:
```json
{
  "clip_id": "abc123",
  "duration_sec": 8.4,
  "video": {"verdict": "fake", "confidence": 0.94},
  "audio": {"verdict": "real", "confidence": 0.88},
  "quadrant": {"label": "FVRA", "probs": {"RVRA":0.02,"RVFA":0.03,"FVRA":0.88,"FVFA":0.07}},
  "localization": {
    "video": [{"start": 1.2, "end": 3.6, "score": 0.91}],
    "audio": []
  },
  "sync_curve": [0.9, 0.88, ...],
  "explain": {"video_heatmap_png": "base64...", "audio_saliency_png": "base64..."},
  "model_version": "david-net-lite-1.0",
  "latency_ms": 1840
}
```

`GET /health`, `GET /version` for ops. Rate-limit + max file size + max duration (e.g., 60 s) enforced server-side.

## 3. FastAPI service

See `api/app.py` and `api/inference.py`. Pipeline: ffmpeg demux → face-track + resample → tensors → DAVID-Net-Lite → post-process → (optional) explainability render → JSON. Model loaded once at startup; async request handling; input validation and safe temp-file cleanup.

## 4. Next.js UI

See `ui/README.md`. Pages:
- **Upload** (drag-drop, progress, client-side duration/size guard).
- **Result**: two verdict cards (video / audio) with confidence gauges, a quadrant badge, a **dual timeline** (video track + audio track) highlighting manipulated intervals, the sync curve, and expandable heatmap overlays.
- **About / limitations**: honest disclaimer that no detector is perfect; show model version + benchmark numbers.

Stack: Next.js (App Router) + TypeScript + Tailwind + a small charting lib for the timeline. Calls the HF Space URL from a server action / route handler (keeps CORS + any token server-side).

## 5. Responsible-use guardrails

- Prominent "**probabilistic, not proof**" disclaimer; never present a verdict as legal certainty.
- No storage of user uploads beyond the request (or opt-in only, with clear notice).
- Model card linked from the UI; log model version with every verdict.
- Abuse throttling; do not expose any generation capability.

## 6. Rollout steps

1. Export Lite checkpoint → HF Model repo.
2. Build Docker Space with ffmpeg + FastAPI; smoke-test `/predict`.
3. Deploy Next.js to Vercel; wire `NEXT_PUBLIC_API_URL` (or server-side proxy).
4. Add health checks, basic analytics, and a feedback button ("was this right?") for future data.
