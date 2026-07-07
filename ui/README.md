# DAVID-Net Web UI (Next.js)

End-user interface: upload a video → see per-modality (video / audio) verdicts, the
authenticity quadrant, temporal manipulation timelines, and the AV-sync curve.

## Stack
- Next.js (App Router) + TypeScript
- Tailwind CSS
- A charting lib for the dual timeline + sync curve (e.g. Recharts or lightweight SVG)
- Calls the FastAPI HF Space via a **server-side route handler** (keeps the API URL/token off the client and sidesteps CORS)

## Scaffold
```bash
npx create-next-app@latest david-net-ui --typescript --tailwind --app --eslint
cd david-net-ui
# set the API base:
echo "API_URL=https://<your-space>.hf.space" >> .env.local
```

## Suggested structure
```
david-net-ui/
  app/
    page.tsx                 # upload screen
    result/page.tsx          # verdict screen
    api/predict/route.ts     # server proxy → forwards multipart to HF Space
  components/
    Uploader.tsx             # drag-drop, size/duration guard, progress
    VerdictCard.tsx          # one card per modality (video / audio) + confidence gauge
    QuadrantBadge.tsx        # RVRA / RVFA / FVRA / FVFA
    DualTimeline.tsx         # video track + audio track, manipulated intervals highlighted
    SyncCurve.tsx            # per-frame agreement line
    Disclaimer.tsx           # "probabilistic, not proof"
  lib/api.ts                 # typed client for the /predict response
```

## Server proxy (app/api/predict/route.ts) — sketch
```ts
export async function POST(req: Request) {
  const form = await req.formData();
  const res = await fetch(`${process.env.API_URL}/predict`, { method: "POST", body: form });
  return new Response(await res.text(), {
    status: res.status,
    headers: { "content-type": "application/json" },
  });
}
```

## Response type (mirror of docs/05_deployment.md)
```ts
export type Verdict = { verdict: "real" | "fake"; confidence: number };
export type PredictResponse = {
  clip_id: string; duration_sec: number;
  video: Verdict; audio: Verdict;
  quadrant: { label: string; probs: Record<string, number> };
  localization: { video: {start:number;end:number;score:number}[]; audio: {start:number;end:number;score:number}[] };
  sync_curve: number[];
  model_version: string; latency_ms: number;
};
```

## UX rules
- Client-side guard: max size + max duration (e.g. 60 s) before upload.
- Always show the disclaimer and the model version next to the verdict.
- Show confidence, never a bare "FAKE" — display probability + calibration note.
- Optional feedback button ("was this right?") to collect (opt-in) evaluation data.

## Deploy
- Push to GitHub → import in Vercel → set `API_URL` env var → deploy.
