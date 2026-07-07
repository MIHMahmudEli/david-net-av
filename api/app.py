"""FastAPI service for DAVID-Net — deploy on a Hugging Face Docker Space.

Endpoints:
  GET  /health   -> liveness
  GET  /version  -> model version
  POST /predict  -> multipart video upload; returns per-modality verdict JSON

Run locally:
  uvicorn api.app:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="DAVID-Net Deepfake Detector", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))  # 50 MB
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from api.inference import DavidNetInference
        _engine = DavidNetInference(
            checkpoint=os.environ.get("DAVID_CHECKPOINT", ""),
            config=os.environ.get("DAVID_CONFIG", "configs/david_net.yaml"),
        )
    return _engine


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"model_version": get_engine().version}


@app.post("/predict")
async def predict(file: UploadFile = File(...), explain: bool = Query(False)):
    if file.content_type is None or not file.content_type.startswith("video"):
        raise HTTPException(status_code=415, detail="Upload a video file.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_BYTES} bytes.")

    suffix = os.path.splitext(file.filename or "clip.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        result = get_engine().predict(tmp.path if hasattr(tmp, "path") else tmp.name, explain=explain)
        return JSONResponse(result)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
