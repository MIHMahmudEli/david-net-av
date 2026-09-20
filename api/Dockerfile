# Hugging Face Docker Space for DAVID-Net. Exposes port 7860 (HF default).
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY . /app
ENV PYTHONPATH=/app
ENV DAVID_CONFIG=/app/configs/david_net_lite.yaml
# Weights are pulled from the public HF model repo at first request; the Space serves
# DAVID-Net-Lite (CPU-friendly). Set DAVID_HF_REPO=MoshinAli/david-net-av for the full
# model, or DAVID_CHECKPOINT=/path/to/david_net.pt for a local file. The checkpoint's
# embedded cfg overrides the architecture fields of DAVID_CONFIG.
ENV DAVID_HF_REPO=MoshinAli/david-net-av-lite

EXPOSE 7860
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
