# Hugging Face Docker Space for DAVID-Net. Exposes port 7860 (HF default).
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY . /app
ENV PYTHONPATH=/app
ENV DAVID_CONFIG=/app/configs/david_net_kaggle.yaml
# Weights are pulled from the public HF model repo at first request
# (override with DAVID_CHECKPOINT=/path/to/david_net.pt to use a local file).
ENV DAVID_HF_REPO=MoshinAli/david-net-av

EXPOSE 7860
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
