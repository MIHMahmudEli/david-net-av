# Hugging Face Docker Space for DAVID-Net. Exposes port 7860 (HF default).
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY . /app
ENV PYTHONPATH=/app
ENV DAVID_CONFIG=/app/configs/david_net.yaml
# ENV DAVID_CHECKPOINT=/app/weights/david_net_lite.pt

EXPOSE 7860
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
