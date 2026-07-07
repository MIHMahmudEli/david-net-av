# Compute & Hardware Strategy — NVIDIA DGX Spark

Training and serving DAVID-Net on **NVIDIA DGX Spark** (GB10 Grace-Blackwell Superchip).

## 1. What the hardware is (the parts that matter here)

| Property | Value | Implication for this project |
|----------|-------|------------------------------|
| GPU | Blackwell, 5th-gen Tensor Cores | Strong low-precision (FP8/FP4) → great for **inference/demo** |
| Unified memory | **128 GB LPDDR5x**, CPU+GPU coherent | Fit large video backbones + long clips + big batches; **capacity is not the constraint** |
| Memory bandwidth | ~273 GB/s | **This is the real constraint.** ~7× lower than an A100 → per-step throughput is bandwidth-bound |
| CPU | 20-core Arm (Grace) | Decode/augmentation can bottleneck → push decode to the GPU (DALI) |
| Networking | ConnectX-7 | Two Sparks can be linked for a 2-node full finetune |
| OS / stack | DGX OS (**ARM64/aarch64**), CUDA, NGC | Some Python wheels lack ARM builds → use NVIDIA NGC containers |
| Form factor / power | Desktop, ~240 W | Can run **always-on** as a self-hosted inference box for the demo |

**One-line takeaway:** huge memory, modest bandwidth. Design the training loop to be *bandwidth-light* (cache features, freeze encoders) and let the 128 GB do the heavy lifting; use the FP4/FP8 Blackwell path for the deployed detector.

## 2. Training strategy: two phases

### Phase A — cached-feature training (primary; do ~90% of runs here)
1. **Preprocess once:** decode → face/mouth crop → resample audio (Section 4). Write tensor shards to local NVMe.
2. **Extract SSL features once:** run frozen VideoMAE (video) and WavLM (audio) over every clip and **cache the token sequences to disk** (`data/feats/<dataset>/<clip_id>.pt`). This is a single forward pass per clip, done once.
3. **Train only the fusion transformer + sync module + heads** on the cached features. No video decode, no backbone forward, tiny per-step memory traffic → fast iteration. All ablations and 3-seed runs live here.

> Why this fits DGX Spark: the expensive, bandwidth-heavy work (backbone forwards over raw pixels/waveforms) is amortized to a one-time cost; the repeated training loop touches only small cached tensors.

**QACP (Stage 0) lives entirely in Phase A:** encoders are frozen, so the synthetic-quadrant
transforms (Griffin-Lim copy-synthesis, self-blending) can be applied offline once, features
cached, and the factorized-SupCon pretraining runs on cached tensors. The 128 GB unified memory
is a direct win here — SupCon benefits from large batches.

### Phase B — end-to-end LoRA finetune (only the best config)
- Unfreeze encoders with **LoRA/adapters** (not full finetune), BF16, gradient checkpointing on the video backbone.
- Batch built with generator-balanced sampling; the 128 GB lets you use a large effective batch (or grad-accumulate) for stable BN/attention stats.
- Expect this to be the slow part; run it 2–4 times, not 20. If it dominates the schedule, link a **second Spark** over ConnectX-7 and shard with FSDP/DDP.

## 3. Precision & throughput

- **Training:** BF16 (Blackwell BF16 is well-supported and stable); reserve FP8 for the encoder forward passes during feature extraction if the framework supports it cleanly. Avoid FP16 (BF16 has better range, no loss-scaling headaches).
- **Inference/deploy:** quantize DAVID-Net-Lite to **FP8/FP4 with TensorRT** for the demo — this is where Blackwell shines and where the ~1 PFLOP number is real.
- **Realistic expectations (verify empirically, don't quote as fact):** treat the Spark as roughly "workstation-class, memory-rich but bandwidth-limited." A cached-feature Phase-A epoch over FakeAVCeleb should be minutes-to-low-hours; a Phase-B end-to-end epoch will be substantially slower. Benchmark one epoch early and set the schedule from the measured number.

## 4. Data pipeline on ARM (avoid the common pitfalls)

- **Run inside an NGC container** (`nvcr.io/nvidia/pytorch:*-py3`) — it's ARM64-ready with CUDA/cuDNN/DALI matched to the driver. Do **not** rely on `pip install` for everything on bare DGX OS; several CV/audio wheels have no aarch64 build.
- **Video decode → NVIDIA DALI** (GPU-accelerated decode + resize + crop). With only 20 Arm cores, CPU decode (OpenCV/decord) will bottleneck the loop; DALI moves it onto the GPU.
- **Face detection/alignment** (RetinaFace/InsightFace) is a **one-time preprocessing** step, not in the training loop. If ARM builds are painful, run this stage on any x86 box or Colab once and ship the cached crops to the Spark. It never needs to run again.
- **`mediapipe` on ARM is fragile** — prefer InsightFace/RetinaFace, or precompute landmarks off-device. Recorded as a known risk.
- Keep shards on the **local NVMe**, not a network mount, so the loader isn't I/O-starved.

## 5. Deployment on/around the Spark

- **Self-host the heavy model:** the Spark can serve **DAVID-Net-Base** locally via the FastAPI service (`api/`) with a TensorRT-optimized FP8/FP4 engine — the always-on desktop form factor makes it a viable private inference server.
- **Public demo stays on HF:** ship **DAVID-Net-Lite** to a Hugging Face Space (CPU/GPU) for the Next.js UI so external users aren't hitting your workstation. The Spark can host the Base model behind it for internal/heavy requests.
- **NVIDIA NIM / Triton** is an optional serving upgrade if you want batched, production-grade inference on the Spark.

## 6. Config knobs to add for this hardware

Add to `configs/*.yaml` (loader already tolerates unknown keys via defaults — wire these when you build the cached-feature loader):
```yaml
precision: bf16            # bf16 (train) | fp8 (feature-extract) | fp4 (deploy)
feature_cache: data/feats  # Phase-A cached SSL features; null = compute on the fly
freeze_encoders: true      # Phase A: true (cached). Phase B: false + LoRA
lora: {enable: false, r: 16, alpha: 32, dropout: 0.05}
grad_checkpointing: true   # for Phase B end-to-end
dali_decode: true          # GPU video decode instead of CPU
num_workers: 8             # Grace has 20 cores; leave headroom for DALI
```

## 7. Checklist before the first big run

- [ ] Running inside an NGC PyTorch container (ARM64) — verify `torch.cuda.is_available()`.
- [ ] Face crops + audio resampled and cached to local NVMe.
- [ ] SSL features extracted once and cached (`feature_cache`).
- [ ] One Phase-A epoch benchmarked → schedule set from the measured time.
- [ ] BF16 confirmed; W&B logging live.
- [ ] TensorRT FP8/FP4 export path smoke-tested for the Lite model.
