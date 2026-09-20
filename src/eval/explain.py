"""Explainability outputs for the paper (Fig. C) and the API `explain=true` path.

For one clip, produces:
  * video saliency  — input-gradient x input attribution of the VIDEO authenticity logit,
                      aggregated per frame to a spatial heat-map (SmoothGrad, 4 samples);
                      shown as overlays on 4 evenly spaced frames
  * audio saliency  — the same attribution of the AUDIO logit w.r.t. the waveform,
                      smoothed to a per-time envelope and overlaid on the log-mel spectrogram
  * sync agreement  — per-window cosine agreement from the sync module
  * localization    — per-frame P(manipulated) for both streams

Gradient-based attribution is used instead of Grad-CAM because the backbones are ViTs
(VideoMAE tubelets, WavLM frames) with no spatial conv feature map; SmoothGrad is the
standard ViT-compatible substitute and needs no hooks.

    python -m src.eval.explain --config <stage1.yaml> --checkpoint best.pt \
        --manifest splits/fakeavceleb/test.jsonl --root-dir <root> --out paper/figures --n 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.datasets import AVDeepfakeDataset, collate
from src.training.train import build_model, availability_masks
from src.utils.config import load_config

QUADRANTS = ["RVRA", "RVFA", "FVRA", "FVFA"]


def _smooth1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")


@torch.no_grad()
def _forward(model, video, audio, v_av, a_av):
    return model(video, audio, v_avail=v_av, a_avail=a_av)


def attribute(model, video, audio, v_av, a_av, n_samples: int = 4, noise: float = 0.05):
    """SmoothGrad input-gradient x input for logit_v (w.r.t. frames) and logit_a (w.r.t. wave).

    video: (1, T, 3, H, W) in [0,1]; audio: (1, N). Returns (frame_maps (T, H, W), audio_env (N,)).
    """
    model.eval()
    sal_v = torch.zeros_like(video)
    sal_a = torch.zeros_like(audio)
    for _ in range(n_samples):
        v = (video + noise * torch.randn_like(video)).clamp(0, 1).requires_grad_(True)
        a = (audio + noise * audio.std() * torch.randn_like(audio)).requires_grad_(True)
        out = model(v, a, v_avail=v_av, a_avail=a_av)
        gv, = torch.autograd.grad(out["logit_v"].float().sum(), v, retain_graph=True)
        ga, = torch.autograd.grad(out["logit_a"].float().sum(), a)
        sal_v += (gv * v).detach().abs()
        sal_a += (ga * a).detach().abs()
    frame_maps = sal_v[0].sum(1) / n_samples                       # (T, H, W)
    # blur each map a little so tubelet-level attributions read as regions
    fm = F.avg_pool2d(frame_maps.unsqueeze(1), 15, stride=1, padding=7).squeeze(1)
    fm = fm / fm.flatten(1).max(1).values.clamp(min=1e-12).view(-1, 1, 1)
    env = sal_a[0].cpu().numpy() / n_samples
    return fm.cpu().numpy(), env


def logmel(wave: np.ndarray, sr: int = 16000, n_fft: int = 400, hop: int = 160, n_mels: int = 64):
    """Numpy log-mel for plotting (no torchaudio dependency)."""
    x = torch.from_numpy(wave).float()
    win = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True).abs() ** 2
    freqs = np.linspace(0, sr / 2, spec.size(0))
    mel_pts = np.linspace(0, 2595 * np.log10(1 + sr / 2 / 700), n_mels + 2)
    hz = 700 * (10 ** (mel_pts / 2595) - 1)
    fb = np.zeros((n_mels, spec.size(0)))
    for m in range(n_mels):
        lo, c, hi = hz[m], hz[m + 1], hz[m + 2]
        fb[m] = np.clip(np.minimum((freqs - lo) / max(c - lo, 1e-6), (hi - freqs) / max(hi - c, 1e-6)), 0, 1)
    mel = torch.from_numpy(fb).float() @ spec
    return torch.log(mel.clamp(min=1e-8)).numpy()


def explain_clip(model, sample: dict, device: str) -> dict:
    batch = collate([sample])
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    v_av, a_av = availability_masks(batch, 0.0)
    out = _forward(model, batch["video"], batch["audio"], v_av, a_av)
    fm, env = attribute(model, batch["video"], batch["audio"], v_av, a_av)
    dur = batch["audio"].size(1) / 16000.0
    return {
        "clip_id": sample["clip_id"],
        "quadrant_true": QUADRANTS[int(sample["quadrant"])],
        "quadrant_pred": QUADRANTS[int(out["logit_quad"].argmax(-1))],
        "p_video_fake": float(torch.sigmoid(out["logit_v"].float())),
        "p_audio_fake": float(torch.sigmoid(out["logit_a"].float())),
        "frames": batch["video"][0].cpu().numpy(),                 # (T, 3, H, W)
        "frame_saliency": fm,                                      # (T, H, W)
        "wave": batch["audio"][0].cpu().numpy(),
        "audio_saliency": env,
        "agreement": out["agreement"][0].float().cpu().numpy() if out["agreement"] is not None else None,
        "loc_v": torch.sigmoid(out["loc_v"][0].float()).cpu().numpy(),
        "loc_a": torch.sigmoid(out["loc_a"][0].float()).cpu().numpy(),
        "duration": dur,
    }


def render(ex: dict, out_path: Path, n_frames_show: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = ex["frames"].shape[0]
    idx = np.linspace(0, T - 1, n_frames_show).round().astype(int)
    fig = plt.figure(figsize=(7.0, 6.0))
    gs = fig.add_gridspec(3, n_frames_show, height_ratios=[1.4, 1.1, 1.0], hspace=0.45, wspace=0.08)
    for j, i in enumerate(idx):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(np.transpose(ex["frames"][i], (1, 2, 0)))
        ax.imshow(ex["frame_saliency"][i], cmap="inferno", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(f"t = {i / max(T - 1, 1) * ex['duration']:.1f} s", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    mel = logmel(ex["wave"])
    ax.imshow(mel, aspect="auto", origin="lower", cmap="magma",
              extent=[0, ex["duration"], 0, mel.shape[0]])
    env = _smooth1d(ex["audio_saliency"], 800)
    env = env / max(env.max(), 1e-12) * mel.shape[0]
    t = np.linspace(0, ex["duration"], len(env))
    ax.plot(t, env, color="cyan", linewidth=0.8, label="audio saliency")
    ax.set_ylabel("mel bin"); ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"P(video fake) = {ex['p_video_fake']:.2f}   P(audio fake) = {ex['p_audio_fake']:.2f}   "
                 f"quadrant: true {ex['quadrant_true']} / pred {ex['quadrant_pred']}", fontsize=8)
    ax = fig.add_subplot(gs[2, :])
    tv = np.linspace(0, ex["duration"], len(ex["loc_v"]))
    ta = np.linspace(0, ex["duration"], len(ex["loc_a"]))
    ax.plot(tv, ex["loc_v"], label="P(video manipulated)")
    ax.plot(ta, ex["loc_a"], label="P(audio manipulated)")
    if ex["agreement"] is not None:
        ts = np.linspace(0, ex["duration"], len(ex["agreement"]))
        ax.plot(ts, (ex["agreement"] + 1) / 2, linestyle="--", label="AV agreement (rescaled)")
    ax.set_ylim(0, 1.02); ax.set_xlabel("Time (s)"); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(ex["clip_id"][:70], fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root-dir", default=None)
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--n", type=int, default=3, help="clips per manipulated quadrant (RVFA, FVRA, FVFA)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.root_dir:
        cfg.root_dir = args.root_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = AVDeepfakeDataset(args.manifest, cfg.shard_root, cfg.n_frames, cfg.audio_len,
                           root_dir=getattr(cfg, "root_dir", None), train=False)
    picked = []
    for q in ("RVFA", "FVRA", "FVFA"):
        cands = [i for i, r in enumerate(ds.records) if r["quadrant"] == q]
        picked += cands[: args.n]
    out = Path(args.out)
    index = []
    for i in picked:
        ex = explain_clip(model, ds[i], device)
        name = f"results_explain_{ex['quadrant_true']}_{i}"
        render(ex, out / name)
        index.append({k: ex[k] for k in ("clip_id", "quadrant_true", "quadrant_pred", "p_video_fake", "p_audio_fake")}
                     | {"figure": name})
        print(f"  {name}: {index[-1]}")
    (out / "explain_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
