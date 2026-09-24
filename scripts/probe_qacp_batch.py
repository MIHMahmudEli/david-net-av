"""Does raising QACP's batch size lift the SupCon terms off their floors, and does it fit?

Run this on Kaggle BEFORE committing ~75 minutes to a Stage-0 rerun:

    python scripts/probe_qacp_batch.py --config /kaggle/working/qacp_config.yaml \\
        --batch-sizes 4,8,16 --steps 40

For each candidate batch size it reports three things:

  no-neg %   share of batches where an axis has NO negative at all. `supcon_loss` then
             minimises at ln(B-1) -- the floor qacp_c never left. Pure label arithmetic,
             no GPU needed, so it prints even on CPU.
  VRAM       peak allocated for one forward/backward. The T4 has 15.6 GB; QACP at
             batch 4 with gradient checkpointing peaked around 4.5 GB.
  floor %    share of optimiser steps whose three terms all sat on a floor. This is the
             number that matters: 0% means the objective still has something to teach.

Falls back to the label-arithmetic section alone when no GPU is present, which is enough
to choose a batch size; the VRAM and floor columns need the real encoders.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic_quadrants import QACP_CLASSES  # noqa: E402
from src.training.losses import at_loss_floor, supcon_floors  # noqa: E402

LABELS = {"RVRA": (0, 0, 0), "RVFA": (0, 1, 0), "FVRA": (1, 0, 0),
          "FVFA": (1, 1, 0), "MISMATCH": (0, 0, 1)}
AXES = ("video", "audio", "sync")


def no_negative_rate(batch_size: int, trials: int = 4000, seed: int = 0) -> dict:
    """Share of batches in which an axis has a single label value, i.e. no negatives."""
    g = torch.Generator().manual_seed(seed)
    hits = {ax: 0 for ax in AXES}
    for _ in range(trials):
        idx = torch.randperm(len(QACP_CLASSES), generator=g).tolist()
        classes = [QACP_CLASSES[idx[i % len(idx)]] for i in range(batch_size)]
        for a, ax in enumerate(AXES):
            if len({LABELS[c][a] for c in classes}) == 1:
                hits[ax] += 1
    return {ax: 100.0 * n / trials for ax, n in hits.items()}


def label_arithmetic(batch_sizes):
    print("label arithmetic -- how often an axis gets no negative at all\n")
    print(f"{'B':>5} {'ln(B-1)':>9} {'video':>9} {'audio':>9} {'sync':>9}")
    print("-" * 46)
    for B in batch_sizes:
        r = no_negative_rate(B)
        print(f"{B:>5} {math.log(max(2, B-1)):>9.4f} "
              + " ".join(f"{r[ax]:>8.1f}%" for ax in AXES))
    print("\nA non-zero sync column is the bug that pinned qacp_c at 1.0986 all run.\n")


def gpu_probe(cfg_path: str, batch_sizes, steps: int):
    from src.utils.config import load_config
    from src.data.datasets import AVDeepfakeDataset
    from src.data.synthetic_quadrants import QACPDataset, QACPBalancedSampler, collate_qacp
    from src.training.train import build_model, move, enable_gradient_checkpointing
    from src.training.losses import qacp_loss
    from torch.utils.data import DataLoader

    cfg = load_config(cfg_path)
    device = torch.device("cuda")
    # same RVRA filter pretrain() uses, so the probe sees the real Stage-0 data
    base = AVDeepfakeDataset(cfg.train_manifest, cfg.shard_root, cfg.n_frames, cfg.audio_len,
                             filt=lambda r: r["video_label"] == 0 and r["audio_label"] == 0,
                             root_dir=getattr(cfg, "root_dir", None), train=True)
    print(f"probing on {len(base)} pristine clips, {steps} optimiser steps per size\n")
    print(f"{'B':>5} {'VRAM peak':>11} {'floor %':>9} {'qacp_v':>9} {'qacp_a':>9} {'qacp_c':>9}")
    print("-" * 60)

    for B in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg.batch_size = B
        ds = QACPDataset(base, views_per_clip=getattr(cfg, "qacp_views_per_clip", 4), seed=cfg.seed)
        ds.set_epoch(0)
        sampler = QACPBalancedSampler(ds, B, seed=cfg.seed)
        sampler.set_epoch(0)
        dl = DataLoader(ds, batch_size=B, sampler=sampler, num_workers=2,
                        collate_fn=collate_qacp, drop_last=True)
        model = build_model(cfg).to(device)
        if getattr(cfg, "gradient_checkpointing", False):
            enable_gradient_checkpointing(model)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

        floor_hits, n, sums = 0, 0, {}
        try:
            for batch in dl:
                batch = move(batch, device)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(batch["video"], batch["audio"])
                    loss, parts = qacp_loss(out, batch, getattr(cfg, "qacp_temperature", 0.1))
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                floor_hits += at_loss_floor(parts, B)
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0.0) + v
                n += 1
                if n >= steps:
                    break
            vram = torch.cuda.max_memory_allocated() / 1e9
            avg = {k: v / max(1, n) for k, v in sums.items()}
            print(f"{B:>5} {vram:>9.2f} GB {100*floor_hits/max(1,n):>8.0f}% "
                  f"{avg.get('qacp_v', 0):>9.4f} {avg.get('qacp_a', 0):>9.4f} {avg.get('qacp_c', 0):>9.4f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>5} {'OOM':>11} {'-':>9} {'-':>9} {'-':>9} {'-':>9}")
        finally:
            del model, opt, dl
            torch.cuda.empty_cache()

    print(f"\nfloors for reference: {', '.join(f'{f:.4f}' for f in supcon_floors(batch_sizes[0]))} at B={batch_sizes[0]}")
    print("Pick the largest B that fits with floor % at 0.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="qacp_config.yaml; omit to run label arithmetic only")
    ap.add_argument("--batch-sizes", default="4,8,16")
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()
    sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]

    label_arithmetic(sizes)
    if not args.config:
        print("no --config given: skipping the GPU probe")
        return
    if not torch.cuda.is_available():
        print("no GPU here: skipping the VRAM/floor probe (run this on Kaggle)")
        return
    gpu_probe(args.config, sizes, args.steps)


if __name__ == "__main__":
    main()
