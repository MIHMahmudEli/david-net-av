"""Train + evaluate any registered baseline under OUR manifests/splits.

The point: identical data, identical splits, identical metrics as DAVID-Net, so
Table 1 / Table 2 comparisons are apples-to-apples. Results are written in the
same results-JSON schema the figure generator consumes.

Usage:
    python -m src.baselines.train_baseline --baseline video-framecnn \
        --train-manifest src/data/splits/fakeavceleb/train.jsonl \
        --test-manifest  src/data/splits/fakeavceleb/test.jsonl \
        --shard-root data/shards/fakeavceleb --epochs 10 \
        --out results/baseline_video-framecnn.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.baselines.models import build_baseline
from src.data.datasets import AVDeepfakeDataset, collate
from src.eval.metrics import per_modality
from src.utils.seed import set_seed


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_baseline(args.baseline).to(device)
    label_key = f"{model.modality}_label"

    root_dir = getattr(args, "root_dir", None)
    train_ds = AVDeepfakeDataset(args.train_manifest, args.shard_root,
                                 args.n_frames, args.audio_len, root_dir=root_dir, train=True)
    test_ds = AVDeepfakeDataset(args.test_manifest, args.shard_root,
                                args.n_frames, args.audio_len, root_dir=root_dir, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()

    model.train()
    steps = 0
    for epoch in range(args.epochs):
        for batch in train_dl:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            opt.zero_grad(set_to_none=True)
            loss = bce(model(batch), batch[label_key].float())
            loss.backward()
            opt.step()
            steps += 1
            if steps % args.log_every == 0:
                print(f"[{args.baseline}] epoch {epoch} step {steps} loss {loss:.4f}")
            if args.dry_run and steps >= 2:
                print("[dry-run] baseline forward/backward OK")
                epoch = args.epochs  # noqa: PLW2901
                break
        if args.dry_run and steps >= 2:
            break

    # ---- evaluate: same metric code as the main model
    model.eval()
    ys, ps, clip_ids = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            ps += torch.sigmoid(model(batch)).cpu().tolist()
            ys += batch[label_key].cpu().tolist()
            clip_ids += batch["clip_id"]

    report = {
        "method": args.baseline,
        "modality": model.modality,
        "seed": args.seed,
        "test_manifest": args.test_manifest,
        "metrics": {model.modality: per_modality(ys, ps)},
        "preds": {"clip_id": clip_ids, "y_true": ys, "y_score": ps},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f)
    print(f"{args.baseline}: {report['metrics']} -> {args.out}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--shard-root", default=None)
    ap.add_argument("--root-dir", default=None, help="media root the manifest rel_paths are relative to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--audio-len", type=int, default=64000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
