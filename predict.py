#!/usr/bin/env python3
"""DAVID-Net: Command-Line Single-Clip Forensic Inference Tool.

Usage:
    python predict.py --input path/to/media.mp4 [--checkpoint checkpoints/model.safetensors] [--device cpu|cuda]
"""
import argparse
import json
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.inference import DavidNetInference
except ImportError:
    from api.inference import DavidNetInference


def main():
    parser = argparse.ArgumentParser(description="DAVID-Net Audio-Visual Deepfake Forensic Inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to video or audio file")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to model weights (.safetensors or .pt)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--device", "-d", type=str, default="auto", help="Compute device (auto, cpu, cuda)")
    parser.add_argument("--explain", action="store_true", help="Generate token-level cross-modal attention saliency")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    device = None if args.device == "auto" else args.device
    engine = DavidNetInference(
        checkpoint=args.checkpoint,
        config=args.config,
        device=device
    )

    is_audio = input_path.suffix.lower() in [".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"]
    if is_audio:
        res = engine.predict_audio(str(input_path), explain=args.explain)
    else:
        res = engine.predict(str(input_path), explain=args.explain)

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
