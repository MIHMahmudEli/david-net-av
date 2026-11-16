#!/usr/bin/env python3
"""DAVID-Net: Benchmark Evaluation Harness.

Usage:
    python evaluate.py --manifest manifests/test_manifest.jsonl [--checkpoint checkpoints/model.safetensors]
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.inference import DavidNetInference
except ImportError:
    from api.inference import DavidNetInference


def main():
    parser = argparse.ArgumentParser(description="DAVID-Net Benchmark Evaluation Harness")
    parser.add_argument("--manifest", "-m", type=str, required=True, help="Path to test manifest JSONL file")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to model weights (.safetensors)")
    parser.add_argument("--device", "-d", type=str, default="auto", help="Device (cpu, cuda)")
    parser.add_argument("--output", "-o", type=str, default="evaluation_results.json", help="Path to save results")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest file {manifest_path} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing DAVID-Net evaluation on manifest: {manifest_path}")
    print("Pretrained checkpoints can be loaded from Hugging Face Hub (MIHMahmudEli/david-net-av-v2).")


if __name__ == "__main__":
    main()
