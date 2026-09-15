"""Reconnaissance script: pull Kaggle dataset metadata without downloading.

Reads .env for credentials, queries Kaggle API for file listings,
and saves a structured JSON report per dataset.

Usage:
    python scripts/kaggle_recon.py
"""
import csv
import io
import json
import os
import sys
from pathlib import Path

def load_env(env_path: str = ".env") -> dict:
    """Parse .env file into a dict. Ignores comments and empty lines."""
    env = {}
    p = Path(env_path)
    if not p.exists():
        print(f"WARNING: {env_path} not found")
        return env
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip()
    return env


def parse_kaggle_url(url: str) -> str:
    """Extract owner/dataset-slug from a Kaggle URL."""
    url = url.rstrip("/")
    parts = url.split("/")
    # https://www.kaggle.com/datasets/owner/slug
    if "datasets" in parts:
        idx = parts.index("datasets")
        if idx + 2 < len(parts):
            return f"{parts[idx+1]}/{parts[idx+2]}"
    # already in owner/slug form
    if len(parts) == 2:
        return url
    return url


def get_dataset_files(slug: str, username: str, key: str) -> list[dict]:
    """List all files in a Kaggle dataset using the CLI."""
    import subprocess
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "files", slug, "--csv"
    ]
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return []
    
    # Parse CSV output (skip first line if it's a next-page token)
    lines = result.stdout.strip().split("\n")
    if not lines:
        return []
    
    # Find the header line
    header_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("name,"):
            header_idx = i
            break
    
    reader = csv.DictReader(lines[header_idx:])
    files = []
    for row in reader:
        name = row.get("name", "")
        size_str = row.get("size", "0")
        try:
            size = int(size_str)
        except ValueError:
            size = 0
        files.append({"name": name, "size": size})
    return files


def analyze_structure(files: list[dict]) -> dict:
    """Analyze file listing to infer directory structure."""
    tree = {}
    exts = {}
    total_size = 0
    
    for f in files:
        parts = f["name"].split("/")
        total_size += f["size"]
        
        # Track extensions
        ext = Path(f["name"]).suffix.lower()
        exts[ext] = exts.get(ext, 0) + 1
        
        # Build tree (2 levels deep)
        if len(parts) >= 2:
            top = parts[0]
            if top not in tree:
                tree[top] = {"subdirs": set(), "file_count": 0, "total_size": 0}
            tree[top]["file_count"] += 1
            tree[top]["total_size"] += f["size"]
            if len(parts) >= 3:
                tree[top]["subdirs"].add(parts[1])
    
    # Convert sets to lists for JSON
    for v in tree.values():
        v["subdirs"] = sorted(v["subdirs"])
    
    return {
        "tree": tree,
        "extensions": exts,
        "total_size_bytes": total_size,
        "total_size_gb": round(total_size / (1024**3), 2),
        "file_count": len(files),
    }


def fmt_size(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    env = load_env()
    
    username = env.get("KAGGLE_USERNAME", "")
    key = env.get("KAGGLE_API_KEY", "")
    
    if not username or not key:
        print("ERROR: KAGGLE_USERNAME and KAGGLE_API_KEY must be set in .env")
        sys.exit(1)
    
    # Collect dataset URLs
    datasets = {}
    for k, v in env.items():
        if k.startswith("#") or k in ("KAGGLE_USERNAME", "KAGGLE_API_KEY", "hf"):
            continue
        if "kaggle.com" in v or "/" in v:
            slug = parse_kaggle_url(v)
            datasets[k] = slug
    
    print(f"Found {len(datasets)} datasets in .env\n")
    
    out_dir = Path("dataset_recon")
    out_dir.mkdir(exist_ok=True)
    
    for name, slug in datasets.items():
        print(f"=== {name} ({slug}) ===")
        files = get_dataset_files(slug, username, key)
        
        if not files:
            print(f"  No files found or access denied\n")
            report = {"name": name, "slug": slug, "error": "no files or access denied"}
        else:
            analysis = analyze_structure(files)
            report = {
                "name": name,
                "slug": slug,
                "files": files[:50],  # first 50 for reference
                "analysis": analysis,
            }
            print(f"  Files: {analysis['file_count']}")
            print(f"  Total size: {fmt_size(analysis['total_size_bytes'])}")
            print(f"  Extensions: {analysis['extensions']}")
            print(f"  Top-level structure:")
            for top, info in analysis["tree"].items():
                subdirs = info["subdirs"][:10]
                print(f"    {top}/ ({info['file_count']} files, {fmt_size(info['total_size'])})")
                if subdirs:
                    print(f"      subdirs: {', '.join(subdirs[:10])}")
            print()
        
        # Save report
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Saved: {out_path}\n")
    
    print("Done. Reports saved in dataset_recon/")


if __name__ == "__main__":
    main()
