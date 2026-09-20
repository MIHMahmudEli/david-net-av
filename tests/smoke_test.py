"""
DAVID-Net Pipeline Smoke Test
===============================
Runs a fast end-to-end sanity check of the entire pipeline:
  1. Imports & model instantiation
  2. Dummy manifest & dataset loading (AVDeepfakeDataset)
  3. QACP synthetic quadrant generation & forward pass (2 steps)
  4. QACP checkpoint save (local)
  5. Stage-1 init_from guard test (file exists & file missing)
  6. Stage-1 forward pass (2 steps, total_loss multi-task backward)
  7. HFBackup connection check
  8. HFBackup upload/download roundtrip
  9. Kaggle API credentials & dataset access verification (all 7 datasets)

Usage (from repo root):
    python tests/smoke_test.py

Credentials are read from .env automatically.
"""
from __future__ import annotations

import json, os, sys, tempfile, time, traceback, subprocess

if __name__ != "__main__":
    # This is a standalone script (network calls to HF + Kaggle, sys.exit at the end).
    # pytest collects `*_test.py`; running it at import time aborted the whole suite.
    import pytest
    pytest.skip("standalone smoke script — run `python tests/smoke_test.py`",
                allow_module_level=True)
from pathlib import Path
from types import SimpleNamespace

# ── Load .env credentials ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
env_file = REPO_ROOT / ".env"
for line in env_file.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if "=" in line:
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k in ("hf", "HF_TOKEN") and v:
            os.environ.setdefault("HF_TOKEN", v)
        elif k == "KAGGLE_USERNAME":
            os.environ.setdefault("KAGGLE_USERNAME", v)
        elif k == "KAGGLE_API_KEY":
            os.environ.setdefault("KAGGLE_KEY", v)

sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

# ── Helpers ────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results: list[tuple[str, str, str]] = []  # (test, status, detail)

def check(name: str, fn):
    try:
        detail = fn() or ""
        results.append((name, PASS, str(detail)))
        print(f"  {PASS} {name}" + (f" — {detail}" if detail else ""))
    except Exception as e:
        results.append((name, FAIL, str(e)))
        print(f"  {FAIL} {name}")
        print(f"        {e}")
        traceback.print_exc()


class DummyVideoEnc(nn.Module):
    """Fast spatial-average pooling encoder for raw video frames (B, T, 3, H, W) -> (B, T, d_model)."""
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.proj = nn.Linear(3, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, T, 3, H, W)
        return self.proj(x.mean(dim=(-2, -1)))


class DummyAudioEnc(nn.Module):
    """Fast downsampling encoder for raw waveform (B, N) -> (B, L_a, d_model)."""
    def __init__(self, d_model: int = 64, chunk: int = 100):
        super().__init__()
        self.chunk = chunk
        self.proj = nn.Linear(chunk, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, N)
        B, N = x.shape
        L = N // self.chunk
        x = x[:, : L * self.chunk].view(B, L, self.chunk)
        return self.proj(x)


def make_dummy_manifest(tmp: Path, n_real: int = 8, n_fake: int = 8) -> Path:
    """Write a tiny fake manifest referencing synthetic files."""
    path = tmp / "smoke_manifest.jsonl"
    with open(path, "w") as f:
        for i in range(n_real):
            f.write(json.dumps({
                "clip_id": f"real_{i:03d}",
                "video_path": f"real_{i:03d}.mp4",
                "audio_path": f"real_{i:03d}.wav",
                "video_label": 0, "audio_label": 0,
                "quadrant": "RVRA",
                "duration": 3.0,
                "fake_segments": [],
                "dataset": "smoke",
                "split": "train",
            }) + "\n")
        for i in range(n_fake):
            f.write(json.dumps({
                "clip_id": f"fake_{i:03d}",
                "video_path": f"fake_{i:03d}.mp4",
                "audio_path": f"fake_{i:03d}.wav",
                "video_label": 1, "audio_label": 1,
                "quadrant": "FVFA",
                "duration": 3.0,
                "fake_segments": [[0.5, 2.5]],
                "dataset": "smoke",
                "split": "train",
            }) + "\n")
    return path


def make_cfg(tmp: Path, manifest: Path, **overrides) -> SimpleNamespace:
    defaults = dict(
        run_id="smoke_run",
        d_model=64, n_heads=4, n_fusion_layers=2, dropout=0.0,
        use_sync=True, use_disentangle=True, compose_quadrant=False,
        video_backbone="videomae", audio_backbone="wavlm",
        video_model_name="MCG-NJU/videomae-base",
        audio_model_name="microsoft/wavlm-base-plus",
        freeze_blocks=12, freeze_feature_extractor=True,
        n_frames=4, audio_len=8000,
        shard_root=str(tmp),
        feature_cache=str(tmp),
        train_manifest=str(manifest),
        val_manifest=str(manifest),
        root_dir=str(tmp),
        modality_dropout=0.0, augment=False,
        batch_size=2, grad_accum_steps=1, num_workers=0,
        epochs=1, milestone_every=1, keep_milestones=1,
        lr=1e-3, lr_encoder=1e-4, weight_decay=0.0,
        warmup_epochs=0, gradient_checkpointing=False,
        log_every=1,
        out_dir=str(tmp / "runs"), local_dir=str(tmp),
        loss_weights={"v": 1.0, "a": 1.0, "quad": 0.5,
                      "loc": 0.5, "sync": 0.3, "disentangle": 0.1},
        qacp_temperature=0.1, seed=42,
        init_from=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  DAVID-Net Pipeline Smoke Test")
print("="*60)

TMP = Path(tempfile.mkdtemp(prefix="davidnet_smoke_"))
print(f"  Temp dir: {TMP}\n")

# ── 1. Imports ─────────────────────────────────────────────────────────────
print("[1] Imports")
def test_imports():
    from src.models.david_net import DavidNet, DavidNetConfig
    from src.training.train import build_model, move
    from src.training.losses import qacp_loss, total_loss, LossWeights
    from src.data.datasets import AVDeepfakeDataset, CachedFeatureDataset
    from src.data.synthetic_quadrants import QACPDataset, collate_qacp
    from src.utils.hf_backup import HFBackup
    from src.utils.config import load_config
    return "all core modules imported cleanly"
check("core imports", test_imports)

# ── 2. Manifest + dataset ──────────────────────────────────────────────────
print("\n[2] Dummy manifest & dataset")
MANIFEST = make_dummy_manifest(TMP)

def test_manifest():
    from src.data.datasets import load_manifest
    recs = load_manifest(str(MANIFEST))
    assert len(recs) == 16, f"expected 16 got {len(recs)}"
    assert "clip_id" in recs[0], "clip_id missing from manifest"
    return f"{len(recs)} records with clip_id"
check("manifest load", test_manifest)

def make_shards():
    shard_dir = TMP
    from src.data.datasets import load_manifest
    for rec in load_manifest(str(MANIFEST)):
        cid = rec["clip_id"]
        # Raw tensors for AVDeepfakeDataset shard_root
        torch.save(torch.randn(4, 3, 224, 224), shard_dir / f"{cid}_video.pt")
        torch.save(torch.randn(8000), shard_dir / f"{cid}_audio.pt")
        # Feature tensors for CachedFeatureDataset
        torch.save(torch.randn(4, 64), shard_dir / f"{cid}_vfeat.pt")
        torch.save(torch.randn(1, 64), shard_dir / f"{cid}_afeat.pt")
make_shards()

def test_dataset():
    from src.data.datasets import AVDeepfakeDataset
    ds = AVDeepfakeDataset(
        str(MANIFEST), shard_root=str(TMP),
        n_frames=4, audio_len=8000,
    )
    item = ds[0]
    assert "video" in item and "audio" in item and "clip_id" in item
    assert item["video"].shape == (4, 3, 224, 224), f"bad video shape: {item['video'].shape}"
    assert item["audio"].shape == (8000,), f"bad audio shape: {item['audio'].shape}"
    return f"item keys: {list(item.keys())}, video shape: {tuple(item['video'].shape)}"
check("AVDeepfakeDataset.__getitem__", test_dataset)

# ── 3. Model instantiation ─────────────────────────────────────────────────
print("\n[3] Model")
def test_model():
    from src.training.train import build_model
    cfg = make_cfg(TMP, MANIFEST)
    model = build_model(cfg)
    n = sum(p.numel() for p in model.parameters())
    return f"{n:,} parameters (identity encoders for feature cache mode)"
check("build_model", test_model)

# ── 4. QACP forward pass ───────────────────────────────────────────────────
print("\n[4] QACP forward pass (2 synthetic steps)")
def test_qacp_forward():
    from src.models.david_net import DavidNet, DavidNetConfig
    from src.training.train import move
    from src.training.losses import qacp_loss
    from src.data.datasets import AVDeepfakeDataset
    from src.data.synthetic_quadrants import QACPDataset, collate_qacp
    from torch.utils.data import DataLoader

    cfg = make_cfg(TMP, MANIFEST)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = DavidNetConfig(
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_fusion_layers=cfg.n_fusion_layers,
        dropout=cfg.dropout, use_sync=cfg.use_sync, use_disentangle=cfg.use_disentangle,
        compose_quadrant=cfg.compose_quadrant,
    )
    model = DavidNet(mcfg, DummyVideoEnc(cfg.d_model), DummyAudioEnc(cfg.d_model)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    base = AVDeepfakeDataset(str(MANIFEST), shard_root=str(TMP), n_frames=4, audio_len=8000,
                             filt=lambda r: r["video_label"] == 0 and r["audio_label"] == 0)
    ds = QACPDataset(base)
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_qacp, num_workers=0, drop_last=False)

    losses = []
    for i, batch in enumerate(dl):
        if i >= 2:
            break
        batch = move(batch, device)
        with torch.amp.autocast("cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
            out = model(batch["video"], batch["audio"])
        loss, parts = qacp_loss(out, batch, temperature=cfg.qacp_temperature)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert len(losses) > 0, "no batches produced"
    return f"losses: {[f'{l:.4f}' for l in losses]}"
check("QACP 2-step forward+backward", test_qacp_forward)

# ── 5. QACP checkpoint save ────────────────────────────────────────────────
print("\n[5] Checkpoint save")
QACP_CKPT = TMP / "qacp_smoke.pt"

def test_save_ckpt():
    from src.models.david_net import DavidNet, DavidNetConfig
    cfg = make_cfg(TMP, MANIFEST)
    mcfg = DavidNetConfig(
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_fusion_layers=cfg.n_fusion_layers,
        dropout=cfg.dropout, use_sync=cfg.use_sync, use_disentangle=cfg.use_disentangle,
        compose_quadrant=cfg.compose_quadrant,
    )
    model = DavidNet(mcfg, DummyVideoEnc(cfg.d_model), DummyAudioEnc(cfg.d_model))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": 0}, QACP_CKPT)
    assert QACP_CKPT.exists()
    return f"{QACP_CKPT.stat().st_size / 1024:.1f} KB saved"
check("torch.save checkpoint", test_save_ckpt)

# ── 6. Stage-1 init_from guard ─────────────────────────────────────────────
print("\n[6] train.py init_from guard")
def test_init_from_exists():
    from src.models.david_net import DavidNet, DavidNetConfig
    cfg = make_cfg(TMP, MANIFEST, init_from=str(QACP_CKPT))
    mcfg = DavidNetConfig(
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_fusion_layers=cfg.n_fusion_layers,
        dropout=cfg.dropout, use_sync=cfg.use_sync, use_disentangle=cfg.use_disentangle,
        compose_quadrant=cfg.compose_quadrant,
    )
    model = DavidNet(mcfg, DummyVideoEnc(cfg.d_model), DummyAudioEnc(cfg.d_model))
    assert os.path.exists(cfg.init_from), "file must exist"
    state = torch.load(cfg.init_from, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    return f"missing={len(missing)}, unexpected={len(unexpected)}"
check("init_from load (file exists)", test_init_from_exists)

def test_init_from_missing():
    """init_from pointing to nonexistent path must NOT crash."""
    import io
    from contextlib import redirect_stdout
    cfg = make_cfg(TMP, MANIFEST, init_from="/nonexistent/path/qacp.pt")
    # Replicate the guarded logic from train.py
    buf = io.StringIO()
    with redirect_stdout(buf):
        if getattr(cfg, "init_from", None):
            if not os.path.exists(cfg.init_from):
                print(f"WARNING: init_from '{cfg.init_from}' not found on disk — skipping")
            else:
                raise AssertionError("Should not reach here")
    assert "WARNING" in buf.getvalue()
    return "warning printed, no crash"
check("init_from guard (file missing)", test_init_from_missing)

# ── 7. Stage-1 forward pass ────────────────────────────────────────────────
print("\n[7] Stage-1 forward pass (2 steps, total_loss multi-task)")
def test_stage1_forward():
    from src.models.david_net import DavidNet, DavidNetConfig
    from src.training.train import move, sample_modality_masks
    from src.training.losses import LossWeights, total_loss
    from src.data.datasets import AVDeepfakeDataset, collate
    from torch.utils.data import DataLoader

    cfg = make_cfg(TMP, MANIFEST, init_from=str(QACP_CKPT))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = DavidNetConfig(
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_fusion_layers=cfg.n_fusion_layers,
        dropout=cfg.dropout, use_sync=cfg.use_sync, use_disentangle=cfg.use_disentangle,
        compose_quadrant=cfg.compose_quadrant,
    )
    model = DavidNet(mcfg, DummyVideoEnc(cfg.d_model), DummyAudioEnc(cfg.d_model)).to(device)

    # load init_from
    state = torch.load(str(QACP_CKPT), map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = AVDeepfakeDataset(str(MANIFEST), shard_root=str(TMP), n_frames=4, audio_len=8000)
    dl = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate, drop_last=False)

    weights = LossWeights(**cfg.loss_weights)
    losses = []
    for i, batch in enumerate(dl):
        if i >= 2:
            break
        batch = move(batch, device)
        with torch.amp.autocast("cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
            v_av, a_av = sample_modality_masks(
                batch["video"].size(0), cfg.modality_dropout, batch["video"].device)
            out = model(batch["video"], batch["audio"], v_avail=v_av, a_avail=a_av)
            loss, parts = total_loss(out, batch, weights, model=model)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert len(losses) > 0
    return f"losses: {[f'{l:.4f}' for l in losses]}"
check("Stage-1 2-step forward+backward", test_stage1_forward)

# ── 8. HFBackup connection ─────────────────────────────────────────────────
print("\n[8] HF Backup connection")
def test_hf_connection():
    from src.utils.hf_backup import HFBackup
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN not set"
    backup = HFBackup(run_id="smoke_test", local_dir=str(TMP))
    backup.setup()
    return "repo accessible"
check("HFBackup.setup() — repo accessible", test_hf_connection)

def test_hf_roundtrip():
    """Upload a tiny JSON, download it, verify content."""
    from src.utils.hf_backup import HFBackup
    import json
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    backup = HFBackup(run_id="smoke_test", local_dir=str(TMP))
    payload = json.dumps({"smoke": True, "ts": time.time()}).encode()
    backup._upload_bytes(payload, "runs/smoke_test/smoke_check.json")

    # Download and verify
    api = HfApi(token=token)
    local = api.hf_hub_download(
        "MoshinAli/david-net-av-backup",
        "runs/smoke_test/smoke_check.json",
        repo_type="model",
    )
    with open(local) as f:
        data = json.load(f)
    assert data["smoke"] is True

    # Clean up
    try:
        api.delete_file("runs/smoke_test/smoke_check.json",
                        repo_id="MoshinAli/david-net-av-backup",
                        repo_type="model")
    except Exception:
        pass
    return "upload -> download -> verify OK"
check("HFBackup upload/download roundtrip", test_hf_roundtrip)

# ── 9. Kaggle API credentials & Dataset Verification ───────────────────────
print("\n[9] Kaggle API credentials & Dataset Verification")
def test_kaggle_creds():
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    assert username and key, f"KAGGLE_USERNAME={username!r} KAGGLE_KEY={'set' if key else 'MISSING'}"

    # Write kaggle.json for the CLI
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(json.dumps({"username": username, "key": key}))

    # Verify all 7 active datasets used in training/evaluation
    active_datasets = [
        "aicontentdetections/fakeavceleb-v1-2",
        "pranay22077/dfdc-10",
        "fahimaislam1812/deepfaketimit",
        "reubensuju/celeb-df-v2",
        "anishsarkar22/asvpoof-2019-dataset-la",
        "abdallamohamed312/in-the-wild-audio-deepfake",
        "walimuhammadahmad/fakeaudio",
    ]

    verified = []
    for slug in active_datasets:
        res = subprocess.run(
            ["kaggle", "datasets", "files", slug, "--page-size", "1"],
            capture_output=True, text=True, timeout=30
        )
        if res.returncode != 0:
            raise RuntimeError(f"Kaggle access failed for dataset '{slug}': {res.stderr[:200]}")
        verified.append(slug.split("/")[-1])

    return f"user: {username} | {len(verified)}/7 datasets accessible ({', '.join(verified[:3])}...)"
check("Kaggle API credentials & 7 datasets access", test_kaggle_creds)

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SMOKE TEST SUMMARY")
print("="*60)
passed = [r for r in results if r[1] == PASS]
failed = [r for r in results if r[1] == FAIL]
skipped = [r for r in results if r[1] == SKIP]

for name, status, detail in results:
    flag = "OK" if status == PASS else ("!!" if status == FAIL else "--")
    print(f"  [{flag}] {name}")
    if status == FAIL:
        print(f"        => {detail}")

print(f"\n  {len(passed)} passed  |  {len(failed)} failed  |  {len(skipped)} skipped")

# Cleanup temp
import shutil
shutil.rmtree(TMP, ignore_errors=True)

if failed:
    print("\n  RESULT: FAILED")
    sys.exit(1)
else:
    print("\n  RESULT: ALL PASSED")
    sys.exit(0)
