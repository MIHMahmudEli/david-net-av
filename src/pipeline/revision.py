"""Canonical pinned revision reader for DAVID-Net-AV.
Single source of truth: reports/CODE_REVISION.json.
"""
from __future__ import annotations

import json
from pathlib import Path

_CACHED_DATA: dict | None = None


def _load_revision_data() -> dict:
    global _CACHED_DATA

    candidates = [
        Path(__file__).resolve().parents[2] / "report" / "final" / "CODE_REVISION.json",
        Path(__file__).resolve().parents[2] / "report" / "CODE_REVISION.json",
        Path(__file__).resolve().parents[2] / "reports" / "CODE_REVISION.json",
        Path("report/final/CODE_REVISION.json"),
        Path("report/CODE_REVISION.json"),
        Path("reports/CODE_REVISION.json"),
        Path("/kaggle/working/Thesis/report/final/CODE_REVISION.json"),
        Path("/kaggle/working/Thesis/report/CODE_REVISION.json"),
        Path("/kaggle/working/Thesis/reports/CODE_REVISION.json"),
    ]
    for c in candidates:
        if c.exists():
            try:
                _CACHED_DATA = json.loads(c.read_text(encoding="utf-8"))
                return _CACHED_DATA
            except Exception:
                pass
    return {}


def get_pinned_revision(group: str | None = None) -> str:
    """Returns the canonical pinned git commit SHA from reports/CODE_REVISION.json.

    Supports per-group mapping {experiment_group: sha} and tag lookup.
    """
    if group and len(group) == 40 and all(c in "0123456789abcdefABCDEF" for c in group):
        return group.strip()

    data = _load_revision_data()
    groups = data.get("groups", {})
    if group and group in groups:
        return str(groups[group]).strip()

    if group and group in data.get("tags", {}):
        return str(data["tags"][group]).strip()

    # Fall back to default group or global code_revision
    rev = groups.get("default") or data.get("code_revision")
    if rev:
        return str(rev).strip()

    return "6793d730aa734433bc3c7bc9ff4f89e0b5dc40d1"


def get_revision_map() -> dict[str, str]:
    """Returns the complete experiment_group -> SHA mapping."""
    data = _load_revision_data()
    return dict(data.get("groups", {}))
