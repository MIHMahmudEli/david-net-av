"""Cell 4 must not download datasets the notebook did not ask for.

The first run of build_cache.ipynb failed after 283 s with "No space left on device":
told to attach FakeAVCeleb alone, Cell 4 began auto-downloading the six datasets that
were not mounted, one of them 96.5 GB, into a 20 GB disk. Nothing was decoded.

These tests run the real cell body out of the committed notebooks, with the two Kaggle
paths redirected into tmp_path and `subprocess.run` recording instead of downloading, so
a future edit that re-widens the allowlist fails here rather than on Kaggle.
"""
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRAIN_NB = REPO / "kaggle_kernel" / "train_kaggle.ipynb"
BUILD_NB = REPO / "kaggle_kernel" / "build_cache.ipynb"


@pytest.fixture(autouse=True)
def _no_real_downloads(monkeypatch):
    """Nothing in this module may shell out, whatever the cell body does.

    The first version of these tests injected a stub into the exec namespace, which the
    cell's own `import subprocess` promptly shadowed. It ran the real
    `kaggle datasets download`, sat on the cell's timeout=3600 for a full hour, and left
    11 GB of a partial dfdc-10.zip in pytest's tmp dir. `_run_cell4` patches
    `subprocess.run` properly now; this is the belt to that pair of braces.
    """
    def _refuse(cmd, *a, **kw):
        raise AssertionError(f"test tried to run a real subprocess: {cmd}")
    monkeypatch.setattr(subprocess, "run", _refuse)


def _cells(nb_path):
    return json.loads(nb_path.read_text(encoding="utf-8"))["cells"]


def _cell4(nb_path):
    for c in _cells(nb_path):
        s = "".join(c["source"])
        if s.startswith("# Cell 4: Discover"):
            return s
    raise AssertionError(f"no Cell 4 in {nb_path.name}")


def _allowlist_preset(nb_path):
    """What the notebook sets for DOWNLOAD_ALLOWLIST before Cell 4, if anything."""
    ns = {}
    for c in _cells(nb_path):
        s = "".join(c["source"])
        if s.startswith("# Cell 4: Discover"):
            break
        if "DOWNLOAD_ALLOWLIST = {" in s:
            exec(compile(s, "<allowlist>", "exec"), ns)
    return ns.get("DOWNLOAD_ALLOWLIST")


def _run_cell4(nb_path, tmp_path, monkeypatch, mounted=()):
    """Execute Cell 4 against a fake /kaggle tree. Returns (datasets, downloaded slugs)."""
    kaggle_input = tmp_path / "input"
    working = tmp_path / "working"
    (kaggle_input / "datasets").mkdir(parents=True)
    working.mkdir()
    for slug in mounted:
        d = kaggle_input / "datasets" / slug
        d.mkdir(parents=True)
        (d / "clip.mp4").write_bytes(b"\x00")

    src = _cell4(nb_path)
    src = src.replace('Path("/kaggle/input")', f"Path(r'{kaggle_input}')")
    src = src.replace('Path("/kaggle/working")', f"Path(r'{working}')")
    assert str(kaggle_input) in src and str(working) in src, "path redirect failed"

    attempted = []

    class _Result:
        returncode, stdout, stderr = 1, "", "stubbed: no network in tests"

    def fake_run(cmd, *a, **kw):
        attempted.append(cmd)
        return _Result()

    # The cell runs `import subprocess` itself, so a stub injected into the namespace is
    # immediately shadowed -- patch the module the cell will import. Without this the
    # test really does shell out to `kaggle datasets download` and hangs for an hour.
    monkeypatch.setattr(subprocess, "run", fake_run)
    ns = {"print": lambda *a, **k: None}
    preset = _allowlist_preset(nb_path)
    if preset is not None:
        ns["DOWNLOAD_ALLOWLIST"] = preset
    exec(compile(src, f"<{nb_path.name}:cell4>", "exec"), ns)
    slugs = [c[c.index("-d") + 1] for c in attempted if "-d" in c]
    return ns["datasets"], slugs


# ------------------------------------------------------------------ the builder
def test_build_notebook_declares_a_fakeavceleb_only_allowlist():
    assert _allowlist_preset(BUILD_NB) == {"fakeavceleb"}


def test_build_notebook_downloads_nothing_when_fakeavceleb_is_mounted(tmp_path, monkeypatch):
    datasets, slugs = _run_cell4(BUILD_NB, tmp_path, monkeypatch,
                                 mounted=["aicontentdetections/fakeavceleb-v1-2"])
    assert set(datasets) == {"fakeavceleb"}
    assert slugs == [], f"the cache builder tried to download {slugs}"


def test_build_notebook_never_reaches_for_the_96gb_corpus(tmp_path, monkeypatch):
    """dfdc-10 is the one that filled the disk. It must not even be attempted."""
    _, slugs = _run_cell4(BUILD_NB, tmp_path, monkeypatch,
                          mounted=["aicontentdetections/fakeavceleb-v1-2"])
    assert not any("dfdc" in s for s in slugs)


# ------------------------------------------------------------------ the trainer
def test_training_notebook_still_downloads_everything_by_default(tmp_path, monkeypatch):
    """The allowlist defaults to the full set, so training behaviour is unchanged."""
    assert _allowlist_preset(TRAIN_NB) is None
    datasets, slugs = _run_cell4(TRAIN_NB, tmp_path, monkeypatch,
                                 mounted=["aicontentdetections/fakeavceleb-v1-2"])
    assert set(datasets) == {"fakeavceleb"}
    assert len(slugs) == 6, f"expected the other six to be attempted, got {slugs}"
    assert any("dfdc" in s for s in slugs)


def test_mounted_datasets_are_used_and_never_downloaded(tmp_path, monkeypatch):
    datasets, slugs = _run_cell4(TRAIN_NB, tmp_path, monkeypatch, mounted=[
        "aicontentdetections/fakeavceleb-v1-2", "pranay22077/dfdc-10",
        "reubensuju/celeb-df-v2"])
    assert {"fakeavceleb", "dfdc-10", "celeb-df-v2"} <= set(datasets)
    assert not any("dfdc" in s or "celeb-df" in s for s in slugs)


# ------------------------------------------------------------------ the disk guard
def test_a_full_disk_stops_downloads_before_they_start():
    """The second guard: free space is measured, not assumed."""
    src = _cell4(BUILD_NB)
    assert "MIN_FREE_GB_TO_DOWNLOAD" in src
    assert "shutil.disk_usage" in src
    i = src.index("def download_dataset")
    j = src.index("dst.mkdir", i)
    assert "MIN_FREE_GB_TO_DOWNLOAD" in src[i:j], \
        "the space check must run BEFORE the download directory is created"
