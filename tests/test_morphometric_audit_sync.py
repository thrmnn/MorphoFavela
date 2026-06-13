"""Regression test for the morphometrics output-sync helper.

The original `if not dest.exists()` guard froze the canonical
morphometrics/svf/ tree, so re-running run_svf_v2 never propagated to
consumers — the cause of the 2026 street-SVF drift. `_sync_outputs`
must OVERWRITE, not skip.
"""

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_morphometric_audit", PROJECT_ROOT / "scripts" / "run_morphometric_audit.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_sync_outputs = _mod._sync_outputs


def test_overwrites_existing_destination(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (dst / "svf_streets.gpkg").write_text("STALE")
    (src / "svf_streets.gpkg").write_text("FRESH")
    (src / "scene.stl").write_text("NEW")

    n_new, n_replaced = _sync_outputs(src, dst, "test")

    assert (dst / "svf_streets.gpkg").read_text() == "FRESH", "stale dest was not overwritten"
    assert (dst / "scene.stl").read_text() == "NEW"
    assert n_replaced == 1
    assert n_new == 1


def test_skips_subdirectories(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "nested").mkdir()
    (src / "a.gpkg").write_text("x")

    n_new, n_replaced = _sync_outputs(src, dst, "test")

    assert n_new == 1 and n_replaced == 0
    assert not (dst / "nested").exists()
