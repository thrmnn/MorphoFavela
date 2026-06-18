"""run_meta.json provenance — the drift-detection guard.

The 2026 street-SVF drift was invisible because SVF outputs carried no
producer provenance. write_run_meta records the git sha so a consumer can
tell whether an output was produced by current code.
"""
import json

from src.svf_v2.io import write_run_meta


def test_writes_git_sha_and_fields(tmp_path):
    p = write_run_meta(tmp_path, "svf_streets", 12345, crs="EPSG:31983", boundary_clipped=True)
    assert p.name == "run_meta.json"
    meta = json.loads(p.read_text())
    assert meta["output_type"] == "svf_streets"
    assert meta["n_points"] == 12345
    assert meta["crs"] == "EPSG:31983"
    assert meta["boundary_clipped"] is True
    assert "git_sha" in meta and meta["git_sha"]      # short sha or 'unknown'
    assert meta["generated_utc"].endswith("+00:00")    # UTC ISO


def test_creates_dir(tmp_path):
    nested = tmp_path / "a" / "b"
    write_run_meta(nested, "svf_grid", 7, grid_spacing=2.0)
    assert (nested / "run_meta.json").exists()
