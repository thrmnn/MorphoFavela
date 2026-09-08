"""Guards for the free-EO satellite reconstruction track.

Two failure modes, both already paid for once:
  1. the reconstruction reading the IPP answer key it is supposed to be scored
     against (leakage), and
  2. mosaicking that silently moves elevations (rasterio.merge with bounds
     shifted GLO-30 by up to 27 m against a direct reprojection).
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

REPO = Path(__file__).resolve().parents[1]
RECON_SRC = REPO / "scripts" / "satellite" / "build_reconstruction.py"
SCORECARD = REPO / "outputs" / "comparative" / "satellite" / "rocinha" / "scorecard.json"

# Tokens that would mean the reconstruction is looking at the answer key.
IPP_TOKENS = ("DTM_RJ", "buildings_RJ_2019", "Favelas_Limit", 'data" / "RJ', "data/RJ/")


def _load_module():
    spec = importlib.util.spec_from_file_location("build_reconstruction", RECON_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reconstruction_never_references_the_ipp_ground_truth():
    """The reconstruction may know where Rocinha is, never what it looks like."""
    tree = ast.parse(RECON_SRC.read_text())
    body = tree.body[1:] if ast.get_docstring(tree) else tree.body
    code = "\n".join(ast.unparse(node) for node in body)
    for token in IPP_TOKENS:
        assert token not in code, f"reconstruction references IPP ground truth: {token!r}"


def test_paste_native_tiles_does_not_resample():
    """Two adjacent tiles on a shared lattice must paste through bit-exact."""
    mod = _load_module()
    res = 1 / 3600
    left = np.arange(12, dtype="float32").reshape(3, 4)
    right = np.arange(100, 112, dtype="float32").reshape(3, 4)
    t_left = rasterio.transform.from_origin(-44.0, -23.0, res, res)
    t_right = rasterio.transform.from_origin(-44.0 + 4 * res, -23.0, res, res)

    native, transform, out_res = mod.paste_native_tiles(
        [(left, t_left, "EPSG:4326", None), (right, t_right, "EPSG:4326", None)]
    )
    assert native.shape == (3, 8)
    assert out_res == pytest.approx(res)
    np.testing.assert_array_equal(native[:, :4], left)
    np.testing.assert_array_equal(native[:, 4:], right)
    assert transform.c == pytest.approx(-44.0)
    assert transform.f == pytest.approx(-23.0)


def test_paste_native_tiles_rejects_mismatched_lattices():
    mod = _load_module()
    a = np.zeros((2, 2), dtype="float32")
    t_a = rasterio.transform.from_origin(-44.0, -23.0, 1 / 3600, 1 / 3600)
    t_b = rasterio.transform.from_origin(-44.0, -23.0, 1 / 1200, 1 / 1200)
    with pytest.raises(SystemExit):
        mod.paste_native_tiles([(a, t_a, "EPSG:4326", None), (a, t_b, "EPSG:4326", None)])


@pytest.mark.skipif(not SCORECARD.exists(), reason="scorecard not built in this checkout")
def test_blocked_heights_are_never_reported_as_numbers():
    """A blocked component must stay null — a plausible guess would read as measured."""
    card = json.loads(SCORECARD.read_text())
    heights = card["heights"]
    assert heights["status"] == "BLOCKED"
    for key in ("per_building_mae_m", "per_building_r2", "grid100m_mean_r2"):
        assert heights[key] is None, f"{key} carries a value while heights are blocked"
