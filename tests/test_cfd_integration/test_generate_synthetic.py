"""Tests for the rectangular_domain_v1 sampler + synthetic provenance.

Data-independent: the rectangular sampler and provenance helper are
exercised directly; the end-to-end `generate_site` test builds a tiny
in-memory campaign_patches.csv (with the v1 domain_*_m columns) and
monkeypatches PROJECT_ROOT, matching the pattern in test_analyze.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_synthetic_cfd_results import (  # noqa: E402
    _generate_samples_rect,
    _provenance,
    generate_site,
)

_REQUIRED_COLS = {"x", "y", "z", "U", "V", "W", "U_mag", "TKE"}


def _to_local(df, cx, cy, direction_deg):
    """Project world (x, y) back into the flow-aligned (s, t) frame."""
    b = np.deg2rad(direction_deg)
    fx, fy = -np.sin(b), -np.cos(b)
    px, py = -fy, fx
    dx = df["x"].to_numpy() - cx
    dy = df["y"].to_numpy() - cy
    s = dx * fx + dy * fy
    t = dx * px + dy * py
    return s, t


def test_rect_sampler_bounds_and_schema():
    cx, cy = 680000.0, 7456000.0
    up, down, lat, sp = 80.0, 160.0, 200.0, 5.0
    rng = np.random.default_rng(0)
    df = _generate_samples_rect(cx, cy, up, down, lat, 1.4, "NE", rng, sp)

    assert _REQUIRED_COLS.issubset(df.columns)
    assert np.allclose(df["z"], 1.5)

    s, t = _to_local(df, cx, cy, 45.0)  # NE = 45°
    # within the rectangle (one spacing of float tolerance)
    assert s.min() >= -up - sp and s.max() <= down + sp
    assert t.min() >= -lat - sp and t.max() <= lat + sp
    # the grid actually reaches both ends of the along-flow axis
    assert s.min() == pytest.approx(-up, abs=sp)
    assert s.max() == pytest.approx(down, abs=sp)


def test_rect_sampler_u_mag_self_consistent():
    rng = np.random.default_rng(1)
    df = _generate_samples_rect(0.0, 0.0, 50.0, 100.0, 120.0, 2.0, "S", rng, 4.0)
    recomputed = np.sqrt(df["U"] ** 2 + df["V"] ** 2 + df["W"] ** 2)
    assert np.allclose(df["U_mag"], recomputed, atol=1e-9)
    assert (df["U_mag"] > 0).all()


def test_rect_sampler_orientation_downstream():
    """Wind from N → air travels south → +s (downstream) is at y < cy."""
    rng = np.random.default_rng(2)
    df = _generate_samples_rect(0.0, 0.0, 60.0, 180.0, 100.0, 6.0, "N", rng, 6.0)
    s, _ = _to_local(df, 0.0, 0.0, 0.0)
    far_down = df.loc[s > 150.0]
    assert len(far_down) > 0
    assert (far_down["y"] < 0).all()


def test_rect_sampler_grid_spacing():
    rng = np.random.default_rng(3)
    sp = 10.0
    df = _generate_samples_rect(0.0, 0.0, 100.0, 100.0, 100.0, 3.0, "E", rng, sp)
    expected = len(np.arange(-100.0, 100.0 + sp, sp)) ** 2
    assert len(df) == expected


def test_provenance_block():
    p = _provenance(seed=42, grid_spacing=2.0, u_ref=6.0)
    assert p["synthetic"] is True
    assert p["sampling"] == "rectangular_domain_v1"
    assert p["grid_spacing_m"] == 2.0
    for key in ("generator", "git_commit", "seed", "model", "generated_utc", "warning"):
        assert key in p
    assert _provenance(42, None, 6.0)["sampling"] == "circular_250m"


@pytest.fixture
def fake_site(tmp_path, monkeypatch):
    rows = [
        {
            "patch_id": "FKE-P01",
            "is_pilot": True,
            "center_x": 0.0,
            "center_y": 0.0,
            "svf": 0.1,
            "lambda_p": 0.9,
            "slope_deg": 25.0,
            "H_mean": 8.0,
            "domain_upstream_m": 60.0,
            "domain_downstream_m": 120.0,
            "domain_lateral_m": 150.0,
        },
        {
            "patch_id": "FKE-P02",
            "is_pilot": True,
            "center_x": 500.0,
            "center_y": 0.0,
            "svf": 0.5,
            "lambda_p": 0.3,
            "slope_deg": 8.0,
            "H_mean": 5.0,
            "domain_upstream_m": 60.0,
            "domain_downstream_m": 120.0,
            "domain_lateral_m": 150.0,
        },
    ]
    sampling = tmp_path / "outputs" / "fakesite" / "sampling_cfd" / "campaign_sampling"
    sampling.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(sampling / "campaign_patches.csv", index=False)
    monkeypatch.setattr(
        "scripts.generate_synthetic_cfd_results.PROJECT_ROOT", tmp_path
    )
    return tmp_path


def test_generate_site_patch_filter_and_synthetic_flag(fake_site):
    out_root = fake_site / "data" / "fakesite" / "cfd_results_synthetic"
    coverage = generate_site(
        site="fakesite",
        out_root=out_root,
        layout="csv",
        grid_spacing=20.0,
        patch_id="FKE-P02",
    )
    assert coverage["n_patches"] == 1
    assert set(coverage["patches"]) == {"FKE-P02"}
    assert not (out_root / "FKE-P01").exists()

    summary = json.loads((out_root / "FKE-P02" / "N" / "summary.json").read_text())
    assert summary["synthetic"] is True
    assert summary["provenance"]["sampling"] == "rectangular_domain_v1"

    df = pd.read_csv(out_root / "FKE-P02" / "N" / "sample_points.csv")
    assert _REQUIRED_COLS.issubset(df.columns)
    assert len(df) > 0


def test_building_aware_field_is_solid_inside_and_directional():
    """Footprints kill the flow; the lee shelter flips with the wind."""
    import geopandas as gpd
    from shapely.geometry import box

    bld = gpd.GeoDataFrame(
        {"height": [12.0]}, geometry=[box(-10, -10, 10, 10)], crs="EPSG:31983"
    )

    def run(direction):
        rng = np.random.default_rng(0)
        return _generate_samples_rect(
            0.0, 0.0, 60.0, 120.0, 80.0, 2.0, direction, rng, 4.0, buildings=bld
        )

    dn = run("N")
    ds = run("S")

    # inside the footprint → essentially no pedestrian flow
    inside = dn[(dn.x.abs() < 8) & (dn.y.abs() < 8)]
    assert len(inside) > 0
    assert inside.U_mag.mean() < 0.5

    # immediate lee vs immediate windward of the block (same direction):
    # u_bg is symmetric in ±s, so a clear deficit is the wake itself.
    # wind-from-N → air travels to world -y; downwind band is y∈[-40,-15].
    def band(d, lo, hi):
        m = (d.y > lo) & (d.y < hi) & (d.x.abs() < 8)
        return d.loc[m, "U_mag"].mean()

    lee_n = band(dn, -40, -15)      # behind block for N
    windward_n = band(dn, 15, 40)   # in front of block for N
    assert lee_n < 0.75 * windward_n

    # same physical band, opposite wind: lee for N is open/windward for S
    lee_s_region_under_s = band(ds, -40, -15)
    assert lee_n < 0.8 * lee_s_region_under_s

    # U_mag stays self-consistent with the components
    rec = np.sqrt(dn.U**2 + dn.V**2 + dn.W**2)
    assert np.allclose(dn.U_mag, rec, atol=1e-9)

    assert _provenance(42, 2.0, 6.0, building_aware=True)["building_aware"] is True
    assert "building_aware" in _provenance(42, 2.0, 6.0, building_aware=True)["model"]


def test_generate_site_legacy_path_still_works(fake_site):
    """grid_spacing=None must preserve the circular-cloud API the
    analyze-pipeline tests depend on."""
    out_root = fake_site / "data" / "fakesite" / "cfd_results"
    generate_site(
        site="fakesite",
        out_root=out_root,
        layout="csv",
        n_samples_per_direction=120,
    )
    df = pd.read_csv(out_root / "FKE-P01" / "N" / "sample_points.csv")
    assert len(df) == 120
    summary = json.loads((out_root / "FKE-P01" / "N" / "summary.json").read_text())
    assert summary["synthetic"] is True  # provenance applies to both modes
