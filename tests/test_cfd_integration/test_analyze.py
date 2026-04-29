"""End-to-end tests for the result-side CFD analysis pipeline.

These run the full chain — synthetic generator → on-disk results →
``scripts.analyze_cfd_results.analyse`` — against a tiny 3-patch
fake site, verifying that all four output artefacts exist, are
non-empty, and have the expected schema.

The synthetic CFD field is deliberately small (3 patches, 600 samples
per direction) so the suite runs in well under 30 s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.analyze_cfd_results as analyse_mod  # noqa: E402
from scripts.generate_synthetic_cfd_results import generate_site  # noqa: E402


@pytest.fixture
def fake_site(tmp_path, monkeypatch):
    """Build a self-contained fake site under tmp_path with all required files.

    Mirrors the IVF layout: outputs/{site}/sampling_cfd/campaign_sampling/
    campaign_patches.csv, outputs/{site}/morphometrics/grid/grid_metrics.gpkg,
    data/{site}/wind_rose.json. The pipeline is rerooted via monkeypatch on
    PROJECT_ROOT in both the generator and orchestrator modules.
    """
    site = "fakesite"

    # 3 patches in a small bounding box, well-separated
    centers = [(0.0, 0.0), (200.0, 0.0), (0.0, 200.0)]
    rows = []
    for i, (cx, cy) in enumerate(centers):
        rows.append(
            {
                "patch_id": f"FKE-P{i + 1:02d}",
                "is_pilot": True,
                "center_x": cx,
                "center_y": cy,
                "stratum_id": "SVF1_SLP1_LP1",
                "svf": 0.1 + 0.3 * i,
                "lambda_p": 0.6 - 0.2 * i,
                "slope_deg": 5.0 + 5.0 * i,
                "porosity": 0.3,
                "sigma_h": 1.5,
                "H_mean": 6.0,
                "H_max_analysis": 12.0,
                "blocken_radius_required": 60.0,
            }
        )
    patches_df = pd.DataFrame(rows)

    sampling_dir = tmp_path / "outputs" / site / "sampling_cfd" / "campaign_sampling"
    sampling_dir.mkdir(parents=True)
    patches_df.to_csv(sampling_dir / "campaign_patches.csv", index=False)

    grid_dir = tmp_path / "outputs" / site / "morphometrics" / "grid"
    grid_dir.mkdir(parents=True)
    grid_rows = []
    zone_id = 0
    for cx, cy in centers:
        for ddy in range(-40, 41, 10):
            for ddx in range(-40, 41, 10):
                gcx = cx + ddx + 5
                gcy = cy + ddy + 5
                grid_rows.append(
                    {
                        "zone_id": zone_id,
                        "centroid_x": gcx,
                        "centroid_y": gcy,
                        "H_mean": 6.5,
                        "geometry": box(gcx - 5, gcy - 5, gcx + 5, gcy + 5),
                    }
                )
                zone_id += 1
    grid_gdf = gpd.GeoDataFrame(grid_rows, crs="EPSG:31983")
    grid_gdf.to_file(grid_dir / "grid_metrics.gpkg", driver="GPKG")

    data_dir = tmp_path / "data" / site
    data_dir.mkdir(parents=True)
    wind_rose = {
        "site": site,
        "source": "synthetic",
        "frequencies": {
            "N": 0.05,
            "NE": 0.10,
            "E": 0.20,
            "SE": 0.25,
            "S": 0.15,
            "SW": 0.10,
            "W": 0.10,
            "NW": 0.05,
        },
        "mean_speeds": {
            "N": 2.0,
            "NE": 3.0,
            "E": 3.5,
            "SE": 4.0,
            "S": 3.0,
            "SW": 2.5,
            "W": 2.0,
            "NW": 1.8,
        },
        "reference_height_m": 10.0,
        "quality_flag": "synthetic",
    }
    with open(data_dir / "wind_rose.json", "w") as f:
        json.dump(wind_rose, f)

    monkeypatch.setattr("scripts.generate_synthetic_cfd_results.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(analyse_mod, "PROJECT_ROOT", tmp_path)
    return site, tmp_path


def test_synthetic_generator_writes_full_set(fake_site):
    site, root = fake_site
    out_root = root / "data" / site / "cfd_results"
    coverage = generate_site(
        site=site,
        out_root=out_root,
        layout="csv",
        n_samples_per_direction=300,
    )
    assert coverage["n_patches"] == 3
    for patch_id in ("FKE-P01", "FKE-P02", "FKE-P03"):
        for direction in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
            d = out_root / patch_id / direction
            assert (d / "sample_points.csv").exists(), f"missing {patch_id}/{direction}/csv"
            assert (d / "summary.json").exists(), f"missing {patch_id}/{direction}/json"


def test_synthetic_generator_parquet_layout(fake_site):
    site, root = fake_site
    out_root = root / "data" / site / "cfd_results"
    generate_site(
        site=site,
        out_root=out_root,
        layout="parquet",
        n_patches=1,
        n_samples_per_direction=200,
    )
    patch_dir = out_root / "FKE-P01"
    expected = {f"wind_{deg:03d}" for deg in (0, 45, 90, 135, 180, 225, 270, 315)}
    actual = {p.name for p in patch_dir.iterdir() if p.is_dir()}
    assert actual == expected
    parquet_files = list((patch_dir / "wind_000").glob("*.parquet"))
    assert len(parquet_files) == 1


def test_orchestrator_full_chain(fake_site):
    site, root = fake_site
    out_root = root / "data" / site / "cfd_results"
    generate_site(
        site=site,
        out_root=out_root,
        layout="csv",
        n_samples_per_direction=600,
    )
    out_dir = root / "outputs" / site / "cfd_analysis"
    coverage = analyse_mod.analyse(site=site, out_root=out_dir)

    assert coverage["site"] == site
    assert coverage["n_patches_expected"] == 3
    assert coverage["n_patches_returned"] == 3

    per_patch_path = out_dir / "per_patch_indicators.csv"
    assert per_patch_path.exists()
    per_patch = pd.read_csv(per_patch_path)
    assert len(per_patch) == 3
    for col in ("annual_U_mean", "annual_stagnation_frac", "annual_TI_mean"):
        assert col in per_patch.columns, f"missing {col}"
        assert per_patch[col].notna().all(), f"{col} has NaN"

    regression_path = out_dir / "predictor_regression.csv"
    assert regression_path.exists()
    regression = pd.read_csv(regression_path)
    assert len(regression) >= 1
    assert {"indicator", "n", "r2"}.issubset(regression.columns)

    grid_path = out_dir / "grid_with_cfd.gpkg"
    assert grid_path.exists()
    grid_with_cfd = gpd.read_file(grid_path)
    assert len(grid_with_cfd) > 0
    assert "annual_cfd_U_mean" in grid_with_cfd.columns

    fig_path = out_dir / "figures" / "fig5_wind_panel.png"
    assert fig_path.exists()
    assert fig_path.stat().st_size > 1000

    coverage_path = out_dir / "coverage.json"
    assert coverage_path.exists()
    with open(coverage_path) as f:
        coverage_loaded = json.load(f)
    assert coverage_loaded["n_patches_returned"] == 3


def test_orchestrator_handles_missing_direction(fake_site):
    """If a direction folder is missing for one patch, the rest still run."""
    site, root = fake_site
    out_root = root / "data" / site / "cfd_results"
    generate_site(site=site, out_root=out_root, layout="csv", n_samples_per_direction=400)

    import shutil

    shutil.rmtree(out_root / "FKE-P02" / "NW")

    out_dir = root / "outputs" / site / "cfd_analysis"
    coverage = analyse_mod.analyse(site=site, out_root=out_dir)
    assert coverage["n_patches_returned"] == 3
    assert "FKE-P02" in coverage["per_patch_directions"]
    assert "NW" not in coverage["per_patch_directions"]["FKE-P02"]
    assert len(coverage["per_patch_directions"]["FKE-P02"]) == 7


def test_orchestrator_handles_missing_patch(fake_site):
    """If a patch folder is missing entirely, it shows up in missing_patches."""
    site, root = fake_site
    out_root = root / "data" / site / "cfd_results"
    generate_site(
        site=site, out_root=out_root, layout="csv", n_patches=2, n_samples_per_direction=400
    )

    out_dir = root / "outputs" / site / "cfd_analysis"
    coverage = analyse_mod.analyse(site=site, out_root=out_dir)
    assert coverage["n_patches_returned"] == 2
    assert "FKE-P03" in coverage["missing_patches"]


def test_orchestrator_autodetects_parquet_layout(fake_site):
    """Airflow-native parquet results should pass through transparently."""
    site, root = fake_site
    out_root = root / "data" / site / "cfd_results"
    generate_site(
        site=site,
        out_root=out_root,
        layout="parquet",
        n_patches=2,
        n_samples_per_direction=400,
    )

    out_dir = root / "outputs" / site / "cfd_analysis"
    coverage = analyse_mod.analyse(site=site, out_root=out_dir)
    assert coverage["n_patches_returned"] == 2
    per_patch = pd.read_csv(out_dir / "per_patch_indicators.csv")
    assert len(per_patch) == 2
    assert per_patch["annual_U_mean"].notna().all()
