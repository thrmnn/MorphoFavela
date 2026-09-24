"""WP-06 geometry-only ventilation table: spec docs/wp06_geometry_spec.md,
deliverable 4's five tests. Synthetic inputs throughout — no dependency on
data/outputs/runs (gitignored, main-checkout-only), so these run anywhere.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[1]

from scripts.run_ventilation_index import count_constraints as reference_count_constraints
from scripts.run_wind_exposure import SECTORS
from src.brisa_solar import wp06_geometry as w


def _synthetic_grid(n: int = 6, cell: float = 10.0) -> gpd.GeoDataFrame:
    """n built 10 m cells in a row, with every column compute_site_table reads."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        cx, cy = i * cell, 0.0
        lf = {f"lambda_f_{s}": float(rng.uniform(0.0, 1.0)) for s in SECTORS}
        rows.append({
            "zone_id": i,
            "centroid_x": cx, "centroid_y": cy,
            "lambda_p": float(rng.uniform(0.0, 0.6)),
            "slope_deg": float(rng.uniform(0.0, 40.0)),
            "porosity": float(rng.uniform(0.0, 1.0)),
            "sigma_h": float(rng.uniform(0.5, 3.0)),
            "aspect_deg": float(rng.uniform(0.0, 360.0)),
            "svf": float(rng.uniform(0.3, 0.9)),
            "H_mean": float(rng.uniform(2.0, 8.0)),
            "building_count": 1,
            **lf,
            "lambda_f_mean": float(np.mean(list(lf.values()))),
            "lambda_f_max": float(np.max(list(lf.values()))),
            "geometry": box(cx - cell / 2, cy - cell / 2, cx + cell / 2, cy + cell / 2),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:31983")


def _synthetic_ground(grid: gpd.GeoDataFrame, pts_per_cell: int = 20) -> pd.DataFrame:
    """1 m ground points scattered inside each grid cell's footprint."""
    rng = np.random.default_rng(1)
    xs, ys, svf, kwh, hrs, ge2h = [], [], [], [], [], []
    for _, row in grid.iterrows():
        cx, cy = row["centroid_x"], row["centroid_y"]
        for _ in range(pts_per_cell):
            xs.append(cx + rng.uniform(-4.9, 4.9))
            ys.append(cy + rng.uniform(-4.9, 4.9))
            svf.append(rng.uniform(0.2, 0.9))
            kwh.append(rng.uniform(500, 1500))
            hrs.append(rng.uniform(0, 8))
            ge2h.append(bool(rng.uniform(0, 8) >= 2.0))
    return pd.DataFrame({
        "x": xs, "y": ys, "svf": svf, "kwh_m2": kwh,
        "hours_winter_solstice": hrs, "ge_2h_winter_solstice": ge2h,
    })


def _synthetic_freq() -> dict:
    rng = np.random.default_rng(2)
    raw = {s: rng.uniform(0.5, 2.0) for s in SECTORS}
    tot = sum(raw.values())
    return {s: raw[s] / tot for s in raw}


# --- (a) written CSV carries only allowlisted columns -----------------------

def test_output_columns_are_allowlisted(tmp_path):
    policy = json.loads((ROOT / "docs" / "p1_column_allowlist.json").read_text())
    legal = set(policy["p1_legal"])

    grid = _synthetic_grid()
    ground = _synthetic_ground(grid)
    freq = _synthetic_freq()
    table = w.compute_site_table(grid, ground, freq, depth_median=15.0)

    assert set(table.columns) <= legal, set(table.columns) - legal

    out_csv = tmp_path / "per_patch_geometry.csv"
    table.to_csv(out_csv, index=False)
    header = out_csv.read_text().split("\n", 1)[0].split(",")
    assert set(header) <= legal, set(header) - legal


# --- (b) count_constraints parity against scripts.run_ventilation_index -----

def test_count_constraints_parity_with_ventilation_index():
    grid = _synthetic_grid(n=10)
    ground = _synthetic_ground(grid)
    freq = _synthetic_freq()
    depth_median = 12.5
    table = w.compute_site_table(grid, ground, freq, depth_median)

    expected = reference_count_constraints(
        grid["lambda_f_mean"].to_numpy(),
        table["open_edge_dist_m"].to_numpy(),
        table["exposure_ratio"].to_numpy(),
        depth_median,
    )
    np.testing.assert_array_equal(table["n_constraints"].to_numpy(), expected)
    np.testing.assert_array_equal(
        table["constraint_vertical"] + table["constraint_lateral"] + table["constraint_directional"],
        table["n_constraints"],
    )


# --- (c) wind exposure equals Sigma freq*lambda_f on a synthetic cell -------

def test_wind_exposure_equals_frequency_weighted_sum():
    grid = _synthetic_grid(n=1)
    ground = _synthetic_ground(grid)
    freq = _synthetic_freq()
    table = w.compute_site_table(grid, ground, freq, depth_median=10.0)

    expected = sum(freq[s] * grid.iloc[0][f"lambda_f_{s}"] for s in SECTORS)
    assert table["wind_exposure"].iloc[0] == pytest.approx(expected, rel=1e-9)


# --- (d) the token lint fails on a planted token, passes on a clean tree ----

def test_token_lint_fails_on_planted_token_passes_clean(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import lint_p1_tokens as lt
    importlib.reload(lt)

    fake_root = tmp_path
    (fake_root / "src" / "brisa_solar").mkdir(parents=True)
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / "docs").mkdir(parents=True)
    (fake_root / "outputs" / "testsite" / "geometry_indicators").mkdir(parents=True)

    clean_py = "def f():\n    return 1  # nothing banned here\n"
    (fake_root / "src" / "brisa_solar" / "dummy.py").write_text(clean_py)
    (fake_root / "scripts" / "lint_p1_probe.py").write_text(clean_py)
    (fake_root / "docs" / "p1_column_allowlist.json").write_text('{"p1_legal": ["patch_id"]}\n')
    (fake_root / "outputs" / "testsite" / "geometry_indicators" / "per_patch_geometry.csv").write_text(
        "patch_id,svf\n1,0.5\n"
    )

    monkeypatch.setattr(lt, "ROOT", fake_root)
    assert lt.main() == 0, "clean synthetic tree must pass"

    (fake_root / "src" / "brisa_solar" / "dummy.py").write_text(
        clean_py + "# this describes a skimming flow regime\n"
    )
    assert lt.main() == 1, "a planted banned token must fail the lint"

    (fake_root / "src" / "brisa_solar" / "dummy.py").write_text(clean_py)
    assert lt.main() == 0, "removing the planted token must restore a clean pass"


# --- WP04MARE: --out-name/--depth-median default to the pre-WP04MARE behaviour

def test_depth_median_is_read_from_a_recorded_run_not_typed(tmp_path):
    run = tmp_path / "runs" / "wp06_geometry_X"
    run.mkdir(parents=True)
    (run / "summary.json").write_text('{"depth_median_m": 12.5}')
    assert w.resolve_depth_median(tmp_path, ["vidigal"], "wp06_geometry_X") == 12.5
    src = (ROOT / "src" / "brisa_solar" / "wp06_geometry.py").read_text()
    assert '"--out-name", default="per_patch_geometry.csv"' in src
    assert "--depth-median\"" not in src


# --- (e) the module imports no CFD code --------------------------------------

def test_module_imports_no_cfd_code():
    probe = (
        "import sys\n"
        "from src.brisa_solar import wp06_geometry\n"
        "leaked = [m for m in sys.modules "
        "if 'cfd_integration' in m or 'analyze_cfd_results' in m]\n"
        "assert not leaked, leaked\n"
        "print('CLEAN')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CLEAN" in result.stdout
