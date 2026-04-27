"""Pytest fixtures for CFD integration tests.

Builds synthetic CFD sample CSVs and summary JSONs that match the
expected output format documented in src/cfd_integration/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_samples() -> pd.DataFrame:
    """Synthetic CFD sample grid at z=1.5m over a 250m circular domain.

    Centred on (0, 0). Uses a simple log-profile-ish wind distribution:
    velocity magnitude increases with distance from the centre
    (mimicking reduced blockage near the domain boundary).
    """
    rng = np.random.default_rng(42)
    # 2m grid spacing → 125 per axis → ~15k points in the circle
    xs = np.arange(-250, 251, 2)
    ys = np.arange(-250, 251, 2)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    # Keep only points within the 250m circle
    mask = (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= 250**2
    pts = pts[mask]

    # U_mag: ramp from 0.3 m/s at center to 3.5 m/s at boundary + noise
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    u_mag = 0.3 + 3.2 * (r / 250.0) + rng.normal(0, 0.15, len(pts))
    u_mag = np.clip(u_mag, 0.05, None)
    # Direction: predominantly east (wind from west)
    u = 0.9 * u_mag + rng.normal(0, 0.05, len(pts))
    v = 0.1 * u_mag + rng.normal(0, 0.05, len(pts))
    w = rng.normal(0, 0.02, len(pts))
    tke = 0.5 * (u_mag * 0.15) ** 2 + rng.uniform(0, 0.01, len(pts))

    return pd.DataFrame(
        {
            "x": pts[:, 0],
            "y": pts[:, 1],
            "z": np.full(len(pts), 1.5),
            "U": u,
            "V": v,
            "W": w,
            "U_mag": u_mag,
            "TKE": tke,
        }
    )


@pytest.fixture
def synthetic_metadata() -> dict:
    return {
        "patch_id": "TST-P01",
        "site": "test_site",
        "wind_direction": "E",
        "wind_speed_ref": 5.0,
        "converged": True,
        "residual_final": 1.2e-5,
        "solver": "simpleFoam",
        "turbulence_model": "kOmegaSST",
        "n_iterations": 2000,
        "wall_clock_s": 3600.0,
    }


@pytest.fixture
def synthetic_patch_csv(tmp_path, synthetic_samples, synthetic_metadata):
    """Write a synthetic sample_points.csv + summary.json to tmp_path."""
    csv_path = tmp_path / "sample_points.csv"
    json_path = tmp_path / "summary.json"
    synthetic_samples.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(synthetic_metadata, f)
    return csv_path, json_path


@pytest.fixture
def synthetic_wind_rose():
    """Simple wind rose: SE-dominated."""
    return {
        "site": "test_site",
        "source": "synthetic",
        "frequencies": {
            "N": 0.05,
            "NE": 0.10,
            "E": 0.15,
            "SE": 0.30,
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
    }
