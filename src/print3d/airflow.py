"""Pull CFD pedestrian-wind results into a ground ventilation field.

The RDP-P20 return is a steady RANS set (simpleFoam / kΩSST), 8 wind directions,
~15 k pedestrian-level (z = 1.5 m) sample points each, carrying |U| and TKE — but
NO local mean age of air (LMA needs a passive-scalar transient run, which this
steady set does not contain). So the airflow texture encodes the available and
strongly LMA-correlated proxy: **wind-rose-weighted mean |U|/U_ref** at pedestrian
level (low speed = stagnant = old air).

The 8 directions do not share sample coordinates, so each is interpolated
(nearest) onto the tile grid, then weighted by the site wind rose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.print3d.model import ROOT

_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_AZIMUTH = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}


def cfd_dir(site: str, patch: str) -> Path:
    return ROOT / "data" / site / "cfd_results" / patch


def has_cfd(site: str, patch: str) -> bool:
    d = cfd_dir(site, patch)
    return d.exists() and any((d / c / "sample_points.csv").exists() for c in _DIRS)


def dominant_wind_azimuth(site: str) -> float:
    wr = json.loads((ROOT / "data" / site / "wind_rose.json").read_text())["frequencies"]
    return float(_AZIMUTH[max(wr, key=wr.get)])


def ventilation_grid(site: str, patch: str, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-cell wind-rose-weighted mean |U|/U_ref (dimensionless). Higher = better
    ventilated; lower = stagnant."""
    wr = json.loads((ROOT / "data" / site / "wind_rose.json").read_text())["frequencies"]
    base = cfd_dir(site, patch)
    cell = np.column_stack([X.ravel(), Y.ravel()])
    acc = np.zeros(len(cell))
    wsum = 0.0
    for d in _DIRS:
        pts = base / d / "sample_points.csv"
        if not pts.exists():
            continue
        df = pd.read_csv(pts)
        u_ref = json.loads((base / d / "summary.json").read_text())["wind_speed_ref"]
        _, nn = cKDTree(df[["x", "y"]].values).query(cell)
        f = wr.get(d, 0.0)
        acc += f * (df["U_mag"].values / u_ref)[nn]
        wsum += f
    return (acc / wsum).reshape(X.shape)
