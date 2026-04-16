"""Wind-rose weighting: combine 8-direction CFD results into annualised metrics.

For each patch, results from 8 wind directions are combined using the local
wind frequency distribution (from INMET or similar station data) weighted
by observed speed per direction.

The speed weighting is important: a low-frequency direction with high speeds
may contribute more to ventilation than a high-frequency calm direction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.cfd_integration.schema import WIND_DIRECTIONS_8, WindRose

logger = logging.getLogger(__name__)


def load_wind_rose(path: Path, site: str) -> WindRose:
    """Load a wind rose from JSON.

    Expected schema (all fields except site + frequencies are optional):
        {
            "site": "vidigal",
            "source": "INMET Alto da Boa Vista 2015-2024",
            "frequencies": {"N": 0.04, "NE": 0.12, ...},
            "mean_speeds":  {"N": 2.1,  "NE": 3.4, ...},
            "reference_height_m": 10.0,
            "station_id": "A652",
            "station_name": "Alto da Boa Vista",
            "station_coords": [-22.9762, -43.2784],
            "time_window_start": "2015-01-01",
            "time_window_end": "2024-12-31",
            "n_observations": 84321,
            "calm_fraction": 0.08,
            "quality_flag": "measured"
        }
    """
    with open(path) as f:
        data = json.load(f)
    coords = data.get("station_coords")
    if coords is not None:
        coords = (float(coords[0]), float(coords[1]))
    return WindRose(
        site=data.get("site", site),
        frequencies=data["frequencies"],
        mean_speeds=data.get("mean_speeds", {}),
        source=data.get("source", ""),
        reference_height_m=data.get("reference_height_m"),
        station_id=data.get("station_id"),
        station_name=data.get("station_name"),
        station_coords=coords,
        time_window_start=data.get("time_window_start"),
        time_window_end=data.get("time_window_end"),
        n_observations=data.get("n_observations"),
        calm_fraction=data.get("calm_fraction"),
        quality_flag=data.get("quality_flag"),
    )


def weighted_by_wind_rose(
    per_direction: dict[str, dict],
    wind_rose: Optional[WindRose] = None,
    weight_by: str = "frequency",
) -> dict:
    """Combine per-direction metrics into a single weighted metric dict.

    Parameters
    ----------
    per_direction : dict
        Mapping {direction: metrics_dict} with metrics like U_mean, stagnation_frac.
        Missing directions are dropped from the weighting.
    wind_rose : WindRose, optional
        If None, uses uniform weights (1/n).
    weight_by : str
        "frequency" — weight by f_dir
        "freq_speed" — weight by f_dir × U_dir (accounts for wind energy contribution)
        "uniform" — equal weighting

    Returns
    -------
    dict with weighted scalar metrics (prefix 'annual_' for wind-rose-weighted).
    """
    directions = [d for d in per_direction if d in WIND_DIRECTIONS_8]
    if not directions:
        return {}

    # Build weight vector
    if wind_rose is None or weight_by == "uniform":
        weights = np.ones(len(directions)) / len(directions)
    else:
        f = np.array([wind_rose.frequencies.get(d, 0.0) for d in directions])
        if weight_by == "freq_speed":
            u = np.array([wind_rose.mean_speeds.get(d, 1.0) for d in directions])
            raw = f * u
        else:  # "frequency"
            raw = f
        if raw.sum() > 0:
            weights = raw / raw.sum()
        else:
            weights = np.ones(len(directions)) / len(directions)

    # Collect metric columns
    all_keys = set()
    for d in directions:
        all_keys.update(per_direction[d].keys())
    # Skip non-numeric / id keys
    skip = {"patch_id", "wind_direction", "n_samples"}
    numeric_keys = sorted(k for k in all_keys if k not in skip)

    out = {"n_directions": len(directions), "weight_method": weight_by}
    for k in numeric_keys:
        vals = []
        wts = []
        for d, w in zip(directions, weights):
            v = per_direction[d].get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(v)
                wts.append(w)
        if vals:
            vals = np.array(vals)
            wts = np.array(wts)
            if wts.sum() > 0:
                out[f"annual_{k}"] = float(np.average(vals, weights=wts))
            else:
                out[f"annual_{k}"] = float(np.mean(vals))
        else:
            out[f"annual_{k}"] = np.nan

    return out


def worst_case_direction(
    per_direction: dict[str, dict], metric: str = "stagnation_frac"
) -> tuple[str, float]:
    """Find the direction with the worst value for a given metric.

    For metrics where higher = worse (stagnation_frac, TI):
        returns (direction, max_value)
    For metrics where lower = worse (U_mean, ACH):
        use negate=True via external logic or wrap in a helper.
    """
    valid = {d: m for d, m in per_direction.items() if metric in m and not np.isnan(m[metric])}
    if not valid:
        return ("", np.nan)
    worst = max(valid.items(), key=lambda kv: kv[1][metric])
    return (worst[0], worst[1][metric])
