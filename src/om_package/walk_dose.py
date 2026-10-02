"""Direct sun dose in the 1 h and 3 h before a walk reached each point.

Per point, the integral of clear-sky direct beam on a horizontal surface
over [t_arrival - H, t_arrival] (Wh/m2), zero while the sun is below the
point's building/geometric horizon. Reuses sun_envelope's clear-sky and
horizon internals, but the window ends at each point's OWN arrival time
(direct_sun_dose uses a fixed slot grid).

CLEAR-SKY ASSUMPTION: no cloud, no tree shade, so this is an UPPER BOUND on
real direct sunlight, and a geometry-derived proxy, not a measurement.

Method: per walk, beam energy per step is computed for every point on a
1-min UTC grid from (first arrival - max H) to the last arrival, integrated
cumulatively (trapezoid), and the cumulative curve is linearly interpolated
at t_arrival and t_arrival - H.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sun_envelope import _beam_horizontal, _chunks, _horizon_array

WALK_DOSE_HOURS = (1, 3)
WALK_DOSE_STEP_MIN = 1


def walk_dose(
    horizons: pd.DataFrame,
    arrivals: pd.DataFrame,
    *,
    lat: float,
    lon: float,
    hours: tuple[int, ...] = WALK_DOSE_HOURS,
    step_min: int = WALK_DOSE_STEP_MIN,
    site_altitude_m: float = 0.0,
) -> pd.DataFrame:
    """Trailing clear-sky direct dose before each point's arrival, per walk.

    horizons: point_id, azimuth_deg, horizon_deg (sun_envelope.horizon_profiles).
    arrivals: point_id, walk_id, t_arrival_utc (tz-aware UTC, NaT outside the
    walk); may hold several walks. Returns point_id, walk_id,
    dose_{h}h_before_wh_m2 per h in ``hours`` (arrivals' row order). NaN for
    NaT arrivals or points absent from ``horizons``. Upper bound (clear sky)."""
    ids, az_grid, H = _horizon_array(horizons)
    row_of = pd.Series(np.arange(len(ids)), index=ids)
    cols = [f"dose_{h}h_before_wh_m2" for h in hours]
    out = arrivals[["point_id", "walk_id"]].reset_index(drop=True).copy()
    for c in cols:
        out[c] = np.nan
    t_all = pd.DatetimeIndex(arrivals["t_arrival_utc"]).tz_convert("UTC")
    rows = row_of.reindex(arrivals["point_id"]).to_numpy()
    usable = (~t_all.isna()) & ~np.isnan(rows)
    walk = arrivals["walk_id"].to_numpy()
    step = pd.Timedelta(minutes=step_min)

    for w in pd.unique(walk[usable]):
        sel = np.flatnonzero(usable & (walk == w))
        t = t_all[sel]
        start = (t.min() - pd.Timedelta(hours=max(hours))).floor(step)
        end = t.max().ceil(step) + step
        grid = pd.date_range(start, end, freq=step)
        pidx = rows[sel].astype(int)
        uniq = np.unique(pidx)
        cum = np.zeros((len(uniq), len(grid)))
        for sl in _chunks(len(uniq), len(grid)):
            e = _beam_horizontal(H[uniq[sl]], az_grid, grid, lat, lon, site_altitude_m, step_min)
            cum[sl, 1:] = np.cumsum(0.5 * (e[:, 1:] + e[:, :-1]), axis=1)
        local = np.searchsorted(uniq, pidx)
        pos = (t - grid[0]).total_seconds().to_numpy() / step.total_seconds()
        at_end = _interp(cum, local, pos)
        for h, c in zip(hours, cols):
            before = _interp(cum, local, pos - h * 60 / step_min)
            out.loc[sel, c] = at_end - before
    return out


def _interp(cum: np.ndarray, rows: np.ndarray, pos: np.ndarray) -> np.ndarray:
    pos = np.clip(pos, 0, cum.shape[1] - 1)
    i0 = np.minimum(np.floor(pos).astype(int), cum.shape[1] - 2)
    f = pos - i0
    return cum[rows, i0] * (1 - f) + cum[rows, i0 + 1] * f
