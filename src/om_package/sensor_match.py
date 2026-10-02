"""Sensor-matched values of point measures along a walk.

A slow sensor carried along the route reads air it has already passed, so
the value comparable to the sensor at point i is an exponentially weighted
mean over the points already passed:

    X_tau(i) = sum_j w_ij X_j / sum_j w_ij,   t_j <= t_i,  dt = t_i - t_j <= 5 tau,
    w_ij = exp(-dt / tau)

tau is the 63 % response time. If only t90 is known, tau = t90 / ln(10).

Method (O(n) per walk and tau, exact for irregular and tied arrival times):
points are sorted by arrival time and the untruncated filter
S_k = S_{k-1} exp(-(t_k - t_{k-1})/tau) + X_k is run once (numerator and
non-NaN weight together, so NaN X are skipped and the mean renormalises).
The 5 tau truncation is exact: terms older than 5 tau sum to
S_p exp(-(t_i - t_p)/tau), where p is the last point with dt > 5 tau
(found by binary search on the sorted times), so subtracting that tail
leaves the windowed sum. Ties (stalls) count with dt = 0 because i takes the
state at the last point sharing its arrival time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TAUS_S = (5, 10, 30, 60)
TRUNCATION_TAUS = 5.0
_MIN_WEIGHT = 1e-6  # a non-empty window has weight >= exp(-5) ~ 6.7e-3; below this is cancellation noise


def tau_from_t90(t90_s: float) -> float:
    """Time constant (s) of a first-order sensor from its 90 % response time."""
    return float(t90_s) / np.log(10.0)


def _tau_label(tau: float) -> str:
    return f"{float(tau):g}"


def _filter(t: np.ndarray, x: np.ndarray, tau: float) -> np.ndarray:
    """t: sorted ascending seconds (float); x: (n, k) with NaN. Returns (n, k)."""
    ok = ~np.isnan(x)
    v = np.concatenate([np.where(ok, x, 0.0), ok.astype(float)], axis=1)
    n = len(t)
    S = np.empty_like(v)
    acc = np.zeros(v.shape[1])
    prev = t[0] if n else 0.0
    for k in range(n):
        acc = acc * np.exp(-(t[k] - prev) / tau) + v[k]
        S[k] = acc
        prev = t[k]
    q = np.searchsorted(t, t, side="right") - 1
    p = np.searchsorted(t, t - TRUNCATION_TAUS * tau, side="left") - 1
    win = S[q].copy()
    has = p >= 0
    ps = np.where(has, p, 0)
    tail = S[ps] * np.exp(-(t - t[ps]) / tau)[:, None]
    win[has] -= tail[has]
    kx = x.shape[1]
    num, den = win[:, :kx], win[:, kx:]
    good = den > _MIN_WEIGHT
    return np.where(good, num / np.where(good, den, 1.0), np.nan)


def sensor_matched(
    arrivals_one_walk: pd.DataFrame,
    values: pd.DataFrame,
    columns: list[str] | None = None,
    taus=DEFAULT_TAUS_S,
) -> pd.DataFrame:
    """Sensor-matched value of each ``columns`` entry of ``values`` for one walk.

    arrivals_one_walk: point_id, walk_id, t_arrival_utc (tz-aware, NaT outside
    the walk). values: point_id + measure columns (default: all numeric
    non-id columns). Returns a wide table in the arrivals' row order:
    point_id, walk_id, <col>_tau<tau>s for every column and tau. NaN where
    the point has no arrival time or no valid X within 5 tau."""
    if columns is None:
        columns = [c for c in values.columns if c != "point_id" and pd.api.types.is_numeric_dtype(values[c])]
    if arrivals_one_walk["walk_id"].dropna().nunique() > 1:
        raise ValueError("arrivals_one_walk holds more than one walk_id")
    base = arrivals_one_walk[["point_id", "walk_id", "t_arrival_utc"]].reset_index(drop=True)
    merged = base.merge(values[["point_id", *columns]].drop_duplicates("point_id"), on="point_id", how="left")
    out = base[["point_id", "walk_id"]].copy()

    t = pd.DatetimeIndex(merged["t_arrival_utc"]).tz_convert("UTC").tz_localize(None)
    valid = ~t.isna()
    secs = (t[valid] - t[valid].min()).total_seconds().to_numpy() if valid.any() else np.empty(0)
    order = np.argsort(secs, kind="stable")
    idx = np.flatnonzero(valid)[order]
    ts = secs[order]
    x = merged[columns].to_numpy(dtype=float)[idx]

    for tau in taus:
        res = np.full((len(merged), len(columns)), np.nan)
        if len(ts):
            res[idx] = _filter(ts, x, float(tau))
        for j, c in enumerate(columns):
            out[f"{c}_tau{_tau_label(tau)}s"] = res[:, j]
    return out
