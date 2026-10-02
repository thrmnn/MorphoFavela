"""Walk timing for OM2: when did each logger walk reach each route point?

The sensor walks (matched CSVs) are the only source of clock time along the
route. Everything downstream that needs "what was the sun/air doing when the
walker was here" (sun envelope per walk, sensor time-constant smoothing)
needs a per-point arrival time, so it is derived once here.

Rules that keep the timeline honest:
- Only fixes matched to edges of the OM2 route count; side-street detours
  are dropped (and counted) rather than projected onto the wrong stretch.
- Walkers only move forward, so distance is a running maximum over time.
  GPS jitter otherwise produces backward steps that would make arrival
  times non-monotone along the route. Time is never reordered.
- Where the distance stalls (walker stopped), a point's arrival is the
  first moment it was reached, not the moment the walker left.
- Arrivals bracketed by a long fix gap are interpolated guesses; they are
  flagged so consumers can down-weight them.
Logger timestamps are UTC; local time is America/Sao_Paulo (no DST).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point

from .routes import UTM23S, WGS84, load_route, route_line_utm
from .sun_envelope import DEFAULT_TZ

#: Fixes further apart than this make the bracketed arrival an interpolation, not a measurement.
GAP_FLAG_S = 60

#: A walk covering less than this share of the route is flagged partial.
PARTIAL_COVERAGE = 0.9

FIX_COLUMNS = ["walk_id", "t_utc", "distance_along_m", "match_status", "edge_on_route"]
ARRIVAL_COLUMNS = ["point_id", "walk_id", "t_arrival_utc", "arrival_source"]

_TO_UTM = Transformer.from_crs(WGS84, UTM23S, always_xy=True)


def _walk_ids(paths: list[Path]) -> list[str]:
    """OM2_<date>_<period>; the duration is appended only when that collides."""
    base = []
    for p in paths:
        parts = p.stem.split("_")
        base.append(f"OM2_{parts[2]}_{parts[3]}")
    counts = pd.Series(base).value_counts()
    return [
        b if counts[b] == 1 else f"{b}_{p.stem.split('_')[4]}"
        for p, b in zip(paths, base)
    ]


def _route_edge_set(route_json: Path) -> set[tuple[int, int]]:
    _, edges = load_route(route_json)
    return {(e.u, e.v) for e in edges} | {(e.v, e.u) for e in edges}


def _read_walk(path: Path, walk_id: str, edge_set: set, line: LineString) -> pd.DataFrame:
    raw = pd.read_csv(path)
    t = pd.to_datetime(raw["timestamp_utc"], utc=True, format="ISO8601", errors="coerce")
    ok = t.notna() & raw["matched_lon"].notna() & raw["matched_lat"].notna()
    raw, t = raw[ok], t[ok]
    on = np.array(
        [(u, v) in edge_set for u, v in zip(raw["matched_edge_u"], raw["matched_edge_v"])],
        dtype=bool,
    )
    x, y = _TO_UTM.transform(raw["matched_lon"].to_numpy(), raw["matched_lat"].to_numpy())
    dist = np.array([line.project(Point(a, b)) for a, b in zip(x, y)])
    df = pd.DataFrame({
        "walk_id": walk_id,
        "t_utc": t.to_numpy(),
        "distance_along_m": dist,
        "match_status": raw["match_status"].to_numpy(),
        "edge_on_route": on,
    })
    df["t_utc"] = pd.to_datetime(df["t_utc"], utc=True)
    return df.sort_values("t_utc", kind="stable").reset_index(drop=True)


def _summarise(walk_id: str, df: pd.DataFrame, route_len: float) -> dict:
    """Times span every row (the walk as logged); distance stats use on-route fixes only.

    A walk with no on-route fixes (it left the route entirely) is kept with
    coverage 0 so the walk count stays honest.
    """
    on = df[df["edge_on_route"]]
    start, end = df["t_utc"].iloc[0], df["t_utc"].iloc[-1]
    start_local = start.tz_convert(DEFAULT_TZ)
    if on.empty:
        first_m = last_m = max_gap = np.nan
        coverage = 0.0
    else:
        first_m, last_m = on["distance_along_m"].iloc[0], on["distance_along_m"].iloc[-1]
        coverage = (last_m - first_m) / route_len
        max_gap = float(on["t_utc"].diff().dt.total_seconds().max()) if len(on) > 1 else 0.0
    return {
        "walk_id": walk_id,
        "date": start_local.date(),
        "period": walk_id.split("_")[2],
        "start_utc": start,
        "end_utc": end,
        "start_local": start_local,
        "end_local": end.tz_convert(DEFAULT_TZ),
        "mid_utc": start + (end - start) / 2,
        "duration_min": (end - start).total_seconds() / 60.0,
        "n_rows": len(df),
        "n_rows_on_route": len(on),
        "share_on_route": len(on) / len(df),
        "share_interpolated": float((on["match_status"] == "interpolated").mean()) if len(on) else np.nan,
        "first_m": first_m,
        "last_m": last_m,
        "coverage_share": coverage,
        "max_gap_s": max_gap,
        "partial": bool(coverage < PARTIAL_COVERAGE),
    }


def load_walks(matched_dir: Path, route_json: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(walks, fixes). fixes holds on-route rows only, distance made monotone."""
    paths = sorted(Path(matched_dir).glob("OM_2_*.csv"))
    ids = _walk_ids(paths)
    edge_set = _route_edge_set(route_json)
    _, line = route_line_utm(route_json)
    summaries, parts = [], []
    for path, wid in zip(paths, ids):
        df = _read_walk(path, wid, edge_set, line)
        on = df[df["edge_on_route"]].reset_index(drop=True)
        on["distance_along_m"] = np.maximum.accumulate(on["distance_along_m"].to_numpy())
        summaries.append(_summarise(wid, pd.concat([on, df[~df["edge_on_route"]]]).sort_values("t_utc", kind="stable"), line.length))
        parts.append(on)
    return pd.DataFrame(summaries), pd.concat(parts, ignore_index=True)[FIX_COLUMNS]


def arrival_times(points: pd.DataFrame, fixes: pd.DataFrame) -> pd.DataFrame:
    """Arrival time of ONE walk at each point (needs point_id, distance_along_m).

    Between the first and last fix, time is interpolated linearly against
    distance. The bracket's upper fix is the first fix at/after the point's
    distance, so a point at a stall returns the stall's first time.
    """
    f = fixes.sort_values("t_utc", kind="stable")
    if f.empty:
        raise ValueError("no fixes; use all_arrivals, which handles off-route walks")
    d = np.maximum.accumulate(f["distance_along_m"].to_numpy())
    t = f["t_utc"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")
    p = points["distance_along_m"].to_numpy(dtype=float)
    inside = (p >= d[0]) & (p <= d[-1])

    hi = np.searchsorted(d, p, side="left").clip(0, len(d) - 1)
    lo = np.maximum(hi - 1, 0)
    span = d[hi] - d[lo]
    frac = np.divide(p - d[lo], span, out=np.zeros_like(p), where=span > 0)
    exact = (d[hi] == p) | (hi == 0)
    t_ns = np.where(exact, t[hi], t[lo] + frac * (t[hi] - t[lo]))
    gappy = inside & ~exact & ((t[hi] - t[lo]) / 1e9 > GAP_FLAG_S)

    arrival = pd.Series(pd.to_datetime(np.round(np.where(inside, t_ns, 0)).astype("int64"), utc=True))
    arrival[~inside] = pd.NaT
    source = np.where(~inside, "outside_walk", np.where(gappy, "gap_interpolated", "gps"))
    return pd.DataFrame({
        "point_id": points["point_id"].to_numpy(),
        "walk_id": f["walk_id"].iloc[0],
        "t_arrival_utc": arrival,
        "arrival_source": source,
    })[ARRIVAL_COLUMNS]


def all_arrivals(points: pd.DataFrame, fixes: pd.DataFrame, walks: pd.DataFrame) -> pd.DataFrame:
    """Long table: every walk x every point."""
    by_walk = dict(tuple(fixes.groupby("walk_id")))
    return pd.concat(
        [_all_outside(points, w) if w not in by_walk else arrival_times(points, by_walk[w])
         for w in walks["walk_id"]],
        ignore_index=True,
    )


def _all_outside(points: pd.DataFrame, walk_id: str) -> pd.DataFrame:
    return pd.DataFrame({
        "point_id": points["point_id"].to_numpy(),
        "walk_id": walk_id,
        "t_arrival_utc": pd.Series(pd.NaT, index=range(len(points)), dtype="datetime64[ns, UTC]"),
        "arrival_source": "outside_walk",
    })[ARRIVAL_COLUMNS]
