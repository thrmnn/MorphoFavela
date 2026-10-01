#!/usr/bin/env python3
"""P-03 — re-aggregate this package's OM2 point table to any segment
length you choose (any of the four buffer radii — 5, 10, 20, 50 m — are
already columns on ``OM2/points.*``; this script only groups points into
segments, it does not recompute buffers).

Standalone: pandas + pyarrow only, no Brisa+ (MorphoFavela) import — this file
travels with the package and works from inside the package directory
alone. Its ``aggregate_to_segments`` function mirrors
``src/om_package/segments.py``'s function of the same name exactly (same
package build, same commit) — if you have the Brisa+ (MorphoFavela) repo checked
out, the repo's own ``scripts/aggregate_om_points.py`` CLI wraps that
library function instead of this copy.

A segment groups consecutive points (by ``distance_along_m``) into
non-overlapping runs of ``segment_length_m``: segment i covers
[i * segment_length_m, (i+1) * segment_length_m). Point count is
conserved: every input point lands in exactly one segment.

Choosing the length: 1 m points are far finer than a sensor's response while
walking, and neighbouring points are strongly correlated. A defensible segment
is L ~ walking speed x k x the sensor's response time constant (k = 3 gives
~95% of a step response). The default (10 m) is a placeholder, not derived
from any sensor; see the README, "Using the data".

Run (from inside the package directory):
    python OM2/aggregate_to_segments.py --points OM2/points.parquet \\
        --segment-m 20 --out OM2/segments_20m.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

#: Placeholder default, not derived from any sensor (see the module docstring).
DEFAULT_SEGMENT_M = 10.0

#: columns that describe identity/position, never averaged — kept in sync
#: with src/om_package/segments.py's _ID_COLS by hand (this file has no
#: repo import to share it from).
_ID_COLS = {"point_id", "route_id", "seq", "distance_along_m", "height_m", "geometry", "x", "y"}


def aggregate_to_segments(
    points_df: pd.DataFrame, segment_length_m: float, distance_col: str = "distance_along_m"
) -> pd.DataFrame:
    """Mean-aggregate numeric point variables into fixed-length segments.

    Returns one row per segment: segment_id, start/end distance_along_m,
    n_points, mean of every other numeric column (NaNs excluded).
    """
    if segment_length_m <= 0:
        raise ValueError("segment_length_m must be > 0")
    df = points_df.copy()
    df["segment_id"] = (df[distance_col] // segment_length_m).astype(int)

    numeric_cols = [
        c
        for c in df.columns
        if c not in _ID_COLS and c != "segment_id" and pd.api.types.is_numeric_dtype(df[c])
    ]

    grouped = df.groupby("segment_id", sort=True)
    agg = grouped[numeric_cols].mean(numeric_only=True)
    agg["n_points"] = grouped.size()
    agg["segment_start_m"] = grouped[distance_col].min()
    agg["segment_end_m"] = grouped[distance_col].max()
    if "route_id" in df.columns:
        agg["route_id"] = grouped["route_id"].first()

    agg = agg.reset_index()
    cols = ["segment_id", "route_id", "segment_start_m", "segment_end_m", "n_points"] + numeric_cols
    cols = [c for c in cols if c in agg.columns]
    return agg[cols]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", required=True, help="path to a points parquet/csv (this package's OM2/points.*)")
    ap.add_argument(
        "--segment-m", "--segment-length-m", dest="segment_m", type=float, default=DEFAULT_SEGMENT_M,
        help=f"segment length in metres (default {DEFAULT_SEGMENT_M:g}, a placeholder, not sensor-derived)",
    )
    ap.add_argument("--out", required=True, help="output path (.parquet or .csv)")
    args = ap.parse_args()

    points_path = Path(args.points)
    df = pd.read_parquet(points_path) if points_path.suffix == ".parquet" else pd.read_csv(points_path)

    segments = aggregate_to_segments(df, args.segment_m)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        segments.to_csv(out_path, index=False)
    else:
        segments.to_parquet(out_path, index=False)

    n_points_in = len(df)
    n_points_out = int(segments["n_points"].sum())
    print(f"[aggregate_to_segments] {n_points_in} points -> {len(segments)} segments of {args.segment_m} m")
    print(f"[aggregate_to_segments] point count conserved: {n_points_in == n_points_out} ({n_points_in} == {n_points_out})")
    print(f"[aggregate_to_segments] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
