"""P-03 (part 2) — segment aggregation: re-aggregate any point-level table
to segments of any length the team chooses. No segments are imposed by
P-02 (points are the only stable unit); this is applied on top, on demand.

A segment groups consecutive points (by seq / distance_along_m) into
non-overlapping runs of ``segment_length_m``: segment i covers
[i * segment_length_m, (i+1) * segment_length_m). Point count is
conserved: every input point lands in exactly one segment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: columns that describe identity/position, never averaged.
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
