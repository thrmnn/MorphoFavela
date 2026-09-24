"""Nearest-neighbour spatial join by KDTree (projected coords, metres)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def nearest_join(
    points_xy: np.ndarray,
    source_xy: np.ndarray,
    source_values: pd.DataFrame,
    max_dist_m: float,
) -> pd.DataFrame:
    """For each row in points_xy, pull the nearest source row's values.

    Returns a DataFrame (same length/order as points_xy) with source_values'
    columns plus ``_join_dist_m``; rows beyond max_dist_m are all-NaN (a
    join gap, not a guess) and ``_join_dist_m`` stays NaN too.
    """
    tree = cKDTree(source_xy)
    dist, idx = tree.query(points_xy, k=1)
    out = source_values.iloc[idx].reset_index(drop=True)
    ok = dist <= max_dist_m
    out = out.where(pd.Series(ok, name="_ok"), other=np.nan)
    out["_join_dist_m"] = np.where(ok, dist, np.nan)
    return out
