"""P-03 (part 1) — buffer variables: any point variable computed from a
circular buffer around each OM2 point, at the four PI-given radii
(5, 10, 20, 50 m).

Buffer variables here are building-footprint derived (airborne, 2019):
  - lambda_p_buffer_{r}: building footprint area / buffer area (plan
    density within the buffer — same definition as the grid's lambda_p,
    src/urban_morphology.py, just computed on a circle instead of a 10 m
    square).
  - building_count_buffer_{r}: count of buildings intersecting the buffer.
  - building_height_mean_buffer_{r}: unweighted mean of 'altura' over
    buildings intersecting the buffer (NaN where none intersect).

Source: data/maré/buildings_extended_300m.gpkg (covers 300 m past the
bairro boundary, so a 50 m buffer never runs off the edge of the
building layer even for points near Maré's edge).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .io_utils import Paths

BUFFER_RADII_M = (5, 10, 20, 50)


def compute_buffer_variables(
    points_gdf: gpd.GeoDataFrame, paths: Paths, radii=BUFFER_RADII_M
) -> pd.DataFrame:
    buildings = gpd.read_file(paths.buildings_extended_300m)[["altura", "geometry"]]
    sindex = buildings.sindex

    out = pd.DataFrame({"point_id": points_gdf["point_id"].to_numpy()})
    geoms = points_gdf.geometry.to_numpy()

    for r in radii:
        buffer_area = np.pi * r * r
        lambda_p = np.full(len(geoms), np.nan)
        counts = np.zeros(len(geoms), dtype=int)
        mean_h = np.full(len(geoms), np.nan)

        for i, pt in enumerate(geoms):
            buf = pt.buffer(r)
            cand_idx = list(sindex.query(buf, predicate="intersects"))
            if not cand_idx:
                lambda_p[i] = 0.0
                counts[i] = 0
                continue
            cand = buildings.iloc[cand_idx]
            inter_area = cand.geometry.intersection(buf).area
            lambda_p[i] = float(inter_area.sum()) / buffer_area
            touching = inter_area > 0
            counts[i] = int(touching.sum())
            if touching.any():
                mean_h[i] = float(cand.loc[touching, "altura"].mean())

        out[f"lambda_p_buffer_{r}m"] = lambda_p
        out[f"building_count_buffer_{r}m"] = counts
        out[f"building_height_mean_buffer_{r}m"] = mean_h

    return out
