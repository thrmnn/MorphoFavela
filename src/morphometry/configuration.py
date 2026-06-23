"""Configuration metrics — the relational fabric the intensity vector misses.

The council's top "what's missing": the signature is strong on intensity (λp, λf,
H) but blind to *configuration* — the favela-defining trait that buildings fuse
into contiguous blocks with shared party walls and no plot subdivision. This module
adds **party-wall ratio** (fraction of building perimeter shared with a neighbour),
the cleanest single discriminator between detached formal fabric and fused favela
fabric. Aggregated to the 10 m grid as a new fabric feature.

Method: a wall is "shared" when it does not lie on the exterior boundary of the
union of all buildings. So per building,
    party_wall_ratio = (perimeter − len(boundary ∩ union_exterior)) / perimeter
clipped to [0, 1]. Vectorized against the dissolved union boundary.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd


def party_wall_ratio(buildings: gpd.GeoDataFrame, snap: float = 0.10) -> np.ndarray:
    """Per-building fraction of perimeter shared with an abutting neighbour.

    Local STRtree neighbour query — each building's boundary is intersected only
    with its handful of near neighbours (dilated by ``snap`` to catch the small
    digitisation gaps common in favela cadastre), not the global union. O(n·k),
    so it scales to the 40k-building sites. Per building, the shared length is the
    boundary lying within ``snap`` of any neighbour, capped at the perimeter.
    """
    from shapely import STRtree

    geoms = buildings.geometry.to_numpy()
    perim = buildings.geometry.length.to_numpy()
    tree = STRtree(geoms)
    shared = np.zeros(len(geoms))
    for i, g in enumerate(geoms):
        bnd = g.boundary
        for j in tree.query(g.buffer(snap)):
            if j == i:
                continue
            shared[i] += bnd.intersection(geoms[j].buffer(snap)).length
    ratio = np.divide(shared, perim, out=np.zeros_like(perim), where=perim > 0)
    return np.clip(ratio, 0.0, 1.0)


def aggregate_to_grid(
    buildings: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, ratio: np.ndarray
) -> pd.DataFrame:
    """Footprint-area-weighted mean party-wall ratio per cell.

    Each building is assigned to the cell containing its representative point
    (point-in-polygon via spatial index — O(n log n), not a full geometric
    overlay), then averaged weighted by footprint area. Buildings are ~10–15 m and
    cells 10 m, so centroid assignment is the right granularity. Cells with no
    building are NaN + ``has_config = False``.
    """
    b = gpd.GeoDataFrame(
        {"pwr": ratio, "barea": buildings.geometry.area.to_numpy()},
        geometry=buildings.geometry.representative_point(), crs=buildings.crs)
    j = gpd.sjoin(b, grid[["zone_id", "geometry"]], how="inner", predicate="within")
    g = j.groupby("zone_id")
    out = pd.DataFrame({
        "party_wall_ratio": g.apply(
            lambda d: np.average(d["pwr"], weights=d["barea"])
            if d["barea"].sum() else np.nan, include_groups=False),
        "n_buildings_cfg": g.size(),
    })
    out = out.reindex(grid["zone_id"].to_numpy())
    out["has_config"] = out["n_buildings_cfg"].notna() & (out["n_buildings_cfg"] > 0)
    out["n_buildings_cfg"] = out["n_buildings_cfg"].fillna(0).astype(int)
    out.index.name = "zone_id"
    return out.reset_index()
