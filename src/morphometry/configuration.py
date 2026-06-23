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


def party_wall_ratio(buildings: gpd.GeoDataFrame, snap: float = 0.20) -> np.ndarray:
    """Per-building fraction of perimeter shared with an abutting neighbour.

    ``snap`` (m) dilates the union boundary so near-touching walls (small
    digitisation gaps common in favela cadastre) still count as shared.
    """
    geom = buildings.geometry
    perim = geom.length.to_numpy()
    union_bnd = geom.union_all().boundary
    if snap > 0:
        union_bnd = union_bnd.buffer(snap)
        # exterior wall = the part of each boundary within the dilated union edge
        ext = geom.boundary.intersection(union_bnd).length.to_numpy()
    else:
        ext = geom.boundary.intersection(union_bnd).length.to_numpy()
    ratio = 1.0 - np.divide(ext, perim, out=np.zeros_like(perim), where=perim > 0)
    return np.clip(ratio, 0.0, 1.0)


def aggregate_to_grid(
    buildings: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, ratio: np.ndarray
) -> pd.DataFrame:
    """Area-weighted mean party-wall ratio per cell (+ contributing building count).

    Area-weighted so a cell's value reflects the fabric covering it, not a count of
    slivers. Cells with no building are NaN + ``has_config = False``.
    """
    b = buildings[["geometry"]].copy()
    b["pwr"] = ratio
    b["barea"] = b.geometry.area
    b["bid"] = np.arange(len(b))
    inter = gpd.overlay(
        b, grid[["zone_id", "geometry"]], how="intersection", keep_geom_type=False
    )
    inter["w"] = inter.geometry.area
    g = inter.groupby("zone_id")
    out = pd.DataFrame({
        "party_wall_ratio": g.apply(
            lambda d: np.average(d["pwr"], weights=d["w"]) if d["w"].sum() else np.nan,
            include_groups=False),
        "n_buildings_cfg": g["bid"].nunique(),
    })
    out = out.reindex(grid["zone_id"].to_numpy())
    out["has_config"] = out["n_buildings_cfg"].notna() & (out["n_buildings_cfg"] > 0)
    out["n_buildings_cfg"] = out["n_buildings_cfg"].fillna(0).astype(int)
    out.index.name = "zone_id"
    return out.reset_index()
