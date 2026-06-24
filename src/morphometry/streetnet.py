"""Street-network / beco (alley) configuration metrics.

The intensity vector (λp, H, λf …) measures *how much* fabric a cell holds; it
is blind to *how the circulation is wired*. This module adds the second
configuration axis the council asked for — the favela's street/alley network —
as two per-cell descriptors that a graph captures but a density field cannot:

- **intersection density** — count of street-network nodes with degree ≥ 3
  (true junctions, not mid-line vertices or dead-ends) falling inside the cell,
  also expressed per hectare. Finer beco lattices branch more often.
- **mean segment length** — average length of the road segments intersecting the
  cell. Shorter segments = finer-grained circulation (the dense Core grain).

We deliberately avoid a graph library: a node is simply a *shared endpoint* of
the LineString segments, snapped to a small tolerance so that hand-digitised
endpoints that should coincide are merged. Degree is the number of distinct
segment ends meeting at that snapped point. This is geopandas/shapely-only.
"""

from __future__ import annotations

import glob
from collections import defaultdict

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

# Endpoints within this distance (m) are treated as the same node. Favela road
# vectors are hand-digitised; 0.5 m merges near-coincident ends without fusing
# genuinely separate junctions in a beco lattice.
SNAP_TOLERANCE_M = 0.5


def find_roads_path(site_dir: str) -> str | None:
    """Locate the road/street shapefile for a site, tolerating naming drift.

    Campaign sites use ``*roads*.shp`` (e.g. ``roads_rocinha.shp``,
    ``Vidigal_roads.shp``) except Maré, which ships ``street_mare.shp``. Test
    fixtures (``*_test.shp``) are skipped. Returns ``None`` if nothing usable
    is found.
    """
    raw = f"{site_dir}/raw"
    cands: list[str] = []
    for pat in ("*roads*.shp", "*_roads.shp", "*street*.shp"):
        cands.extend(glob.glob(f"{raw}/{pat}"))
    cands = sorted({c for c in cands if not c.lower().endswith("_test.shp")})
    return cands[0] if cands else None


def _snap_key(x: float, y: float, tol: float) -> tuple[int, int]:
    """Quantise a coordinate to the snapping grid so coincident ends collide."""
    return (round(x / tol), round(y / tol))


def build_network_nodes(
    roads: gpd.GeoDataFrame, tol: float = SNAP_TOLERANCE_M
) -> gpd.GeoDataFrame:
    """Derive street-network nodes with their degree from road LineStrings.

    A node is a snapped shared endpoint of the *whole-feature* lines (not the
    intermediate vertices). Degree = number of feature-endpoints meeting there.
    Degree ≥ 3 marks a true junction; degree 1 is a dead-end; degree 2 is a
    pass-through (kept, but not counted as an intersection).

    Returns a GeoDataFrame of node points with a ``degree`` column, in the
    roads' CRS.
    """
    endpoint_count: dict[tuple[int, int], int] = defaultdict(int)
    rep_coord: dict[tuple[int, int], tuple[float, float]] = {}

    for geom in roads.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = [geom] if isinstance(geom, LineString) else (
            list(geom.geoms) if isinstance(geom, MultiLineString) else []
        )
        for part in parts:
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            for end in (coords[0], coords[-1]):
                key = _snap_key(end[0], end[1], tol)
                endpoint_count[key] += 1
                rep_coord.setdefault(key, (end[0], end[1]))

    keys = list(endpoint_count.keys())
    pts = [Point(*rep_coord[k]) for k in keys]
    deg = [endpoint_count[k] for k in keys]
    return gpd.GeoDataFrame({"degree": deg}, geometry=pts, crs=roads.crs)


def cell_streetnet_metrics(
    grid: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    tol: float = SNAP_TOLERANCE_M,
) -> pd.DataFrame:
    """Per-cell intersection density and mean segment length.

    Parameters
    ----------
    grid : GeoDataFrame
        Must carry ``zone_id`` and the cell polygon geometry, in the same CRS as
        ``roads`` (the caller is responsible for reprojection).
    roads : GeoDataFrame
        Road/street LineStrings.

    Returns
    -------
    DataFrame indexed implicitly, one row per grid cell, with columns:
    ``zone_id``, ``intersection_count`` (degree ≥ 3 nodes in the cell),
    ``intersection_density_ha`` (per hectare), ``mean_segment_len_m`` (mean
    length of segments intersecting the cell; NaN where no road touches the
    cell), ``n_segments``.
    """
    nodes = build_network_nodes(roads, tol=tol)
    junctions = nodes[nodes["degree"] >= 3].copy()

    # Whole-feature segments (one row per original line) keep "segment length"
    # meaning the digitised street trecho, the natural unit of beco grain.
    seg = roads[roads.geometry.notna()].copy()
    seg = seg[~seg.geometry.is_empty]
    seg = seg.explode(index_parts=False, ignore_index=True)
    seg = seg[seg.geometry.geom_type == "LineString"].copy()
    seg["seg_len"] = seg.geometry.length

    cell_area_ha = grid.geometry.area / 1.0e4

    # Junction count per cell via spatial join.
    if len(junctions):
        jj = gpd.sjoin(
            junctions[["geometry"]], grid[["zone_id", "geometry"]],
            how="inner", predicate="within",
        )
        jcount = jj.groupby("zone_id").size()
    else:
        jcount = pd.Series(dtype=int)

    # Mean segment length per cell: segments that intersect the cell.
    if len(seg):
        ss = gpd.sjoin(
            seg[["seg_len", "geometry"]], grid[["zone_id", "geometry"]],
            how="inner", predicate="intersects",
        )
        seg_agg = ss.groupby("zone_id")["seg_len"].agg(["mean", "size"])
    else:
        seg_agg = pd.DataFrame(columns=["mean", "size"])

    out = pd.DataFrame({"zone_id": grid["zone_id"].to_numpy()})
    out["_area_ha"] = cell_area_ha.to_numpy()
    out["intersection_count"] = (
        out["zone_id"].map(jcount).fillna(0).astype(int)
    )
    out["intersection_density_ha"] = out["intersection_count"] / out["_area_ha"]
    out["mean_segment_len_m"] = out["zone_id"].map(
        seg_agg["mean"] if "mean" in seg_agg else pd.Series(dtype=float)
    )
    out["n_segments"] = (
        out["zone_id"].map(
            seg_agg["size"] if "size" in seg_agg else pd.Series(dtype=float)
        ).fillna(0).astype(int)
    )
    return out.drop(columns="_area_ha")


def summarise_by_type(
    cells: pd.DataFrame, type_col: str = "morphotype_smooth"
) -> pd.DataFrame:
    """Aggregate per-cell streetnet metrics by morphotype (T0–T5).

    ``cells`` must carry the streetnet columns plus ``type_col``. Cells whose
    type is NaN are dropped. Mean segment length is averaged only over cells a
    road touches (NaN-aware), so a sparse type isn't biased by empty cells.
    """
    df = cells.dropna(subset=[type_col]).copy()
    df[type_col] = df[type_col].astype(int)
    g = df.groupby(type_col)
    summary = pd.DataFrame({
        "n_cells": g.size(),
        "mean_intersection_density_ha": g["intersection_density_ha"].mean(),
        "mean_intersection_count": g["intersection_count"].mean(),
        "mean_segment_len_m": g["mean_segment_len_m"].mean(),
        "median_segment_len_m": g["mean_segment_len_m"].median(),
        "frac_cells_with_road": g["n_segments"].apply(lambda s: (s > 0).mean()),
    })
    return summary.reset_index().rename(columns={type_col: "morphotype"})
