"""Maré study-area geometry: the union of the 16 Redes da Maré communities
(data/maré/neighbourhoods.gpkg, layer "communities"; provenance in
data/maré/neighbourhoods_provenance.json, built by
scripts/data_utils/build_mare_neighbourhoods.py), intersected with the site
DATA EXTENT — the bairro polygon (data/maré/raw/mare_boundary.shp) that
building footprints and the DTM are actually clipped to.

Two boundaries exist for Maré and must never be conflated (PI-approved
2026-09-23):

- STUDY AREA  = union(16 communities) ∩ DATA EXTENT — governs which
  cells/observers/buildings count in Maré's statistics, and is what gets
  outlined on the site sheet and the interactive dashboard.
- DATA EXTENT = the bairro polygon itself — the near_boundary / edge-halo
  15 m logic (scripts/build_site_dashboard.py, scripts/build_html_dashboard.py)
  must keep measuring distance to THIS boundary: buildings actually stop at
  the bairro edge, not at an interior community-to-community line, and
  swapping in the study-area edge there would flag interior community
  borders as "edge" — exactly the bug this module exists to prevent.

Marcílio Dias, one of the 16 communities, lies ~2 km north of bairro Maré
and is excluded from the study area. The exclusion is computed by GEOMETRY
(the community's own share of area inside the data extent), never by name:
data/maré/neighbourhoods.gpkg's own QA (share_in_official_bairro) measures
Marcílio Dias at 0.0% inside the bairro against >= 99.1% for every other
community, so INSIDE_EXTENT_MIN_SHARE = 0.5 cleanly separates the two
groups without hardcoding "Marcílio Dias" as a name to drop.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

# Hardcoded, not Path(__file__).resolve().parents[2]: this module is also
# imported from a git worktree that has no data/ of its own (see other
# ROOT-hardcoding scripts, e.g. scripts/build_site_dashboard.py,
# scripts/build_html_dashboard.py) — every caller reads data/outputs from
# the one main checkout regardless of which checkout's copy of this file runs.
ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")
NEIGHBOURHOODS_GPKG = ROOT / "data" / "maré" / "neighbourhoods.gpkg"
BAIRRO_SHP = ROOT / "data" / "maré" / "raw" / "mare_boundary.shp"
PROVENANCE_JSON = ROOT / "data" / "maré" / "neighbourhoods_provenance.json"

# A community with less than this share of its own area inside the data
# extent is excluded from the study area. Geometry-based cutoff, not a
# per-name list: every included community actually measures >= 0.99, the
# one excluded community measures 0.0 (see module docstring).
INSIDE_EXTENT_MIN_SHARE = 0.5


def _union(geoseries) -> "shapely.Geometry":
    return geoseries.union_all() if hasattr(geoseries, "union_all") else geoseries.unary_union


def load_data_extent(root: Path = ROOT) -> "shapely.Geometry":
    """The bairro polygon (data extent): buildings/DTM are clipped to this,
    never to the study area. Used unmodified by near_boundary/edge-halo
    logic everywhere in the codebase."""
    bairro_shp = root / "data" / "maré" / "raw" / "mare_boundary.shp"
    bairro = gpd.read_file(bairro_shp)
    if bairro.crs is None:
        bairro = bairro.set_crs(31983)
    else:
        bairro = bairro.to_crs(31983)
    return _union(bairro.geometry)


def load_communities(root: Path = ROOT) -> gpd.GeoDataFrame:
    """All 16 Redes da Maré communities, unfiltered."""
    gpkg = root / "data" / "maré" / "neighbourhoods.gpkg"
    comm = gpd.read_file(gpkg, layer="communities")
    if comm.crs is None:
        comm = comm.set_crs(31983)
    else:
        comm = comm.to_crs(31983)
    return comm


def load_study_area(root: Path = ROOT) -> dict:
    """Build the study-area geometry and the data-extent-based in/out split.

    Returns a dict with:
      communities  — all 16, with an added boolean column `in_extent`
      included     — the (typically 15) communities inside the data extent
      excluded     — the (typically 1, Marcílio Dias) communities outside it
      data_extent  — the bairro polygon geometry (shapely), unclipped
      study_area   — union(included) ∩ data_extent (shapely)
    """
    extent_geom = load_data_extent(root)
    comm = load_communities(root)
    share = comm.geometry.intersection(extent_geom).area / comm.geometry.area
    comm = comm.assign(in_extent=(share.to_numpy() >= INSIDE_EXTENT_MIN_SHARE))
    included = comm[comm["in_extent"]].reset_index(drop=True)
    excluded = comm[~comm["in_extent"]].reset_index(drop=True)
    if len(included) == 0:
        raise ValueError("no community intersects the Maré data extent — check the crosswalk / geometry")
    study_area_geom = _union(included.geometry).intersection(extent_geom)
    return {
        "communities": comm,
        "included": included,
        "excluded": excluded,
        "data_extent": extent_geom,
        "study_area": study_area_geom,
    }


def within_mask(x: np.ndarray, y: np.ndarray, geom) -> np.ndarray:
    """Vectorised point-in-polygon test against `geom`, bbox-prefiltered
    first (same pattern as scripts/mare_definition_sensitivity.py's
    `_within`, applied here to arrays instead of a DataFrame)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0, y0, x1, y1 = geom.bounds
    in_box = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    out = np.zeros(len(x), dtype=bool)
    if in_box.any():
        out[in_box] = shapely.contains_xy(geom, x[in_box], y[in_box])
    return out
