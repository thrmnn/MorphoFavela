"""Maré study-area geometry — THIN WRAPPER over src.sites.territory (SITETERR,
2026-09-24). Kept only because scripts/build_site_dashboard.py,
scripts/build_html_dashboard.py and scripts/render_mare_territory_map.py
already import `mare_study_area as msa` and call `msa.load_study_area()` /
`msa.within_mask()`; new code should call `src.sites.territory.load_territory`
directly instead.

Two boundaries exist for Maré and must never be conflated (PI-approved
2026-09-23):

- STUDY AREA  = union(16 communities) ∩ DATA EXTENT — governs which
  cells/observers/buildings count in Maré's statistics, and is what gets
  outlined on the site sheet and the interactive dashboard. Registry:
  config/sites.yaml maré.study_area (kind: subunits_union_in_extent).
- DATA EXTENT = the bairro polygon itself — the near_boundary / edge-halo
  15 m logic (scripts/build_site_dashboard.py, scripts/build_html_dashboard.py)
  must keep measuring distance to THIS boundary: buildings actually stop at
  the bairro edge, not at an interior community-to-community line, and
  swapping in the study-area edge there would flag interior community
  borders as "edge" — exactly the bug this module exists to prevent.

Marcílio Dias, one of the 16 communities, lies ~2 km north of bairro Maré
and is excluded from the study area. The exclusion is computed by GEOMETRY
(the community's own share of area inside the data extent, config/sites.yaml
maré.study_area.share_threshold = 0.5), never by name — see
src.sites.territory.build_study_area.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from src.sites.territory import ROOT_DEFAULT, load_sites_config, load_territory, within_mask  # noqa: F401

# Hardcoded, not Path(__file__).resolve().parents[2]: this module is also
# imported from a git worktree that has no data/ of its own — every caller
# reads data/outputs from the one main checkout regardless of which
# checkout's copy of this file runs. src.sites.territory.load_territory
# takes the same `root` explicitly, so this default only matters for callers
# (still) invoking this module's own functions with no argument.
ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")

# Backward-compatible constants some existing tests/callers read directly
# (tests/test_mare_study_area.py). Sourced from config/sites.yaml — not
# retyped — so they can never drift from what load_territory("maré") itself
# uses.
_MARE_CFG = load_sites_config()["maré"]
BAIRRO_SHP = ROOT / "data" / _MARE_CFG["data_extent"]
NEIGHBOURHOODS_GPKG = ROOT / "data" / _MARE_CFG["subunits"]["file"]
PROVENANCE_JSON = ROOT / "data" / _MARE_CFG["subunits"]["provenance"]
INSIDE_EXTENT_MIN_SHARE = _MARE_CFG["study_area"]["share_threshold"]


def load_data_extent(root: Path = ROOT) -> "shapely.Geometry":  # noqa: F821
    """The bairro polygon (data extent): buildings/DTM are clipped to this,
    never to the study area."""
    return load_territory("maré", root=root).data_extent


def load_communities(root: Path = ROOT) -> gpd.GeoDataFrame:
    """All 16 Redes da Maré communities, unfiltered (no `in_extent` column)."""
    from src.sites.territory import load_sites_config, load_subunits
    cfg = load_sites_config()["maré"]
    comm, _ = load_subunits(cfg, root)
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
    t = load_territory("maré", root=root)
    if t.subunits is None or len(t.subunits_included) == 0:
        raise ValueError("no community intersects the Maré data extent — check the crosswalk / geometry")
    return {
        "communities": t.subunits,
        "included": t.subunits_included,
        "excluded": t.subunits_excluded,
        "data_extent": t.data_extent,
        "study_area": t.study_area,
    }
