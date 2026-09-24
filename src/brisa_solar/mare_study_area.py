"""Maré study-area geometry — THIN WRAPPER over src.sites.territory (SITETERR,
2026-09-24). Kept only because scripts/build_site_dashboard.py,
scripts/build_html_dashboard.py and scripts/render_mare_territory_map.py
already import `mare_study_area as msa` and call `msa.load_study_area()` /
`msa.within_mask()`; new code should call `src.sites.territory.load_territory`
directly instead.

Two boundaries exist for Maré and must never be conflated:

- STUDY AREA  = the IPP Territórios Sociais complex outline (territory 03,
  a single contiguous polygon) — governs which cells/observers/buildings
  count in Maré's statistics, and is what gets outlined on the site sheet
  and the interactive dashboard. Registry: config/sites.yaml
  maré.study_area (kind: polygon_file). PROMOTED 2026-09-24 (PI ruling,
  brisaverse resolved_decisions id mare_site_study_area) from the previous
  definition, union(16 communities) ∩ data extent — that definition is kept
  as a declared candidate/history entry (config/sites.yaml
  maré.study_area_candidates, id communities_union_in_extent) so it stays
  reproducible. Unlike that old definition, the outline covers ground
  between the 16 communities too (canals, streets, open land) — every
  study-area point/cell gets exactly one subunit label, a community name
  or the literal "between communities" (src.sites.territory.label_subunits,
  BETWEEN_SUBUNITS_LABEL).
- DATA EXTENT = the bairro polygon itself — the near_boundary / edge-halo
  15 m logic (scripts/build_site_dashboard.py, scripts/build_html_dashboard.py)
  must keep measuring distance to THIS boundary: buildings actually stop at
  the bairro edge, not at an interior community-to-community line, and
  swapping in the study-area edge there would flag interior community
  borders as "edge" — exactly the bug this module exists to prevent. The
  study area is NOT a subset of the data extent any more (the outline
  extends ~13,400 m² beyond the bairro polygon — different source,
  independently digitized; see tests/test_mare_study_area.py).

Marcílio Dias, one of the 16 communities, lies ~2 km north of bairro Maré
and outside the outline, and is excluded from the study area. The exclusion
is computed by GEOMETRY (the community's own share of area inside the
ACTIVE study area, src.sites.territory.SUBUNIT_MEMBERSHIP_THRESHOLD = 0.5),
never by name — see src.sites.territory.load_territory.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from src.sites.territory import (  # noqa: F401
    ROOT_DEFAULT, SUBUNIT_MEMBERSHIP_THRESHOLD, load_sites_config, load_territory, within_mask,
)

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
# Not read from config/sites.yaml maré.study_area any more: that key only
# carries a `share_threshold` for the subunits_union_in_extent kind, and
# the active kind is polygon_file since 2026-09-24 (PI ruling). Membership
# of a subunit in the ACTIVE study area is now a generic property computed
# by src.sites.territory.load_territory for any kind — this constant is
# that same threshold, re-exported under its old name for callers that
# still read it directly (tests/test_mare_study_area.py).
INSIDE_EXTENT_MIN_SHARE = SUBUNIT_MEMBERSHIP_THRESHOLD


def load_data_extent(root: Path = ROOT) -> "shapely.Geometry":  # noqa: F821
    """The bairro polygon (data extent): buildings/DTM are clipped to this,
    never to the study area."""
    return load_territory("maré", root=root).data_extent


def load_communities(root: Path = ROOT) -> gpd.GeoDataFrame:
    """All 16 Redes da Maré communities, unfiltered (no `in_study_area` column)."""
    from src.sites.territory import load_sites_config, load_subunits
    cfg = load_sites_config()["maré"]
    comm, _ = load_subunits(cfg, root)
    return comm


def load_study_area(root: Path = ROOT) -> dict:
    """Build the ACTIVE study-area geometry (config/sites.yaml maré.study_area
    — the IPP Territórios Sociais outline since 2026-09-24) and the
    study-area-based in/out split of the 16 communities.

    Returns a dict with:
      communities  — all 16, with added columns `in_study_area` /
                     `share_in_study_area` (src.sites.territory.load_territory)
      included     — the (typically 15) communities inside the study area
      excluded     — the (typically 1, Marcílio Dias) communities outside it
      data_extent  — the bairro polygon geometry (shapely), unclipped
      study_area   — the active study area (shapely) — the IPP outline, not
                     a subset of data_extent (see module docstring)
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
