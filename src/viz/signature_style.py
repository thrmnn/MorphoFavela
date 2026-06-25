"""Shared visual language for the morpho-signature figures.

One palette system, applied everywhere (the spine of the viz plan):
- categorical morphotype hues = Okabe–Ito, colourblind-safe, saturated mid-tones,
  used on points/polygons/dendrogram leaves;
- NULL/no-support = neutral grey, reserved (no type is grey);
- sequential (prioritization) and diverging (z-scores, deltas) ramps pass through
  white inside heatmap fills, so the three colour roles never collide.

Decisions: docs/visualization_plan.md ("Decisions from expert review").
"""

from __future__ import annotations

from matplotlib.colors import ListedColormap

# Okabe–Ito, assigned to the lambda_p type order (T0 sparse → T5 dense).
TYPE_COLORS = {
    0: "#56B4E9",  # sky blue
    1: "#009E73",  # bluish green
    2: "#E69F00",  # orange
    3: "#F0E442",  # yellow  (flatland-only type)
    4: "#D55E00",  # vermillion
    5: "#CC79A7",  # reddish purple
}
NULL_COLOR = "#E0E0E0"        # no street support / unbuilt — reserved grey
SEQUENTIAL = "YlOrBr"         # prioritization intensity (a hue no type owns)
DIVERGING = "RdBu_r"          # z-scores / flat-vs-terrain delta, centred at 0

# Names RE-DERIVED 2026-06-25 after the dissolved-λf re-baseline (k=6 on the
# party-wall-corrected frontal density; ARI 0.23 vs the summed-λf typology, so the
# membership re-sorted). Same two-axis scheme: density ladder Fringe →
# Consolidated → Core × terrain/condition modifier. Stats are the new fabric +
# experience centroids. The two flatland types (1, 5) are now the conditional pair.
TYPE_NAMES = {
    0: "T0 · Open Fringe",          # open low-rise edge: λp 0.10, slope 5°, SVF 0.66, sun 7h
    1: "T1 · Flatland Consolidated",  # flat (1°) mid-density mixed-rise: λp 0.57, H 7.4 (flatland-specific)
    2: "T2 · Hillside Fringe",      # steep (20°) low-rise: λp 0.59, H 4.9, canyon from terrain
    3: "T3 · Shaded Consolidated",  # steep (17°) tall dense: λp 0.65, λf 1.42, H/W 2.6, 0 h sun
    4: "T4 · Hillside Core",        # steep (18°) maximal-coverage: λp 0.90, sun-starved
    5: "T5 · Saturated Core",       # flat (1°) λp 1.0 interior: H 7.6, H/W 1.5 (flatland-specific)
}
# Short labels (no "T# ·" prefix) for compact figures.
TYPE_LABEL = {c: n.split(" · ")[1] for c, n in TYPE_NAMES.items()}

# Block-scale MORPHOTOPE (tissue) names — deliberately a DISTINCT vocabulary from the
# cell-level MORPHOTYPE names (no Fringe/Consolidated/Core reuse) so the two levels are
# never confused. Tissues are areas (M0–M4); morphotypes are the cell alphabet (T0–T5).
# Ordered by ascending Saturated-Core cell share.
MORPHOTOPE_LABEL = {
    0: "Low-rise Slope Tissue",   # 87% T1 cells, uniform low fabric on slope
    1: "Mixed Flatland Tissue",   # diverse flat mix (T0/T3/T5), highest diversity
    2: "Compact Hillside Tissue", # 46% T4 + 30% T5 cells, recurs in 4 sites
    3: "Mixed Dense Tissue",      # heterogeneous dense (T5/T4/T3)
    4: "Compact Flatland Tissue", # 53% T3 + 46% T5 cells, flatland-specific
}


def type_cmap(k: int = 6) -> ListedColormap:
    """ListedColormap over morphotype 0..k-1 in the canonical palette order."""
    return ListedColormap([TYPE_COLORS[c] for c in range(k)])


def type_color_list(k: int = 6) -> list[str]:
    return [TYPE_COLORS[c] for c in range(k)]
