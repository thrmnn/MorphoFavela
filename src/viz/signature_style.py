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

# Provisional short descriptors (data-checked at gallery build; refine after review)
# Council-proposed names (2026-06-19), pending user sign-off. Spine:
# Footing → Plateau → Core (densification→enclosure) × a {flat, steep} slope switch.
TYPE_NAMES = {
    0: "T0 · Open Footing",            # sparse flat single-storey fringe (λp 0.22, SVF 0.65)
    1: "T1 · Stepped Footing",         # low-rise on ~19° slope (H/W from terrain)
    2: "T2 · Massing Plateau",         # flat consolidated mid-rise, sky intact (conditional)
    3: "T3 · Shaded Plateau",          # flat dense, daylight lost by frontal density (conditional)
    4: "T4 · Cliff Stack",             # steep dense hillside, H/W 2.6, fully sun-starved
    5: "T5 · Saturated Core",          # λp=1 maxed flat interior, deep-canyon 0.89, H/W 3.5
}


def type_cmap(k: int = 6) -> ListedColormap:
    """ListedColormap over morphotype 0..k-1 in the canonical palette order."""
    return ListedColormap([TYPE_COLORS[c] for c in range(k)])


def type_color_list(k: int = 6) -> list[str]:
    return [TYPE_COLORS[c] for c in range(k)]
