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
TYPE_NAMES = {
    0: "T0 · sparse, tall-scattered",
    1: "T1 · open low-rise",
    2: "T2 · uniform open low-rise",
    3: "T3 · flatland mid-rise (site-specific)",
    4: "T4 · dense steep hillside",
    5: "T5 · densest, deep-canyon",
}


def type_cmap(k: int = 6) -> ListedColormap:
    """ListedColormap over morphotype 0..k-1 in the canonical palette order."""
    return ListedColormap([TYPE_COLORS[c] for c in range(k)])


def type_color_list(k: int = 6) -> list[str]:
    return [TYPE_COLORS[c] for c in range(k)]
