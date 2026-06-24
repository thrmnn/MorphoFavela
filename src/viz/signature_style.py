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
# Names (refined 2026-06-19, pending user sign-off). Two-axis scheme:
# density ladder  Fringe → Consolidated → Core  ×  condition modifier
# (Open / Hillside / Shaded / Saturated). Density rises T0→T5; T1,T4 are the steep pair.
TYPE_NAMES = {
    0: "T0 · Open Fringe",          # sparse flat single-storey edge (λp 0.22, SVF 0.65, sun 7h)
    1: "T1 · Hillside Fringe",      # low-rise on ~19° slope; canyon from terrain, not mass
    2: "T2 · Open Consolidated",    # flat, dense, sky still intact (SVF 0.46; conditional)
    3: "T3 · Shaded Consolidated",  # flat, dense, daylight lost to frontal density (conditional)
    4: "T4 · Hillside Core",        # steep dense hillside, H/W 2.6, fully sun-starved
    5: "T5 · Saturated Core",       # λp=1 maxed flat interior, deep-canyon 0.89, H/W 3.5
}
# Short labels (no "T# ·" prefix) for compact figures.
TYPE_LABEL = {c: n.split(" · ")[1] for c, n in TYPE_NAMES.items()}

# Block-scale morphotope (tissue) names — from the cell-type composition of each
# (2026-06-19, pending user sign-off). M0→M4 ordered by ascending Saturated-Core share.
MORPHOTOPE_LABEL = {
    0: "Stepped Hillside",      # 87% T1 (Hillside Fringe), uniform low slope fabric
    1: "Mixed Flatland",        # T0/T3/T5 mix, highest diversity
    2: "Dense Hillside Core",   # 46% T4 + 30% T5, recurs in 4 sites
    3: "Transitional Dense",    # mixed dense (T5/T4/T3), most heterogeneous
    4: "Flat Dense Core",       # 53% T3 + 46% T5, flatland-specific
}


def type_cmap(k: int = 6) -> ListedColormap:
    """ListedColormap over morphotype 0..k-1 in the canonical palette order."""
    return ListedColormap([TYPE_COLORS[c] for c in range(k)])


def type_color_list(k: int = 6) -> list[str]:
    return [TYPE_COLORS[c] for c in range(k)]
