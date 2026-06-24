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
# Names (FINAL, user pre-authorized 2026-06-19). Two-axis scheme:
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
