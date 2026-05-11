#!/usr/bin/env python3
"""Figure 0.7 (proposition C) — Framework synthesis (graphical abstract).

Communicates the whole pipeline in one image. Designed to double as
the manuscript's graphical-abstract / closing figure.

Panels
------
A   Pipeline flow diagram. 5 input boxes → CFD synthesizer → 4-state
    classifier → diagnostic models → figure outputs.
B   Per-site headline-number grid. One column per site, rows show
    compound %, sun-deprived %, dominant predictor, climate flip %.
C   Four "what we learned" callout cards keyed to Figs 0.3 / 0.4 /
    0.5 / 0.6, each with the supporting number and a 1-line take-away.

Run:
    python docs/manuscript/figures/fig_0_7_proposition_framework.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
sys.path.insert(0, str(PROJECT_ROOT / "docs" / "manuscript" / "figures"))

from fig_style import (  # noqa: E402
    SITE_LABELS,
    SITE_ORDER,
    WIDTH_DOUBLE,
    apply_style,
)
from fig_0_4_diagnostic import (  # noqa: E402
    STATE_COLORS,
    TYPOLOGY_OF,
    build_diagnostic_grid,
    state_fractions,
)

EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

INPUT_COLOR = "#E8F1F8"
PROCESS_COLOR = "#FFF4E0"
OUTPUT_COLOR = "#F0E6F2"
EDGE_COLOR = "#7a7a7a"
ARROW_COLOR = "#888888"

INPUTS = [
    ("Building\nfootprints", "🏘"),
    ("Terrain\nDTM", "🗻"),
    ("Wind rose\n(INMET)", "🌬"),
    ("Solar\nray-cast", "☀"),
    ("Morphometry\n(SVF, λp …)", "📐"),
]
PROCESSES = [
    ("Synthetic /\ncampaign CFD"),
    ("4-state\nclassifier"),
    ("Diagnostic\nmodels"),
]
OUTPUTS = [
    "Fig 0.3 — patches",
    "Fig 0.4 — diagnosis *",
    "Fig 0.5 — predictors",
    "Fig 0.6 — climate",
]


# ---------------------------------------------------------------------------
# Panel A — pipeline flow
# ---------------------------------------------------------------------------
def _box(ax, xy, w, h, label, fc, ec=EDGE_COLOR, fontsize=6.5,
         fontweight="normal"):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.0,rounding_size=0.025",
                         facecolor=fc, edgecolor=ec, linewidth=0.6, zorder=3)
    ax.add_patch(box)
    cx = xy[0] + w / 2
    cy = xy[1] + h / 2
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color="#222", fontweight=fontweight, linespacing=1.15, zorder=4)


def _arrow(ax, p0, p1, lw=0.7):
    arr = FancyArrowPatch(p0, p1, arrowstyle="-|>",
                          mutation_scale=8, lw=lw, color=ARROW_COLOR,
                          zorder=2)
    ax.add_patch(arr)


def draw_panel_a(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("white")

    # Input column at x=0.02..0.16; 5 stacked boxes.
    input_x = 0.02
    input_w = 0.14
    input_h = 0.14
    input_xs = []
    input_centers = []
    for i, (label, _icon) in enumerate(INPUTS):
        y = 0.84 - i * 0.18
        _box(ax, (input_x, y), input_w, input_h, label,
             fc=INPUT_COLOR, fontsize=6, fontweight="normal")
        input_xs.append((input_x + input_w, y + input_h / 2))
        input_centers.append((input_x + input_w / 2, y + input_h / 2))

    # Process boxes at x=0.30..0.50, 0.55..0.70, 0.75..0.90.
    proc_w = 0.16
    proc_h = 0.30
    proc_y = 0.40
    proc_positions = []
    for i, label in enumerate(PROCESSES):
        x = 0.30 + i * 0.22
        _box(ax, (x, proc_y), proc_w, proc_h, label,
             fc=PROCESS_COLOR, fontsize=7, fontweight="bold")
        proc_positions.append((x, x + proc_w, proc_y + proc_h / 2))

    # Output column at x=0.95? No, let's put it underneath the processes.
    # Output boxes in a row at the bottom: y=0.05..0.18
    out_w = 0.20
    out_h = 0.13
    out_y = 0.06
    out_positions = []
    for i, label in enumerate(OUTPUTS):
        x = 0.10 + i * 0.22
        is_headline = "★" in label
        _box(ax, (x, out_y), out_w, out_h, label,
             fc=OUTPUT_COLOR, fontsize=6.0,
             fontweight="bold" if is_headline else "normal")
        out_positions.append((x + out_w / 2, out_y + out_h))

    # Arrows: inputs → first process (CFD or classifier depending on which).
    # Footprints + DTM → CFD ; Wind rose → CFD ; Solar → classifier ;
    # Morphometry → classifier + diagnostic models.
    routing = {
        0: 0,  # Footprints -> CFD
        1: 0,  # DTM -> CFD
        2: 0,  # Wind rose -> CFD
        3: 1,  # Solar -> classifier
        4: 1,  # Morphometry -> classifier
    }
    for i, (px, py) in enumerate(input_xs):
        target_x = proc_positions[routing[i]][0]
        target_cy = proc_positions[routing[i]][2]
        _arrow(ax, (px + 0.005, py), (target_x - 0.005, target_cy), lw=0.6)

    # Process chain: CFD → classifier → diagnostic models.
    for i in range(len(proc_positions) - 1):
        x0 = proc_positions[i][1]
        x1 = proc_positions[i + 1][0]
        cy = proc_positions[i][2]
        _arrow(ax, (x0 + 0.005, cy), (x1 - 0.005, cy), lw=1.0)

    # Arrows from process row to outputs.
    # Output 0 (Fig 0.3) ← classifier ; 1 (Fig 0.4 ★) ← classifier ;
    # 2 (Fig 0.5) ← diagnostic models ; 3 (Fig 0.6) ← classifier + climate scale.
    arrows_to_out = {0: 1, 1: 1, 2: 2, 3: 1}
    for i, (cx, cy) in enumerate(out_positions):
        src_proc = proc_positions[arrows_to_out[i]]
        src_x = (src_proc[0] + src_proc[1]) / 2
        src_y = proc_y
        _arrow(ax, (src_x, src_y - 0.005), (cx, cy + 0.005), lw=0.6)

    # An extra arrow from morphometry → diagnostic models to indicate
    # the predictor flow.
    ax.annotate("", xy=(proc_positions[2][0] - 0.005, proc_positions[2][2] - 0.10),
                xytext=(input_x + input_w + 0.005, input_centers[4][1] - 0.03),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=0.5,
                                linestyle="dashed", connectionstyle="arc3,rad=-0.3",
                                mutation_scale=7), zorder=2)

    # Headers
    ax.text(input_x + input_w / 2, 0.985, "INPUTS",
            ha="center", va="top", fontsize=6.5, fontweight="bold",
            color="#444")
    ax.text(0.51, 0.985, "PIPELINE",
            ha="center", va="top", fontsize=6.5, fontweight="bold",
            color="#444")
    ax.text(0.51, 0.245, "OUTPUTS  (manuscript figures)",
            ha="center", va="top", fontsize=6.5, fontweight="bold",
            color="#444")


# ---------------------------------------------------------------------------
# Panel B — headline numbers
# ---------------------------------------------------------------------------
def draw_panel_b(ax, headline: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    n = len(SITE_ORDER)
    col_w = 0.95 / n
    rows = [
        ("Typology", "typ"),
        ("Compound %", "compound"),
        ("Sun-fail %", "sun"),
        ("Vent-fail %", "vent"),
        ("Climate Δ %", "climate"),
    ]
    n_rows = len(rows)
    row_h = 0.78 / n_rows
    header_y = 0.92

    # Site headers (column titles).
    for i, site in enumerate(SITE_ORDER):
        cx = 0.025 + (i + 0.5) * col_w
        ax.text(cx, header_y, SITE_LABELS[site],
                ha="center", va="center", fontsize=6.5, fontweight="bold",
                color="#222")

    # Row labels (leftmost column).
    label_x = 0.018
    for r, (row_label, _) in enumerate(rows):
        ry = header_y - 0.08 - r * row_h
        ax.text(label_x, ry, row_label, ha="left", va="center",
                fontsize=6, color="#444")

    # Values.
    for i, site in enumerate(SITE_ORDER):
        cx = 0.025 + (i + 0.5) * col_w
        for r, (_, key) in enumerate(rows):
            ry = header_y - 0.08 - r * row_h
            v = headline.get(site, {}).get(key, None)
            if v is None:
                txt = "—"
                col = "#aaa"
            elif key == "typ":
                txt = v
                col = "#222"
            elif key == "compound":
                txt = f"{v:.1f}"
                col = STATE_COLORS["compound"]
            elif key == "sun":
                txt = f"{v:.1f}"
                col = STATE_COLORS["sun"]
            elif key == "vent":
                txt = f"{v:.1f}"
                col = STATE_COLORS["vent"]
            elif key == "climate":
                txt = f"+{v:.1f}"
                col = "#a0522d"
            else:
                txt = str(v); col = "#222"
            ax.text(cx, ry, txt, ha="center", va="center", fontsize=7,
                    fontweight="bold", color=col)

    # Faint row separators.
    for r in range(n_rows):
        ry = header_y - 0.08 - (r + 0.5) * row_h
        ax.plot([0.015, 0.985], [ry, ry], color="#eeeeee", lw=0.4, zorder=1)


# ---------------------------------------------------------------------------
# Panel C — callout cards
# ---------------------------------------------------------------------------
def draw_panel_c(ax, callouts: list[dict]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    n = len(callouts)
    pad = 0.012
    card_w = (1 - (n + 1) * pad) / n
    card_h = 0.88
    y = 0.06
    for i, c in enumerate(callouts):
        x = pad + i * (card_w + pad)
        _box(ax, (x, y), card_w, card_h, "",
             fc="#FAFAFA", ec="#cccccc", fontsize=6)
        # Header.
        ax.text(x + 0.012, y + card_h - 0.015,
                c["title"], ha="left", va="top",
                fontsize=7, fontweight="bold", color=c["color"])
        # Big number.
        ax.text(x + card_w / 2, y + card_h * 0.55,
                c["number"], ha="center", va="center",
                fontsize=18, fontweight="bold", color=c["color"])
        # Caption.
        ax.text(x + 0.012, y + 0.012, c["caption"],
                ha="left", va="bottom", fontsize=5.2, color="#444",
                linespacing=1.30)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    apply_style()
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("Building diagnostic grids for headline numbers ...")
    grids = {s: build_diagnostic_grid(s) for s in SITE_ORDER}

    # Headline numbers per site.
    headline: dict[str, dict] = {}
    for site in SITE_ORDER:
        g = grids[site]
        fr = state_fractions(g)
        n_cls = int(g["state"].notna().sum())
        if n_cls == 0:
            headline[site] = {"typ": TYPOLOGY_OF[site]}
            continue
        # Sun-fail = sun + compound. Vent-fail = vent + compound.
        sun_fail_pct = fr["sun"] + fr["compound"]
        vent_fail_pct = fr["vent"] + fr["compound"]
        # Climate Δ: % cells that flip to compound under −15 % U.
        u = g["annual_cfd_U_mean"]
        sun = g["solar_hours_winter"]
        cls = u.notna() & sun.notna()
        base = (u >= 1.0) & (sun >= 2.0)
        u15 = u * 0.85
        clim_compound = (~(u15 >= 1.0)) & (~(sun >= 2.0))
        # Cells that were adequate or sun-only, now compound.
        prior_safe = cls & ((u >= 1.0))
        flipped = prior_safe & clim_compound
        delta_pct = 100.0 * int(flipped.sum()) / max(int(cls.sum()), 1)
        headline[site] = {
            "typ": TYPOLOGY_OF[site],
            "compound": fr["compound"],
            "sun": sun_fail_pct,
            "vent": vent_fail_pct,
            "climate": delta_pct,
        }
        print(f"  {site:<22} compound={fr['compound']:.1f}%  "
              f"sun={sun_fail_pct:.1f}%  vent={vent_fail_pct:.1f}%  "
              f"climate_Δ={delta_pct:.1f}%")

    # Pooled compound% (excluding pending sites).
    comp_vals = [h["compound"] for h in headline.values()
                 if h.get("compound") is not None]
    avg_compound = float(np.mean(comp_vals)) if comp_vals else 0.0

    # Callout cards.
    callouts = [
        {
            "title": "Fig 0.3 — performance",
            "color": "#0072B2",
            "number": "4/5",
            "caption": ("4 representative\npatches mapped.\nMaré solar pending."),
        },
        {
            "title": "Fig 0.4 * — diagnosis",
            "color": STATE_COLORS["compound"],
            "number": f"{avg_compound:.0f}%",
            "caption": ("Mean compound-\nfailure across\nfavelas; clustered,\nnot random."),
        },
        {
            "title": "Fig 0.5 — mechanism",
            "color": "#a0522d",
            "number": "SVF",
            "caption": ("Dominant predictor\nfor vent + sun fail.\nChangepoint at\nSVF = 0.12."),
        },
        {
            "title": "Fig 0.6 — climate",
            "color": "#D55E00",
            "number": "-15%",
            "caption": ("Wind-stilling factor\npulls 4–10 % of\ncells across the\nLawson threshold."),
        },
    ]

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.85))
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[2.4, 1.0, 1.2],
        hspace=0.22,
        left=0.04, right=0.97, top=0.93, bottom=0.04,
    )

    ax_a = fig.add_subplot(outer[0])
    draw_panel_a(ax_a)

    ax_b = fig.add_subplot(outer[1])
    draw_panel_b(ax_b, headline)

    ax_c = fig.add_subplot(outer[2])
    draw_panel_c(ax_c, callouts)

    # Panel labels.
    for ax, label in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c")]:
        pos = ax.get_position()
        fig.text(pos.x0 - 0.012, pos.y1 + 0.005, label,
                 fontsize=9, fontweight="bold", va="bottom", ha="left")

    fig.text(0.5, 0.985,
             "Framework synthesis: inputs → CFD → 4-state classifier → diagnostic models → six manuscript figures",
             ha="center", va="top", fontsize=8.5, fontweight="bold",
             color="#1a1a1a")
    fig.text(0.5, 0.968,
             "PROPOSITION C — graphical-abstract synthesis. Reusable for cover letter; pairs with Figs 0.4 / 0.5 / 0.6 as the prescriptive package.",
             ha="center", va="top", fontsize=5.5, style="italic",
             color="#a0522d")

    out_png = EXPORTS_DIR / "fig_0_7_proposition_framework.png"
    out_svg = EXPORTS_DIR / "fig_0_7_proposition_framework.svg"
    print(f"Saving {out_png.name} + {out_svg.name} ...")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
