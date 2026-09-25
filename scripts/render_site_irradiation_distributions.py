"""A site's annual ground irradiation as distributions against the city —
the single-definition sibling of render_mare_irradiation_distributions.py.

Maré gets its own module (kept untouched by this one) because it is the
only P1 site with a contested pair of boundary definitions (A = 6 IPP
favela polygons vs E = IPP complex outline; PI ruling, resolved_decisions
id mare_site_study_area) — panel 1/2 there plot both. Every other site's
P1 definition is just its config/sites.yaml citywide_rule match, singular,
so this module draws one line/one bar series, not a pair.

Two or three panels, decided by whether the site has real subunits
(territory.subunits is not None — today: Complexo do Alemão, 15 favela
parts; Rio das Pedras, 2): (1) city-vs-site histogram, (2) the share of
the site's ground in each citywide decile, and, only when subunits exist,
(3) each subunit's spread of citywide percentiles, listed north to south,
via src.sites.territory.label_subunits (the same generic helper Territory
itself exposes, not a re-derivation). Vidigal and Rocinha (each a single
"Isolada" polygon, no subunits) get no panel 3 here — build_site_dashboard.py
keeps them on the v3 grid-row sheet instead of this FOLHA4 layout, since a
one-box "boxplot" would tell the reader nothing panel 1 doesn't already.

`compute_distributions` (the numbers) and `draw_distributions_top` /
`draw_distributions_bottom` (the panels, onto a caller-supplied figure +
gridspec slot) split the same way Maré's module does, for the same reason:
scripts/build_site_dashboard.py's FOLHA4 sheet embeds these directly.

    python scripts/render_site_irradiation_distributions.py --site complexo_do_alemao
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))
ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")

from src.brisa_solar.wp05_full import match_favela_group  # noqa: E402
from src.brisa_solar.wp07_ledger import RUN_OF_RECORD  # noqa: E402
from src.sites.territory import (  # noqa: E402
    BETWEEN_SUBUNITS_LABEL,
    label_subunits,
    load_sites_config,
    load_territory,
    normalize_site_key,
)

# Same palette as render_mare_irradiation_distributions.py — one source
# would be cleaner, but that module is under locked test contracts
# (docs/critic/folha4_council_2026-09-24.md round 3) and this one is not,
# so the constants are duplicated rather than risking an import coupling
# that makes a future Maré-only edit ripple into every other site's sheet.
INK, MUTED, GRID, CITY, DEF_A = "#1b1b1b", "#6b6b66", "#e4e3dc", "#c9c8c0", "#2a78d6"
BETWEEN = BETWEEN_SUBUNITS_LABEL

DISTRIBUTIONS_RC = {
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
}


def compute_distributions(site: str, root: Path = ROOT) -> dict:
    """All numbers behind panels 1-3, and nothing else — no drawing."""
    site = normalize_site_key(site)
    territory = load_territory(site, root=root)
    cfg = load_sites_config()[site]

    df = pd.read_parquet(root / "runs" / RUN_OF_RECORD["wp05"] / "wp05_full.parquet",
                         columns=["x", "y", "favela_id", "kwh_m2"])
    city = np.sort(df["kwh_m2"].to_numpy())
    city = city[np.isfinite(city)]

    def pct(v):
        return 100.0 * np.searchsorted(city, v, side="right") / len(city)

    polys, method = match_favela_group(
        gpd.read_file(root / "data" / "RJ" / "Favelas_Limit_2019.shp"),
        cfg["citywide_rule"]["target"])
    a = df[df["favela_id"].isin(polys["cod_favela"].astype(int))].copy()
    a["p"] = pct(a["kwh_m2"].to_numpy())

    order = None
    if territory.subunits is not None:
        a["sub"] = label_subunits(a["x"].to_numpy(), a["y"].to_numpy(), territory)
        # Points a study-area-membership epsilon outside territory.subunits'
        # source geometry (label_subunits checks against territory.study_area,
        # which for these sites is the same union(citywide) `a` is already
        # filtered to, so this is a handful of edge cells at most) get None
        # and are simply left out of panel 3 — still counted in panels 1/2.
        present = a.loc[a["sub"].notna()]
        name_col = "name" if "name" in territory.subunits.columns else territory.subunits.columns[0]
        by_lat = territory.subunits.assign(cy=territory.subunits.geometry.centroid.y)
        north_to_south = by_lat.sort_values("cy", ascending=False)[name_col]
        order = [c for c in north_to_south if (present["sub"] == c).any()]
        if (present["sub"] == BETWEEN).any():
            order = order + [BETWEEN]

    deciles = np.quantile(city, np.linspace(0, 1, 11))
    label_a = f"{territory.display_name} — {len(polys)} IPP favela polygon(s), P1 definition ({method})"

    return dict(
        site=site, display_name=territory.display_name,
        city=city, a=a, a_p=a["p"].to_numpy(), deciles=deciles,
        label_a=label_a, order=order,
        run_of_record=RUN_OF_RECORD["wp05"],
    )


def draw_distributions_top(fig, spec, data: dict) -> tuple:
    """Panels 1 & 2, single-definition versions of
    render_mare_irradiation_distributions.draw_distributions_top."""
    city, a, a_p, deciles = data["city"], data["a"], data["a_p"], data["deciles"]
    label_a, display_name = data["label_a"], data["display_name"]

    gs = spec.subgridspec(1, 2, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, deciles[-1], 90)
    mids = 0.5 * (bins[1:] + bins[:-1])
    ax.fill_between(mids, np.histogram(city, bins, density=True)[0], color=CITY, lw=0,
                    label="Rio — all ground cells")
    ax.plot(mids, np.histogram(a["kwh_m2"], bins, density=True)[0], color=DEF_A, lw=2, label=label_a)
    for q in deciles[1:-1]:
        ax.axvline(q, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("annual irradiation at ground (kWh/m²·yr)")
    ax.set_yticks([])
    ax.set_title("1 · Whole distributions, not one number", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(deciles[1], ax.get_ylim()[1] * 0.97, "  city deciles", color=MUTED, fontsize=6.5, va="top")
    ax1 = ax

    ax = fig.add_subplot(gs[0, 1])
    idx = np.arange(10)
    share = np.histogram(a_p, np.linspace(0, 100, 11))[0] / len(a_p) * 100
    ax.bar(idx, share, 0.55, color=DEF_A, label=label_a)
    ax.text(idx[0], share[0] + 1, f"{share[0]:.0f}%", ha="center", fontsize=7, color=INK)
    ax.axhline(10, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.text(9.6, 10.6, "city = 10% in each", ha="right", fontsize=6.5, color=MUTED)
    ax.set_xticks(idx, [f"D{i + 1}" for i in idx])
    ax.set_xlabel("citywide irradiation decile  (D1 = darkest tenth of the city's ground)")
    ax.set_ylabel(f"share of {display_name}'s ground cells (%)")
    # Fixed headroom floor, not matplotlib's default ~5% autoscale margin:
    # a low-peak site (Alemão's deciles top out near 16%) left almost no
    # gap between the top y-tick label and the top spine, so the title
    # (loc="left", default pad) sat on the same row as that tick label and
    # visually overlapped it — confirmed against Maré/Rio das Pedras (both
    # much higher peaks, so autoscale alone gave them enough headroom) in
    # the round-2 council. Deriving headroom from this panel's own data
    # (never a constant tuned to one site) plus a fixed title pad fixes it
    # for every site's own peak, not just the ones already tall enough.
    ax.set_ylim(0, max(float(share.max()) * 1.15 + 4.0, 20.0))
    ax.set_title(f"2 · Where {display_name}'s ground falls among the city's deciles",
                loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax2 = ax

    return ax1, ax2


def draw_distributions_bottom(fig, spec, data: dict):
    """Panel 3, only when the site has subunits to break it down by
    (`data['order']` is None otherwise). Returns None without drawing
    anything when there is nothing to draw — caller decides whether to
    still reserve the gridspec row (build_site_dashboard.py's
    build_folha4_site does not call this at all for such a site)."""
    order = data["order"]
    if not order:
        return None
    a = data["a"]
    n = len(order)
    # A category count-scaled, centered sub-axes instead of always taking
    # the row's full physical width: at full width a low subunit count
    # (Rio das Pedras: 2) left its two boxes "stranded" with roughly half
    # the sheet blank on either side (round-2 council, blocking) because
    # matplotlib's own autoscale margin is a fixed *fraction* of the data
    # range regardless of how few categories that range spans. frac is
    # derived from this panel's own category count (never a constant tuned
    # to one site) and capped below 1.0 even at the high end — Alemão's 16
    # categories (15 + "between") span the same physical row as before,
    # just with a hairline margin instead of edge-to-edge, which is also
    # what stopped its leftmost rotated tick label ("Rua Armando Sodré")
    # from clipping against the raw canvas edge (round-2 council).
    frac = float(np.clip(n / 10.0, 0.30, 0.97))
    side = (1.0 - frac) / 2.0
    sub = spec.subgridspec(1, 3, width_ratios=[side, frac, side], wspace=0.0)
    ax = fig.add_subplot(sub[0, 1])
    data_by_order = [a.loc[a["sub"] == o, "p"].to_numpy() for o in order]
    bp = ax.boxplot(data_by_order, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED))
    for box, o in zip(bp["boxes"], order):
        between = o == BETWEEN
        box.set(facecolor="#f2f1ec" if between else DEF_A, edgecolor=MUTED if between else DEF_A, alpha=0.55)
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.text(len(order) + 0.45, 51, "city median", fontsize=6.5, color=MUTED, ha="right")
    ax.set_xticks(range(1, len(order) + 1), [f"{o}\n(n={len(d):,})" for o, d in zip(order, data_by_order)],
                 rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("citywide percentile of each ground cell")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("3 · Each part's spread against the city — listed north to south, not ranked",
                loc="left", fontsize=10, color=INK)
    return (ax,)


def provenance_note(data: dict) -> str:
    return (f"WP-05 run of record {data['run_of_record']} · ground lattice · "
            "decile = one tenth of the citywide distribution, darkest to brightest · "
            "box = interquartile range (the middle 50% of values), whiskers = 1.5× that range · "
            "staged for PI review, ethics-gate before release")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True)
    args = p.parse_args()

    data = compute_distributions(args.site, ROOT)
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams.update(DISTRIBUTIONS_RC)
        has_panel3 = data["order"] is not None
        fig = plt.figure(figsize=(13, 9.2 if has_panel3 else 6.0), dpi=200)
        gs = fig.add_gridspec(2 if has_panel3 else 1, 1,
                              height_ratios=[1, 1.25] if has_panel3 else [1], hspace=0.42)
        draw_distributions_top(fig, gs[0, 0], data)
        if has_panel3:
            draw_distributions_bottom(fig, gs[1, 0], data)
        fig.text(0.01, 0.005, provenance_note(data), fontsize=6.5, color=MUTED)
        out = ROOT / "outputs" / data["site"] / "territory" / "site_irradiation_distributions.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
