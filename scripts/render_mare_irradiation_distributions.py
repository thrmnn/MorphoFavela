"""Maré's annual ground irradiation as distributions against the city, not one
percentile: (1) Maré's density over the city's for two definitions of Maré,
(2) the share of Maré's ground in each citywide decile, (3) each community's
spread of citywide percentiles, listed north to south — never ranked.
Per-community contrasts are staged for the PI and ethics-gated before release.

`compute_distributions` (the numbers) and `draw_distributions` (the three
panels, onto a caller-supplied figure + gridspec slot) are split out so a
second sheet can embed the identical analysis without re-deriving it —
FOLHA4's Maré site sheet (scripts/build_site_dashboard.py) calls both
directly rather than reading this module's own PNG or re-computing
percentiles/deciles itself. `main()` below is unchanged in output: it calls
the same two functions into a standalone figure.

    python scripts/render_mare_irradiation_distributions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# sys.path uses the RUNNING checkout's own root (may be a worktree ahead of
# ROOT); ROOT itself stays hardcoded at the main checkout for data/outputs —
# same split as scripts/build_site_dashboard.py, which imports this module
# directly and needs the two to agree when it runs from a worktree.
_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))
ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")

from src.brisa_solar.wp05_full import match_favela_group  # noqa: E402
from src.brisa_solar.wp07_ledger import RUN_OF_RECORD  # noqa: E402

OUT = ROOT / "outputs" / "maré" / "territory" / "mare_irradiation_distributions.png"
OUTLINE = ROOT / "data" / "maré" / "raw" / "ipp_territorios_sociais_territorio03.gpkg"
COMMUNITIES = ROOT / "data" / "maré" / "neighbourhoods.gpkg"
BETWEEN = "between communities"
INK, MUTED, GRID, CITY, DEF_A, DEF_E = "#1b1b1b", "#6b6b66", "#e4e3dc", "#c9c8c0", "#2a78d6", "#eb6834"

# Local style for these three panels — deliberately different from a host
# sheet's own rcParams (e.g. build_site_dashboard.py's INK/PAPER masthead
# palette): every caller applies it via `matplotlib.rc_context` around the
# `draw_distributions` call, never by mutating global rcParams, so the panels
# render identically standalone or embedded and never leak into a
# neighbouring panel's style.
DISTRIBUTIONS_RC = {
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
}


def _within(df: pd.DataFrame, geom) -> pd.DataFrame:
    x0, y0, x1, y1 = geom.bounds
    box = df[(df.x >= x0) & (df.x <= x1) & (df.y >= y0) & (df.y <= y1)]
    return box[shapely.contains_xy(geom, box.x.to_numpy(), box.y.to_numpy())].copy()


def compute_distributions(root: Path = ROOT) -> dict:
    """All numbers behind the three panels, and nothing else — no drawing.
    Single source for both this module's own PNG and any embedding caller;
    percentiles/deciles/community assignment are computed exactly once."""
    df = pd.read_parquet(root / "runs" / RUN_OF_RECORD["wp05"] / "wp05_full.parquet",
                         columns=["x", "y", "favela_id", "kwh_m2"])
    city = np.sort(df["kwh_m2"].to_numpy())
    city = city[np.isfinite(city)]

    def pct(v):
        return 100.0 * np.searchsorted(city, v, side="right") / len(city)

    polys, _ = match_favela_group(gpd.read_file(root / "data" / "RJ" / "Favelas_Limit_2019.shp"), "Maré")
    a = df[df["favela_id"].isin(polys["cod_favela"].astype(int))]
    e = _within(df, gpd.read_file(root / "data" / "maré" / "raw" / "ipp_territorios_sociais_territorio03.gpkg")
                .to_crs(31983).union_all())
    comm = gpd.read_file(root / "data" / "maré" / "neighbourhoods.gpkg", layer="communities")
    e["sub"] = BETWEEN
    for _, r in comm.iterrows():
        e.loc[shapely.contains_xy(r.geometry, e.x.to_numpy(), e.y.to_numpy()), "sub"] = r["community"]
    e["p"] = pct(e["kwh_m2"].to_numpy())
    a_p = pct(a["kwh_m2"].to_numpy())
    deciles = np.quantile(city, np.linspace(0, 1, 11))
    label_a = f"Maré — {len(polys)} IPP favela polygons (P1 f1, definition A)"
    label_e = "Maré — IPP complex outline (site sheet, definition E)"
    north_to_south = comm.assign(cy=comm.geometry.centroid.y).sort_values("cy", ascending=False)["community"]
    order = [c for c in north_to_south if (e["sub"] == c).any()] + [BETWEEN]

    return dict(
        city=city, a=a, e=e, a_p=a_p, deciles=deciles,
        label_a=label_a, label_e=label_e, comm=comm, order=order,
        run_of_record=RUN_OF_RECORD["wp05"],
    )


def draw_distributions_top(fig, spec, data: dict) -> tuple:
    """Panels 1 & 2 (city-vs-Maré histogram, decile-share bars) side by
    side. Split out from `draw_distributions` (FOLHA4 round 3) because a
    single hspace fraction inside one shared sub-gridspec scales with
    whatever outer-cell height the caller gives it — fine for
    `draw_distributions`'s own equal-height callers, but FOLHA4's hero and
    no-hero variants give the distributions block very different outer
    heights (~7.6 vs ~11.3 page-ratio units), so a fixed *fraction* gap
    ballooned into ~250px of dead space in no-hero while staying tight in
    hero (round-2 council finding). Returning top/bottom as separate specs
    lets the host sheet place a fixed-*ratio* spacer row between them on
    its own outer gridspec — the same absolute-gap-regardless-of-variant
    trick already used for the gap before the caveat strip
    (SPACER_BEFORE_CAVEATS in build_site_dashboard.py)."""
    city, a, e, a_p, deciles = data["city"], data["a"], data["e"], data["a_p"], data["deciles"]
    label_a, label_e = data["label_a"], data["label_e"]

    gs = spec.subgridspec(1, 2, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, deciles[-1], 90)
    mids = 0.5 * (bins[1:] + bins[:-1])
    ax.fill_between(mids, np.histogram(city, bins, density=True)[0], color=CITY, lw=0, label="Rio — all ground cells")
    for values, col, lab in [(a["kwh_m2"], DEF_A, label_a), (e["kwh_m2"], DEF_E, label_e)]:
        ax.plot(mids, np.histogram(values, bins, density=True)[0], color=col, lw=2, label=lab)
    for q in deciles[1:-1]:
        ax.axvline(q, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("annual irradiation at ground (kWh/m²·yr)")
    ax.set_yticks([])
    ax.set_title("1 · Whole distributions, not one number", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(deciles[1], ax.get_ylim()[1] * 0.97, "  city deciles", color=MUTED, fontsize=6.5, va="top")
    ax1 = ax

    ax = fig.add_subplot(gs[0, 1])
    idx, w = np.arange(10), 0.38
    for off, values, col, lab in [(-w / 2, a_p, DEF_A, "A · IPP favela polygons"),
                                   (w / 2, e["p"].to_numpy(), DEF_E, "E · IPP complex outline")]:
        share = np.histogram(values, np.linspace(0, 100, 11))[0] / len(values) * 100
        ax.bar(idx + off, share, w - 0.04, color=col, label=lab)
        ax.text(idx[0] + off, share[0] + 1, f"{share[0]:.0f}%", ha="center", fontsize=7, color=INK)
    ax.axhline(10, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.text(9.6, 10.6, "city = 10% in each", ha="right", fontsize=6.5, color=MUTED)
    ax.set_xticks(idx, [f"D{i + 1}" for i in idx])
    ax.set_xlabel("citywide irradiation decile  (D1 = darkest tenth of the city's ground)")
    ax.set_ylabel("share of Maré's ground cells (%)")
    ax.set_title("2 · Where Maré's ground falls among the city's deciles", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax2 = ax

    return ax1, ax2


def draw_distributions_bottom(fig, spec, data: dict) -> tuple:
    """Panel 3 alone (each community's spread of citywide percentiles,
    listed north to south). Split out from `draw_distributions` — see
    `draw_distributions_top`'s docstring for why."""
    e, order = data["e"], data["order"]

    ax = fig.add_subplot(spec)
    data_by_order = [e.loc[e["sub"] == o, "p"].to_numpy() for o in order]
    bp = ax.boxplot(data_by_order, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for box, o in zip(bp["boxes"], order):
        between = o == BETWEEN
        box.set(facecolor="#f2f1ec" if between else DEF_E, edgecolor=MUTED if between else DEF_E, alpha=0.55)
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.text(len(order) + 0.45, 51, "city median", fontsize=6.5, color=MUTED, ha="right")
    ax.set_xticks(range(1, len(order) + 1), [f"{o}\n(n={len(d):,})" for o, d in zip(order, data_by_order)],
                 rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("citywide percentile of each ground cell")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("3 · Each community's spread against the city — listed north to south, not ranked",
                loc="left", fontsize=10, color=INK)
    ax3 = ax

    return (ax3,)


def draw_distributions(fig, spec, data: dict) -> tuple:
    """Combined convenience wrapper: all three panels into one 2-row
    sub-gridspec carved out of `spec` (a SubplotSpec — pass
    `fig.add_gridspec(1, 1)[0, 0]` for "the whole figure", or a cell of a
    host sheet's own outer gridspec to embed). Caller is responsible for
    the `matplotlib.rc_context(DISTRIBUTIONS_RC)` wrapper (see module
    docstring) — this function only draws. Safe here because this
    wrapper's two callers (this module's own `main()` and, historically,
    FOLHA4 round 1/2) always give it one fixed-height figure — the
    variant-height mismatch that broke a shared hspace fraction only
    shows up when a host sheet's hero/no-hero variants hand the block
    very different outer heights, which is why FOLHA4 now calls
    `draw_distributions_top`/`draw_distributions_bottom` directly instead
    of this wrapper (see build_site_dashboard.py)."""
    gs = spec.subgridspec(2, 1, height_ratios=[1, 1.25], hspace=0.42)
    ax1, ax2 = draw_distributions_top(fig, gs[0, 0], data)
    ax3, = draw_distributions_bottom(fig, gs[1, 0], data)
    return ax1, ax2, ax3


def provenance_note(data: dict) -> str:
    """One plain-English line: what produced these numbers, a working
    glossary for the stats terms panels 1-3 use with no other gloss on
    the page (decile, IQR, length-weighted — round-2/3 council finding:
    a reader without a stats background has nothing to go on for these,
    print has no hover the way the interactive twin's <dfn> tooltips do),
    and their release status — no code-level identifiers beyond the run
    id itself, which is a citable artefact name, not an internal
    algorithm parameter."""
    return (f"WP-05 run of record {data['run_of_record']} · ground lattice · "
            "decile = one tenth of the citywide distribution, darkest to brightest · "
            "box = interquartile range (the middle 50% of values), whiskers = 1.5× that range · "
            "length-weighted = each road segment counted by its length, not as one point · "
            "staged for PI review, ethics-gate before release")


def main() -> int:
    data = compute_distributions(ROOT)
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams.update(DISTRIBUTIONS_RC)
        fig = plt.figure(figsize=(13, 9.2), dpi=200)
        spec = fig.add_gridspec(1, 1)[0, 0]
        draw_distributions(fig, spec, data)
        fig.text(0.01, 0.005, provenance_note(data), fontsize=6.5, color=MUTED)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
