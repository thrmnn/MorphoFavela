"""Maré's annual ground irradiation as distributions against the city, not one
percentile: (1) Maré's density over the city's for two definitions of Maré,
(2, FOLHA4 round 4 — PI 2026-09-27: "some graphs don't make sense such as the
decile decomposition" — the decile-share bars this panel used to be are gone;
nothing else in this repo drew them, so they are deleted rather than kept
dead) each community's spread of citywide percentiles, ranked best to worst
by median. Per-community contrasts are staged for the PI and ethics-gated
before release.

`compute_distributions` (the numbers) and `draw_distributions` (the panels,
onto a caller-supplied figure + gridspec slot) are split out so a second
sheet can embed the identical analysis without re-deriving it — FOLHA4's
Maré site sheet (scripts/build_site_dashboard.py) calls both directly rather
than reading this module's own PNG or re-computing percentiles/deciles
itself. `main()` below calls the same functions into a standalone figure.

`compute_community_stats` is the one place per-community median citywide
percentile / cell count / rank are computed — panel 3 (ranked ordering) and
build_site_dashboard.py's community choropleth (FOLHA4 round 4) both call it
rather than deriving their own groupby, so the numbers on the map and the
numbers on the box plot are the same numbers by construction, not two
independently-computed answers that could drift apart.

    python scripts/render_mare_irradiation_distributions.py
"""

from __future__ import annotations

import math
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

    community_stats = compute_community_stats(
        comm["community"].tolist(), e["sub"].to_numpy(), e["p"].to_numpy())

    return dict(
        city=city, a=a, e=e, a_p=a_p, deciles=deciles,
        label_a=label_a, label_e=label_e, comm=comm, order=order,
        community_stats=community_stats,
        run_of_record=RUN_OF_RECORD["wp05"],
    )


def compute_community_stats(names: list, sub, p) -> list[dict]:
    """Per-community median citywide percentile, cell count and rank — the
    ONE place this is computed, so panel 3 (ranked box plots) and
    build_site_dashboard.py's community choropleth (FOLHA4 round 4) draw
    numbers that are identical by construction, never two independently
    -grouped answers. `sub`/`p` are per-ground-cell arrays (data['e']['sub']
    /data['e']['p']); `names` is every community's own name (order does not
    matter — the return is always sorted by rank, so a shuffled or
    duplicated `names` input reorders/duplicates the OUTPUT rows but never
    changes any single community's own median/n_cells/rank, see
    tests/test_render_mare_irradiation_distributions.py).

    A community with zero matching cells is never dropped the way
    compute_distributions()'s own `order` list silently drops one (real
    data: Marcílio Dias, excluded from the Maré study area by geometry —
    PI ruling 2026-09-24, config/sites.yaml's definition_note). It gets a
    row here too — `n_cells=0`, `median_percentile=None`, `rank=None` — so
    a caller that maps every row to a map/axis label still accounts for
    every named community instead of one silently vanishing. `number` is
    assigned 1..N by rank (ranked communities first, then any zero-cell
    ones), and is what the hero map / choropleth / panel 3 all key their
    numbering off of."""
    sub = np.asarray(sub, dtype=object)
    p = np.asarray(p, dtype=float)
    rows = []
    for name in names:
        mask = sub == name
        n = int(mask.sum())
        rows.append({
            "name": name, "n_cells": n,
            "median_percentile": float(np.median(p[mask])) if n else None,
        })
    ranked = sorted((r for r in rows if r["n_cells"] > 0), key=lambda r: -r["median_percentile"])
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    flagged = [r for r in rows if r["n_cells"] == 0]
    for r in flagged:
        r["rank"] = None
    out = ranked + flagged
    for i, r in enumerate(out, start=1):
        r["number"] = i
    return out


def community_color_limits(community_stats: list) -> tuple:
    """The choropleth's colour scale (FOLHA4 round 5, F2): every community
    median here is far below the city median (50), so the old fixed
    0..100 percentile scale left every polygon pale — the ramp's upper
    half was dead space no community ever reached. vmin/vmax are the data
    range of the community medians themselves, rounded OUTWARD to the
    nearest 5 (floor for vmin, ceil for vmax) so a small render-to-render
    jitter in the underlying WP-05 run can't flip a boundary community's
    fill by a hair — never typed constants, and never rounded inward
    (that would clip the extreme community's own colour). Marcílio Dias
    (n_cells=0, median_percentile=None) is excluded from the range the
    same way it is excluded from the ranked box plot — a null median
    cannot widen or narrow a numeric range.

    Proof this cannot silently freeze into a typed constant: TEST_RED in
    tests/test_render_mare_irradiation_distributions.py feeds two
    different median sets through this function and asserts the two
    outputs differ — a hardcoded return value fails that test by
    construction."""
    medians = [r["median_percentile"] for r in community_stats
               if r.get("median_percentile") is not None]
    if not medians:
        raise ValueError("community_color_limits: no community has a median_percentile")
    vmin = 5.0 * math.floor(min(medians) / 5.0)
    vmax = 5.0 * math.ceil(max(medians) / 5.0)
    if vmax <= vmin:
        vmax = vmin + 5.0
    return vmin, vmax


def draw_distributions_top(fig, spec, data: dict, panel_num: int = 1) -> tuple:
    """The city-vs-Maré histogram panel alone. Named `_top` (rather than
    renamed to match its now-single panel) so it keeps slotting into the
    same outer-gridspec row build_site_dashboard.py already gives it.
    FOLHA4 round 4 (PI 2026-09-27, "some graphs don't make sense such as
    the decile decomposition") removed the second, decile-share-bars axis
    this used to draw beside this panel — nothing else in this repo read
    it, so it is gone rather than kept dead. The colour scale this
    histogram's x-axis implies (0..deciles[-1]) is also what
    build_site_dashboard.py's hero map colours its street observers by, so
    map and histogram read as one (FOLHA4 round 4 F2).

    `panel_num` (round 5): the sheet numbers panels in reading order (hero
    map, choropleth, whole distributions, per-community spread) — a caller
    with a hero map and a choropleth ahead of this panel passes 3;
    standalone callers (this module's own main(), below) keep the default
    1."""
    city, a, e, deciles = data["city"], data["a"], data["e"], data["deciles"]
    label_a, label_e = data["label_a"], data["label_e"]

    ax = fig.add_subplot(spec)
    bins = np.linspace(0, deciles[-1], 90)
    mids = 0.5 * (bins[1:] + bins[:-1])
    ax.fill_between(mids, np.histogram(city, bins, density=True)[0], color=CITY, lw=0, label="Rio — all ground cells")
    for values, col, lab in [(a["kwh_m2"], DEF_A, label_a), (e["kwh_m2"], DEF_E, label_e)]:
        ax.plot(mids, np.histogram(values, bins, density=True)[0], color=col, lw=2, label=lab)
    for q in deciles[1:-1]:
        ax.axvline(q, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("annual irradiation at ground (kWh/m²·yr)")
    ax.set_yticks([])
    ax.set_title(f"{panel_num} · Whole distributions, not one number", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(deciles[1], ax.get_ylim()[1] * 0.97, "  city deciles", color=MUTED, fontsize=6.5, va="top")

    return (ax,)


def draw_distributions_bottom(fig, spec, data: dict, panel_num: int = 2) -> tuple:
    """The ranked box-plot panel alone: each community's spread of
    citywide percentiles, ordered by rank (best median first) rather than
    north-to-south (FOLHA4 round 4, PI 2026-09-27: ranking is the more
    legible ordering once the sheet also carries a ranked choropleth).
    Numbers on the x labels are `compute_community_stats`'s own `number`
    field — the same numbers the choropleth and (for the 15 communities it
    shows) the hero map carry. 'between communities' (ground inside the
    study area but no named community) is dropped from this ranked view —
    it is not a community and has no number to share with the map; it
    stays visible on the unranked standalone review PNG only through
    `data['order']`, which this function no longer reads.

    `panel_num` (round 5) — see draw_distributions_top's docstring; this
    panel is always the LAST on the page, so a caller passes it its
    draw_distributions_top `panel_num + 1` (build_folha4_mare's
    `folha4_mare_panel_numbers` names both together)."""
    community_stats = data["community_stats"]
    ranked = [r for r in community_stats if r["rank"] is not None]
    flagged = [r for r in community_stats if r["rank"] is None]
    rows = ranked + flagged
    e = data["e"]

    ax = fig.add_subplot(spec)
    box_positions, box_data = [], []
    for i, r in enumerate(rows, start=1):
        if r["n_cells"] > 0:
            box_positions.append(i)
            box_data.append(e.loc[e["sub"] == r["name"], "p"].to_numpy())
    if box_data:
        bp = ax.boxplot(box_data, positions=box_positions, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        for box in bp["boxes"]:
            box.set(facecolor=DEF_E, edgecolor=DEF_E, alpha=0.55)
    for i, r in enumerate(rows, start=1):
        if r["n_cells"] == 0:
            ax.text(i, 15, "no cells\n(excluded from\nstudy area)", ha="center", va="center",
                    fontsize=6, color=MUTED, style="italic")
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.text(0.55, 53, "city median", fontsize=6.5, color=MUTED, ha="left")
    ax.set_xticks(range(1, len(rows) + 1),
                 [f"{r['number']} {r['name']}\n(n={r['n_cells']:,})" for r in rows],
                 rotation=35, ha="right", fontsize=7)
    ax.set_xlim(0.3, len(rows) + 0.7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("citywide percentile of each ground cell")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title(f"{panel_num} · Each community's spread against the city — listed by rank, best median first",
                loc="left", fontsize=10, color=INK)

    return (ax,)


def draw_distributions(fig, spec, data: dict) -> tuple:
    """Combined convenience wrapper: both panels into one 2-row
    sub-gridspec carved out of `spec` (a SubplotSpec — pass
    `fig.add_gridspec(1, 1)[0, 0]` for "the whole figure", or a cell of a
    host sheet's own outer gridspec to embed). Caller is responsible for
    the `matplotlib.rc_context(DISTRIBUTIONS_RC)` wrapper (see module
    docstring) — this function only draws."""
    gs = spec.subgridspec(2, 1, height_ratios=[1, 1.25], hspace=0.42)
    ax1, = draw_distributions_top(fig, gs[0, 0], data)
    ax3, = draw_distributions_bottom(fig, gs[1, 0], data)
    return ax1, ax3


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
