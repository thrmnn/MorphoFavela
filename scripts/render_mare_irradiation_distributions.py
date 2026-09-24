"""Maré's annual ground irradiation as distributions against the city, not one
percentile: (1) Maré's density over the city's for two definitions of Maré,
(2) the share of Maré's ground in each citywide decile, (3) each community's
spread of citywide percentiles, listed north to south — never ranked.
Per-community contrasts are staged for the PI and ethics-gated before release.

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brisa_solar.wp05_full import match_favela_group  # noqa: E402
from src.brisa_solar.wp07_ledger import RUN_OF_RECORD  # noqa: E402

OUT = ROOT / "outputs" / "maré" / "territory" / "mare_irradiation_distributions.png"
OUTLINE = ROOT / "data" / "maré" / "raw" / "ipp_territorios_sociais_territorio03.gpkg"
COMMUNITIES = ROOT / "data" / "maré" / "neighbourhoods.gpkg"
BETWEEN = "between communities"
INK, MUTED, GRID, CITY, DEF_A, DEF_E = "#1b1b1b", "#6b6b66", "#e4e3dc", "#c9c8c0", "#2a78d6", "#eb6834"


def _within(df: pd.DataFrame, geom) -> pd.DataFrame:
    x0, y0, x1, y1 = geom.bounds
    box = df[(df.x >= x0) & (df.x <= x1) & (df.y >= y0) & (df.y <= y1)]
    return box[shapely.contains_xy(geom, box.x.to_numpy(), box.y.to_numpy())].copy()


def main() -> int:
    df = pd.read_parquet(ROOT / "runs" / RUN_OF_RECORD["wp05"] / "wp05_full.parquet",
                         columns=["x", "y", "favela_id", "kwh_m2"])
    city = np.sort(df["kwh_m2"].to_numpy())
    city = city[np.isfinite(city)]

    def pct(v):
        return 100.0 * np.searchsorted(city, v, side="right") / len(city)

    polys, _ = match_favela_group(gpd.read_file(ROOT / "data" / "RJ" / "Favelas_Limit_2019.shp"), "Maré")
    a = df[df["favela_id"].isin(polys["cod_favela"].astype(int))]
    e = _within(df, gpd.read_file(OUTLINE).to_crs(31983).union_all())
    comm = gpd.read_file(COMMUNITIES, layer="communities")
    e["sub"] = BETWEEN
    for _, r in comm.iterrows():
        e.loc[shapely.contains_xy(r.geometry, e.x.to_numpy(), e.y.to_numpy()), "sub"] = r["community"]
    e["p"] = pct(e["kwh_m2"].to_numpy())
    a_p = pct(a["kwh_m2"].to_numpy())
    deciles = np.quantile(city, np.linspace(0, 1, 11))
    label_a = f"Maré — {len(polys)} IPP favela polygons (P1 f1, definition A)"
    label_e = "Maré — IPP complex outline (site sheet, definition E)"

    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                             "xtick.color": MUTED, "ytick.color": MUTED,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig = plt.figure(figsize=(13, 9.2), dpi=200)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.42, wspace=0.22)

        ax = fig.add_subplot(gs[0, 0])
        bins = np.linspace(0, deciles[-1], 90)
        mids = 0.5 * (bins[1:] + bins[:-1])
        ax.fill_between(mids, np.histogram(city, bins, density=True)[0], color=CITY, lw=0, label="Rio — all ground cells")
        for data, col, lab in [(a["kwh_m2"], DEF_A, label_a), (e["kwh_m2"], DEF_E, label_e)]:
            ax.plot(mids, np.histogram(data, bins, density=True)[0], color=col, lw=2, label=lab)
        for q in deciles[1:-1]:
            ax.axvline(q, color=GRID, lw=0.8, zorder=0)
        ax.set_xlabel("annual irradiation at ground (kWh/m²·yr)")
        ax.set_yticks([])
        ax.set_title("1 · Whole distributions, not one number", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.text(deciles[1], ax.get_ylim()[1] * 0.97, "  city deciles", color=MUTED, fontsize=6.5, va="top")

        ax = fig.add_subplot(gs[0, 1])
        idx, w = np.arange(10), 0.38
        for off, data, col, lab in [(-w / 2, a_p, DEF_A, "A · IPP favela polygons"), (w / 2, e["p"].to_numpy(), DEF_E, "E · IPP complex outline")]:
            share = np.histogram(data, np.linspace(0, 100, 11))[0] / len(data) * 100
            ax.bar(idx + off, share, w - 0.04, color=col, label=lab)
            ax.text(idx[0] + off, share[0] + 1, f"{share[0]:.0f}%", ha="center", fontsize=7, color=INK)
        ax.axhline(10, color=MUTED, lw=1, ls=(0, (3, 2)))
        ax.text(9.6, 10.6, "city = 10% in each", ha="right", fontsize=6.5, color=MUTED)
        ax.set_xticks(idx, [f"D{i + 1}" for i in idx])
        ax.set_xlabel("citywide irradiation decile  (D1 = darkest tenth of the city's ground)")
        ax.set_ylabel("share of Maré's ground cells (%)")
        ax.set_title("2 · Where Maré's ground falls among the city's deciles", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7, loc="upper right")

        ax = fig.add_subplot(gs[1, :])
        north_to_south = comm.assign(cy=comm.geometry.centroid.y).sort_values("cy", ascending=False)["community"]
        order = [c for c in north_to_south if (e["sub"] == c).any()] + [BETWEEN]
        data = [e.loc[e["sub"] == o, "p"].to_numpy() for o in order]
        bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        for box, o in zip(bp["boxes"], order):
            between = o == BETWEEN
            box.set(facecolor="#f2f1ec" if between else DEF_E, edgecolor=MUTED if between else DEF_E, alpha=0.55)
        ax.axhline(50, color=MUTED, lw=1, ls=(0, (3, 2)))
        ax.text(len(order) + 0.45, 51, "city median", fontsize=6.5, color=MUTED, ha="right")
        ax.set_xticks(range(1, len(order) + 1), [f"{o}\n(n={len(d):,})" for o, d in zip(order, data)], rotation=35, ha="right", fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_ylabel("citywide percentile of each ground cell")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title("3 · Each community's spread against the city — listed north to south, not ranked", loc="left", fontsize=10, color=INK)
        fig.text(0.01, 0.005, f"WP-05 run of record {RUN_OF_RECORD['wp05']} · ground lattice · box = interquartile range, "
                 "whiskers 1.5×IQR · staged for PI review, ethics-gate before release", fontsize=6.5, color=MUTED)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
