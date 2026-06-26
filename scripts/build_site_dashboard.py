"""
Folha de Rua — per-site street-level SVF + solar dashboard.

A3-portrait static PNG + atomic per-panel exports + low-fidelity PDF and a
web-1200 thumbnail. One parameterised entry point handles every site via
--site. Reads only existing pipeline outputs; never regenerates SVF or solar.

Run:
    python scripts/build_site_dashboard.py --site rocinha
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde, pearsonr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/theo/MorphoFavela")
sys.path.insert(0, str(ROOT))

from src.svf_v2.paths import AREA_FILES
from src.viz.folha.sites import SHEET_NUMBER

TYPOLOGY = {
    "vidigal": ("hillside canyon", "#2C5F8D"),
    "rocinha": ("hillside canyon", "#2C5F8D"),
    "complexo_do_alemao": ("hillside canyon", "#2C5F8D"),
    "maré": ("dense low-rise grid", "#6B7280"),
    "mare": ("dense low-rise grid", "#6B7280"),
    "riodaspedras": ("flat-dense alleys", "#5B7C5B"),
}

SITE_DISPLAY = {
    "vidigal": "Vidigal",
    "rocinha": "Rocinha",
    "complexo_do_alemao": "Complexo do Alemão",
    "maré": "Maré",
    "mare": "Maré",
    "riodaspedras": "Rio das Pedras",
}

STRIP_ORDER = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]

# Brazilian Portuguese street-class abbrevs from per-site segments. Used for
# labelling the ridgelines. R == Rua, Trv == Travessa, Etr == Estrada, Bc ==
# Beco, Via == Via, Esc == Escadaria, Lrg == Largo, Tun == Túnel, Aetr ==
# Auto-Estrada, Cam == Caminho, Vila == Vila, Srv == Servidão.
CLASS_LABELS = {
    "R": "Rua",
    "Trv": "Travessa",
    "Etr": "Estrada",
    "Bc": "Beco",
    "Via": "Via",
    "Esc": "Escadaria",
    "Lrg": "Largo",
    "Tun": "Túnel",
    "Aetr": "Auto-estrada",
    "Cam": "Caminho",
    "Vila": "Vila",
    "Srv": "Servidão",
}

INK = "#1A1A1A"
PAPER = "#FAFAF7"
ACCENT = "#D97706"
SVF_FILL = "#2C5F8D"
SOLAR_FILL = "#D97706"
MAGENTA = "#C026D3"
MUTED = "#6B7280"
GREEN = "#5B7C5B"
RED = "#B91C1C"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "savefig.facecolor": PAPER,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "pdf.fonttype": 42,
})


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "0000000"


def site_paths(site: str) -> dict:
    files = AREA_FILES[site]
    raw = ROOT / "data" / site / "raw"
    out = ROOT / "outputs" / site
    return {
        "boundary": raw / files["boundary"],
        "buildings": raw / files["footprints"],
        "svf": out / "morphometrics" / "svf" / "svf_streets.gpkg",
        "solar": out / "morphometrics" / "svf" / "svf_streets_solar.gpkg",
        "segments": out / "morphometrics" / "svf" / "svf_streets_segments.gpkg",
        "observers": out / "sampling_streets" / "observers.gpkg",
        "manifest": out / "sampling_streets" / "manifest.json",
    }


def load_site(site: str, issues: list) -> dict:
    p = site_paths(site)
    svf = gpd.read_file(p["svf"])
    try:
        solar = gpd.read_file(p["solar"])
    except Exception as e:
        issues.append(f"{site}: solar gpkg missing or unreadable ({e})")
        solar = None
    try:
        seg = gpd.read_file(p["segments"])
    except Exception as e:
        issues.append(f"{site}: segments gpkg missing ({e})")
        seg = None
    boundary = gpd.read_file(p["boundary"])
    if boundary.crs is None or boundary.crs.to_epsg() != 31983:
        boundary = boundary.to_crs(31983)
    try:
        buildings = gpd.read_file(p["buildings"])
        if buildings.crs is None or buildings.crs.to_epsg() != 31983:
            buildings = buildings.to_crs(31983)
    except Exception as e:
        issues.append(f"{site}: buildings unreadable ({e})")
        buildings = None
    with open(p["manifest"]) as f:
        manifest = json.load(f)
    return dict(
        svf=svf, solar=solar, seg=seg, boundary=boundary,
        buildings=buildings, manifest=manifest,
    )


def compute_stats(d: dict) -> dict:
    svf = d["svf"]
    solar = d["solar"]
    seg = d["seg"]
    boundary = d["boundary"]
    manifest = d["manifest"]

    n_obs = len(svf)
    road_km = float(seg.geometry.length.sum() / 1000.0) if seg is not None else float("nan")
    area_km2 = float(boundary.geometry.area.sum() / 1e6)
    obs_per_km2 = n_obs / area_km2 if area_km2 > 0 else float("nan")

    # near_boundary: 15 m inward buffer (kept geometry outside the buffer)
    inner = boundary.buffer(-15.0)
    inner_union = inner.union_all() if hasattr(inner, "union_all") else inner.unary_union
    near = ~svf.geometry.within(inner_union)
    edge_share = float(near.mean())

    # length-weighted mean SVF, segment-level
    if seg is not None and "svf_mean" in seg.columns:
        w = seg.geometry.length
        valid = seg["svf_mean"].notna() & (w > 0)
        if valid.any():
            mean_svf = float(np.average(seg.loc[valid, "svf_mean"], weights=w[valid]))
        else:
            mean_svf = float("nan")
    else:
        mean_svf = float(svf["svf"].mean())

    # Pearson — observers not near boundary, with valid solar
    if solar is not None and "solar_hours_annual" in solar.columns:
        m = solar.set_index(["street_id", "distance_along"])
        s = svf.set_index(["street_id", "distance_along"])
        merged = s[["svf", "geometry", "offset_distance"]].join(
            m[["solar_hours_annual", "sunshine_ratio_mean"]], how="inner",
        ).reset_index()
        merged_near = ~gpd.GeoSeries(merged["geometry"], crs=svf.crs).within(inner_union)
        merged["near_boundary"] = merged_near.values
        keep = ~merged["near_boundary"] & merged["svf"].notna() & merged["solar_hours_annual"].notna()
        if keep.sum() > 10:
            r, _ = pearsonr(merged.loc[keep, "svf"], merged.loc[keep, "solar_hours_annual"])
        else:
            r = float("nan")
    else:
        merged = None
        r = float("nan")

    offset_frac = manifest.get("qa", {}).get("offset_fraction", float("nan"))

    return dict(
        n_obs=n_obs, road_km=road_km, area_km2=area_km2,
        obs_per_km2=obs_per_km2, edge_share=edge_share,
        mean_svf=mean_svf, pearson=r, offset_frac=offset_frac,
        merged=merged, inner_union=inner_union, near=near,
    )


# --- panel renderers ---------------------------------------------------------


def draw_masthead(ax, site: str, stats: dict, sha: str, build_date: str, folha_nn: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # ruled top and bottom
    ax.axhline(1.0, color=INK, lw=0.4)
    ax.axhline(0.0, color=INK, lw=0.4)

    ax.text(0.005, 0.62, "MORPHOFAVELA", fontsize=14, fontweight="bold",
            family="DejaVu Sans", color=INK, va="center", ha="left")

    ax.text(0.5, 0.55, SITE_DISPLAY.get(site, site.title()),
            fontsize=32, family="DejaVu Serif", color=INK,
            ha="center", va="center")

    typ_label, typ_color = TYPOLOGY[site]
    right_lines = [
        f"Folha {folha_nn}/05 · EPSG:31983",
        f"build {build_date} · {sha}",
    ]
    ax.text(0.995, 0.78, right_lines[0], fontsize=8, family="DejaVu Sans Mono",
            ha="right", va="center", color=INK)
    ax.text(0.995, 0.52, right_lines[1], fontsize=8, family="DejaVu Sans Mono",
            ha="right", va="center", color=INK)
    ax.text(0.995, 0.22, typ_label.upper(), fontsize=9, family="DejaVu Sans",
            fontweight="bold", ha="right", va="center", color=typ_color)


def draw_identity_card(ax, site: str, stats: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    obs_per_road_km = (
        stats["n_obs"] / stats["road_km"] if stats["road_km"] > 0 else float("nan")
    )
    cells = [
        ("n_obs", f"{stats['n_obs']:,}"),
        ("road km", f"{stats['road_km']:.1f}"),
        ("obs / road-km", f"{obs_per_road_km:,.0f}"),
        ("edge share %", f"{stats['edge_share']*100:.1f}"),
        ("mean SVF", f"{stats['mean_svf']:.3f}"),
        ("ρ(SVF, sol_h)", f"{stats['pearson']:.2f}" if not np.isnan(stats['pearson']) else "n/a"),
    ]
    n = len(cells)
    for i, (lab, val) in enumerate(cells):
        x0 = i / n
        x1 = (i + 1) / n
        cx = (x0 + x1) / 2
        # vertical thin divider
        if i > 0:
            ax.plot([x0, x0], [0.05, 0.95], color=INK, lw=0.3)
        # value-cell tint for edge_share severity
        face = None
        if lab == "edge share %":
            es = stats["edge_share"] * 100
            if es > 20:
                face = RED
            elif es > 15:
                face = ACCENT
        if face:
            ax.add_patch(Rectangle((x0 + 0.005, 0.05), (x1 - x0) - 0.01, 0.9,
                                   facecolor=face, alpha=0.18, edgecolor="none",
                                   transform=ax.transAxes))
        ax.text(cx, 0.60, val, fontsize=13, family="DejaVu Sans Mono",
                ha="center", va="center", color=INK)
        ax.text(cx, 0.22, lab.upper(), fontsize=7.5, family="DejaVu Sans",
                fontweight="bold", ha="center", va="center", color=MUTED)
        if lab == "mean SVF":
            ax.text(cx, 0.05, "length-weighted, segment-level",
                    fontsize=6.5, style="italic", ha="center", va="center",
                    color=MUTED)


def draw_hero_map(ax, site: str, d: dict, stats: dict) -> None:
    ax.set_facecolor(PAPER)
    boundary = d["boundary"]
    buildings = d["buildings"]
    svf = d["svf"]

    # Buildings basemap
    if buildings is not None:
        try:
            buildings.plot(ax=ax, color="#D9D9D6", edgecolor="none",
                           linewidth=0, zorder=1)
        except Exception:
            pass

    # Boundary
    boundary.boundary.plot(ax=ax, color=INK, linewidth=0.6, zorder=3)

    # Edge halo: 15 m inward buffer — hatched ring + dashed inner outline
    # (the hatch is too quiet alone at A3 print scale, per audit feedback)
    try:
        inner = boundary.buffer(-15.0)
        halo = boundary.difference(inner)
        halo_gs = gpd.GeoSeries(halo, crs=boundary.crs)
        halo_gs.plot(ax=ax, facecolor="none", edgecolor=INK,
                     linewidth=0.0, hatch="////", alpha=0.20, zorder=2)
        inner_gs = gpd.GeoSeries(inner, crs=boundary.crs)
        inner_gs.boundary.plot(ax=ax, color=INK, linewidth=0.5,
                               linestyle="--", zorder=2.5, alpha=0.6)
    except Exception:
        pass

    # observer classification
    is_zero = svf["svf"] == 0.0
    is_offset = (svf["offset_distance"] > 2.5) & ~is_zero
    normal = ~is_zero & ~is_offset

    cmap = mpl.colormaps.get_cmap("YlGnBu_r")
    norm_v = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    if normal.any():
        ax.scatter(
            svf.loc[normal].geometry.x, svf.loc[normal].geometry.y,
            c=cmap(norm_v(svf.loc[normal, "svf"].values)),
            s=1.6, alpha=0.85, linewidths=0, zorder=4,
        )
    if is_offset.any():
        ax.scatter(
            svf.loc[is_offset].geometry.x, svf.loc[is_offset].geometry.y,
            facecolors="none",
            edgecolors=cmap(norm_v(svf.loc[is_offset, "svf"].values)),
            s=4.0, linewidths=0.6, zorder=5,
        )
    if is_zero.any():
        # Larger, bolder cross with faint halo so dense-alley svf==0 streaks
        # remain legible as discrete glyphs rather than continuous strokes
        # (orthogonal-grid failure mode flagged in riodaspedras review).
        ax.scatter(
            svf.loc[is_zero].geometry.x, svf.loc[is_zero].geometry.y,
            marker="o", facecolors=MAGENTA, edgecolors="none",
            s=3.0, alpha=0.30, linewidths=0, zorder=6,
        )
        ax.scatter(
            svf.loc[is_zero].geometry.x, svf.loc[is_zero].geometry.y,
            marker="+", c=MAGENTA, s=20.0, linewidths=0.9, zorder=7,
        )

    minx, miny, maxx, maxy = boundary.total_bounds
    pad = 0.04 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Scalebar 200 m bottom-left
    bar_y = miny - pad + (maxy - miny) * 0.02
    bar_x0 = minx - pad + (maxx - minx) * 0.04
    bar_x1 = bar_x0 + 200.0
    ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color=INK, lw=1.4)
    ax.text((bar_x0 + bar_x1) / 2, bar_y + (maxy - miny) * 0.012,
            "200 m", fontsize=7, family="DejaVu Sans Mono",
            ha="center", va="bottom", color=INK)
    # North arrow top-left of bar
    arr_x = bar_x0
    arr_y0 = bar_y + (maxy - miny) * 0.04
    arr_y1 = arr_y0 + (maxy - miny) * 0.04
    ax.annotate("", xy=(arr_x, arr_y1), xytext=(arr_x, arr_y0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.text(arr_x + (maxx - minx) * 0.004, arr_y1, "N",
            fontsize=7, family="DejaVu Sans Mono", color=INK,
            ha="left", va="center")

    # colorbar inset for SVF
    cbar_ax = ax.inset_axes([0.02, 0.86, 0.30, 0.025])
    sm = mpl.cm.ScalarMappable(norm=norm_v, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=6, length=2, pad=2)
    cb.set_label("SVF", fontsize=7, color=INK)

    # Locator inset top-right: all five site boundaries in their geographic
    # Rio-metro positions, current site filled orange. No external coastline
    # tile dependency — works offline from the data already on disk.
    try:
        loc_ax = ax.inset_axes([0.78, 0.78, 0.20, 0.20])
        loc_ax.set_facecolor("#F0EFEA")
        for sname in STRIP_ORDER:
            try:
                b = gpd.read_file(site_paths(sname)["boundary"]).to_crs(31983)
                if sname == site:
                    b.plot(ax=loc_ax, facecolor=ACCENT, edgecolor=INK,
                           linewidth=0.6, zorder=3)
                else:
                    b.plot(ax=loc_ax, facecolor="#BFC4BD", edgecolor=INK,
                           linewidth=0.3, zorder=2)
            except Exception:
                pass
        loc_ax.set_aspect("equal")
        loc_ax.set_xticks([])
        loc_ax.set_yticks([])
        for spine in loc_ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color(INK)
        loc_ax.set_title("Rio metro · 5 sites", fontsize=6.5, pad=2,
                         color=INK)
    except Exception:
        pass


def draw_hero_legend(ax, counts: dict | None = None) -> None:
    ax.axis("off")
    counts = counts or {}

    def _lab(base, key):
        c = counts.get(key)
        return f"{base}  (n={c:,})" if c is not None else base

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#2C5F8D", markersize=5,
               label=_lab("resolved observer", "resolved")),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="none", markeredgecolor="#2C5F8D",
               markeredgewidth=0.8, markersize=6,
               label=_lab("offset > 2.5 m (low-confidence)", "offset")),
        Line2D([0], [0], marker="+", color=MAGENTA,
               markersize=8, linestyle="None",
               label=_lab("SVF == 0 (unresolved)", "unresolved")),
        mpatches.Patch(facecolor="#D9D9D6", edgecolor="none",
                       label="building footprints"),
        mpatches.Patch(facecolor="none", edgecolor=INK, hatch="////",
                       label="15 m edge-halo zone"),
    ]
    ax.legend(handles=handles, loc="center left", fontsize=8,
              frameon=False, ncol=2, columnspacing=1.2,
              handletextpad=0.6)


def _ridge_panel(ax, data_by_class, x_range, fill_color, x_label,
                 metric_label=None, metric_by_class=None) -> None:
    ax.set_facecolor(PAPER)
    keys = [k for k, v in data_by_class.items() if len(v) >= 30]
    sparse = {k: v for k, v in data_by_class.items() if 10 <= len(v) < 30}
    absent = {k: v for k, v in data_by_class.items() if len(v) < 10}

    # order by descending count
    keys = sorted(keys, key=lambda k: -len(data_by_class[k]))
    sparse_keys = sorted(sparse.keys(), key=lambda k: -len(sparse[k]))
    absent_keys = sorted(absent.keys(), key=lambda k: -len(absent[k]))

    rows = keys + sparse_keys + absent_keys
    n_rows = max(len(rows), 1)
    ax.set_xlim(*x_range)
    ax.set_ylim(-0.5, n_rows)

    x = np.linspace(x_range[0], x_range[1], 400)
    for i, k in enumerate(rows):
        y0 = n_rows - 1 - i
        vals = data_by_class[k]
        label = CLASS_LABELS.get(k, k)
        n = len(vals)
        if n >= 30:
            try:
                kde = gaussian_kde(vals, bw_method=0.25)
                y = kde(x)
                y = y / y.max() * 0.85
                ax.fill_between(x, y0, y0 + y, color=fill_color, alpha=0.35,
                                linewidth=0)
                ax.plot(x, y0 + y, color=INK, lw=1.0)
                med = float(np.median(vals))
                ax.plot([med, med], [y0, y0 + 0.18], color=INK, lw=1.6)
                ax.text(x_range[1] * 1.005, y0 + 0.45,
                        f"{label}  n={n}", fontsize=7.5,
                        family="DejaVu Sans", va="center", ha="left",
                        color=INK)
                if metric_by_class and k in metric_by_class:
                    ax.text(x_range[0] + (x_range[1]-x_range[0]) * 0.02,
                            y0 + 0.6,
                            f"{metric_label}={metric_by_class[k]:.2f}",
                            fontsize=6.5, family="DejaVu Sans Mono",
                            color=MUTED, va="center", ha="left")
            except Exception:
                pass
        elif n >= 10:
            try:
                kde = gaussian_kde(vals, bw_method=0.3)
                y = kde(x)
                y = y / y.max() * 0.7
                ax.plot(x, y0 + y, color=MUTED, lw=0.8, linestyle="--")
                ax.text(x_range[1] * 1.005, y0 + 0.45,
                        f"{label}  n={n} — sparse",
                        fontsize=7, color=MUTED, va="center", ha="left",
                        style="italic")
            except Exception:
                pass
        else:
            ax.plot([x_range[0], x_range[1]], [y0, y0],
                    color=MUTED, lw=0.4, linestyle="--")
            ax.text((x_range[0] + x_range[1]) / 2, y0 + 0.12,
                    f"{label}: n<10 — not sampled" if n > 0 else
                    f"{label}: absent",
                    fontsize=7, color=MUTED, va="center", ha="center",
                    style="italic")

    ax.set_xlabel(x_label, fontsize=9)
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="x", labelsize=7)


def _quadrant_groups(svf_or_solar, boundary, value_col: str):
    """Bin observers into NE/SE/SW/NW from boundary centroid; return dict of arrays."""
    bg = boundary.geometry
    centroid = (bg.union_all() if hasattr(bg, "union_all") else bg.unary_union).centroid
    cx = centroid.x
    cy = centroid.y
    dx = svf_or_solar.geometry.x.values - cx
    dy = svf_or_solar.geometry.y.values - cy
    # azimuth: 0 = N (+y), clockwise → E
    ang = (np.degrees(np.arctan2(dx, dy))) % 360.0
    bins = [("NE", 0, 90), ("SE", 90, 180), ("SW", 180, 270), ("NW", 270, 360)]
    vals = svf_or_solar[value_col].values
    out = {}
    for lab, lo, hi in bins:
        m = (ang >= lo) & (ang < hi) & np.isfinite(vals)
        if m.sum() > 0:
            out[lab] = vals[m]
    return out


def draw_svf_ridgeline(ax, d: dict) -> None:
    svf = d["svf"]
    seg = d["seg"]
    if seg is not None and "tipo_logra" in seg.columns:
        cls_by_seg = seg.set_index("street_id")["tipo_logra"].to_dict()
        svf_local = svf.copy()
        svf_local["cls"] = svf_local["street_id"].map(cls_by_seg)
        by_class = {
            k: v["svf"].dropna().values
            for k, v in svf_local.dropna(subset=["cls"]).groupby("cls")
        }
        _ridge_panel(ax, by_class, (0, 1.0), SVF_FILL, "SVF")
        ax.set_xticks(np.arange(0, 1.01, 0.1))
        return
    # Fallback: compass quadrants (Maré case — no usable highway class column)
    by_class = _quadrant_groups(svf, d["boundary"], "svf")
    _ridge_panel(ax, by_class, (0, 1.0), SVF_FILL, "SVF")
    ax.set_xticks(np.arange(0, 1.01, 0.1))


def draw_solar_ridgeline(ax, d: dict, site: str) -> None:
    seg = d["seg"]
    solar = d["solar"]
    if solar is None or "solar_hours_annual" not in solar.columns:
        ax.text(0.5, 0.5, "solar data unavailable",
                ha="center", va="center", color=MUTED, fontsize=9)
        ax.axis("off")
        return

    if seg is not None and "tipo_logra" in seg.columns:
        cls_by_seg = seg.set_index("street_id")["tipo_logra"].to_dict()
        s = solar.copy()
        s["cls"] = s["street_id"].map(cls_by_seg)
        by_class = {
            k: v["solar_hours_annual"].dropna().values
            for k, v in s.dropna(subset=["cls"]).groupby("cls")
        }
        median_ratio_by_class = {
            k: float(v["sunshine_ratio_mean"].dropna().median())
            for k, v in s.dropna(subset=["cls"]).groupby("cls")
            if v["sunshine_ratio_mean"].notna().any()
        }
    else:
        # Compass-quadrant fallback for Maré
        by_class = _quadrant_groups(solar, d["boundary"], "solar_hours_annual")
        sun_by_class = _quadrant_groups(solar, d["boundary"], "sunshine_ratio_mean")
        median_ratio_by_class = {
            k: float(np.nanmedian(v)) for k, v in sun_by_class.items()
        }
    # solar_hours_annual is daily-averaged (range ~0–12 h/day), not an annual sum
    _ridge_panel(ax, by_class, (0, 12), SOLAR_FILL, "solar hours / day (annual mean)",
                 metric_label="r̄", metric_by_class=median_ratio_by_class)
    ax.set_xticks(np.arange(0, 13, 2))

    # Sunshine-ratio badge. r̄ = mean(solar_hours/day) / available daylight.
    # Rio's ~22.9°S → ~11–13.5 h daylight, so an open observer should sit
    # in [0.40, 0.85]. Deep canyons and flat-dense alleys legitimately fall
    # below 0.40 from mutual shading; that's a shaded regime, not a unit
    # bug. >0.85 site median would be suspicious.
    overall = solar["sunshine_ratio_mean"].dropna()
    if len(overall) > 0:
        med = float(overall.median())
        if 0.40 <= med <= 0.85:
            ax.text(0.985, 0.97, f"open-sky regime ✓ (r̄={med:.2f})",
                    transform=ax.transAxes,
                    fontsize=7, color=GREEN, ha="right", va="top")
        elif med < 0.40:
            ax.text(0.985, 0.97, f"shaded regime ✓ (r̄={med:.2f})",
                    transform=ax.transAxes,
                    fontsize=7, color=SVF_FILL, ha="right", va="top")
        else:
            ax.text(0.985, 0.97, f"check (r̄={med:.2f})",
                    transform=ax.transAxes,
                    fontsize=7, color=RED, ha="right", va="top")


def draw_hexbin(ax, d: dict, stats: dict) -> None:
    merged = stats.get("merged")
    if merged is None:
        ax.text(0.5, 0.5, "no joinable solar data",
                ha="center", va="center", color=MUTED)
        ax.axis("off")
        return
    keep = ~merged["near_boundary"] & merged["svf"].notna() \
        & merged["solar_hours_annual"].notna()
    n_excl = (~keep).sum()
    pct_excl = n_excl / len(merged) * 100
    sub = merged.loc[keep]

    hb = ax.hexbin(sub["svf"], sub["solar_hours_annual"], gridsize=30,
                   cmap="Greys", mincnt=1, linewidths=0.0)
    ax.set_xlim(0, 1)
    y_top = float(sub["solar_hours_annual"].max()) * 1.05 if len(sub) else 12.0
    ax.set_ylim(0, min(12.5, y_top))
    ax.set_xlabel("SVF", fontsize=9)
    ax.set_ylabel("solar hours / day (annual mean)", fontsize=9)
    ax.tick_params(labelsize=7)

    # quantile regression q=0.5 fallback: simple linear fit on medians per bin
    try:
        from scipy.optimize import minimize
        x = sub["svf"].values
        y = sub["solar_hours_annual"].values

        def q_loss(beta):
            a, b = beta
            resid = y - (a * x + b)
            return np.sum(np.where(resid >= 0, 0.5 * resid, -0.5 * resid))
        res = minimize(q_loss, x0=[8.0, 1.5], method="Nelder-Mead")
        a, b = res.x
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, a * xs + b, color=INK, lw=0.8)
    except Exception:
        pass

    if len(sub) > 10:
        r, _ = pearsonr(sub["svf"], sub["solar_hours_annual"])
        ax.text(0.98, 0.97, f"r = {r:.2f}", transform=ax.transAxes,
                fontsize=10, family="DejaVu Sans Mono", ha="right", va="top")
    ax.text(0.02, -0.18, f"edge n={int(n_excl)} excluded ({pct_excl:.1f}% of network)",
            transform=ax.transAxes, fontsize=7, color=MUTED, ha="left", va="top")

    # GMM bimodality check — skip when SVF×solar is strongly linear
    # (|r| > 0.9 implies a single elongated cluster, not two modes)
    try:
        r_check, _ = pearsonr(sub["svf"], sub["solar_hours_annual"])
    except Exception:
        r_check = 0.0
    try:
        from sklearn.mixture import GaussianMixture
        X = np.column_stack([
            (sub["svf"] - sub["svf"].mean()) / sub["svf"].std(),
            (sub["solar_hours_annual"] - sub["solar_hours_annual"].mean())
            / sub["solar_hours_annual"].std(),
        ])
        if len(X) > 50 and abs(r_check) < 0.9:
            g1 = GaussianMixture(n_components=1, random_state=0).fit(X)
            g2 = GaussianMixture(n_components=2, random_state=0).fit(X)
            if g2.bic(X) < g1.bic(X) - 10:
                centroids = g2.means_
                centroids_real = np.column_stack([
                    centroids[:, 0] * sub["svf"].std() + sub["svf"].mean(),
                    centroids[:, 1] * sub["solar_hours_annual"].std()
                    + sub["solar_hours_annual"].mean(),
                ])
                ax.scatter(centroids_real[:, 0], centroids_real[:, 1],
                           marker="+", c=MAGENTA, s=80, linewidths=1.5,
                           zorder=10)
                ax.text(0.02, 0.97, "2 modes — canyon / open",
                        transform=ax.transAxes, fontsize=8, style="italic",
                        color=MAGENTA, va="top", ha="left")
    except Exception:
        pass

    cb_ax = ax.inset_axes([1.02, 0.0, 0.025, 1.0])
    cb = plt.colorbar(hb, cax=cb_ax)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("count", fontsize=7)


def draw_small_multiples(ax, current_site: str, issues: list) -> None:
    n = len(STRIP_ORDER)
    # subaxes
    ax.axis("off")
    fig = ax.figure
    bbox = ax.get_position()
    width = bbox.width / n * 0.95
    h = bbox.height * 0.92
    for i, site in enumerate(STRIP_ORDER):
        x0 = bbox.x0 + (bbox.width / n) * i + (bbox.width / n - width) / 2
        y0 = bbox.y0 + bbox.height * 0.03
        sub = fig.add_axes([x0, y0, width, h])
        sub.set_facecolor(PAPER)
        try:
            paths_i = site_paths(site)
            b = gpd.read_file(paths_i["boundary"]).to_crs(31983)
            try:
                buildings = gpd.read_file(paths_i["buildings"]).to_crs(31983)
                buildings.plot(ax=sub, color="#D9D9D6", linewidth=0, zorder=1)
            except Exception:
                pass
            b.boundary.plot(ax=sub, color=INK, linewidth=0.4, zorder=2)
            sv = gpd.read_file(paths_i["svf"])
            sub.scatter(sv.geometry.x, sv.geometry.y,
                        c=sv["svf"].values, cmap="YlGnBu_r",
                        vmin=0, vmax=1, s=0.18, alpha=0.7, linewidths=0,
                        zorder=3)
            minx, miny, maxx, maxy = b.total_bounds
            pad = 0.04 * max(maxx - minx, maxy - miny)
            sub.set_xlim(minx - pad, maxx + pad)
            sub.set_ylim(miny - pad, maxy + pad)
        except Exception as e:
            issues.append(f"small_multiples:{site}: {e}")
        sub.set_aspect("equal")
        sub.set_xticks([])
        sub.set_yticks([])
        for spine in sub.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_color(INK)
        if site == current_site:
            for spine in sub.spines.values():
                spine.set_linewidth(2.8)
                spine.set_color(ACCENT)
            # 1 pt inner white halo so the orange "you are here" border
            # pops against neighbours of similar dark-ink stroke.
            sub.patch.set_edgecolor(PAPER)
            sub.patch.set_linewidth(0.0)
        sub.set_title(SITE_DISPLAY.get(site, site), fontsize=8,
                      color=INK, pad=2)


def draw_caveats(ax, site: str, stats: dict) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axhline(1.0, color=INK, lw=0.4)

    edge_pct = stats["edge_share"] * 100
    obs_per_km2 = stats["obs_per_km2"]
    road_km = stats["road_km"]
    n_obs = stats["n_obs"]
    offset_pct = stats["offset_frac"] * 100

    caveats = [
        ("[H1] EDGE HALO",
         f"{edge_pct:.1f}% of observers fall within 15 m of the boundary; "
         "external building footprints beyond the site are not modelled, "
         "so perimeter SVF and solar are biased high. Affected zone is "
         "hatched on the hero map."),
        ("[H2] OBSERVER DENSITY",
         f"{obs_per_km2:.0f} obs/km² ({road_km:.1f} km of network, "
         f"n={n_obs:,}). Cross-site density spreads up to 2.3×; raw counts "
         "are not directly comparable across sites."),
        ("[M2] SOLAR UNITS",
         "solar_hours_annual and sunshine_ratio_mean are canonical; the "
         "legacy solar_hours column is deprecated and not used. r̄ in "
         "[0.45, 0.85] is consistent with Rio (~22.9°S) latitude."),
        ("[M3] MEAN SVF · [L1] OFFSET",
         f"Mean SVF reported length-weighted at segment level; pre-1bffc3e "
         f"runs were observer-weighted and over-counted short segments. "
         f"{offset_pct:.1f}% of observers were repositioned > 2.5 m from "
         "the original sample location (low-confidence glyphs on map)."),
    ]
    n = len(caveats)
    for i, (head, body) in enumerate(caveats):
        x0 = i / n + 0.005
        ax.text(x0, 0.88, head, fontsize=7, fontweight="bold",
                family="DejaVu Sans Mono", color=MUTED,
                ha="left", va="top")
        ax.text(x0, 0.72, body, fontsize=7.5, color=INK,
                ha="left", va="top", wrap=True,
                family="DejaVu Sans",
                bbox=None)
        # text wrap by manual width
    ax.text(0.995, 0.05, "See technical_report §4.2, §6",
            fontsize=6.5, style="italic", color=MUTED,
            ha="right", va="bottom")


def _wrap_text(s: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(s, width=width))


def draw_caveats_v2(ax, site: str, stats: dict) -> None:
    """Cleaner caveat strip with proper wrapping."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axhline(1.0, color=INK, lw=0.4)

    edge_pct = stats["edge_share"] * 100
    obs_per_km2 = stats["obs_per_km2"]
    road_km = stats["road_km"]
    n_obs = stats["n_obs"]
    offset_pct = stats["offset_frac"] * 100

    caveats = [
        ("[H1] EDGE HALO",
         f"{edge_pct:.1f}% of observers fall within 15 m of the "
         "boundary; external footprints beyond the site are not "
         "modelled, so perimeter SVF and solar are biased high. "
         "Affected zone is hatched on the hero map."),
        ("[H2] OBSERVER DENSITY",
         f"{obs_per_km2:.0f} obs/km² ({road_km:.1f} km of "
         f"network, n={n_obs:,}). Cross-site density spreads up "
         "to 2.3×; raw counts are not directly comparable."),
        ("[M2] SOLAR UNITS",
         "solar_hours_annual and sunshine_ratio_mean are canonical; "
         "legacy solar_hours is deprecated. Sunshine ratio in "
         "[0.45, 0.85] is consistent with Rio's ~22.9°S latitude."),
        ("[M3/L1] MEAN SVF · OFFSET",
         "Mean SVF is length-weighted segment-level; pre-1bffc3e "
         f"runs were observer-weighted. {offset_pct:.1f}% of "
         "observers were repositioned > 2.5 m (low-confidence "
         "glyphs on map)."),
    ]
    n = len(caveats)
    col_w = 1.0 / n
    for i, (head, body) in enumerate(caveats):
        cx = i * col_w + 0.01
        ax.text(cx, 0.92, head, fontsize=7.5, fontweight="bold",
                family="DejaVu Sans Mono", color=MUTED,
                ha="left", va="top")
        body_wrapped = _wrap_text(body, 42)
        ax.text(cx, 0.78, body_wrapped, fontsize=7, color=INK,
                ha="left", va="top", family="DejaVu Sans",
                linespacing=1.35)
    ax.text(0.995, 0.04, "See technical_report §4.2, §6",
            fontsize=6.5, style="italic", color=MUTED,
            ha="right", va="bottom")


# --- atomic exports ---------------------------------------------------------


def _save_atom(out_dir: Path, name: str, renderer, figsize, **kwargs):
    fig = plt.figure(figsize=figsize, dpi=200, facecolor=PAPER)
    ax = fig.add_subplot(111)
    renderer(ax, **kwargs)
    p = out_dir / f"{name}.png"
    fig.savefig(p, dpi=200, facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
    return p


def build_dashboard(site: str) -> dict:
    issues: list = []
    panels: list = []

    out_dir = ROOT / "outputs" / "_distribution" / "site_dashboards" / site
    atoms = out_dir / "atoms"
    out_dir.mkdir(parents=True, exist_ok=True)
    atoms.mkdir(parents=True, exist_ok=True)

    d = load_site(site, issues)
    stats = compute_stats(d)
    sha = git_sha()
    build_date = datetime.date.today().isoformat()
    folha_nn = SHEET_NUMBER.get(site, "00")

    # A3 portrait: 297 x 420 mm → 11.69 x 16.54 in
    fig = plt.figure(figsize=(11.69, 16.54), dpi=200, facecolor=PAPER)
    gs = fig.add_gridspec(
        nrows=8, ncols=2,
        height_ratios=[1.1, 0.7, 6.0, 0.6, 2.2, 2.2, 2.6, 1.2],
        width_ratios=[1, 1],
        left=0.04, right=0.97, top=0.985, bottom=0.015,
        hspace=0.35, wspace=0.20,
    )

    ax_mast = fig.add_subplot(gs[0, :])
    draw_masthead(ax_mast, site, stats, sha, build_date, folha_nn)
    panels.append("masthead")

    ax_id = fig.add_subplot(gs[1, :])
    draw_identity_card(ax_id, site, stats)
    panels.append("identity_card")

    ax_hero = fig.add_subplot(gs[2, :])
    draw_hero_map(ax_hero, site, d, stats)
    ax_hero.set_title("Street-level SVF — observer network on building fabric",
                      fontsize=11, color=INK, pad=8, loc="left")
    panels.append("hero_map")

    ax_legend = fig.add_subplot(gs[3, :])
    svf_for_counts = d["svf"]
    _zero = (svf_for_counts["svf"] == 0.0)
    _off = (svf_for_counts["offset_distance"] > 2.5) & ~_zero
    hero_counts = {
        "resolved": int((~_zero & ~_off).sum()),
        "offset": int(_off.sum()),
        "unresolved": int(_zero.sum()),
    }
    draw_hero_legend(ax_legend, counts=hero_counts)

    ax_svf = fig.add_subplot(gs[4, 0])
    draw_svf_ridgeline(ax_svf, d)
    ax_svf.set_title("SVF distribution by street class",
                     fontsize=10, color=INK, pad=4, loc="left")
    panels.append("svf_ridgeline")

    ax_sol = fig.add_subplot(gs[4, 1])
    draw_solar_ridgeline(ax_sol, d, site)
    ax_sol.set_title("Annual solar access by street class",
                     fontsize=10, color=INK, pad=4, loc="left")
    panels.append("solar_ridgeline")

    ax_hex = fig.add_subplot(gs[5, :])
    draw_hexbin(ax_hex, d, stats)
    ax_hex.set_title("SVF × solar — physical consistency check",
                     fontsize=10, color=INK, pad=4, loc="left")
    panels.append("svf_solar_hexbin")

    ax_strip = fig.add_subplot(gs[6, :])
    draw_small_multiples(ax_strip, site, issues)
    ax_strip.set_title("Cross-site comparison — shared SVF ramp",
                       fontsize=10, color=INK, pad=4, loc="left")
    panels.append("small_multiples_strip")

    ax_cav = fig.add_subplot(gs[7, :])
    draw_caveats_v2(ax_cav, site, stats)
    panels.append("caveat_strip")

    a3_path = out_dir / f"folha_{site}_A3.png"
    fig.savefig(a3_path, dpi=220, facecolor=PAPER, bbox_inches=None)

    pdf_path = out_dir / f"folha_{site}.pdf"
    fig.savefig(pdf_path, facecolor=PAPER)

    # Web 1200 thumb
    web_path = out_dir / f"folha_{site}_web1200.png"
    fig.savefig(web_path, dpi=85, facecolor=PAPER)

    plt.close(fig)

    # Atomic exports — render each panel into its own figure
    try:
        _save_atom(atoms, "masthead", lambda ax: draw_masthead(
            ax, site, stats, sha, build_date, folha_nn),
                  figsize=(11.69, 1.1))
    except Exception as e:
        issues.append(f"atom masthead: {e}")

    try:
        _save_atom(atoms, "identity_card", lambda ax: draw_identity_card(
            ax, site, stats), figsize=(11.69, 0.9))
    except Exception as e:
        issues.append(f"atom identity_card: {e}")

    try:
        fig2 = plt.figure(figsize=(11.69, 9.0), dpi=200, facecolor=PAPER)
        ax2 = fig2.add_subplot(111)
        draw_hero_map(ax2, site, d, stats)
        fig2.savefig(atoms / "hero_map.png", dpi=200, facecolor=PAPER,
                     bbox_inches="tight")
        plt.close(fig2)
    except Exception as e:
        issues.append(f"atom hero_map: {e}")

    try:
        fig2 = plt.figure(figsize=(5.85, 3.5), dpi=200, facecolor=PAPER)
        ax2 = fig2.add_subplot(111)
        draw_svf_ridgeline(ax2, d)
        fig2.savefig(atoms / "svf_ridgeline.png", dpi=200,
                     facecolor=PAPER, bbox_inches="tight")
        plt.close(fig2)
    except Exception as e:
        issues.append(f"atom svf_ridgeline: {e}")

    try:
        fig2 = plt.figure(figsize=(5.85, 3.5), dpi=200, facecolor=PAPER)
        ax2 = fig2.add_subplot(111)
        draw_solar_ridgeline(ax2, d, site)
        fig2.savefig(atoms / "solar_ridgeline.png", dpi=200,
                     facecolor=PAPER, bbox_inches="tight")
        plt.close(fig2)
    except Exception as e:
        issues.append(f"atom solar_ridgeline: {e}")

    try:
        fig2 = plt.figure(figsize=(10.0, 5.0), dpi=200, facecolor=PAPER)
        ax2 = fig2.add_subplot(111)
        draw_hexbin(ax2, d, stats)
        fig2.savefig(atoms / "svf_solar_hexbin.png", dpi=200,
                     facecolor=PAPER, bbox_inches="tight")
        plt.close(fig2)
    except Exception as e:
        issues.append(f"atom svf_solar_hexbin: {e}")

    try:
        fig2 = plt.figure(figsize=(11.69, 3.0), dpi=200, facecolor=PAPER)
        ax2 = fig2.add_subplot(111)
        ax2.set_position([0.02, 0.05, 0.96, 0.85])
        draw_small_multiples(ax2, site, issues)
        fig2.savefig(atoms / "small_multiples_strip.png", dpi=200,
                     facecolor=PAPER)
        plt.close(fig2)
    except Exception as e:
        issues.append(f"atom small_multiples: {e}")

    try:
        _save_atom(atoms, "caveat_strip",
                   lambda ax: draw_caveats_v2(ax, site, stats),
                   figsize=(11.69, 1.6))
    except Exception as e:
        issues.append(f"atom caveat: {e}")

    # metadata
    meta = {
        "site": site,
        "site_display": SITE_DISPLAY.get(site, site),
        "build_date": build_date,
        "git_sha7": sha,
        "panels_built": panels,
        "stats": {
            "n_obs": int(stats["n_obs"]),
            "road_km": float(stats["road_km"]),
            "area_km2": float(stats["area_km2"]),
            "obs_per_km2": float(stats["obs_per_km2"]),
            "edge_share": float(stats["edge_share"]),
            "mean_svf_length_weighted": float(stats["mean_svf"]),
            "pearson_svf_solar": None if np.isnan(stats["pearson"])
                                 else float(stats["pearson"]),
            "offset_fraction": float(stats["offset_frac"]),
        },
        "issues": issues,
        "outputs": {
            "A3_png": str(a3_path),
            "pdf": str(pdf_path),
            "web1200": str(web_path),
            "atoms_dir": str(atoms),
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return dict(
        a3_path=a3_path, pdf=pdf_path, web=web_path,
        metadata=out_dir / "metadata.json",
        stats=meta["stats"], issues=issues, panels=panels,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True)
    args = p.parse_args()
    site = args.site
    result = build_dashboard(site)
    print(f"A3 PNG: {result['a3_path']}")
    print(f"PDF:    {result['pdf']}")
    print(f"web1200: {result['web']}")
    print(f"metadata: {result['metadata']}")
    if result["issues"]:
        print("ISSUES:")
        for i in result["issues"]:
            print(f"  - {i}")


if __name__ == "__main__":
    main()
