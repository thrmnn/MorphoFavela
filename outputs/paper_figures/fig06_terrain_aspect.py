"""Figure 6 — SVF↔solar dissociation on sloped terrain (cross-site).

Per-quadrant median traces of winter-solstice solar hours as a function
of SVF, for every site whose street-level solar pipeline has run.
Filter `slope ≥ 5°` is applied so the dissociation is read on cells
where terrain aspect can actually matter — flatland sites with very
few sloped cells will show thin panels (which is itself the story).

Small-multiples layout (2 × 3): Vidigal, Rocinha, Complexo do Alemão on
the top row; Rio das Pedras, Maré (pending), and a per-site quadrant-
mean summary on the bottom row. The spatial expression of the same
dissociation is in `figS_terrain_aspect_spatial.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (  # noqa: E402
    PROJECT_ROOT,
    SITE_LABELS,
    SITE_ORDER,
    WIDTH_DOUBLE,
    apply_style,
    save_fig,
)

sys.path.insert(0, str(PROJECT_ROOT))
from src.morphometry.aspect import aspect_quadrant  # noqa: E402

QUADRANT_COLORS = {
    "N": "#1F6FB0",
    "E": "#2CA02C",
    "S": "#D62728",
    "W": "#FF7F0E",
}
QUADRANTS = ("N", "E", "S", "W")
SLOPE_THRESHOLD = 5.0
MIN_BIN_N = 30
MIN_PANEL_N = 50


def _build_compass_cmap() -> ListedColormap:
    """Cyclic colormap: N(0°) blue → E(90°) green → S(180°) red → W(270°) orange → N."""
    from matplotlib.colors import to_rgb

    anchors = [
        (0.0, QUADRANT_COLORS["N"]),
        (0.25, QUADRANT_COLORS["E"]),
        (0.50, QUADRANT_COLORS["S"]),
        (0.75, QUADRANT_COLORS["W"]),
        (1.0, QUADRANT_COLORS["N"]),
    ]
    n_steps = 256
    rgb = np.zeros((n_steps, 3))
    for i in range(n_steps):
        t = i / (n_steps - 1)
        for j in range(len(anchors) - 1):
            t0, c0 = anchors[j]
            t1, c1 = anchors[j + 1]
            if t0 <= t <= t1:
                local = (t - t0) / (t1 - t0)
                rgb0 = np.array(to_rgb(c0))
                rgb1 = np.array(to_rgb(c1))
                rgb[i] = rgb0 * (1 - local) + rgb1 * local
                break
    return ListedColormap(rgb, name="compass")


def _load_site(site: str) -> gpd.GeoDataFrame | None:
    solar_path = (
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    )
    grid_path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    if not solar_path.exists():
        return None
    solar = gpd.read_file(solar_path)
    grid = gpd.read_file(grid_path)
    if solar.crs != grid.crs:
        solar = solar.to_crs(grid.crs)
    keep = grid[["geometry", "aspect_deg", "slope_deg"]]
    pts = gpd.sjoin(solar, keep, how="left", predicate="within").drop(columns="index_right")
    pts = pts.dropna(subset=["aspect_deg", "svf", "solar_hours_winter"])
    pts = pts[pts["slope_deg"].fillna(0) >= SLOPE_THRESHOLD].copy()
    if pts.empty:
        return None
    pts["quadrant"] = aspect_quadrant(pts["aspect_deg"].values)
    return pts


def _quadrant_trace(
    sub: gpd.GeoDataFrame,
    svf_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centres = 0.5 * (svf_edges[:-1] + svf_edges[1:])
    idx = np.digitize(sub["svf"].values, svf_edges) - 1
    idx = np.clip(idx, 0, len(centres) - 1)
    med = np.full(len(centres), np.nan)
    q25 = np.full(len(centres), np.nan)
    q75 = np.full(len(centres), np.nan)
    counts = np.zeros(len(centres), dtype=int)
    for i in range(len(centres)):
        vals = sub["solar_hours_winter"].values[idx == i]
        counts[i] = len(vals)
        if len(vals) > 0:
            med[i] = np.median(vals)
            q25[i] = np.percentile(vals, 25)
            q75[i] = np.percentile(vals, 75)
    return centres, med, q25, q75, counts


def _draw_panel(
    ax: plt.Axes,
    pts: gpd.GeoDataFrame | None,
    site: str,
    cmap: ListedColormap,
    *,
    is_pending: bool = False,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 10.5)
    ax.axhline(0, color="0.85", lw=0.4, zorder=0)
    ax.set_title(SITE_LABELS[site], fontsize=8, pad=3, fontweight="bold", color="#222")

    if is_pending or pts is None or len(pts) < MIN_PANEL_N:
        ax.text(
            0.5,
            0.5,
            "(pending)" if is_pending else f"sparse\nn = {0 if pts is None else len(pts)}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#888",
            style="italic",
        )
        ax.tick_params(axis="both", labelsize=5.5, length=2, width=0.4, pad=2)
        return

    ax.scatter(
        pts["svf"],
        pts["solar_hours_winter"],
        c=pts["aspect_deg"] / 360.0,
        cmap=cmap,
        s=2.0,
        alpha=0.14,
        linewidths=0,
        zorder=1,
    )
    svf_edges = np.linspace(0, 1, 11)
    for q in QUADRANTS:
        sub = pts[pts["quadrant"] == q]
        if sub.empty:
            continue
        centres, med, q25, q75, counts = _quadrant_trace(sub, svf_edges)
        finite = np.isfinite(med)
        solid = finite & (counts >= MIN_BIN_N)
        if solid.any():
            ax.fill_between(
                centres[solid],
                q25[solid],
                q75[solid],
                color=QUADRANT_COLORS[q],
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                centres[solid],
                med[solid],
                color=QUADRANT_COLORS[q],
                lw=1.6,
                marker="o",
                ms=3.0,
                zorder=3,
            )
        weak = finite & (counts < MIN_BIN_N) & (counts > 0)
        if weak.any():
            ax.plot(
                centres[weak],
                med[weak],
                color=QUADRANT_COLORS[q],
                lw=0.9,
                ls=(0, (2, 1.5)),
                marker="o",
                ms=2.5,
                mfc="white",
                mew=0.7,
                alpha=0.7,
                zorder=3,
            )

    ax.text(
        0.99,
        0.02,
        f"n = {len(pts):,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color="0.4",
    )
    ax.tick_params(axis="both", labelsize=5.5, length=2, width=0.4, pad=2)


def _quadrant_means(pts: gpd.GeoDataFrame) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for q in QUADRANTS:
        sub = pts[pts["quadrant"] == q]
        if len(sub) == 0:
            out[q] = (np.nan, 0)
        else:
            out[q] = (float(sub["solar_hours_winter"].mean()), len(sub))
    return out


def _draw_summary(
    ax: plt.Axes, per_site: dict[str, dict[str, tuple[float, int]]], ordered: list[str]
) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Header
    ax.text(
        0.0,
        0.97,
        "Quadrant means (winter solstice, slope ≥ 5°)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color="#222",
    )

    # Column header
    col_x = [0.34, 0.50, 0.66, 0.82]
    for q, x in zip(QUADRANTS, col_x):
        ax.text(
            x,
            0.86,
            q,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            color=QUADRANT_COLORS[q],
        )
    ax.text(
        0.95,
        0.86,
        "N − S",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        fontweight="bold",
        color="#444",
    )

    y = 0.78
    for site in ordered:
        if site not in per_site:
            ax.text(
                0.0,
                y,
                SITE_LABELS[site],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6,
                color="#888",
                style="italic",
            )
            ax.text(
                0.95,
                y,
                "pending",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.5,
                color="#888",
                style="italic",
            )
            y -= 0.105
            continue
        means = per_site[site]
        n_total = sum(n for _, n in means.values())
        ax.text(
            0.0,
            y,
            f"{SITE_LABELS[site]} (n={n_total:,})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            color="#222",
        )
        for q, x in zip(QUADRANTS, col_x):
            v, n = means[q]
            if np.isfinite(v):
                ax.text(
                    x,
                    y,
                    f"{v:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=6,
                    color="#222",
                )
            else:
                ax.text(
                    x,
                    y,
                    "—",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=6,
                    color="#aaa",
                )
        n_v, _ = means["N"]
        s_v, _ = means["S"]
        if np.isfinite(n_v) and np.isfinite(s_v):
            gap = n_v - s_v
            color = "#1F6FB0" if gap > 0 else "#D62728"
            ax.text(
                0.95,
                y,
                f"{gap:+.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6,
                fontweight="bold",
                color=color,
            )
        y -= 0.105

    ax.text(
        0.0,
        y - 0.05,
        "Hillside sites (Vidigal, Rocinha) show N − S > 0:\n"
        "south-facing slopes are structurally shaded on the\n"
        "winter solstice in the southern hemisphere. Flatland\n"
        "fabrics have too few sloped cells to support the test.\n"
        "Maré pending: street-solar pipeline not yet run.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color="#444",
        linespacing=1.5,
    )


def _draw_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], color=QUADRANT_COLORS["N"], lw=1.6, marker="o", ms=3, label="N-facing"),
        Line2D([0], [0], color=QUADRANT_COLORS["E"], lw=1.6, marker="o", ms=3, label="E-facing"),
        Line2D([0], [0], color=QUADRANT_COLORS["S"], lw=1.6, marker="o", ms=3, label="S-facing"),
        Line2D([0], [0], color=QUADRANT_COLORS["W"], lw=1.6, marker="o", ms=3, label="W-facing"),
        Patch(facecolor="0.7", edgecolor="none", alpha=0.4, label="±IQR (n ≥ 30)"),
        Line2D(
            [0],
            [0],
            color="0.4",
            lw=0.9,
            ls=(0, (2, 1.5)),
            marker="o",
            ms=2.5,
            mfc="white",
            label="n < 30 (sparse)",
        ),
    ]
    ax.legend(
        handles=handles,
        title="terrain aspect quadrant",
        title_fontsize=6,
        fontsize=5.5,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        frameon=False,
        ncol=1,
        handlelength=1.5,
    )


def main() -> None:
    apply_style()
    cmap = _build_compass_cmap()

    print("Loading per-site solar+morph point sets ...")
    panel_data: dict[str, gpd.GeoDataFrame | None] = {}
    quadrant_means: dict[str, dict[str, tuple[float, int]]] = {}
    for site in SITE_ORDER:
        pts = _load_site(site)
        panel_data[site] = pts
        if pts is not None:
            quadrant_means[site] = _quadrant_means(pts)
            mn = quadrant_means[site]
            print(
                f"  {site:<22} n={len(pts):5,}  "
                f"N={mn['N'][0]:.2f}  E={mn['E'][0]:.2f}  "
                f"S={mn['S'][0]:.2f}  W={mn['W'][0]:.2f}  "
                f"N-S={mn['N'][0] - mn['S'][0]:+.2f}"
            )
        else:
            print(f"  {site:<22} (no solar — pending)")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.62))
    outer = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.36,
        wspace=0.16,
        left=0.06,
        right=0.97,
        top=0.90,
        bottom=0.08,
    )

    # Site order on the grid: Vidigal, Rocinha, CDA (top); RdP, Maré, summary (bottom).
    placements = [
        ("vidigal", (0, 0)),
        ("rocinha", (0, 1)),
        ("complexo_do_alemao", (0, 2)),
        ("riodaspedras", (1, 0)),
        ("maré", (1, 1)),
    ]
    axes: list[plt.Axes] = []
    for site, (r, c) in placements:
        ax = fig.add_subplot(outer[r, c])
        is_pending = site == "maré"
        _draw_panel(ax, panel_data.get(site), site, cmap, is_pending=is_pending)
        if r == 1:
            ax.set_xlabel("SVF", fontsize=6.5, labelpad=2)
        if c == 0:
            ax.set_ylabel("hours of direct winter sun", fontsize=6.5, labelpad=2)
        axes.append(ax)

    # Bottom-right: quadrant-mean summary table + legend stacked.
    summary_gs = outer[1, 2].subgridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.10)
    ax_sum = fig.add_subplot(summary_gs[0, 0])
    _draw_summary(ax_sum, quadrant_means, list(SITE_ORDER))
    ax_leg = fig.add_subplot(summary_gs[1, 0])
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    for sp in ax_leg.spines.values():
        sp.set_visible(False)
    _draw_legend(ax_leg)

    fig.suptitle(
        "SVF and direct winter sun decouple on sloped terrain — five sites",
        fontsize=9,
        y=0.97,
        fontweight="bold",
        color="#1a1a1a",
    )
    fig.text(
        0.5,
        0.935,
        "Per-aspect-quadrant median solar hours vs SVF on cells with "
        "slope ≥ 5°. Winter sun in the southern hemisphere bypasses "
        "S-facing slopes; this dissociation is visible on hillside sites "
        "and absent on flatland fabrics that have too few sloped cells.",
        ha="center",
        va="top",
        fontsize=5.8,
        style="italic",
        color="#666",
    )

    save_fig(fig, "fig06_terrain_aspect")


if __name__ == "__main__":
    main()
