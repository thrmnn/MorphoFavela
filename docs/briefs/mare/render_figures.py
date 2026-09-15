#!/usr/bin/env python3
"""Render the Maré brief's figures fresh from the outputs of record, plus
copy the one already-band-classed publishable figure and write
figure_manifest.json.

All rendered maps: band-classed (4-5 discrete classes), no basemap, no
coordinate axes/ticks, scale bar + north arrow only. Called from
build_brief.py; can also run standalone.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = Path(__file__).resolve().parent
SITE = "maré"

ACCENT = "#2A5FA5"
BAND_CMAP_5 = ListedColormap(["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"])
BAND_CMAP_4 = ListedColormap(["#eff3ff", "#9ecae1", "#4292c6", "#08519c"])


def _no_coord_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _scale_bar_and_north(ax, gdf, length_m=200):
    minx, miny, maxx, maxy = gdf.total_bounds
    span = maxx - minx
    x0 = minx + 0.05 * span
    y0 = miny + 0.05 * (maxy - miny)
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", linewidth=2, solid_capstyle="butt")
    ax.text(x0 + length_m / 2, y0 + 0.015 * (maxy - miny), f"{length_m} m",
            ha="center", va="bottom", fontsize=6)
    nx = maxx - 0.06 * span
    ny = miny + 0.10 * (maxy - miny)
    ax.annotate("N", xy=(nx, ny + 0.06 * (maxy - miny)), xytext=(nx, ny),
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.2))


def _band_classes(values, edges):
    labels = [f"{edges[i]:.2g}–{edges[i+1]:.2g}" for i in range(len(edges) - 1)]
    return BoundaryNorm(edges, len(edges) - 1), labels


def _plot_grid_layer(ax, gdf, col, edges, cmap, title):
    norm, labels = _band_classes(gdf[col], edges)
    gdf.plot(column=col, ax=ax, cmap=cmap, norm=norm, edgecolor="none", missing_kwds={"color": "#f2f2f2"})
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    _no_coord_axes(ax)
    return norm, labels, cmap


def render_built_form_maps(grid: gpd.GeoDataFrame, out_path: Path) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.6))
    layers = [
        ("lambda_p", [0, 0.2, 0.4, 0.6, 0.8, 1.0], BAND_CMAP_5, "Plan density (λp)"),
        ("H_mean", None, BAND_CMAP_5, "Mean building height"),
        ("porosity", [0, 0.2, 0.4, 0.6, 0.8, 1.0], BAND_CMAP_5, "Porosity"),
        ("svf", [0, 0.2, 0.4, 0.6, 0.8, 1.0], BAND_CMAP_5, "Sky View Factor"),
    ]
    h_vals = grid.loc[grid["H_mean"].notna(), "H_mean"]
    h_edges = list(np.quantile(h_vals, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    for ax, (col, edges, cmap, title) in zip(axes.flat, layers):
        e = edges if edges is not None else h_edges
        _plot_grid_layer(ax, grid, col, e, cmap, title)
    _scale_bar_and_north(axes.flat[0], grid)
    fig.suptitle("Maré — built-form indicators (10 m grid)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return {
        "classes": {
            "lambda_p": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "H_mean_m": [round(float(x), 2) for x in h_edges],
            "porosity": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "svf": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }
    }


def render_street_svf_map(segments: gpd.GeoDataFrame, out_path: Path) -> dict:
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    norm, labels = _band_classes(segments["svf_median"], edges)
    segments.plot(column="svf_median", ax=ax, cmap=BAND_CMAP_5, norm=norm, linewidth=1.2)
    ax.set_title("Maré — street-segment Sky View Factor", fontsize=9)
    ax.set_aspect("equal")
    _no_coord_axes(ax)
    _scale_bar_and_north(ax, segments, length_m=500)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return {"classes": {"svf_median": edges}}


def render_distributions(grid: gpd.GeoDataFrame, segments: gpd.GeoDataFrame, out_path: Path) -> dict:
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.6))
    panels = [
        (grid["lambda_p"], "λp (grid)"),
        (grid.loc[grid["H_mean"].notna(), "H_mean"], "Mean height, m (grid)"),
        (grid["svf"], "SVF (grid)"),
        (segments["svf_median"], "SVF (street segments)"),
    ]
    for ax, (series, title) in zip(axes, panels):
        ax.hist(series.dropna(), bins=24, color=ACCENT, alpha=0.85)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylabel("cells" if "grid" in title else "segments", fontsize=6)
    fig.suptitle("Maré — indicator distributions", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return {}


def render_wind_rose(wind_rose: dict, out_path: Path) -> dict:
    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    freqs = [wind_rose["frequencies"][s] for s in sectors]
    angles = np.deg2rad(np.linspace(0, 360, len(sectors), endpoint=False))
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(111, projection="polar")
    ax.bar(angles, freqs, width=2 * np.pi / len(sectors) * 0.85, color=ACCENT, alpha=0.85, edgecolor="white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(sectors, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title(f"Maré — wind-direction frequency\n({wind_rose['station_name']})", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return {}


def render_all(outputs_root: Path, figures_dir: Path) -> list[dict]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    grid = gpd.read_file(outputs_root / SITE / "morphometrics" / "grid" / "grid_metrics.gpkg")
    segments = gpd.read_file(outputs_root / SITE / "svf_v2" / "svf_streets_segments.gpkg")
    data_root = outputs_root.parent / "data"
    wind_rose = json.loads((data_root / SITE / "wind_rose.json").read_text())

    manifest = []

    p = figures_dir / "fig_built_form_maps.png"
    extra = render_built_form_maps(grid, p)
    manifest.append({
        "file": p.name, "class": "band-classed map, freshly rendered",
        "layers": ["lambda_p", "H_mean", "porosity", "svf"],
        "n_classes": 5, "basemap": False, "coordinate_ticks": False,
        "source": "morphometrics/grid/grid_metrics.gpkg", **extra,
    })

    p = figures_dir / "fig_street_svf_map.png"
    extra = render_street_svf_map(segments, p)
    manifest.append({
        "file": p.name, "class": "band-classed map, freshly rendered",
        "layers": ["street_svf"], "n_classes": 5, "basemap": False,
        "coordinate_ticks": False, "source": "svf_v2/svf_streets_segments.gpkg", **extra,
    })

    p = figures_dir / "fig_distributions.png"
    render_distributions(grid, segments, p)
    manifest.append({
        "file": p.name, "class": "distribution (histogram), freshly rendered",
        "layers": ["lambda_p", "H_mean", "svf", "street_svf"],
        "basemap": False, "coordinate_ticks": False,
        "source": "morphometrics/grid/grid_metrics.gpkg, svf_v2/svf_streets_segments.gpkg",
    })

    p = figures_dir / "fig_wind_rose.png"
    render_wind_rose(wind_rose, p)
    manifest.append({
        "file": p.name, "class": "wind rose, freshly rendered",
        "layers": ["wind_frequency"], "basemap": False, "coordinate_ticks": False,
        "source": "data/maré/wind_rose.json",
    })

    src_diag = outputs_root / SITE / "paper_figures" / "fig_maré_diagnostic_map.png"
    dst_diag = figures_dir / "fig_mare_diagnostic_map.png"
    shutil.copy2(src_diag, dst_diag)
    manifest.append({
        "file": dst_diag.name, "class": "publishable (already band-classed; copied verbatim)",
        "layers": ["diagnostic_classification"], "basemap": False, "coordinate_ticks": False,
        "source": "paper_figures/fig_maré_diagnostic_map.png",
    })

    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, required=True)
    ap.add_argument("--figures-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--manifest-path", type=Path, default=HERE / "figure_manifest.json")
    args = ap.parse_args()

    manifest = render_all(args.outputs_root, args.figures_dir)
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"render_figures: wrote {len(manifest)} figures -> {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
