"""Shared style module for all Brisa+ Nature Cities paper figures.

Defines site colors, morphometric palettes, rcParams, and helper functions.
Every figure script imports this module to ensure visual consistency.

Usage:
    from fig_style import *
    apply_style()
    fig, ax = plt.subplots(figsize=SIZE_DOUBLE)
    # ... plot ...
    save_fig(fig, "fig01_study_sites")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  Nature Cities sizing (mm → inches)
# ═══════════════════════════════════════════════════════════════

_MM_TO_IN = 1 / 25.4

WIDTH_SINGLE = 88 * _MM_TO_IN  # 3.46 in
WIDTH_1_5 = 136 * _MM_TO_IN  # 5.35 in
WIDTH_DOUBLE = 180 * _MM_TO_IN  # 7.09 in
HEIGHT_MAX = 230 * _MM_TO_IN  # 9.06 in

SIZE_SINGLE = (WIDTH_SINGLE, WIDTH_SINGLE * 0.85)
SIZE_DOUBLE = (WIDTH_DOUBLE, WIDTH_DOUBLE * 0.55)
SIZE_DOUBLE_TALL = (WIDTH_DOUBLE, HEIGHT_MAX)

DPI = 600

# ═══════════════════════════════════════════════════════════════
#  Site colors — colorblind-safe, warm=hillside, cool=flatland
#  Based on Tol muted palette
# ═══════════════════════════════════════════════════════════════

SITE_ORDER = [
    "vidigal",
    "rocinha",
    "complexo_do_alemao",
    "riodaspedras",
    "maré",
]

SITE_COLORS = {
    "vidigal": "#CC6677",  # warm rose (hillside)
    "rocinha": "#DDCC77",  # warm sand (hillside)
    "complexo_do_alemao": "#999933",  # olive (mixed)
    "riodaspedras": "#44AA99",  # teal (flatland)
    "maré": "#332288",  # indigo (flatland)
}

SITE_LABELS = {
    "vidigal": "Vidigal",
    "rocinha": "Rocinha",
    "complexo_do_alemao": "C. do Alemão",
    "riodaspedras": "Rio das Pedras",
    "maré": "Maré",
}

SITE_TYPES = {
    "vidigal": "hillside",
    "rocinha": "hillside",
    "complexo_do_alemao": "mixed",
    "riodaspedras": "flatland",
    "maré": "flatland",
}

SITE_MARKERS = {
    "vidigal": "o",
    "rocinha": "s",
    "complexo_do_alemao": "D",
    "riodaspedras": "^",
    "maré": "v",
}

# ═══════════════════════════════════════════════════════════════
#  Morphometric palettes (perceptually uniform)
# ═══════════════════════════════════════════════════════════════

CMAP_SVF = "cividis"
CMAP_LAMBDA_P = "magma"
CMAP_LAMBDA_F = "inferno"
CMAP_SLOPE = "viridis"
CMAP_POROSITY = "cividis_r"
CMAP_SIGMA_H = "magma"

# Diverging for risk/threshold encoding (future CFD figures)
CMAP_RISK = "RdYlBu_r"  # red = failure, blue = adequate

# ═══════════════════════════════════════════════════════════════
#  Stratum thresholds
# ═══════════════════════════════════════════════════════════════

SVF_THRESHOLDS = [0.15, 0.30]
SLOPE_THRESHOLD = 15.0
LAMBDA_P_THRESHOLD = 0.5

# ═══════════════════════════════════════════════════════════════
#  Style application
# ═══════════════════════════════════════════════════════════════


def apply_style() -> None:
    """Set rcParams for Nature Cities publication figures."""
    plt.rcParams.update(
        {
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.title_fontsize": 7,
            # Axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.4,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            # Grid
            "axes.grid": False,
            # Colors — white backgrounds for Nature Cities
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            # Lines
            "lines.linewidth": 0.8,
            "patch.linewidth": 0.4,
            # Math text
            "mathtext.default": "regular",
        }
    )


# ═══════════════════════════════════════════════════════════════
#  Save helper
# ═══════════════════════════════════════════════════════════════


def save_fig(fig: plt.Figure, name: str, dpi: int = DPI) -> None:
    """Save figure as both SVG and PNG to exports/."""
    svg_path = EXPORTS_DIR / f"{name}.svg"
    png_path = EXPORTS_DIR / f"{name}.png"
    kw = dict(bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(svg_path, format="svg", **kw)
    fig.savefig(png_path, format="png", dpi=dpi, **kw)
    plt.close(fig)
    print(f"  Saved {svg_path.name} + {png_path.name}")


# ═══════════════════════════════════════════════════════════════
#  Map helpers
# ═══════════════════════════════════════════════════════════════


def add_scalebar(ax, length_m: float | None = None, loc: str = "lower left") -> None:
    """Add a minimal scale bar to a map axes."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ex = xlim[1] - xlim[0]
    ey = ylim[1] - ylim[0]

    if length_m is None:
        candidates = [10, 20, 50, 100, 200, 500, 1000, 2000]
        length_m = min(candidates, key=lambda v: abs(v - ex * 0.18))

    pad_x = ex * 0.03
    pad_y = ey * 0.03
    bar_h = ey * 0.006

    if "lower" in loc:
        y0 = ylim[0] + pad_y
    else:
        y0 = ylim[1] - pad_y - bar_h
    if "left" in loc:
        x0 = xlim[0] + pad_x
    else:
        x0 = xlim[1] - pad_x - length_m

    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            length_m,
            bar_h,
            facecolor="k",
            edgecolor="k",
            linewidth=0.3,
            zorder=20,
            clip_on=False,
        )
    )

    label = f"{length_m / 1000:.0f} km" if length_m >= 1000 else f"{length_m:.0f} m"
    ax.text(
        x0 + length_m / 2,
        y0 + bar_h + ey * 0.006,
        label,
        ha="center",
        va="bottom",
        fontsize=5,
        zorder=20,
        clip_on=False,
    )


def add_north_arrow(ax, loc: str = "upper right", size: float = 0.04) -> None:
    """Add a minimal north arrow."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ex = xlim[1] - xlim[0]
    ey = ylim[1] - ylim[0]
    arrow_len = ey * size

    if "right" in loc:
        cx = xlim[1] - ex * 0.04
    else:
        cx = xlim[0] + ex * 0.04
    if "upper" in loc:
        base_y = ylim[1] - ey * 0.04 - arrow_len
    else:
        base_y = ylim[0] + ey * 0.04

    ax.annotate(
        "",
        xy=(cx, base_y + arrow_len),
        xytext=(cx, base_y),
        arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8),
        zorder=20,
    )
    ax.text(
        cx,
        base_y + arrow_len + ey * 0.008,
        "N",
        ha="center",
        va="bottom",
        fontsize=5,
        fontweight="bold",
        zorder=20,
    )


def clean_map_axes(ax) -> None:
    """Remove all ticks and spines from a map axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def hillshade(dtm: np.ndarray, res: float, az: float = 315, alt: float = 45) -> np.ndarray:
    """Compute hillshade from a DTM array."""
    dy, dx = np.gradient(dtm, res)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az_r, alt_r = np.radians(az), np.radians(alt)
    shade = np.sin(alt_r) * np.cos(slope) + np.cos(alt_r) * np.sin(slope) * np.cos(az_r - aspect)
    return np.clip(shade, 0, 1)


def stat_annotation(ax, text: str, loc: str = "upper left") -> None:
    """Place a statistical annotation (r, n, p) inside the plot."""
    x = 0.03 if "left" in loc else 0.97
    y = 0.97 if "upper" in loc else 0.03
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=5.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )


# ═══════════════════════════════════════════════════════════════
#  Data loading helpers
# ═══════════════════════════════════════════════════════════════


def load_grid(site: str):
    """Load the morphometric grid for a site."""
    import geopandas as gpd

    path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"Grid not found: {path}")
    return gpd.read_file(path)


def load_buildings(site: str, extended: bool = True):
    """Load building footprints (extended or original)."""
    import geopandas as gpd

    if extended:
        path = PROJECT_ROOT / "data" / site / "buildings_extended_300m.gpkg"
        if path.exists():
            return gpd.read_file(path)
    # Fallback to original via paths module
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.svf_v2.paths import resolve_paths

    _, fp, _ = resolve_paths(site)
    return gpd.read_file(fp)


def load_boundary(site: str):
    """Load the favela boundary polygon."""
    import sys

    import geopandas as gpd

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.svf_v2.paths import resolve_boundary

    bp = resolve_boundary(site)
    if bp is None:
        raise FileNotFoundError(f"No boundary for {site}")
    return gpd.read_file(bp)


def load_campaign_patches(site: str):
    """Load campaign patch allocations for a site."""
    import pandas as pd

    path = (
        PROJECT_ROOT
        / "outputs"
        / site
        / "sampling_cfd"
        / "campaign_sampling"
        / "campaign_patches.csv"
    )
    if not path.exists():
        # Fallback to pilot
        path = (
            PROJECT_ROOT
            / "outputs"
            / site
            / "sampling_cfd"
            / "pilot_sampling"
            / "pilot_patches.csv"
        )
    if not path.exists():
        raise FileNotFoundError(f"No patches found for {site}")
    return pd.read_csv(path)


def load_dtm_path(site: str, extended: bool = True) -> Path:
    """Get DTM path for a site."""
    if extended:
        path = PROJECT_ROOT / "data" / site / "dtm_extended_300m.tif"
        if path.exists():
            return path
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.svf_v2.paths import resolve_paths

    dtm, _, _ = resolve_paths(site)
    return Path(dtm)
