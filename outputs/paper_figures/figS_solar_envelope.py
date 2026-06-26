"""Parameterised street-level solar envelope (winter / annual / summer).

Three panels on the same point set, same colour scale [0, 12] hours,
same buildings backdrop. Site-agnostic: the canonical Vidigal version
(``fig07_solar_envelope_vidigal.py``) is a 4-line wrapper that
delegates here; per-site SI figures (Rocinha, Complexo do Alemão,
Rio das Pedras, Maré) call ``render_envelope("rocinha")``, etc.

The colour ramp (``YlOrRd_r``: deep red 0 h → orange → pale yellow 12 h)
matches the legacy ``solar_access_streets.png`` from the
2026-03 single-date run, so visual continuity with anything previously
shown to collaborators is preserved.

Inputs per site:

* ``outputs/{site}/morphometrics/svf/svf_streets_solar.gpkg`` —
  produced by ``scripts/run_street_solar.py``.
* ``data/{site}/<buildings>.shp`` — via ``fig_style.load_buildings``
  for the backdrop layer.

Outputs (under ``outputs/paper_figures/exports/``):

* ``{stem}.png`` / ``{stem}.svg`` — the combined 1×3 figure.
* ``{stem}_winter.png``, ``{stem}_annual.png``, ``{stem}_summer.png``
  — standalone single-panel versions, useful for slide decks where
  one season is the focus.

The shared 0–12 h scale across all sites makes Maré (flat, ~all-summer)
and Vidigal (sloped, ~all-winter-shaded) directly comparable.

Usage::

    python outputs/paper_figures/figS_solar_envelope.py --site rocinha
    python outputs/paper_figures/figS_solar_envelope.py --site maré --stem fig07d_solar_mare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (
    PROJECT_ROOT,
    SITE_LABELS,
    WIDTH_DOUBLE,
    add_north_arrow,
    add_scalebar,
    apply_style,
    clean_map_axes,
    load_buildings,
    save_fig,
)

VMAX_HOURS = 12.0  # shared colour ceiling so panels are directly comparable
CMAP = "YlOrRd_r"  # legacy palette (reversed): deep red (0 h) → orange → pale yellow (12 h)


def _load(site: str) -> gpd.GeoDataFrame:
    path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    if not path.exists():
        raise FileNotFoundError(
            f"Solar gpkg missing for {site!r}: {path}\n"
            f"Run `python scripts/run_street_solar.py --site {site}` first."
        )
    return gpd.read_file(path)


def _draw_panel(
    ax: plt.Axes,
    pts: gpd.GeoDataFrame,
    column: str,
    title: str,
    buildings: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
) -> object:
    xmin, ymin, xmax, ymax = bbox
    buildings.plot(ax=ax, facecolor="0.86", edgecolor="0.62", linewidth=0.18, zorder=1)
    sc = ax.scatter(
        pts.geometry.x,
        pts.geometry.y,
        c=pts[column],
        cmap=CMAP,
        vmin=0,
        vmax=VMAX_HOURS,
        s=4,
        alpha=0.95,
        linewidths=0,
        zorder=2,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    clean_map_axes(ax)
    ax.set_title(title, fontsize=8, loc="left", pad=2)

    # Inline stats card (the only quantitative anchor in the standalone panel).
    mean_h = float(pts[column].mean())
    med_h = float(pts[column].median())
    zero_pct = 100.0 * float((pts[column] <= 0.01).mean())
    ax.text(
        0.02,
        0.98,
        f"mean {mean_h:4.2f} h\nmedian {med_h:4.2f} h\nzero-share {zero_pct:4.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.92),
        zorder=4,
    )
    return sc


def _save_standalone_panel(
    pts: gpd.GeoDataFrame,
    column: str,
    title: str,
    buildings: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE * 0.6, WIDTH_DOUBLE * 0.55))
    sc = _draw_panel(ax, pts, column=column, title=title, buildings=buildings, bbox=bbox)
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.7, pad=0.04, aspect=24)
    cbar.set_label("hours of direct sun", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    add_scalebar(ax, length_m=200)
    add_north_arrow(ax)
    fig.tight_layout()
    save_fig(fig, name)


def render_envelope(
    site: str,
    stem: str | None = None,
    suptitle: str | None = None,
    save_standalones: bool = True,
    standalone_stems: dict[str, str] | None = None,
) -> None:
    """Render the 1×3 winter / annual / summer envelope for one site.

    Parameters
    ----------
    site : str
        Site key, e.g. ``"rocinha"``.
    stem : str, optional
        Filename stem for the combined PNG. Defaults to
        ``figS_solar_envelope_{site_safe}``.
    suptitle : str, optional
        Override the figure suptitle. Defaults to a generic
        site-aware caption.
    save_standalones : bool, default True
        Also write the three single-panel PNGs.
    standalone_stems : dict[str, str], optional
        Override the per-panel filename stems. Keys must be
        ``"winter"``, ``"annual"``, ``"summer"``; values are full
        filename stems. Defaults to ``{stem}_{season}`` per panel.
        Used by the canonical fig07 wrapper to keep the
        ``fig07a_solar_winter.png`` legacy naming.
    """
    apply_style()
    pts = _load(site)
    buildings = load_buildings(site, extended=False)
    if buildings.crs != pts.crs:
        buildings = buildings.to_crs(pts.crs)

    bbox_arr = pts.total_bounds
    pad = 30.0
    bbox = (
        bbox_arr[0] - pad,
        bbox_arr[1] - pad,
        bbox_arr[2] + pad,
        bbox_arr[3] + pad,
    )

    site_safe = site.replace("ã", "a").replace(" ", "_")
    site_label = SITE_LABELS.get(site, site.capitalize())
    base = stem or f"figS_solar_envelope_{site_safe}"

    if standalone_stems is None:
        standalone_stems = {
            "winter": f"{base}_winter",
            "annual": f"{base}_annual",
            "summer": f"{base}_summer",
        }

    panels = [
        (
            "solar_hours_winter",
            "(a) winter solstice — 21 Jun (worst case)",
            standalone_stems["winter"],
        ),
        (
            "solar_hours_annual",
            "(b) annual proxy — mean of 4 ref. dates",
            standalone_stems["annual"],
        ),
        (
            "solar_hours_summer",
            "(c) summer solstice — 21 Dec (best case)",
            standalone_stems["summer"],
        ),
    ]

    if save_standalones:
        for column, title, name in panels:
            _save_standalone_panel(
                pts, column=column, title=title, buildings=buildings, bbox=bbox, name=name
            )

    # Combined 1×3 figure.
    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.55))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.06],
        hspace=0.12,
        wspace=0.04,
        left=0.02,
        right=0.985,
        top=0.84,
        bottom=0.10,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[1, :])

    last_sc = None
    for ax, (column, title, _) in zip(axes, panels):
        last_sc = _draw_panel(ax, pts, column=column, title=title, buildings=buildings, bbox=bbox)

    add_scalebar(axes[0], length_m=200)
    add_north_arrow(axes[2])

    cbar = fig.colorbar(last_sc, cax=cax, orientation="horizontal")
    cbar.set_label("hours of direct sun  (shared scale 0 – 12 h)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    if suptitle is None:
        suptitle = (
            f"{site_label} — street-level solar envelope.  "
            "Same points, same scale, three reference cases.\n"
            "Winter is the worst case; summer the best; the annual proxy is the figure to use "
            "for year-round comfort and PV studies."
        )
    fig.suptitle(suptitle, fontsize=8.5, y=0.97)

    save_fig(fig, base)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--site", required=True, help="Site key (e.g. 'rocinha').")
    parser.add_argument("--stem", default=None, help="Override filename stem.")
    parser.add_argument(
        "--no-standalones",
        action="store_true",
        help="Skip the three single-panel PNGs.",
    )
    args = parser.parse_args()
    render_envelope(site=args.site, stem=args.stem, save_standalones=not args.no_standalones)
    return 0


if __name__ == "__main__":
    sys.exit(main())
