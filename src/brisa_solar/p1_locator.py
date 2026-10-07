"""P1 Figure 1 — study-site locator, drawn in the shared P1 house style.

Replaces outputs/paper_figures/fig01_study_sites.py for the manuscript (that
script stays for the older technical-report figure). Differences that matter:
sites in the fixed order everywhere; the outlines are the run-of-record IPP
2019 favela polygons the solar numbers use (wp05_full.match_favela_group), so
for Maré the inset shows the six modelled polygons inside the wider complex;
no building counts (each figure names its own denominator); the citywide
urban fabric the percentiles are computed on is drawn as a faint extent so the
"citywide distribution" has a visible referent.

Map hygiene (red_lines.md §5): no coordinate axes or graticule, footprints are
rasterised in the SVG so no per-building geometry is extractable, and the
fabric extent is an occupancy mask at OVERVIEW_PIXEL_M, never per-point values.

Run: python -m src.brisa_solar.p1_locator   -> runs/p1_locator_<UTC>/
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from . import p1_style as style  # noqa: E402
from .constants import REPO_ROOT  # noqa: E402
from .wp07_ledger import FAVELAS, RUN_OF_RECORD  # noqa: E402

#: Rendering choices, not measurements.
OVERVIEW_DTM_DECIMATE = 6
OVERVIEW_PIXEL_M = 100.0
INSET_PAD_FRAC = 0.12
BUILDINGS_REL = "data/RJ/buildings_RJ_2019_utm.gpkg"
FAVELAS_REL = "data/RJ/Favelas_Limit_2019.shp"
DTM_REL = "data/RJ/DTM_RJ.tif"
OTHER_BUILDINGS = "#DCD3C3"

#: Hand-set label offsets (points) on the overview so the five names never
#: collide; the leader line, not the text box, carries the location.
LABEL_OFFSET = {
    "vidigal": (-10, -22, "right", "top"),
    "rocinha": (-26, 16, "right", "bottom"),
    "complexo_do_alemao": (-8, 22, "right", "bottom"),
    "mare": (18, 12, "left", "bottom"),
    "riodaspedras": (-6, -22, "center", "top"),
}


def _hillshade(dtm: np.ndarray, res: float, az: float = 315, alt: float = 45) -> np.ndarray:
    dy, dx = np.gradient(dtm, res)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az_r, alt_r = np.radians(az), np.radians(alt)
    shade = np.sin(alt_r) * np.cos(slope) + np.cos(alt_r) * np.sin(slope) * np.cos(az_r - aspect)
    return np.clip(shade, 0, 1)


def _scalebar(ax, length_m: float, label: str, x0_frac: float = 0.04, y0_frac: float = 0.05) -> None:
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    bx = x0 + (x1 - x0) * x0_frac
    by = y0 + (y1 - y0) * y0_frac
    ax.plot([bx, bx + length_m], [by, by], color="black", linewidth=1.6, solid_capstyle="butt",
            zorder=20)
    ax.text(bx + length_m / 2, by + (y1 - y0) * 0.02, label, ha="center", va="bottom",
            fontsize=style.BASE_PT, zorder=20,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))


def _nice_length(span_m: float) -> float:
    target = span_m * 0.25
    for v in (50, 100, 200, 250, 500, 1000, 2000, 5000, 10000):
        if v >= target * 0.6:
            return float(v)
    return 10000.0


def _site_polygons(repo_root: Path) -> dict:
    from src.config import EXPECTED_CRS
    from .wp05_full import match_favela_group

    gdf = gpd.read_file(repo_root / FAVELAS_REL)
    if gdf.crs is not None:
        gdf = gdf.to_crs(EXPECTED_CRS)
    out = {}
    for slug in style.SITE_ORDER:
        polys, _ = match_favela_group(gdf, FAVELAS[slug])
        out[slug] = polys
    return out, gdf


def _fabric_mask(repo_root: Path, bounds) -> tuple[np.ndarray, tuple]:
    """Occupancy of the citywide sampling frame at OVERVIEW_PIXEL_M — where
    the pooled citywide distribution lives, nothing about its values."""
    xmin, ymin, xmax, ymax = bounds
    nx = int(np.ceil((xmax - xmin) / OVERVIEW_PIXEL_M))
    ny = int(np.ceil((ymax - ymin) / OVERVIEW_PIXEL_M))
    occ = np.zeros(nx * ny, dtype=bool)
    pf = pq.ParquetFile(repo_root / "runs" / RUN_OF_RECORD["wp05"] / "frame_cells.parquet")
    for batch in pf.iter_batches(columns=["x", "y"], batch_size=1_000_000):
        x = batch.column(0).to_numpy()
        y = batch.column(1).to_numpy()
        c = np.clip(((x - xmin) / OVERVIEW_PIXEL_M).astype(np.int64), 0, nx - 1)
        r = np.clip(((ymax - y) / OVERVIEW_PIXEL_M).astype(np.int64), 0, ny - 1)
        occ[r * nx + c] = True
    return occ.reshape(ny, nx), (xmin, xmax, ymin, ymax)


def render(repo_root: Path, out_dir: Path) -> dict:
    fig_id = "p1_fig1_study_sites"
    polys, all_favelas = _site_polygons(repo_root)
    with plt.rc_context(style.rc()):
        style._alias_arial()
        fig = plt.figure(figsize=(style.WIDTH_DOUBLE_IN, 122 * style.MM))
        ax = fig.add_axes([0.0, 0.33, 1.0, 0.67])

        with rasterio.open(repo_root / DTM_REL) as src:
            f = OVERVIEW_DTM_DECIMATE
            dtm = src.read(1, out_shape=(src.height // f, src.width // f)).astype(float)
            if src.nodata is not None:
                dtm[dtm == src.nodata] = np.nan
            dtm[np.abs(dtm) > 1e6] = np.nan
            b = src.bounds
            shade = _hillshade(np.nan_to_num(dtm, nan=0.0), src.res[0] * f)
        land = np.isfinite(dtm)
        rgba = np.zeros(dtm.shape + (4,))
        rgba[..., :3] = (0.93 + 0.07 * shade[..., None]) * np.array([0.96, 0.95, 0.92])
        rgba[..., :3] -= (1 - shade[..., None]) * 0.28
        rgba[..., 3] = land.astype(float)
        extent = (b.left, b.right, b.bottom, b.top)
        ax.imshow(np.clip(rgba, 0, 1), extent=extent, interpolation="bilinear", zorder=0)

        occ, occ_ext = _fabric_mask(repo_root, (b.left, b.bottom, b.right, b.top))
        ax.imshow(np.where(occ, 1.0, np.nan), extent=occ_ext, cmap=ListedColormap(["#CFC6E3"]),
                  alpha=0.55, interpolation="nearest", zorder=1)

        for slug in style.SITE_ORDER:
            p = polys[slug]
            p.plot(ax=ax, facecolor=style.SITE_COLORS[slug], edgecolor="black", linewidth=0.3, zorder=3)
            c = p.geometry.union_all().centroid
            dx, dy, ha, va = LABEL_OFFSET[slug]
            ax.annotate(FAVELAS[slug], xy=(c.x, c.y), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=style.BASE_PT, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="#222222", lw=0.5, shrinkA=0, shrinkB=2),
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)
            ax.plot([c.x], [c.y], marker=style.SITE_MARKERS[slug], markersize=5,
                    color=style.SITE_COLORS[slug], markeredgecolor="black", markeredgewidth=0.5, zorder=5)
        ax.set_xlim(b.left, b.right)
        ax.set_ylim(b.bottom, b.top)
        ax.set_aspect("equal")
        ax.set_axis_off()
        _scalebar(ax, 10000.0, "10 km", x0_frac=0.03, y0_frac=0.06)
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        nx_, ny_ = x0 + (x1 - x0) * 0.035, y0 + (y1 - y0) * 0.80
        ax.annotate("N", xy=(nx_, ny_ + (y1 - y0) * 0.10), xytext=(nx_, ny_), ha="center", va="top",
                    fontsize=style.BASE_PT, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))
        style.panel_letter(ax, "A", x=0.01, y=0.97)
        ax.texts[-1].set_ha("left")
        ax.texts[-1].set_va("top")

        # key for the overview
        from matplotlib.patches import Patch
        ax.legend([Patch(fc="#CFC6E3", alpha=0.55, ec="none"), Patch(fc="white", ec="black", lw=0.3)],
                  ["urban fabric of the citywide distribution", "study favela (IPP 2019 polygons)"],
                  loc="lower right", frameon=False, fontsize=style.BASE_PT, handlelength=1.2)

        # ---- B: insets, fixed order ------------------------------------------
        n = len(style.SITE_ORDER)
        gap = 0.012
        w = (1.0 - gap * (n + 1)) / n
        for k, slug in enumerate(style.SITE_ORDER):
            axi = fig.add_axes([gap + k * (w + gap), 0.035, w, 0.25])
            p = polys[slug]
            xmin, ymin, xmax, ymax = p.total_bounds
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            # fit the polygons into the slot's own aspect so every inset fills it
            aspect = (0.25 * fig.get_figheight()) / (w * fig.get_figwidth())
            half_w = max((xmax - xmin) / 2, (ymax - ymin) / 2 / aspect) * (1 + 2 * INSET_PAD_FRAC)
            half_h = half_w * aspect
            bbox = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
            bld = gpd.read_file(repo_root / BUILDINGS_REL, bbox=bbox)
            inside = bld.geometry.intersects(p.geometry.union_all())
            bld[~inside].plot(ax=axi, facecolor=OTHER_BUILDINGS, edgecolor="none", rasterized=True, zorder=1)
            bld[inside].plot(ax=axi, facecolor=style.SITE_COLORS[slug], edgecolor="none",
                             rasterized=True, zorder=2)
            p.boundary.plot(ax=axi, color="black", linewidth=0.7, zorder=3)
            axi.set_xlim(bbox[0], bbox[2])
            axi.set_ylim(bbox[1], bbox[3])
            axi.set_aspect("equal")
            axi.set_xticks([])
            axi.set_yticks([])
            for sp in axi.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.4)
            axi.text(0.03, 0.97, FAVELAS[slug], transform=axi.transAxes, ha="left", va="top",
                     fontsize=style.BASE_PT, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=10)
            L = _nice_length(bbox[2] - bbox[0])
            lbl = f"{L / 1000:g} km" if L >= 1000 else f"{L:g} m"
            _scalebar(axi, L, lbl, x0_frac=0.05, y0_frac=0.05)
            if k == 0:
                style.panel_letter(axi, "B", x=0.0, y=1.02)
                axi.texts[-1].set_ha("left")
        fig.text(1.0 - gap, 0.292, "outline: modelled area (IPP 2019 favela polygons); coloured: building "
                 "footprints inside it; beige: other footprints", ha="right", va="bottom", fontsize=style.BASE_PT)

        min_pt = style.min_text_pt(fig)
        assert min_pt >= style.MIN_PT, f"{fig_id}: text at {min_pt} pt"
        svg_path, png_path = out_dir / f"{fig_id}.svg", out_dir / f"{fig_id}.png"
        fig.savefig(svg_path, format="svg", dpi=300)
        fig.savefig(png_path, format="png", dpi=style.DPI)
        plt.close(fig)

    raw = svg_path.read_text()
    import re
    texts = " ".join(re.findall(r"<text\b[^>]*>(.*?)</text>", raw, re.S))
    return {
        "id": fig_id, "status": "produced", "svg_path": svg_path.name, "png_path": png_path.name,
        "release_class_proposed": "staged",
        "sources": {"favela_polygons": FAVELAS_REL, "buildings": BUILDINGS_REL, "dtm": DTM_REL,
                    "fabric_extent": f"runs/{RUN_OF_RECORD['wp05']}/frame_cells.parquet "
                                     f"(occupancy at {OVERVIEW_PIXEL_M:g} m)"},
        "matched_polygons": {s: sorted(polys[s]["nome"].astype(str).tolist()) for s in style.SITE_ORDER},
        "checklist": {
            "no_coordinate_text": re.search(r"(?<!\d)\d{6,7}(?!\d)", texts) is None,
            "no_axes_ticks": True,
            "footprints_rasterised_in_svg": "<image" in raw,
            "sites_fixed_order": list(style.SITE_ORDER),
            "min_text_pt": round(float(min_pt), 2),
            "final_width_mm": round(float(style.WIDTH_DOUBLE_IN * 25.4), 1),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    repo_root = Path(args.repo_root)
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "runs" / (
        "p1_locator_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = render(repo_root, out_dir)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    manifest = {"_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_sha": sha,
                "figures": {fig["id"]: fig}}
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({"out_dir": str(out_dir), "figure": fig["png_path"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
