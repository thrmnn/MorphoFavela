"""Version A (realistic) sunlight-texture tile — 3 STL variants for review.

    python scripts/build_texture_tile.py                    # VDG-P07, all 3 variants
    python scripts/build_texture_tile.py --site maré --patch MAR-P20

Produces, from one shared base tile (true terrain + smooth buildings), three
ground-only sun-hours texture treatments as separate watertight STLs:
  V1 stipple · V2 contour bands · V3 directional hatching
plus one review figure and a per-variant print-risk + recommendation note.
Output → outputs/{site}/print/texture_tile/.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.print3d.model import ROOT
from src.print3d.texture import VARIANTS, build_tile, sample_tile

CLASS_COLORS = ["#f4c430", "#f5deb3", "#8aa1b4", "#334155"]  # Class1 sun … Class4 shade
CLASS_LABELS = ["Class 1 · >6 h", "Class 2 · 4–6 h", "Class 3 · 2–4 h", "Class 4 · <2 h"]
VARIANT_TITLE = {"stipple": "V1 · stippling", "contour": "V2 · contour bands", "hatch": "V3 · hatching"}


def _texture_hillshade(disp, ground_mask, ax, title):
    """Show the texture relief (not the macro terrain) so pits/grooves/hatches read."""
    ls = LightSource(azdeg=315, altdeg=30)
    d = np.where(ground_mask, disp, np.nan)
    filled = np.where(np.isnan(d), 0.0, d)
    rgb = ls.shade(filled, cmap=plt.cm.gray, vert_exag=60, blend_mode="soft")
    rgb[~ground_mask] = [0.62, 0.49, 0.36, 1.0]  # buildings tinted, untextured
    ax.imshow(rgb, origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def _detail(disp, ground_mask, ax, title, crop):
    y0, y1, x0, x1 = crop
    _texture_hillshade(disp[y0:y1, x0:x1], ground_mask[y0:y1, x0:x1], ax, title)


def review_figure(tile, results, out_png):
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(
        f"Version A — realistic sunlight-texture tile · {tile.site}/{tile.patch} · "
        f"150 mm · winter sun-hours on ground only",
        fontsize=14, fontweight="bold", x=0.01, ha="left",
    )

    # input sun-class map
    ax = fig.add_subplot(2, 4, 1)
    cmap = ListedColormap(CLASS_COLORS)
    sc = np.where(tile.ground_mask, tile.shade, np.nan)  # 0..3 = Class1..4
    ax.imshow(sc, origin="upper", cmap=cmap, vmin=0, vmax=3)
    bld = np.ma.masked_where(tile.ground_mask, np.ones_like(tile.shade))
    ax.imshow(bld, origin="upper", cmap=ListedColormap(["#9c6f52"]))
    ax.set_title("input · winter sun-hours class", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS]
    ax.legend(handles, CLASS_LABELS, fontsize=6, loc="lower left", framealpha=0.9)

    # a mixed-shade crop window (center third) for the detail row
    n = tile.shade.shape[0]
    crop = (n // 3, n // 3 + n // 4, n // 3, n // 3 + n // 4)

    for k, (variant, (mesh, stats, disp)) in enumerate(results.items()):
        _texture_hillshade(disp, tile.ground_mask, fig.add_subplot(2, 4, 2 + k),
                           f"{VARIANT_TITLE[variant]} · {stats.texture_depth_mm} mm deep")
        _detail(disp, tile.ground_mask, fig.add_subplot(2, 4, 6 + k),
                f"{VARIANT_TITLE[variant]} — detail (≈{tile.tile_mm/4:.0f} mm)", crop)

    fig.text(0.01, 0.01,
             "Texture is a vertical z-displacement on ground cells only; building roofs stay smooth. "
             "Hillshade exaggerates the sub-mm relief so the pattern reads.",
             fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.subplots_adjust(hspace=0.18)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=125)
    plt.close(fig)
    return out_png


def print_time_hours(height_mm, footprint_mm2, layer=0.2):
    """Very rough FDM estimate (slicer-confirmed needed): layer count × a
    footprint-scaled per-layer time. Bracketed, not precise."""
    layers = height_mm / layer
    per_layer_s = 22 + footprint_mm2 / 900  # travel+perimeter+infill, coarse
    lo = layers * per_layer_s / 3600
    return lo, lo * 1.5


NOTES = {
    "stipple": (
        "Pits are 1.0 mm⌀ × 0.5 mm hemispheres on a jittered grid (spacing 4.0/2.5/1.5 mm "
        "for Class 2/3/4). At a 0.4 mm nozzle a 1 mm pit is ~2.5 nozzle-widths and 0.5 mm is "
        "2–3 layers — it will register but rounds off; the density gradient survives even when "
        "individual pits blur. Robust: pits are isolated concavities, no thin standing walls. "
        "Resin renders each pit crisply."),
    "contour": (
        "Continuous 0.6 mm × 0.4 mm grooves on class boundaries + a 0.2 mm/class step-down. "
        "0.6 mm ≈ 1.5 nozzle-widths, so grooves print as a shallow valley, not a crisp line, and "
        "the 0.2 mm steps are ~1 layer — near the FDM floor. RISK: where a class boundary runs "
        "parallel to a steep terrain break the groove competes with the slope and reads ambiguously "
        "(flagged cells sit on >20° ground). Best on flatter fabric; resin holds the line width."),
    "hatch": (
        "Parallel 0.5 mm × 0.3 mm grooves along the shadow azimuth, spacing by class. 0.5 mm is "
        "~1.25 nozzle-widths and 0.3 mm is the stated FDM engraving floor — the shallowest of the "
        "three, most at risk of being smeared by a 0.4 mm nozzle. On slopes >20° the fixed-azimuth "
        "grooves alias against the draped grid. Directionality is legible; absolute depth is marginal "
        "on FDM, fine on resin."),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="vidigal")
    ap.add_argument("--patch", default="VDG-P07")
    ap.add_argument("--tile-mm", type=float, default=150.0)
    ap.add_argument("--cell", type=float, default=0.30, help="print grid cell, mm")
    args = ap.parse_args()

    tile = sample_tile(args.site, args.patch, tile_mm=args.tile_mm, model_cell_mm=args.cell)
    out = ROOT / "outputs" / args.site / "print" / "texture_tile"
    out.mkdir(parents=True, exist_ok=True)

    dist = {CLASS_LABELS[3 - s].split(" ·")[0]: int((tile.shade == s).sum())
            for s in range(4)}
    print(f"\n  tile {args.site}/{args.patch}  {args.tile_mm:.0f} mm  1:{int(round(1000/tile.mm_per_m))}  "
          f"grid {tile.shade.shape[1]}×{tile.shade.shape[0]} @ {tile.world_cell:.2f} m")
    print(f"  ground sun-class cells: {dist}")

    results = {}
    for variant in VARIANTS:
        mesh, stats, disp = build_tile(tile, variant)
        stem = f"{args.patch}_texture_{variant}"
        mesh.export(out / f"{stem}.stl")
        (out / f"{stem}.json").write_text(json.dumps(asdict(stats), indent=2))
        results[variant] = (mesh, stats, disp)
        w, d, h = stats.model_mm
        lo, hi = print_time_hours(h, w * d)
        mb = (out / f"{stem}.stl").stat().st_size / 1e6
        print(f"\n  {VARIANT_TITLE[variant]}")
        print(f"    {w}×{d}×{h} mm · texture {stats.texture_depth_mm} mm · "
              f"{stats.triangles:,} tris · {mb:.0f} MB · watertight {stats.watertight}")
        print(f"    est. FDM print (0.2 mm, rough): {lo:.0f}–{hi:.0f} h")
        print(f"    note: {NOTES[variant]}")

    fig = review_figure(tile, results, out / f"{args.patch}_texture_review.png")
    print(f"\n  review figure -> {fig.relative_to(ROOT)}")
    print("\n  RECOMMENDATION for FDM at this scale: V1 stippling. Isolated concave "
          "pits have no thin standing walls or single-layer floors to lose, so the "
          "shade gradient survives a 0.4 mm nozzle where V2's 0.2 mm steps and V3's "
          "0.3 mm grooves sit at/under the FDM floor. Print V2/V3 in resin, or FDM "
          "V1. All three watertight; texture is vertical (printable on slopes).")


if __name__ == "__main__":
    main()
