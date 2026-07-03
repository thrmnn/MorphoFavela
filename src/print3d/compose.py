"""Compose a raw draped surface into a premium, printable artifact.

Takes the terrain + building heightfield and returns a single watertight
heightfield that looks like a museum / urban-planning model rather than a raw
extrusion: a flat framed border (the terrain sits in a shallow well), a chamfered
outer edge, a recessed water plane where the DTM drops below sea level, and an
engraved nameplate (site · scale · north arrow · scale bar) cut into the front
border. Everything is done as vertical displacement on one padded grid, so there
is no boolean CSG and the result is watertight by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


@dataclass
class Composed:
    X: np.ndarray
    Y: np.ndarray
    surf: np.ndarray
    water: np.ndarray        # bool, padded grid
    frame: np.ndarray        # bool, the flat border (padded)
    built: np.ndarray        # bool, building cells (padded)
    engrave: np.ndarray      # bool, engraved nameplate cells (padded)
    world_cell: float


def _text_mask(Xc, Yc, s, x_anchor, y_center, cap_w, *, max_width=None, ha="left", prop=None):
    """Boolean mask of grid cells covered by text `s`, cap-height `cap_w` (world
    units), vertically centred on `y_center`. If the rendered width would exceed
    `max_width`, the whole label is scaled down so it always fits (never clips).
    `ha`: 'left' anchors the left edge at x_anchor, 'center' centres on x_anchor,
    'right' anchors the right edge. Even-odd fill handles letter holes."""
    tp = TextPath((0, 0), s, size=1.0, prop=prop or FontProperties(weight="bold"))
    ext = tp.get_extents()
    if ext.height <= 0 or ext.width <= 0:
        return np.zeros(Xc.shape, bool)
    scale = cap_w / ext.height
    if max_width is not None and ext.width * scale > max_width:
        scale = max_width / ext.width
    w = ext.width * scale
    if ha == "center":
        dx = x_anchor - (ext.x0 * scale + w / 2)
    elif ha == "right":
        dx = x_anchor - (ext.x0 * scale + w)
    else:
        dx = x_anchor - ext.x0 * scale
    dy = y_center - (ext.y0 * scale + ext.height * scale / 2)
    verts = Affine2D().scale(scale).translate(dx, dy).transform(tp.vertices)
    path = MplPath(verts, tp.codes)
    return path.contains_points(np.column_stack([Xc.ravel(), Yc.ravel()])).reshape(Xc.shape)


def _north_arrow_mask(Xc, Yc, cx, cy, size):
    """A north-pointing chevron + 'N' above it, centred on (cx,cy); world units."""
    a = size
    tri = MplPath([(cx, cy + a), (cx - a * 0.5, cy - a * 0.55),
                   (cx, cy - a * 0.25), (cx + a * 0.5, cy - a * 0.55), (cx, cy + a)])
    pts = np.column_stack([Xc.ravel(), Yc.ravel()])
    m = tri.contains_points(pts).reshape(Xc.shape)
    m |= _text_mask(Xc, Yc, "N", cx, cy + a * 1.7, a * 0.8, ha="center")
    return m


def _scalebar_mask(Xc, Yc, x0, y0, seg_w, n_seg, bar_h):
    """Alternating filled ticks: n_seg segments of width seg_w, height bar_h."""
    m = np.zeros(Xc.shape, bool)
    for k in range(n_seg):
        if k % 2 == 0:
            xa, xb = x0 + k * seg_w, x0 + (k + 1) * seg_w
            m |= (Xc >= xa) & (Xc < xb) & (Yc >= y0) & (Yc < y0 + bar_h)
    # a thin baseline under the ticks
    m |= (Xc >= x0) & (Xc < x0 + n_seg * seg_w) & (Yc >= y0 - bar_h * 0.3) & (Yc < y0)
    return m


def compose(
    ground: np.ndarray,
    building_h: np.ndarray,
    *,
    x0: float, y0: float, world_cell: float,
    mm_per_m: float,
    label: str,
    scale_denom: int,
    sea_level_m: float = 0.5,
    border_mm: float = 6.0,
    engrave_mm: float = 0.6,
    chamfer_mm: float = 1.0,
    water_drop_mm: float = 1.2,
) -> Composed:
    ny, nx = ground.shape
    surf = ground + building_h
    built = building_h > 0

    water = ground <= sea_level_m
    land = surf[~water] if (~water).any() else surf
    frame_z = float(np.min(land))
    if water.any():
        surf = surf.copy()
        surf[water] = frame_z - water_drop_mm / mm_per_m

    # Asymmetric border: a thin mat on three sides, a wider strip along the bottom
    # to hold the nameplate (so the text never fights the terrain or clips).
    side = max(2, int(round((border_mm / mm_per_m) / world_cell)))
    botb = max(side + 2, int(round((2.4 * border_mm / mm_per_m) / world_cell)))
    surf = np.pad(surf, ((side, botb), (side, side)), constant_values=frame_z)
    water = np.pad(water, ((side, botb), (side, side)), constant_values=False)
    built = np.pad(built, ((side, botb), (side, side)), constant_values=False)
    frame = np.zeros(surf.shape, bool)
    frame[:side, :] = frame[-botb:, :] = frame[:, :side] = frame[:, -side:] = True

    # chamfer: bevel the outermost ring of the frame downward
    cham = max(1, int(round((chamfer_mm / mm_per_m) / world_cell)))
    drop = chamfer_mm / mm_per_m
    for r in range(cham):
        t = (cham - r) / cham * drop
        surf[r, :] -= t; surf[-1 - r, :] -= t
        surf[:, r] -= t; surf[:, -1 - r] -= t

    Ny, Nx = surf.shape
    gx0, gy1 = x0 - side * world_cell, y0 + (ny - 1 + side) * world_cell
    xs = gx0 + (np.arange(Nx) + 0.5) * world_cell
    ys = gy1 - (np.arange(Ny) + 0.5) * world_cell
    Xc, Yc = np.meshgrid(xs, ys)

    # --- nameplate laid out inside the bottom strip, with margins so nothing clips ---
    eng_m = engrave_mm / mm_per_m
    x_lo, x_hi = xs[side], xs[-side - 1]          # inner (terrain-width) span
    frame_w, y_bot = x_hi - x_lo, ys[-1]          # bottom edge of the plate
    strip_h = botb * world_cell
    inset = strip_h * 0.16                         # keep clear of the chamfered edge

    # upper band: the name, centred, occupying the top ~40% of the strip
    name = _text_mask(Xc, Yc, label.upper(), (x_lo + x_hi) / 2,
                      y_bot + strip_h * 0.72, strip_h * 0.30,
                      max_width=frame_w - 2 * inset, ha="center")
    # lower band: 1:scale (left) and a scale bar with its metre label (right)
    info_cap = strip_h * 0.19
    scale_txt = _text_mask(Xc, Yc, f"1:{scale_denom:,}", x_lo + inset,
                           y_bot + strip_h * 0.26, info_cap,
                           max_width=frame_w * 0.36, ha="left")
    seg_w, avail = 50.0, frame_w * 0.40
    n_seg = max(2, min(5, int(avail / seg_w)))
    bar_x = x_hi - inset - n_seg * seg_w
    scalebar = _scalebar_mask(Xc, Yc, bar_x, y_bot + strip_h * 0.17, seg_w, n_seg, info_cap * 0.7)
    scale_lbl = _text_mask(Xc, Yc, f"{int(n_seg * seg_w)} m", x_hi - inset,
                           y_bot + strip_h * 0.40, info_cap * 0.8,
                           max_width=avail, ha="right")
    # north arrow lives in the tall right-hand border near the top (fits its width)
    north = _north_arrow_mask(Xc, Yc, (xs[-side] + xs[-1]) / 2, ys[max(3, 2 * side)],
                              side * world_cell * 0.34)
    engrave = (name | scale_txt | scalebar | scale_lbl | north) & frame
    surf[engrave] -= eng_m

    return Composed(Xc, Yc, surf, water, frame, built, engrave, world_cell)
