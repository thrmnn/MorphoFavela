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


def _text_mask(Xc, Yc, s, x0, y0, height_w, prop=None, ha="left"):
    """Boolean mask of grid cells covered by text `s`, baseline at (x0,y0),
    cap-height `height_w` (world units). Even-odd fill handles letter holes."""
    tp = TextPath((0, 0), s, size=1.0, prop=prop or FontProperties(weight="bold"))
    ext = tp.get_extents()
    if ext.height <= 0:
        return np.zeros(Xc.shape, bool)
    scale = height_w / ext.height
    dx = -ext.x0 * scale + x0
    if ha == "center":
        dx = x0 - (ext.x0 + ext.width / 2) * scale
    tr = Affine2D().scale(scale).translate(dx, -ext.y0 * scale + y0)
    verts = tr.transform(tp.vertices)
    path = MplPath(verts, tp.codes)
    pts = np.column_stack([Xc.ravel(), Yc.ravel()])
    return path.contains_points(pts).reshape(Xc.shape)


def _north_arrow_mask(Xc, Yc, cx, cy, size):
    """A north-pointing chevron outline + 'N' near (cx,cy); size in world units."""
    a = size
    tri = MplPath([(cx, cy + a), (cx - a * 0.5, cy - a * 0.55),
                   (cx, cy - a * 0.25), (cx + a * 0.5, cy - a * 0.55), (cx, cy + a)])
    pts = np.column_stack([Xc.ravel(), Yc.ravel()])
    m = tri.contains_points(pts).reshape(Xc.shape)
    m |= _text_mask(Xc, Yc, "N", cx - a * 0.32, cy + a * 1.25, a * 0.9)
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

    # pad a flat border (the frame/mat) around the terrain
    bcells = max(2, int(round((border_mm / mm_per_m) / world_cell)))
    P = ((bcells, bcells), (bcells, bcells))
    surf = np.pad(surf, P, constant_values=frame_z)
    water = np.pad(water, P, constant_values=False)
    built = np.pad(built, P, constant_values=False)
    frame = np.zeros(surf.shape, bool)
    frame[:bcells, :] = frame[-bcells:, :] = frame[:, :bcells] = frame[:, -bcells:] = True

    # chamfer: bevel the outermost ring of the frame downward
    cham = max(1, int(round((chamfer_mm / mm_per_m) / world_cell)))
    drop = chamfer_mm / mm_per_m
    for r in range(cham):
        t = (cham - r) / cham * drop
        surf[r, :] -= t; surf[-1 - r, :] -= t
        surf[:, r] -= t; surf[:, -1 - r] -= t

    Ny, Nx = surf.shape
    gx0, gy1 = x0 - bcells * world_cell, y0 + (ny - 1 + bcells) * world_cell
    cols = np.arange(Nx); rows = np.arange(Ny)
    xs = gx0 + (cols + 0.5) * world_cell
    ys = gy1 - (rows + 0.5) * world_cell
    Xc, Yc = np.meshgrid(xs, ys)

    # engrave the front (south) border: NAME · 1:scale · scale bar; north arrow NE
    border_w = bcells * world_cell
    south_y = ys[-bcells:].mean()
    cap = border_w * 0.42
    name = _text_mask(Xc, Yc, label.upper(), gx0 + border_w * 1.2,
                      south_y + cap * 0.7, cap)
    scaletxt = _text_mask(Xc, Yc, f"1:{scale_denom:,}", gx0 + border_w * 1.2,
                          south_y - cap * 0.9, cap * 0.55)
    # scale bar: 5 segments of 50 m (world) near bottom-right of the south border
    seg_w = 50.0
    n_seg = max(2, min(5, int((xs[-bcells] - (gx0 + border_w)) / seg_w)))
    bar_x = xs[-bcells - int(n_seg * seg_w / world_cell)] if n_seg else gx0
    scalebar = _scalebar_mask(Xc, Yc, bar_x, south_y - cap * 0.5, seg_w, n_seg, cap * 0.5)
    scalelbl = _text_mask(Xc, Yc, f"{int(n_seg*seg_w)} m", bar_x, south_y + cap * 0.4, cap * 0.5)
    north = _north_arrow_mask(Xc, Yc, xs[-bcells // 2 - 1], ys[bcells + 2], border_w * 0.45)
    engrave = (name | scaletxt | scalebar | scalelbl | north) & frame
    surf[engrave] -= engrave_mm / mm_per_m

    return Composed(Xc, Yc, surf, water, frame, built, engrave, world_cell)
