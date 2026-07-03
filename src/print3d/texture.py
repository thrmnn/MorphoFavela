"""Version A (realistic) sunlight-texture tiles for 3D print.

A 150 mm square tile of a CFD analysis patch: true terrain, buildings extruded at
real height with smooth roofs, and street-level *winter* sun-hours encoded as a
relief texture on the ground surface ONLY. Three texture treatments (stippling,
contour bands, directional hatching) are produced as separate watertight STL
variants from one shared base.

Design choices that matter for print fidelity:

* The whole tile is a single heightfield solid — terrain + building heights baked
  into one draped surface — so it is always a single watertight manifold with no
  boolean CSG. Building cells keep a flat (smooth) roof; only ground cells
  (no building) receive texture, exactly as the brief requires.
* Texture is applied as a *vertical* z-displacement, not along the true surface
  normal. On a slope a vertical dimple is the printable choice (no overhangs for
  FDM), and at these feature depths the difference from a normal-aligned cut is
  sub-0.1 mm. The steep-slope legibility the brief asks for is preserved; the
  deviation is documented per variant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import trimesh
from rasterio.enums import MergeAlg, Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.spatial import cKDTree

from src.print3d.model import NODATA_GUARD, ROOT, heightfield_solid

# Winter sun-hours class edges → shade level. Class 1 (>6 h, full sun) = smooth;
# shade rises to Class 4 (<2 h, deep shade) which carries the densest texture.
CLASS_EDGES = (2.0, 4.0, 6.0)
SHADE_SPACING_MM = {1: 4.0, 2: 2.5, 3: 1.5}  # shade 1=Class2 … 3=Class4
PIT_DIA_MM, PIT_DEPTH_MM = 1.0, 0.5
GROOVE_W_MM, GROOVE_D_MM, GROOVE_STEP_MM = 0.6, 0.4, 0.2
HATCH_W_MM, HATCH_D_MM = 0.5, 0.3


# Legend metadata per field: (tile-title tag, 4 class labels smooth→densest).
FIELD_LEGEND = {
    "sunlight": ("winter sun-hours", [
        "Class 1 · >6 h", "Class 2 · 4–6 h", "Class 3 · 2–4 h", "Class 4 · <2 h"]),
    "ventilation": ("pedestrian ventilation (|U|/U_ref quartiles)", [
        "Q1 · best-ventilated", "Q2", "Q3", "Q4 · most stagnant"]),
}


@dataclass
class Tile:
    site: str
    patch: str
    tile_mm: float
    mm_per_m: float
    world_cell: float
    X: np.ndarray
    Y: np.ndarray
    ground: np.ndarray       # terrain elevation, m
    bld_h: np.ndarray        # building height above ground, m (0 = open ground)
    shade: np.ndarray        # 0 smooth (best) … 3 densest (worst) on ground; -1 under buildings
    azimuth_deg: float       # groove direction for hatching (shadow / prevailing wind)
    field: str = "sunlight"

    @property
    def ground_mask(self) -> np.ndarray:
        return self.bld_h <= 0

    @property
    def base_surface(self) -> np.ndarray:
        return self.ground + self.bld_h


def _patch_dir(site: str, patch: str, sampling: str) -> Path:
    return ROOT / "outputs" / site / "sampling_cfd" / sampling / "patches" / patch


def sample_tile(
    site: str,
    patch: str,
    tile_mm: float = 150.0,
    model_cell_mm: float = 0.30,
    sampling: str = "campaign_sampling",
    field: str = "sunlight",
) -> Tile:
    pdir = _patch_dir(site, patch, sampling)
    meta = json.loads((pdir / "patch_meta.json").read_text())
    cx, cy = meta["center_x"], meta["center_y"]
    half = meta.get("analysis_patch_diameter", 100.0) / 2.0

    mm_per_m = tile_mm / (2 * half)
    world_cell = model_cell_mm / mm_per_m
    n = int(round(2 * half / world_cell))
    x0, y1 = cx - half, cy + half
    transform = from_origin(x0, y1, world_cell, world_cell)

    ground = np.empty((n, n), dtype="float32")
    with rasterio.open(pdir / "terrain.tif") as r:
        reproject(
            source=rasterio.band(r, 1), destination=ground,
            src_transform=r.transform, src_crs=r.crs,
            dst_transform=transform, dst_crs=r.crs,
            src_nodata=(r.nodata if r.nodata is not None else 3.4e38), dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    ground = ground.astype(float)
    ground[ground > NODATA_GUARD] = np.nan
    if np.isnan(ground).any():
        ground[np.isnan(ground)] = np.nanmin(ground)

    b = gpd.read_file(pdir / "buildings.gpkg")
    b = b[(b["altura"].notna()) & (b["altura"] > 0) & b.geometry.notna() & ~b.geometry.is_empty]
    b = b.sort_values("altura")
    bld_h = rasterize(
        ((g, float(a)) for g, a in zip(b.geometry, b["altura"])),
        out_shape=(n, n), transform=transform, fill=0.0,
        merge_alg=MergeAlg.replace, dtype="float32",
    ).astype(float)

    cols = np.arange(n) + 0.5
    xs = x0 + cols * world_cell
    ys = y1 - cols * world_cell
    X, Y = np.meshgrid(xs, ys)

    ground_mask = bld_h <= 0
    if field == "sunlight":
        # nearest street-solar winter sun-hours → fixed sun-hour classes
        sp = gpd.read_file(f"outputs/{site}/morphometrics/svf/svf_streets_solar.gpkg")
        sp_xy = np.column_stack([sp.geometry.x.values, sp.geometry.y.values])
        _, nn = cKDTree(sp_xy).query(np.column_stack([X.ravel(), Y.ravel()]))
        value = sp["solar_hours_winter"].values[nn].reshape(n, n)
        shade = (3 - np.digitize(value, CLASS_EDGES)).astype(int)  # 0 full-sun … 3 deep
        azimuth = float(meta.get("winter_shadow_azimuth_deg", 340.0))
    elif field == "ventilation":
        # wind-rose-weighted pedestrian |U|/U_ref → quartile classes (relative,
        # because the 8-direction mean is spatially flat in absolute terms)
        from src.print3d.airflow import dominant_wind_azimuth, ventilation_grid
        value = ventilation_grid(site, patch, X, Y)
        edges = np.quantile(value[ground_mask], [0.25, 0.5, 0.75])
        shade = (3 - np.digitize(value, edges)).astype(int)  # 0 best-vent … 3 stagnant
        azimuth = dominant_wind_azimuth(site)  # grooves along prevailing wind
    else:
        raise ValueError(f"unknown field {field!r}")
    shade[bld_h > 0] = -1
    return Tile(site, patch, tile_mm, mm_per_m, world_cell, X, Y, ground, bld_h, shade, azimuth, field)


# --------------------------------------------------------------------------- #
# Texture treatments — each returns a ground-only vertical displacement (m > 0).
# --------------------------------------------------------------------------- #

def stipple_displacement(tile: Tile, seed: int = 0) -> np.ndarray:
    """V1: hemispherical pits, jittered grid, denser with shade class."""
    n = tile.shade.shape[0]
    disp = np.zeros((n, n))
    r_w = (PIT_DIA_MM / 2) / tile.mm_per_m
    depth_w = PIT_DEPTH_MM / tile.mm_per_m
    rr = max(1, int(np.ceil(r_w / tile.world_cell)))
    rng = np.random.default_rng(seed)
    for s, spacing_mm in SHADE_SPACING_MM.items():
        step = spacing_mm / tile.mm_per_m / tile.world_cell  # spacing in cells
        gx = np.arange(rr, n - rr, step)
        for cyf in gx:
            for cxf in gx:
                iy = int(cyf + rng.uniform(-0.35, 0.35) * step)
                ix = int(cxf + rng.uniform(-0.35, 0.35) * step)
                if not (rr <= iy < n - rr and rr <= ix < n - rr):
                    continue
                if tile.shade[iy, ix] != s:  # pit belongs to this class's cells only
                    continue
                yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
                d = np.hypot(xx, yy) * tile.world_cell
                dz = np.where(d < r_w, depth_w * np.sqrt(np.clip(1 - (d / r_w) ** 2, 0, 1)), 0.0)
                win = disp[iy - rr:iy + rr + 1, ix - rr:ix + rr + 1]
                np.maximum(win, dz, out=win)
    disp[~tile.ground_mask] = 0.0
    return disp


def contour_displacement(tile: Tile) -> np.ndarray:
    """V2: engrave grooves on class boundaries + a subtle per-class step-down."""
    n = tile.shade.shape[0]
    s = np.where(tile.ground_mask, tile.shade, -1)
    # boundary cells: a ground cell adjacent (4-neigh) to a different ground class
    diff = np.zeros((n, n), bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = np.roll(np.roll(s, dy, 0), dx, 1)
        diff |= tile.ground_mask & (sh >= 0) & (sh != s)
    # dilate to groove width
    half_w = max(1, int(round((GROOVE_W_MM / tile.mm_per_m / tile.world_cell) / 2)))
    band = diff.copy()
    for _ in range(half_w):
        band |= np.roll(band, 1, 0) | np.roll(band, -1, 0) | np.roll(band, 1, 1) | np.roll(band, -1, 1)
    band &= tile.ground_mask
    disp = np.zeros((n, n))
    step = GROOVE_STEP_MM / tile.mm_per_m
    disp[tile.ground_mask] = step * tile.shade[tile.ground_mask]  # gentle per-class lowering
    disp[band] += GROOVE_D_MM / tile.mm_per_m
    return disp


def hatch_displacement(tile: Tile) -> np.ndarray:
    """V3: parallel grooves along the shadow azimuth, denser with shade class."""
    n = tile.shade.shape[0]
    th = np.deg2rad(tile.azimuth_deg)
    # signed distance across the groove direction, in mm on the model
    proj = (tile.X * np.cos(th) + tile.Y * np.sin(th))
    proj_mm = (proj - proj.min()) * tile.mm_per_m
    depth_w = HATCH_D_MM / tile.mm_per_m
    disp = np.zeros((n, n))
    for s, spacing_mm in SHADE_SPACING_MM.items():
        on = (proj_mm % spacing_mm) < HATCH_W_MM
        sel = on & tile.ground_mask & (tile.shade == s)
        disp[sel] = depth_w
    return disp


# --------------------------------------------------------------------------- #

VARIANTS = {
    "stipple": stipple_displacement,
    "contour": contour_displacement,
    "hatch": hatch_displacement,
}


@dataclass
class TileStats:
    site: str
    patch: str
    variant: str
    tile_mm: float
    scale_denom: int
    grid: tuple[int, int]
    model_mm: tuple[float, float, float]
    texture_depth_mm: float
    triangles: int
    watertight: bool


def build_tile(tile: Tile, variant: str, base_thickness: float = 4.0):
    disp = VARIANTS[variant](tile)
    surf = tile.base_surface - disp  # displacement is downward (into ground)
    floor_z = float(np.min(surf)) - base_thickness / tile.mm_per_m

    mesh = heightfield_solid(tile.X, tile.Y, surf, floor_z)
    mesh.apply_translation([-tile.X.min(), -tile.Y.min(), -floor_z])
    mesh.apply_scale(tile.mm_per_m)

    ext = mesh.extents
    stats = TileStats(
        site=tile.site, patch=tile.patch, variant=variant, tile_mm=tile.tile_mm,
        scale_denom=int(round(1000 / tile.mm_per_m)),
        grid=(tile.shade.shape[1], tile.shade.shape[0]),
        model_mm=(round(ext[0], 1), round(ext[1], 1), round(ext[2], 1)),
        texture_depth_mm=round(float(disp.max()) * tile.mm_per_m, 2),
        triangles=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
    )
    return mesh, stats, disp
