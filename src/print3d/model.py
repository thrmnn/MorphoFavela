"""Build a watertight, scaled 3D-print STL from a CFD analysis patch.

The patch is a 100 m core (``analysis_patch_diameter``) cut from the larger CFD
domain. We sample the terrain on its native 5 m grid over the core's bounding
square, lift it into a closed solid plinth, extrude each building footprint into
a solid prism embedded a few metres into that plinth (so nothing floats or
detaches in the print), then move to a local origin and scale to model mm.

Resin note: this emits a *solid* watertight model; hollowing + drain holes are
best done in the slicer (Lychee/Chitubox), which place them more reliably than a
mesh-level offset would.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import trimesh
from rasterio.windows import from_bounds
from shapely.geometry import box
from trimesh.creation import extrude_polygon

ROOT = Path(__file__).resolve().parents[2]
NODATA_GUARD = 1e30  # terrain.tif uses a 3.4e38 float sentinel for no-data


@dataclass
class PrintStats:
    site: str
    patch_id: str
    scale_denom: int
    n_buildings: int
    model_mm: tuple[float, float, float]
    relief_mm: float
    triangles: int
    watertight: bool
    single_manifold: bool
    tallest_building_mm: float


def _patch_dir(site: str, patch_id: str, sampling: str) -> Path:
    return ROOT / "outputs" / site / "sampling_cfd" / sampling / "patches" / patch_id


def _sample_terrain_core(terrain_path: Path, cx: float, cy: float, half: float):
    """Return (X, Y, Z) grids of valid terrain over the [c±half] square."""
    with rasterio.open(terrain_path) as r:
        win = from_bounds(cx - half, cy - half, cx + half, cy + half, r.transform)
        z = r.read(1, window=win).astype(float)
        wt = r.window_transform(win)
    z[z > NODATA_GUARD] = np.nan
    if np.isnan(z).any():  # fill rare gaps with the local minimum so the plinth stays closed
        z[np.isnan(z)] = np.nanmin(z)
    ny, nx = z.shape
    cols = np.arange(nx) + 0.5
    rows = np.arange(ny) + 0.5
    xs = wt.c + cols * wt.a
    ys = wt.f + rows * wt.e
    X, Y = np.meshgrid(xs, ys)
    return X, Y, z


def heightmap_to_solid(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, floor_z: float) -> trimesh.Trimesh:
    """Close a draped heightmap into a watertight solid plinth down to ``floor_z``."""
    ny, nx = Z.shape
    top = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    bot = top.copy()
    bot[:, 2] = floor_z
    verts = np.vstack([top, bot])
    n = ny * nx

    def idx(i, j):
        return i * nx + j

    faces = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            faces += [[a, b, c], [a, c, d]]                  # top (up)
            faces += [[n + a, n + c, n + b], [n + a, n + d, n + c]]  # bottom (down)
    # side walls along the four borders
    for j in range(nx - 1):
        for i in (0, ny - 1):
            a, b = idx(i, j), idx(i, j + 1)
            faces += [[a, b, n + b], [a, n + b, n + a]]
    for i in range(ny - 1):
        for j in (0, nx - 1):
            a, b = idx(i, j), idx(i + 1, j)
            faces += [[a, n + b, b], [a, n + a, n + b]]

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def _building_prisms(buildings: gpd.GeoDataFrame, embed: float) -> list[trimesh.Trimesh]:
    parts = []
    for geom, base, alt in zip(buildings.geometry, buildings["base"], buildings["altura"]):
        if geom is None or geom.is_empty or alt is None or alt <= 0:
            continue
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            if poly.area < 1.0:
                continue
            prism = extrude_polygon(poly, float(alt) + embed, engine="triangle")
            prism.apply_translation([0, 0, float(base) - embed])
            parts.append(prism)
    return parts


def _fuse(parts: list[trimesh.Trimesh], solid: bool) -> tuple[trimesh.Trimesh, bool]:
    """Boolean-union the parts into one watertight manifold when manifold3d is
    available; otherwise concatenate the (individually closed) solids."""
    if solid:
        try:
            merged = trimesh.boolean.union(parts, engine="manifold")
            return merged, True
        except Exception:  # noqa: BLE001 — fall back to a sliceable soup of closed solids
            pass
    return trimesh.util.concatenate(parts), False


def build_print_model(
    site: str,
    patch_id: str,
    scale_denom: int = 1000,
    embed: float = 2.0,
    base_thickness: float = 3.0,
    solid: bool = True,
    sampling: str = "campaign_sampling",
) -> tuple[trimesh.Trimesh, PrintStats]:
    """Assemble a watertight, model-mm print mesh for one analysis patch.

    scale_denom: 1000 → a 100 m patch becomes a 100 mm (10 cm) model; 500 → 20 cm.
    embed: metres each building is sunk into the terrain so it fuses (no floaters).
    base_thickness: model-mm thickness of the solid plinth below the lowest ground.
    solid: boolean-union terrain+buildings into one watertight manifold (manifold3d);
        if unavailable, fall back to a sliceable union of individually closed solids.
    """
    pdir = _patch_dir(site, patch_id, sampling)
    meta = json.loads((pdir / "patch_meta.json").read_text())
    cx, cy = meta["center_x"], meta["center_y"]
    half = meta.get("analysis_patch_diameter", 100.0) / 2.0

    X, Y, Z = _sample_terrain_core(pdir / "terrain.tif", cx, cy, half)
    z_min = float(np.min(Z))

    buildings = gpd.read_file(pdir / "buildings.gpkg")
    core = box(cx - half, cy - half, cx + half, cy + half)
    buildings = buildings.clip(core)
    buildings = buildings[~buildings.geometry.is_empty & buildings.geometry.notna()]

    mm_per_m = 1000.0 / scale_denom
    floor_z = z_min - base_thickness / mm_per_m  # plinth thickness is defined in model mm

    terrain = heightmap_to_solid(X, Y, Z, floor_z)
    parts = [terrain, *_building_prisms(buildings, embed)]
    mesh, fused = _fuse(parts, solid)

    mesh.apply_translation([-cx, -cy, -floor_z])
    mesh.apply_scale(mm_per_m)

    ext = mesh.extents
    tallest_mm = float(buildings["altura"].max()) * mm_per_m if len(buildings) else 0.0
    stats = PrintStats(
        site=site,
        patch_id=patch_id,
        scale_denom=scale_denom,
        n_buildings=int(len(buildings)),
        model_mm=(round(ext[0], 1), round(ext[1], 1), round(ext[2], 1)),
        relief_mm=round((float(np.ptp(Z))) * mm_per_m, 1),
        triangles=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        single_manifold=bool(fused),
        tallest_building_mm=round(tallest_mm, 1),
    )
    return mesh, stats


def save_model(mesh: trimesh.Trimesh, stats: PrintStats, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stats.patch_id}_1to{stats.scale_denom}"
    path = out_dir / f"{stem}.stl"
    mesh.export(path)
    (out_dir / f"{stem}.json").write_text(json.dumps(asdict(stats), indent=2))
    return path
