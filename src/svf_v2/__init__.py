"""SVF v2: Sky View Factor computation in world coordinates (EPSG:31983)."""

from src.svf_v2.compute import (
    compute_svf,
    compute_svf_raycasting,
    generate_sky_directions,
    generate_tregenza_patches,
)
from src.svf_v2.facades import compute_facade_svf, compute_facade_solar_potential
from src.svf_v2.sampling import (
    sample_facade_points,
    sample_grid_points,
    sample_street_points,
)
from src.svf_v2.scene import (
    build_scene,
    build_terrain_mesh,
    build_building_meshes,
    sample_dtm_at_points,
)
from src.svf_v2.paths import resolve_paths, resolve_boundary, AREA_FILES
from src.svf_v2.io import (
    save_grid_results,
    save_street_results,
    save_facade_results,
    save_scene_stl,
)

__all__ = [
    # Computation
    "compute_svf",
    "compute_svf_raycasting",
    "generate_sky_directions",
    "generate_tregenza_patches",
    # Facades
    "compute_facade_svf",
    "compute_facade_solar_potential",
    # Sampling
    "sample_facade_points",
    "sample_grid_points",
    "sample_street_points",
    # Scene
    "build_scene",
    "build_terrain_mesh",
    "build_building_meshes",
    "sample_dtm_at_points",
    # Paths
    "resolve_paths",
    "resolve_boundary",
    "AREA_FILES",
    # I/O
    "save_grid_results",
    "save_street_results",
    "save_facade_results",
    "save_scene_stl",
]
