"""SVF v2: Sky View Factor computation in world coordinates (EPSG:31983)."""

from src.svf_v2.compute import (
    compute_svf,
    compute_svf_raycasting,
    generate_sky_directions,
    generate_tregenza_patches,
)
from src.svf_v2.facades import compute_facade_solar_potential, compute_facade_svf
from src.svf_v2.io import (
    save_facade_results,
    save_grid_results,
    save_scene_stl,
    save_street_results,
)
from src.svf_v2.paths import AREA_FILES, resolve_boundary, resolve_paths
from src.svf_v2.sampling import (
    sample_facade_points,
    sample_grid_points,
    sample_street_points,
)
from src.svf_v2.scene import (
    build_building_meshes,
    build_scene,
    build_terrain_mesh,
    sample_dtm_at_points,
)
from src.svf_v2.utils import (
    compute_ground_mask,
    extract_terrain_surface,
    filter_buildings,
    filter_isolated_buildings,
    generate_ground_points,
    load_building_footprints,
    load_mesh,
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
    # Utilities (migrated from svf_utils)
    "load_mesh",
    "extract_terrain_surface",
    "filter_buildings",
    "filter_isolated_buildings",
    "load_building_footprints",
    "compute_ground_mask",
    "generate_ground_points",
]
