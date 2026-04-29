"""Integration tests for scene construction with real data."""

import numpy as np
import pytest
import pyvista as pv

# Expected building counts after filtering (approximate)
EXPECTED_BUILDINGS = {
    "vidigal_tls": (300, 500),
    "vidigal": (2000, 4000),
    "riodaspedras": (7000, 11000),
}


@pytest.mark.integration
class TestBuildTerrainMesh:
    def test_terrain_has_points(self, area_paths):
        from src.svf_v2.scene import build_terrain_mesh

        dtm_path = area_paths[0]
        mesh = build_terrain_mesh(dtm_path, subsample=4)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_terrain_z_range(self, area_paths):
        from src.svf_v2.scene import build_terrain_mesh

        dtm_path = area_paths[0]
        mesh = build_terrain_mesh(dtm_path, subsample=4)
        zs = mesh.points[:, 2]
        assert np.all(np.isfinite(zs)), "Terrain mesh has NaN vertices"
        assert zs.min() >= -10, f"Suspiciously low Z: {zs.min()}"
        assert zs.max() <= 600, f"Suspiciously high Z: {zs.max()}"


@pytest.mark.integration
class TestBuildBuildingMeshes:
    def test_building_count(self, area_name, area_paths):
        from src.svf_v2.scene import build_building_meshes

        dtm_path, fp_path, _ = area_paths
        mesh, gdf = build_building_meshes(fp_path, dtm_path, area=area_name)
        lo, hi = EXPECTED_BUILDINGS.get(area_name, (1, 100000))
        assert lo <= len(gdf) <= hi, f"{area_name}: {len(gdf)} buildings (expected {lo}-{hi})"
        assert mesh is not None, "Building mesh should not be None"
        assert mesh.n_points > 0

    def test_building_z_plausible(self, area_paths):
        from src.svf_v2.scene import build_building_meshes

        dtm_path, fp_path, _ = area_paths
        mesh, _ = build_building_meshes(fp_path, dtm_path)
        zs = mesh.points[:, 2]
        assert zs.min() >= -10, f"Building Z too low: {zs.min()}"
        assert zs.max() <= 500, f"Building Z too high: {zs.max()}"


@pytest.mark.integration
class TestBuildScene:
    def test_combined_larger_than_terrain(self, area_scene, area_paths):
        scene_mesh, terrain, gdf = area_scene
        assert scene_mesh.n_points > terrain.n_points, (
            "Combined scene should have more points than terrain alone"
        )

    def test_scene_bbox_contains_footprints(self, area_scene):
        scene_mesh, _, gdf = area_scene
        fp_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        mesh_bounds = scene_mesh.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
        # Mesh x range should encompass footprint x range (with 100m tolerance)
        assert mesh_bounds[0] <= fp_bounds[0] + 100
        assert mesh_bounds[1] >= fp_bounds[2] - 100
        # Mesh y range
        assert mesh_bounds[2] <= fp_bounds[1] + 100
        assert mesh_bounds[3] >= fp_bounds[3] - 100


@pytest.mark.integration
class TestSTLExport:
    def test_stl_roundtrip(self, area_scene, tmp_path):
        scene_mesh, _, _ = area_scene
        stl_path = tmp_path / "scene.stl"
        scene_mesh.save(str(stl_path), binary=True)
        assert stl_path.exists()
        assert stl_path.stat().st_size > 0
        # Roundtrip: re-read and verify
        reloaded = pv.read(str(stl_path))
        assert reloaded.n_cells > 0
