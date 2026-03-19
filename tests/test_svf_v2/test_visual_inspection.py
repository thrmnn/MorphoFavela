"""Integration tests for visual inspection outputs (per area)."""

import pyvista as pv
import pytest


@pytest.mark.integration
class TestSTLExportPerArea:
    def test_stl_export(self, area_name, area_scene, tmp_path):
        from src.svf_v2.io import save_scene_stl

        scene_mesh, _, _ = area_scene
        stl_path = save_scene_stl(scene_mesh, tmp_path / f"{area_name}.stl")
        assert stl_path.exists()
        assert stl_path.stat().st_size > 1000, "STL file suspiciously small"
        reloaded = pv.read(str(stl_path))
        assert reloaded.n_cells > 0

    def test_vtk_export(self, area_name, area_scene, tmp_path):
        from src.svf_v2.io import save_scene_vtk

        scene_mesh, _, _ = area_scene
        vtk_path = save_scene_vtk(scene_mesh, tmp_path / f"{area_name}.vtk")
        assert vtk_path.exists()
        reloaded = pv.read(str(vtk_path))
        assert reloaded.n_points == scene_mesh.n_points


@pytest.mark.integration
class TestAlignmentPlotPerArea:
    def test_alignment_plot(self, area_name, area_paths, area_scene, tmp_path):
        from src.svf_v2.io import plot_alignment_check
        import geopandas as gpd
        import rasterio

        _, _, gdf = area_scene
        _, _, roads_path = area_paths
        dtm_path = area_paths[0]

        roads_gdf = gpd.read_file(roads_path)
        with rasterio.open(dtm_path) as src:
            if roads_gdf.crs != src.crs:
                roads_gdf = roads_gdf.to_crs(src.crs)

        png_path = plot_alignment_check(
            gdf,
            tmp_path / f"{area_name}_align.png",
            roads_gdf=roads_gdf,
            title=f"{area_name}",
        )
        assert png_path.exists()
        assert png_path.stat().st_size > 1000


@pytest.mark.integration
class TestSceneBBoxVsDTM:
    def test_bbox_matches_dtm(self, area_name, area_paths, area_scene):
        import rasterio

        scene_mesh, _, _ = area_scene
        dtm_path = area_paths[0]

        with rasterio.open(dtm_path) as src:
            dtm_bounds = src.bounds  # left, bottom, right, top

        mesh_bounds = scene_mesh.bounds  # xmin, xmax, ymin, ymax, zmin, zmax
        tol = 200  # metres tolerance (subsampled terrain may shrink slightly)

        assert mesh_bounds[0] >= dtm_bounds.left - tol, (
            f"Mesh xmin {mesh_bounds[0]} far from DTM left {dtm_bounds.left}"
        )
        assert mesh_bounds[1] <= dtm_bounds.right + tol
        assert mesh_bounds[2] >= dtm_bounds.bottom - tol
        assert mesh_bounds[3] <= dtm_bounds.top + tol
