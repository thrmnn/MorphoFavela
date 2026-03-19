"""Tests for I/O functions: file creation and roundtrips."""

import numpy as np
import pyvista as pv
import geopandas as gpd
from shapely.geometry import Point, box


class TestSaveGridResults:
    def test_creates_all_files(self, tmp_path):
        from src.svf_v2.io import save_grid_results

        # Need a grid large enough for rasterisation (>= 2 cols and 2 rows)
        xs = np.arange(0, 20, 2.0)
        ys = np.arange(0, 20, 2.0)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.full_like(xx, 10.0)
        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        rng = np.random.default_rng(42)
        svf = rng.uniform(0.3, 0.9, len(pts))

        save_grid_results(pts, svf, "EPSG:31983", tmp_path, grid_spacing=2.0)

        assert (tmp_path / "svf_grid.gpkg").exists()
        assert (tmp_path / "svf_grid.csv").exists()
        assert (tmp_path / "svf_grid.tif").exists()
        assert (tmp_path / "svf_heatmap.png").exists()

        # Verify GPKG contents
        gdf = gpd.read_file(tmp_path / "svf_grid.gpkg")
        assert len(gdf) == len(pts)
        assert "svf" in gdf.columns


class TestSaveStreetResults:
    def test_creates_files(self, tmp_path):
        from src.svf_v2.io import save_street_results

        gdf = gpd.GeoDataFrame(
            {
                "svf": [0.5, 0.6, 0.7],
                "street_id": [0, 0, 1],
                "z": [10.0, 10.1, 10.2],
            },
            geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
            crs="EPSG:31983",
        )
        save_street_results(gdf, tmp_path)
        assert (tmp_path / "svf_streets.gpkg").exists()
        assert (tmp_path / "svf_streets_map.png").exists()


class TestSaveSceneSTL:
    def test_stl_roundtrip(self, tmp_path):
        from src.svf_v2.io import save_scene_stl

        # Create a simple mesh
        x = np.linspace(0, 5, 6)
        y = np.linspace(0, 5, 6)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        mesh = pv.StructuredGrid(xx, yy, zz).extract_surface()

        stl_path = save_scene_stl(mesh, tmp_path / "test.stl")
        assert stl_path.exists()
        assert stl_path.stat().st_size > 0

        reloaded = pv.read(str(stl_path))
        assert reloaded.n_cells > 0

    def test_vtk_roundtrip(self, tmp_path):
        from src.svf_v2.io import save_scene_vtk

        mesh = pv.Plane(i_size=10, j_size=10)
        vtk_path = save_scene_vtk(mesh, tmp_path / "test.vtk")
        assert vtk_path.exists()
        reloaded = pv.read(str(vtk_path))
        assert reloaded.n_points == mesh.n_points


class TestAlignmentPlot:
    def test_creates_png(self, tmp_path):
        from src.svf_v2.io import plot_alignment_check

        fp_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 5, 5), box(10, 10, 15, 15)],
            crs="EPSG:31983",
        )
        png_path = plot_alignment_check(fp_gdf, tmp_path / "align.png")
        assert png_path.exists()
        assert png_path.stat().st_size > 0

    def test_with_roads_and_points(self, tmp_path):
        from src.svf_v2.io import plot_alignment_check
        from shapely.geometry import LineString

        fp_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 5, 5)], crs="EPSG:31983",
        )
        roads_gdf = gpd.GeoDataFrame(
            geometry=[LineString([(0, 2.5), (10, 2.5)])], crs="EPSG:31983",
        )
        pts = np.array([[1, 2.5, 0], [3, 2.5, 0], [7, 2.5, 0]])

        png_path = plot_alignment_check(
            fp_gdf, tmp_path / "align2.png",
            roads_gdf=roads_gdf, sample_points=pts,
        )
        assert png_path.exists()


class TestSaveFacadeResults:
    def test_creates_files(self, tmp_path):
        from src.svf_v2.io import save_facade_results

        gdf = gpd.GeoDataFrame(
            {
                "svf": [0.4, 0.5],
                "facade_azimuth": [90.0, 270.0],
                "height_above_ground": [2.0, 4.0],
                "building_id": [0, 0],
            },
            geometry=[Point(0, 0), Point(1, 0)],
            crs="EPSG:31983",
        )
        fp_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 2, 2)], crs="EPSG:31983",
        )
        save_facade_results(gdf, tmp_path, footprints_gdf=fp_gdf)
        assert (tmp_path / "svf_facades.gpkg").exists()
        assert (tmp_path / "svf_facades_summary.gpkg").exists()
        assert (tmp_path / "svf_facades_orientation.png").exists()
        assert (tmp_path / "svf_facades_height.png").exists()
