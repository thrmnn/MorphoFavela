"""Integration tests for point sampling with real data."""

import numpy as np
import pytest


@pytest.mark.integration
class TestGridSampling:
    def test_grid_points_count(self, area_name, area_paths, area_scene):
        from src.svf_v2.sampling import sample_grid_points

        dtm_path = area_paths[0]
        _, _, gdf = area_scene
        pts = sample_grid_points(
            dtm_path, gdf, grid_spacing=20.0, pedestrian_height=1.5
        )
        assert len(pts) > 50, f"Too few grid points: {len(pts)}"
        assert pts.shape[1] == 3

    def test_grid_z_finite(self, area_paths, area_scene):
        from src.svf_v2.sampling import sample_grid_points

        dtm_path = area_paths[0]
        _, _, gdf = area_scene
        pts = sample_grid_points(dtm_path, gdf, grid_spacing=20.0)
        assert np.all(np.isfinite(pts[:, 2])), "Grid points have NaN Z values"

    def test_grid_z_within_dtm_range(self, area_paths, area_scene):
        import rasterio

        from src.svf_v2.sampling import sample_grid_points

        dtm_path = area_paths[0]
        _, _, gdf = area_scene
        pts = sample_grid_points(dtm_path, gdf, grid_spacing=20.0)

        with rasterio.open(dtm_path) as src:
            dtm = src.read(1)
            valid = dtm[np.isfinite(dtm) & (dtm < 10000)]

        # Grid Z (minus pedestrian height) should be within DTM range
        grid_z = pts[:, 2] - 1.5  # remove pedestrian height
        assert grid_z.min() >= valid.min() - 5, "Grid Z below DTM min"
        assert grid_z.max() <= valid.max() + 5, "Grid Z above DTM max"

    def test_grid_excludes_buildings(self, area_paths, area_scene):
        """Verify a random sample of grid points are outside building footprints."""
        import geopandas as gpd
        from shapely.geometry import Point

        from src.svf_v2.sampling import sample_grid_points

        dtm_path = area_paths[0]
        _, _, gdf = area_scene
        pts = sample_grid_points(dtm_path, gdf, grid_spacing=20.0)

        # Check 100 random points
        n_check = min(100, len(pts))
        indices = np.random.default_rng(42).choice(len(pts), n_check, replace=False)
        check_pts = gpd.GeoDataFrame(
            geometry=[Point(pts[i, 0], pts[i, 1]) for i in indices],
            crs=gdf.crs,
        )
        joined = gpd.sjoin(check_pts, gdf[["geometry"]], how="left", predicate="within")
        n_inside = joined["index_right"].notna().sum()
        assert n_inside == 0, f"{n_inside}/{n_check} grid points are inside buildings"


@pytest.mark.integration
class TestStreetSampling:
    def test_street_points_count(self, area_paths):
        from src.svf_v2.sampling import sample_street_points

        _, _, roads_path = area_paths
        dtm_path = area_paths[0]
        gdf = sample_street_points(roads_path, dtm_path, spacing=5.0)
        assert len(gdf) > 0, "No street sample points"
        assert "street_id" in gdf.columns
        assert "z" in gdf.columns
        assert "z_observer" in gdf.columns

    def test_street_z_finite(self, area_paths):
        from src.svf_v2.sampling import sample_street_points

        _, _, roads_path = area_paths
        dtm_path = area_paths[0]
        gdf = sample_street_points(roads_path, dtm_path, spacing=10.0)
        assert np.all(np.isfinite(gdf["z"].values)), "Street points have NaN Z"


@pytest.mark.integration
class TestFacadeSampling:
    def test_facade_points_count(self, area_paths, area_scene):
        from src.svf_v2.sampling import sample_facade_points

        dtm_path = area_paths[0]
        _, _, gdf = area_scene
        facade_gdf = sample_facade_points(
            gdf,
            dtm_path,
            vertical_spacing=5.0,
            horizontal_spacing=5.0,
        )
        assert len(facade_gdf) > 0, "No facade sample points"
        assert "normal_x" in facade_gdf.columns
        assert "building_id" in facade_gdf.columns

    def test_facade_normals_unit_length(self, area_paths, area_scene):
        from src.svf_v2.sampling import sample_facade_points

        dtm_path = area_paths[0]
        _, _, gdf = area_scene
        facade_gdf = sample_facade_points(
            gdf,
            dtm_path,
            vertical_spacing=5.0,
            horizontal_spacing=5.0,
        )
        nx = facade_gdf["normal_x"].values
        ny = facade_gdf["normal_y"].values
        norms = np.sqrt(nx**2 + ny**2)
        np.testing.assert_allclose(
            norms, 1.0, atol=0.01, err_msg="Facade normals are not unit vectors"
        )
