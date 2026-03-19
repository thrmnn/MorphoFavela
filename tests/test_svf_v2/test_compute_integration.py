"""Integration tests for SVF computation on real data (small subsets)."""

import numpy as np
import pytest


@pytest.mark.integration
class TestSVFGrid:
    def test_svf_grid_10pts(self, area_paths, area_scene):
        from src.svf_v2.sampling import sample_grid_points
        from src.svf_v2.compute import compute_svf

        dtm_path = area_paths[0]
        scene_mesh, _, gdf = area_scene

        pts = sample_grid_points(dtm_path, gdf, grid_spacing=20.0)
        subset = pts[:10]

        svf = compute_svf(subset, scene_mesh, backend="raycasting", n_sky_patches=20)
        assert len(svf) == len(subset)
        assert np.all(np.isfinite(svf)), "SVF has NaN values"
        assert np.all((svf >= 0) & (svf <= 1)), "SVF out of [0,1] range"
        assert svf.mean() > 0, "Mean SVF should be > 0"


@pytest.mark.integration
class TestSVFStreets:
    def test_svf_street_10pts(self, area_paths, area_scene):
        from src.svf_v2.sampling import sample_street_points
        from src.svf_v2.compute import compute_svf

        dtm_path, _, roads_path = area_paths
        scene_mesh, _, _ = area_scene

        street_gdf = sample_street_points(roads_path, dtm_path, spacing=10.0)
        if len(street_gdf) == 0:
            pytest.skip("No street points for this area")

        subset = street_gdf.head(10)
        observers = np.column_stack(
            [
                np.array([p.x for p in subset.geometry]),
                np.array([p.y for p in subset.geometry]),
                subset["z_observer"].values,
            ]
        )

        svf = compute_svf(observers, scene_mesh, backend="raycasting", n_sky_patches=20)
        assert len(svf) == len(subset)
        assert np.all(np.isfinite(svf))
        assert np.all((svf >= 0) & (svf <= 1))


@pytest.mark.integration
class TestSVFFacades:
    def test_svf_facade_10pts(self, area_paths, area_scene):
        from src.svf_v2.sampling import sample_facade_points
        from src.svf_v2.facades import compute_facade_svf

        dtm_path = area_paths[0]
        scene_mesh, _, gdf = area_scene

        facade_gdf = sample_facade_points(
            gdf,
            dtm_path,
            vertical_spacing=5.0,
            horizontal_spacing=5.0,
        )
        if len(facade_gdf) == 0:
            pytest.skip("No facade points for this area")

        subset = facade_gdf.head(10).copy()
        result = compute_facade_svf(subset, scene_mesh, n_sky_patches=20)

        assert "svf" in result.columns
        svf = result["svf"].values
        assert np.all(np.isfinite(svf))
        assert np.all((svf >= 0) & (svf <= 1))
