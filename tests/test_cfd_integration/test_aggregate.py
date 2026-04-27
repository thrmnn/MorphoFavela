"""Tests for spatial aggregation: CFD samples → 100m-diameter circular patch → 10m grid cells."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from src.cfd_integration.aggregate import (
    aggregate_to_domain,
    aggregate_to_grid,
    aggregate_to_patch,
)
from src.cfd_integration.schema import CFDPatchResult, PatchSimulationMetadata


@pytest.fixture
def patch_result(synthetic_samples, synthetic_metadata):
    meta = PatchSimulationMetadata(**synthetic_metadata)
    return CFDPatchResult(metadata=meta, samples=synthetic_samples)


def _make_grid(center_x=0.0, center_y=0.0, extent=100.0):
    """Create a 10m grid covering the 100m analysis patch."""
    rows = []
    half = extent / 2
    zone_id = 0
    for cy in np.arange(center_y - half + 5, center_y + half + 5, 10):
        for cx in np.arange(center_x - half + 5, center_x + half + 5, 10):
            rows.append(
                {
                    "zone_id": zone_id,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "geometry": box(cx - 5, cy - 5, cx + 5, cy + 5),
                    "H_mean": 6.5,  # typical favela
                }
            )
            zone_id += 1
    return gpd.GeoDataFrame(rows, crs="EPSG:31983")


class TestAggregateToPatch:
    def test_crops_to_analysis_patch(self, patch_result):
        """Only samples within the central 100m-diameter circle should be used."""
        result = aggregate_to_patch(
            patch_result, patch_center_xy=(0, 0), analysis_patch_diameter=100
        )
        assert result["n_samples"] > 0
        # Our synthetic domain is 250m radius, 100m patch should have a
        # small fraction of the total samples
        total_samples = len(patch_result.samples)
        assert result["n_samples"] < total_samples

    def test_required_keys(self, patch_result):
        result = aggregate_to_patch(patch_result, (0, 0))
        required = {
            "U_mean",
            "U_median",
            "U_p10",
            "U_p90",
            "stagnation_frac",
            "TKE_mean",
            "vent_efficiency",
        }
        assert required.issubset(result.keys())

    def test_ach_when_canopy_height_given(self, patch_result):
        result = aggregate_to_patch(patch_result, (0, 0), canopy_height=6.5)
        assert "ach_patch" in result
        assert result["ach_patch"] > 0

    def test_ventilation_efficiency_in_range(self, patch_result):
        result = aggregate_to_patch(patch_result, (0, 0))
        # In synthetic data, U_mean in central 100m is small (ramps from 0.3)
        # U_ref = 5.0. Efficiency should be well below 1.
        assert 0 < result["vent_efficiency"] < 1

    def test_offset_patch_still_works(self, patch_result):
        # Patch centred at (100, 0) — should capture higher velocities
        result_center = aggregate_to_patch(patch_result, (0, 0))
        result_offset = aggregate_to_patch(patch_result, (100, 0))
        assert result_offset["U_mean"] > result_center["U_mean"]

    def test_empty_patch_gracefully_handled(self, patch_result):
        # Patch far outside the domain → no samples
        result = aggregate_to_patch(patch_result, (10000, 10000))
        assert result["n_samples"] == 0


class TestAggregateToGrid:
    def test_produces_cell_level_metrics(self, patch_result):
        grid = _make_grid()
        cells = aggregate_to_grid(patch_result, grid, (0, 0))
        # 10×10 grid at 10m spacing → centroids at ±5…±45; 80 centroids fall
        # inside the 50m-radius circle (π/4 × 100 = 78.54, rounded up to 80
        # by the discrete centroid pattern).
        assert len(cells) == 80
        assert "cfd_U_mean" in cells.columns
        assert "cfd_ach" in cells.columns
        # At least some cells have valid data
        assert cells["cfd_n_samples"].sum() > 0

    def test_drops_cells_outside_patch(self, patch_result):
        # Build a grid larger than the analysis patch
        grid = _make_grid(extent=200)  # 20×20 cells at 10m
        assert len(grid) == 400
        cells = aggregate_to_grid(
            patch_result, grid, (0, 0), analysis_patch_diameter=100
        )
        # Only centroids inside the 100m-diameter circle survive — 80 cells.
        assert len(cells) == 80

    def test_tags_with_patch_and_direction(self, patch_result):
        grid = _make_grid()
        cells = aggregate_to_grid(patch_result, grid, (0, 0))
        assert cells["cfd_patch_id"].unique().tolist() == ["TST-P01"]
        assert cells["cfd_wind_direction"].unique().tolist() == ["E"]


class TestAggregateToDomain:
    def test_covers_more_samples(self, patch_result):
        """The 250m domain should contain more samples than the 100m patch."""
        patch_result_inner = aggregate_to_patch(patch_result, (0, 0), 100)
        domain_result = aggregate_to_domain(patch_result, (0, 0), 250)
        assert domain_result["n_samples_domain"] > patch_result_inner["n_samples"]
