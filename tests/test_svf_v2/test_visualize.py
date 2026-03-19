"""Tests for src/svf_v2/visualize.py."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from src.svf_v2.visualize import plot_svf_comparison, plot_svf_dashboard


# ---------------------------------------------------------------------------
# Fixtures – synthetic GeoDataFrames
# ---------------------------------------------------------------------------


def _make_street_gdf(
    n: int = 200,
    mean_svf: float = 0.6,
    std_svf: float = 0.15,
    with_street_id: bool = False,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """Build a synthetic street-point GeoDataFrame with an ``svf`` column."""
    rng = np.random.default_rng(seed)
    svf = np.clip(rng.normal(mean_svf, std_svf, n), 0.0, 1.0)
    xs = rng.uniform(0, 1000, n)
    ys = rng.uniform(0, 1000, n)
    geom = [Point(x, y) for x, y in zip(xs, ys)]
    data = {"svf": svf, "geometry": geom}
    if with_street_id:
        data["street_id"] = rng.integers(0, 25, n)
    return gpd.GeoDataFrame(data, crs="EPSG:32723")


@pytest.fixture
def informal_gdf() -> gpd.GeoDataFrame:
    """Synthetic informal-settlement SVF data (low mean)."""
    return _make_street_gdf(n=300, mean_svf=0.43, std_svf=0.12, seed=1)


@pytest.fixture
def planned_gdf() -> gpd.GeoDataFrame:
    """Synthetic planned-grid SVF data (high mean)."""
    return _make_street_gdf(n=500, mean_svf=0.90, std_svf=0.06, seed=2)


@pytest.fixture
def street_gdf_with_segments() -> gpd.GeoDataFrame:
    """Street GeoDataFrame that includes a street_id column."""
    return _make_street_gdf(
        n=400, mean_svf=0.55, std_svf=0.18, with_street_id=True, seed=3
    )


@pytest.fixture
def street_gdf_no_segments() -> gpd.GeoDataFrame:
    """Street GeoDataFrame without a street_id column."""
    return _make_street_gdf(n=200, mean_svf=0.65, std_svf=0.10, seed=4)


# ---------------------------------------------------------------------------
# Tests – plot_svf_comparison
# ---------------------------------------------------------------------------


class TestPlotSvfComparison:
    """Tests for the comparative KDE + violin figure."""

    def test_creates_output_file(self, tmp_path, informal_gdf, planned_gdf):
        out = tmp_path / "comparison.png"
        result = plot_svf_comparison(
            {"Vidigal": informal_gdf, "Cidade de Deus": planned_gdf}, out
        )
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_area(self, tmp_path, informal_gdf):
        out = tmp_path / "single.png"
        result = plot_svf_comparison({"Vidigal": informal_gdf}, out)
        assert result == out
        assert out.exists()

    def test_three_areas(self, tmp_path, informal_gdf, planned_gdf):
        third = _make_street_gdf(n=150, mean_svf=0.70, std_svf=0.10, seed=99)
        out = tmp_path / "three.png"
        result = plot_svf_comparison(
            {"A": informal_gdf, "B": planned_gdf, "C": third}, out
        )
        assert result == out
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path, informal_gdf, planned_gdf):
        out = tmp_path / "sub" / "dir" / "comparison.png"
        result = plot_svf_comparison(
            {"Vidigal": informal_gdf, "CDD": planned_gdf}, out
        )
        assert result == out
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests – plot_svf_dashboard
# ---------------------------------------------------------------------------


class TestPlotSvfDashboard:
    """Tests for the 2x2 dashboard figure."""

    def test_creates_output_with_segments(self, tmp_path, street_gdf_with_segments):
        out = tmp_path / "dashboard.png"
        result = plot_svf_dashboard(street_gdf_with_segments, out, title="Test Area")
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_output_without_segments(self, tmp_path, street_gdf_no_segments):
        out = tmp_path / "dashboard_noseg.png"
        result = plot_svf_dashboard(street_gdf_no_segments, out)
        assert result == out
        assert out.exists()

    def test_with_roads_overlay(self, tmp_path, street_gdf_with_segments):
        from shapely.geometry import LineString

        roads = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(0, 0), (1000, 1000)]),
                    LineString([(0, 500), (1000, 500)]),
                ]
            },
            crs="EPSG:32723",
        )
        out = tmp_path / "dashboard_roads.png"
        result = plot_svf_dashboard(
            street_gdf_with_segments, out, roads_gdf=roads, title="With Roads"
        )
        assert result == out
        assert out.exists()

    def test_no_title(self, tmp_path, street_gdf_with_segments):
        out = tmp_path / "dashboard_notitle.png"
        result = plot_svf_dashboard(street_gdf_with_segments, out)
        assert result == out
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path, street_gdf_no_segments):
        out = tmp_path / "nested" / "path" / "dashboard.png"
        result = plot_svf_dashboard(street_gdf_no_segments, out)
        assert result == out
        assert out.exists()
