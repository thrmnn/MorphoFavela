"""Tests for src/solar/visualize.py — solar visualization smoke tests.

Each test verifies that the corresponding plot function creates an output
file without raising an exception.  The visualization module is expected
to be created in parallel; these tests will fail with ImportError until
the module exists.
"""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Point

from src.solar.visualize import (
    plot_solar_access,
    plot_solar_seasonal_panel,
    plot_solar_distribution,
    plot_solar_dashboard,
    plot_solar_sun_path,
    plot_solar_irradiance_map,
    plot_solar_vs_svf,
    plot_solar_comparison,
)


# ===================================================================
# Tests: plot_solar_access
# ===================================================================


class TestPlotSolarAccess:
    def test_creates_file(self, sample_street_gdf, tmp_path):
        gdf = sample_street_gdf.copy()
        gdf["solar_hours"] = [3.0, 5.0, 8.0]
        out = tmp_path / "solar.png"
        plot_solar_access(gdf, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ===================================================================
# Tests: plot_solar_seasonal_panel
# ===================================================================


class TestPlotSolarSeasonalPanel:
    def test_creates_file(self, sample_street_gdf, tmp_path):
        gdf = sample_street_gdf.copy()
        # The function auto-detects columns matching "solar_hours_20*"
        gdf["solar_hours_2026-06-21"] = [2.0, 5.0, 8.0]
        gdf["solar_hours_2026-12-21"] = [4.0, 7.0, 10.0]
        gdf["solar_hours_2026-03-20"] = [3.0, 6.0, 9.0]
        gdf["solar_hours_2026-09-22"] = [2.5, 5.5, 8.5]
        out = tmp_path / "panel.png"
        plot_solar_seasonal_panel(gdf, out)
        assert out.exists()


# ===================================================================
# Tests: plot_solar_distribution
# ===================================================================


class TestPlotSolarDistribution:
    def test_creates_file(self, sample_street_gdf, tmp_path):
        gdf = sample_street_gdf.copy()
        gdf["solar_hours_mean"] = [2.0, 5.0, 8.0]
        out = tmp_path / "dist.png"
        plot_solar_distribution(gdf, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ===================================================================
# Tests: plot_solar_dashboard
# ===================================================================


class TestPlotSolarDashboard:
    def test_creates_file(self, sample_street_gdf, tmp_path):
        gdf = sample_street_gdf.copy()
        gdf["solar_hours_mean"] = [2.0, 5.0, 8.0]
        out = tmp_path / "dash.png"
        plot_solar_dashboard(gdf, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ===================================================================
# Tests: plot_solar_sun_path
# ===================================================================


class TestPlotSunPath:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "sunpath.png"
        plot_solar_sun_path(-22.97, -43.17, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ===================================================================
# Tests: plot_solar_irradiance_map
# ===================================================================


class TestPlotSolarIrradianceMap:
    def test_creates_file(self, sample_street_gdf, tmp_path):
        gdf = sample_street_gdf.copy()
        gdf["irradiance_mean"] = [500.0, 1200.0, 2500.0]
        out = tmp_path / "irr.png"
        plot_solar_irradiance_map(gdf, out)
        assert out.exists()


# ===================================================================
# Tests: plot_solar_vs_svf
# ===================================================================


class TestPlotSolarVsSvf:
    def test_creates_file(self, sample_street_gdf, tmp_path):
        gdf = sample_street_gdf.copy()
        gdf["solar_hours_mean"] = [2.0, 5.0, 8.0]
        out = tmp_path / "svf_vs_solar.png"
        plot_solar_vs_svf(gdf, out)
        assert out.exists()


# ===================================================================
# Tests: plot_solar_comparison
# ===================================================================


class TestPlotSolarComparison:
    def test_creates_file(self, sample_street_gdf, tmp_path):
        gdf = sample_street_gdf.copy()
        gdf["solar_hours_mean"] = [2.0, 5.0, 8.0]
        out = tmp_path / "compare.png"
        plot_solar_comparison({"vidigal": gdf}, out)
        assert out.exists()

    def test_multiple_areas(self, sample_street_gdf, tmp_path):
        gdf1 = sample_street_gdf.copy()
        gdf1["solar_hours_mean"] = [2.0, 5.0, 8.0]
        gdf2 = sample_street_gdf.copy()
        gdf2["solar_hours_mean"] = [6.0, 7.0, 9.0]
        out = tmp_path / "compare_multi.png"
        plot_solar_comparison({"vidigal": gdf1, "copacabana": gdf2}, out)
        assert out.exists()
