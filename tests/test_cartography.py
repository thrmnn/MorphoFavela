"""Tests for src/cartography.py."""

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely.geometry import Point, box

from src.cartography import (
    add_north_arrow,
    add_scale_bar,
    add_settlement_boundary,
    apply_publication_style,
    format_utm_axes,
    get_svf_cmap,
    style_buildings_by_height,
)

matplotlib.use("Agg")


@pytest.fixture
def utm_ax():
    """Axes with UTM-like coordinate extent."""
    fig, ax = plt.subplots()
    ax.set_xlim(680000, 681000)
    ax.set_ylim(7460000, 7461000)
    yield ax
    plt.close(fig)


@pytest.fixture
def buildings_gdf():
    """Synthetic building footprints with heights."""
    geoms = [box(i * 10, 0, i * 10 + 8, 8) for i in range(5)]
    return gpd.GeoDataFrame(
        {"altura": [3.0, 6.0, 9.0, 12.0, 15.0]},
        geometry=geoms,
        crs="EPSG:31983",
    )


@pytest.fixture
def boundary_gdf():
    """Synthetic settlement boundary."""
    return gpd.GeoDataFrame(
        geometry=[box(0, 0, 50, 50)],
        crs="EPSG:31983",
    )


class TestAddScaleBar:
    def test_auto_size(self, utm_ax):
        add_scale_bar(utm_ax)
        # Should not raise; patches added
        assert len(utm_ax.patches) > 0

    def test_explicit_length(self, utm_ax):
        add_scale_bar(utm_ax, length_m=200)
        assert len(utm_ax.patches) > 0

    def test_locations(self, utm_ax):
        for loc in ("lower left", "lower right", "upper left", "upper right"):
            add_scale_bar(utm_ax, location=loc)


class TestAddNorthArrow:
    def test_adds_annotation(self, utm_ax):
        add_north_arrow(utm_ax)
        # Check that text "N" was added
        texts = [t.get_text() for t in utm_ax.texts]
        assert "N" in texts


class TestFormatUtmAxes:
    def test_with_labels(self, utm_ax):
        format_utm_axes(utm_ax, show_labels=True)

    def test_without_labels(self, utm_ax):
        format_utm_axes(utm_ax, show_labels=False)


class TestAddSettlementBoundary:
    def test_draws_boundary(self, utm_ax, boundary_gdf):
        add_settlement_boundary(utm_ax, boundary_gdf)


class TestStyleBuildingsByHeight:
    def test_with_height_column(self, utm_ax, buildings_gdf):
        style_buildings_by_height(utm_ax, buildings_gdf)

    def test_without_height_column(self, utm_ax, buildings_gdf):
        gdf_no_height = buildings_gdf.drop(columns=["altura"])
        style_buildings_by_height(utm_ax, gdf_no_height)

    def test_none_gdf(self, utm_ax):
        style_buildings_by_height(utm_ax, None)

    def test_empty_gdf(self, utm_ax):
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:31983")
        style_buildings_by_height(utm_ax, gdf)


class TestGetSvfCmap:
    def test_returns_cividis(self):
        assert get_svf_cmap() == "cividis"


class TestApplyPublicationStyle:
    def test_sets_rcparams(self):
        apply_publication_style()
        assert plt.rcParams["axes.facecolor"] == "white"
        assert plt.rcParams["figure.facecolor"] == "white"
