"""Unit tests for the streetnet beco-configuration core functions."""

from __future__ import annotations

import os
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.morphometry.streetnet import (  # noqa: E402
    build_network_nodes,
    cell_streetnet_metrics,
    summarise_by_type,
)


def _plus_network() -> gpd.GeoDataFrame:
    """A '+' of four arms meeting at the origin → one degree-4 node."""
    lines = [
        LineString([(0, 0), (10, 0)]),
        LineString([(0, 0), (-10, 0)]),
        LineString([(0, 0), (0, 10)]),
        LineString([(0, 0), (0, -10)]),
    ]
    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:31983")


def test_degree_at_central_junction():
    nodes = build_network_nodes(_plus_network())
    central = nodes[nodes.geometry.distance(nodes.geometry.iloc[0].centroid) < 1]
    # The origin node has degree 4; the four tips have degree 1.
    assert nodes["degree"].max() == 4
    assert (nodes["degree"] == 1).sum() == 4
    assert (nodes["degree"] >= 3).sum() == 1


def test_snap_merges_near_coincident_endpoints():
    # Two segments whose ends are 0.2 m apart should snap into one degree-2 node.
    roads = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (5, 0)]),
            LineString([(5.2, 0), (10, 0)]),
        ],
        crs="EPSG:31983",
    )
    nodes = build_network_nodes(roads, tol=0.5)
    assert (nodes["degree"] == 2).sum() == 1


def test_cell_metrics_counts_junction_and_segment_length():
    grid = gpd.GeoDataFrame(
        {"zone_id": [1]}, geometry=[box(-10, -10, 10, 10)], crs="EPSG:31983"
    )
    cells = cell_streetnet_metrics(grid, _plus_network())
    row = cells.iloc[0]
    assert row["intersection_count"] == 1
    assert row["n_segments"] == 4
    # Each arm is 10 m long.
    assert abs(row["mean_segment_len_m"] - 10.0) < 1e-6
    # 400 m^2 cell = 0.04 ha → 1 junction / 0.04 = 25 per ha.
    assert abs(row["intersection_density_ha"] - 25.0) < 1e-6


def test_summarise_by_type_drops_nan_and_groups():
    cells = pd.DataFrame({
        "zone_id": [1, 2, 3],
        "intersection_count": [1, 0, 2],
        "intersection_density_ha": [25.0, 0.0, 50.0],
        "mean_segment_len_m": [10.0, float("nan"), 5.0],
        "n_segments": [4, 0, 3],
        "morphotype_smooth": [4, 4, None],
    })
    summary = summarise_by_type(cells)
    # The NaN-type cell is dropped; only type 4 survives.
    assert summary["morphotype"].tolist() == [4]
    assert summary.loc[0, "n_cells"] == 2
    assert abs(summary.loc[0, "frac_cells_with_road"] - 0.5) < 1e-9
