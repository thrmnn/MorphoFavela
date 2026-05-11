"""Tests for src/solar/animation.py — wide-format sunlit GIS layer for dataviz."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import pyvista as pv
from shapely.geometry import Point

from src.solar.animation import (
    _timestamp_label,
    build_animation_manifest,
    sunlit_matrix_to_wide_gdf,
)
from src.solar.compute import compute_sunlit_matrix
from src.solar.sun import sun_position_to_direction


def _frame(time_str: str, alt: float, az: float) -> dict:
    return {
        "time": pd.Timestamp(time_str, tz="America/Sao_Paulo"),
        "altitude_deg": alt,
        "azimuth_deg": az,
        "direction": sun_position_to_direction(alt, az),
    }


class TestTimestampLabel:
    def test_basic(self):
        ts = pd.Timestamp("2026-06-21 07:00:00", tz="America/Sao_Paulo")
        assert _timestamp_label(ts) == "T0700"

    def test_afternoon(self):
        ts = pd.Timestamp("2026-06-21 14:30:00", tz="America/Sao_Paulo")
        assert _timestamp_label(ts) == "T1430"


class TestSunlitMatrixToWideGdf:
    def test_columns_and_shape(self):
        observers = np.array(
            [
                [0.0, 0.0, 1.5],
                [1.0, 0.0, 1.5],
                [0.0, 1.0, 1.5],
            ]
        )
        sunlit = np.array(
            [
                [True, False],
                [True, True],
                [False, True],
            ]
        )
        frames = [
            _frame("2026-06-21 07:00:00", 8.0, 56.0),
            _frame("2026-06-21 08:00:00", 19.0, 49.0),
        ]
        gdf, manifest = sunlit_matrix_to_wide_gdf(
            observers=observers,
            sunlit_matrix=sunlit,
            sun_frames=frames,
            crs="EPSG:31983",
            interval_minutes=60,
        )

        assert "lit_T0700" in gdf.columns
        assert "lit_T0800" in gdf.columns
        assert "geometry" in gdf.columns
        assert gdf.crs.to_string().endswith("31983")
        assert len(gdf) == 3

        # Boolean dtype preserved
        assert gdf["lit_T0700"].dtype == bool
        # solar_hours = row-sum * step
        assert gdf["solar_hours"].tolist() == [1.0, 2.0, 1.0]
        # Truth values intact
        assert gdf["lit_T0700"].tolist() == [True, True, False]
        assert gdf["lit_T0800"].tolist() == [False, True, True]

        # Manifest reflects counts
        assert manifest["n_points"] == 3
        assert manifest["n_frames"] == 2
        cols = [f["col"] for f in manifest["frames"]]
        assert cols == ["lit_T0700", "lit_T0800"]
        n_sunlit = [f["n_sunlit"] for f in manifest["frames"]]
        assert n_sunlit == [2, 2]

    def test_include_solar_hours_false(self):
        observers = np.array([[0.0, 0.0, 1.5]])
        sunlit = np.array([[True, False]])
        frames = [
            _frame("2026-06-21 07:00:00", 8.0, 56.0),
            _frame("2026-06-21 08:00:00", 19.0, 49.0),
        ]
        gdf, _ = sunlit_matrix_to_wide_gdf(
            observers=observers,
            sunlit_matrix=sunlit,
            sun_frames=frames,
            crs="EPSG:31983",
            interval_minutes=60,
            include_solar_hours=False,
        )
        assert "solar_hours" not in gdf.columns

    def test_mismatched_dimensions_raise(self):
        observers = np.array([[0.0, 0.0, 1.5]])
        sunlit = np.array([[True, False]])
        frames = [_frame("2026-06-21 07:00:00", 8.0, 56.0)]  # only 1 frame
        with pytest.raises(ValueError):
            sunlit_matrix_to_wide_gdf(
                observers=observers,
                sunlit_matrix=sunlit,
                sun_frames=frames,
                crs="EPSG:31983",
                interval_minutes=60,
            )

    def test_observer_row_mismatch(self):
        observers = np.array([[0.0, 0.0, 1.5], [1.0, 0.0, 1.5]])
        sunlit = np.array([[True]])  # only 1 row
        frames = [_frame("2026-06-21 07:00:00", 8.0, 56.0)]
        with pytest.raises(ValueError):
            sunlit_matrix_to_wide_gdf(
                observers=observers,
                sunlit_matrix=sunlit,
                sun_frames=frames,
                crs="EPSG:31983",
                interval_minutes=60,
            )


class TestEndToEndOnSyntheticScene:
    """Closed box → all-shaded, open ground → all-sunlit, verified via the helper."""

    def test_open_ground_all_lit(self):
        ground = pv.Plane(
            center=(5, 5, 0),
            direction=(0, 0, 1),
            i_size=10,
            j_size=10,
            i_resolution=4,
            j_resolution=4,
        )
        observers = np.array([[5.0, 5.0, 1.5], [3.0, 3.0, 1.5]])
        frames = [
            _frame("2026-06-21 09:00:00", 45.0, 90.0),
            _frame("2026-06-21 12:00:00", 60.0, 180.0),
        ]
        sun_dirs = np.stack([f["direction"] for f in frames], axis=0)

        sunlit = compute_sunlit_matrix(observers, sun_dirs, ground)
        gdf, manifest = sunlit_matrix_to_wide_gdf(
            observers=observers,
            sunlit_matrix=sunlit,
            sun_frames=frames,
            crs="EPSG:31983",
            interval_minutes=60,
        )
        assert gdf["lit_T0900"].all()
        assert gdf["lit_T1200"].all()
        assert all(f["n_sunlit"] == 2 for f in manifest["frames"])

    def test_enclosed_box_all_shaded(self):
        box = pv.Box(bounds=(0, 10, 0, 10, 0, 10))
        observers = np.array([[5.0, 5.0, 5.0]])
        frames = [
            _frame("2026-06-21 09:00:00", 45.0, 90.0),
            _frame("2026-06-21 12:00:00", 60.0, 180.0),
        ]
        sun_dirs = np.stack([f["direction"] for f in frames], axis=0)

        sunlit = compute_sunlit_matrix(observers, sun_dirs, box)
        gdf, manifest = sunlit_matrix_to_wide_gdf(
            observers=observers,
            sunlit_matrix=sunlit,
            sun_frames=frames,
            crs="EPSG:31983",
            interval_minutes=60,
        )
        assert not gdf["lit_T0900"].any()
        assert not gdf["lit_T1200"].any()
        assert all(f["n_sunlit"] == 0 for f in manifest["frames"])


class TestBuildAnimationManifest:
    def test_wraps_frame_manifest_grid_mode(self):
        frame_manifest = {"n_points": 100, "n_frames": 11, "interval_minutes": 60, "frames": []}
        full = build_animation_manifest(
            site="vidigal",
            date="2026-06-21",
            timezone="America/Sao_Paulo",
            crs="EPSG:31983",
            grid_spacing_m=5.0,
            evaluation_height_m=1.5,
            hour_start=7,
            hour_end=17,
            frame_manifest=frame_manifest,
        )
        assert full["site"] == "vidigal"
        assert full["date"] == "2026-06-21"
        assert full["crs"] == "EPSG:31983"
        assert full["hour_window_local"] == [7, 17]
        assert full["n_frames"] == 11
        assert full["frames"] == []
        assert full["grid_spacing_m"] == 5.0
        assert "observer_source" not in full

    def test_wraps_frame_manifest_street_mode(self):
        frame_manifest = {"n_points": 6876, "n_frames": 11, "interval_minutes": 60, "frames": []}
        full = build_animation_manifest(
            site="vidigal",
            date="2026-06-21",
            timezone="America/Sao_Paulo",
            crs="EPSG:31983",
            hour_start=7,
            hour_end=17,
            frame_manifest=frame_manifest,
            observer_source="outputs/vidigal/morphometrics/svf/svf_streets_solar.gpkg",
        )
        assert full["observer_source"].endswith("svf_streets_solar.gpkg")
        assert "grid_spacing_m" not in full
        assert "evaluation_height_m" not in full


class TestBaseGdfAppendMode:
    """Append lit_T* columns to a pre-built GeoDataFrame (street-points case)."""

    def _base_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                "street_id": [0, 0, 1],
                "z_observer": [101.5, 102.5, 95.5],
                "svf": [0.42, 0.31, 0.78],
                "solar_hours_winter": [2.5, 1.0, 5.5],
            },
            geometry=[Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)],
            crs="EPSG:31983",
        )

    def test_preserves_original_columns(self):
        base = self._base_gdf()
        observers = np.array([[0.0, 0.0, 101.5], [1.0, 0.0, 102.5], [0.0, 1.0, 95.5]])
        sunlit = np.array([[True, False], [True, True], [False, True]])
        frames = [
            _frame("2026-06-21 09:00:00", 27.0, 45.0),
            _frame("2026-06-21 12:00:00", 43.0, 358.0),
        ]
        gdf, manifest = sunlit_matrix_to_wide_gdf(
            observers=observers,
            sunlit_matrix=sunlit,
            sun_frames=frames,
            interval_minutes=60,
            base_gdf=base,
            solar_hours_col="lit_hours_total",
        )
        # All originals preserved
        assert "street_id" in gdf.columns
        assert "svf" in gdf.columns
        assert "solar_hours_winter" in gdf.columns
        # Pre-existing solar_hours_winter must NOT be overwritten
        assert gdf["solar_hours_winter"].tolist() == [2.5, 1.0, 5.5]
        # New convenience column uses a distinct name
        assert "lit_hours_total" in gdf.columns
        assert "solar_hours" not in gdf.columns  # default name unused in append mode
        assert gdf["lit_hours_total"].tolist() == [1.0, 2.0, 1.0]
        # lit_T* columns added
        assert "lit_T0900" in gdf.columns
        assert "lit_T1200" in gdf.columns
        # CRS inherited from base
        assert gdf.crs == base.crs
        # Geometry preserved
        assert (gdf.geometry == base.geometry).all()
        # Manifest count correct
        assert manifest["n_points"] == 3
        assert manifest["n_frames"] == 2

    def test_base_gdf_row_count_mismatch_raises(self):
        base = self._base_gdf().iloc[:2]  # only 2 rows
        observers = np.array([[0.0, 0.0, 101.5], [1.0, 0.0, 102.5], [0.0, 1.0, 95.5]])
        sunlit = np.array([[True, False], [True, True], [False, True]])
        frames = [
            _frame("2026-06-21 09:00:00", 27.0, 45.0),
            _frame("2026-06-21 12:00:00", 43.0, 358.0),
        ]
        with pytest.raises(ValueError):
            sunlit_matrix_to_wide_gdf(
                observers=observers,
                sunlit_matrix=sunlit,
                sun_frames=frames,
                interval_minutes=60,
                base_gdf=base,
            )

    def test_does_not_mutate_input(self):
        base = self._base_gdf()
        original_cols = set(base.columns)
        observers = np.array([[0.0, 0.0, 101.5], [1.0, 0.0, 102.5], [0.0, 1.0, 95.5]])
        sunlit = np.array([[True, False], [True, True], [False, True]])
        frames = [
            _frame("2026-06-21 09:00:00", 27.0, 45.0),
            _frame("2026-06-21 12:00:00", 43.0, 358.0),
        ]
        sunlit_matrix_to_wide_gdf(
            observers=observers,
            sunlit_matrix=sunlit,
            sun_frames=frames,
            interval_minutes=60,
            base_gdf=base,
        )
        assert set(base.columns) == original_cols
