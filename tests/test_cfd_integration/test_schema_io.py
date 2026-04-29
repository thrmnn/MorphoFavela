"""Tests for schema dataclasses and CSV/JSON I/O."""

import json

import pandas as pd
import pytest

from src.cfd_integration.io import load_patch_csv
from src.cfd_integration.schema import (
    WIND_DIRECTIONS_8,
    CFDPatchResult,
    PatchSimulationMetadata,
    WindRose,
)


class TestWindDirections:
    def test_8_directions(self):
        assert len(WIND_DIRECTIONS_8) == 8
        assert set(WIND_DIRECTIONS_8) == {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
        # Bearings from north, clockwise
        assert WIND_DIRECTIONS_8["N"] == 0
        assert WIND_DIRECTIONS_8["E"] == 90
        assert WIND_DIRECTIONS_8["S"] == 180
        assert WIND_DIRECTIONS_8["W"] == 270


class TestWindRose:
    def test_normalization(self):
        # Frequencies not summing to 1 get normalised
        wr = WindRose(site="x", frequencies={"N": 2.0, "E": 2.0})
        assert wr.frequencies["N"] == pytest.approx(0.5)
        assert wr.frequencies["E"] == pytest.approx(0.5)

    def test_no_normalization_needed(self):
        wr = WindRose(site="x", frequencies={"N": 0.5, "E": 0.5})
        assert wr.frequencies["N"] == pytest.approx(0.5)

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValueError, match="Unknown wind direction"):
            WindRose(site="x", frequencies={"XY": 1.0})


class TestCFDPatchResult:
    def test_requires_mandatory_cols(self):
        meta = PatchSimulationMetadata(
            patch_id="P01",
            site="x",
            wind_direction="N",
            wind_speed_ref=5.0,
        )
        # Missing U_mag
        bad = pd.DataFrame({"x": [0], "y": [0], "z": [1.5], "U": [1], "V": [0], "W": [0]})
        with pytest.raises(ValueError, match="missing columns"):
            CFDPatchResult(metadata=meta, samples=bad)

    def test_valid_construction(self, synthetic_metadata, synthetic_samples):
        meta = PatchSimulationMetadata(**synthetic_metadata)
        result = CFDPatchResult(metadata=meta, samples=synthetic_samples)
        assert result.metadata.patch_id == "TST-P01"
        assert len(result.samples) > 1000


class TestLoadPatchCSV:
    def test_roundtrip(self, synthetic_patch_csv):
        csv_path, json_path = synthetic_patch_csv
        result = load_patch_csv(csv_path, json_path)
        assert result.metadata.patch_id == "TST-P01"
        assert result.metadata.wind_direction == "E"
        assert result.metadata.wind_speed_ref == 5.0
        assert "U_mag" in result.samples.columns
        assert len(result.samples) > 1000

    def test_autocomputes_umag(self, tmp_path, synthetic_metadata):
        """If U_mag is missing but U/V/W present, it's auto-computed."""
        samples = pd.DataFrame(
            {
                "x": [0, 1],
                "y": [0, 1],
                "z": [1.5, 1.5],
                "U": [3.0, 0.0],
                "V": [4.0, 0.0],
                "W": [0.0, 0.0],
                "TKE": [0.1, 0.1],
            }
        )
        csv = tmp_path / "sample_points.csv"
        samples.to_csv(csv, index=False)
        with open(tmp_path / "summary.json", "w") as f:
            json.dump(synthetic_metadata, f)

        result = load_patch_csv(csv)
        assert "U_mag" in result.samples.columns
        assert result.samples.iloc[0]["U_mag"] == pytest.approx(5.0)

    def test_missing_csv(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_patch_csv(tmp_path / "does_not_exist.csv")

    def test_missing_metadata(self, tmp_path, synthetic_samples):
        csv = tmp_path / "sample_points.csv"
        synthetic_samples.to_csv(csv, index=False)
        # No summary.json written
        with pytest.raises(FileNotFoundError, match="metadata"):
            load_patch_csv(csv)
