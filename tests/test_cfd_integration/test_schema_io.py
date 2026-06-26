"""Tests for schema dataclasses and CSV/JSON I/O."""

import json

import pandas as pd
import pytest

from src.cfd_integration.io import (
    _normalize_wind_direction,
    load_campaign_results,
    load_patch_csv,
    load_patch_parquet,
)
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


class TestProvenanceAndContract:
    """S4: synthetic/provenance fields + wind_direction contract enforcement."""

    def test_legacy_metadata_defaults(self, tmp_path, synthetic_samples, synthetic_metadata):
        """A real/legacy summary.json without the new keys loads with
        synthetic=False / provenance=None — backward compatible."""
        assert "synthetic" not in synthetic_metadata
        assert "provenance" not in synthetic_metadata
        csv = tmp_path / "sample_points.csv"
        synthetic_samples.to_csv(csv, index=False)
        with open(tmp_path / "summary.json", "w") as f:
            json.dump(synthetic_metadata, f)

        result = load_patch_csv(csv)
        assert result.metadata.synthetic is False
        assert result.metadata.provenance is None

    def test_synthetic_stamped_metadata(self, tmp_path, synthetic_samples, synthetic_metadata):
        """A synthetic-stamped summary.json round-trips synthetic=True and
        the provenance block."""
        meta = dict(synthetic_metadata)
        meta["synthetic"] = True
        meta["provenance"] = {
            "generator": "scripts/generate_synthetic_cfd_results.py",
            "warning": "SYNTHETIC — not a CFD solve.",
        }
        csv = tmp_path / "sample_points.csv"
        synthetic_samples.to_csv(csv, index=False)
        with open(tmp_path / "summary.json", "w") as f:
            json.dump(meta, f)

        result = load_patch_csv(csv)
        assert result.metadata.synthetic is True
        assert result.metadata.provenance["generator"].endswith(
            "generate_synthetic_cfd_results.py"
        )

    def test_invalid_wind_direction_raises(self, tmp_path, synthetic_samples, synthetic_metadata):
        """An unknown wind_direction in summary.json is a contract violation."""
        meta = dict(synthetic_metadata)
        meta["wind_direction"] = "NORTHEAST"
        csv = tmp_path / "sample_points.csv"
        synthetic_samples.to_csv(csv, index=False)
        with open(tmp_path / "summary.json", "w") as f:
            json.dump(meta, f)

        with pytest.raises(ValueError, match="Unknown wind_direction"):
            load_patch_csv(csv)

    def test_wind_nnn_direction_accepted(self, tmp_path, synthetic_samples, synthetic_metadata):
        """The documented wind_NNN form is a valid summary.json direction."""
        meta = dict(synthetic_metadata)
        meta["wind_direction"] = "wind_045"
        csv = tmp_path / "sample_points.csv"
        synthetic_samples.to_csv(csv, index=False)
        with open(tmp_path / "summary.json", "w") as f:
            json.dump(meta, f)

        result = load_patch_csv(csv)
        assert result.metadata.wind_direction == "wind_045"


class TestNormalizeWindDirection:
    """The Airflow producer uses wind_NNN/ directories instead of N/, NE/, etc.
    The normalizer accepts both and returns the canonical cardinal name."""

    def test_cardinal_passes_through(self):
        for d in WIND_DIRECTIONS_8:
            assert _normalize_wind_direction(d) == d

    def test_airflow_degree_form(self):
        assert _normalize_wind_direction("wind_000") == "N"
        assert _normalize_wind_direction("wind_045") == "NE"
        assert _normalize_wind_direction("wind_090") == "E"
        assert _normalize_wind_direction("wind_135") == "SE"
        assert _normalize_wind_direction("wind_180") == "S"
        assert _normalize_wind_direction("wind_225") == "SW"
        assert _normalize_wind_direction("wind_270") == "W"
        assert _normalize_wind_direction("wind_315") == "NW"

    def test_unknown_returns_none(self):
        assert _normalize_wind_direction("wind_022") is None  # off-axis
        assert _normalize_wind_direction("wind_NN") is None
        assert _normalize_wind_direction("north") is None
        assert _normalize_wind_direction("") is None
        assert _normalize_wind_direction("wind_") is None


class TestLoadPatchParquet:
    def test_roundtrip_single_file(self, tmp_path, synthetic_samples, synthetic_metadata):
        parquet = tmp_path / "samples.parquet"
        synthetic_samples.to_parquet(parquet, index=False)
        with open(tmp_path / "summary.json", "w") as f:
            json.dump(synthetic_metadata, f)

        result = load_patch_parquet(parquet)
        assert result.metadata.patch_id == "TST-P01"
        assert result.metadata.wind_direction == "E"
        assert "U_mag" in result.samples.columns
        assert len(result.samples) == len(synthetic_samples)

    def test_concat_multiple_parquets(self, tmp_path, synthetic_samples, synthetic_metadata):
        """Airflow may emit one parquet per processor decomposition; concat them."""
        half = len(synthetic_samples) // 2
        a = synthetic_samples.iloc[:half]
        b = synthetic_samples.iloc[half:]
        a.to_parquet(tmp_path / "samples_0.parquet", index=False)
        b.to_parquet(tmp_path / "samples_1.parquet", index=False)
        with open(tmp_path / "summary.json", "w") as f:
            json.dump(synthetic_metadata, f)

        result = load_patch_parquet(
            [tmp_path / "samples_0.parquet", tmp_path / "samples_1.parquet"]
        )
        assert len(result.samples) == len(synthetic_samples)

    def test_autocomputes_umag(self, tmp_path, synthetic_metadata):
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
        parquet = tmp_path / "samples.parquet"
        samples.to_parquet(parquet, index=False)
        with open(tmp_path / "summary.json", "w") as f:
            json.dump(synthetic_metadata, f)

        result = load_patch_parquet(parquet)
        assert "U_mag" in result.samples.columns
        assert result.samples.iloc[0]["U_mag"] == pytest.approx(5.0)

    def test_missing_parquet(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_patch_parquet(tmp_path / "does_not_exist.parquet")

    def test_empty_path_list(self):
        with pytest.raises(ValueError, match="at least one parquet"):
            load_patch_parquet([])


class TestLoadCampaignAutodetect:
    """`load_campaign_results` should transparently handle both layouts."""

    def _write_summary(self, dir_path, patch_id, wind_direction):
        with open(dir_path / "summary.json", "w") as f:
            json.dump(
                {
                    "patch_id": patch_id,
                    "site": "test_site",
                    "wind_direction": wind_direction,
                    "wind_speed_ref": 5.0,
                    "converged": True,
                },
                f,
            )

    def test_ivf_native_layout(self, tmp_path, synthetic_samples):
        """Cardinal direction dirs + sample_points.csv → loads cleanly."""
        root = tmp_path / "cfd_results"
        for direction in ["N", "E"]:
            d = root / "TST-P01" / direction
            d.mkdir(parents=True)
            synthetic_samples.to_csv(d / "sample_points.csv", index=False)
            self._write_summary(d, "TST-P01", direction)

        result = load_campaign_results(site="test_site", results_root=root)
        assert sorted(result.directions_for("TST-P01")) == ["E", "N"]

    def test_airflow_native_layout(self, tmp_path, synthetic_samples):
        """wind_NNN/ dirs + parquet → mapped to cardinal in the result."""
        root = tmp_path / "cfd_results"
        for deg, cardinal in [("000", "N"), ("090", "E")]:
            d = root / "TST-P01" / f"wind_{deg}"
            d.mkdir(parents=True)
            synthetic_samples.to_parquet(d / "samples.parquet", index=False)
            self._write_summary(d, "TST-P01", cardinal)

        result = load_campaign_results(site="test_site", results_root=root)
        # Cardinal keys, even though the on-disk dirs are wind_NNN
        assert sorted(result.directions_for("TST-P01")) == ["E", "N"]

    def test_mixed_layouts_per_direction(self, tmp_path, synthetic_samples):
        """One direction CSV, another parquet — both load."""
        root = tmp_path / "cfd_results"

        # MorphoFavela native: N
        d_n = root / "TST-P01" / "N"
        d_n.mkdir(parents=True)
        synthetic_samples.to_csv(d_n / "sample_points.csv", index=False)
        self._write_summary(d_n, "TST-P01", "N")

        # Airflow native: NE (wind_045)
        d_ne = root / "TST-P01" / "wind_045"
        d_ne.mkdir(parents=True)
        synthetic_samples.to_parquet(d_ne / "samples.parquet", index=False)
        self._write_summary(d_ne, "TST-P01", "NE")

        result = load_campaign_results(site="test_site", results_root=root)
        assert sorted(result.directions_for("TST-P01")) == ["N", "NE"]
