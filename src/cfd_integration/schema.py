"""Data contracts for CFD results.

All CFD result payloads are defined here. The OpenFOAM-side agent must
produce CSVs that map to CFDSamplePoint, plus summary.json per patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# 8 cardinal directions as bearing from north (degrees, meteorological convention)
WIND_DIRECTIONS_8 = {
    "N": 0.0,
    "NE": 45.0,
    "E": 90.0,
    "SE": 135.0,
    "S": 180.0,
    "SW": 225.0,
    "W": 270.0,
    "NW": 315.0,
}


@dataclass
class CFDSamplePoint:
    """A single sampled point from a CFD result at pedestrian height (1.5m).

    Matches the CSV columns produced by the OpenFOAM post-processing step
    (see `src/cfd_integration/README.md` for the exact CSV schema).
    """

    x: float  # UTM easting (m)
    y: float  # UTM northing (m)
    z: float  # elevation above ground (m), nominally 1.5
    U: float  # velocity x-component (m/s), east-positive
    V: float  # velocity y-component (m/s), north-positive
    W: float  # velocity z-component (m/s), up-positive
    U_mag: float  # |U| = sqrt(U² + V² + W²)  (m/s)
    TKE: float  # turbulent kinetic energy (m²/s²)
    p: float = 0.0  # pressure (Pa), optional


@dataclass
class PatchSimulationMetadata:
    """Per-simulation metadata (one patch × one wind direction)."""

    patch_id: str
    site: str
    wind_direction: str  # "N", "NE", ... "NW"
    wind_speed_ref: float  # reference velocity at inlet (m/s, at 10m height)
    converged: bool = True  # did the simulation converge?
    residual_final: Optional[float] = None  # final residual (e.g., 1e-5)
    solver: str = "simpleFoam"
    turbulence_model: str = "kOmegaSST"
    n_iterations: int = 0
    wall_clock_s: float = 0.0


@dataclass
class CFDPatchResult:
    """All sample points for one patch × one wind direction.

    Sample points are the DataFrame (one row per sample, columns matching
    CFDSamplePoint fields). Metadata is a separate dataclass.
    """

    metadata: PatchSimulationMetadata
    samples: pd.DataFrame

    def __post_init__(self):
        required_cols = {"x", "y", "z", "U", "V", "W", "U_mag", "TKE"}
        missing = required_cols - set(self.samples.columns)
        if missing:
            raise ValueError(
                f"CFDPatchResult samples missing columns: {missing}. "
                f"Got: {sorted(self.samples.columns)}"
            )


@dataclass
class WindRose:
    """Wind rose: frequency of occurrence for each of the 8 cardinal directions.

    Used to weight CFD results across directions into an annualised ensemble.
    Frequencies should sum to 1.0 (or will be normalised).

    Optionally includes a speed distribution per direction (mean wind speed
    from INMET, used to scale dimensionless CFD results), plus provenance
    metadata so readers can assess confidence (station identity, time
    window, observation count, calm fraction, anemometer reference height,
    and a free-form quality flag).
    """

    site: str
    frequencies: dict[str, float]  # e.g., {"N": 0.05, "NE": 0.12, ...}
    mean_speeds: dict[str, float] = field(default_factory=dict)  # m/s per direction
    source: str = ""  # human-readable provenance line
    reference_height_m: Optional[float] = None  # anemometer height, nominally 10 m
    station_id: Optional[str] = None  # INMET code (e.g., "A652")
    station_name: Optional[str] = None  # e.g., "Alto da Boa Vista"
    station_coords: Optional[tuple[float, float]] = None  # (lat, lon)
    time_window_start: Optional[str] = None  # ISO date (e.g., "2015-01-01")
    time_window_end: Optional[str] = None  # ISO date
    n_observations: Optional[int] = None  # total observations in the window
    calm_fraction: Optional[float] = None  # fraction of obs with |U| < 0.5 m/s
    quality_flag: Optional[str] = None  # "measured" | "gap-filled" | "placeholder-prior"

    def __post_init__(self):
        # Normalise frequencies
        total = sum(self.frequencies.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            self.frequencies = {k: v / total for k, v in self.frequencies.items()}
        # Validate direction keys
        for d in self.frequencies:
            if d not in WIND_DIRECTIONS_8:
                raise ValueError(f"Unknown wind direction: {d}")

    @property
    def is_placeholder(self) -> bool:
        """True if this rose is a climatological prior, not station data."""
        return self.quality_flag == "placeholder-prior"


@dataclass
class CFDCampaignResult:
    """All CFD results for one site: N patches × up to 8 directions each."""

    site: str
    patches: dict[str, dict[str, CFDPatchResult]]
    # e.g., patches["VDG-P01"]["N"] = CFDPatchResult(...)
    wind_rose: Optional[WindRose] = None

    def patch_ids(self) -> list[str]:
        return sorted(self.patches.keys())

    def directions_for(self, patch_id: str) -> list[str]:
        return sorted(self.patches.get(patch_id, {}).keys())

    def n_simulations(self) -> int:
        return sum(len(v) for v in self.patches.values())
