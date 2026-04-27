"""CFD result integration — map CFD outputs onto the 10m morphometric grid.

Ingests OpenFOAM CFD results (per-patch, per-wind-direction) and aggregates
them to the 10m grid cells within each patch's 100m analysis zone.
Supports wind-rose weighting across 8 cardinal directions.

Public API:
    load_patch_csv, load_campaign_results      — I/O
    aggregate_to_grid, aggregate_to_patch       — spatial aggregation
    weighted_by_wind_rose                       — 8-direction ensemble
    ach, stagnation_fraction,
    turbulent_intensity, velocity_magnitude    — health-relevant metrics

See `src/cfd_integration/README.md` for the CFD result file format spec
that the OpenFOAM-side agent is expected to produce.
"""

from src.cfd_integration.aggregate import (
    aggregate_to_grid,
    aggregate_to_patch,
)
from src.cfd_integration.io import (
    load_campaign_results,
    load_patch_csv,
    load_patch_vtu,
)
from src.cfd_integration.metrics import (
    ach,
    stagnation_fraction,
    turbulent_intensity,
    velocity_magnitude,
)
from src.cfd_integration.schema import (
    WIND_DIRECTIONS_8,
    CFDCampaignResult,
    CFDPatchResult,
    CFDSamplePoint,
    WindRose,
)
from src.cfd_integration.weighting import (
    load_wind_rose,
    weighted_by_wind_rose,
)

__all__ = [
    "CFDSamplePoint",
    "CFDPatchResult",
    "CFDCampaignResult",
    "WindRose",
    "WIND_DIRECTIONS_8",
    "load_patch_csv",
    "load_patch_vtu",
    "load_campaign_results",
    "aggregate_to_grid",
    "aggregate_to_patch",
    "velocity_magnitude",
    "stagnation_fraction",
    "turbulent_intensity",
    "ach",
    "weighted_by_wind_rose",
    "load_wind_rose",
]
