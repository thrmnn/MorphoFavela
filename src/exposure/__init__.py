"""Composite morphological-environmental deprivation indices.

The two deprivation scripts in `scripts/` (unit-level and raster-level)
share the same underlying formulas; those formulas live here so any
change is a single point of edit.

Public API:
    solar_deficit       — 1 - hours/reference, clipped [0, 1]
    ventilation_deficit — 1 - (svf + porosity) / 2, clipped [0, 1]
    hotspot_index       — equal-weighted mean of the three deficits

Type-specific concerns (numpy NaN propagation, pandas percentile
ranking, plotting, I/O) stay in the calling scripts; only the
arithmetic is shared.
"""

from src.exposure.deprivation import (
    DEFAULT_SOLAR_REFERENCE_HOURS,
    hotspot_index,
    solar_deficit,
    ventilation_deficit,
)

__all__ = [
    "DEFAULT_SOLAR_REFERENCE_HOURS",
    "solar_deficit",
    "ventilation_deficit",
    "hotspot_index",
]
