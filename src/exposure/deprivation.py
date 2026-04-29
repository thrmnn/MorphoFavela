"""Composite morphological-environmental deprivation index — shared formulas.

The unit-level (`scripts/compute_deprivation_index.py`) and raster-level
(`scripts/compute_deprivation_index_raster.py`) variants both compute the
same three deficit dimensions and combine them into a hotspot index.
Centralising the math here means a change to the formula is a single
edit and the two variants can't silently drift apart.

The functions are deliberately type-agnostic: any input that supports
the standard arithmetic protocol (`+`, `-`, `/`, `*`) and a `.clip()`
method works — pandas Series, pandas DataFrame columns, and numpy
ndarrays all pass.

Formulas:
  solar_deficit    = clip(1 - hours / reference, 0, 1)
  ventilation_def  = clip(1 - (svf + porosity) / 2, 0, 1)
  hotspot_index    = (solar_def + vent_def + occ_score) / 3

Type-specific concerns stay in callers:
  - numpy NaN propagation
  - pandas percentile ranking for occupancy_score
  - plotting and file I/O
"""

from __future__ import annotations

# Default reference for solar deficit when callers don't pass one.
# 3.0 hours/day is the WHO healthy-window threshold; deficit = 1 below
# that, 0 if the unit/pixel meets or exceeds it.
DEFAULT_SOLAR_REFERENCE_HOURS = 3.0


def solar_deficit(solar_hours, reference: float = DEFAULT_SOLAR_REFERENCE_HOURS):
    """Solar-access deficit normalised to [0, 1].

    Returns ``clip(1 - solar_hours / reference, 0, 1)``. Higher values
    mean more deficit (less direct sunlight).
    """
    deficit = 1.0 - (solar_hours / reference)
    return deficit.clip(0.0, 1.0)


def ventilation_deficit(svf, porosity):
    """Ventilation deficit normalised to [0, 1].

    Combines sky-view factor and sectional porosity as a single proxy
    for ventilation potential, then takes the deficit:
    ``clip(1 - (svf + porosity) / 2, 0, 1)``.
    """
    score = (svf + porosity) / 2.0
    deficit = 1.0 - score
    return deficit.clip(0.0, 1.0)


def hotspot_index(solar_def, vent_def, occ_score):
    """Composite hotspot index — equal-weighted mean of the three deficits.

    Each input must already be in [0, 1]. The result is in [0, 1] where
    1 means maximum compounded environmental deprivation across all
    three dimensions.
    """
    return (solar_def + vent_def + occ_score) / 3.0
