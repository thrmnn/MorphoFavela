"""Environmental exposure metrics, deprivation formulas, and visualisation.

Two layers:

* ``src.exposure.deprivation`` — pure arithmetic for the three deprivation
  formulas (solar, ventilation, hotspot). Type-agnostic; works on numpy
  arrays and pandas Series alike. Used by the two ``compute_deprivation_*``
  scripts so they can't silently drift.
* ``src.exposure.sky_exposure`` — zone-level metrics + composite exposure
  index + the two plotting routines. Heavyweight (geopandas, matplotlib);
  used by the legacy zone-analysis pipeline.

Public API (re-exported here for stable callers):
    Deprivation formulas:
        solar_deficit, ventilation_deficit, hotspot_index,
        DEFAULT_SOLAR_REFERENCE_HOURS
    Zone-level metrics + composite:
        compute_zone_solar_deficit, compute_zone_svf_deficit,
        compute_zone_building_density, compute_exposure_index
    Visualisation:
        plot_exposure_panel, plot_exposure_bivariate
"""

from src.exposure.deprivation import (
    DEFAULT_SOLAR_REFERENCE_HOURS,
    hotspot_index,
    solar_deficit,
    ventilation_deficit,
)
from src.exposure.sky_exposure import (
    compute_exposure_index,
    compute_zone_building_density,
    compute_zone_solar_deficit,
    compute_zone_svf_deficit,
    plot_exposure_bivariate,
    plot_exposure_panel,
)

__all__ = [
    "DEFAULT_SOLAR_REFERENCE_HOURS",
    "solar_deficit",
    "ventilation_deficit",
    "hotspot_index",
    "compute_zone_solar_deficit",
    "compute_zone_svf_deficit",
    "compute_zone_building_density",
    "compute_exposure_index",
    "plot_exposure_panel",
    "plot_exposure_bivariate",
]
