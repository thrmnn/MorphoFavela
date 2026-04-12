"""Health-relevant wind metrics computed from CFD samples.

All functions take arrays of sample values and return scalar metrics.
Vectorised for batch computation across grid cells.
"""

from __future__ import annotations

import numpy as np


def velocity_magnitude(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute |U| = sqrt(U² + V² + W²)."""
    return np.sqrt(u**2 + v**2 + w**2)


def stagnation_fraction(
    u_mag: np.ndarray, threshold: float = 0.5
) -> float:
    """Fraction of sample points with |U| below threshold.

    Parameters
    ----------
    u_mag : array
        Velocity magnitudes (m/s).
    threshold : float
        Stagnation threshold (m/s). Default 0.5 — a commonly cited limit
        for 'calm' air where pollutant accumulation becomes significant.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if len(u_mag) == 0:
        return np.nan
    return float((u_mag < threshold).sum() / len(u_mag))


def turbulent_intensity(tke: np.ndarray, u_ref: float) -> np.ndarray:
    """Turbulent intensity: TI = sqrt(2/3 × TKE) / U_ref.

    Parameters
    ----------
    tke : array
        Turbulent kinetic energy (m²/s²).
    u_ref : float
        Reference velocity (e.g., inlet velocity at 10m height).

    Returns
    -------
    array
        Dimensionless turbulent intensity.
    """
    if u_ref <= 0:
        return np.full_like(tke, np.nan, dtype=float)
    return np.sqrt(2.0 / 3.0 * tke) / u_ref


def ach(
    u_mag: np.ndarray,
    canopy_height: float,
    cell_area: float,
    averaging_height: float = 1.5,
) -> float:
    """Air change rate (per hour) for an urban canopy cell.

    For a square cell of side L receiving wind advectively through one face,
    the simplest physical scaling is:

        Q_in   = U × L × H_canopy        (volume flux through one face)
        V_cell = L² × H_canopy
        ACH    = 3600 × Q_in / V_cell = 3600 × U / L

    This is the standard urban-climate formulation (Buccolieri et al. 2010,
    Hang et al. 2012). It's a mean-flow approximation — true ACH includes
    turbulent exchange which can be significant in high-TKE cells. Use
    cfd_U_mean and cfd_TKE_mean from aggregate_to_grid for a richer picture.

    Parameters
    ----------
    u_mag : array
        Velocity magnitudes at pedestrian height within the cell (m/s).
    canopy_height : float
        Urban canopy height (m), typically H_mean — used for documentation
        only; the formula is independent of canopy height because Q and V
        both scale with it.
    cell_area : float
        Cell plan area (m²), e.g., 100 for 10m cells.
    averaging_height : float
        Height at which U was sampled (m, informational).

    Returns
    -------
    float
        Air change rate (h⁻¹).
    """
    if len(u_mag) == 0 or canopy_height <= 0 or cell_area <= 0:
        return np.nan
    u_mean = float(np.mean(u_mag))
    cell_length = np.sqrt(cell_area)
    # ACH = 3600 × U / L
    return 3600.0 * u_mean / cell_length


def low_wind_percentile(u_mag: np.ndarray, pct: float = 10) -> float:
    """Return the p-th percentile of |U| (default 10th).

    Low percentiles characterise the worst-case stagnation conditions —
    the cells that will be poorly ventilated during calm periods.
    """
    if len(u_mag) == 0:
        return np.nan
    return float(np.percentile(u_mag, pct))


def canyon_ventilation_efficiency(
    u_mag_in_canyon: np.ndarray, u_ref: float
) -> float:
    """Canyon ventilation efficiency: mean(|U|) / U_ref.

    Ratio of in-canyon wind speed to the reference (free-stream) velocity.
    Lower values indicate worse ventilation. Typical values:
        > 0.4 : well-ventilated
        0.2-0.4 : moderate
        < 0.2 : poorly ventilated (health concern)
    """
    if len(u_mag_in_canyon) == 0 or u_ref <= 0:
        return np.nan
    return float(np.mean(u_mag_in_canyon) / u_ref)


# Default thresholds for reporting (configurable in campaign analysis)
HEALTH_THRESHOLDS = {
    "u_mag_stagnation": 0.5,       # m/s, below = calm/stagnant
    "ach_poor": 3.0,                # h⁻¹, below = inadequate ventilation
    "ventilation_efficiency_poor": 0.2,  # canyon U / U_ref
    "TI_high": 0.5,                 # high turbulence (dispersion-dominated)
}
