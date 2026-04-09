"""
Clear-sky irradiance models for urban solar access analysis.

Direct beam: Hottel (1976) clear-sky transmittance.
Diffuse: Liu & Jordan (1960) isotropic model weighted by Sky View Factor.

References
----------
- Hottel, H.C. (1976). A simple model for estimating the transmittance of
  direct solar radiation through clear atmospheres. *Solar Energy*, 18, 129-134.
- Liu, B.Y.H. & Jordan, R.C. (1960). The interrelationship and characteristic
  distribution of direct, diffuse, and total solar radiation. *Solar Energy*, 4(3).
- Reindl, D.T., Beckman, W.A. & Duffie, J.A. (1990). Diffuse fraction
  correlations. *Solar Energy*, 45(1), 1-7.
- Spencer, J.W. (1971). Fourier series representation of the position of the
  sun. *Search*, 2(5), 172.
- Duffie, J.A. & Beckman, W.A. (2013). *Solar Engineering of Thermal Processes*.
  4th ed., Wiley.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOLAR_CONSTANT = 1361.0  # W/m^2 (total solar irradiance)


# ---------------------------------------------------------------------------
# Extra-terrestrial irradiance
# ---------------------------------------------------------------------------


def _extraterrestrial_irradiance(day_of_year: int) -> float:
    """Solar constant corrected for Earth-Sun distance (Spencer 1971)."""
    B = 2 * np.pi * (day_of_year - 1) / 365.0
    return SOLAR_CONSTANT * (
        1.00011
        + 0.034221 * np.cos(B)
        + 0.00128 * np.sin(B)
        + 0.000719 * np.cos(2 * B)
        + 0.000077 * np.sin(2 * B)
    )


# ---------------------------------------------------------------------------
# Direct normal irradiance — Hottel (1976)
# ---------------------------------------------------------------------------


def direct_normal_irradiance(
    altitude_deg: float,
    day_of_year: int,
    site_elevation_m: float = 100.0,
) -> float:
    """Clear-sky DNI using Hottel (1976) transmittance model.

    Hottel's beam transmittance for a clear atmosphere is:

        tau_b = a0 + a1 * exp(-k / sin(alpha))

    where ``a0``, ``a1``, ``k`` are climate-type coefficients corrected for
    site altitude.  This implementation uses the *tropical* climate type
    (appropriate for Rio de Janeiro) with altitude correction from
    Duffie & Beckman (2013), Table 2.8.1.

    Parameters
    ----------
    altitude_deg : float
        Solar altitude above the horizon in degrees.
    day_of_year : int
        Day of the year (1-365).
    site_elevation_m : float
        Site elevation above sea level in metres (default 100 m).

    Returns
    -------
    float
        Clear-sky direct normal irradiance in W/m^2.
        Returns 0 if the sun is at or below the horizon.
    """
    if altitude_deg <= 0:
        return 0.0

    sin_alt = np.sin(np.radians(altitude_deg))

    # Site altitude in km
    A = site_elevation_m / 1000.0

    # Hottel coefficients for *tropical* climate type (Duffie & Beckman Table 2.8.1)
    # Base values at sea level for 23-degree latitude (tropical):
    #   r0 = 0.4237, r1 = 0.5055, rk = 0.2711
    # Altitude corrections:
    #   a0 = r0 * (0.4237 - 0.00821*(6 - A)^2)  -- simplified Hottel formulation
    # Using Duffie & Beckman correction factors for altitude:
    a0_star = 0.4237 - 0.00821 * (6.0 - A) ** 2
    a1_star = 0.5055 + 0.00595 * (6.5 - A) ** 2
    k_star = 0.2711 + 0.01858 * (2.5 - A) ** 2

    # Climate-type correction for tropical (Duffie & Beckman Table 2.8.1)
    # r0=0.95, r1=0.98, rk=1.02
    a0 = 0.95 * a0_star
    a1 = 0.98 * a1_star
    k = 1.02 * k_star

    # Beam transmittance
    tau_b = a0 + a1 * np.exp(-k / sin_alt)

    # Extra-terrestrial irradiance on the normal plane
    I_ext = _extraterrestrial_irradiance(day_of_year)

    dni = I_ext * tau_b
    return float(max(0.0, dni))


# ---------------------------------------------------------------------------
# Direct horizontal irradiance
# ---------------------------------------------------------------------------


def direct_horizontal_irradiance(dni: float, altitude_deg: float) -> float:
    """Direct irradiance on a horizontal surface.

    Parameters
    ----------
    dni : float
        Direct normal irradiance (W/m^2).
    altitude_deg : float
        Solar altitude above horizon in degrees.

    Returns
    -------
    float
        DHI = DNI * sin(altitude).  Returns 0 if altitude <= 0.
    """
    if altitude_deg <= 0 or dni <= 0:
        return 0.0
    return float(dni * np.sin(np.radians(altitude_deg)))


# ---------------------------------------------------------------------------
# Diffuse horizontal irradiance
# ---------------------------------------------------------------------------


def diffuse_horizontal_irradiance(
    altitude_deg: float,
    day_of_year: int,
    site_elevation_m: float = 100.0,
) -> float:
    """Estimate clear-sky diffuse horizontal irradiance.

    Uses a simplified Reindl et al. (1990) clear-sky diffuse fraction.
    For clear-sky conditions the diffuse component is typically 10-15% of
    the global irradiance.  We estimate:

        I_d = I_ext * sin(alpha) * C_d

    where ``C_d`` is a diffuse coefficient that varies with air mass and
    atmospheric clarity.  For a tropical clear sky, C_d ~ 0.12.

    As an improvement over a flat coefficient, we use the relationship from
    Reindl et al. (1990) for the clear-sky diffuse fraction, simplified to:

        k_d = 0.147  (clear-sky average for tropical latitudes)

    so that I_d = k_d * GHI_clearsky, approximated as:

        I_d = k_d * I_ext * tau_b * sin(alpha) / (1 - k_d)

    This avoids circular dependency by using the beam transmittance directly.

    Parameters
    ----------
    altitude_deg : float
        Solar altitude in degrees.
    day_of_year : int
        Day of the year (1-365).
    site_elevation_m : float
        Site elevation in metres.

    Returns
    -------
    float
        Diffuse horizontal irradiance in W/m^2.
    """
    if altitude_deg <= 0:
        return 0.0

    # Get beam irradiance on horizontal surface
    dni = direct_normal_irradiance(altitude_deg, day_of_year, site_elevation_m)
    I_bh = direct_horizontal_irradiance(dni, altitude_deg)

    # Clear-sky diffuse fraction (Reindl et al., 1990)
    # For clear skies (kt > 0.6), k_d is typically 0.12–0.17
    # We use a slightly altitude-dependent coefficient:
    # Higher sun = lower diffuse fraction (clearer path)
    sin_alt = np.sin(np.radians(altitude_deg))
    k_d = 0.165 - 0.025 * sin_alt  # ranges ~0.14 (zenith) to ~0.165 (horizon)

    # Diffuse = k_d / (1 - k_d) * beam_horizontal
    # This comes from: GHI = I_bh + I_d, and k_d = I_d / GHI
    # => I_d = k_d * GHI = k_d * (I_bh + I_d) => I_d = k_d / (1 - k_d) * I_bh
    if I_bh <= 0:
        # Very low sun: use a minimum diffuse from atmospheric scattering
        I_ext = _extraterrestrial_irradiance(day_of_year)
        return float(max(0.0, 0.06 * I_ext * sin_alt))

    I_d = k_d / (1.0 - k_d) * I_bh
    return float(max(0.0, I_d))


# ---------------------------------------------------------------------------
# Instantaneous GHI (single point, single timestep)
# ---------------------------------------------------------------------------


def compute_instantaneous_ghi(
    altitude_deg: float,
    day_of_year: int,
    svf: float = 1.0,
    is_sunlit: bool = True,
    site_elevation_m: float = 100.0,
) -> float:
    """Global Horizontal Irradiance for one timestep at one point.

    Combines direct beam (if sunlit) and diffuse (weighted by SVF):

        GHI = (DNI * sin(alt)  if sunlit  else 0) + I_diffuse * SVF

    The SVF weighting on diffuse radiation follows the isotropic diffuse
    model of Liu & Jordan (1960): the fraction of the sky dome visible
    from the point scales the diffuse component linearly.

    Parameters
    ----------
    altitude_deg : float
        Solar altitude in degrees.
    day_of_year : int
        Day of year (1-365).
    svf : float
        Sky View Factor at the point (0-1).  Default 1.0 (unobstructed).
    is_sunlit : bool
        Whether the point receives direct beam at this timestep.
    site_elevation_m : float
        Site elevation in metres.

    Returns
    -------
    float
        Instantaneous GHI in W/m^2.
    """
    if altitude_deg <= 0:
        return 0.0

    # Direct component
    if is_sunlit:
        dni = direct_normal_irradiance(altitude_deg, day_of_year, site_elevation_m)
        I_direct = direct_horizontal_irradiance(dni, altitude_deg)
    else:
        I_direct = 0.0

    # Diffuse component (isotropic, weighted by SVF)
    I_diffuse = diffuse_horizontal_irradiance(
        altitude_deg, day_of_year, site_elevation_m
    )
    I_diffuse_weighted = I_diffuse * svf

    return float(I_direct + I_diffuse_weighted)


# ---------------------------------------------------------------------------
# Daily integration (single point)
# ---------------------------------------------------------------------------


def integrate_daily_irradiance(
    sun_positions: list[tuple[float, float]],
    sunlit_flags: np.ndarray,
    svf: float,
    interval_minutes: int,
    day_of_year: int,
    site_elevation_m: float = 100.0,
) -> float:
    """Integrate GHI over one day for a single point.

    Uses the trapezoidal-like approximation: each timestep represents
    ``interval_minutes`` of constant irradiance.

    Parameters
    ----------
    sun_positions : list[tuple[float, float]]
        List of ``(altitude_deg, azimuth_deg)`` for each timestep.
    sunlit_flags : np.ndarray
        Boolean array of length ``len(sun_positions)``; True if the point
        is sunlit at that timestep.
    svf : float
        Sky View Factor at the point (0-1).
    interval_minutes : int
        Time step between sun positions (minutes).
    day_of_year : int
        Day of year (1-365).
    site_elevation_m : float
        Site elevation in metres.

    Returns
    -------
    float
        Daily irradiation in Wh/m^2/day.
    """
    total_wh = 0.0
    hours_per_step = interval_minutes / 60.0

    for i, (alt, _az) in enumerate(sun_positions):
        ghi = compute_instantaneous_ghi(
            altitude_deg=alt,
            day_of_year=day_of_year,
            svf=svf,
            is_sunlit=bool(sunlit_flags[i]),
            site_elevation_m=site_elevation_m,
        )
        total_wh += ghi * hours_per_step

    return float(total_wh)


# ---------------------------------------------------------------------------
# Vectorized daily irradiance for N points
# ---------------------------------------------------------------------------


def compute_daily_irradiance_array(
    sun_positions: list[tuple[float, float]],
    sunlit_matrix: np.ndarray,
    svf_array: np.ndarray,
    interval_minutes: int,
    day_of_year: int,
    site_elevation_m: float = 100.0,
) -> np.ndarray:
    """Vectorized daily irradiation for N points.

    Pre-computes the irradiance components per timestep, then broadcasts
    across all points for efficiency.

    Parameters
    ----------
    sun_positions : list[tuple[float, float]]
        List of ``(altitude_deg, azimuth_deg)`` for M timesteps.
    sunlit_matrix : np.ndarray
        Boolean array of shape ``(N, M)``; True if point i is sunlit at
        timestep j.
    svf_array : np.ndarray
        SVF values for N points, shape ``(N,)``.
    interval_minutes : int
        Time step between sun positions (minutes).
    day_of_year : int
        Day of year (1-365).
    site_elevation_m : float
        Site elevation in metres.

    Returns
    -------
    np.ndarray
        Daily irradiation for each point in Wh/m^2/day, shape ``(N,)``.
    """
    n_sun = len(sun_positions)
    hours_per_step = interval_minutes / 60.0

    # Pre-compute per-timestep irradiance components
    # direct_h[j] = DNI * sin(alt) for timestep j  (beam on horizontal)
    # diffuse_h[j] = diffuse horizontal irradiance for timestep j
    direct_h = np.zeros(n_sun, dtype=np.float64)
    diffuse_h = np.zeros(n_sun, dtype=np.float64)

    for j, (alt, _az) in enumerate(sun_positions):
        if alt <= 0:
            continue
        dni = direct_normal_irradiance(alt, day_of_year, site_elevation_m)
        direct_h[j] = direct_horizontal_irradiance(dni, alt)
        diffuse_h[j] = diffuse_horizontal_irradiance(alt, day_of_year, site_elevation_m)

    # Ensure shapes are correct
    sunlit = np.asarray(sunlit_matrix, dtype=np.float64)  # (N, M)
    svf = np.asarray(svf_array, dtype=np.float64)  # (N,)

    # Direct component: sum over timesteps where sunlit
    # (N, M) * (M,) -> (N, M), then sum over M -> (N,)
    direct_total = np.sum(sunlit * direct_h[np.newaxis, :], axis=1)  # (N,)

    # Diffuse component: SVF-weighted, summed over all above-horizon timesteps
    # SVF is constant per point; diffuse depends only on sun position
    diffuse_sum = np.sum(diffuse_h)  # scalar: total diffuse over all timesteps
    diffuse_total = svf * diffuse_sum  # (N,)

    # Convert to Wh/m^2
    irradiance_wh = (direct_total + diffuse_total) * hours_per_step

    return irradiance_wh
