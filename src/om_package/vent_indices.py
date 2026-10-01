"""Geometry-derived ventilation PROXIES per OM2 point, for a given wind
direction. Every column is a proxy built from 2019 building geometry; none is
measured or simulated air temperature or airflow. The geometry epoch is a
parameter set (``points`` frame, ``buffer_m``, horizon arrays), so swapping
in 2024 ALS/footprints means passing the new inputs, not editing this module.

Wind direction is meteorological: degrees clockwise from north, the bearing
the wind blows FROM. The upwind azimuth is therefore the wind direction.

Proxies (column suffix ``_proxy``):
  windward_lambda_f_proxy      frontal-area density facing the wind, circular
                               linear interpolation of the 8 lambda_f_<dir>
                               columns (the 10 m grid cell nearest the point).
  canyon_alignment_deg_proxy   angle between the street axis and the wind,
                               folded to 0-90 (0 = along the street,
                               channelling; 90 = across).
  upwind_shelter_deg_proxy     horizon angle at the upwind azimuth from the
                               point horizon profiles (shade.py
                               point_horizon_profiles): the obstruction the
                               wind meets.
  z0_m_proxy, zd_m_proxy       Macdonald et al. (1998) roughness length and
                               displacement height (below).
  open_space_fraction_proxy    1 - lambda_p in the point buffer.

Macdonald, R.W., Griffiths, R.F., Hall, D.J. (1998), "An improved method for
the estimation of surface roughness of obstacle arrays", Atmospheric
Environment 32(11), 1857-1864. With H the mean building height, lambda_p the
plan density, lambda_f the frontal-area density, kappa = 0.4 (von Karman):

    zd / H = 1 + A**(-lambda_p) * (lambda_p - 1)
    z0 / H = (1 - zd/H) * exp( -[ 0.5 * beta * (Cd / kappa**2)
                                  * (1 - zd/H) * lambda_f ] ** -0.5 )

with A = 4.43, beta = 1.0, Cd = 1.2 (the paper's staggered-array constants).
H, lambda_p come from the point buffer (buffer_m); lambda_f is the windward
value at the 10 m grid cell, so z0 and zd are wind-direction specific only
through lambda_f.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import DEFAULT_ROOT, Paths
from .ventilation import prevailing_wind_bearing_deg
from .wind_obs import wind_at

MACDONALD_A = 4.43
MACDONALD_BETA = 1.0
MACDONALD_CD = 1.2
VON_KARMAN = 0.4
DEFAULT_BUFFER_M = 50

_DIR_BEARINGS = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
LAMBDA_F_COLS = [f"lambda_f_{d}" for d in _DIR_BEARINGS]

INDEX_COLUMNS = {
    "windward_lambda_f_proxy": "windward frontal-area density, proxy",
    "canyon_alignment_deg_proxy": "street axis vs wind angle 0-90 deg, proxy",
    "upwind_shelter_deg_proxy": "horizon angle at the upwind azimuth, proxy",
    "z0_m_proxy": "Macdonald 1998 roughness length, proxy",
    "zd_m_proxy": "Macdonald 1998 displacement height, proxy",
    "open_space_fraction_proxy": "1 - plan density in the buffer, proxy",
}


def windward_lambda_f(lambda_f_by_dir: np.ndarray, wind_dir_deg) -> np.ndarray:
    """Circular linear interpolation of the 8 per-direction columns.
    lambda_f_by_dir is (n, 8) in N, NE, ... NW order; wind_dir_deg a scalar
    or length-n array."""
    lf = np.asarray(lambda_f_by_dir, float)
    w = np.broadcast_to(np.asarray(wind_dir_deg, float) % 360.0, (lf.shape[0],))
    pos = w / 45.0
    lo = np.floor(pos).astype(int) % 8
    hi = (lo + 1) % 8
    frac = pos - np.floor(pos)
    rows = np.arange(lf.shape[0])
    return lf[rows, lo] * (1 - frac) + lf[rows, hi] * frac


def canyon_alignment_deg(street_orientation_deg, wind_dir_deg) -> np.ndarray:
    """Angle between the street axis and the wind axis, folded to 0-90. A
    street is undirected (axis mod 180) and wind blows along or against it
    equally, so both are folded: 0 = along the street, 90 = across."""
    d = np.abs(np.asarray(street_orientation_deg, float) - np.asarray(wind_dir_deg, float)) % 180.0
    return np.minimum(d, 180.0 - d)


def upwind_shelter_deg(horizon_deg: np.ndarray, azimuths_deg: np.ndarray, wind_dir_deg) -> np.ndarray:
    """Horizon angle (n_points x n_az profile) at the azimuth nearest the
    upwind direction. Same nearest-azimuth rule as shade.is_shaded."""
    h = np.asarray(horizon_deg, float)
    az = np.asarray(azimuths_deg, float)
    w = np.broadcast_to(np.asarray(wind_dir_deg, float) % 360.0, (h.shape[0],))
    diff = np.abs(w[:, None] - az[None, :]) % 360.0
    diff = np.minimum(diff, 360.0 - diff)
    return h[np.arange(h.shape[0]), np.argmin(diff, axis=1)]


def macdonald_zd_z0(lambda_p, lambda_f, h_mean_m, a: float = MACDONALD_A, beta: float = MACDONALD_BETA,
                    cd: float = MACDONALD_CD, kappa: float = VON_KARMAN):
    """(zd, z0) in metres by Macdonald et al. (1998); see module docstring.
    NaN where H or lambda_p is NaN; z0 = 0 where lambda_f = 0."""
    lp = np.clip(np.asarray(lambda_p, float), 0.0, 1.0)
    lf = np.maximum(np.asarray(lambda_f, float), 0.0)
    h = np.asarray(h_mean_m, float)
    zd_h = 1.0 + a ** (-lp) * (lp - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = 0.5 * beta * (cd / kappa ** 2) * (1.0 - zd_h) * lf
        z0_h = np.where(inner > 0, (1.0 - zd_h) * np.exp(-(inner ** -0.5)), 0.0)
    z0_h = np.where(np.isnan(lf) | np.isnan(lp), np.nan, z0_h)
    return zd_h * h, z0_h * h


def prevailing_direction_deg(root=DEFAULT_ROOT) -> float:
    """Frequency-weighted circular mean of the 2015-2024 SBGL climatology
    (data/maré/wind_rose.json), as ventilation.py uses. A circular mean of a
    spread rose is a summary bearing, not necessarily a mode."""
    return prevailing_wind_bearing_deg(Paths(root).wind_rose_json)


def compute_indices(points: pd.DataFrame, wind_dir_deg: float, horizon_deg: np.ndarray | None = None,
                    azimuths_deg: np.ndarray | None = None, buffer_m: int = DEFAULT_BUFFER_M) -> pd.DataFrame:
    """All six proxies for every point at one wind direction. ``points`` is
    the package points table (lambda_f_<dir>, street_orientation_deg,
    lambda_p_buffer_<r>m, building_height_mean_buffer_<r>m). The shelter
    column is NaN if no horizon profile is supplied."""
    lf = windward_lambda_f(points[LAMBDA_F_COLS].to_numpy(), wind_dir_deg)
    lp = points[f"lambda_p_buffer_{buffer_m}m"].to_numpy(float)
    h = points[f"building_height_mean_buffer_{buffer_m}m"].to_numpy(float)
    zd, z0 = macdonald_zd_z0(lp, lf, h)
    out = pd.DataFrame({"point_id": points["point_id"].to_numpy()})
    out["wind_dir_deg"] = float(wind_dir_deg)
    out["windward_lambda_f_proxy"] = lf
    out["canyon_alignment_deg_proxy"] = canyon_alignment_deg(points["street_orientation_deg"].to_numpy(float), wind_dir_deg)
    if horizon_deg is not None:
        out["upwind_shelter_deg_proxy"] = upwind_shelter_deg(horizon_deg, azimuths_deg, wind_dir_deg)
    else:
        out["upwind_shelter_deg_proxy"] = np.nan
    out["z0_m_proxy"] = z0
    out["zd_m_proxy"] = zd
    out["open_space_fraction_proxy"] = 1.0 - lp
    return out


def indices_at_prevailing(points, horizon_deg=None, azimuths_deg=None, root=DEFAULT_ROOT,
                          buffer_m: int = DEFAULT_BUFFER_M) -> pd.DataFrame:
    return compute_indices(points, prevailing_direction_deg(root), horizon_deg, azimuths_deg, buffer_m)


def indices_at_observed(points, timestamp_utc, horizon_deg=None, azimuths_deg=None, obs=None,
                        root=DEFAULT_ROOT, buffer_m: int = DEFAULT_BUFFER_M) -> pd.DataFrame | None:
    """Proxies at the SBGL direction nearest timestamp_utc (see
    wind_obs.wind_at); None when no usable observation is within 60 min."""
    w = wind_at(timestamp_utc, obs=obs, root=root)
    if w is None:
        return None
    out = compute_indices(points, w["drct"], horizon_deg, azimuths_deg, buffer_m)
    out["sbgl_valid_utc"] = w["valid_utc"]
    out["sbgl_speed_ms"] = w["speed_ms"]
    return out
