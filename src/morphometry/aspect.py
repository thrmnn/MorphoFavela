"""Terrain-aspect helpers.

Aspect is a circular quantity: the bearing (0-360 deg from north,
clockwise) of the steepest downslope direction. Linear OLS treats
0 deg and 359 deg as opposite, which is wrong, so consumers either
(a) encode it as a (sin, cos) pair, or (b) bin it into compass
quadrants. Helpers for both, plus the cosine alignment with a wind
direction used as a CFD covariate.

Conventions
-----------
- ``aspect_deg``    bearing of downslope direction, 0=N, 90=E, 180=S, 270=W.
- ``wind_dir_deg``  meteorological "wind FROM" bearing, same N-clockwise.
  ``WIND_DIRECTIONS_8`` in ``src.cfd_integration.schema`` encodes the
  same convention (N=0, E=90, ...).
- A south-facing slope (aspect=180) hit by a southerly wind
  (wind FROM south = 180) is windward — they "point at each other"
  in the geometric sense — so we define alignment so that this case
  returns +1.

NaN handling: any NaN in input propagates as NaN; helpers never
silently substitute zeros (which would bias regressions).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

QUADRANT_EDGES = (45.0, 135.0, 225.0, 315.0)
QUADRANT_LABELS = ("N", "E", "S", "W")


def aspect_to_sincos(aspect_deg: np.ndarray | Iterable[float] | float) -> tuple[np.ndarray, np.ndarray]:
    """Encode aspect as orthogonal (sin, cos) covariates for OLS.

    Returns two arrays the same shape as the input. NaN-in -> NaN-out.
    """
    a = np.asarray(aspect_deg, dtype=float)
    rad = np.deg2rad(a)
    return np.sin(rad), np.cos(rad)


def aspect_wind_alignment(
    aspect_deg: np.ndarray | Iterable[float] | float,
    wind_dir_deg: float,
) -> np.ndarray:
    """Cosine alignment between terrain aspect and a wind direction.

    Both arguments use the meteorological convention (bearing FROM
    north, clockwise). +1 means the slope faces directly into the
    wind (windward), -1 means it points directly away (lee), 0 is
    perpendicular. This is the natural covariate for "does facing
    the wind matter?" in a CFD regression.
    """
    a = np.asarray(aspect_deg, dtype=float)
    return np.cos(np.deg2rad(a - wind_dir_deg))


def aspect_quadrant(aspect_deg: np.ndarray | Iterable[float] | float) -> np.ndarray:
    """Bin aspect into the four compass quadrants centred on N/E/S/W.

    Bin centres are the cardinal directions; edges fall on the
    inter-cardinal directions (45, 135, 225, 315). NaN-in -> NaN-out
    (returned as the empty string '' so the caller can mask without
    a separate boolean array).
    """
    a = np.asarray(aspect_deg, dtype=float)
    out = np.empty(a.shape, dtype=object)
    nan_mask = np.isnan(a)
    a_mod = np.where(nan_mask, 0.0, a) % 360.0
    # N quadrant wraps the 0/360 boundary
    n_mask = (a_mod >= QUADRANT_EDGES[3]) | (a_mod < QUADRANT_EDGES[0])
    e_mask = (a_mod >= QUADRANT_EDGES[0]) & (a_mod < QUADRANT_EDGES[1])
    s_mask = (a_mod >= QUADRANT_EDGES[1]) & (a_mod < QUADRANT_EDGES[2])
    w_mask = (a_mod >= QUADRANT_EDGES[2]) & (a_mod < QUADRANT_EDGES[3])
    out[n_mask] = "N"
    out[e_mask] = "E"
    out[s_mask] = "S"
    out[w_mask] = "W"
    out[nan_mask] = ""
    return out


def dominant_wind_direction_deg(wind_rose) -> float | None:
    """Frequency-weighted mean wind direction (deg from N, clockwise).

    Uses the per-bin frequencies on a ``WindRose`` (see
    ``src.cfd_integration.schema``). Direction labels are looked up
    in ``WIND_DIRECTIONS_8``. Computed as the angle of the mean
    unit vector — the right thing for a circular quantity.

    Returns ``None`` if the rose has no positive-frequency bins.
    """
    from src.cfd_integration.schema import WIND_DIRECTIONS_8

    freqs = []
    bearings = []
    for label, bearing in WIND_DIRECTIONS_8.items():
        f = wind_rose.frequencies.get(label, 0.0)
        if f > 0:
            freqs.append(f)
            bearings.append(bearing)
    if not freqs:
        return None
    rad = np.deg2rad(bearings)
    sx = np.sum(np.array(freqs) * np.sin(rad))
    cx = np.sum(np.array(freqs) * np.cos(rad))
    return float(np.rad2deg(np.arctan2(sx, cx)) % 360.0)
