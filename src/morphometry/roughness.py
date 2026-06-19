"""Aerodynamic roughness (z0, zd) from morphometry — UMEP-driven (R-A).

Thin, tested wrapper around the vendored UMEP roughness calculator
(`vendor/umep_processing/util/RoughnessCalcFunctionV2.py`, Kent & Grimmond), which
implements Macdonald (1998), Kanda (2013), Raupach (1994), Millward-Hopkins (2011)
and rule-of-thumb. We do NOT re-derive the formulas — we wrap UMEP per cell and per
wind sector, add extrapolation flags for the favela regime, and (elsewhere) anchor
against CFD. Rationale + equations: docs/roughness_plan.md.

UMEP signature: RoughnessCalc(method, zH, fai, pai, zMax, zSdev) -> (zd, z0), with
zH=mean height, fai=frontal-area index λf (directional), pai=plan-area index λp,
zMax=max height, zSdev=σH.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "umep_processing" / "util"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
from RoughnessCalcFunctionV2 import RoughnessCalc  # noqa: E402

DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
METHODS = ("Kan", "Mho", "Mac", "Rau")  # Kanda primary; Macdonald = σH-free baseline
PAI_ENVELOPE_MAX = 0.5  # all methods calibrated below ~0.5


def roughness(method: str, zH, fai, pai, zMax, zSdev) -> tuple[float, float]:
    """(zd, z0) for one cell/sector; NaN if inputs are unphysical/missing."""
    vals = np.array([zH, fai, pai, zMax, zSdev], dtype=float)
    if not np.isfinite(vals).all() or zH <= 0 or zMax <= 0 or fai < 0 or pai <= 0:
        return (np.nan, np.nan)
    zd, z0 = RoughnessCalc(method, float(zH), float(fai), float(pai),
                           float(zMax), float(zSdev))
    return (float(zd), float(z0))


def roughness_vec(method, zH, fai, pai, zMax, zSdev):
    """Vectorized roughness over arrays → (zd_array, z0_array), NaN-guarded."""
    zH, fai, pai, zMax, zSdev = (np.asarray(a, float)
                                 for a in (zH, fai, pai, zMax, zSdev))
    zd = np.full(zH.shape, np.nan)
    z0 = np.full(zH.shape, np.nan)
    ok = (np.isfinite(zH) & np.isfinite(fai) & np.isfinite(pai)
          & np.isfinite(zMax) & np.isfinite(zSdev)
          & (zH > 0) & (zMax > 0) & (fai >= 0) & (pai > 0))
    for i in np.flatnonzero(ok):
        d, z = RoughnessCalc(method, float(zH[i]), float(fai[i]), float(pai[i]),
                             float(zMax[i]), float(zSdev[i]))
        zd[i], z0[i] = float(d), float(z)
    return zd, z0


def extrapolation_flags(zH, pai, zMax, zSdev) -> dict:
    """Per-cell flags marking estimates outside the methods' calibration envelope."""
    zH, pai, zMax, zSdev = (np.asarray(a, float) for a in (zH, pai, zMax, zSdev))
    X = np.where(zMax > 0, (zSdev + zH) / zMax, np.nan)  # Kanda predictor, capped ≤1
    return {
        "flag_pai_over_envelope": pai > PAI_ENVELOPE_MAX,
        "flag_kanda_X_saturated": np.isfinite(X) & (X >= 0.98),
        "kanda_X": X,
    }


def method_spread_z0(zH, fai, pai, zMax, zSdev, methods=METHODS):
    """z0 by each method (dict) + the max-min spread = morphometric uncertainty."""
    import warnings

    z0s = {m: roughness_vec(m, zH, fai, pai, zMax, zSdev)[1] for m in methods}
    stack = np.vstack([z0s[m] for m in methods])
    with warnings.catch_warnings():  # all-NaN columns (unbuilt cells) → NaN, no noise
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
    return z0s, spread
