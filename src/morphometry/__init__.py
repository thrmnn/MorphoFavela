"""Morphometric analysis pipeline for informal settlement audit and reporting.

Provides grid-based computation of urban morphometric indicators, SVF audit,
publication-quality figure generation, and PDF report output.

Typical usage::

    python scripts/run_morphometric_audit.py --area vidigal
"""

from src.morphometry.audit import audit_svf
from src.morphometry.grid import compute_grid_morphometrics
from src.morphometry.indicators import (
    compute_lambda_f_directional,
    compute_porosity,
    compute_slope_aspect,
)

__all__ = [
    "compute_grid_morphometrics",
    "compute_porosity",
    "compute_slope_aspect",
    "compute_lambda_f_directional",
    "audit_svf",
]
