"""P1 (configuration C′) pipeline: citywide ground-level solar access.

A thin orchestration layer over engines that already exist and are already
validated at site scale — `src.solar` (pvlib ray-cast sun access) and
`src.svf_v2` (GPU/CPU Tregenza sky-view factor). This package adds the EPW
cumulative-sky weighting, the terrain/building decomposition, the citywide
stratified sample, and the run-manifest/acceptance machinery around them.

Boundaries that matter:
  * P1's ventilation axis is DESCRIPTIVE GEOMETRY ONLY. Nothing in this package
    may read a CFD-derived column (scripts/lint_p1_columns.py enforces it).
  * One sky resolution end-to-end: constants.P1_SKY_PATCHES.
  * Existing solar results are clear-sky and are not annual-insolation claims
    until recomputed or relabelled under an explicit sky_model.
"""
from __future__ import annotations

from .constants import P1_SKY_PATCHES, load_params

__all__ = ["P1_SKY_PATCHES", "load_params"]
