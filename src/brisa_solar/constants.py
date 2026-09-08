"""Constants shared by every P1 (C′) work package.

The sky-patch count is the one number that MUST be identical on both sides of
the citywide-percentile claim: a site's percentile position is meaningless if
its SVF and irradiation were sampled on a different sky discretization than the
citywide distribution it is being ranked against. Plan v1.0 specified Reinhart
577 for sites and Tregenza 145 citywide, which violates its own one-resolution
rule; 145 wins because it is what src/svf_v2 actually implements and what
docs/GPU_SVF_EXACT_VALIDATION.md validated (577 was never implemented).

Import this — never write a patch count as a literal in a WP module.
"""
from __future__ import annotations

from pathlib import Path

import yaml

#: Tregenza sky-patch count for every P1 number feeding the citywide percentile.
P1_SKY_PATCHES = 145

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "config" / "params.yaml"


def load_params() -> dict:
    """Return config/params.yaml. WP entry points hash their section into the run manifest."""
    with PARAMS_PATH.open() as fh:
        return yaml.safe_load(fh)
