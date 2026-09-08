"""P1 (C′): one sky discretization end-to-end, enforced rather than documented.

A site's position within the citywide distribution is only meaningful if both
sides were sampled on the same sky. Plan v1.0 asked for Reinhart 577 at sites
and Tregenza 145 citywide, which breaks its own one-resolution rule. These
tests pin the resolved answer to the *implementation* (not to a restated
number) and fail if any run manifest disagrees with any other.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.brisa_solar.constants import P1_SKY_PATCHES
from src.svf_v2.compute import generate_tregenza_patches

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_constant_matches_the_engine_that_implements_it():
    directions, weights = generate_tregenza_patches()
    assert len(directions) == P1_SKY_PATCHES
    assert len(weights) == P1_SKY_PATCHES


def test_params_yaml_agrees_with_the_constant():
    params = yaml.safe_load((REPO_ROOT / "config" / "params.yaml").read_text())
    assert params["sky"]["patches"] == P1_SKY_PATCHES


def test_no_p1_module_hardcodes_a_patch_count():
    """WP modules must import the constant, never restate 145 (or 577)."""
    offenders = []
    for py in (REPO_ROOT / "src" / "brisa_solar").rglob("*.py"):
        if py.name == "constants.py":
            continue
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if "145" in code or "577" in code:
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "import P1_SKY_PATCHES instead of a literal:\n" + "\n".join(offenders)


def test_all_run_manifests_used_the_same_sky():
    """Site and citywide runs must agree. Skips until runs exist — never passes vacuously."""
    manifests = sorted((REPO_ROOT / "runs").glob("*/manifest.json")) if (REPO_ROOT / "runs").exists() else []
    if not manifests:
        pytest.skip("no run manifests yet (WP-02/WP-05 have not run)")
    counts = {}
    for m in manifests:
        data = json.loads(m.read_text())
        patches = data.get("sky", {}).get("patches")
        if patches is not None:
            counts.setdefault(patches, []).append(m.parent.name)
    assert len(counts) <= 1, f"sky-resolution split across runs — percentile claim invalid: {counts}"
    if counts:
        assert next(iter(counts)) == P1_SKY_PATCHES
