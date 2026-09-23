"""Invariants for the Maré neighbourhood layer (scripts/data_utils/build_mare_neighbourhoods.py)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data_utils"))

import build_mare_neighbourhoods as bmn  # noqa: E402


def test_crosswalk_names_sixteen_distinct_communities():
    names = [c for c, _, _, _ in bmn.CROSSWALK]
    assert len(names) == len(set(names)) == 16


def test_no_polygon_is_claimed_by_two_communities():
    parts = [p for _, ps, _, _ in bmn.CROSSWALK for p in ps]
    assert len(parts) == len(set(parts))


def test_every_inferred_match_says_why():
    for community, _, match, note in bmn.CROSSWALK:
        assert match in {"exact", "inferred"}, community
        if match == "inferred":
            assert note, f"{community}: an inferred match must carry its reason"


@pytest.mark.skipif(not bmn.OUT.exists(), reason="layer not built on this machine")
def test_built_layer_matches_the_crosswalk_and_does_not_overlap():
    import geopandas as gpd

    layer = gpd.read_file(bmn.OUT, layer="communities")
    assert sorted(layer["community"]) == sorted(c for c, _, _, _ in bmn.CROSSWALK)
    qa = json.loads(bmn.PROVENANCE.read_text())["qa"]
    assert qa["pairwise_overlaps_m2"] == []
    assert qa["n_communities"] == 16
