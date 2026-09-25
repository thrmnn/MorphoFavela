"""Tests for scripts/registry_join.py — the single badge/caption formatter
corrective step 3 requires every consumer to share
(docs/critic/incident_dashboard_loop_2026-09-25.md, charter phase D).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import registry_join as rj  # noqa: E402


# --------------------------------------------------------------- release_badge

def test_release_badge_staged_wins_over_release_class_text():
    assert rj.release_badge({"state": "staged", "release_class": "publishable"}) == "staged"


def test_release_badge_withheld_by_state():
    assert rj.release_badge({"state": "withheld"}) == "withheld"


def test_release_badge_withheld_by_release_class_text():
    assert rj.release_badge({"release_class": "PNG withheld pending review"}) == "withheld"


def test_release_badge_publishable():
    assert rj.release_badge({"state": "promoted", "release_class": "PNG + SVG `publishable`"}) == "publishable"


def test_release_badge_none_row_is_unclassified():
    assert rj.release_badge(None) == "unclassified"


def test_release_badge_no_match_is_unclassified():
    assert rj.release_badge({"state": "promoted", "release_class": "not in §5 registry"}) == "unclassified"


# ------------------------------------------------------------------- badge_text

def test_badge_text_current_lifecycle_is_bare_release_word():
    assert rj.badge_text({"release": "staged", "lifecycle": "current"}) == "staged"


def test_badge_text_non_current_lifecycle_gets_suffix():
    assert rj.badge_text({"release": "staged", "lifecycle": "draft"}) == "staged · draft"


def test_badge_text_missing_release_falls_back_to_unclassified():
    assert rj.badge_text({"lifecycle": "current"}) == "unclassified"


def test_badge_text_none_node_is_unclassified():
    assert rj.badge_text(None) == "unclassified"


# ---------------------------------------------------------------- caption_text

def test_caption_text_combines_family_site_wp():
    node = {"family": "wp07_figures", "site": "vidigal", "wp": "WP07"}
    assert rj.caption_text(node) == "wp07_figures · vidigal · WP07"


def test_caption_text_omits_missing_site():
    node = {"family": "wp07_figures", "wp": "WP07"}
    assert rj.caption_text(node) == "wp07_figures · WP07"


def test_caption_text_none_node_is_unclassified():
    assert rj.caption_text(None) == "unclassified"


# ------------------------------------------------------------------ node_by_path

def test_node_by_path_indexes_only_figure_nodes_with_a_path():
    registry = {"nodes": {
        "fam:x": {"kind": "family"},
        "art:x::a": {"kind": "figure", "path": "runs/r/a.png"},
        "art:x::b": {"kind": "figure", "path": None},
    }}
    idx = rj.node_by_path(registry)
    assert set(idx) == {"runs/r/a.png"}


# ------------------------------------------------------------------ join_p1_release

def test_join_p1_release_sets_release_guardian_paper_ref(tmp_path, monkeypatch):
    brisaverse_root = tmp_path / "brisaverse"
    (brisaverse_root / "shared" / "facts").mkdir(parents=True)
    import json
    (brisaverse_root / "shared" / "facts" / "p1_artifacts.json").write_text(json.dumps({
        "artifacts": [{
            "id": "f1_citywide_position", "state": "staged", "cited_in_outline": True,
            "run_of_record": "wp07_figures_X", "image_url": "/x/f1_citywide_position.png",
            "guardian": {"verdict": "CLEAR"},
        }],
    }))
    nodes = {
        "art:wp07_figures::wp07_figures_X::f1": {
            "kind": "figure", "parent": "run:wp07_figures_X",
            "path": "runs/wp07_figures_X/f1_citywide_position.png",
        },
    }
    n = rj.join_p1_release(nodes, tmp_path / "runs", brisaverse_root)
    assert n == 1
    node = nodes["art:wp07_figures::wp07_figures_X::f1"]
    assert node["release"] == "staged"
    assert node["guardian_verdict"] == "CLEAR"
    assert node["paper_ref"] == "f1_citywide_position"


def test_join_p1_release_no_match_leaves_release_none():
    nodes = {"art:x::y::z": {"kind": "figure", "parent": "run:y", "path": "runs/y/z.png"}}
    n = rj.join_p1_release(nodes, Path("/nonexistent/runs"), Path("/nonexistent/brisaverse"))
    assert n == 0
    assert nodes["art:x::y::z"].get("release") is None
    assert rj.badge_text(nodes["art:x::y::z"]) == "unclassified"


# --------------------------------------------------------------------------
# Self-proof: this suite can go red. A badge formatter that drops the
# lifecycle suffix (the exact class of regression corrective step 3 exists
# to prevent — a surface quietly reverting to its own private formatting)
# must fail this test, not pass it.
# --------------------------------------------------------------------------

def test_self_test_a_broken_badge_formatter_is_caught():
    def _broken_badge_text(node):
        return (node or {}).get("release") or "unclassified"  # drops lifecycle — the bug class

    node = {"release": "staged", "lifecycle": "superseded"}
    correct = rj.badge_text(node)
    broken = _broken_badge_text(node)
    assert correct != broken, "self-test failed to construct a distinguishing case"


# --------------------------------------------------------------------------
# Cross-repo mirror: brisaverse's shared/lib/registry_badge.py must produce
# byte-identical badge/caption text to this module's, for the same node —
# the two repos can't share a Python package, so this is what keeps the
# mirrored copy honest (corrective step 3: "a test in each repo").
# --------------------------------------------------------------------------

def test_badge_text_matches_brisaverse_mirror():
    brisaverse_lib = Path.home() / "SCL" / "SCR" / "brisaverse" / "shared" / "lib"
    if not (brisaverse_lib / "registry_badge.py").is_file():
        import pytest
        pytest.skip("brisaverse checkout not present in this environment")
    sys.path.insert(0, str(brisaverse_lib))
    import registry_badge as mirror  # noqa: E402
    cases = [
        {"release": "staged", "lifecycle": "current"},
        {"release": "withheld", "lifecycle": "draft"},
        {"release": None, "lifecycle": "current"},
        None,
    ]
    for node in cases:
        assert mirror.badge_text(node) == rj.badge_text(node)
        assert mirror.caption_text(node) == rj.caption_text(node)
