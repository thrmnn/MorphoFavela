"""Tests for scripts/build_results_registry.py (charter: organization_charter.md
§2). Every fixture is a throwaway tree under tmp_path — this generator's own
module-level path constants are monkeypatched so these tests never touch the
real ~150-run-directory repo or the sibling brisaverse checkout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_results_registry as brr  # noqa: E402


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "MorphoFavela"
    brisa = tmp_path / "brisaverse"
    runs = root / "runs"
    outputs = root / "outputs"
    config = root / "config"

    _write(config / "sites.yaml", yaml.dump({
        "sites": {"vidigal": {"display_name": "Vidigal"}, "rocinha": {"display_name": "Rocinha"}},
    }))
    _write_json(brisa / "shared" / "facts" / "tasks.json", {"open_decisions": [], "resolved_decisions": []})

    monkeypatch.setattr(brr, "ROOT", root)
    monkeypatch.setattr(brr, "DATA_ROOT", root)
    monkeypatch.setattr(brr, "RUNS", runs)
    monkeypatch.setattr(brr, "OUTPUTS", outputs)
    monkeypatch.setattr(brr, "CONFIG", config)
    monkeypatch.setattr(brr, "REGISTRY_OUT", outputs / "_registry")
    monkeypatch.setattr(brr, "BASELINE_PATH", root / "registry" / "baseline.json")
    monkeypatch.setattr(brr, "WP_YAML", config / "work_packages.yaml")
    monkeypatch.setattr(brr, "BRISA", brisa)
    monkeypatch.setattr(brr, "TASKS_JSON", brisa / "shared" / "facts" / "tasks.json")

    return {"root": root, "runs": runs, "outputs": outputs, "config": config}


def _wp_yaml(families: dict) -> dict:
    return {"work_packages": {"WP07": {"title": "P1 solar results", "papers": ["p1"], "families": families}},
            "unassigned": {}}


def test_two_runs_sharing_a_slug_get_distinct_run_scoped_ids(env):
    """The round-1 prototype bug: `art:<family>::<slug>` (no run component)
    let a later-scanned run silently overwrite an earlier run's node with
    the same slug in the registry dict — a superseded run's figure could
    then read as CURRENT. This is the regression test the O2 task asked
    for: two runs of the same family sharing a slug must produce two
    coexisting nodes, with lifecycle correctly set on each, and exactly one
    `current` alias pointing at the newer run."""
    runs = env["runs"]
    config = env["config"]
    _write(config / "work_packages.yaml", yaml.dump(_wp_yaml({"wp07_figures": {"script": "src/x.py"}})))

    for run_id, status in [("wp07_figures_20260915T190806Z", "produced"), ("wp07_figures_20260917T125201Z", "produced")]:
        run_dir = runs / run_id
        _write(run_dir / "f1_citywide_position.png")
        _write_json(run_dir / "figure_manifest.json", {
            "_utc": brr._run_utc_from_name(run_id),
            "figures": {"f1_citywide_position": {"id": "f1_citywide_position", "status": status,
                                                   "png_path": "f1_citywide_position.png"}},
        })

    reg = brr.build()
    nodes = reg["nodes"]

    old_id = "art:wp07_figures::wp07_figures_20260915T190806Z::f1_citywide_position"
    new_id = "art:wp07_figures::wp07_figures_20260917T125201Z::f1_citywide_position"
    assert old_id in nodes, "the older run's artifact must not be overwritten"
    assert new_id in nodes, "the newer run's artifact must exist"
    assert nodes[old_id]["path"] != None and nodes[new_id]["path"] != None

    assert nodes[old_id]["lifecycle"] == "superseded"
    assert nodes[new_id]["lifecycle"] == "current"

    alias_id = "art:wp07_figures::current::f1_citywide_position"
    assert alias_id in nodes
    assert nodes[alias_id]["kind"] == "alias"
    assert nodes[alias_id]["target"] == new_id, "the current alias must point at the NEWER run, never the older one"

    run_node_old = nodes["run:wp07_figures_20260915T190806Z"]
    run_node_new = nodes["run:wp07_figures_20260917T125201Z"]
    assert run_node_old["lifecycle"] == "superseded"
    assert run_node_new["lifecycle"] == "current"
    assert run_node_old["superseded_by"] == "run:wp07_figures_20260917T125201Z"
    assert run_node_new["superseded_by"] is None

    assert nodes["fam:wp07_figures"]["head_run"] == "run:wp07_figures_20260917T125201Z"


def test_static_root_family_uses_two_part_id_no_run_axis(env):
    config = env["config"]
    outputs = env["outputs"]
    _write(config / "work_packages.yaml", yaml.dump(_wp_yaml({"p1_exports": {"static_root": "outputs/paper_figures/exports"}})))
    _write(outputs / "paper_figures" / "exports" / "fig01_composite.png")

    reg = brr.build()
    assert "art:p1_exports::fig01_composite" in reg["nodes"]
    assert reg["nodes"]["fam:p1_exports"]["static_root"] is True


def test_skipped_figure_is_draft_and_gets_no_current_alias(env):
    config = env["config"]
    runs = env["runs"]
    _write(config / "work_packages.yaml", yaml.dump(_wp_yaml({"wp07_figures": {"script": "src/x.py"}})))
    run_dir = runs / "wp07_figures_20260917T125201Z"
    _write_json(run_dir / "figure_manifest.json", {
        "_utc": "2026-09-17T12:52:02Z",
        "figures": {"f2_skipped": {"id": "f2_skipped", "status": "skipped", "reason": "no data"}},
    })

    reg = brr.build()
    node = reg["nodes"]["art:wp07_figures::wp07_figures_20260917T125201Z::f2_skipped"]
    assert node["lifecycle"] == "draft"
    assert "art:wp07_figures::current::f2_skipped" not in reg["nodes"]


def test_unclassified_excludes_only_hash_matched_hub_review_mirrors(env):
    """figure_organization_spec.md §1 exempts a HASH MATCH under
    outputs/_hub/**/outputs/_review/** as a copy of a registered original —
    it does not exempt the whole directory. A file there with unique
    content (no registered artifact shares its hash) is real,
    unregistered content and must surface as unclassified, never be
    silently dropped (organization_charter.md: 'a figure with no register
    row is shown as unclassified, never hidden'). Regression test for the
    O2 verifier finding: the round-1 fix blanket-excluded EXCLUDED_TOP_DIRS
    regardless of hash, hiding 9 real files on the live repo."""
    config = env["config"]
    outputs = env["outputs"]
    _write(config / "work_packages.yaml",
           yaml.dump(_wp_yaml({"p1_exports": {"static_root": "outputs/paper_figures/exports"}})))
    _write(outputs / "paper_figures" / "exports" / "fig01_composite.png", "registered content")

    # genuine copy of the registered artifact -> excluded (hash matches)
    _write(outputs / "_hub" / "mirror" / "fig01_composite.png", "registered content")
    # real, unique content under an otherwise-mirror dir -> NOT excluded
    _write(outputs / "_hub" / "docs" / "explainer.png", "unique explainer content")
    _write(outputs / "_review" / "2026-09-17" / "x.png", "unique review content")
    _write(outputs / "orphan_dir" / "mystery.png", "unique orphan content")

    reg = brr.build()
    assert reg["unclassified"]["count"] == 3
    assert reg["unclassified"]["by_folder"] == {
        "_hub/docs": 1, "_review/2026-09-17": 1, "orphan_dir": 1,
    }


def test_unclassified_collapses_byte_identical_copies_to_one_canonical_row(env):
    """O8 cleanup (figure_organization_spec.md §1/§6): 'Rows with the same
    content_hash become one canonical row, with the other paths in
    copies[].' A review-folder mirror that duplicates an already-unclassified
    original must not double the unclassified count — it collapses to one
    counted row, and the duplicate path is named in copies[], never
    dropped. The canonical path is always the real (non-_review/_hub)
    location when one exists, never the dated snapshot."""
    config = env["config"]
    outputs = env["outputs"]
    _write(config / "work_packages.yaml", yaml.dump(_wp_yaml({})))
    _write(outputs / "cidade_de_deus" / "svf_v2" / "svf_dashboard.png", "same bytes")
    _write(outputs / "_review" / "2026-09-24" / "sweep" / "svf_dashboard.png", "same bytes")
    _write(outputs / "_review" / "2026-09-17" / "sweep" / "svf_dashboard.png", "same bytes")

    reg = brr.build()
    assert reg["unclassified"]["count"] == 1
    assert reg["unclassified"]["duplicate_files_collapsed"] == 2
    canonical = "outputs/cidade_de_deus/svf_v2/svf_dashboard.png"
    assert canonical in reg["unclassified"]["copies"]
    assert set(reg["unclassified"]["copies"][canonical]) == {
        "outputs/_review/2026-09-24/sweep/svf_dashboard.png",
        "outputs/_review/2026-09-17/sweep/svf_dashboard.png",
    }


def test_archived_bytes_counted_separately_from_unclassified(env):
    """O8 cleanup: bytes moved to outputs/_archive/** (the cleanup pass —
    'archive, never delete') stop inflating the PI's 'needs review'
    unclassified count and show in their own 'archived' bucket instead;
    the row is never lost (organization_charter.md §4: 'Nothing leaves the
    PI's view. Archiving moves bytes, never rows')."""
    config = env["config"]
    outputs = env["outputs"]
    _write(config / "work_packages.yaml", yaml.dump(_wp_yaml({})))
    _write(outputs / "_archive" / "cross_site" / "old_map.png", "archived content")
    _write(outputs / "live_dir" / "new_map.png", "live content")

    reg = brr.build()
    assert reg["unclassified"]["count"] == 1
    assert reg["archived"]["count"] == 1
    assert "outputs/_archive/cross_site/old_map.png" in reg["archived"]["by_folder"] or \
        reg["archived"]["by_folder"] == {"_archive/cross_site": 1}
    assert reg["counts"]["archived"] == 1
    assert reg["counts"]["unclassified"] == 1


def test_orphan_run_family_reported(env):
    config = env["config"]
    runs = env["runs"]
    _write(config / "work_packages.yaml", yaml.dump(_wp_yaml({})))
    _write_json(runs / "ghost_family_20260101T000000Z" / "figure_manifest.json",
                {"_utc": "2026-01-01T00:00:00Z", "figures": {}})

    reg = brr.build()
    assert "ghost_family" in reg["orphan_run_families"]


def test_baseline_written_once(env):
    config = env["config"]
    _write(config / "work_packages.yaml", yaml.dump(_wp_yaml({})))
    import importlib
    argv = sys.argv
    sys.argv = ["build_results_registry.py"]
    try:
        assert brr.main() == 0
    finally:
        sys.argv = argv
    baseline = json.loads(brr.BASELINE_PATH.read_text())
    assert baseline["unclassified_count"] == 0
    # a second build must not overwrite an already-recorded baseline
    baseline_path = brr.BASELINE_PATH
    baseline_path.write_text(json.dumps({"unclassified_count": 42, "recorded_utc": "x", "note": "manual"}))
    sys.argv = ["build_results_registry.py"]
    try:
        brr.main()
    finally:
        sys.argv = argv
    assert json.loads(baseline_path.read_text())["unclassified_count"] == 42
