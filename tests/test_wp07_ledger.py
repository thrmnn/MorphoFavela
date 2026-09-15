"""WP-07A numbers ledger: spec docs/wp07_ledger_spec.md, deliverable 5's
seven tests (a)-(g). Reads only from the runs of record and skips cleanly
when a run folder is absent (never passes vacuously against a wrong number,
only against a missing input)."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.brisa_solar import wp07_ledger as w

RUNS = ROOT / "runs"


def _runs_present() -> bool:
    return all((RUNS / run_id).exists() for run_id in w.RUN_OF_RECORD.values())


pytestmark = pytest.mark.skipif(not _runs_present(), reason="a run of record is absent")


@pytest.fixture(scope="module")
def ledger():
    return w.build_ledger(ROOT)


def _walk_lists(obj, path=""):
    if isinstance(obj, list):
        yield path, obj
        for i, v in enumerate(obj):
            yield from _walk_lists(v, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_lists(v, f"{path}/{k}")


# --- (a) round-trip: every entry's value equals the source re-read at json_pointer ---

def test_round_trip_every_entry_against_its_own_source(ledger):
    cache: dict[str, dict] = {}
    for entry_id, entry in ledger["entries"].items():
        src = entry["source"]
        key = src["file"]
        if key not in cache:
            cache[key] = json.loads((ROOT / src["file"]).read_text())
        expected = w.resolve_pointer(cache[key], src["json_pointer"])
        assert expected == entry["value"], (
            f"{entry_id}: source {src['file']}#{src['json_pointer']} = {expected!r} "
            f"!= ledger value {entry['value']!r}"
        )


# --- (b) ids unique; every source file is one of the runs of record ---

def test_ids_unique_and_sources_are_runs_of_record(ledger):
    ids = list(ledger["entries"])
    assert len(ids) == len(set(ids)), "duplicate ledger ids"

    record_run_ids = set(w.RUN_OF_RECORD.values())
    for entry_id, entry in ledger["entries"].items():
        run_id = entry["source"]["run_id"]
        assert run_id in record_run_ids, f"{entry_id}: run_id {run_id} not a run of record"
        file_path = entry["source"]["file"]
        assert file_path.startswith(f"runs/{run_id}/"), (
            f"{entry_id}: source file {file_path} does not live under runs/{run_id}/"
        )
        assert (ROOT / file_path).exists(), f"{entry_id}: source file {file_path} missing"
        assert entry_id == entry["id"], f"dict key {entry_id} != entry['id'] {entry['id']}"


# --- (c) no list longer than 12 anywhere in the ledger document ---

def test_no_list_longer_than_12(ledger):
    offenders = [(path, len(lst)) for path, lst in _walk_lists(ledger) if len(lst) > 12]
    assert not offenders, f"list(s) longer than 12 elements: {offenders}"


# --- (d) any entry whose id contains non_favela/formal/contrast is reviewer-defence-only ---

def test_contrast_ids_are_reviewer_defence_only(ledger):
    # No favela-vs-non-favela/formal contrast exists in the current runs of
    # record (distribution.json carries no such field), so this loop is
    # expected to find nothing to flag today — it still guards the rule for
    # the day such a field is added and copied into the ledger.
    flagged_tokens = ("non_favela", "formal", "contrast")
    for entry_id, entry in ledger["entries"].items():
        if any(tok in entry_id for tok in flagged_tokens):
            assert entry["release_class"] == "reviewer-defence-only", (
                f"{entry_id}: favela-vs-non-favela/formal contrast must be reviewer-defence-only, "
                f"got {entry['release_class']}"
            )


# --- (e) derived spread equals a direct recomputation from sensitivity.json ---

def test_derived_spread_matches_direct_recomputation(ledger):
    g3_path = RUNS / w.RUN_OF_RECORD["g3"] / "sensitivity.json"
    g3 = json.loads(g3_path.read_text())
    variants = g3["variants"]
    assert len(variants) == 9

    for slug, display in w.FAVELAS.items():
        vals = [
            v["study_favelas"][display]["svf_percentile_of_citywide_median"]
            for v in variants
        ]
        expected_spread = max(vals) - min(vals)
        got = ledger["derived"]["spread"][slug]["svf_percentile_spread_max_minus_min"]
        assert got == pytest.approx(expected_spread, abs=1e-9), slug

    locked = next(
        v for v in variants
        if (v["fabric_coverage_threshold"], v["fabric_footprint_distance_m"]) == w.LOCKED_VARIANT
    )
    expected_rank = sorted(
        w.FAVELAS,
        key=lambda s: locked["study_favelas"][w.FAVELAS[s]]["svf_percentile_of_citywide_median"],
        reverse=True,
    )
    assert ledger["derived"]["rank_under_locked_domain"] == expected_rank

    expected_invariant = all(
        sorted(
            w.FAVELAS,
            key=lambda s: v["study_favelas"][w.FAVELAS[s]]["svf_percentile_of_citywide_median"],
            reverse=True,
        ) == expected_rank
        for v in variants
    )
    assert ledger["derived"]["rank_invariant_across_grid"] == expected_invariant


# --- (f) ledger.md carries no banned token ---

def test_ledger_markdown_has_no_banned_token(ledger, tmp_path):
    import lint_p1_tokens as lt
    importlib.reload(lt)

    md = w.render_markdown(ledger)
    hits = lt._scan_lines(md.split("\n"), "ledger.md")
    assert not hits, hits


# --- (g) five-favela SVF percentiles equal WP-05's; g3.* base-variant matches to 0.1 point ---

def test_favela_percentiles_match_wp05_and_g3_base_variant(ledger):
    wp05_path = RUNS / w.RUN_OF_RECORD["wp05"] / "distribution.json"
    wp05 = json.loads(wp05_path.read_text())

    g3_path = RUNS / w.RUN_OF_RECORD["g3"] / "sensitivity.json"
    g3 = json.loads(g3_path.read_text())
    locked = next(
        v for v in g3["variants"]
        if (v["fabric_coverage_threshold"], v["fabric_footprint_distance_m"]) == w.LOCKED_VARIANT
    )
    grid_slug = w._grid_slug(*w.LOCKED_VARIANT)

    for slug, display in w.FAVELAS.items():
        wp05_pct = wp05["study_favelas"][display]["svf"]["citywide_percentile_position"]
        ledger_pct = ledger["entries"][f"favela.{slug}.svf.percentile"]["value"]
        assert ledger_pct == pytest.approx(wp05_pct, abs=1e-9), slug

        g3_pct = locked["study_favelas"][display]["svf_percentile_of_citywide_median"]
        g3_ledger_pct = ledger["entries"][f"g3.grid_{grid_slug}.{slug}.svf_percentile"]["value"]
        assert g3_ledger_pct == pytest.approx(g3_pct, abs=1e-9), slug
        assert g3_ledger_pct == pytest.approx(wp05_pct, abs=0.1), slug


# --- write_ledger produces a tracked-shape run folder ---

def test_write_ledger_writes_json_md_manifest(tmp_path):
    run_dir = w.write_ledger(ROOT, tmp_path / "wp07_ledger_test")
    assert (run_dir / "ledger.json").exists()
    assert (run_dir / "ledger.md").exists()
    assert (run_dir / "manifest.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["sky"]["patches"] == w.P1_SKY_PATCHES
