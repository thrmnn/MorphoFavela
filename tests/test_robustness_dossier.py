"""ROBUST — reviewer robustness dossier: spec docs/robustness_dossier_spec.md.

Every test here reads only from the main checkout (this worktree carries no
data/, outputs/, or heavy run artifacts) and skips cleanly if that checkout
is absent — never passes vacuously against a wrong number, only against a
missing input (the same discipline as tests/test_wp07_ledger.py).

The one assertion that matters most: no dossier row's value differs from the
source it cites — re-resolved independently here, not merely re-printed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_robustness_dossier as brd  # noqa: E402
import lint_p1_tokens as lt  # noqa: E402

from src.brisa_solar.constants import P1_SKY_PATCHES  # noqa: E402
from src.brisa_solar.wp07_ledger import resolve_pointer  # noqa: E402

MAIN_ROOT = brd.DEFAULT_MAIN_ROOT

pytestmark = pytest.mark.skipif(
    not MAIN_ROOT.exists(), reason="main checkout absent (this worktree has no data/outputs/heavy runs)"
)


@pytest.fixture(scope="module")
def dossier():
    return brd.build_dossier(MAIN_ROOT)


def _all_rows(dossier):
    for sec_name, sec in dossier["sections"].items():
        for r in sec.get("rows", []):
            yield sec_name, r
    yield "engine_acceptance", dossier["sections"]["engine_acceptance"]["chosen_variant_quote"]


def _values_equal(expected, actual) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        return expected == pytest.approx(actual, rel=1e-12, abs=1e-12)
    return expected == actual


# --- (a) every row's value equals a fresh, independent re-resolution of its cited source ---

def test_every_row_matches_its_cited_source(dossier):
    file_cache: dict[str, dict | str] = {}

    def _read_json(file_str: str) -> dict:
        if file_str not in file_cache:
            path = brd._resolve_cited_path(MAIN_ROOT, file_str)
            file_cache[file_str] = json.loads(path.read_text())
        return file_cache[file_str]

    def _read_text(file_str: str) -> str:
        key = f"TEXT::{file_str}"
        if key not in file_cache:
            path = brd._resolve_cited_path(MAIN_ROOT, file_str)
            file_cache[key] = path.read_text()
        return file_cache[key]

    checked = 0
    for sec_name, r in _all_rows(dossier):
        src = r["source"]
        if "computed_from" in src:
            inputs = [resolve_pointer(_read_json(f["file"]), f["json_pointer"])
                      for f in src["computed_from"]]
            fn = brd.COMPUTE_FNS[src["fn"]]
            expected = fn(*inputs)
        elif "text_substring_of" in src:
            text = _read_text(src["text_substring_of"])
            assert r["value"] in text, (
                f"{sec_name}/{r['id']}: quoted text is no longer a substring of "
                f"{src['text_substring_of']}"
            )
            checked += 1
            continue
        else:
            doc = _read_json(src["file"])
            expected = resolve_pointer(doc, src["json_pointer"])

        assert _values_equal(expected, r["value"]), (
            f"{sec_name}/{r['id']}: source {src} = {expected!r} != dossier value {r['value']!r}"
        )
        checked += 1

    assert checked > 60, f"expected >60 rows checked across all sections, got {checked}"


# --- (b) row ids are unique within each section ---

def test_row_ids_unique_per_section(dossier):
    for sec_name, sec in dossier["sections"].items():
        ids = [r["id"] for r in sec.get("rows", [])]
        assert len(ids) == len(set(ids)), f"{sec_name}: duplicate row ids {ids}"


# --- (c) the L1 red line: no favela-vs-formal/non-favela/contrast field anywhere ---

def test_l1_favela_vs_formal_contrast_guard_is_clear(dossier):
    assert dossier["_meta"]["l1_contrast_check"]["clear"] is True, (
        f"possible favela-vs-formal (L1) field(s): {dossier['_meta']['l1_contrast_check']['flagged_ids']}"
    )


# --- (d) engine acceptance: the three crossref numbers + the three analytic self-checks exist ---

def test_engine_acceptance_has_crossref_numbers_and_self_checks(dossier):
    sec = dossier["sections"]["engine_acceptance"]
    ids = {r["id"] for r in sec["rows"]}
    assert {"engine.crossref.r", "engine.crossref.median_abs_delta",
            "engine.crossref.p95_abs_delta"} <= ids
    assert len(sec["analytic_self_checks"]) == 3
    for check in sec["analytic_self_checks"]:
        doc_path = MAIN_ROOT / check["defined_in"].split("#")[0]
        assert doc_path.exists(), f"{check['name']}: {doc_path} missing"
        test_file, test_fn = check["verified_by"].split("::")
        test_path = MAIN_ROOT / test_file
        assert test_path.exists(), f"{check['name']}: {test_path} missing"
        assert f"def {test_fn}(" in test_path.read_text(), (
            f"{check['name']}: {test_fn} not defined in {test_file}"
        )


# --- (e) G3: position moves, ordering does not — the honest framing, verbatim, with both numbers ---

def test_g3_says_position_moves_ordering_does_not_with_both_numbers(dossier):
    sec = dossier["sections"]["domain_sensitivity_g3"]
    narrative = sec["narrative"]
    assert "position" in narrative and "moves" in narrative
    assert "ordering" in narrative and "does not" in narrative

    row_by_id = {r["id"]: r["value"] for r in sec["rows"]}
    spread_min = row_by_id["g3.spread.min_across_favelas"]
    spread_max = row_by_id["g3.spread.max_across_favelas"]
    assert f"{spread_min:.3g}" in narrative
    assert f"{spread_max:.3g}" in narrative
    assert row_by_id["g3.rank_invariant_across_grid"] is True
    assert str(row_by_id["g3.rank_invariant_across_grid"]) in narrative


def test_g3_spread_and_rank_match_direct_recomputation_from_sensitivity_json(dossier):
    """Independent of the ledger's own tests: recompute straight from the g3_domain
    run file and check the dossier's derived numbers agree — never through the
    ledger's own derivation code twice."""
    from src.brisa_solar.wp07_ledger import FAVELAS, LOCKED_VARIANT, RUN_OF_RECORD

    g3_path = MAIN_ROOT / "runs" / RUN_OF_RECORD["g3"] / "sensitivity.json"
    g3 = json.loads(g3_path.read_text())
    variants = g3["variants"]

    sec = dossier["sections"]["domain_sensitivity_g3"]
    spread_rows = {r["id"]: r["value"] for r in sec["rows"] if r["id"].startswith("g3.spread.")
                   and r["id"] not in ("g3.spread.min_across_favelas", "g3.spread.max_across_favelas")}

    for slug, display in FAVELAS.items():
        vals = [v["study_favelas"][display]["svf_percentile_of_citywide_median"] for v in variants]
        expected_spread = max(vals) - min(vals)
        got = spread_rows[f"g3.spread.{slug}.svf_percentile"]
        assert got == pytest.approx(expected_spread, abs=1e-9), slug

    locked = next(v for v in variants
                  if (v["fabric_coverage_threshold"], v["fabric_footprint_distance_m"]) == LOCKED_VARIANT)
    expected_rank = sorted(FAVELAS, key=lambda s: locked["study_favelas"][FAVELAS[s]]
                            ["svf_percentile_of_citywide_median"], reverse=True)
    rank_row = next(r["value"] for r in sec["rows"] if r["id"] == "g3.rank_under_locked_domain")
    assert rank_row == expected_rank


# --- (f) sky resolution: imported, never typed; exactly one resolution across every manifest ---

def test_sky_resolution_is_imported_and_single_across_every_manifest(dossier):
    sec = dossier["sections"]["sky_resolution"]
    assert sec["p1_sky_patches"] == P1_SKY_PATCHES
    assert sec["single_resolution"] is True
    assert sec["distinct_patch_counts"] == [P1_SKY_PATCHES]
    assert sec["n_manifests_scanned"] > 0
    assert str(P1_SKY_PATCHES) in sec["narrative"]


# --- (g) irradiance: the percentage is computed here, never copied from _crosscheck ---

def test_irradiance_percentage_is_computed_not_copied(dossier):
    sec = dossier["sections"]["irradiance_input"]
    row_by_id = {r["id"]: r for r in sec["rows"]}
    pct_row = row_by_id["epw.santos_vs_galeao_ghi_pct_diff"]
    assert pct_row["source"].get("computed_from"), "percentage must be a computed row, not a direct copy"
    epw = json.loads((MAIN_ROOT / "data" / "epw" / "epw_inventory.json").read_text())
    galeao = epw["galeao"]["annual_ghi_kwh_m2"]
    santos = epw["santos_dumont"]["annual_ghi_kwh_m2"]
    expected = (santos - galeao) / galeao * 100.0
    assert pct_row["value"] == pytest.approx(expected, rel=1e-12)
    # and it must NOT equal a hand-copy of the file's own _crosscheck field by construction
    # (same numeric neighbourhood is fine; the row's source must never be that pointer)
    for f in pct_row["source"]["computed_from"]:
        assert "_crosscheck" not in f["json_pointer"]


# --- (h) second-axis validity: definition cited, wp06 shares present for all 5 sites ---

def test_second_axis_validity_cites_ventaxis_and_lists_all_sites(dossier):
    from src.brisa_solar.wp07_ledger import SITES

    sec = dossier["sections"]["second_axis_validity"]
    ids = {r["id"] for r in sec["rows"]}
    assert "wp06.definition_of_record_quote" in ids
    for slug in SITES:
        assert f"wp06.{slug}.n" in ids
        for k in "0123":
            assert f"wp06.{slug}.share_n{k}" in ids


# --- (i) G2: reported as a negative result with a confirmed confound; open card named, not resolved ---

def test_g2_is_negative_with_confound_and_names_the_open_card(dossier):
    sec = dossier["sections"]["ground_truth_g2"]
    assert sec["open_card"] == "g2_validity_floor"
    narrative = sec["narrative"].lower()
    assert "confound" in narrative
    assert "negative" in narrative
    assert "not resolve" in narrative or "does not resolve" in narrative
    row_by_id = {r["id"]: r["value"] for r in sec["rows"]}
    # the confound: shared-observer-elevation variants read higher r than the baseline
    assert row_by_id["g2.confound.shared_elevation_variant_f.lt1p5m.r"] > row_by_id["g2.confound.baseline_variant_a.lt1p5m.r"]
    assert row_by_id["g2.street.floor_class"] == "<1.5m"


# --- (j) coverage: citywide count/resolution + per-site n + a generated (never typed) epoch table ---

def test_coverage_epoch_table_is_generated_from_live_file_metadata(dossier):
    import os

    sec = dossier["sections"]["coverage"]
    assert len(sec["epoch_table"]) >= 3 * 6  # citywide + 5 sites x {boundary, footprints, dtm}
    for e in sec["epoch_table"]:
        p = MAIN_ROOT / e["path"]
        st = os.stat(p)
        assert e["size_bytes"] == st.st_size, e["path"]

    from src.brisa_solar.wp07_ledger import SITES
    row_ids = {r["id"] for r in sec["rows"]}
    for slug in SITES:
        assert f"coverage.site.{slug}.ground_n" in row_ids


# --- (k) declared limitations: canopy descope + façade NOT accepted + zero façade ledger entries ---

def test_declared_limitations_are_stated_and_verified_live(dossier):
    sec = dossier["sections"]["declared_limitations"]
    assert sec["n_dsm_files_on_disk"] == 0, "a DSM file exists on disk — the canopy-descope limitation is stale"
    assert sec["n_facade_entries_in_ledger"] == 0, "a façade number leaked into the ledger — façade was NOT accepted"
    row_by_id = {r["id"]: r["value"] for r in sec["rows"]}
    assert row_by_id["limitations.facade.riodaspedras_r"] < row_by_id["limitations.facade.floor_r"]
    assert row_by_id["limitations.facade.vidigal_r"] < row_by_id["limitations.facade.floor_r"]
    assert "canopy" in dossier["_meta"]["purpose"].lower() or "canopy" in sec["narrative"].lower()
    assert "not accepted" in sec["narrative"].lower() or "NOT accepted" in sec["narrative"]


# --- (l) never the 2 h floor's WHO misattribution; Athens Charter only, if it's mentioned at all ---

def test_never_says_who_for_the_2h_floor(dossier):
    text = json.dumps(dossier)
    assert "WHO" not in text


# --- (m) the ledger used is the newest wp07_ledger_*/ledger.json on disk (or the task-given one) ---

def test_ledger_used_is_the_newest_on_disk(dossier):
    import glob

    hits = sorted(glob.glob(str(MAIN_ROOT / "runs" / "wp07_ledger_*" / "ledger.json")))
    newest_rel = str(Path(hits[-1]).relative_to(MAIN_ROOT))
    assert dossier["_meta"]["ledger_used"]["path"] == newest_rel


# --- (n) write_dossier produces a tracked-shape run folder; its markdown carries no banned token ---

def test_write_dossier_writes_json_md_manifest_and_is_token_clean(tmp_path):
    run_dir = brd.write_dossier(brd.THIS_REPO_ROOT, MAIN_ROOT, tmp_path / "robustness_test")
    assert (run_dir / "dossier.json").exists()
    assert (run_dir / "dossier.md").exists()
    assert (run_dir / "manifest.json").exists()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["sky"]["patches"] == P1_SKY_PATCHES
    assert manifest["staged_only"] is True

    md = (run_dir / "dossier.md").read_text()
    hits = lt._scan_lines(md.split("\n"), "dossier.md")
    assert not hits, hits


# --- (o) the output globs this dossier writes to are already inside lint_p1_tokens.py's scan ---

def test_robustness_output_globs_are_covered_by_lint_p1_tokens():
    assert "runs/robustness_*/**/*.md" in lt.P1_SOURCE_GLOBS
    assert "runs/robustness_*/**/*.json" in lt.P1_SOURCE_GLOBS
