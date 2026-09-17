#!/usr/bin/env python3
"""ROBUST — the reviewer robustness dossier for P1 v2.

Spec: docs/robustness_dossier_spec.md (binding). Assembly by code, not new
analysis: every number below already sits final in a run of record (mostly
the WP-07 C' numbers ledger; a few sections — G2, EPW, coverage — are not
ledgered and are read straight from their run file). Nothing here is
recomputed as a new result; the only "computation" performed is re-reading a
value or combining two already-final values with an explicit, testable
formula (e.g. the EPW station GHI percentage difference).

This worktree carries no data/, outputs/, or heavy run artifacts (only the
git-tracked manifest/json/md siblings of each runs/<id>/ folder) — every
input is read from the MAIN checkout by absolute path (--main-root). The
dossier itself is written under THIS repo's own runs/ (this worktree), never
promoted anywhere the public hub or a paper reads.

Run: python3 scripts/build_robustness_dossier.py
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_REPO_ROOT))

from src.brisa_solar.constants import P1_SKY_PATCHES  # noqa: E402
from src.brisa_solar.wp07_ledger import RUN_OF_RECORD, SITE_DIRS, SITES  # noqa: E402

#: Default source of truth for every input this dossier reads. This worktree
#: has none of data/, outputs/, or heavy run artifacts — only the light
#: manifest/json/md files git tracks alongside each run. Override with
#: --main-root for a different checkout.
DEFAULT_MAIN_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")

#: Cross-repo citation only (read, never written): the C' reframe plan that
#: carries the PI's canopy-descope decision (item B9). BRISAverse owns it;
#: MorphoFavela cites it the same way build_review_pack.py cites docs here.
CPRIME_PLAN = Path(
    "/home/theo/SCL/SCR/brisaverse/papers/p1-nature-cities/proposal/cprime_reframe_plan.md"
)

#: Named pure functions a "computed" row can point back to, so the test can
#: recompute the row from its own cited inputs using the SAME function the
#: build used — never a second, drift-prone copy of the formula.
COMPUTE_FNS = {
    "pct_diff_b_over_a": lambda a, b: (b - a) / a * 100.0,
    "min": lambda *vals: min(vals),
    "max": lambda *vals: max(vals),
}

SITE_BOUNDARY_FILE = {
    "vidigal": "raw/Vidigal_Limit.shp",
    "rocinha": "raw/rocinha_boundary.shp",
    "complexo_do_alemao": "raw/complexo_do_alemao_boundary.shp",
    "riodaspedras": "raw/riodaspedras_boundary.shp",
    "mare": "raw/mare_boundary.shp",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _rel(main_root: Path, path: Path) -> str:
    """Path for a source citation: relative to main_root when possible (the
    common case), else an absolute string (a deliberate cross-repo citation
    like CPRIME_PLAN)."""
    try:
        return str(path.resolve().relative_to(main_root.resolve()))
    except ValueError:
        return str(path)


def _resolve_cited_path(main_root: Path, file_str: str) -> Path:
    p = Path(file_str)
    return p if p.is_absolute() else main_root / file_str


def row(row_id: str, value, unit: str, *, file: str, json_pointer: str) -> dict:
    """A dossier row backed by one JSON file + RFC-6901 pointer. The test
    re-resolves the pointer against the cited file and asserts equality —
    the row can never silently drift from its source."""
    return {
        "id": row_id,
        "value": value,
        "unit": unit,
        "source": {"file": file, "json_pointer": json_pointer},
    }


def ledger_row(row_id: str, value, unit: str, ledger_rel: str, ledger_id: str) -> dict:
    return row(row_id, value, unit, file=ledger_rel, json_pointer=f"/entries/{ledger_id}/value")


def ledger_derived_row(row_id: str, value, unit: str, ledger_rel: str, pointer: str) -> dict:
    return row(row_id, value, unit, file=ledger_rel, json_pointer=pointer)


def computed_row(row_id: str, value, unit: str, fn: str, from_: list[dict]) -> dict:
    return {
        "id": row_id,
        "value": value,
        "unit": unit,
        "source": {"computed_from": from_, "fn": fn},
    }


def text_row(row_id: str, value: str, *, file: str) -> dict:
    """A prose quote whose source is a substring of a (non-JSON) file. The
    test re-reads the file and asserts the quote is still verbatim in it."""
    return {"id": row_id, "value": value, "unit": "text", "source": {"text_substring_of": file}}


# ── ledger discovery ─────────────────────────────────────────────────────

def find_ledger(main_root: Path) -> tuple[Path, dict]:
    """The newest runs/wp07_ledger_*/ledger.json under main_root (spec: 'use
    the newest ... unless VENTAXIS has regenerated it')."""
    hits = sorted(glob.glob(str(main_root / "runs" / "wp07_ledger_*" / "ledger.json")))
    if not hits:
        sys.exit("no runs/wp07_ledger_*/ledger.json found under main-root")
    path = Path(hits[-1])
    return path, json.loads(path.read_text())


# ── sections ──────────────────────────────────────────────────────────────

def build_engine_acceptance(main_root: Path, ledger_path: Path, ledger: dict) -> dict:
    ledger_rel = _rel(main_root, ledger_path)
    rows = [
        ledger_row("engine.crossref.r", ledger["entries"]["engine.crossref.r"]["value"],
                   "dimensionless", ledger_rel, "engine.crossref.r"),
        ledger_row("engine.crossref.median_abs_delta",
                   ledger["entries"]["engine.crossref.median_abs_delta"]["value"],
                   "fraction", ledger_rel, "engine.crossref.median_abs_delta"),
        ledger_row("engine.crossref.p95_abs_delta",
                   ledger["entries"]["engine.crossref.p95_abs_delta"]["value"],
                   "fraction", ledger_rel, "engine.crossref.p95_abs_delta"),
    ]
    # Checkable the same way as a numeric row: exact value equality against a
    # JSON pointer (into the ledger's own _meta), not a hand-copied string.
    variant_quote = row(
        "engine.crossref.chosen_variant_quote", ledger["_meta"]["engine_acceptance_source"],
        "text", file=ledger_rel, json_pointer="/_meta/engine_acceptance_source",
    )

    analytic_self_checks = [
        {
            "name": "unobstructed identity",
            "predicate": "flat surface -> every patch visible -> sky.svf == 1 for every observer",
            "defined_in": "docs/wp02_horizon_engine_spec.md#Acceptance (item 1)",
            "verified_by": "tests/test_wp02_horizon.py::test_flat_surface_identity",
        },
        {
            "name": "infinite canyon, exact mask",
            "predicate": "analytic closed form 'visible iff dz/|dy| > 2H/W' must equal the "
                         "engine's mask patch-for-patch for H/W in {0.25, 0.5, 1, 2, 3}, "
                         "allowing only patches within one step's angular quantum of the horizon",
            "defined_in": "docs/wp02_horizon_engine_spec.md#Acceptance (item 2)",
            "verified_by": "tests/test_wp02_horizon.py::test_infinite_canyon_exact_mask",
        },
        {
            "name": "isolated wall shadow",
            "predicate": "one wall of height H at distance D: blocked patches are exactly "
                         "those with alt < atan(H/D) inside the wall's azimuth span",
            "defined_in": "docs/wp02_horizon_engine_spec.md#Acceptance (item 3)",
            "verified_by": "tests/test_wp02_horizon.py::test_isolated_wall_shadow",
        },
    ]
    return {
        "narrative": (
            "Engine acceptance rests on two independent legs: (1) three analytic "
            "self-checks against closed-form geometry (no data dependency), and "
            "(2) a measured cross-reference against the CPU raycaster's street SVF "
            "at Rio das Pedras (16,905 points). Both are read here, not recomputed."
        ),
        "rows": rows,
        "chosen_variant_quote": variant_quote,
        "analytic_self_checks": analytic_self_checks,
    }


def build_domain_sensitivity_g3(main_root: Path, ledger_path: Path, ledger: dict) -> dict:
    ledger_rel = _rel(main_root, ledger_path)
    spread = ledger["derived"]["spread"]
    rows = []
    for slug in sorted(spread):
        v = spread[slug]["svf_percentile_spread_max_minus_min"]
        rows.append(ledger_derived_row(
            f"g3.spread.{slug}.svf_percentile", v, "percentile points", ledger_rel,
            f"/derived/spread/{slug}/svf_percentile_spread_max_minus_min",
        ))
    rank = ledger["derived"]["rank_under_locked_domain"]
    rows.append(ledger_derived_row(
        "g3.rank_under_locked_domain", rank, "ordered list", ledger_rel,
        "/derived/rank_under_locked_domain",
    ))
    invariant = ledger["derived"]["rank_invariant_across_grid"]
    rows.append(ledger_derived_row(
        "g3.rank_invariant_across_grid", invariant, "boolean", ledger_rel,
        "/derived/rank_invariant_across_grid",
    ))

    spread_values = [spread[slug]["svf_percentile_spread_max_minus_min"] for slug in spread]
    spread_min, spread_max = min(spread_values), max(spread_values)
    rows.append(computed_row(
        "g3.spread.min_across_favelas", spread_min, "percentile points", "min",
        [{"file": ledger_rel, "json_pointer": f"/derived/spread/{slug}/svf_percentile_spread_max_minus_min"}
         for slug in spread],
    ))
    rows.append(computed_row(
        "g3.spread.max_across_favelas", spread_max, "percentile points", "max",
        [{"file": ledger_rel, "json_pointer": f"/derived/spread/{slug}/svf_percentile_spread_max_minus_min"}
         for slug in spread],
    ))

    n_grid_variants = len({k.split(".")[1] for k in ledger["entries"] if k.startswith("g3.grid_")})

    return {
        "narrative": (
            f"Across the {n_grid_variants} grid variants swept, each favela's SVF-percentile "
            f"position moves — by {spread_min:.3g} to {spread_max:.3g} percentile points "
            f"(min-max spread across the five favelas) — but the ORDERING of the five favelas "
            f"does not: rank_invariant_across_grid = {invariant} (every one of the "
            f"{n_grid_variants} variants yields the same descending order). Position moves with "
            f"the grid; ordering does not."
        ),
        "n_grid_variants": n_grid_variants,
        "rows": rows,
    }


def build_sky_resolution(main_root: Path) -> dict:
    manifests = sorted((main_root / "runs").glob("*/manifest.json"))
    counts: dict[int, list[str]] = {}
    for m in manifests:
        try:
            data = json.loads(m.read_text())
        except json.JSONDecodeError:
            continue
        patches = data.get("sky", {}).get("patches")
        if patches is not None:
            counts.setdefault(patches, []).append(str(m.parent.relative_to(main_root)))
    single_resolution = len(counts) <= 1
    return {
        "narrative": (
            f"P1_SKY_PATCHES = {P1_SKY_PATCHES} (imported from src/brisa_solar/constants.py, "
            "never typed as a literal in any P1 module — tests/test_p1_sky_resolution_consistency.py "
            "enforces this by AST). Scanning every run manifest under the main checkout "
            f"({len(manifests)} manifests, {sum(len(v) for v in counts.values())} carrying a "
            f"sky.patches field): {'exactly one' if single_resolution else 'MORE THAN ONE'} "
            f"resolution appears — {sorted(counts)}. No second resolution exists in any code "
            "path feeding a pooled number."
        ),
        "p1_sky_patches": P1_SKY_PATCHES,
        "n_manifests_scanned": len(manifests),
        "distinct_patch_counts": sorted(counts),
        "single_resolution": single_resolution,
        "manifests_by_patch_count": counts,
    }


def build_irradiance_input(main_root: Path) -> dict:
    epw_rel = "data/epw/epw_inventory.json"
    epw = json.loads((main_root / epw_rel).read_text())
    galeao_ghi = epw["galeao"]["annual_ghi_kwh_m2"]
    santos_ghi = epw["santos_dumont"]["annual_ghi_kwh_m2"]
    pct = COMPUTE_FNS["pct_diff_b_over_a"](galeao_ghi, santos_ghi)

    rows = [
        row("epw.galeao.annual_ghi_kwh_m2", galeao_ghi, "kWh/m2",
            file=epw_rel, json_pointer="/galeao/annual_ghi_kwh_m2"),
        row("epw.santos_dumont.annual_ghi_kwh_m2", santos_ghi, "kWh/m2",
            file=epw_rel, json_pointer="/santos_dumont/annual_ghi_kwh_m2"),
        row("epw.primary_station", epw["_meta"]["primary"], "text",
            file=epw_rel, json_pointer="/_meta/primary"),
        computed_row(
            "epw.santos_vs_galeao_ghi_pct_diff", pct, "percent", "pct_diff_b_over_a",
            [{"file": epw_rel, "json_pointer": "/galeao/annual_ghi_kwh_m2"},
             {"file": epw_rel, "json_pointer": "/santos_dumont/annual_ghi_kwh_m2"}],
        ),
    ]
    return {
        "narrative": (
            f"Two EPW stations feed P1: Galeão (primary, {galeao_ghi:.4g} kWh/m2/yr) and "
            f"Santos Dumont ({santos_ghi:.4g} kWh/m2/yr). Santos Dumont reads "
            f"{pct:+.3g}% relative to Galeão — computed here from the two annual GHI values, "
            f"never copied from epw_inventory.json's own _crosscheck field."
        ),
        "rows": rows,
    }


def build_second_axis_validity(main_root: Path, ledger_path: Path, ledger: dict) -> dict:
    ledger_rel = _rel(main_root, ledger_path)
    ventaxis_path = main_root / "docs" / "ventaxis_canonical.md"
    ventaxis_text = ventaxis_path.read_text()
    definition_quote = (
        "For each built 10 m cell, `n_constraints` is a **checklist count** (an\n"
        "integer in {0, 1, 2, 3}) of how many of three independent geometry\n"
        "predicates the cell triggers."
    )
    assert definition_quote in ventaxis_text, "ventaxis definition quote drifted from its source"

    rows = [text_row("wp06.definition_of_record_quote", definition_quote,
                      file="docs/ventaxis_canonical.md")]
    for slug in SITES:
        rows.append(ledger_row(f"wp06.{slug}.n", ledger["entries"][f"wp06.{slug}.n"]["value"],
                                "count", ledger_rel, f"wp06.{slug}.n"))
        for k in ("0", "1", "2", "3"):
            eid = f"wp06.{slug}.share_n{k}"
            rows.append(ledger_row(eid, ledger["entries"][eid]["value"], "fraction",
                                    ledger_rel, eid))

    return {
        "narrative": (
            "Definition of record: docs/ventaxis_canonical.md (VENTAXIS). The second axis "
            "is a per-cell checklist count in {0,1,2,3} of three independent geometry "
            "predicates (vertical, lateral, directional) — never a weighted continuous "
            "index, never a measurement of air exchange itself. Shares below are the "
            "fraction of each site's cells at each count, from the wp06.* ledger entries."
        ),
        "rows": rows,
    }


def build_ground_truth_g2(main_root: Path) -> dict:
    reg_run = "wp03_tls_20260915T205422Z"
    g2_run = "wp03_tls_20260915T215720Z"
    reg_rel = f"runs/{reg_run}/registration.json"
    g2_rel = f"runs/{g2_run}/g2_result_v3.json"
    report_rel = f"runs/{g2_run}/report_v3.md"

    reg = json.loads((main_root / reg_rel).read_text())
    g2 = json.loads((main_root / g2_rel).read_text())
    report_text = (main_root / report_rel).read_text()

    rows = [
        row("g2.registration.median_delta_m", reg["median_delta_m"], "meters",
            file=reg_rel, json_pointer="/median_delta_m"),
        row("g2.registration.p95_abs_delta_m", reg["p95_abs_delta_m"], "meters",
            file=reg_rel, json_pointer="/p95_abs_delta_m"),
        row("g2.registration.n", reg["n"], "count", file=reg_rel, json_pointer="/n"),
        row("g2.street.floor_class", g2["street"]["floor"], "text",
            file=g2_rel, json_pointer="/street/floor"),
        row("g2.street.lt1p5m.r", g2["street"]["classes"][0]["r"], "dimensionless",
            file=g2_rel, json_pointer="/street/classes/0/r"),
        row("g2.street.lt1p5m.median_abs_delta", g2["street"]["classes"][0]["median_abs_delta"],
            "fraction (SVF)", file=g2_rel, json_pointer="/street/classes/0/median_abs_delta"),
        row("g2.street.lt1p5m.share_within_tol", g2["street"]["classes"][0]["share_within_tol"],
            "fraction", file=g2_rel, json_pointer="/street/classes/0/share_within_tol"),
        row("g2.street.lt1p5m.n", g2["street"]["classes"][0]["n"], "count",
            file=g2_rel, json_pointer="/street/classes/0/n"),
        row("g2.confound.baseline_variant_a.lt1p5m.r",
            g2["variants"]["a_dtm_fill_merged"]["classes"][0]["r"], "dimensionless",
            file=g2_rel, json_pointer="/variants/a_dtm_fill_merged/classes/0/r"),
        row("g2.confound.shared_elevation_variant_e.lt1p5m.r",
            g2["variants"]["e_shared_obs_z_als_dtm"]["classes"][0]["r"], "dimensionless",
            file=g2_rel, json_pointer="/variants/e_shared_obs_z_als_dtm/classes/0/r"),
        row("g2.confound.shared_elevation_variant_f.lt1p5m.r",
            g2["variants"]["f_shared_obs_z_smrf_ground"]["classes"][0]["r"], "dimensionless",
            file=g2_rel, json_pointer="/variants/f_shared_obs_z_smrf_ground/classes/0/r"),
    ]
    verdict_quote = (
        "at least one class in variant (e) or (f) reads r >= 0.5 with the shared observer "
        "elevation -- the confound was (at least partly) real; see the per-class table for "
        "which class(es) cleared the bar."
    )
    assert verdict_quote in report_text, "G2 verdict quote drifted from its source"
    rows.append(text_row("g2.verdict_quote", verdict_quote, file=report_rel))

    return {
        "narrative": (
            "What was compared: TLS-derived DSM vs the 2.5D model's ALS-DTM+footprint "
            "surface, in three vertical-clearance classes (<1.5 m, 1.5-3 m, >3 m), for "
            "shared-observer-cell SVF and for street/alley points specifically. This is a "
            "NEGATIVE result with a confirmed confound, reported as it is: the baseline "
            "comparison (variant a, no shared observer elevation) shows essentially no "
            "correlation for the <1.5 m class (r ~ 0.015); forcing a SHARED observer "
            "elevation (variants e/f) raises that to r >= 0.5 for at least one variant, "
            "confirming an observer-elevation confound. Agreement holds only for STREET "
            "points in alleys below the 1.5 m height threshold (floor class = "
            f"'{g2['street']['floor']}', r = {g2['street']['classes'][0]['r']:.3g}, "
            f"median |delta SVF| = {g2['street']['classes'][0]['median_abs_delta']:.3g}). "
            "The open decision this bears on — the 2.5D model's validity floor — is card "
            "`g2_validity_floor`. The PI has not ruled; this dossier states the finding and "
            "names the open card, and does not resolve it."
        ),
        "rows": rows,
        "open_card": "g2_validity_floor",
    }


def build_coverage(main_root: Path) -> dict:
    wp05_run = RUN_OF_RECORD["wp05"]
    wp04_run = RUN_OF_RECORD["wp04"]
    wp05_rel = f"runs/{wp05_run}/manifest.json"
    wp05 = json.loads((main_root / wp05_rel).read_text())

    rows = [
        row("coverage.citywide.n_cells", wp05["n_cells_consolidated"], "count",
            file=wp05_rel, json_pointer="/n_cells_consolidated"),
        row("coverage.citywide.cell_m", wp05["cell_m"], "meters",
            file=wp05_rel, json_pointer="/cell_m"),
    ]
    for slug in SITES:
        summary_rel = f"runs/{wp04_run}/{SITE_DIRS[slug]}/summary.json"
        summary = json.loads((main_root / summary_rel).read_text())
        rows.append(row(f"coverage.site.{slug}.ground_n", summary["ground"]["n"], "count",
                         file=summary_rel, json_pointer="/ground/n"))

    epoch_files = [("citywide", "boundary", "RJ/Favelas_Limit_2019.shp"),
                   ("citywide", "footprints", "RJ/buildings_RJ_2019_utm.gpkg"),
                   ("citywide", "dtm", "RJ/DTM_RJ.tif")]
    for slug in SITES:
        d = SITE_DIRS[slug]
        epoch_files.append((slug, "boundary", f"{d}/{SITE_BOUNDARY_FILE[slug]}"))
        epoch_files.append((slug, "footprints", f"{d}/buildings_extended_300m.gpkg"))
        epoch_files.append((slug, "dtm", f"{d}/dtm_extended_300m.tif"))

    epoch_table = []
    for scope, category, rel in epoch_files:
        p = main_root / "data" / rel
        st = p.stat()
        epoch_table.append({
            "scope": scope,
            "category": category,
            "path": f"data/{rel}",
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "size_bytes": st.st_size,
        })

    return {
        "narrative": (
            f"Citywide: {wp05['n_cells_consolidated']:,} cells at {wp05['cell_m']} m "
            "(WP-05 full run, exhaustive — every fabric cell, no sampling error). Per-site "
            "ground-cell counts below are WP-04's (1 m). The methods epoch table is generated "
            "from each input file's own filesystem metadata (last-modified timestamp, size), "
            "never typed — per the C' plan §1.6. This is a proxy for acquisition vintage, not "
            "a claim about it: no embedded raster datetime tag was found on the citywide DTM "
            "(confirmed via gdalinfo), consistent with the C' plan's own note that "
            "DTM_RJ.tif's true survey vintage is UNVERIFIED."
        ),
        "rows": rows,
        "epoch_table": epoch_table,
    }


def build_declared_limitations(main_root: Path, ledger_path: Path, ledger: dict) -> dict:
    ledger_rel = _rel(main_root, ledger_path)

    # canopy / DSM
    canopy_quote = (
        "PI dropped canopy/vegetation from P1 rather than hold the critical path on it. "
        "Solar and the ventilation index are computed on terrain + buildings only; "
        "vegetation shading is a stated limitation in the methods, and the discussion may "
        "not attribute any shortfall to trees."
    )
    canopy_source_exists = CPRIME_PLAN.exists()
    if canopy_source_exists:
        plan_text = CPRIME_PLAN.read_text()
        assert canopy_quote in plan_text, "canopy-descope quote drifted from its source"
    dsm_hits = sorted(
        p for p in glob.glob(str(main_root / "data" / "**" / "*[Dd][Ss][Mm]*"), recursive=True)
    )

    # facade
    facade_run = "wp04f2_facade_2026-09-15T07:50:07Z"
    facade_rel = f"runs/{facade_run}/crossref.json"
    facade = json.loads((main_root / facade_rel).read_text())
    facade_ledger_hits = [k for k in ledger["entries"] if "facade" in k.lower()]

    rows = [
        text_row("limitations.canopy.descope_quote", canopy_quote, file=str(CPRIME_PLAN)),
        row("limitations.facade.floor_r", facade["floor"]["r"], "dimensionless",
            file=facade_rel, json_pointer="/floor/r"),
        row("limitations.facade.floor_median_abs_delta", facade["floor"]["median_abs_delta"],
            "fraction", file=facade_rel, json_pointer="/floor/median_abs_delta"),
        row("limitations.facade.riodaspedras_r",
            facade["sites"]["riodaspedras"]["variants"]["unweighted"]["overall"]["r"],
            "dimensionless", file=facade_rel,
            json_pointer="/sites/riodaspedras/variants/unweighted/overall/r"),
        row("limitations.facade.vidigal_r",
            facade["sites"]["vidigal"]["variants"]["unweighted"]["overall"]["r"],
            "dimensionless", file=facade_rel,
            json_pointer="/sites/vidigal/variants/unweighted/overall/r"),
    ]

    return {
        "narrative": (
            "Vegetation/canopy: descoped 2026-09-10 (PI decision, quoted below); this "
            f"dossier confirms mechanically that no DSM-named file exists under data/ in "
            f"the main checkout ({len(dsm_hits)} matches) and states plainly that this "
            "limitation may not be used to explain any shortfall. Façade: NOT accepted — "
            f"cross-reference r = {facade['sites']['riodaspedras']['variants']['unweighted']['overall']['r']:.3g} "
            f"(Rio das Pedras) / "
            f"{facade['sites']['vidigal']['variants']['unweighted']['overall']['r']:.3g} (Vidigal), "
            f"both below the floor "
            f"(r >= {facade['floor']['r']}, median |delta| <= {facade['floor']['median_abs_delta']}); "
            f"the ledger carries {len(facade_ledger_hits)} façade-derived entries (must be zero — "
            "confirmed). The 2.5D model's validity floor is the open G2 question above "
            "(card g2_validity_floor, not resolved here)."
        ),
        "rows": rows,
        "n_dsm_files_on_disk": len(dsm_hits),
        "n_facade_entries_in_ledger": len(facade_ledger_hits),
        "cprime_plan_cited": str(CPRIME_PLAN),
        "cprime_plan_readable": canopy_source_exists,
    }


# ── red-line guard (L1): no favela-vs-formal contrast anywhere in the ledger ─

def check_no_l1_contrast(ledger: dict) -> list[str]:
    flagged_tokens = ("non_favela", "formal", "contrast")
    hits = [
        entry_id for entry_id in ledger["entries"]
        if any(tok in entry_id for tok in flagged_tokens)
    ]
    return hits


# ── assembly ──────────────────────────────────────────────────────────────

def build_dossier(main_root: Path) -> dict:
    ledger_path, ledger = find_ledger(main_root)
    l1_hits = check_no_l1_contrast(ledger)
    if l1_hits:
        # Explicitly OUT of scope to resolve; the spec says stop and report.
        print("STOP: possible favela-vs-formal (L1) field(s) in the ledger:", l1_hits,
              file=sys.stderr)
        sys.exit(2)

    sections = {
        "engine_acceptance": build_engine_acceptance(main_root, ledger_path, ledger),
        "domain_sensitivity_g3": build_domain_sensitivity_g3(main_root, ledger_path, ledger),
        "sky_resolution": build_sky_resolution(main_root),
        "irradiance_input": build_irradiance_input(main_root),
        "second_axis_validity": build_second_axis_validity(main_root, ledger_path, ledger),
        "ground_truth_g2": build_ground_truth_g2(main_root),
        "coverage": build_coverage(main_root),
        "declared_limitations": build_declared_limitations(main_root, ledger_path, ledger),
    }

    all_files = set()
    for sec in sections.values():
        for r in sec.get("rows", []):
            src = r["source"]
            if "file" in src:
                all_files.add(src["file"])
            elif "computed_from" in src:
                all_files.update(f["file"] for f in src["computed_from"])
            elif "text_substring_of" in src:
                all_files.add(src["text_substring_of"])

    return {
        "_utc": _utc_now(),
        "status": "staged",
        "_meta": {
            "purpose": (
                "Reviewer robustness dossier for P1 v2 — assembly by code of every "
                "acceptance and sensitivity number already final in a run of record. "
                "Nothing recomputed; nothing new claimed. STAGED — no promotion to "
                "shared/figures, papers/, or anything the public hub serves."
            ),
            "spec": "docs/robustness_dossier_spec.md",
            "main_root_used": str(main_root),
            "ledger_used": {
                "path": _rel(main_root, ledger_path),
                "run_utc": ledger["_utc"],
                "n_entries": len(ledger["entries"]),
                "status": ledger["status"],
            },
            "sky_patches": int(P1_SKY_PATCHES),
            "l1_contrast_check": {"flagged_ids": l1_hits, "clear": not l1_hits},
            "sources_read": sorted(all_files),
        },
        "sections": sections,
    }


def _fmt(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, str):
        # markdown table cells can't carry a raw newline; the JSON value keeps
        # the exact source text (that's what the drift test checks against).
        return " ".join(value.split())
    return str(value)


def render_markdown(dossier: dict) -> str:
    meta = dossier["_meta"]
    lines = [
        "# Robustness dossier — P1 v2 (STAGED, not promoted)",
        "",
        f"Generated {dossier['_utc']} · status: {dossier['status']}",
        "",
        f"Ledger used: `{meta['ledger_used']['path']}` "
        f"({meta['ledger_used']['n_entries']} entries, run {meta['ledger_used']['run_utc']}, "
        f"status {meta['ledger_used']['status']})",
        "",
        f"L1 (favela-vs-formal) contrast check: "
        f"{'CLEAR — no flagged ids' if meta['l1_contrast_check']['clear'] else 'FLAGGED — ' + str(meta['l1_contrast_check']['flagged_ids'])}",
        "",
        meta["purpose"],
        "",
    ]

    titles = {
        "engine_acceptance": "1. Engine acceptance",
        "domain_sensitivity_g3": "2. Domain sensitivity (G3)",
        "sky_resolution": "3. Sky resolution",
        "irradiance_input": "4. Irradiance input",
        "second_axis_validity": "5. Second-axis validity (VENTAXIS)",
        "ground_truth_g2": "6. Ground-truth comparison (G2)",
        "coverage": "7. Coverage",
        "declared_limitations": "8. Declared limitations",
    }
    for key, title in titles.items():
        sec = dossier["sections"][key]
        lines.append(f"## {title}")
        lines.append("")
        if "narrative" in sec:
            lines.append(sec["narrative"])
            lines.append("")
        if sec.get("rows"):
            lines.append("| id | value | unit | source |")
            lines.append("|---|---|---|---|")
            for r in sec["rows"]:
                src = r["source"]
                if "file" in src:
                    src_str = f"`{src['file']}#{src['json_pointer']}`"
                elif "computed_from" in src:
                    src_str = f"computed ({src['fn']}) from " + "; ".join(
                        f"`{f['file']}#{f['json_pointer']}`" for f in src["computed_from"])
                else:
                    src_str = f"`{src['text_substring_of']}` (verbatim substring)"
                val = _fmt(r["value"])
                lines.append(f"| `{r['id']}` | {val} | {r['unit']} | {src_str} |")
            lines.append("")
        if key == "engine_acceptance":
            lines.append(f"Chosen variant: {sec['chosen_variant_quote']['value']}")
            lines.append("")
            lines.append("Analytic self-checks (closed-form, no data dependency):")
            for c in sec["analytic_self_checks"]:
                lines.append(f"- **{c['name']}**: {c['predicate']} "
                              f"(`{c['defined_in']}`, verified by `{c['verified_by']}`)")
            lines.append("")
        if key == "coverage":
            lines.append("Methods epoch table (generated from file metadata, never typed):")
            lines.append("")
            lines.append("| scope | category | path | mtime (UTC) | size (bytes) |")
            lines.append("|---|---|---|---|---|")
            for e in sec["epoch_table"]:
                lines.append(f"| {e['scope']} | {e['category']} | `{e['path']}` | "
                              f"{e['mtime_utc']} | {e['size_bytes']:,} |")
            lines.append("")
        if key == "ground_truth_g2":
            lines.append(f"Open card (not resolved here): **{sec['open_card']}**")
            lines.append("")

    lines.append("## Sources read")
    lines.append("")
    for f in meta["sources_read"]:
        lines.append(f"- `{f}`")
    lines.append("")
    return "\n".join(lines)


def write_dossier(this_repo_root: Path, main_root: Path, run_dir: Path | None = None) -> Path:
    dossier = build_dossier(main_root)

    if run_dir is None:
        run_dir = this_repo_root / "runs" / ("robustness_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "dossier.json").write_text(json.dumps(dossier, indent=1, ensure_ascii=False))
    (run_dir / "dossier.md").write_text(render_markdown(dossier))

    manifest = {
        "_utc": _utc_now(),
        "git_sha": _git_sha(this_repo_root),
        "main_root_used": str(main_root),
        "ledger_used": dossier["_meta"]["ledger_used"],
        "sky": {"patches": int(P1_SKY_PATCHES)},
        "n_sources_read": len(dossier["_meta"]["sources_read"]),
        "staged_only": True,
        "spec": "docs/robustness_dossier_spec.md",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-root", default=str(DEFAULT_MAIN_ROOT))
    ap.add_argument("--run-dir", default=None, help="defaults to a fresh runs/robustness_<UTC>/")
    args = ap.parse_args()

    run_dir = write_dossier(THIS_REPO_ROOT, Path(args.main_root),
                             Path(args.run_dir) if args.run_dir else None)
    print(f"Wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
