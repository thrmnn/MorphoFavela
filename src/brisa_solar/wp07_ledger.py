"""WP-07A — the C′ numbers ledger (C′ reframe, phase 1 of WP-07's numbers side).

Spec: docs/wp07_ledger_spec.md. Both PI decisions (`wp05_run_design`,
`g3_domain`) landed 2026-09-15, so every headline number WP-04/05/06 and G3
produced is FINAL — even though the source run folders themselves still
carry the stale "PROVISIONAL" string from before the decision. This module
copies those numbers BY CODE from their run of record into one ledger
(`runs/wp07_ledger_<UTC>/ledger.json` + `ledger.md`), each entry carrying a
JSON-pointer back to the exact value it was read from, so every number here
is mechanically re-verifiable and never hand-typed.

Not in scope: paper prose (the PI writes every sentence), façade numbers
(façade was not accepted), per-cell data (never a ledger entry), any run
folder other than the five runs of record below.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .constants import P1_SKY_PATCHES, REPO_ROOT

#: Runs of record (docs/wp07_ledger_spec.md "Runs of record"). Never modified,
#: never re-read from anywhere else.
RUN_OF_RECORD = {
    "wp05": "wp05_full_20260914T215419Z",
    "wp04": "wp04_sites_20260914T230606Z",
    "g3": "g3_domain_20260915T042927Z",
    "wp06": "wp06_geometry_20260915T052604Z",
    "wp02_crossref": "wp02_horizon_20260914T195630Z",
}

#: engine.crossref.* comes from this run's crossref_diagnostic.json, variant
#: A_nearest_sampling (march_sampling="nearest", r=0.9945) — the nearest-cell
#: march named in the spec, not the bilinear-march baseline in the same file
#: (r=0.988) or the sibling run's single-site crossref (also bilinear, r=0.988).
CROSSREF_FILE = "crossref_diagnostic.json"
CROSSREF_VARIANT = "A_nearest_sampling"

#: slug -> the exact key study_favelas/per-favela dicts use in the source JSON.
FAVELAS = {
    "vidigal": "Vidigal",
    "rocinha": "Rocinha",
    "complexo_do_alemao": "Complexo do Alemão",
    "mare": "Maré",
    "riodaspedras": "Rio das Pedras",
}

#: slug -> the run-folder / summary-key spelling WP-04 and WP-06 use on disk.
SITE_DIRS = {
    "vidigal": "vidigal",
    "rocinha": "rocinha",
    "complexo_do_alemao": "complexo_do_alemao",
    "riodaspedras": "riodaspedras",
    "mare": "maré",
}
SITES = list(SITE_DIRS)

CITYWIDE_PERCENTILES = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
SITE_PERCENTILES = ["p10", "p25", "p50", "p75", "p90"]

#: the 9 grid variants g3_domain swept (fabric_coverage_threshold, fabric_footprint_distance_m).
#: Excludes the wp04_polygon_interior universe on purpose — see docs/wp07_ledger_spec.md
#: deliverable 2 ("exclude the WP-04 polygon-interior universe").
LOCKED_VARIANT = (0.1, 10.0)

DECIDED_BY = ["wp05_run_design 2026-09-15", "g3_domain 2026-09-15"]


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


def _grid_slug(threshold: float, distance: float) -> str:
    return f"{round(threshold * 100):03d}_{round(distance):02d}"


def resolve_pointer(doc, pointer: str):
    """Minimal RFC-6901 JSON-pointer resolution (dict keys + list indices).

    The single mechanism every ledger value goes through, so an entry's value
    is never a hand-typed number — it is always whatever this function reads
    back out of the source document at the recorded pointer.
    """
    if pointer in ("", "/"):
        return doc
    cur = doc
    for raw in pointer.lstrip("/").split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def _add(entries: dict, doc, file_rel: str, entry_id: str, pointer: str,
         run_id: str, run_utc: str, unit: str, release_class: str = "publishable-candidate") -> None:
    if entry_id in entries:
        raise ValueError(f"duplicate ledger id: {entry_id}")
    entries[entry_id] = {
        "id": entry_id,
        "value": resolve_pointer(doc, pointer),
        "unit": unit,
        "source": {
            "file": file_rel,
            "json_pointer": pointer,
            "run_id": run_id,
            "run_utc": run_utc,
        },
        "status": "final",
        "decided_by": list(DECIDED_BY),
        "release_class": release_class,
    }


def _load(repo_root: Path, run_id: str, filename: str) -> tuple[dict, str]:
    path = repo_root / "runs" / run_id / filename
    doc = json.loads(path.read_text())
    return doc, str(path.relative_to(repo_root))


def _build_citywide(entries: dict, wp05: dict, wp05_rel: str, run_id: str) -> None:
    run_utc = wp05["_utc"]
    for metric, unit in (("svf", "fraction"), ("kwh_m2", "kWh/m2")):
        for pct in CITYWIDE_PERCENTILES:
            _add(entries, wp05, wp05_rel, f"citywide.{metric}.{pct}",
                 f"/citywide/{metric}/{pct}", run_id, run_utc, unit)


def _build_favelas(entries: dict, wp05: dict, wp05_rel: str, run_id: str) -> None:
    run_utc = wp05["_utc"]
    for slug, display in FAVELAS.items():
        for metric, unit in (("svf", "fraction"), ("kwh_m2", "kWh/m2")):
            base = f"/study_favelas/{display}/{metric}"
            _add(entries, wp05, wp05_rel, f"favela.{slug}.{metric}.median",
                 f"{base}/median", run_id, run_utc, unit)
            _add(entries, wp05, wp05_rel, f"favela.{slug}.{metric}.percentile",
                 f"{base}/citywide_percentile_position", run_id, run_utc, "percentile")
            _add(entries, wp05, wp05_rel, f"favela.{slug}.{metric}.iqr_low",
                 f"{base}/iqr/0", run_id, run_utc, unit)
            _add(entries, wp05, wp05_rel, f"favela.{slug}.{metric}.iqr_high",
                 f"{base}/iqr/1", run_id, run_utc, unit)


def _build_sites(entries: dict, repo_root: Path, run_id: str) -> None:
    metrics = (
        ("svf", "svf", "fraction"),
        ("kwh_m2", "kwh_m2", "kWh/m2"),
        ("sun_h_winter", "direct_sun_hours_winter_solstice", "hours"),
        ("sun_h_equinox", "direct_sun_hours_equinox", "hours"),
    )
    for slug in SITES:
        summary, rel = _load(repo_root, run_id, f"{SITE_DIRS[slug]}/summary.json")
        run_utc = summary["_utc"]
        for surface in ("ground", "street"):
            for metric_slug, key, unit in metrics:
                for pct in SITE_PERCENTILES:
                    _add(entries, summary, rel, f"site.{slug}.{surface}.{metric_slug}.{pct}",
                         f"/{surface}/{key}/{pct}", run_id, run_utc, unit)
        for share_key in summary["ground"]["threshold_shares"]:
            _add(entries, summary, rel, f"site.{slug}.ground.{share_key}",
                 f"/ground/threshold_shares/{share_key}", run_id, run_utc, "fraction")


def _build_g3(entries: dict, g3: dict, g3_rel: str, run_id: str) -> None:
    run_utc = g3["_utc"]
    for i, variant in enumerate(g3["variants"]):
        slug_grid = _grid_slug(variant["fabric_coverage_threshold"], variant["fabric_footprint_distance_m"])
        for slug, display in FAVELAS.items():
            _add(entries, g3, g3_rel, f"g3.grid_{slug_grid}.{slug}.svf_percentile",
                 f"/variants/{i}/study_favelas/{display}/svf_percentile_of_citywide_median",
                 run_id, run_utc, "percentile")


def _build_wp06(entries: dict, wp06: dict, wp06_rel: str, run_id: str) -> None:
    run_utc = wp06["_utc"]
    for slug in SITES:
        dirname = SITE_DIRS[slug]
        _add(entries, wp06, wp06_rel, f"wp06.{slug}.n",
             f"/per_site/{dirname}/n", run_id, run_utc, "count")
        for k in ("0", "1", "2", "3"):
            _add(entries, wp06, wp06_rel, f"wp06.{slug}.share_n{k}",
                 f"/per_site/{dirname}/shares/{k}", run_id, run_utc, "fraction")


def _build_engine(entries: dict, crossref: dict, crossref_rel: str, run_id: str) -> None:
    run_utc = crossref["_utc"]
    base = f"/variants/{CROSSREF_VARIANT}"
    _add(entries, crossref, crossref_rel, "engine.crossref.r",
         f"{base}/r", run_id, run_utc, "dimensionless")
    _add(entries, crossref, crossref_rel, "engine.crossref.median_abs_delta",
         f"{base}/median_abs_delta", run_id, run_utc, "fraction")
    _add(entries, crossref, crossref_rel, "engine.crossref.p95_abs_delta",
         f"{base}/p95_abs_delta", run_id, run_utc, "fraction")


def _favela_svf_percentile(variant: dict, display: str) -> float:
    return variant["study_favelas"][display]["svf_percentile_of_citywide_median"]


def build_derived(g3: dict) -> dict:
    """Deliverable 2: per-favela max-min SVF-percentile spread across the 9
    grid variants (wp04_polygon_interior excluded), and the locked-domain
    rank order with an across-grid invariance flag."""
    variants = g3["variants"]

    spread = {}
    for slug, display in FAVELAS.items():
        vals = [_favela_svf_percentile(v, display) for v in variants]
        spread[slug] = {
            "id": f"g3.spread.{slug}.svf",
            "svf_percentile_spread_max_minus_min": max(vals) - min(vals),
        }

    locked = next(
        v for v in variants
        if (v["fabric_coverage_threshold"], v["fabric_footprint_distance_m"]) == LOCKED_VARIANT
    )
    rank_under_locked_domain = sorted(
        FAVELAS, key=lambda slug: _favela_svf_percentile(locked, FAVELAS[slug]), reverse=True
    )

    rank_invariant = all(
        sorted(FAVELAS, key=lambda slug: _favela_svf_percentile(v, FAVELAS[slug]), reverse=True)
        == rank_under_locked_domain
        for v in variants
    )

    return {
        "spread": spread,
        "rank_under_locked_domain": rank_under_locked_domain,
        "rank_invariant_across_grid": rank_invariant,
    }


def build_ledger(repo_root: Path) -> dict:
    repo_root = Path(repo_root)

    wp05, wp05_rel = _load(repo_root, RUN_OF_RECORD["wp05"], "distribution.json")
    g3, g3_rel = _load(repo_root, RUN_OF_RECORD["g3"], "sensitivity.json")
    wp06, wp06_rel = _load(repo_root, RUN_OF_RECORD["wp06"], "summary.json")
    crossref, crossref_rel = _load(repo_root, RUN_OF_RECORD["wp02_crossref"], CROSSREF_FILE)

    entries: dict[str, dict] = {}
    _build_citywide(entries, wp05, wp05_rel, RUN_OF_RECORD["wp05"])
    _build_favelas(entries, wp05, wp05_rel, RUN_OF_RECORD["wp05"])
    _build_sites(entries, repo_root, RUN_OF_RECORD["wp04"])
    _build_g3(entries, g3, g3_rel, RUN_OF_RECORD["g3"])
    _build_wp06(entries, wp06, wp06_rel, RUN_OF_RECORD["wp06"])
    _build_engine(entries, crossref, crossref_rel, RUN_OF_RECORD["wp02_crossref"])

    derived = build_derived(g3)

    meta = {
        "supersession": (
            "Both PI decisions (wp05_run_design, g3_domain) were decided 2026-09-15; "
            "every source run's own 'PROVISIONAL — ... untapped' status string is "
            "superseded by that decision. Every entry in this ledger is status=final "
            "regardless of what the source file's own status field still says."
        ),
        "facade_exclusion": (
            "WP-04's facade surface is excluded throughout — facade was not accepted "
            "for release; only ground and street summaries are copied into this ledger."
        ),
        "sky_patches": int(P1_SKY_PATCHES),
        "rounding": "values unrounded; ledger.md rounds to 3 significant figures for reading only",
        "derived_formula": (
            "derived.spread[<favela>] = max(svf_percentile_of_citywide_median across the "
            "9 g3_domain grid variants) - min(same) for that favela, EXCLUDING the "
            "wp04_polygon_interior universe. derived.rank_under_locked_domain = the 5 "
            "favelas sorted descending by svf_percentile_of_citywide_median under the "
            "locked 0.10/10 grid variant (g3_domain, 2026-09-15). "
            "derived.rank_invariant_across_grid = True iff every one of the 9 grid "
            "variants yields that identical descending order."
        ),
        "engine_acceptance_source": (
            f"{crossref_rel}#/variants/{CROSSREF_VARIANT} — chosen because "
            f"march_sampling='nearest' is the nearest-cell march named in "
            "docs/wp07_ledger_spec.md (r≈0.995), not the bilinear-march baseline "
            "in the same file (r≈0.988) nor the sibling run's single-site crossref "
            "(also bilinear-march, r≈0.988)."
        ),
        "decision_provenance_pointers": {
            "domain_status": "config/params.yaml#/domain/status",
            "sampling_run_design": "config/params.yaml#/sampling/run_design",
            "sampling_cell_m_status": "config/params.yaml#/sampling/cell_m_status",
        },
        "definition_of_record": {
            "wp06.*": "docs/ventaxis_canonical.md",
        },
        "runs_of_record": RUN_OF_RECORD,
    }

    return {
        "_utc": _utc_now(),
        "status": "final",
        "_meta": meta,
        "entries": entries,
        "derived": derived,
    }


def _fmt_3sig(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.3g}"


def render_markdown(ledger: dict) -> str:
    meta = ledger["_meta"]
    lines = [
        "# WP-07 C′ numbers ledger",
        "",
        f"Generated {ledger['_utc']} · status: {ledger['status']}",
        "",
        "## _meta",
        "",
    ]
    for key in ("supersession", "facade_exclusion", "rounding", "derived_formula", "engine_acceptance_source"):
        lines.append(f"- **{key}**: {meta[key]}")
    lines.append(f"- **sky_patches**: {meta['sky_patches']}")
    lines.append("- **decision_provenance_pointers**:")
    for k, v in meta["decision_provenance_pointers"].items():
        lines.append(f"  - {k}: `{v}`")
    lines.append("- **definition_of_record**:")
    for k, v in meta["definition_of_record"].items():
        lines.append(f"  - {k}: `{v}`")
    lines.append("- **runs_of_record**:")
    for k, v in meta["runs_of_record"].items():
        lines.append(f"  - {k}: `{v}`")
    lines.append("")

    lines.append("## Entries")
    lines.append("")
    lines.append(f"{len(ledger['entries'])} entries. Values rounded to 3 significant figures for reading; ledger.json carries the unrounded values.")
    lines.append("")
    lines.append("| id | value | unit | release_class | run_id |")
    lines.append("|---|---|---|---|---|")
    for entry_id in sorted(ledger["entries"]):
        e = ledger["entries"][entry_id]
        lines.append(f"| {entry_id} | {_fmt_3sig(e['value'])} | {e['unit']} | {e['release_class']} | {e['source']['run_id']} |")
    lines.append("")

    lines.append("## Derived")
    lines.append("")
    lines.append(f"- **rank_under_locked_domain** (descending SVF percentile, 0.10/10 grid): {', '.join(ledger['derived']['rank_under_locked_domain'])}")
    lines.append(f"- **rank_invariant_across_grid**: {ledger['derived']['rank_invariant_across_grid']}")
    lines.append("")
    lines.append("| favela | svf percentile spread (max-min, 9 grid variants) |")
    lines.append("|---|---|")
    for slug in sorted(ledger["derived"]["spread"]):
        s = ledger["derived"]["spread"][slug]
        lines.append(f"| {slug} | {_fmt_3sig(s['svf_percentile_spread_max_minus_min'])} |")
    lines.append("")

    return "\n".join(lines)


def write_ledger(repo_root: Path, run_dir: Path | None = None) -> Path:
    repo_root = Path(repo_root)
    ledger = build_ledger(repo_root)

    if run_dir is None:
        run_dir = repo_root / "runs" / ("wp07_ledger_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "ledger.json").write_text(json.dumps(ledger, indent=1, ensure_ascii=False))
    (run_dir / "ledger.md").write_text(render_markdown(ledger))

    manifest = {
        "_utc": _utc_now(),
        "git_sha": _git_sha(repo_root),
        "sky": {"patches": int(P1_SKY_PATCHES)},
        "source_runs": RUN_OF_RECORD,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--run-dir", default=None, help="defaults to a fresh runs/wp07_ledger_<UTC>/")
    args = ap.parse_args()

    run_dir = write_ledger(Path(args.repo_root), Path(args.run_dir) if args.run_dir else None)
    print(f"Wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
