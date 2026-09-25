#!/usr/bin/env python3
"""Per-subunit morphology — site-agnostic CLI over src.sites.subunit_morphology.

Generalises docs/briefs/mare/mare_subunits.py's per-neighbourhood Maré table
(density/height/footprint/sky-view from WP-06's grid, winter-sun/annual-
irradiation from a WP-04 ground-point run) onto any OTHER campaign site that
declares `subunits` in config/sites.yaml — today that is complexo_do_alemao
and riodaspedras (their IPP favela polygons under the complexo). Vidigal and
Rocinha have no subunits declared (each is one polygon, "Isolada" per
config/sites.yaml's comment) and are skipped with a printed note, not an
error. Maré is also skipped here: it keeps its own dedicated brief script
(docs/briefs/mare/mare_subunits.py), which additionally fits a Maré-internal
fabric clustering this module does not generalise (see that module's
docstring for why).

Descriptive only: rows are geographic order (north to south by mean grid-
cell y), never ranked, and BETWEEN_SUBUNITS_LABEL ("between communities") is
kept for study-area ground inside no named subunit.

Writes, per site, under one timestamped run directory:
    runs/subunit_morphology_<UTC>/<site>/subunit_morphology.csv
    runs/subunit_morphology_<UTC>/<site>/summary.json
    runs/subunit_morphology_<UTC>/manifest.json

Usage:
    python scripts/build_subunit_morphology.py
    python scripts/build_subunit_morphology.py --site complexo_do_alemao riodaspedras
    python scripts/build_subunit_morphology.py --root /home/theo/SCL/SCR/MorphoFavela
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sites.subunit_morphology import MissingSource, build_subunit_table  # noqa: E402
from src.sites.territory import BETWEEN_SUBUNITS_LABEL, load_sites_config  # noqa: E402

#: Sites this script never runs for, and why — read at call time so a new
#: config/sites.yaml entry is picked up without touching this file.
SKIP = {
    "maré": "keeps its own dedicated script (docs/briefs/mare/mare_subunits.py, "
            "which also fits a Maré-internal fabric clustering this module does not generalise)",
}


def eligible_sites(sites_cfg: dict) -> tuple[list[str], dict[str, str]]:
    """(sites to run, {skipped site: reason}) from config/sites.yaml — a
    site is eligible iff it declares `subunits` and isn't in SKIP."""
    run, skipped = [], {}
    for site, cfg in sites_cfg.items():
        if site in SKIP:
            skipped[site] = SKIP[site]
        elif cfg.get("subunits") is None:
            skipped[site] = "no subunits declared (single polygon, \"Isolada\")"
        else:
            run.append(site)
    return run, skipped


def _write_site(dest: Path, site: str, table) -> dict:
    site_dir = dest / site
    site_dir.mkdir(parents=True, exist_ok=True)
    csv_path = site_dir / "subunit_morphology.csv"
    table.to_csv(csv_path, index=False)
    summary = {
        "site": site,
        "wp04_run": table.attrs.get("wp04_run"),
        "n_subunits": int(len(table)),
        "n_cells_total": int(table["n_cells"].sum()),
        "has_between_subunits": bool((table["name"] == BETWEEN_SUBUNITS_LABEL).any()),
        "row_order": "north to south (descending mean grid-cell y) — geographic, never ranked",
        "columns": list(table.columns),
        "csv": str(csv_path.relative_to(dest.parent.parent)),
    }
    (site_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="checkout that owns data/outputs/runs (default: this script's own repo; "
                         "pass the main checkout's absolute path when running from a worktree)")
    ap.add_argument("--site", nargs="+", default=None,
                    help="restrict to these site keys (default: every eligible site)")
    args = ap.parse_args()

    root = args.root.resolve()
    outputs_root = root / "outputs"
    runs_root = root / "runs"
    sites_cfg = load_sites_config()

    eligible, skipped = eligible_sites(sites_cfg)
    sites = args.site if args.site else eligible
    unknown = [s for s in sites if s not in sites_cfg]
    if unknown:
        raise SystemExit(f"unknown site(s) {unknown} — config/sites.yaml has: {sorted(sites_cfg)}")

    for s in sites:
        if s in SKIP:
            print(f"subunit_morphology: skip {s} — {SKIP[s]}")
    sites = [s for s in sites if s not in SKIP]

    utc = dt.datetime.now(dt.timezone.utc)
    dest = runs_root / f"subunit_morphology_{utc:%Y%m%dT%H%M%SZ}"
    dest.mkdir(parents=True)

    run_summaries = {}
    errors = {}
    for site in sites:
        try:
            table = build_subunit_table(site, outputs_root, runs_root, root=root)
        except MissingSource as e:
            errors[site] = str(e)
            print(f"subunit_morphology: SKIP {site} — {e}")
            continue
        summary = _write_site(dest, site, table)
        run_summaries[site] = summary
        print(f"subunit_morphology: {site} — {summary['n_subunits']} subunits "
              f"({summary['n_cells_total']} cells) -> {dest / site / 'subunit_morphology.csv'}")

    manifest = {
        "_utc": utc.isoformat().replace("+00:00", "Z"),
        "sites_run": list(run_summaries),
        "sites_skipped_by_config": skipped,
        "sites_missing_source": errors,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(dest / "manifest.json")

    return 1 if (not run_summaries and sites) else 0


if __name__ == "__main__":
    raise SystemExit(main())
