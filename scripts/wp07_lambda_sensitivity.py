"""Sensitivity of P1's geometry-constraint count to the lambda_f threshold.

PI ruling 2026-10-08: 0.65 is a study-defined cut (by analogy with Oke's street
height-to-width ratio of 0.65); report 0.5 and 0.8. Reads the WP-06 per-cell tables of the
run of record, re-derives only the vertical flag and n_constraints
(lateral/directional flags are threshold-independent), and reuses
wp07_crosstab.crosstab and wp07_round2.stratified_crosstab unchanged. The 0.65
column is asserted equal to the crosstab/round-2 runs of record, the ledger and
the WP-06 shares; nothing under runs/ other than the new run dir is written.
Aggregates only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brisa_solar.constants import LAMBDA_F_CONSTRAINT_MIN  # noqa: E402
from src.brisa_solar.wp07_crosstab import crosstab  # noqa: E402
from src.brisa_solar.wp07_ledger import RUN_OF_RECORD  # noqa: E402
from src.brisa_solar.wp07_round2 import SITE_DIRS, CROSSTAB_IN_NAME, stratified_crosstab  # noqa: E402

THRESHOLDS = (0.5, 0.65, 0.8)
LEDGER_RUN = "wp07_ledger_20261007T215336Z"
SITE_ORDER = ["vidigal", "rocinha", "complexo_do_alemao", "mare", "riodaspedras"]
COLS = ["lambda_f_mean", "share_ge_2h_winter", "sun_h_winter_p50",
        "constraint_vertical", "constraint_lateral", "constraint_directional", "n_constraints"]


XTAB_COLS = ["share_ge_2h_winter", "sun_h_winter_p50", "n_constraints"]


def load_tables(root: Path) -> dict[str, pd.DataFrame]:
    return {
        slug: pd.read_csv(root / "outputs" / SITE_DIRS[slug] / "geometry_indicators" / CROSSTAB_IN_NAME, usecols=COLS)
        for slug in SITE_ORDER
    }


def with_threshold(table: pd.DataFrame, thr: float) -> pd.DataFrame:
    t = table.copy()
    t["constraint_vertical"] = (np.nan_to_num(t["lambda_f_mean"].to_numpy(), nan=0.0) >= thr).astype(int)
    t["n_constraints"] = t["constraint_vertical"] + t["constraint_lateral"] + t["constraint_directional"]
    return t


def _direction(rows: list[dict]) -> dict:
    pop = [r["point_deficit"] for r in rows if r["cells"] > 0]
    return {
        "top_class_above_bottom_class": bool(pop[-1] > pop[0]) if len(pop) > 1 else None,
        "monotone_nondecreasing": bool(all(b >= a for a, b in zip(pop, pop[1:]))),
    }


def block(table: pd.DataFrame, thr: float) -> dict:
    t = with_threshold(table, thr)
    n = len(t)
    ground = t.dropna(subset=["share_ge_2h_winter", "sun_h_winter_p50"])
    rows = crosstab(t[XTAB_COLS])
    return {
        "cells_total": int(n),
        "cells_ground_sampled": int(len(ground)),
        "share_lambda_f_constraint_all_cells": float(t["constraint_vertical"].mean()),
        "share_lambda_f_constraint_ground_sampled": float(ground["constraint_vertical"].mean()),
        "constraint_count_share_all_cells": {
            str(k): float((t["n_constraints"] == k).mean()) for k in range(4)
        },
        "gradient": {"rows": rows, **_direction(rows)},
        "stratified": stratified_crosstab(t),
    }


def compute(tables: dict[str, pd.DataFrame]) -> dict:
    pooled = pd.concat([tables[slug] for slug in SITE_DIRS], ignore_index=True)  # run-of-record row order
    out = {}
    for thr in THRESHOLDS:
        out[f"{thr:g}"] = {
            "per_site": {slug: block(tables[slug], thr) for slug in SITE_ORDER},
            "pooled": block(pooled, thr),
        }
    return out


def verify_record(result: dict, root: Path) -> dict:
    """Assert the 0.65 column equals every run-of-record value that depends on it."""
    assert LAMBDA_F_CONSTRAINT_MIN == 0.65, "constant changed; 0.65 column is no longer the run of record"
    col = result["0.65"]
    xt = json.loads((root / "runs" / RUN_OF_RECORD["crosstab"] / "summary.json").read_text())
    r2 = json.loads((root / "runs" / RUN_OF_RECORD["round2"] / "summary.json").read_text())
    wp06 = json.loads((root / "runs" / RUN_OF_RECORD["wp06"] / "summary.json").read_text())
    ledger = json.loads((root / "runs" / LEDGER_RUN / "ledger.json").read_text())["entries"]
    n_checked = 0

    def eq(a, b, what):
        nonlocal n_checked
        assert a == b, f"0.65 does not reproduce {what}: {a!r} != {b!r}"
        n_checked += 1

    for slug in SITE_ORDER + ["pooled"]:
        mine = col["pooled"] if slug == "pooled" else col["per_site"][slug]
        disk = "pooled" if slug == "pooled" else SITE_DIRS[slug]
        eq(mine["cells_total"], xt["per_site"][disk]["cells_total"], f"crosstab cells_total {slug}")
        for rec, row in zip(xt["per_site"][disk]["rows"], mine["gradient"]["rows"]):
            for f in ("n_constraints", "cells", "share_of_cells", "point_deficit", "cell_deficit"):
                a, b = row[f], rec[f]
                eq(a, b, f"crosstab {slug} n{rec['n_constraints']} {f}")
            for f in ("cells", "point_deficit"):
                eq(row[f], ledger[f"crosstab.{slug}.n{row['n_constraints']}.{f}"]["value"], f"ledger crosstab.{slug}.n{row['n_constraints']}.{f}")
        eq(mine["stratified"], r2["stratified"][slug], f"round2 stratified {slug}")
        for v in ("v0", "v1"):
            for o in ("o0", "o1", "o2"):
                for f in ("cells", "point_deficit", "cell_deficit"):
                    eq(mine["stratified"][v][o][f], ledger[f"xtab_strat.{slug}.{v}.{o}.{f}"]["value"],
                       f"ledger xtab_strat.{slug}.{v}.{o}.{f}")
        if slug != "pooled":
            eq(mine["cells_total"], wp06["per_site"][disk]["n"], f"wp06 n {slug}")
            for k in range(4):
                eq(mine["constraint_count_share_all_cells"][str(k)], wp06["per_site"][disk]["shares"][str(k)],
                   f"wp06 share_n{k} {slug}")
    return {"values_checked": n_checked, "against": [RUN_OF_RECORD["crosstab"], RUN_OF_RECORD["round2"],
                                                     RUN_OF_RECORD["wp06"], LEDGER_RUN]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.repo_root)
    result = compute(load_tables(root))
    check = verify_record(result, root)
    run_dir = root / "runs" / ("wp07_lambda_sens_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True)
    summary = {
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ruling": "PI 2026-10-08: lambda_f cut is study-defined (analogy with Oke's street height-to-width ratio of 0.65); report 0.5 and 0.8",
        "thresholds": list(THRESHOLDS),
        "run_of_record_threshold": LAMBDA_F_CONSTRAINT_MIN,
        "site_order": SITE_ORDER,
        "inputs": {"geometry_csv": CROSSTAB_IN_NAME, "wp06_run": RUN_OF_RECORD["wp06"]},
        "reproduction_check_0.65": check,
        "gradient_measure": "point_deficit = cell-weighted share of ground points below the 2 h winter floor; cell_deficit = share of cells with median ground point below 2 h",
        "results": result,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"Wrote {run_dir} ({check['values_checked']} values reproduced at 0.65)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
