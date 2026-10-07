"""WP-07 — descriptive cross-tabulation of the two C′ axes.

Sunlight deficit (Athens Charter 2 h floor, winter solstice, ground points)
against the WP-06 geometry-only constraint count (0–3), per site and pooled.
No pass/fail cut is applied to the constraint count: choosing one is a PI
decision (ruling 2026-10-01: the C′ draft reports the full cross-tab).

Two deficit measures per (site, n) class, both from per_patch_geometry.csv:
- ``point_deficit``: cell-weighted mean of 1 − share_ge_2h_winter, i.e. the
  share of ground points below the floor, averaged over cells;
- ``cell_deficit``: share of cells whose median ground point is below the floor
  (sun_h_winter_p50 < 2 h).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.brisa_solar.wp06_geometry import SITES

FLOOR_H = 2.0
COLUMNS = ["share_ge_2h_winter", "sun_h_winter_p50", "n_constraints"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def crosstab(table: pd.DataFrame) -> list[dict]:
    """One row per constraint count 0–3; cells without a ground sample drop out."""
    t = table.dropna(subset=["share_ge_2h_winter", "sun_h_winter_p50"])
    n_total = len(t)
    rows = []
    for k in range(4):
        sub = t[t["n_constraints"] == k]
        rows.append({
            "n_constraints": k,
            "cells": int(len(sub)),
            "share_of_cells": float(len(sub) / n_total) if n_total else float("nan"),
            "point_deficit": float((1.0 - sub["share_ge_2h_winter"]).mean()) if len(sub) else float("nan"),
            "cell_deficit": float((sub["sun_h_winter_p50"] < FLOOR_H).mean()) if len(sub) else float("nan"),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/theo/SCL/SCR/MorphoFavela")
    ap.add_argument("--wp06-run-id", default="wp06_geometry_20260915T052604Z")
    ap.add_argument(
        "--in-name", default="per_patch_geometry.csv",
        help="CSV under outputs/<site>/geometry_indicators/ — the wp06 run's --out-name",
    )
    args = ap.parse_args()

    root = Path(args.data_root)
    run_dir = root / "runs" / ("wp07_crosstab_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True)

    inputs, frames, per_site, records = {}, [], {}, []
    for site in SITES:
        path = root / "outputs" / site / "geometry_indicators" / args.in_name
        inputs[str(path.relative_to(root))] = _sha256(path)
        table = pd.read_csv(path, usecols=COLUMNS)
        frames.append(table)
        per_site[site] = {"cells_total": int(len(table)), "rows": crosstab(table)}
        records += [{"site": site, **r} for r in per_site[site]["rows"]]
    pooled = pd.concat(frames, ignore_index=True)
    per_site["pooled"] = {"cells_total": int(len(pooled)), "rows": crosstab(pooled)}
    records += [{"site": "pooled", **r} for r in per_site["pooled"]["rows"]]

    pd.DataFrame(records).to_csv(run_dir / "crosstab.csv", index=False)
    summary = {
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "floor_h": FLOOR_H,
        "wp06_run_id": args.wp06_run_id,
        "inputs_sha256": inputs,
        "per_site": per_site,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
