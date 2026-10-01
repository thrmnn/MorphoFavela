"""P-07 — quality report: coverage mask along OM2, known gaps.

For every variable column in the package's point table, reports how many
points have a valid (non-null) value and lists the point_ids that don't
(join-distance gaps from formvars.py/ventilation.py, or a point off the
edge of a source layer). Also records the PENDING items that are not
computed at all in this version.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

#: Items still owed in this version. None: the former four were descoped.
PENDING_ITEMS: list[str] = []

#: Dropped from v0.1.3 by PI decision om_v013_descope (2026-10-01; still out of scope in v0.2.0) — a
#: deliberate cut, not a gap; candidates for a later version. Listed so the
#: quality report and the data dictionary agree.
DESCOPE_DECISION = "om_v013_descope"
DESCOPED_ITEMS = [
    "sky_view_factor_terrestrial",
    "tree_shade",
    "airborne_vs_terrestrial_comparison",
    "height_change_2024_2026",
]


def coverage_report(df: pd.DataFrame, variable_cols: list[str], extra: dict | None = None) -> dict:
    n = len(df)
    per_column = {}
    for col in variable_cols:
        if col not in df.columns:
            per_column[col] = {"present": False}
            continue
        valid = df[col].notna()
        per_column[col] = {
            "present": True,
            "n_valid": int(valid.sum()),
            "n_total": n,
            "coverage_fraction": float(valid.sum()) / n if n else 0.0,
            "missing_point_ids": df.loc[~valid, "point_id"].tolist() if "point_id" in df.columns else [],
        }
    report = {
        "n_points": n,
        "columns": per_column,
        "pending_items": PENDING_ITEMS,
        "descoped_items": DESCOPED_ITEMS,
        "descoped_by": DESCOPE_DECISION,
    }
    if "route_geometry_flag" in df.columns:
        # PI ruling 2026-09-24 (must-fix 1): count points where the
        # OSM-inferred route falls inside a building or off the street.
        report["route_geometry_flagged_points"] = int(df["route_geometry_flag"].sum())
    if extra:
        # P-10 / P-11 table-level entries (class shares, clock agreement,
        # wind-observation counts): measured by the build, never typed.
        report.update(extra)
    return report


def write_quality_report(df: pd.DataFrame, variable_cols: list[str], out_dir: Path, stem: str = "p07_quality_report",
                         extra: dict | None = None):
    report = coverage_report(df, variable_cols, extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2))

    rows = []
    for col, stats in report["columns"].items():
        if not stats.get("present"):
            rows.append({"variable": col, "present": False, "coverage_fraction": None, "n_valid": None, "n_total": report["n_points"]})
        else:
            rows.append(
                {
                    "variable": col,
                    "present": True,
                    "coverage_fraction": stats["coverage_fraction"],
                    "n_valid": stats["n_valid"],
                    "n_total": stats["n_total"],
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"{stem}.csv", index=False)
    return report
