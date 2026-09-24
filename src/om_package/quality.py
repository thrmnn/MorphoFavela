"""P-07 — quality report: coverage mask along OM2, known gaps.

For every variable column in the package's point table, reports how many
points have a valid (non-null) value and lists the point_ids that don't
(join-distance gaps from formvars.py/ventilation.py, or a point off the
edge of a source layer). Also records the PENDING items that are not
computed at all in v0.1.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

#: Not computed anywhere in v0.1 — no column exists for these; listed here
#: so the quality report and the data dictionary agree.
PENDING_ITEMS = [
    "sky_view_factor_terrestrial",
    "building_shade_per_5min",
    "tree_shade",
    "airborne_vs_terrestrial_comparison",
    "height_change_2024_2026",
]


def coverage_report(df: pd.DataFrame, variable_cols: list[str]) -> dict:
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
    return {
        "n_points": n,
        "columns": per_column,
        "pending_items": PENDING_ITEMS,
    }


def write_quality_report(df: pd.DataFrame, variable_cols: list[str], out_dir: Path, stem: str = "p07_quality_report"):
    report = coverage_report(df, variable_cols)
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
