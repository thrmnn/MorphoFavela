"""Phase 3: migrate per_patch_indicators.csv to rectangular-domain v1 schema.

Augments ``outputs/{site}/cfd_analysis/per_patch_indicators.csv`` with the
new columns from ``outputs/comparative/cfd_methodology/audit_v1.csv`` and
drops the deprecated cylindrical-domain column ``blocken_radius_required``.

The audit script is read-only; this is the writer that lands the new
fields into the canonical per-patch indicator file. Idempotent: re-running
on an already-migrated CSV is a no-op (existing new columns are refreshed
from audit_v1.csv).

Run::

    python scripts/migrate_indicators_rectangular_v1.py            # dry-run
    python scripts/migrate_indicators_rectangular_v1.py --apply    # write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITES = ["vidigal", "riodaspedras", "rocinha", "complexo_do_alemao", "maré"]
AUDIT_CSV = PROJECT_ROOT / "outputs" / "comparative" / "cfd_methodology" / "audit_v1.csv"

# Cylindrical-domain artifact to drop. blocken_ok and cfd_domain_radius
# are listed in the original migration prompt but never actually exist in
# the indicator files — left out of DEPRECATED to avoid noisy warnings.
DEPRECATED = ["blocken_radius_required"]

WIND_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
NEW_COLUMNS = [
    "lambda_p_patch",
    *[f"lambda_f_{d}" for d in WIND_DIRS],
    "lambda_f_mean",
    "lambda_f_max",
    "lambda_f_max_dir",
    "H_mean",  # already in source, but rewritten consistently from audit
    "domain_upstream_m",
    "domain_downstream_m",
    "domain_lateral_m",
    "domain_top_m",
    "domain_blockage_frontal_m2",
    "domain_blockage_cross_section_m2",
    "domain_blockage_ratio",
    "domain_blockage_ok",
    "lambda_f_blockage_diag",
    "source_data_required_m",
    "source_data_extent_m",
    "source_data_ok",
    "eligible",
]


def _migrate_one(audit: pd.DataFrame, site: str, apply: bool) -> tuple[int, int]:
    path = PROJECT_ROOT / "outputs" / site / "cfd_analysis" / "per_patch_indicators.csv"
    if not path.exists():
        print(f"  [SKIP] {site}: indicator file missing at {path}")
        return 0, 0

    df = pd.read_csv(path)
    n_before = len(df.columns)

    # Drop deprecated columns (silently if absent).
    drop_cols = [c for c in DEPRECATED if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    site_audit = audit[audit["site"] == site].copy()
    if site_audit.empty:
        print(f"  [WARN] {site}: no rows in audit_v1.csv — skipping")
        return 0, 0

    # Drop any existing NEW_COLUMNS so the merge is idempotent.
    present_new = [c for c in NEW_COLUMNS if c in df.columns]
    if present_new:
        df = df.drop(columns=present_new)

    audit_cols = ["patch_id"] + [c for c in NEW_COLUMNS if c in site_audit.columns]
    df = df.merge(site_audit[audit_cols], on="patch_id", how="left")

    n_after = len(df.columns)
    n_added = n_after - n_before + len(drop_cols)
    n_eligible = int(df["eligible"].sum()) if "eligible" in df.columns else 0

    print(
        f"  {site:24s}  rows={len(df):3d}  cols {n_before}→{n_after}  "
        f"+{n_added} new, -{len(drop_cols)} deprecated  eligible={n_eligible}/{len(df)}"
    )

    if apply:
        df.to_csv(path, index=False)
    return n_added, n_eligible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the migrated CSV in place. Without this flag the script is a dry-run.",
    )
    args = parser.parse_args()

    if not AUDIT_CSV.exists():
        print(f"ERROR: audit file missing at {AUDIT_CSV}", file=sys.stderr)
        print("Run scripts/audit_rectangular_domain.py first.", file=sys.stderr)
        return 1

    audit = pd.read_csv(AUDIT_CSV)
    expected_cols = {"site", "patch_id", *NEW_COLUMNS}
    missing = expected_cols - set(audit.columns)
    if missing:
        print(f"ERROR: audit file is missing columns: {sorted(missing)}", file=sys.stderr)
        return 1

    print(f"  audit:  {AUDIT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  rows:   {len(audit)}")
    print(f"  mode:   {'APPLY (writing)' if args.apply else 'DRY-RUN'}")
    print()

    total_added, total_eligible = 0, 0
    for site in SITES:
        added, elig = _migrate_one(audit, site, apply=args.apply)
        total_added += added
        total_eligible += elig

    print()
    print(f"  total eligible: {total_eligible}/{len(audit)}")
    if not args.apply:
        print()
        print("  DRY-RUN — no files written. Re-run with --apply to commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
