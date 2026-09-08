#!/usr/bin/env python3
"""Guard P1's geometry-only ventilation axis against CFD leakage (C′ reframe).

Three checks, all reading the policy from docs/p1_column_allowlist.json so the
column lists live in exactly one place:

  1. schema drift  — the source CFD table still has the 52-column header the
     policy was written against (a renamed/added column silently escapes an
     allowlist written for the old schema).
  2. artifact leak — no P1-bound table carries a banned column.
  3. code leak     — no banned column name appears in P1 pipeline source.

Run: python3 scripts/lint_p1_columns.py [--strict]
Wired into .claude/verify-cmd, so it runs before every push.
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "docs" / "p1_column_allowlist.json"

# Where P1's own artifacts live. A banned column here is a real leak.
P1_ARTIFACT_GLOBS = [
    "outputs/*/geometry_indicators/*.csv",
    "outputs/p1_cprime/**/*.csv",
    "runs/*/artifacts/*.csv",
]
# P1 pipeline source. A banned name here means code is reaching for CFD output.
P1_SOURCE_GLOBS = ["src/brisa_solar/**/*.py"]


def fail(msg: str) -> None:
    print(f"  FAIL {msg}")


def main() -> int:
    pol = json.loads(POLICY.read_text())
    legal = set(pol["p1_legal"])
    banned = set(pol["p1_banned_cfd_output"]["columns"]) | set(
        pol["p1_banned_cfd_setup"]["columns"]
    )
    errors = 0

    # 1. schema drift against the source table the policy describes
    src = pol["_source_schema"]
    seen_hashes = set()
    files = sorted(glob.glob(str(ROOT / "outputs/*/cfd_analysis/per_patch_indicators.csv")))
    for f in files:
        header = Path(f).read_text().split("\n", 1)[0].strip()
        seen_hashes.add(hashlib.md5(header.encode()).hexdigest()[:8])
        cols = header.split(",")
        unknown = [c for c in cols if c not in legal and c not in banned]
        if unknown:
            fail(f"{Path(f).relative_to(ROOT)}: {len(unknown)} column(s) classified by neither "
                 f"list — classify them in the policy before P1 code touches this file: {unknown}")
            errors += 1
    if files and seen_hashes != {src["header_md5_8"]}:
        fail(f"source schema drift: header md5 {sorted(seen_hashes)} != policy "
             f"{src['header_md5_8']} — re-verify the P1/P3 column split, then update the policy.")
        errors += 1

    # 2. banned columns in P1-bound artifacts
    for pattern in P1_ARTIFACT_GLOBS:
        for f in glob.glob(str(ROOT / pattern), recursive=True):
            header = Path(f).read_text().split("\n", 1)[0].strip()
            hit = sorted(banned.intersection(header.split(",")))
            if hit:
                fail(f"{Path(f).relative_to(ROOT)}: CFD-derived column(s) in a P1 artifact: {hit}")
                errors += 1

    # 3. banned column names referenced in P1 pipeline source
    for pattern in P1_SOURCE_GLOBS:
        for f in glob.glob(str(ROOT / pattern), recursive=True):
            text = Path(f).read_text()
            for name in sorted(banned):
                if re.search(rf"\b{re.escape(name)}\b", text):
                    fail(f"{Path(f).relative_to(ROOT)}: references banned column '{name}'")
                    errors += 1

    n_art = sum(len(glob.glob(str(ROOT / p), recursive=True)) for p in P1_ARTIFACT_GLOBS)
    n_src = sum(len(glob.glob(str(ROOT / p), recursive=True)) for p in P1_SOURCE_GLOBS)
    if errors:
        print(f"lint_p1_columns: {errors} problem(s) — P1 must stay geometry-only (C′).")
        return 1
    print(f"lint_p1_columns: OK ✓ — {len(files)} source table(s) on the pinned 52-column schema; "
          f"{n_art} P1 artifact(s) and {n_src} P1 source file(s) clean of "
          f"{len(banned)} banned CFD columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
