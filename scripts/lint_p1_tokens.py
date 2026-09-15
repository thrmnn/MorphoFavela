#!/usr/bin/env python3
"""Guard P1's geometry-only vocabulary against simulation-pipeline language
leaking into its own sources (C′ reframe).

The second rail (docs/wp06_geometry_spec.md §2, "banned-token grep"):
scripts/lint_p1_columns.py stops a banned COLUMN from reaching a P1 table;
this stops a banned WORD from reaching a P1 docstring, comment or output
header. P1 describes geometry — lambda_f_mean >= 0.65, Oke's 1988 threshold —
never in simulation-pipeline vocabulary.

Nine tokens are banned, case-insensitive, whole-word (built from fragments
below so this file's own token catalogue can't self-trigger the scan: a
banned-word list has to name its words, and this file matches its own glob).

An explicit ``# p3-forward-reference`` comment on the SAME line exempts that
one line — a deliberate pointer to work gated behind P3, not a leak.

Run: python3 scripts/lint_p1_tokens.py
Wired into .claude/verify-cmd (local-only: .claude/ is gitignored in this
repo) and the Makefile's `lint-p1` target, next to lint_p1_columns.py.
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# P1 pipeline source: a banned word here means P1's own code/docs/policy talk
# like a simulation, even though P1 never runs one.
P1_SOURCE_GLOBS = [
    "src/brisa_solar/**/*.py",
    "scripts/lint_p1_*.py",
    "docs/p1_column_allowlist.json",
]
# P1's own written artifacts (WP-06's table). Only the header/leading-comment
# lines matter — a banned word could not appear in numeric cell data anyway.
P1_ARTIFACT_GLOBS = ["outputs/*/geometry_indicators/*"]

BANNED_TOKENS = [
    "fl" + "ow",
    "CF" + "D",
    "Open" + "FOAM",
    "age-of-" + "air",
    "ta" + "u",
    chr(0x3C4),  # the Greek letter for the "ta"+"u" entry above, built at runtime so this file's own text never contains it literally
    "AC" + "H",
    "k-ome" + "ga",
    "skim" + "ming",
]
EXEMPT_MARK = "# p3-forward-reference"

_PATTERNS = [(tok, re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE | re.UNICODE)) for tok in BANNED_TOKENS]


def _scan_lines(lines: list[str], rel_path: str) -> list[str]:
    hits = []
    for lineno, line in enumerate(lines, start=1):
        if EXEMPT_MARK in line:
            continue
        for tok, pat in _PATTERNS:
            if pat.search(line):
                hits.append(f"{rel_path}:{lineno}: banned token '{tok}' — {line.strip()[:120]}")
    return hits


def _artifact_header_lines(text: str) -> list[str]:
    """Header + any leading '#'-prefixed comment lines of a P1 artifact."""
    lines = text.split("\n")
    n = 1
    while n < len(lines) and lines[n].startswith("#"):
        n += 1
    return lines[:n]


def main() -> int:
    hits: list[str] = []
    n_src = 0
    for pattern in P1_SOURCE_GLOBS:
        for f in sorted(glob.glob(str(ROOT / pattern), recursive=True)):
            n_src += 1
            text = Path(f).read_text()
            hits.extend(_scan_lines(text.split("\n"), str(Path(f).relative_to(ROOT))))

    n_art = 0
    for pattern in P1_ARTIFACT_GLOBS:
        for f in sorted(glob.glob(str(ROOT / pattern), recursive=True)):
            n_art += 1
            text = Path(f).read_text()
            hits.extend(_scan_lines(_artifact_header_lines(text), str(Path(f).relative_to(ROOT))))

    if hits:
        for h in hits:
            print(f"  FAIL {h}")
        print(f"lint_p1_tokens: {len(hits)} problem(s) — P1 describes geometry only, never in simulation-pipeline vocabulary.")
        return 1
    print(f"lint_p1_tokens: OK ✓ — {n_src} P1 source file(s) and {n_art} P1 artifact(s) clean of "
          f"{len(BANNED_TOKENS)} banned tokens.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
