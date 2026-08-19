#!/usr/bin/env python3
"""ADVISORY (never fails the gate): flag module-level ALL-CAPS numeric constants
in validation/audit/check scripts, so a value that SHOULD be derived from the
artifact it validates doesn't quietly calcify into a hardcoded one.

The case study: audit_morphotope_stability.py hardcoded REF_K = 5 for seven
weeks after build_morphotope.py moved on to k=3 — the published bootstrap ARI
kept defending a partition that no longer existed on disk. Nothing failed,
because a stale defence still runs clean.

This is a coarse heuristic, not a proof: RADIUS, N_BOOT, SUBSAMPLE_FRAC are
legitimate fixed hyperparameters, not artifacts to derive. Advisory-only until
it has run for a cycle and the allowlist below is known-complete — a false
positive that blocks a commit is worse than a missed one that gets caught at
cycle-close review.

Annotate a legitimate fixed constant with `# derived-from: <n/a — parameter>`
or add its name to ALLOWLIST below; annotate a value that IS derived elsewhere
with `# derived-from: <path/to/producer.py>` on the same line.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TARGET_GLOBS = ["scripts/audit_*.py", "scripts/*check*.py", "scripts/*validat*.py"]

# Known, reviewed fixed hyperparameters — not artifacts to re-derive.
ALLOWLIST = {
    "RADIUS", "N_BOOT", "SUBSAMPLE_FRAC", "RANDOM_STATE", "SILHOUETTE_SAMPLE",
    "PATCH_DIAMETER_M", "BLOCKAGE_GATE", "SAFETY_MARGIN_M",  # campaign geometry specs, not fitted values
}

CONST_RE = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*=\s*[0-9][0-9.eE+-]*\s*(#.*)?$")


def scan_file(path: Path):
    hits = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        m = CONST_RE.match(line.strip())
        if not m:
            continue
        name, comment = m.group(1), m.group(2) or ""
        if name in ALLOWLIST or "derived-from" in comment:
            continue
        hits.append((i, name, line.strip()))
    return hits


def main():
    findings = []
    seen = set()
    for pattern in TARGET_GLOBS:
        for f in ROOT.glob(pattern):
            if f in seen:
                continue
            seen.add(f)
            for lineno, name, line in scan_file(f):
                findings.append((f.relative_to(ROOT), lineno, name, line))

    if not findings:
        print("check_derived_constants: OK — no un-annotated numeric constants in validation/audit scripts.")
        return 0

    print(f"check_derived_constants: ADVISORY — {len(findings)} un-annotated constant(s):")
    for rel, lineno, name, line in findings:
        print(f"  {rel}:{lineno}  {line}")
    print(
        "  Each is either a fixed hyperparameter (add to ALLOWLIST in this script) or a\n"
        "  value that should be derived from its producer (add `# derived-from: <path>`\n"
        "  and, ideally, actually derive it — see audit_morphotope_stability.py's fix)."
    )
    return 0  # advisory: never blocks. Promote to hard-fail once the allowlist is stable.


if __name__ == "__main__":
    sys.exit(main())
