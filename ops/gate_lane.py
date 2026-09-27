#!/usr/bin/env python3
"""Fast/slow lane dispatcher for MorphoFavela's push gate
(staging_prod_ruling_2026-09-25.md §3 + §7 phase 1).

Classifies the diff about to be pushed by path, picks the matching gate
chain, and runs it. Fast lane (always): pytest + linting. Slow lane (if diff
touches scripts/emit_cockpit.py, scripts/build_project_hub.py,
scripts/audit_hub_graph.py, scripts/build_results_registry.py,
scripts/check_registry.py, or outputs/_hub/**): full chain. A misclassification
must fail stricter, never looser.

    python3 ops/gate_lane.py
    python3 ops/gate_lane.py --self-test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FAST = "fast"
SLOW = "slow"

SLOW_TRIGGERS = {
    "scripts/emit_cockpit.py",
    "scripts/build_project_hub.py",
    "scripts/audit_hub_graph.py",
    "scripts/build_results_registry.py",
    "scripts/check_registry.py",
}

FAST_CMD = (
    "python3 -m pytest tests/ -m \"not integration\" -q && "
    "python3 scripts/lint_p1_columns.py && "
    "python3 scripts/lint_p1_tokens.py"
)
SLOW_CMD = (
    "python3 -m pytest tests/ -m \"not integration\" -q && "
    "python3 scripts/lint_p1_columns.py && "
    "python3 scripts/lint_p1_tokens.py && "
    "[ ! -f scripts/check_registry.py ] || python3 scripts/check_registry.py --self-test"
)


def classify(paths: set[str]) -> str:
    if not paths:
        return SLOW  # no diff read -> ambiguous -> stricter
    # Check if any path triggers slow lane
    for path in paths:
        if path in SLOW_TRIGGERS or path.startswith("outputs/_hub/"):
            return SLOW
    return FAST


def changed_paths(repo_root: Path) -> set[str]:
    """Paths of the diff about to be pushed: staged changes if any exist,
    else commits ahead of the upstream tracking branch, else commits ahead
    of origin/main. Any git failure returns an empty set (-> slow lane)."""
    def run(args: list[str]) -> list[str]:
        r = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return []
        return [ln for ln in r.stdout.splitlines() if ln.strip()]

    staged = run(["diff", "--cached", "--name-only"])
    if staged:
        return set(staged)

    upstream = run(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream:
        return set(run(["diff", "--name-only", "@{u}..HEAD"]))

    ahead = run(["diff", "--name-only", "origin/main...HEAD"])
    return set(ahead)


def self_test() -> int:
    fixtures = [
        ({"scripts/lint_p1_columns.py"}, FAST),
        ({"scripts/emit_cockpit.py"}, SLOW),
        ({"scripts/build_project_hub.py"}, SLOW),
        ({"scripts/audit_hub_graph.py"}, SLOW),
        ({"scripts/build_results_registry.py"}, SLOW),
        ({"scripts/check_registry.py"}, SLOW),
        ({"outputs/_hub/foo.html"}, SLOW),
        ({"tests/test_foo.py"}, FAST),
        ({"docs/some_unrelated_file.md"}, FAST),  # unclassified -> fast
        (set(), SLOW),  # no diff found -> slow
    ]

    fails = []
    for paths, expected in fixtures:
        got = classify(paths)
        if got != expected:
            fails.append((paths, expected, got))

    # Prove the fixtures actually discriminate (RO-73 / gate-contract: a
    # check must prove it can go red before its green is trusted). An
    # "always fast" classifier — the exact failure mode this dispatcher
    # exists to prevent, slow-lane work slipping through as fast — must be
    # caught by at least one fixture.
    always_fast = lambda _paths: FAST  # noqa: E731
    sabotage_caught = any(always_fast(p) != expected for p, expected in fixtures)

    if fails:
        for paths, expected, got in fails:
            print(f"SELF-TEST FAIL: {sorted(paths) or '(empty)'} expected={expected} got={got}", file=sys.stderr)
        return 1
    if not sabotage_caught:
        print("SELF-TEST FAIL: fixtures do not discriminate a looser (always-fast) classifier", file=sys.stderr)
        return 1

    print(f"SELF-TEST PASSED: {len(fixtures)} fixtures, sabotage-detection confirmed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    repo_root = Path(__file__).resolve().parent.parent
    paths = changed_paths(repo_root)
    lane = classify(paths)
    print(f"gate_lane: lane={lane} files={sorted(paths) or '(none — ambiguous)'}")
    cmd = FAST_CMD if lane == FAST else SLOW_CMD
    return subprocess.run(["bash", "-c", cmd], cwd=repo_root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
