#!/usr/bin/env python3
"""PreToolUse hook: check report-sync state before a git commit.

Wired in `.claude/settings.json` under `hooks.PreToolUse` with a Bash
matcher. Fires on every Bash call but exits 0 (no-op) unless the
command is a `git commit`.

Encodes the high-confidence subset of the triggers documented in
`CLAUDE.md` "Technical report" — i.e. the rules that are mechanical
enough to check without LLM judgment:

  - `.md` and `.pdf` must be staged together (one without the other = FAIL).
  - For every staged `outputs/paper_figures/*.py`, a matching PNG under
    `docs/technical_report/figures/` should be staged in the same commit.

Triggers that need interpretation (was that `scripts/X.py` change
pipeline-relevant?) are listed as **advisory**, not blocking — the
`report-sync-auditor` agent does the full LLM-backed pass when the
user invokes it explicitly.

Hook exit semantics in Claude Code:
  exit 0           → allow tool, silent
  exit 2 + stderr  → block tool, show stderr to Claude
  other non-zero   → non-blocking error

This hook exits **2 when there is a FAIL finding** (hard rules: .md ↔
.pdf pairing, paper-figure script without matching PNG copy) and **0
otherwise**. WARN and advisory findings are surfaced via stderr but do
not block the commit. The FAIL set is the same set covered by the
unit-test suite at tests/test_hooks/test_check_report_sync.py — that
15-test FP-rate floor is what authorizes the blocking flip.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def _git(args: list[str]) -> str:
    """Run a git command in the repo root and return stdout (empty on failure)."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout if result.returncode == 0 else ""
    except FileNotFoundError:
        return ""


def _staged_files() -> list[str]:
    """Files staged for the in-flight commit (`git diff --cached --name-only`)."""
    out = _git(["diff", "--cached", "--name-only"])
    return [line.strip() for line in out.splitlines() if line.strip()]


# Triggers that *might* need a report update; we surface them as advisory
# rather than block, because deciding whether the change is pipeline-
# relevant requires interpreting the diff.
_ADVISORY_TRIGGER_PATTERNS = [
    (re.compile(r"^scripts/(?!debug/|data_utils/|shell/)[^/]+\.py$"), "§3–§7 (pipeline script)"),
    (re.compile(r"^src/morphometry/.*\.py$"), "§4 (morphometry)"),
    (re.compile(r"^src/morphology_metrics\.py$"), "§4.2 (morphometry indicators)"),
    (re.compile(r"^src/cfd_integration/.*\.py$"), "§7 (CFD integration)"),
    (re.compile(r"^scripts/run_(pilot|campaign)_sampling\.py$"), "§6 (sampling allocation)"),
    (re.compile(r"^data/[^/]+/cfd_results/"), "§7.4, §11 (CFD results ingested)"),
]


# Source paths that count as "feature/fix code" for the tests-with-feature
# advisory. Test files (under tests/) and the report itself are excluded
# automatically. Excluded subtrees mirror the pipeline-trigger glob above.
_CODE_PATH_PATTERN = re.compile(
    r"^(src/.*\.py|scripts/(?!debug/|data_utils/|shell/)[^/]+\.py)$"
)
_TEST_PATH_PATTERN = re.compile(r"^tests/.*\.py$")

# Match `feat(...)` / `feat: ` / `fix(...)` / `fix: ` either at the start
# of a line (heredoc form: `git commit -m "$(cat <<'EOF'\nfeat(...)`)
# or right after an opening quote (inline form: `git commit -m "feat(...)"`).
# The quote/line-start anchor avoids matching "feat" inside prose like
# "docs: explain the recent feat(svf) addition".
_FEAT_FIX_RE = re.compile(r"""(?:^|["'`])(feat|fix)[(:]""", re.MULTILINE)


def check(
    staged: list[str], commit_command: str = ""
) -> tuple[list[str], list[str], list[str]]:
    """Return (fails, warns, advisories) lists of human-readable messages.

    ``commit_command`` is the raw string of the in-flight ``git commit``
    invocation. Used to gate commit-type-sensitive rules (e.g. the
    tests-with-feature advisory only fires on ``feat(...)`` / ``fix(...)``).
    Pass empty string to skip those rules.
    """
    fails: list[str] = []
    warns: list[str] = []
    advisories: list[str] = []

    md = "docs/technical_report/technical_report.md"
    pdf = "docs/technical_report/technical_report.pdf"

    md_in = md in staged
    pdf_in = pdf in staged

    # Hard rule: .md ↔ .pdf must be staged together
    if md_in and not pdf_in:
        fails.append(
            "technical_report.md is staged but technical_report.pdf is not — "
            "rebuild the PDF (`python docs/technical_report/build_pdf.py`) "
            "and stage it in the same commit"
        )
    if pdf_in and not md_in:
        fails.append(
            "technical_report.pdf is staged but technical_report.md is not — "
            "this usually means a stale build artefact got committed; "
            "either include the source change or unstage the PDF"
        )

    # Hard rule: paper-figure script change → matching PNG copy
    paper_figs_changed = [s for s in staged if re.match(r"^outputs/paper_figures/.*\.py$", s)]
    for f in paper_figs_changed:
        # Heuristic: scripts named figXX_*.py produce figures named figXX_*.png
        name = Path(f).stem
        # Only flag if the script name follows the figXX or fig_NAME convention
        if not re.match(r"^(fig\d+|fig[A-Z]\d+|fig_)", name):
            continue
        expected_png = f"docs/technical_report/figures/{name}.png"
        if expected_png not in staged:
            warns.append(
                f"{f} changed but {expected_png} not staged — re-run the figure script "
                f"and copy the PNG into docs/technical_report/figures/ if the figure changed"
            )

    # Soft rule: a figure was regenerated but the .md was not updated.
    # This catches the "stale numbers next to a refreshed figure" pattern
    # — the report cites figures by reference, and a figure rebuild often
    # implies the cited numbers changed.
    figs_changed = [
        s for s in staged if re.match(r"^docs/technical_report/figures/.+\.(png|pdf|svg)$", s)
    ]
    if figs_changed and not md_in:
        warns.append(
            "Report figure(s) staged without technical_report.md: "
            + ", ".join(Path(f).name for f in figs_changed)
            + ". If the figure update changed any numbers cited in the report "
            "(r², slope, n, etc.), update the corresponding §-block. Cosmetic "
            "tweaks (color, labels, dpi) can be ignored."
        )

    # Advisory: list paths that *might* need a report touch
    triggered = []
    for f in staged:
        for pattern, section in _ADVISORY_TRIGGER_PATTERNS:
            if pattern.match(f):
                triggered.append(f"{f} → {section}")
                break

    if triggered and not md_in:
        advisories.append("The following paths *might* require a technical_report.md update:")
        for t in triggered:
            advisories.append(f"  • {t}")
        advisories.append(
            "Run the `report-sync-auditor` agent for a full LLM-backed check, "
            "or explicitly confirm the changes are not pipeline-relevant."
        )

    # Tests-with-feature advisory: feat()/fix() commits that touch src/ or
    # scripts/ should ideally stage a tests/ file in the same commit. We
    # only fire when commit_command suggests a feat/fix (regex match);
    # other types (chore/docs/style/refactor/test) are not covered.
    if commit_command and _FEAT_FIX_RE.search(commit_command):
        code_changed = [s for s in staged if _CODE_PATH_PATTERN.match(s)]
        tests_changed = [s for s in staged if _TEST_PATH_PATTERN.match(s)]
        if code_changed and not tests_changed:
            advisories.append(
                "feat/fix without staged test — confirm intentional or stage "
                "a tests/ file. Code paths changed: "
                + ", ".join(code_changed[:5])
                + ("…" if len(code_changed) > 5 else "")
            )

    return fails, warns, advisories


def main() -> int:
    # Read PreToolUse event from stdin
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # silent no-op on malformed events

    if event.get("tool_name") != "Bash":
        return 0
    command = event.get("tool_input", {}).get("command", "")
    if not command:
        return 0

    # Match `git commit` (and `git commit -m ...`, `git commit -a`, etc.)
    # but NOT `git log --grep "commit"` or other false positives.
    if not re.search(r"\bgit\s+commit\b", command):
        return 0

    # Skip --amend reflows (those are explicit overrides of the last commit)
    if "--amend" in command:
        return 0

    staged = _staged_files()
    if not staged:
        return 0  # nothing staged → nothing to check

    fails, warns, advisories = check(staged, commit_command=command)

    if not (fails or warns or advisories):
        return 0  # silent success

    header = "[report-sync hook] BLOCKING:" if fails else "[report-sync hook] advisory:"
    parts = [header]
    for f in fails:
        parts.append(f"  FAIL: {f}")
    for w in warns:
        parts.append(f"  WARN: {w}")
    for a in advisories:
        parts.append(a)
    print("\n".join(parts), file=sys.stderr)

    # Block on FAIL (hard rules); WARN/advisory surface via stderr only.
    return 2 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
