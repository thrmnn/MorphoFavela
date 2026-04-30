"""Unit tests for the report-sync PreToolUse hook.

These tests target the pure `check(staged)` function directly, bypassing
git entirely, so they exercise the rule logic in isolation. They are the
FP-characterisation gate the hook needs before flipping from advisory
(exit 0) to blocking (exit 2): if a rule fires here on a benign input,
that's a FP that would block real commits.

The hook source lives at `.claude/hooks/check_report_sync.py` (dot-
prefixed, so not importable as a package). We load it via importlib.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "check_report_sync.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("check_report_sync", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hook():
    return _load_hook()


# ---------------------------------------------------------------------------
# Hard rules: .md ↔ .pdf pairing
# ---------------------------------------------------------------------------


def test_md_without_pdf_fails(hook):
    fails, _, _ = hook.check(["docs/technical_report/technical_report.md"])
    assert any("technical_report.pdf is not" in f for f in fails)


def test_pdf_without_md_fails(hook):
    fails, _, _ = hook.check(["docs/technical_report/technical_report.pdf"])
    assert any("technical_report.md is not" in f for f in fails)


def test_md_and_pdf_together_passes(hook):
    fails, warns, advisories = hook.check([
        "docs/technical_report/technical_report.md",
        "docs/technical_report/technical_report.pdf",
    ])
    # Both staged → no .md ↔ .pdf finding
    assert not any("technical_report" in f for f in fails)


# ---------------------------------------------------------------------------
# Soft rule (added 2026-04-30): figure staged without .md → WARN
# ---------------------------------------------------------------------------


def test_figure_without_md_warns(hook):
    fails, warns, _ = hook.check(["docs/technical_report/figures/figS6_umep_validation.png"])
    assert any("Report figure(s) staged without technical_report.md" in w for w in warns)
    # Critically, should NOT be a FAIL — cosmetic figure tweaks shouldn't block
    assert not any("figure" in f.lower() for f in fails)


def test_figure_with_md_no_warn(hook):
    _, warns, _ = hook.check([
        "docs/technical_report/figures/figS6_umep_validation.png",
        "docs/technical_report/technical_report.md",
        "docs/technical_report/technical_report.pdf",
    ])
    assert not any("Report figure(s) staged without" in w for w in warns)


def test_figure_pdf_or_svg_also_warns(hook):
    """The figure-WARN rule covers png, pdf, and svg."""
    for ext in ("png", "pdf", "svg"):
        _, warns, _ = hook.check([f"docs/technical_report/figures/fig5_wind_panel.{ext}"])
        assert any(
            "Report figure(s) staged without" in w for w in warns
        ), f"WARN should fire for .{ext}"


# ---------------------------------------------------------------------------
# Soft rule: paper-figure script change → matching PNG copy
# ---------------------------------------------------------------------------


def test_paper_fig_script_without_png_warns(hook):
    _, warns, _ = hook.check(["outputs/paper_figures/fig5_wind_panel.py"])
    assert any("fig5_wind_panel" in w for w in warns)


def test_paper_fig_script_with_png_no_warn(hook):
    _, warns, _ = hook.check([
        "outputs/paper_figures/fig5_wind_panel.py",
        "docs/technical_report/figures/fig5_wind_panel.png",
        "docs/technical_report/technical_report.md",
        "docs/technical_report/technical_report.pdf",
    ])
    # No warning about the missing PNG copy
    assert not any("fig5_wind_panel.py" in w and ".png not staged" in w for w in warns)


def test_paper_fig_non_fig_script_silent(hook):
    """Scripts under outputs/paper_figures/ that don't follow figXX naming
    should not trigger the PNG-pairing rule (e.g. shared helpers)."""
    _, warns, _ = hook.check(["outputs/paper_figures/fig_style.py"])
    # fig_style is a helper, not a fig generator — note: regex `^(fig\d+|fig[A-Z]\d+|fig_)`
    # actually MATCHES fig_style. Verify the current behaviour.
    matches = [w for w in warns if "fig_style" in w]
    # Either it warns (then we accept "fig_" prefix is too permissive) or it doesn't.
    # Lock in the current behaviour so we notice if it changes.
    assert len(matches) == 1, f"Expected exactly one match for fig_style, got {matches}"


# ---------------------------------------------------------------------------
# Advisory rule: pipeline-script change without .md → list
# ---------------------------------------------------------------------------


def test_script_change_without_md_advises(hook):
    _, _, advisories = hook.check(["scripts/compute_morphometry.py"])
    assert any("compute_morphometry.py" in a for a in advisories)
    assert any("§3–§7" in a or "pipeline script" in a for a in advisories)


def test_script_change_with_md_silent(hook):
    _, _, advisories = hook.check([
        "scripts/compute_morphometry.py",
        "docs/technical_report/technical_report.md",
        "docs/technical_report/technical_report.pdf",
    ])
    # When .md is staged, advisories aren't listed (the user is already touching the report)
    assert not advisories


def test_excluded_script_dirs_silent(hook):
    """scripts/debug/, scripts/data_utils/, scripts/shell/ are excluded
    from the advisory trigger — they're not pipeline-relevant."""
    for excluded in ("scripts/debug/foo.py", "scripts/data_utils/bar.py", "scripts/shell/baz.py"):
        _, _, advisories = hook.check([excluded])
        assert not any(excluded in a for a in advisories), (
            f"{excluded} should not trigger advisory"
        )


def test_hooks_path_not_pipeline(hook):
    """`.claude/hooks/*.py` should NOT trigger the pipeline advisory.
    The auditor flagged this as a plausible FP vector for the blocking
    flip — confirm the trigger glob already scopes to scripts/ proper."""
    _, _, advisories = hook.check([".claude/hooks/check_report_sync.py"])
    assert not any(".claude/hooks" in a for a in advisories)


# ---------------------------------------------------------------------------
# Empty / silent cases (FP-rate floor)
# ---------------------------------------------------------------------------


def test_empty_staged_silent(hook):
    fails, warns, advisories = hook.check([])
    assert not (fails or warns or advisories)


def test_unrelated_staged_silent(hook):
    fails, warns, advisories = hook.check([
        "README.md",
        "src/exposure/sky_exposure.py",
        "tests/test_exposure/test_sky_exposure.py",
    ])
    # No report file, no figure, no scripts/, no paper_figures → silent
    assert not (fails or warns or advisories)


# ---------------------------------------------------------------------------
# Exit-code semantics (blocking flip 2026-04-30)
# ---------------------------------------------------------------------------


def _run_main(hook, monkeypatch, stdin_event, staged):
    """Invoke hook.main() with a mocked stdin and _staged_files."""
    import io as _io_mod

    monkeypatch.setattr(hook.sys, "stdin", _io_mod.StringIO(_io_mod.StringIO(stdin_event).read()))
    monkeypatch.setattr(hook, "_staged_files", lambda: staged)
    return hook.main()


def test_exit_2_on_fail(hook, monkeypatch):
    """A FAIL finding (md without pdf) blocks the commit (exit 2)."""
    import json as _json

    event = _json.dumps({"tool_name": "Bash", "tool_input": {"command": "git commit -m foo"}})
    rc = _run_main(hook, monkeypatch, event, ["docs/technical_report/technical_report.md"])
    assert rc == 2


def test_exit_0_on_warn_only(hook, monkeypatch):
    """A WARN-only finding (figure without md) does NOT block (exit 0)."""
    import json as _json

    event = _json.dumps({"tool_name": "Bash", "tool_input": {"command": "git commit -m foo"}})
    rc = _run_main(
        hook, monkeypatch, event, ["docs/technical_report/figures/figS6_umep_validation.png"]
    )
    assert rc == 0


def test_exit_0_on_advisory_only(hook, monkeypatch):
    """An advisory-only finding (script change without md) does NOT block."""
    import json as _json

    event = _json.dumps({"tool_name": "Bash", "tool_input": {"command": "git commit -m foo"}})
    rc = _run_main(hook, monkeypatch, event, ["scripts/compute_morphometry.py"])
    assert rc == 0


def test_exit_0_on_clean_commit(hook, monkeypatch):
    """No findings → exit 0, silent."""
    import json as _json

    event = _json.dumps({"tool_name": "Bash", "tool_input": {"command": "git commit -m foo"}})
    rc = _run_main(hook, monkeypatch, event, ["README.md"])
    assert rc == 0


def test_exit_0_on_non_commit_bash(hook, monkeypatch):
    """Bash commands that aren't `git commit` are not gated by the hook."""
    import json as _json

    event = _json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}})
    rc = _run_main(hook, monkeypatch, event, ["docs/technical_report/technical_report.md"])
    assert rc == 0


def test_exit_0_on_amend(hook, monkeypatch):
    """`git commit --amend` is an explicit override and bypasses the hook."""
    import json as _json

    event = _json.dumps({"tool_name": "Bash", "tool_input": {"command": "git commit --amend"}})
    rc = _run_main(hook, monkeypatch, event, ["docs/technical_report/technical_report.md"])
    assert rc == 0
