"""Runs scripts/checks/check_derived_constants.py — advisory, never fails the
build (see that script's own docstring). This test just exercises it so a
broken check doesn't silently stop running; the audit_morphotope_stability.py
k=5/k=3 case study is why this check exists at all."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_check_derived_constants_runs_clean():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "checks" / "check_derived_constants.py")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
