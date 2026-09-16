"""VENTAXIS — the C' definition of record for P1's second axis
(docs/ventaxis_cprime_spec.md). One test file, three assertions:

(a) docs/ventaxis_canonical.md exists;
(b) it is inside scripts/lint_p1_tokens.py's own P1_SOURCE_GLOBS — checked by
    calling the lint's own glob expansion, never by re-typing the glob;
(c) the WP-07 ledger's _meta.definition_of_record resolves to a file that
    exists.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lint_p1_tokens as lt  # noqa: E402

from src.brisa_solar import wp07_ledger as w  # noqa: E402

DOC = ROOT / "docs" / "ventaxis_canonical.md"


def test_ventaxis_definition_of_record_is_railed_and_wired():
    # (a) the doc exists
    assert DOC.exists(), DOC

    # (b) it is inside the lint's own P1_SOURCE_GLOBS — via the lint's own
    # glob expansion, never a re-typed copy of the pattern
    matched = set()
    for pattern in lt.P1_SOURCE_GLOBS:
        matched.update(glob.glob(str(lt.ROOT / pattern), recursive=True))
    assert str(DOC) in matched, (
        f"{DOC} not covered by any of lt.P1_SOURCE_GLOBS={lt.P1_SOURCE_GLOBS}"
    )

    # (c) the ledger's _meta.definition_of_record resolves to a file that exists
    ledger = w.build_ledger(ROOT)
    dor = ledger["_meta"]["definition_of_record"]
    assert dor, "ledger _meta.definition_of_record is empty"
    for family, rel_path in dor.items():
        target = ROOT / rel_path
        assert target.exists(), f"definition_of_record[{family!r}] -> {target} does not exist"


def test_vertical_constraint_threshold_cannot_diverge_from_the_e2_script():
    """P1 imports the frontal-area-density threshold from its own constants home,
    not from the June E2 script whose identifier names a flow regime. The value
    must stay identical in both places or the June index and the P1 ledger would
    silently describe different cells."""
    from scripts.run_ventilation_index import SKIM_MIN as _e2_value

    from src.brisa_solar.constants import LAMBDA_F_CONSTRAINT_MIN

    assert LAMBDA_F_CONSTRAINT_MIN == _e2_value

    # and the P1 module must not reach into the E2 script for it
    src = (ROOT / "src" / "brisa_solar" / "wp06_geometry.py").read_text()
    assert "LAMBDA_F_CONSTRAINT_MIN" in src
    assert "import SKIM_MIN" not in src and "SKIM_MIN," not in src
