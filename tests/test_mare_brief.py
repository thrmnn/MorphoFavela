"""Gate for docs/briefs/mare/ (see docs/mare_brief_spec.md):

(a) every ${id} in the template resolves against mare_numbers.json (except
    ${pi_contact}, deliberately left unfilled for the PI);
(b) no digit sequence in the template outside placeholders — with three
    narrow, documented exceptions: morphotype codes T0-T5 (taxonomy labels,
    not measured numbers), a citation year in parentheses e.g. "(1988)",
    and the spec-mandated literal "Point 26" (Athens Charter locator);
(c) figure_manifest.json lists only allowed classes and no excluded
    basename patterns (slope, aspect, per-building height, risk, tb, cfd);
(d) the rendered (filled) markdown has no banned tokens per
    scripts/lint_p1_tokens.py's scanner, and none of "deficit",
    "formal city", "WHO";
(e) the PDF exists and has <= 8 pages.

Skips cleanly when the real outputs/ tree is absent (this repo's outputs/
and data/ are gitignored; a worktree checkout has neither).
"""
from __future__ import annotations

import importlib.util
import json
import re
import string
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BRIEF_DIR = ROOT / "docs" / "briefs" / "mare"
OUTPUTS_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela/outputs")

# Build into a throwaway copy of the brief directory: the builder writes next to
# its own file (HERE), and rebuilding the TRACKED pdf/json from a test run left
# the repo dirty after every full suite (weasyprint/matplotlib bytes differ run to run).
BUILD_DIR = Path(tempfile.mkdtemp(prefix="mare_brief_test_")) / "mare"
shutil.copytree(BRIEF_DIR, BUILD_DIR)

os.environ["MORPHOFAVELA_ROOT"] = str(ROOT)
sys.path.insert(0, str(BUILD_DIR))
import build_brief  # noqa: E402
import collect_numbers  # noqa: E402
from numbers_format import format_entry  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LINT_P1 = _load_module("lint_p1_tokens", ROOT / "scripts" / "lint_p1_tokens.py")

SRC_MD = BRIEF_DIR / "mare_morphology_brief.src.md"
NUMBERS_JSON = BUILD_DIR / "mare_numbers.json"
MANIFEST_JSON = BUILD_DIR / "figure_manifest.json"
PDF = BUILD_DIR / "mare_morphology_brief.pdf"
TRACKED_PDF = BRIEF_DIR / "mare_morphology_brief.pdf"

EXCLUDED_BASENAME_PATTERNS = ["slope", "aspect", "risk", "tb", "cfd"]
# per-building height maps are distinguished from the allowed *aggregated*
# H_mean grid map by an explicit "per_building" / "per-building" marker
EXCLUDED_BASENAME_REGEXES = [re.compile(p, re.IGNORECASE) for p in EXCLUDED_BASENAME_PATTERNS] + [
    re.compile(r"per.?building.*height|height.*per.?building", re.IGNORECASE),
    re.compile(r"per.?building", re.IGNORECASE),
]

ALLOWED_FIGURE_CLASSES_PREFIXES = (
    "band-classed map, freshly rendered",
    "distribution (histogram), freshly rendered",
    "wind rose, freshly rendered",
    "publishable (already band-classed",
)


@pytest.fixture(scope="module")
def built():
    if not OUTPUTS_ROOT.exists():
        pytest.skip(f"outputs tree absent: {OUTPUTS_ROOT}")
    try:
        rc = build_brief.build(OUTPUTS_ROOT)
    except collect_numbers.MissingSource as e:
        pytest.skip(f"missing source file: {e}")
    if rc == 2:
        pytest.skip("build_brief skipped: a source of record is missing on this checkout")
    assert rc == 0, "build_brief failed (pandoc/weasyprint)"
    if not NUMBERS_JSON.exists():
        pytest.skip("mare_numbers.json not produced (collect_numbers SKIP)")
    return json.loads(NUMBERS_JSON.read_text())


TEMPLATE_TEXT = SRC_MD.read_text()
PLACEHOLDER_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def test_a_all_placeholders_resolve(built):
    ids_in_template = set(PLACEHOLDER_RE.findall(TEMPLATE_TEXT))
    assert "pi_contact" in ids_in_template, "template must keep ${pi_contact} as a literal placeholder"
    required = ids_in_template - {"pi_contact"}
    missing = required - set(built.keys())
    assert not missing, f"template placeholders with no mare_numbers.json entry: {sorted(missing)}"
    # and every one actually formats without raising
    for id_ in required:
        format_entry(built[id_])
    # confirm the built PDF/markdown really did leave ${pi_contact} unfilled
    filled = string.Template(TEMPLATE_TEXT).substitute(
        {k: format_entry(v) for k, v in built.items() if k != "pi_contact"}
        | {"pi_contact": "${pi_contact}"}
    )
    assert "${pi_contact}" in filled


def test_b_no_typed_numbers_outside_placeholders():
    stripped = PLACEHOLDER_RE.sub("", TEMPLATE_TEXT)
    stripped = re.sub(r"\bT[0-5]\b", "", stripped)  # morphotype taxonomy codes
    stripped = re.sub(r"\(\d{4}\)", "", stripped)  # citation years, e.g. "(1988)"
    stripped = stripped.replace("Point 26", "")  # spec-mandated Athens Charter locator
    hits = [
        (stripped[:m.start()].count("\n") + 1, m.group(0))
        for m in re.finditer(r"[0-9]+", stripped)
    ]
    assert not hits, f"typed number(s) outside placeholders: {hits}"


def test_c_figure_manifest_classes_and_basenames(built):
    if not MANIFEST_JSON.exists():
        pytest.skip("figure_manifest.json not produced")
    manifest = json.loads(MANIFEST_JSON.read_text())
    assert manifest, "figure_manifest.json is empty"
    for entry in manifest:
        assert entry["class"].startswith(ALLOWED_FIGURE_CLASSES_PREFIXES), (
            f"disallowed figure class: {entry['class']!r} for {entry['file']}"
        )
        assert entry.get("basemap") is False
        assert entry.get("coordinate_ticks") is False
        for rx in EXCLUDED_BASENAME_REGEXES:
            assert not rx.search(entry["file"]), (
                f"figure {entry['file']} matches excluded basename pattern {rx.pattern!r}"
            )


BANNED_PLAIN_WORDS = ["deficit", "formal city", "who"]


def test_d_no_banned_tokens(built):
    filled = build_brief.fill_template(built)
    for word in ["deficit", "formal city"]:
        assert word not in filled.lower(), f"banned word present: {word!r}"
    # "WHO" as a standalone word (not inside e.g. "WHOLE"/"whole")
    assert not re.search(r"\bWHO\b", filled), "banned token 'WHO' present (use Athens Charter, Point 26)"

    tmp_path = BUILD_DIR / "_test_tmp_lint_target.md"
    tmp_path.write_text(filled)
    try:
        hits = LINT_P1._scan_lines(filled.split("\n"), tmp_path.name)
    finally:
        tmp_path.unlink(missing_ok=True)
    assert not hits, f"lint_p1_tokens hits in rendered brief: {hits}"


def test_e_pdf_exists_and_page_count(built):
    assert PDF.exists(), f"missing {PDF}"
    assert TRACKED_PDF.exists(), f"the committed brief PDF is missing: {TRACKED_PDF}"
    n_pages = _pdf_page_count(PDF)
    assert n_pages is not None, "could not determine PDF page count (no pypdf, no pdfinfo)"
    assert n_pages <= 8, f"PDF has {n_pages} pages, spec caps at 8"


def _pdf_page_count(pdf_path: Path) -> int | None:
    try:
        import pypdf

        return len(pypdf.PdfReader(str(pdf_path)).pages)
    except ImportError:
        pass
    try:
        result = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return None
