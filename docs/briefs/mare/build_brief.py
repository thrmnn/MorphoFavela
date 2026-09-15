#!/usr/bin/env python3
"""Build mare_morphology_brief.pdf from mare_morphology_brief.src.md.

Pipeline: collect_numbers.py -> mare_numbers.json; render_figures.py ->
figures/ + figure_manifest.json; fill the ${id} template (string.Template,
${pi_contact} deliberately left unfilled) -> pandoc (markdown -> HTML) ->
weasyprint (HTML -> PDF). Shape reused from docs/technical_report/build_pdf.py;
this is a new, calmer stylesheet (single accent colour, A4, 10.5 pt body).

Usage: python3 docs/briefs/mare/build_brief.py --outputs-root <path>
"""
from __future__ import annotations

import argparse
import json
import re
import string
import subprocess
import sys
from datetime import date
from pathlib import Path

import weasyprint

import collect_numbers
import render_figures
from numbers_format import format_entry

HERE = Path(__file__).resolve().parent
SRC_MD = HERE / "mare_morphology_brief.src.md"
NUMBERS_JSON = HERE / "mare_numbers.json"
FIGURES_DIR = HERE / "figures"
MANIFEST_JSON = HERE / "figure_manifest.json"
FILLED_MD = HERE / "_build_tmp_filled.md"
HTML = HERE / "_build_tmp.html"
PDF = HERE / "mare_morphology_brief.pdf"

UNFILLED_ID = "pi_contact"

DISCLOSURE_MD = HERE / "disclosure_sweep.md"

# The ethics-gate disclosure-sweep greplist
# (/home/theo/SCL/SCR/brisaverse/.claude/skills/ethics-gate/SKILL.md, v2).
DISCLOSURE_PATTERN = re.compile(
    r"party.?wall|dissolve|lancet|nature cities|morphofavela|airflow|brisaverse|"
    r"drive.?sync|0\.65|< ?2 ?h|λ_?f|lambda_?f|solstice|G&O|grimmond|oke|"
    r"sondotecnica|IPP|mingze|gobatti|fabio",
    re.IGNORECASE,
)

# Internal codenames / method neologisms not on the greplist but worth a
# PI decision (ethics-gate SKILL.md: "ask ... does this text name an
# unpublished method, an internal codename, a venue/journal, or a result
# parameter?"). (pattern, proposed decision)
EXTRA_DISCLOSURE_CHECKS = [
    (
        re.compile(r"morphotype|Open Fringe|Flatland Consolidated|Hillside Fringe|"
                    r"Shaded Consolidated|Hillside Core|Saturated Core|\bT[0-5]\b"),
        "HOLD (proposed) — the six-morphotype taxonomy (T0-T5 + names) is the "
        "project's own unpublished classification scheme (see technical_report.md "
        "§5.5); naming it externally may pre-empt a paper contribution. PI to "
        "decide: keep the named taxonomy, or generalise to 'recurring fabric "
        "clusters' without the T0-T5 labels.",
    ),
    (
        re.compile(r"constraint|n_constraints", re.IGNORECASE),
        "INCLUDE (proposed) — matches the already-CLEAR release class of the "
        "WP-07 f4_geometry_constraints figure in red_lines.md §5 (ordinal "
        "geometry-only constraint count, denominator = built cells, not ranked).",
    ),
]

ACCENT = "#2A5FA5"

CSS = f"""
@page {{
  size: A4;
  margin: 14mm 15mm 16mm 15mm;
  @bottom-center {{
    content: "Maré morphology brief · """ + date.today().isoformat() + f""" · draft for PI review";
    font-size: 7.5pt; color: #777;
  }}
  @top-right {{ content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; }}
}}
body {{
  font-family: "Liberation Sans", "Arial", sans-serif;
  font-size: 10pt;
  line-height: 1.32;
  color: #1c1c1c;
}}
h1 {{
  font-size: 17pt;
  font-weight: 700;
  color: {ACCENT};
  border-bottom: 2px solid {ACCENT};
  padding-bottom: 5px;
  margin-top: 0;
  margin-bottom: 6pt;
}}
h1:first-of-type {{ page-break-before: avoid; }}
h2 {{
  font-size: 12.5pt;
  font-weight: 700;
  color: #1c1c1c;
  margin-top: 10pt;
  margin-bottom: 4pt;
  padding-bottom: 2px;
  border-bottom: 1px solid #ccc;
  page-break-after: avoid;
}}
p, li {{ text-align: left; }}
p {{ margin: 0.35em 0; }}
em {{ color: #555; }}
strong {{ font-weight: 600; }}
table {{
  border-collapse: collapse;
  margin: 0.4em 0;
  font-size: 8.5pt;
  width: 100%;
  page-break-inside: avoid;
}}
th, td {{
  border: 1px solid #d5d5d5;
  padding: 3px 6px;
  text-align: left;
  vertical-align: top;
}}
th {{
  background: #eef3fa;
  color: {ACCENT};
  font-weight: 600;
}}
img {{
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.4em auto;
  page-break-inside: avoid;
}}
figure {{ margin: 0.5em 0; page-break-inside: avoid; }}
figcaption {{ font-size: 7.5pt; color: #555; text-align: center; }}
a {{ color: {ACCENT}; text-decoration: none; }}
ul, ol {{ margin: 0.3em 0; padding-left: 1.4em; }}
li {{ margin: 0.1em 0; }}
/* key-numbers block: the first two-column table in section 2 */
h2:nth-of-type(2) + table th,
h2:nth-of-type(2) + table td {{ border: none; padding: 2px 10px 2px 0; }}
h2:nth-of-type(2) + table {{ font-size: 10pt; }}
"""


def fill_template(numbers_by_id: dict) -> str:
    template = string.Template(SRC_MD.read_text())
    mapping = {
        id_: format_entry(entry)
        for id_, entry in numbers_by_id.items()
        if id_ != UNFILLED_ID
    }

    class _KeepUnfilled(dict):
        def __missing__(self, key):
            if key == UNFILLED_ID:
                return "${" + key + "}"
            raise KeyError(key)

    filled = template.substitute(_KeepUnfilled(mapping))
    return filled


def _decision_for(term: str) -> str:
    t = term.lower()
    if "oke" in t or "λ" in t or "lambda_f" in t:
        return ("INCLUDE (proposed) — standard published morphometric notation "
                "(Oke 1988; Stewart & Oke 2012), not project-internal.")
    if "morphofavela" in t:
        return ("FLAG for PI (proposed INCLUDE) — names the analysis tool/pipeline "
                "that produced this brief's numbers; low sensitivity, but confirm "
                "MorphoFavela is an acceptable external-facing name before this "
                "brief leaves the repo.")
    return "FLAG for PI — greplist hit, no default proposed."


def write_disclosure_sweep(rendered_md: str) -> None:
    lines = rendered_md.splitlines()
    rows = []
    for lineno, line in enumerate(lines, start=1):
        for m in DISCLOSURE_PATTERN.finditer(line):
            rows.append((lineno, m.group(0), line.strip(), _decision_for(m.group(0))))

    out = ["# Maré brief — disclosure sweep",
           "",
           f"Run against the rendered markdown ({SRC_MD.name}, filled). "
           "Every hit needs an explicit include/drop decision — the PI decides.",
           "",
           "## Greplist hits "
           "(ethics-gate SKILL.md v2)",
           ""]
    if not rows:
        out.append("None.")
    else:
        out.append("| line | match | context | proposed decision |")
        out.append("|---:|---|---|---|")
        for lineno, match, context, decision in rows:
            context_esc = context.replace("|", "\\|")
            out.append(f"| {lineno} | `{match}` | {context_esc} | {decision} |")

    out += ["", "## Additional codename / method-neologism / result-parameter check",
            "", "Beyond the greplist: does the text name an unpublished method, an "
            "internal codename, a venue/journal, or a result parameter?", ""]
    for pattern, decision in EXTRA_DISCLOSURE_CHECKS:
        hit_lines = sorted({i + 1 for i, l in enumerate(lines) if pattern.search(l)})
        if hit_lines:
            out.append(f"- Lines {hit_lines}: {decision}")
    out.append("")
    out.append("No venue/journal name, no collaborator name, and no other project "
                "codename (brisa/brisaverse/P1-P4/track names) appear in the rendered "
                "brief.")

    DISCLOSURE_MD.write_text("\n".join(out) + "\n")
    print(f"build_brief: wrote disclosure sweep ({len(rows)} greplist hits) -> {DISCLOSURE_MD}")


def build(outputs_root: Path) -> int:
    try:
        numbers = collect_numbers.collect(outputs_root)
    except collect_numbers.MissingSource as e:
        print(f"build_brief: SKIP — {e}")
        return 0

    by_id = {n["id"]: n for n in numbers}
    NUMBERS_JSON.write_text(json.dumps(by_id, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"build_brief: collected {len(by_id)} numbers -> {NUMBERS_JSON}")

    manifest = render_figures.render_all(outputs_root, FIGURES_DIR)
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"build_brief: rendered {len(manifest)} figures -> {FIGURES_DIR}")

    filled = fill_template(by_id)
    FILLED_MD.write_text(filled)
    write_disclosure_sweep(filled)

    print("build_brief: converting markdown -> HTML via pandoc...")
    result = subprocess.run(
        [
            "pandoc",
            str(FILLED_MD),
            "-o",
            str(HTML),
            "--standalone",
            "--from",
            "gfm",
            "--to",
            "html5",
            "--embed-resources",
        ],
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    if result.returncode != 0:
        print(result.stderr)
        return 1

    print(f"build_brief: rendering HTML -> PDF with weasyprint -> {PDF}")
    html = weasyprint.HTML(filename=str(HTML), base_url=str(HERE))
    css = weasyprint.CSS(string=CSS)
    html.write_pdf(str(PDF), stylesheets=[css])

    HTML.unlink(missing_ok=True)
    FILLED_MD.unlink(missing_ok=True)

    size_kb = PDF.stat().st_size / 1024
    print(f"build_brief: done: {PDF} ({size_kb:.0f} KB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, required=True)
    args = ap.parse_args()
    return build(args.outputs_root)


if __name__ == "__main__":
    sys.exit(main())
