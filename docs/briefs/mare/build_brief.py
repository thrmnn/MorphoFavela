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
ACCENT_TINT = "#eef3fa"

# Font pair from what weasyprint sees on this laptop (fc-list : family):
# body Lato, display Fraunces.
CSS = f"""
@page {{
  size: A4;
  margin: 13mm 14mm 15mm 14mm;
  @bottom-center {{
    content: "Maré morphology brief · """ + date.today().isoformat() + f""" · draft for PI review";
    font-size: 7.5pt; color: #777;
  }}
  @top-right {{ content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; }}
}}
body {{
  font-family: "Lato", "Liberation Sans", "Arial", sans-serif;
  font-size: 10pt;
  line-height: 1.28;
  color: #1c1c1c;
  counter-reset: brief-figure;
  max-width: 100%;
  margin: 0;
  padding: 0;
}}
h1, h2 {{ font-family: "Fraunces", "Bitstream Charter", Georgia, serif; }}
h1 {{
  font-size: 19pt;
  font-weight: 700;
  color: {ACCENT};
  border-bottom: 2px solid {ACCENT};
  padding-bottom: 5px;
  margin-top: 0;
  margin-bottom: 6pt;
}}
h1:first-of-type {{ page-break-before: avoid; }}
h2 {{
  font-size: 13.5pt;
  font-weight: 600;
  color: #1c1c1c;
  margin-top: 9pt;
  margin-bottom: 4pt;
  padding-bottom: 2px;
  border-bottom: 1px solid #ccc;
  page-break-after: avoid;
}}
p, li {{ text-align: left; }}
p {{ margin: 0.32em 0; }}
em {{ color: #555; }}
strong {{ font-weight: 600; }}
table {{
  /* pandoc's embedded default stylesheet sets table display to block
     (a web responsive-table reset) which defeats the CSS table-layout
     algorithm entirely — override back to real table layout. */
  display: table;
  border-collapse: collapse;
  margin: 0.4em 0;
  font-size: 8.5pt;
  width: 100%;
  table-layout: fixed;
  page-break-inside: avoid;
}}
th, td {{
  border: 1px solid #d5d5d5;
  padding: 3px 6px;
  text-align: left;
  vertical-align: top;
}}
th {{
  background: {ACCENT_TINT};
  color: {ACCENT};
  font-weight: 600;
}}
img {{
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.3em auto;
  page-break-inside: avoid;
}}
figure {{
  margin: 0.5em 0;
  page-break-inside: avoid;
  counter-increment: brief-figure;
}}
figcaption {{ font-size: 7.5pt; color: #555; text-align: center; padding: 0 4mm; }}
figcaption::before {{
  content: "Figure " counter(brief-figure) ". ";
  font-weight: 700;
  color: {ACCENT};
}}
a {{ color: {ACCENT}; text-decoration: none; }}
ul, ol {{ margin: 0.3em 0; padding-left: 1.4em; }}
li {{ margin: 0.1em 0; }}

/* key-numbers card: the first two-column table in section 2 ("at a glance") */
h2:nth-of-type(2) + table {{
  border-collapse: separate;
  border-spacing: 0;
  background: {ACCENT_TINT};
  border-radius: 7px;
  font-size: 9.8pt;
  page-break-inside: avoid;
}}
h2:nth-of-type(2) + table tr:first-child td {{ padding-top: 8px; }}
h2:nth-of-type(2) + table tr:last-child td {{ padding-bottom: 8px; }}
h2:nth-of-type(2) + table td {{
  border: none;
  border-bottom: 1px solid #dbe6f3;
  padding: 4px 14px;
  color: #444;
}}
h2:nth-of-type(2) + table tr:last-child td {{ border-bottom: none; }}
h2:nth-of-type(2) + table td:last-child {{
  color: {ACCENT};
  font-weight: 700;
  text-align: right;
}}
h2:nth-of-type(2) + table td:first-child {{ width: 60%; }}
h2:nth-of-type(2) + table td:last-child {{ width: 40%; }}

/* data-inventory table: fixed column widths so no header wraps ragged */
h2:nth-of-type(3) + table {{ table-layout: fixed; }}
h2:nth-of-type(3) + table th:nth-child(1),
h2:nth-of-type(3) + table td:nth-child(1) {{ width: 18%; }}
h2:nth-of-type(3) + table th:nth-child(2),
h2:nth-of-type(3) + table td:nth-child(2) {{ width: 13%; }}
h2:nth-of-type(3) + table th:nth-child(3),
h2:nth-of-type(3) + table td:nth-child(3) {{ width: 21%; }}
h2:nth-of-type(3) + table th:nth-child(4),
h2:nth-of-type(3) + table td:nth-child(4) {{ width: 21%; }}
h2:nth-of-type(3) + table th:nth-child(5),
h2:nth-of-type(3) + table td:nth-child(5) {{ width: 27%; }}
"""


# Provisional disclosure default (brisaverse tasks.json
# _meta.provisional_default_policy: reversible, no external dependency, PI
# tap overrides): the T0-T5 morphotype taxonomy is generalised to lettered
# "fabric cluster" labels unless --named-morphotypes restores the named
# variant. See disclosure_sweep.md.
NAMED_MORPHOTYPES = [
    ("T0", "Open Fringe"),
    ("T1", "Flatland Consolidated"),
    ("T2", "Hillside Fringe"),
    ("T3", "Shaded Consolidated"),
    ("T4", "Hillside Core"),
    ("T5", "Saturated Core"),
]
GENERALISED_CLUSTER_LABELS = ["A", "B", "C", "D", "E", "F"]

NAMED_MORPHOTYPE_NARRATIVE = (
    "Maré's fabric is dominated by T5 Saturated Core (λp near its maximum), the "
    "flatland-conditional type associated with the tight, near-fully-covered "
    "block interiors of the original housing-project layout; T4 Hillside Core, "
    "the type universal across all five campaign sites, is present as a "
    "secondary component. T1 and T5 are present only where flat buildable land "
    "exists, which is why they concentrate at the two flatland sites rather "
    "than recurring campaign-wide."
)
GENERALISED_MORPHOTYPE_NARRATIVE = (
    "Maré's fabric is dominated by one cluster with plan density near its "
    "maximum, associated with the tight, near-fully-covered block interiors of "
    "the original housing-project layout; a second cluster, common across all "
    "five campaign sites, is present as a secondary component. Two of the six "
    "clusters occur only where flat buildable land exists, which is why they "
    "concentrate at the two flatland sites rather than recurring campaign-wide."
)


def _morphotype_composition_md(numbers_by_id: dict, named_morphotypes: bool) -> str:
    intro = (
        "The campaign's fabric-vector clustering assigns each built cell to one "
        "of six recurring fabric clusters. Maré's composition:"
    )
    header = "Morphotype" if named_morphotypes else "Fabric cluster"
    rows = []
    for i in range(6):
        pct = format_entry(numbers_by_id[f"mare_morphotype_T{i}_pct"])
        if named_morphotypes:
            code, name = NAMED_MORPHOTYPES[i]
            label = f"{code} — {name}"
        else:
            label = GENERALISED_CLUSTER_LABELS[i]
        rows.append(f"<tr><td>{label}</td><td>{pct}%</td></tr>")
    # Raw HTML (not a pandoc pipe table): weasyprint's automatic table-layout
    # algorithm does not stretch short-content tables to fill width:100% even
    # with table-layout:fixed set, so column widths are pinned explicitly via
    # <colgroup> here rather than left to the layout algorithm.
    table = (
        "<table>\n"
        '<colgroup><col style="width:55%"><col style="width:45%"></colgroup>\n'
        f"<thead><tr><th>{header}</th><th>Share of built cells</th></tr></thead>\n"
        "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n"
        "</table>"
    )
    narrative = NAMED_MORPHOTYPE_NARRATIVE if named_morphotypes else GENERALISED_MORPHOTYPE_NARRATIVE
    return f"{intro}\n\n{table}\n\n{narrative}"


def fill_template(numbers_by_id: dict, named_morphotypes: bool = False) -> str:
    template = string.Template(SRC_MD.read_text())
    mapping = {
        id_: format_entry(entry)
        for id_, entry in numbers_by_id.items()
        if id_ != UNFILLED_ID
    }
    mapping["morphotype_composition"] = _morphotype_composition_md(numbers_by_id, named_morphotypes)

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


def write_disclosure_sweep(rendered_md: str, named_morphotypes: bool) -> None:
    lines = rendered_md.splitlines()
    rows = []
    for lineno, line in enumerate(lines, start=1):
        for m in DISCLOSURE_PATTERN.finditer(line):
            rows.append((lineno, m.group(0), line.strip(), _decision_for(m.group(0))))

    mode_line = (
        "ENABLED — this build rendered the named T0–T5 variant."
        if named_morphotypes else
        "not passed — this build rendered the default, generalised (lettered A–F) variant."
    )
    out = ["# Maré brief — disclosure sweep",
           "",
           f"Run against the rendered markdown ({SRC_MD.name}, filled). "
           "Every hit needs an explicit include/drop decision — the PI decides.",
           "",
           "## Provisional disclosure default",
           "",
           "brisaverse tasks.json `_meta.provisional_default_policy`: reversible, "
           "no external dependency, PI tap overrides. The T0–T5 morphotype "
           "taxonomy (codes and names) is the project's own unpublished "
           "classification scheme (technical_report.md §5.5); by default it does "
           "NOT appear in this brief — the composition passage in 'Maré among "
           "the five campaign sites' describes six lettered fabric clusters "
           "(A–F) by share, with a plain description of the dominant cluster and "
           "no taxonomy codes or names. Pass `--named-morphotypes` to "
           "build_brief.py to restore the named T0–T5 variant for PI review.",
           "",
           f"`--named-morphotypes`: {mode_line}",
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


def build(outputs_root: Path, named_morphotypes: bool = False) -> int:
    try:
        numbers = collect_numbers.collect(outputs_root)
    except collect_numbers.MissingSource as e:
        print(f"build_brief: SKIP — {e}")
        return 2  # distinct from success so callers cannot mistake a skip for a build

    by_id = {n["id"]: n for n in numbers}
    NUMBERS_JSON.write_text(json.dumps(by_id, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"build_brief: collected {len(by_id)} numbers -> {NUMBERS_JSON}")

    manifest = render_figures.render_all(outputs_root, FIGURES_DIR)
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"build_brief: rendered {len(manifest)} figures -> {FIGURES_DIR}")

    filled = fill_template(by_id, named_morphotypes=named_morphotypes)
    FILLED_MD.write_text(filled)
    write_disclosure_sweep(filled, named_morphotypes)

    print("build_brief: converting markdown -> HTML via pandoc...")
    # Deliberately NOT --standalone: pandoc's standalone html5 template embeds
    # its own default <style> (incl. `table { display: block; }`, a
    # responsive-table reset) which is CSS author-origin and silently beats
    # our own stylesheet — passed to weasyprint as user-origin — for any
    # property both set, regardless of selector specificity or source order.
    # A bare fragment sidesteps that entirely; CSS lives solely in `CSS` below.
    result = subprocess.run(
        [
            "pandoc",
            str(FILLED_MD),
            "-o",
            str(HTML),
            "--from",
            "gfm+implicit_figures",
            "--to",
            "html5",
        ],
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    if result.returncode != 0:
        print(result.stderr)
        return 1

    print(f"build_brief: rendering HTML -> PDF with weasyprint -> {PDF}")
    html = weasyprint.HTML(filename=str(HTML), base_url=str(HERE), encoding="utf-8")
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
    ap.add_argument(
        "--named-morphotypes",
        action="store_true",
        help="restore the T0-T5 morphotype taxonomy labels and names "
        "(default: generalised, lettered 'fabric cluster' labels — see "
        "disclosure_sweep.md)",
    )
    args = ap.parse_args()
    return build(args.outputs_root, named_morphotypes=args.named_morphotypes)


if __name__ == "__main__":
    sys.exit(main())
