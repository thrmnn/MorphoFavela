"""Render package markdown to PDF: pandoc (md -> HTML) then weasyprint
(HTML -> PDF), the same pipeline as docs/technical_report/build_pdf.py
(pdflatex chokes on the Unicode in the README; weasyprint does not).

Two documents use it: report.pdf (report_css: A4, 2.5 cm margins, one
figure per block at text width) and README.pdf (readme_css: same page,
smaller type and tight tables for the column lists). Both carry the running
header "Octopus OM2 data package <version>" and a footer with the page
number and the use terms.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import weasyprint

from .package_docs import USE_TERMS

#: Must match src/om_package/fig_style.FONT_FAMILY, so text and figures share one family.
FONT_FAMILY = "DejaVu Sans"


def _css_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _page_css(version: str, footer: str) -> str:
    """A4, 2.5 cm margins (text width 16 cm), running header and footer."""
    return f"""
@page {{
  size: A4;
  margin: 25mm 25mm 25mm 25mm;
  @top-left {{ content: "Octopus OM2 data package {_css_string(version)}"; font-family: "{FONT_FAMILY}";
               font-size: 7.5pt; color: #6b6b6b; vertical-align: bottom; padding-bottom: 4mm; }}
  @bottom-left {{ content: "{_css_string(footer)}"; font-family: "{FONT_FAMILY}"; font-size: 7.5pt;
                  color: #6b6b6b; vertical-align: top; padding-top: 4mm; }}
  @bottom-right {{ content: counter(page); font-family: "{FONT_FAMILY}"; font-size: 7.5pt; color: #6b6b6b;
                   vertical-align: top; padding-top: 4mm; }}
}}
html {{ font-family: "{FONT_FAMILY}", sans-serif; }}
body {{ max-width: none !important; margin: 0 !important; padding: 0 !important; hyphens: manual; }}
a {{ color: #0f5f57; text-decoration: none; }}
code {{ font-family: "DejaVu Sans Mono", monospace; font-size: 0.86em; background: #f3f3f3; padding: 0 1.5pt; }}
"""


def report_css(version: str, footer: str = USE_TERMS) -> str:
    """Report: readable body, figures at text width with their caption kept
    on the same page. Every chapter (h2) starts a new page, except one marked
    .keep-on-page; tables never split across pages."""
    return _page_css(version, footer) + """
body { font-size: 9.4pt; line-height: 1.36; color: #1d1d1f; }
h1 { font-size: 17pt; line-height: 1.2; margin: 0 0 8pt 0; color: #111; }
.titleblock { margin: 0 0 10pt 0; padding-bottom: 6pt; border-bottom: 0.8pt solid #333; }
.titleblock h1 { font-size: 20pt; line-height: 1.16; margin: 0 0 4pt 0; }
.titleblock .subtitle { font-size: 12pt; line-height: 1.3; color: #333; margin: 0 0 8pt 0; }
.titleblock .byline { font-size: 9.6pt; color: #111; margin: 0 0 3pt 0; }
.titleblock .issue { font-size: 9pt; color: #555; margin: 0; }
h2 { font-size: 12.5pt; color: #111; margin: 0 0 6pt 0; padding-bottom: 2pt;
     border-bottom: 0.6pt solid #b9b9b9; break-before: page; break-after: avoid; }
h2.keep-on-page { break-before: auto; margin-top: 14pt; }
h3 { font-size: 10.4pt; color: #111; margin: 12pt 0 4pt 0; break-after: avoid; }
ul { margin: 2pt 0 8pt 0; padding-left: 14pt; }
li { margin: 0 0 3pt 0; }
p { margin: 0 0 5pt 0; orphans: 3; widows: 3; }
strong { color: #111; }
table { border-collapse: collapse; width: 100%; font-size: 7.1pt; line-height: 1.22; margin: 4pt 0 8pt 0;
        break-inside: avoid; }
th, td { border-bottom: 0.4pt solid #c8c8c8; padding: 1.3pt 5pt 1.3pt 0; text-align: left; vertical-align: top;
         overflow-wrap: anywhere; }
th { border-bottom: 0.8pt solid #333; font-weight: bold; }
tr { break-inside: avoid; }
td:first-child { width: 33%; }
td:nth-child(2) { width: 40%; }
td code { font-size: 6.8pt; background: none; padding: 0; }
figure { margin: 6pt 0 8pt 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; }
figcaption { font-size: 8.4pt; line-height: 1.35; color: #444; margin: 3pt 0 0 0; text-align: left; }
"""


def readme_css(version: str, footer: str = USE_TERMS) -> str:
    """README: denser text, wrapped code and compact tables for the column lists."""
    return _page_css(version, footer) + """
body { font-size: 8.8pt; line-height: 1.4; color: #1a1a1a; }
h1 { font-size: 16pt; margin: 0 0 6pt 0; }
h2 { font-size: 12pt; margin: 14pt 0 4pt 0; border-bottom: 0.6pt solid #b9b9b9; break-after: avoid; }
h3 { font-size: 10pt; margin: 10pt 0 3pt 0; break-after: avoid; }
p, li { overflow-wrap: anywhere; margin: 0 0 4pt 0; }
pre { padding: 5pt 7pt; border: 0.5pt solid #ddd; background: #f6f6f6; white-space: pre-wrap; font-size: 7.6pt;
      break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { margin: 0 0 8pt 0; padding: 3pt 8pt; border-left: 2.5pt solid #0f5f57; background: #f4f6f7; }
table { border-collapse: collapse; margin: 4pt 0 8pt 0; font-size: 7pt; line-height: 1.3; width: 100%; }
th, td { border-bottom: 0.4pt solid #c8c8c8; padding: 2pt 4pt 2pt 0; text-align: left; vertical-align: top;
         overflow-wrap: anywhere; }
th { border-bottom: 0.8pt solid #333; }
tr { break-inside: avoid; }
td code { font-size: 6.6pt; background: none; padding: 0; }
td:first-child { width: 34%; }
td:nth-child(2) { width: 14%; }
"""


def render_markdown_pdf(md: Path, pdf: Path, *, css: str, title: str, md_format: str = "gfm") -> Path:
    """Render md to pdf; image paths resolve relative to md's directory.
    Raises if pandoc fails (the document must ship or the build must say
    why)."""
    md, pdf = Path(md), Path(pdf)
    html = md.parent / f"_{md.stem}_build_tmp.html"
    result = subprocess.run(
        ["pandoc", str(md), "-o", str(html), "--standalone", "--from", md_format, "--to", "html5",
         "--embed-resources", "--metadata", f"pagetitle={title}"],
        capture_output=True, text=True, cwd=md.parent,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pandoc failed rendering {md}: {result.stderr}")
    # pandoc sizes pipe-table columns from the dash counts (all equal here);
    # let the auto layout decide.
    html.write_text(re.sub(r"<colgroup>.*?</colgroup>", "", html.read_text(encoding="utf-8"), flags=re.S),
                    encoding="utf-8")
    try:
        weasyprint.HTML(filename=str(html), base_url=str(md.parent)).write_pdf(
            str(pdf), stylesheets=[weasyprint.CSS(string=css)]
        )
    finally:
        html.unlink(missing_ok=True)
    return pdf


def render_readme_pdf(package_dir: Path) -> Path:
    """Write package_dir/README.pdf from package_dir/README.md."""
    package_dir = Path(package_dir)
    version = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))["package_version"]
    return render_markdown_pdf(package_dir / "README.md", package_dir / "README.pdf",
                               css=readme_css(version),
                               title=f"Octopus OM2 data package {version}")
