"""Render package markdown to PDF: pandoc (md -> HTML) then weasyprint
(HTML -> PDF), the same pipeline as docs/technical_report/build_pdf.py
(pdflatex chokes on the Unicode in the README; weasyprint does not).

Two documents use it: README.pdf (the technical README, whose wide spec
table needs a small font and aggressive wrapping to fit A4) and
report.pdf (the short human report, see report.py, with readable body
text and full-width figures, styled by report_css).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import weasyprint

CSS = """
@page {
  size: A4;
  margin: 16mm 13mm 16mm 13mm;
  @top-left { content: "Maré morphology, OM2 - data package"; font-size: 8pt; color: #666; }
  @top-right { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; }
}
body { font-family: "Liberation Sans", "Arial", sans-serif; font-size: 9pt; line-height: 1.4; color: #1a1a1a; }
h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 5px; margin-top: 0; }
h2 { font-size: 13pt; margin-top: 16pt; border-bottom: 1px solid #bbb; page-break-after: avoid; }
h3 { font-size: 11pt; page-break-after: avoid; }
p, li { overflow-wrap: anywhere; }
code, pre { font-family: "Liberation Mono", "Consolas", monospace; font-size: 8pt; background: #f5f5f5; overflow-wrap: anywhere; }
pre { padding: 6px 8px; border: 1px solid #ddd; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; }
blockquote { margin: 0.6em 0; padding-left: 0.8em; border-left: 3px solid #bbb; color: #333; }
table { border-collapse: collapse; margin: 0.7em 0; font-size: 7pt; line-height: 1.3; width: 100%; table-layout: auto; }
th:first-child, td:first-child { white-space: nowrap; }
td:nth-child(3), th:nth-child(3) { min-width: 17mm; }
td:nth-child(5), th:nth-child(5) { min-width: 22mm; }
th, td { border: 1px solid #bbb; padding: 2px 4px; text-align: left; vertical-align: top; overflow-wrap: anywhere; word-break: break-word; }
th { background: #eee; }
tr { page-break-inside: avoid; }
table code { font-size: 6.5pt; }
a { color: #2A5FA5; text-decoration: none; }
"""


def report_css(version: str) -> str:
    """Stylesheet of the human report: A4, one figure per section at full
    width, a small running header and footer. The version goes in the
    header so a printed page says which package it describes."""
    return """
@page {
  size: A4;
  margin: 17mm 18mm 16mm 18mm;
  @top-left { content: "Octopus OM2 data package, VERSION"; font-size: 7.5pt; color: #6b6b6b; }
  @top-right { content: "Internal review draft"; font-size: 7.5pt; color: #6b6b6b; }
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 7.5pt; color: #6b6b6b; }
}
@page :first { @top-left { content: none; } @top-right { content: none; } }
html { font-family: "Source Sans 3", "Liberation Sans", "Arial", sans-serif; }
body { font-size: 10.5pt; line-height: 1.45; color: #1d1d1f; }
h1 { font-family: "Source Serif 4", "Liberation Serif", Georgia, serif; font-size: 21pt; line-height: 1.15;
     margin: 0 0 3pt 0; color: #111; }
h2 { font-family: "Source Serif 4", "Liberation Serif", Georgia, serif; font-size: 14pt; color: #111;
     margin: 14pt 0 5pt 0; padding-bottom: 2pt; border-bottom: 0.6pt solid #b9b9b9; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 0 0 4pt 0; color: #0f5f57; break-after: avoid; }
p { margin: 0 0 5pt 0; orphans: 3; widows: 3; }
ol, ul { margin: 2pt 0 6pt 0; padding-left: 16pt; }
li { margin: 0 0 2.5pt 0; }
strong { color: #111; }
.meta p { font-size: 9pt; color: #555; margin: 0 0 9pt 0; padding-bottom: 6pt; border-bottom: 1.2pt solid #111; }
.figsec { break-inside: avoid; margin: 0 0 12pt 0; padding-top: 4pt; }
figure { margin: 6pt 0 0 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; width: auto; height: auto; }
img.hero { max-height: 112mm; }
img.map { max-height: 150mm; }
img.tall { max-height: 160mm; }
img.wide { max-height: 112mm; }
figcaption { font-size: 8.8pt; line-height: 1.35; color: #444; margin: 4pt 0 0 0; text-align: left; }
section.keep, .keep { break-inside: avoid; }
.care { break-inside: avoid; background: #f4f6f7; border-left: 3pt solid #0f5f57; padding: 2pt 10pt 4pt 10pt;
        margin: 10pt 0; }
.care h2 { border-bottom: none; margin-top: 6pt; }
a { color: #0f5f57; text-decoration: none; }
""".replace("VERSION", version)


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
    # pandoc sizes pipe-table columns from the dash counts (all equal here),
    # which squeezes id/status; let the browser-style auto layout decide.
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
    return render_markdown_pdf(package_dir / "README.md", package_dir / "README.pdf",
                               css=CSS, title="Maré morphology, OM2 - data package")
