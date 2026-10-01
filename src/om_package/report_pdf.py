"""Render the package README.md to README.pdf: pandoc (md -> HTML) then
weasyprint (HTML -> PDF), the same pipeline as
docs/technical_report/build_pdf.py (pdflatex chokes on the Unicode in the
README; weasyprint does not).

Wide tables (the spec conformance table has long evidence cells) use a
small font and aggressive wrapping so they fit the
A4 page instead of running off it.
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


def render_readme_pdf(package_dir: Path) -> Path:
    """Write package_dir/README.pdf from package_dir/README.md; returns the
    PDF path. Raises if pandoc fails (the report must ship or the build
    must say why)."""
    package_dir = Path(package_dir)
    md = package_dir / "README.md"
    html = package_dir / "_README_build_tmp.html"
    pdf = package_dir / "README.pdf"
    result = subprocess.run(
        ["pandoc", str(md), "-o", str(html), "--standalone", "--from", "gfm", "--to", "html5",
         "--embed-resources", "--metadata", "pagetitle=Maré morphology, OM2 - data package"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pandoc failed rendering {md}: {result.stderr}")
    # pandoc sizes pipe-table columns from the dash counts (all equal here),
    # which squeezes id/status; let the browser-style auto layout decide.
    html.write_text(re.sub(r"<colgroup>.*?</colgroup>", "", html.read_text(encoding="utf-8"), flags=re.S),
                    encoding="utf-8")
    try:
        weasyprint.HTML(filename=str(html), base_url=str(package_dir)).write_pdf(
            str(pdf), stylesheets=[weasyprint.CSS(string=CSS)]
        )
    finally:
        html.unlink(missing_ok=True)
    return pdf
