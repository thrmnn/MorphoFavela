#!/usr/bin/env python3
"""Build technical_report.pdf from technical_report.md.

Pipeline: pandoc (markdown -> HTML) -> weasyprint (HTML -> PDF).
Chosen over pdflatex because the markdown uses Unicode characters (λ, σ,
°, ≥, ≤, ×, →) that pdflatex chokes on, and xelatex is not installed.

Usage: python docs/technical_report/build_pdf.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import weasyprint

HERE = Path(__file__).resolve().parent
MD = HERE / "technical_report.md"
HTML = HERE / "_build_tmp.html"
PDF = HERE / "technical_report.pdf"

CSS = """
@page {
  size: A4;
  margin: 18mm 15mm 18mm 15mm;
  @top-left { content: "Brisa+ Technical Report"; font-size: 8pt; color: #666; }
  @top-right { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; }
}
body {
  font-family: "Liberation Sans", "Arial", sans-serif;
  font-size: 9.5pt;
  line-height: 1.42;
  color: #1a1a1a;
  max-width: 100%;
}
h1 {
  font-size: 18pt;
  font-weight: 700;
  color: #111;
  border-bottom: 2px solid #333;
  padding-bottom: 6px;
  margin-top: 0;
  page-break-before: auto;
}
h1:first-of-type { page-break-before: avoid; }
h2 {
  font-size: 14pt;
  font-weight: 700;
  color: #222;
  margin-top: 20pt;
  padding-bottom: 2px;
  border-bottom: 1px solid #bbb;
  page-break-after: avoid;
}
h3 {
  font-size: 11.5pt;
  font-weight: 600;
  color: #333;
  margin-top: 14pt;
  page-break-after: avoid;
}
h4 { font-size: 10pt; font-weight: 600; margin-top: 10pt; }
p, li { text-align: justify; }
p { margin: 0.5em 0; }
strong { font-weight: 600; }
hr { border: none; border-top: 1px solid #ccc; margin: 1em 0; }
code, pre {
  font-family: "Liberation Mono", "Consolas", monospace;
  font-size: 8.5pt;
  background: #f5f5f5;
}
code { padding: 0.1em 0.3em; border-radius: 2px; }
pre {
  padding: 8px 10px;
  border-radius: 3px;
  border: 1px solid #ddd;
  white-space: pre-wrap;
  word-wrap: break-word;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; }
blockquote {
  margin: 0.6em 0;
  padding-left: 0.8em;
  border-left: 3px solid #bbb;
  color: #555;
  font-style: italic;
}
table {
  border-collapse: collapse;
  margin: 0.8em 0;
  font-size: 8.5pt;
  width: 100%;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #ccc;
  padding: 3px 6px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #f0f0f0;
  font-weight: 600;
}
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.5em auto;
  page-break-inside: avoid;
}
figure { margin: 1em 0; page-break-inside: avoid; }
figcaption { font-size: 8pt; color: #555; text-align: center; }
a { color: #2A5FA5; text-decoration: none; }
a:hover { text-decoration: underline; }
ul, ol { margin: 0.4em 0; padding-left: 1.6em; }
li { margin: 0.15em 0; }
/* Table of contents styling */
nav#TOC { page-break-after: always; }
nav#TOC > ul { list-style: none; padding-left: 0; }
nav#TOC ul ul { padding-left: 1.2em; }
nav#TOC li { margin: 0.2em 0; }
"""


def build():
    print(f"Reading  {MD}")
    if not MD.exists():
        sys.exit(f"Not found: {MD}")

    print("Converting markdown -> HTML via pandoc...")
    result = subprocess.run(
        [
            "pandoc",
            str(MD),
            "-o", str(HTML),
            "--standalone",
            "--toc",
            "--toc-depth=2",
            "--metadata", "title=Brisa+ Technical Report",
            "--from", "gfm+tex_math_dollars",
            "--to", "html5",
            "--embed-resources",
            "--mathjax",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr)
        sys.exit("pandoc failed")

    print(f"Rendering HTML -> PDF with weasyprint -> {PDF}")
    html = weasyprint.HTML(filename=str(HTML), base_url=str(HERE))
    css = weasyprint.CSS(string=CSS)
    html.write_pdf(str(PDF), stylesheets=[css])

    # Cleanup
    HTML.unlink(missing_ok=True)

    size_mb = PDF.stat().st_size / 1024 / 1024
    print(f"Done: {PDF} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build()
