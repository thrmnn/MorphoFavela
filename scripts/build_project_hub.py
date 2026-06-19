"""Build the MorphoFavela project hub — one navigable page linking every
reviewable part of the project (review checklist, per-favela dashboards, figure
galleries, deliverables, plans, decision logs). Markdown docs (incl. the
technical report, with inline figures) render to styled in-browser pages;
artifacts are discovered by existence so the hub degrades gracefully.

Uses the project-agnostic `hubkit` engine (vendored from the project-hub skill).
Serve the repo root and open / (lands on the hub via index.html redirect):
    python -m http.server 8773 --bind 0.0.0.0 --directory <repo-root>

    python scripts/build_project_hub.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hubkit import (  # noqa: E402
    badge,
    breadcrumb,
    card,
    git_provenance,
    page,
    render_doc_page,
    section,
)

OUT = ROOT / "outputs" / "_hub"
DOCS = OUT / "docs"

SITE_NAMES = {
    "vidigal": "Vidigal", "rocinha": "Rocinha", "riodaspedras": "Rio das Pedras",
    "complexo_do_alemao": "Complexo do Alemão", "maré": "Maré",
    "borel": "Borel", "jacarezinho": "Jacarezinho",
    "morro_do_juramento": "Morro do Juramento",
}
CAMPAIGN = ["vidigal", "rocinha", "riodaspedras", "complexo_do_alemao", "maré"]
DASH = ROOT / "outputs" / "_distribution" / "html_dashboards"

# Static sections of markdown docs / live reports (kind: doc | live).
DOC_SECTIONS = {
    "Figure galleries & reports": [
        ("/outputs/cross_site/signature/figures_v2/index.html",
         "Signature figures (v2)",
         "Expert-reviewed morphotype figure set — click to enlarge.", "live"),
        ("/outputs/comparative/vidigal_vs_mingze/report/index.html",
         "Mingze solar comparison", "Vidigal Ladybug-vs-raycast report.", "live"),
    ],
    "Plans & decisions": [
        ("/docs/roughness_plan.md", "Roughness-estimation plan",
         "z0/zd from morphometry for CFD; SOTA brief + equations (Kanda 2013).", "doc"),
        ("/docs/roughness_decisions.md", "Roughness decision log",
         "R-A choices + findings (zd>H_mean 70–93%; λp>0.5 mostly out-of-envelope).", "doc"),
        ("/docs/morpho_signature_plan.md", "Morpho-signature plan",
         "The 3-workstream track plan + literature brief.", "doc"),
        ("/docs/morpho_signature_decisions.md", "Decision log",
         "Every methodological tradeoff (WS-0…WS-B), append-only.", "doc"),
        ("/docs/visualization_plan.md", "Visualization plan",
         "Figure spec + synthesized 3-expert review.", "doc"),
        ("/ROADMAP.md", "Roadmap", "Phase status + recently completed.", "doc"),
        ("/README.md", "README", "Repo overview + structure.", "doc"),
    ],
}

CALLOUT = """<section><h2 id="review">Start here — review &amp; decisions</h2>
<div style="background:#fff;border:1px solid #e3e7ec;border-left:5px solid #1a6fb5;
border-radius:10px;padding:14px 18px">
<p style="margin:0 0 8px">Page through the <b>Signature figures</b> gallery and the
per-favela <b>Sites</b>, then the inputs I need (priority order):</p>
<ol style="margin:0 0 4px">
<li><b>Finalize the 6 morphotype names</b> — gates every caption + the paper narrative.</li>
<li><b>Figure keep / refine</b> across the 9 figures (incl. the priority map).</li>
<li><b>WS-B follow-ups</b> — CFD-anchor overlay + boundary transects (in progress).</li>
</ol></div></section>"""


def _doc_card(url, name, desc, prov):
    src = ROOT / url.lstrip("/")
    back = breadcrumb([("← Project hub", "../index.html"), (src.stem, None)])
    render_doc_page(src, DOCS / f"{src.stem}.html", crumb=back, provenance=prov)
    return card(name, desc, f"docs/{src.stem}.html", meta=url, kind="doc")


def sites_section(prov):
    cards = []
    for s in CAMPAIGN + [s for s in SITE_NAMES if s not in CAMPAIGN]:
        idx = DASH / s / "index.html"
        if idx.exists():
            tag = "campaign site" if s in CAMPAIGN else "calibration site"
            cards.append(card(SITE_NAMES[s], f"Interactive per-favela dashboard — {tag}.",
                              "/" + str(idx.relative_to(ROOT)), kind="ok"))
    if (DASH / "index.html").exists():
        cards.append(card("All sites — interactive index",
                          "Combined dashboard index for every favela.",
                          "/" + str((DASH / "index.html").relative_to(ROOT)), kind="info"))
    return section("Sites", cards, anchor="sites")


def deliverables_section(prov):
    cards = []
    tr_md = ROOT / "docs/technical_report/technical_report.md"
    tr_pdf = ROOT / "docs/technical_report/technical_report.pdf"
    if tr_md.exists():
        back = breadcrumb([("← Project hub", "../index.html"), ("Technical report", None)])
        render_doc_page(tr_md, DOCS / "technical_report.html", crumb=back,
                        provenance=prov, base="/docs/technical_report/")
        cards.append(card("Technical report", "Full report — fast HTML view, figures inline.",
                          "docs/technical_report.html", kind="ok"))
    if tr_pdf.exists():
        mb = tr_pdf.stat().st_size // 1_000_000
        cards.append(card(f"Technical report — PDF ({mb} MB)",
                          "Canonical typeset PDF; large file, opens in a new tab.",
                          "/docs/technical_report/technical_report.pdf", kind="info"))
    return section("Deliverables", cards, anchor="deliverables")


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    prov = git_provenance(ROOT, "scripts/build_project_hub.py")

    blocks = [CALLOUT, sites_section(prov)]
    titles = ["Review", "Sites"]
    for title, items in DOC_SECTIONS.items():
        cards = []
        for url, name, desc, kind in items:
            if not (ROOT / url.lstrip("/")).exists():
                continue
            cards.append(_doc_card(url, name, desc, prov) if kind == "doc"
                         else card(name, desc, url, meta=url, kind="ok"))
        blocks.append(section(title, cards, anchor=title.split()[0].lower()))
        titles.append(title)
    blocks.insert(2, deliverables_section(prov))
    titles.insert(2, "Deliverables")

    n_sites = sum((DASH / s / "index.html").exists() for s in SITE_NAMES)
    n_figs = len(glob.glob(str(ROOT / "outputs/cross_site/signature/figures_v2/*.png")))
    nav = " ".join(f'<a href="#{t.split()[0].lower()}">{t}</a>' for t in titles)
    sub = (f'{badge("ok", f"{n_sites} site dashboards")} '
           f'{badge("info", f"{n_figs} figures")} &nbsp; {nav}')
    (OUT / "index.html").write_text(
        page("MorphoFavela — project hub", sub, "".join(blocks), provenance=prov))
    print(f"hub written: {len(titles)} sections, {n_sites} site dashboards")


if __name__ == "__main__":
    main()
