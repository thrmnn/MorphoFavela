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
    toc_sections,
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
        ("/docs/morphology_overview.md", "▶ Morphology overview (start here)",
         "Goal-first walkthrough: the signature, the 6 morphotypes, validation, "
         "and the honest roughness limit — with figures and captions.", "doc"),
        ("/outputs/cross_site/signature/figures_v2/index.html",
         "Signature & roughness figures",
         "The full grouped gallery — click any figure to enlarge.", "live"),
        ("/outputs/comparative/vidigal_vs_mingze/report/index.html",
         "Mingze solar comparison", "Vidigal Ladybug-vs-raycast report.", "live"),
    ],
    "Plans & decisions": [
        ("/docs/typology_predictor_plan.md", "Typology-as-predictor plan",
         "Use the morphotype/morphotope typology to predict environmental failure; "
         "LOSO transfer, variance decomposition, blind risk map.", "doc"),
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

GAL = "/outputs/cross_site/signature/figures_v2/index.html"


def build_callout(prov):
    """Top panel: newest results (direct links) + the live work queue, so new
    figures are never hard to find."""
    if (ROOT / "docs/work_queue.md").exists():
        render_doc_page(ROOT / "docs/work_queue.md", DOCS / "work_queue.html",
                        crumb=breadcrumb([("← Project hub", "../index.html"),
                                          ("Work queue", None)]), provenance=prov)
    hub = "/outputs/_hub"
    latest = [
        ("Typology → environmental failure", f"{GAL}#prioritization",
         "the money figure: type predicts WHO-2h sun failure (14%→73%)"),
        ("Block-scale morphotopes", f"{GAL}#morphotope", "5 tissues, 4/5 recur"),
        ("Morphology overview (start here)", f"{hub}/docs/morphology_overview.html",
         "goal-first walkthrough, all figures + captions"),
        ("Configuration: party-wall", f"{GAL}#signature", "favela fabric is fused 0.6–1.0"),
        ("Roughness validity", f"{GAL}#roughness", "per-cell z0/zd invalid 53–75%"),
    ]
    items = "".join(
        f'<li><a href="{u}" style="color:#0a5;font-weight:600;'
        f'text-decoration:none">{n}</a> <span style="color:#888">— {d}</span></li>'
        for n, u, d in latest)
    return (f'<section><h2 id="latest">🆕 Latest &amp; work queue</h2>'
            f'<div style="background:#fff;border:1px solid #e3e7ec;border-left:5px '
            f'solid #1a7f4b;border-radius:10px;padding:14px 18px">'
            f'<p style="margin:0 0 6px;font-weight:600">Newest results — click straight in:</p>'
            f'<ul style="margin:0 0 10px;padding-left:18px;line-height:1.7">{items}</ul>'
            f'<p style="margin:0"><a href="/outputs/_hub/docs/work_queue.html" '
            f'style="font-weight:700">📋 Full work queue →</a> &nbsp; what is in '
            f'progress, queued, and gated.</p>'
            f'</div></section>')


def _doc_card(url, name, desc, prov):
    src = ROOT / url.lstrip("/")
    back = breadcrumb([("← Project hub", "../index.html"), (src.stem, None)])
    render_doc_page(src, DOCS / f"{src.stem}.html", crumb=back, provenance=prov)
    return card(name, desc, f"/outputs/_hub/docs/{src.stem}.html", meta=url, kind="doc")


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

    blocks = [build_callout(prov), sites_section(prov)]
    titles = ["Latest", "Sites"]
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
    sub = (f'{badge("ok", f"{n_sites} site dashboards")} '
           f'{badge("info", f"{n_figs} figures")}')
    sidebar = toc_sections([(t.split()[0].lower(), t) for t in titles])
    (OUT / "index.html").write_text(
        page("MorphoFavela — project hub", sub, "".join(blocks),
             provenance=prov, sidebar=sidebar))
    print(f"hub written: {len(titles)} sections, {n_sites} site dashboards")


if __name__ == "__main__":
    main()
