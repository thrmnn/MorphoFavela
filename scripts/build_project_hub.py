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
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hubkit import (
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
        ("/docs/autonomous_loop_plan.md", "Autonomous-loop plan + top blockers",
         "Ranked blockers to remove for a multi-hour parallel-agent loop.", "doc"),
        ("/docs/tr_audit.md", "TR coherence audit",
         "Bulletproofing punch list — criticals done, medium queued.", "doc"),
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

# TR §5.6 anchor in the rendered technical-report HTML page.
VENT_TR = "/outputs/_hub/docs/technical_report.html#56-geometric-ventilation-tendencies--gated-pre-cfd"
VENT_EXPORTS = "/outputs/paper_figures/exports"
# Geometry-only ventilation tendencies (§5.6) — each an image card that zooms
# in a lightbox and links through to its captioned place in the report.
VENT_FIGS = [
    ("ventilation_index.png", "Multi-constraint index (0–3)",
     "Count of geometry-only ventilation constraints triggered per cell "
     "(skimming · deep · wind-aligned). 24.2 % triply constrained pooled; "
     "flatland RdP 55 %/Maré 32 % ≫ hillside ~7 %. Checklist count, not a sum."),
    ("ventilation_susceptibility.png", "Susceptibility (regime × depth)",
     "λf flow regime crossed with lateral depth; worst class skimming ∩ deep "
     "= 41.8 % pooled. Two axes kept separate, never summed."),
    ("lateral_connectivity.png", "Lateral connectivity (depth)",
     "Distance from each built cell to the nearest open edge; pooled median "
     "31.6 m. ρ(depth, λf) = +0.487 → doubly constrained fabric."),
    ("wind_exposure.png", "Effective wind exposure",
     "Directional λf weighted by the measured wind rose; near-isotropic "
     "(ratio median 1.007) → directional alignment is a 2nd-order effect."),
]


MZ_DIR = "/outputs/comparative/mingze_facade"
FACADE_FIGS = [
    ("/mingze_facade_crosscheck.png", "Façade vs street + cross-check (ours vs Ladybug)",
     "Our raycast street field cross-checked against Mingze's independent Ladybug "
     "run: ordering agrees, Maré least deprived, our winter worst-day ≈1.9× the "
     "annual envelope. Façade deprivation 50–56 % floor, Rocinha 72 %."),
    ("/figures/fig2_facade_deprivation.png", "Façade deprivation by settlement (Mingze)",
     "Share of façade area below WHO 2 h/day: Rocinha 72 %, others 50–56 %."),
    ("/figures/fig3_drivers.png", "Drivers: height-invariance + valley→ridge (Mingze)",
     "Taller ≠ brighter (flat lines); hillside sun rises valley-floor→ridge."),
    ("/figures/fig4_street_gini.png", "Street-sun inequality / Gini (Mingze)",
     "Rocinha most unequal (Gini 0.70); p10–median–p90 of direct sun hours."),
    ("/figures/fig1_seasonal_riodaspedras.png", "Seasonal 3-D solar model — Rio das Pedras (Mingze)",
     "Annual direct-sun on the full 3-D fabric at 3 m façade resolution, four seasonal dates."),
]


SIG_DIR = "/outputs/cross_site/signature/figures_v2"
SIG_GAL = f"{SIG_DIR}/index.html"
# The project's own headline result — promoted to a first-class hero section so a
# first-time reviewer meets the contribution before any partner cross-check.
HEADLINE_FIGS = [
    ("typology_failure_lookup.png", "fig-typology_failure_lookup",
     "Morphotype predicts winter-sun failure (14 % → 73 %)",
     "The headline result: cell morphotype alone predicts the share of street "
     "observers below the WHO 2 h/day winter-sun floor, rising 14 % → 73 % across "
     "the six types and transferring leave-one-site-out."),
    ("typology_blind_riskmap.png", "fig-typology_blind_riskmap",
     "Blind cross-site winter-sun risk map (8 favelas)",
     "One continuous fabric-vector model maps WHO-2h failure risk across 5 campaign "
     "+ 3 calibration favelas; beats the morphotype-rate blind map."),
    ("typology_variance.png", "fig-typology_variance",
     "Why it transfers: morphotype 17 % ≫ site 2 %",
     "Morphotype explains 17 % of the winter-sun-failure variance vs 2 % for site "
     "and 0.7 % for their interaction — so the type→failure mapping is portable."),
]


def headline_section(prov):
    """Hero: the money figure(s) as zoomable image cards, deep-linked to their
    captioned place in the signature gallery."""
    cards = [
        card(title, desc, f"{SIG_GAL}#{anchor}", img=f"{SIG_DIR}/{fn}",
             meta="headline result · WHO 2 h/day winter floor", kind="ok",
             badge_label="Headline")
        for fn, anchor, title, desc in HEADLINE_FIGS
        if (ROOT / f"{SIG_DIR}/{fn}".lstrip("/")).exists()
    ]
    return section("Headline result — morphotype predicts winter-sun failure",
                   cards, anchor="headline")


def facade_solar_section(prov):
    """Façade-level solar (independent Ladybug run) + our street cross-check."""
    cards = [
        card(title, desc, MZ_DIR + href, img=MZ_DIR + href,
             meta="façade-level solar · WHO 2 h/day", kind="info",
             badge_label="Partner")
        for href, title, desc in FACADE_FIGS
        if (ROOT / (MZ_DIR + href).lstrip("/")).exists()
    ]
    return section("Solar access — façade & street (Mingze / Ladybug + our cross-check)",
                   cards, anchor="facade-solar")


def maup_section(prov):
    """MAUP resolution-curve figure (5–30 m)."""
    fig = "/outputs/comparative/maup/maup_resolution_curve.png"
    if not (ROOT / fig.lstrip("/")).exists():
        return ""
    c = card("MAUP resolution curve (5–30 m)",
             "Flow-regime shares, λf/σH medians, and per-site skimming vs cell "
             "size. Monotonic drift; cross-site ordering preserved (Spearman "
             "ρ = 0.90). Absolute shares must be quoted at the 10 m lock.",
             fig, img=fig, meta="TR §10.9 · dissolved λf", kind="info")
    return section("Grid-resolution sensitivity (MAUP)", [c], anchor="maup")


def ventilation_section(prov):
    """Dedicated §5.6 gallery so the geometry-only ventilation tendencies are a
    first-class hub item (image cards with lightbox, linking to the report)."""
    cards = [
        card(title, desc, VENT_TR, img=f"{VENT_EXPORTS}/{fn}",
             meta="pre-CFD tendency · not adequacy (τ CFD-gated)", kind="ok")
        for fn, title, desc in VENT_FIGS
        if (ROOT / "outputs/paper_figures/exports" / fn).exists()
    ]
    return section("Geometric ventilation tendencies (§5.6)", cards, anchor="ventilation")


def _render_latest_item(n, u, d):
    """One <li> for the Latest changelog: a trailing '— NEW' becomes a pill, and
    both the label and gloss are HTML-escaped (no raw & reaches the page)."""
    m = re.search(r"\s*[—-]\s*NEW\s*$", n)
    label = n[:m.start()] if m else n
    pill = (badge("info", "NEW") + " ") if m else ""
    return (f'<li>{pill}<a href="{u}">{html.escape(label)}</a> '
            f'<span class="gloss">— {html.escape(d)}</span></li>')


def build_callout(prov):
    """Top panel: newest results (direct links) + the live work queue, so new
    figures are never hard to find."""
    if (ROOT / "docs/work_queue.md").exists():
        render_doc_page(ROOT / "docs/work_queue.md", DOCS / "work_queue.html",
                        crumb=breadcrumb([("← Project hub", "../index.html"),
                                          ("Work queue", None)]), provenance=prov)
    hub = "/outputs/_hub"
    tr = f"{hub}/docs/technical_report.html"
    # newest first — each lands on the EXACT figure / TR section it documents
    latest = [
        ("Façade-level solar + Ladybug cross-check (§5.4.1) — NEW",
         "#facade-solar",
         "Mingze's façade run (50–72 % WHO-2h deprivation) + our street cross-check: "
         "ordering agrees, Maré least deprived, Rocinha façade outlier 72 %"),
        ("MAUP resolution curve 5–30 m (§10.9) — NEW",
         "#maup",
         "full grid-size sweep; monotonic regime drift, cross-site ordering "
         "preserved (Spearman ρ = 0.90)"),
        ("Geometric ventilation tendencies (§5.6) — 4-figure gallery",
         "#ventilation",
         "lateral depth · regime×depth susceptibility · wind exposure · "
         "multi-constraint index (0–3); geometry-only, τ-gated"),
        ("TR §6.6 roughness — invalidity caveat added",
         f"{tr}#66-aerodynamic-roughness-z0-zd",
         "the bulletproofing fix: per-cell z0/zd invalid 53–75%, envelope is the result"),
        ("TR §5.5 Morphological Typology & Signature",
         f"{tr}#55-morphological-typology-signature",
         "the signature in the report; morphotype (cell) vs morphotope (tissue)"),
        ("Typology → environmental failure (money figure)",
         f"{GAL}#fig-typology_failure_lookup",
         "type predicts WHO-2h sun failure 14%→73%, transfers leave-one-site-out"),
        ("Variance: type vs site vs interaction",
         f"{GAL}#fig-typology_variance",
         "morphotype 17% vs site 2% vs 0.7% interaction → the mapping transfers"),
        ("Block-scale morphotope (tissue) maps", f"{GAL}#fig-morphotope_maps",
         "5 tissues, distinct from the cell types; 4/5 recur"),
        ("TR coherence audit (punch list)", f"{hub}/docs/tr_audit.html",
         "the weaknesses being bulletproofed — criticals done, medium queued"),
        ("Morphology overview (start here)", f"{hub}/docs/morphology_overview.html",
         "goal-first walkthrough, all figures + captions"),
    ]
    items = "".join(_render_latest_item(*t) for t in latest)
    return (f'<section><h2 id="latest">Latest &amp; work queue</h2>'
            f'<div class="callout">'
            f'<p class="lead">Newest results — click straight in:</p>'
            f'<ul>{items}</ul>'
            f'<p class="more"><a href="/outputs/_hub/docs/work_queue.html">'
            f'📋 Full work queue →</a> '
            f'<span class="gloss">what is in progress, queued, and gated.</span></p>'
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
            camp = s in CAMPAIGN
            tag = "campaign site" if camp else "calibration site"
            cards.append(card(SITE_NAMES[s], f"Interactive per-favela dashboard — {tag}.",
                              "/" + str(idx.relative_to(ROOT)),
                              kind="ok" if camp else "info",
                              badge_label="Campaign" if camp else "Calibration"))
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
                          "docs/technical_report.html", kind="ok",
                          badge_label="Report"))
    if tr_pdf.exists():
        mb = tr_pdf.stat().st_size // 1_000_000
        cards.append(card("Technical report — PDF",
                          "Canonical typeset PDF; large file, opens in a new tab.",
                          "/docs/technical_report/technical_report.pdf", kind="info",
                          badge_label=f"PDF · {mb} MB"))
    return section("Deliverables", cards, anchor="deliverables")


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    prov = git_provenance(ROOT, "scripts/build_project_hub.py")

    # Single ordered source of truth: (anchor, sidebar_label, html). The anchor
    # matches the id the section builder emits; the label is a human heading, so the
    # sidebar never shows a machine slug. Project-owned contribution (headline, TR)
    # precedes the partner façade cross-check.
    sections = [
        ("latest", "Latest & work queue", build_callout(prov)),
        ("headline", "Headline result", headline_section(prov)),
        ("deliverables", "Deliverables", deliverables_section(prov)),
        ("ventilation", "Ventilation tendencies (§5.6)", ventilation_section(prov)),
        ("maup", "Grid sensitivity (MAUP)", maup_section(prov)),
        ("facade-solar", "Solar access — façade & street", facade_solar_section(prov)),
        ("sites", "Sites", sites_section(prov)),
    ]
    for title, items in DOC_SECTIONS.items():
        cards = []
        for url, name, desc, kind in items:
            if not (ROOT / url.lstrip("/")).exists():
                continue
            cards.append(_doc_card(url, name, desc, prov) if kind == "doc"
                         else card(name, desc, url, meta=url, kind="ok"))
        anchor = title.split()[0].lower()
        sections.append((anchor, title, section(title, cards, anchor=anchor)))

    sections = [(a, lbl, h) for a, lbl, h in sections if h]  # degrade-by-existence
    body = "".join(h for _, _, h in sections)
    sidebar = toc_sections([(a, lbl) for a, lbl, _ in sections])

    n_sites = sum((DASH / s / "index.html").exists() for s in SITE_NAMES)
    n_figs = len(glob.glob(str(ROOT / "outputs/cross_site/signature/figures_v2/*.png")))
    sub = (f'{badge("ok", f"{n_sites} site dashboards")} '
           f'{badge("info", f"{n_figs} figures")}')
    (OUT / "index.html").write_text(
        page("MorphoFavela — project hub", sub, body,
             provenance=prov, sidebar=sidebar))
    print(f"hub written: {len(sections)} sections, {n_sites} site dashboards")


if __name__ == "__main__":
    main()
