"""Build the MorphoFavela project hub — one navigable page linking every
reviewable part of the project (plans, decision logs, deliverables, figure
galleries, reports, dashboards). Markdown docs are rendered to styled in-browser
pages; artifacts are discovered by existence so the hub degrades gracefully.

Uses the project-agnostic `hubkit` engine (vendored from the project-hub skill).
Serve the repo root and open /outputs/_hub/index.html:
    python -m http.server 8773 --directory <repo-root>

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

# (root-relative url, title, description, kind). kind: doc (md→rendered) | pdf | live
SECTIONS = {
    "Plans & decisions": [
        ("/docs/morpho_signature_plan.md", "Morpho-signature plan",
         "The 3-workstream track plan + literature brief.", "doc"),
        ("/docs/morpho_signature_decisions.md", "Decision log",
         "Every methodological tradeoff (WS-0…WS-A.2), append-only.", "doc"),
        ("/docs/visualization_plan.md", "Visualization plan",
         "Figure spec + synthesized 3-expert review.", "doc"),
        ("/ROADMAP.md", "Roadmap", "Phase status + recently completed.", "doc"),
    ],
    "Deliverables": [
        ("/docs/technical_report/technical_report.pdf", "Technical report (PDF)",
         "The project's primary deliverable document.", "pdf"),
        ("/README.md", "README", "Repo overview + structure.", "doc"),
    ],
    "Figure galleries & reports": [
        ("/outputs/cross_site/signature/figures_v2/index.html",
         "Signature figures (v2)",
         "Expert-reviewed morphotype figure set — click to enlarge.", "live"),
        ("/outputs/comparative/vidigal_vs_mingze/report/index.html",
         "Mingze solar comparison", "Vidigal Ladybug-vs-raycast report.", "live"),
    ],
}
KIND_BADGE = {"doc": "doc", "pdf": "info", "live": "ok"}


def _discover_dashboards():
    out = []
    for p in sorted(glob.glob(str(ROOT / "outputs/_distribution/**/*.html"), recursive=True)):
        out.append(("/" + str(Path(p).relative_to(ROOT)),
                    Path(p).stem.replace("_", " "), "Site dashboard.", "live"))
    return out


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    prov = git_provenance(ROOT, "scripts/build_project_hub.py")

    sections = dict(SECTIONS)
    dash = _discover_dashboards()
    if dash:
        sections["Dashboards"] = dash

    blocks, n_links = [], 0
    for title, items in sections.items():
        cards = []
        for url, name, desc, kind in items:
            src = ROOT / url.lstrip("/")
            if not src.exists():
                continue
            if kind == "doc" and src.suffix == ".md":
                back = breadcrumb([("← Project hub", "../index.html"), (src.stem, None)])
                render_doc_page(src, DOCS / f"{src.stem}.html", crumb=back, provenance=prov)
                href = f"docs/{src.stem}.html"
            else:
                href = url
            cards.append(card(name, desc, href, meta=url, kind=KIND_BADGE.get(kind, "doc")))
            n_links += 1
        blocks.append(section(title, cards, anchor=title.split()[0].lower()))

    n_sites = len(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet")))
    n_figs = len(glob.glob(str(ROOT / "outputs/cross_site/signature/figures_v2/*.png")))
    nav = " ".join(f'<a href="#{t.split()[0].lower()}">{t}</a>' for t in sections)
    sub = (f'{badge("ok", f"{n_sites} sites")} {badge("info", f"{n_figs} figures")} '
           f'&nbsp; {nav}')
    (OUT / "index.html").write_text(
        page("MorphoFavela — project hub", sub, "".join(blocks), provenance=prov))
    print(f"hub written to {OUT/'index.html'} ({n_links} links, {len(sections)} sections)")


if __name__ == "__main__":
    main()
