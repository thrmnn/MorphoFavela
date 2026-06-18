"""Build a project hub — one navigable page linking every reviewable part of the
project (plans, decision logs, deliverables, figure galleries, reports,
dashboards). Discovers artifacts by existence so it degrades gracefully.

Serve the repo root and open /outputs/_hub/index.html:
    python -m http.server 8773 --directory <repo-root>

    python scripts/build_project_hub.py
"""

from __future__ import annotations

import glob
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "_hub"


def _git(*args, default="?"):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


# (root-relative url, title, description, kind) — kind drives the badge
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


def _discover_dashboards():
    items = []
    for p in sorted(glob.glob(str(ROOT / "outputs/_distribution/**/*.html"),
                              recursive=True)):
        rel = "/" + str(Path(p).relative_to(ROOT))
        items.append((rel, Path(p).stem.replace("_", " "), "Site dashboard.", "live"))
    return items


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sections = {k: [i for i in v if _exists(i[0].lstrip("/"))]
                for k, v in SECTIONS.items()}
    dash = _discover_dashboards()
    if dash:
        sections["Dashboards"] = dash

    n_sites = len(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet")))
    n_figs = len(glob.glob(str(ROOT / "outputs/cross_site/signature/figures_v2/*.png")))
    badge = {"doc": "#6c757d", "pdf": "#b5341f", "live": "#1a7f4b"}

    blocks = []
    for title, items in sections.items():
        cards = "".join(
            f"""<a class="card" href="{url}" target="_blank">
      <span class="badge" style="background:{badge.get(kind, '#888')}">{kind}</span>
      <h3>{name}</h3><p>{desc}</p></a>"""
            for url, name, desc, kind in items)
        blocks.append(f"<section><h2 id='{title.split()[0].lower()}'>{title}</h2>"
                      f"<div class='grid'>{cards}</div></section>")

    nav = "".join(f"<a href='#{t.split()[0].lower()}'>{t}</a>" for t in sections)
    html = f"""<!doctype html><meta charset=utf-8>
<title>MorphoFavela — project hub</title>
<style>
 :root{{--fg:#1d1d1f;--mut:#666}}
 *{{box-sizing:border-box}} body{{margin:0;font:15px/1.55 system-ui,sans-serif;color:var(--fg);background:#f5f5f7}}
 header{{padding:28px 40px;background:#fff;border-bottom:1px solid #e3e3e6}}
 h1{{margin:0;font-size:22px}} .sub{{color:var(--mut);margin-top:6px;font-size:13px}}
 nav{{position:sticky;top:0;background:#fffd;backdrop-filter:blur(6px);padding:10px 40px;border-bottom:1px solid #e3e3e6;display:flex;gap:18px;flex-wrap:wrap;z-index:5}}
 nav a{{color:#0a5;text-decoration:none;font-size:13px;font-weight:600}}
 main{{padding:8px 40px 60px}} section{{margin-top:28px}} h2{{font-size:16px;margin:0 0 12px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}}
 .card{{display:block;background:#fff;border:1px solid #e3e3e6;border-radius:10px;padding:16px;text-decoration:none;color:inherit;transition:.12s}}
 .card:hover{{border-color:#0a5;transform:translateY(-2px);box-shadow:0 4px 14px #0001}}
 .card h3{{margin:8px 0 4px;font-size:15px}} .card p{{margin:0;color:var(--mut);font-size:13px}}
 .badge{{display:inline-block;color:#fff;font-size:10px;text-transform:uppercase;letter-spacing:.04em;padding:2px 7px;border-radius:20px}}
</style>
<header>
 <h1>MorphoFavela — project hub</h1>
 <div class="sub">branch <b>{_git('rev-parse', '--abbrev-ref', 'HEAD')}</b> ·
   {_git('rev-parse', '--short', 'HEAD')} · {n_sites} sites with feature tables ·
   {n_figs} signature figures · regenerate: <code>python scripts/build_project_hub.py</code></div>
</header>
<nav>{nav}</nav>
<main>{''.join(blocks)}</main>"""
    (OUT / "index.html").write_text(html)
    print(f"hub written to {OUT/'index.html'} "
          f"({sum(len(v) for v in sections.values())} links)")


if __name__ == "__main__":
    main()
