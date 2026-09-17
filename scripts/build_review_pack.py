#!/usr/bin/env python3
"""Render the PI's promotion-review pack into the mirror the hub already serves.

Reads the methodology and intermediate-result documents behind the staged WP-07
figures and writes them as HTML next to the figures, in review order, with an
index. Nothing crosses the release boundary: outputs/_hub/ is the read-only
mirror the PI sees behind Access. Re-run after any run of record changes.
Run: python3 scripts/build_review_pack.py
"""
from __future__ import annotations
import glob, html, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "_hub" / "wp07_staged" / "review"


def latest(pattern: str) -> Path:
    hits = sorted(glob.glob(str(ROOT / pattern)))
    if not hits:
        sys.exit(f"no file matches {pattern}")
    return Path(hits[-1])


# Review order: what the figures show, the numbers behind them, how they were
# computed, then the validation results that bound the claims.
SECTIONS = [
    ("The numbers behind every figure", [
        (latest("runs/wp07_ledger_*/ledger.md"), "WP-07 ledger — every headline number, copied by code from its run of record"),
    ]),
    ("Methodology", [
        (ROOT / "docs/wp02_horizon_engine_spec.md", "The horizon engine — per-patch sky visibility on the GPU and how it was accepted"),
        (ROOT / "docs/wp07_ledger_spec.md", "How the ledger is built and what makes a number final"),
        (ROOT / "docs/wp07_figures_spec.md", "What each of the four figures is specified to show"),
        (ROOT / "docs/ventaxis_canonical.md", "The second axis — definition of record for the geometry-constraint count"),
    ]),
    ("Validation and sensitivity", [
        (latest("runs/g3_domain_*/sensitivity.md"), "G3 — domain sensitivity across nine grid variants (the position moves, the ordering does not)"),
        (latest("runs/wp03_tls_*/report_v3.md"), "G2 — TLS ground truth vs the 2.5D model (a negative result with a confound, stated as it is)"),
        (latest("runs/wp04f2_facade_*/crossref.md"), "Façade cross-reference — why the façade layer was NOT accepted"),
    ]),
]



# ── Per-work-package results pages, generated from the ledger ─────────────────
# The PI asked for "the results of WP4 and 6": the cards linked a manifest JSON,
# which is not a result for a reader. One page per run of record, every entry
# the ledger attributes to that run, one table with all sites as columns where
# the id has a site segment. Nothing typed; regenerated with the pack.
import json, re
LEDGER = latest("runs/wp07_ledger_*/ledger.json")
led = json.loads(LEDGER.read_text())
runs_of_record = led["_meta"].get("runs_of_record", {})
SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "mare"]
WP_TITLES = {
    "wp04": "WP-04 — site decomposition: direct-sun hours and the 2 h floor (Athens Charter 1943, Point 26), ground and street, both reference days",
    "wp06": "WP-06 — the second axis: geometry-constraint count shares per site",
    "wp05": "WP-05 — citywide distribution and each favela's percentile position",
    "g3": "G3 — domain sensitivity across nine grid variants",
    "wp02": "WP-02 — engine acceptance against the CPU reference",
}
wp_pages = []
for wp, run_id in runs_of_record.items():
    ents = {k: v for k, v in led["entries"].items() if v.get("source", {}).get("run_id") == run_id}
    if not ents:
        continue
    # rows keyed by the id with the site segment removed, columns = sites
    table = {}
    for k, v in ents.items():
        parts = k.split(".")
        site = next((s for s in parts if s in SITES), None)
        key = ".".join(x for x in parts if x != site) if site else k
        table.setdefault(key, {})[site or "value"] = v["value"]
    cols = SITES if any(any(s in r for s in SITES) for r in table.values()) else ["value"]
    md = [f"# {WP_TITLES.get(wp.split('_')[0], wp)}", "",
          f"Run of record: `{run_id}` · {len(ents)} ledger entries, every one `status: final` · ledger `{LEDGER.parent.name}`", "",
          "| id | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for key in sorted(table):
        vals = [table[key].get(c) for c in cols]
        fmt = lambda x: "" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))
        md.append(f"| `{key}` | " + " | ".join(fmt(x) for x in vals) + " |")
    src = OUT / f"_results_{wp}.md"; src.write_text("\n".join(md) + "\n")
    wp_pages.append((src, f"{wp.upper()} results — {len(ents)} numbers, all sites, from run {run_id}"))
SECTIONS.insert(0, ("Results by work package", wp_pages))


OUT.mkdir(parents=True, exist_ok=True)
rows = []
for title, docs in SECTIONS:
    # <nav>, not a bare <ul> — a link outside a real nav/card is PROSE-ONLY to
    # scripts/audit_hub_graph.py's reachability gate (docs/hub_wp_structure_spec.md).
    rows.append(f"<h2>{html.escape(title)}</h2><nav><ul>")
    for src, blurb in docs:
        dst = OUT / (src.stem + ".html")
        subprocess.run(["pandoc", str(src), "-f", "gfm", "-t", "html5", "-s",
                        "--metadata", f"title={src.stem}", "-o", str(dst)], check=True)
        # pandoc's standalone template adds an IE conditional loading html5shiv
        # from a protocol-relative CDN URL; the hub build's no-root-absolute
        # guard refuses it (correctly — the mirror must be self-contained).
        html_txt = dst.read_text()
        html_txt = "\n".join(l for l in html_txt.splitlines() if "html5shiv" not in l and "<!--[if" not in l and "<![endif]" not in l)
        dst.write_text(html_txt)
        rel = src.relative_to(ROOT)
        rows.append(f'<li><a href="{dst.name}">{html.escape(blurb)}</a><br>'
                    f'<small><code>{html.escape(str(rel))}</code></small></li>')
    rows.append("</ul></nav>")

index = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Promotion review pack</title>
<style>
 body{{font:16px/1.55 -apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:52rem;margin:0 auto;padding:1.5rem 1rem;color:#1d1a16;background:#fbf9f5}}
 h1{{font-size:1.6rem;margin:.2rem 0 .4rem}} h2{{font-size:1.1rem;margin:1.6rem 0 .5rem;color:#8a651c}}
 li{{margin:.55rem 0}} a{{color:#b6482b;display:inline-block;min-height:44px;line-height:1.4;padding:.35rem 0}}
 .lead{{color:#5a554d}} small{{color:#8a857c}}
</style></head><body>
<p class="lead"><a href="../">← the four staged figures</a></p>
<h1>Promotion review pack</h1>
<p class="lead">Read in this order: the figures (previous page), the numbers behind them, how they were computed, then what bounds the claims. Everything here is generated from the runs of record; the promotion decision itself is <b>wp07_figure_promotion</b> on /ops.</p>
{''.join(rows)}
</body></html>"""
(OUT / "index.html").write_text(index)
print(f"wrote {OUT.relative_to(ROOT)}/index.html + {sum(len(d) for _, d in SECTIONS)} documents")
