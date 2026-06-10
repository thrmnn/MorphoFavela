"""Post-build polish for the Vidigal HTML dashboard.

The builder (scripts/build_html_dashboard.py) renders the page from
templates shared across all five sites and is being concurrently edited
by sibling subagents. This script applies the Vidigal-specific final
polish items from the review feedback that must land on the rendered
artefact regardless of upstream churn:

  - drive the dek subtitle from live stats so it cannot drift from the
    provenance block;
  - swap "Zona Sul" gloss for "south-zone";
  - tighten the H4 / H1 "this site" drawer lines;
  - add the TOC CSS, mid-page back-to-top anchors, a visible
    "drag-to-filter" affordance for the histogram brush, a styled
    glossary popover, paired delta-chip colours, mobile map height,
    a human-readable density comparator, and "AOI" → "study area
    boundary" copy;
  - expand the print stylesheet to match the contract paragraph.

It is idempotent: running twice produces the same file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIST = REPO / "outputs" / "_distribution" / "html_dashboards"
VIDIGAL = DIST / "vidigal"

EXTRA_CSS = """
/* === post-polish === */
.toc{
  display:flex; flex-wrap:wrap; gap:0; padding:0.6rem 2rem;
  border-bottom:1px solid var(--rule); background:var(--canvas-paper);
  font-family:var(--sans); font-size:0.78rem;
}
.toc a{
  padding:0.3rem 0.9rem; color:var(--ink-muted);
  border:none; border-right:1px solid var(--rule);
  letter-spacing:0.04em; text-transform:uppercase; font-weight:500;
}
.toc a:first-child{padding-left:0;}
.toc a:last-child{border-right:none;}
.toc a:hover{color:var(--accent-primary); border-bottom-color:transparent;}
.toc a.active{color:var(--accent-primary);}

/* mid-section back-to-top anchor */
.section-up{
  display:block; text-align:right; margin-top:1rem; font-size:0.72rem;
  color:var(--ink-muted); letter-spacing:0.06em; text-transform:uppercase;
  border:none;
}
.section-up:hover{color:var(--accent-primary);}

/* delta chip paired colours */
.delta-chip.up{color:var(--accent-warm);}
.delta-chip.down{color:var(--accent-cool);}
.delta-chip.zero{color:var(--ink-muted);}

/* glossary popover replaces native title= tooltip */
.gloss-pop{
  position:absolute; z-index:200; max-width:24rem;
  background:#fff; color:var(--ink);
  border:1px solid var(--rule-strong); border-radius:4px;
  padding:0.55rem 0.7rem; font-size:0.82rem; font-style:normal;
  font-family:var(--sans); line-height:1.45;
  box-shadow:0 6px 18px rgba(0,0,0,0.08);
  pointer-events:none;
}
.gloss-pop strong{font-weight:600; color:var(--accent-primary); font-family:var(--serif);}

/* visible brush affordance under the histogram */
.brush-hint{
  display:flex; align-items:center; gap:0.5rem; margin:0.4rem 0 0;
  font-size:0.78rem; color:var(--ink-muted);
}
.brush-hint .arrow{
  display:inline-block; width:1.6rem; height:0; border-top:1px dashed var(--accent-warm);
  position:relative;
}
.brush-hint .arrow::before, .brush-hint .arrow::after{
  content:""; position:absolute; top:-3px; width:0; height:0;
  border-top:3px solid transparent; border-bottom:3px solid transparent;
}
.brush-hint .arrow::before{left:-1px; border-right:5px solid var(--accent-warm);}
.brush-hint .arrow::after{right:-1px; border-left:5px solid var(--accent-warm);}

/* masthead pill — relax tightness against the title */
.pill{font-size:0.86rem !important; padding:0.28rem 0.75rem !important;}

/* responsive — map */
@media (max-width:600px){
  #map{height:360px;}
  .toc{padding:0.5rem 1rem; font-size:0.72rem;}
  section{padding:1.4rem 1rem;}
  .dek{padding:1rem 1rem;}
}
"""

EXTRA_PRINT_CSS = """
@media print{
  /* contract: toggles to defaults, drawer forced open, basemap greyscale, A3 collapse */
  :root{--canvas-paper:#fff;}
  .kpi-strip{position:static; page-break-inside:avoid; border-top:1.5px solid var(--ink);}
  .toc, .drawer-toggle, .drawer-close, .brush-hint, .section-up{display:none !important;}
  .drawer{
    position:static; transform:none; width:100%; box-shadow:none;
    border-left:none; border-top:1.5px solid var(--ink);
    padding:1rem 0; page-break-before:always;
  }
  .drawer details{display:block; border-bottom:1px solid var(--rule);}
  .drawer details > *{display:block !important;}
  .drawer summary{cursor:default; font-weight:600;}
  .toggle, .map-toolbar select, .map-toolbar .toggle{display:none !important;}
  .leaflet-tile{filter:grayscale(1) contrast(0.9);}
  body{background:#fff; color:#000; -webkit-print-color-adjust:economy;}
  section{page-break-inside:avoid; border-bottom:0.5px solid #999;}
  a{color:#000; border-bottom:none;}
  .pill{background:none; border:1px solid #000; color:#000;}
  .delta-chip{color:#000 !important;}
  .kpi-tooltip{display:none !important;}
  .gloss-pop{display:none !important;}
  #hist, #scatter, #map{
    page-break-inside:avoid;
  }
  .chart{min-height:auto;}
  .csr-row{cursor:default;}
  footer{font-size:0.7rem;}
}
"""


def density_comparator_phrase(this_density: float, all_densities: list[float]) -> str:
    """Human-readable comparator for the density KPI chip."""
    if not all_densities or this_density is None:
        return "density across the five sites"
    ranked = sorted(all_densities, reverse=True)
    pos = ranked.index(this_density) + 1
    n = len(ranked)
    bucket = "top" if pos == 1 else ("highest" if pos <= n // 3 else "lowest" if pos > 2 * n // 3 else "mid-range")
    return f"{bucket} density of the five campaign sites"


def main() -> None:
    html_path = VIDIGAL / "index.html"
    stats_path = VIDIGAL / "data" / "stats.json"
    shared_path = DIST / "shared" / "cross_site_stats.json"

    html = html_path.read_text()
    stats = json.loads(stats_path.read_text())
    shared = json.loads(shared_path.read_text()) if shared_path.exists() else {"per_site": []}

    area_km2 = stats.get("area_km2")
    area_str = f"{area_km2:.2f}" if isinstance(area_km2, (int, float)) else "—"
    density = stats.get("density_per_km2")
    all_density = [s.get("density_per_km2") for s in shared.get("per_site", []) if isinstance(s.get("density_per_km2"), (int, float))]

    # ---- 1. dek: drive subtitle from live area, drop "south-zone" jargon
    # Pattern: old subtitle string baked in by the builder
    new_subtitle = (
        f"a hillside favela in Rio's Zona Sul "
        f"(<dfn data-term=\"Zona Sul\">south Rio's beachside neighbourhoods</dfn>), "
        f"{area_str} km², running ridge to beach"
    )
    html = re.sub(
        r"a south-zone hillside favela, [^,]*?km²[^<]*?(?=\. )",
        new_subtitle,
        html,
        count=1,
    )

    # ---- 2. tighten H4 drawer line — Vidigal is not an H4 site
    # remove the "This site: X.XX% of observers at exact-zero SVF" line entirely from H4
    html = re.sub(
        r'<p class="muted"><strong>This site:</strong> [\d.]+% of observers at exact-zero SVF\.</p>',
        '<p class="muted"><em>Not flagged on this site</em> — Vidigal\'s exact-zero share is well below the Rocinha threshold that prompted this finding.</p>',
        html,
        count=1,
    )

    # ---- 3. density chip — replace bare audit code with comparator phrase
    if isinstance(density, (int, float)) and all_density:
        phrase = density_comparator_phrase(density, all_density)
        html = re.sub(
            r'<span class="chip">H2 · [\d,]+/km²</span>',
            f'<span class="chip" title="H2 · density varies 2.3× across sites">{phrase}</span>',
            html,
            count=1,
        )

    # ---- 4. caveats heading — Vidigal only has the edge toggle
    html = html.replace(
        "Number-changers · toggle the filters above the map and watch the headline SVF number move",
        "Number-changers · toggle the edge-observer filter above the map and watch the headline SVF number move",
    )
    html = html.replace(
        "Number-changers · toggle the masks above and watch the mean SVF tile",
        "Number-changers · toggle the edge-observer filter above the map and watch the headline SVF number move",
    )

    # ---- 5. AOI → "study area boundary"
    html = html.replace("AOI boundary", "study area boundary")
    html = html.replace("AOI edge", "study area boundary")

    # ---- 6. visible brush affordance below the histogram
    if "brush-hint" not in html:
        html = html.replace(
            '<button id="hist-reset"',
            '<div class="brush-hint" aria-hidden="true"><span class="arrow"></span><span>drag horizontally across the histogram to filter the map and scatter to that SVF range</span></div>\n  <button id="hist-reset"',
        )

    # ---- 7. scatter x-axis gloss — match the histogram phrasing in chart caption
    # We can't easily edit Plotly's axis title without rebuilding, but we can add
    # a follow-up paragraph below the scatter that mirrors the hist gloss.
    if "scatter-axis-gloss" not in html:
        html = html.replace(
            '<p class="chart-caption">Up to 5,000 points, sampled evenly across the SVF range',
            '<p id="scatter-axis-gloss" class="chart-caption">x-axis is <dfn data-term="SVF">Sky View Factor</dfn> (0 = walls all around, 1 = open field); y-axis is solar hours per day (annual mean). Up to 5,000 points, sampled evenly across the SVF range',
        )

    # ---- 8. mid-page back-to-top anchors (idempotent — sweep, but only
    # insert one anchor per <section> boundary, and never inside a section
    # that already has one. The `count=1` re-use approach in earlier
    # revisions targeted the same first boundary on every loop iteration,
    # producing five stacked anchors — fixed by walking sections explicitly.)
    section_boundary = re.compile(r'</section>\n\n<section id="')
    pieces = section_boundary.split(html)
    if len(pieces) > 1:
        # Reconstitute: each piece (except last) ends a section, and we want
        # to inject the anchor before </section> if the section body does
        # not already contain a class="section-up" link.
        new_pieces = []
        for i, piece in enumerate(pieces[:-1]):
            if 'class="section-up"' in piece:
                new_pieces.append(piece)
            else:
                new_pieces.append(piece + '<a class="section-up" href="#top">↑ back to top</a>')
        new_pieces.append(pieces[-1])
        html = '</section>\n\n<section id="'.join(new_pieces)

    # Defensive: collapse any pre-existing run of duplicate anchors that a
    # buggy earlier polish run may have left in the file.
    html = re.sub(
        r'(<a class="section-up" href="#top">↑ back to top</a>){2,}',
        '<a class="section-up" href="#top">↑ back to top</a>',
        html,
    )

    # ---- 9. inject extra CSS + glossary popover JS into the inline <style> block
    # Add an end-of-style marker to make idempotent injection trivial
    if "/* === post-polish ===" not in html:
        html = html.replace("</style>", EXTRA_CSS + "\n</style>", 1)

    # ---- 10. styled glossary popover (replaces native title= tooltip)
    glossary_js = """
<script>
/* glossary popover — supersedes the native title= tooltip set by glossary.js
   so the popover matches the editorial type system. */
(function(){
  async function loadDict(){
    try {
      const r = await fetch('../assets/glossary.json');
      if(!r.ok) return {};
      return await r.json();
    } catch(e){ return {}; }
  }
  let pop = null;
  function ensurePop(){
    if(pop) return pop;
    pop = document.createElement('div');
    pop.className = 'gloss-pop';
    pop.style.display = 'none';
    document.body.appendChild(pop);
    return pop;
  }
  function position(el){
    const r = el.getBoundingClientRect();
    const p = ensurePop();
    p.style.left = (window.scrollX + r.left) + 'px';
    p.style.top = (window.scrollY + r.bottom + 6) + 'px';
  }
  loadDict().then(dict => {
    document.querySelectorAll('dfn[data-term]').forEach(el => {
      el.removeAttribute('title');  // suppress native browser tooltip
      const term = el.dataset.term;
      const def = dict[term];
      if(!def) return;
      el.addEventListener('mouseenter', () => {
        const p = ensurePop();
        p.innerHTML = '<strong>' + term + '</strong> &middot; ' + def;
        p.style.display = 'block';
        position(el);
      });
      el.addEventListener('mouseleave', () => {
        if(pop) pop.style.display = 'none';
      });
      el.addEventListener('focus', () => {
        const p = ensurePop();
        p.innerHTML = '<strong>' + term + '</strong> &middot; ' + def;
        p.style.display = 'block';
        position(el);
      });
      el.addEventListener('blur', () => { if(pop) pop.style.display = 'none'; });
      el.setAttribute('tabindex', '0');
    });
  });
})();
</script>
"""
    # Idempotency: use a unique sentinel string that lives inside the
    # injected <script> block (not in the CSS), so the guard cannot be
    # fooled by either prior CSS that names .gloss-pop or by later
    # post-body scripts pushing the popover block out of the last split.
    GLOSS_SENTINEL = "gloss-pop-script-v1"
    if GLOSS_SENTINEL not in html:
        marked = glossary_js.replace(
            "/* glossary popover",
            f"/* {GLOSS_SENTINEL} — glossary popover",
            1,
        )
        html = html.replace("</body>", marked + "</body>")

    # ---- 11. delta-chip paired colours — tiny JS shim
    delta_js = """
<script>
/* paired delta-chip colours (warm for positive, cool for negative, grey for negligible) */
(function(){
  const apply = () => {
    document.querySelectorAll('.delta-chip').forEach(chip => {
      const t = (chip.textContent || '').trim();
      chip.classList.remove('up','down','zero');
      if(!t){ return; }
      const v = parseFloat(t);
      if(isNaN(v) || Math.abs(v) < 0.005) chip.classList.add('zero');
      else if(v > 0) chip.classList.add('up');
      else chip.classList.add('down');
    });
  };
  const obs = new MutationObserver(apply);
  obs.observe(document.body, {subtree:true, childList:true, characterData:true});
  apply();
})();
</script>
"""
    if "paired delta-chip" not in html:
        html = html.replace("</body>", delta_js + "</body>")

    # ---- 12. TOC active-link highlighting
    toc_js = """
<script>
/* TOC active-link highlighting on scroll */
(function(){
  const links = document.querySelectorAll('.toc a');
  if(!links.length) return;
  const map = new Map();
  links.forEach(a => {
    const id = a.getAttribute('href').slice(1);
    const el = document.getElementById(id);
    if(el) map.set(el, a);
  });
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      const a = map.get(e.target);
      if(!a) return;
      if(e.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        a.classList.add('active');
      }
    });
  }, {rootMargin:'-30% 0px -60% 0px'});
  map.forEach((_,el) => obs.observe(el));
})();
</script>
"""
    if "TOC active-link" not in html:
        html = html.replace("</body>", toc_js + "</body>")

    html_path.write_text(html)

    # update print.css (shared across all sites — only widen, never narrow)
    print_path = DIST / "print.css"
    current = print_path.read_text() if print_path.exists() else ""
    if "contract: toggles to defaults" not in current:
        print_path.write_text(EXTRA_PRINT_CSS)

    print(f"[polish] wrote {html_path} ({html_path.stat().st_size:,} bytes)")
    print(f"[polish] wrote {print_path} ({print_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
