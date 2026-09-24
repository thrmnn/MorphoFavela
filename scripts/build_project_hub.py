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

import html
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import audit_hub_graph  # the same reachability walk backs outputs/_hub/map.html
from hubkit import (
    badge,
    breadcrumb,
    card,
    git_provenance,
    page,
    relativize_page,
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
    ],
    "CFD hand-off (exports)": [
        ("/docs/roughness_wall_treatment_explainer.md",
         "▶ Roughness & wall treatment — plain-language guide (start here)",
         "What z₀/z_d mean, the two-roles rule, why favela density breaks the estimate, and "
         "how z₀ becomes a wall boundary condition — with urban-physics schematics.", "doc"),
        ("/docs/cfd_parameter_estimation_plan.md", "CFD parameter-estimation plan + hand-off spec",
         "The morphometric→CFD inlet/wall spec: NaN-safe floored z0 + provenance "
         "(shipped), the MAR-P07 pilot request, F/I deferred-gated on it. "
         "/autoplan-reviewed, validate-first.", "doc"),
        ("/outputs/cross_site/roughness/patch_roughness.csv",
         "patch_roughness.csv — the CFD-inlet export (119 patches)",
         "The artifact the CFD repo consumes: z0_kan(θ) + NaN-safe floored z0 + "
         "flag_z0_floored + kanda_precfd_v1 provenance stamp, one row per patch.", "live"),
    ],
    "Plans & decisions": [
        ("/docs/autonomous_loop_plan.md", "Autonomous-loop plan + top blockers",
         "Ranked blockers to remove for a multi-hour parallel-agent loop.", "doc"),
        ("/docs/tr_audit.md", "TR coherence audit",
         "Bulletproofing punch list — criticals done, medium queued.", "doc"),
        ("/docs/dashboard_improvement_plan.md", "Dashboard improvement plan",
         "Council audit + prioritised backlog for the hub itself (this page).", "doc"),
        ("/docs/planetary_health_plan.md", "Planetary-health living plan",
         "Orchestrator + expert-panel track: the council of experts, the "
         "vitamin-D→TB pivot, the first real health number (TB×sun-deficit "
         "ρ=+0.80), and the append-only cycle log.", "doc"),
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


THUMB_W = 880  # display width cap for landing-page card thumbnails


def _png_size(path):
    """(width, height) from a PNG IHDR chunk — stdlib only, no Pillow."""
    import struct
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
        return struct.unpack(">II", head[16:24])
    return None


def _make_thumb(src: Path, img_url: str):
    """Write an ≤THUMB_W-wide thumbnail into outputs/_hub/thumbs/ and return its
    hub URL. Requires Pillow; returns None if unavailable so the hub degrades to
    full-res + intrinsic dimensions with no hard dependency."""
    try:
        from PIL import Image
    except ImportError:
        return None
    thumbs = OUT / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    name = img_url.lstrip("/").replace("/", "__")
    out = thumbs / name
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        with Image.open(src) as im:
            im.thumbnail((THUMB_W, THUMB_W * 20))  # cap width, preserve aspect
            im.save(out)
    return f"/outputs/_hub/thumbs/{name}"


def _img_attrs(img_url: str) -> dict:
    """{thumb,width,height} for a hub figure: intrinsic dimensions (stops layout
    shift) plus a downscaled thumbnail when the source is wide and Pillow exists."""
    src = ROOT / img_url.lstrip("/")
    size = _png_size(src) if src.exists() else None
    if not size:
        return {}
    w, h = size
    if w > THUMB_W and (thumb := _make_thumb(src, img_url)):
        return {"thumb": thumb, "width": THUMB_W, "height": round(h * THUMB_W / w)}
    return {"width": w, "height": h}


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


# ── Work packages — learned from the ledger, never hand-typed (HUBWP Ph.2) ──
# The ledger's own `_meta.runs_of_record` map is the source of truth for which
# runs are "of record" and what their ids are (docs/hub_wp_structure_spec.md).
# WP_RUN_ORDER/WP_TASK_ID/WP_LABEL are category labels this project already
# uses everywhere (WP-02, WP-04, ... in specs/tasks.json) — not run ids, not
# measured values — so declaring them here to fix display order + which
# tasks.json id backs each card's title does not violate "never type a
# number/run id/path that exists in a file"; every id, UTC, status and number
# on the cards below is read from ledger.json / the run's own manifest.
WP_RUN_ORDER = ["wp02_crossref", "wp04", "wp05", "g3", "wp06"]
WP_TASK_ID = {
    "wp02_crossref": "WP02", "wp04": "WP04", "wp05": "WP05FULL",
    "g3": "G3CARD", "wp06": "WP06",
}
WP_LABEL = {
    "wp02_crossref": "WP-02", "wp04": "WP-04", "wp05": "WP-05",
    "g3": "G3", "wp06": "WP-06",
}
BRISAVERSE_TASKS = Path.home() / "SCL" / "SCR" / "brisaverse" / "shared" / "facts" / "tasks.json"


def _latest_ledger_path() -> Path | None:
    hits = sorted(ROOT.glob("runs/wp07_ledger_*/ledger.json"))
    return hits[-1] if hits else None


def _tasks_titles() -> dict:
    """{task id: title} from brisaverse/shared/facts/tasks.json, read not
    written. Empty dict (cards degrade to the manifest-only description) if
    the sibling repo isn't checked out on this machine."""
    if not BRISAVERSE_TASKS.exists():
        return {}
    try:
        tasks = json.loads(BRISAVERSE_TASKS.read_text()).get("tasks", [])
    except (json.JSONDecodeError, OSError):
        return {}
    return {t["id"]: t.get("title", "") for t in tasks if "id" in t}


def _entries_for_run(entries: dict, run_id: str) -> dict:
    return {eid: e for eid, e in entries.items()
            if e.get("source", {}).get("run_id") == run_id}


def _fmt_num(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _headline_sample(run_entries: dict, k: int = 6) -> list:
    """Up to `k` entries spread evenly across the sorted id list — a
    deterministic, data-driven sample (never a hand-picked id) so every run's
    card shows real, spread-out headline numbers instead of always the same
    alphabetically-first handful."""
    ids = sorted(run_entries)
    if len(ids) <= k:
        chosen = ids
    else:
        step = len(ids) / k
        chosen = [ids[int(i * step)] for i in range(k)]
    return [(eid, run_entries[eid]["value"], run_entries[eid].get("unit", ""))
            for eid in chosen]


def _run_utc(run_dir: Path, run_entries: dict) -> str:
    for name in ("manifest.json", "summary.json"):
        f = run_dir / name
        if f.exists():
            try:
                utc = json.loads(f.read_text()).get("_utc")
            except (json.JSONDecodeError, OSError):
                utc = None
            if utc:
                return utc
    return next((e["source"]["run_utc"] for e in run_entries.values()), "")


WP_REPORTS_DIR = "wp_reports"  # never name this (or a segment of it) "runs" —
# build_mirror_manifest()/_is_withheld() blanket-forbids that exact path
# segment anywhere under outputs/_hub/ as a per-cell-layer guard (L1), and a
# root-absolute href into runs/ would otherwise auto-mirror to exactly that
# segment. So every run report is copied/rendered explicitly into this
# repo-owned, already-"outputs/"-prefixed directory instead.


def _copy_report_file(src: Path, run_name: str) -> str:
    dst = DOCS / WP_REPORTS_DIR / f"{run_name}__{src.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    return f"/outputs/_hub/docs/{WP_REPORTS_DIR}/{dst.name}"


def _run_report_link(run_dir: Path, prov) -> tuple:
    """(filename, href) for the run's own report: its first *.md (rendered to
    HTML like any other doc, so it reads like the rest of the hub) if one
    exists, else summary.json, else the largest non-manifest *.json, else
    manifest.json — discovered by glob, never a hand-typed filename."""
    mds = sorted(run_dir.glob("*.md"))
    if mds:
        src = mds[0]
        out = DOCS / WP_REPORTS_DIR / f"{run_dir.name}__{src.stem}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        back = breadcrumb([("← Project hub", "../../index.html"), (src.stem, None)])
        render_doc_page(src, out, crumb=back, provenance=prov,
                        base=_doc_base(src), root=ROOT, mirror_dir=None)
        return src.name, f"/outputs/_hub/docs/{WP_REPORTS_DIR}/{run_dir.name}__{src.stem}.html"
    summary = run_dir / "summary.json"
    if summary.exists():
        return "summary.json", _copy_report_file(summary, run_dir.name)
    others = sorted(p for p in run_dir.glob("*.json") if p.name != "manifest.json")
    if others:
        return others[0].name, _copy_report_file(others[0], run_dir.name)
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        return "manifest.json", _copy_report_file(manifest, run_dir.name)
    return run_dir.name, None


def _wp_gate_note(meta: dict, run_id: str, run_entries: dict, ledger_path: Path) -> str:
    """What this run feeds downstream — read from the ledger's own _meta
    pointers (definition_of_record, engine_acceptance_source) plus the plain
    fact that it feeds this ledger, never an invented claim."""
    prefixes = {eid.split(".", 1)[0] for eid in run_entries}
    parts = []
    for k, v in meta.get("definition_of_record", {}).items():
        if k.split(".", 1)[0] in prefixes:
            parts.append(f"definition of record: {v}")
    eas = meta.get("engine_acceptance_source", "")
    if run_id and run_id in eas:
        parts.append(f"engine acceptance source: {eas}")
    parts.append(f"feeds {ledger_path.relative_to(ROOT)} ({len(run_entries)} numbers)")
    return " · ".join(parts)


def work_packages_section(prov):
    """One card per run of record, in C′ causal order, sourced entirely from
    the ledger's own `_meta.runs_of_record` map (docs/hub_wp_structure_spec.md
    Ph.2 item 1) — never a hand-typed run id or number."""
    ledger_path = _latest_ledger_path()
    if ledger_path is None:
        return ""
    ledger = json.loads(ledger_path.read_text())
    meta = ledger.get("_meta", {})
    runs_of_record = meta.get("runs_of_record", {})
    entries = ledger.get("entries", {})
    titles = _tasks_titles()

    cards = []
    for key in WP_RUN_ORDER:
        run_id = runs_of_record.get(key)
        if not run_id:
            continue
        run_dir = ROOT / "runs" / run_id
        if not run_dir.exists():
            continue
        run_entries = _entries_for_run(entries, run_id)
        label = WP_LABEL.get(key, key)
        title = titles.get(WP_TASK_ID.get(key, ""), "")
        status = next((e.get("status") for e in run_entries.values()), ledger.get("status", ""))
        utc = _run_utc(run_dir, run_entries)
        report_name, report_href = _run_report_link(run_dir, prov)
        nums = _headline_sample(run_entries)
        nums_txt = " · ".join(f"{eid}={_fmt_num(v)}{(' ' + u) if u else ''}"
                              for eid, v, u in nums)
        desc = title or f"{len(run_entries)} ledger numbers sourced from this run."
        if nums_txt:
            desc += f" Headline: {nums_txt}."
        meta_txt = (f"{run_id} · {utc} · status: {status} · report: {report_name} · "
                    f"{_wp_gate_note(meta, run_id, run_entries, ledger_path)}")
        cards.append(card(f"{label} — run of record", desc, report_href,
                          meta=meta_txt, kind="ok", badge_label=label, new_tab=False))
    return section("Work packages — runs of record (C′ causal order)", cards,
                   anchor="work-packages")


def headline_section(prov):
    """Hero: the headline figure(s) as zoomable image cards, deep-linked to their
    captioned place in the signature gallery."""
    cards = [
        card(title, desc, f"{SIG_GAL}#{anchor}", img=f"{SIG_DIR}/{fn}",
             meta="headline result · WHO 2 h/day winter floor", kind="ok",
             badge_label="Headline", **_img_attrs(f"{SIG_DIR}/{fn}"))
        for fn, anchor, title, desc in HEADLINE_FIGS
        if (ROOT / f"{SIG_DIR}/{fn}".lstrip("/")).exists()
    ]
    return section("Headline result — morphotype predicts winter-sun failure",
                   cards, anchor="headline")


def facade_solar_section(prov):
    """Façade-level solar (independent Ladybug run) + our street cross-check."""
    cards = [
        card(title, desc, MZ_DIR + href, img=MZ_DIR + href,
             meta="5-favela façade · WHO 2 h/day", kind="info",
             badge_label="Partner", **_img_attrs(MZ_DIR + href))
        for href, title, desc in FACADE_FIGS
        if (ROOT / (MZ_DIR + href).lstrip("/")).exists()
    ]
    # the detailed single-site accuracy report belongs with the Mingze work, scoped
    vdg = "/outputs/comparative/vidigal_vs_mingze/report/index.html"
    if (ROOT / vdg.lstrip("/")).exists():
        cards.append(card("Vidigal accuracy report (detailed)",
                          "Vidigal only — our raycast vs Mingze's Ladybug: per-observer "
                          "agreement + four ranked disagreement hypotheses.",
                          vdg, meta="Vidigal only · Ladybug ↔ raycast",
                          kind="info", badge_label="Partner"))
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
             fig, img=fig, meta="TR §10.9 · dissolved λf", kind="info",
             **_img_attrs(fig))
    return section("Grid-resolution sensitivity (MAUP)", [c], anchor="maup")


def ventilation_section(prov):
    """Dedicated §5.6 gallery so the geometry-only ventilation tendencies are a
    first-class hub item (image cards with lightbox, linking to the report)."""
    cards = [
        card(title, desc, VENT_TR, img=f"{VENT_EXPORTS}/{fn}",
             meta="pre-CFD tendency · not adequacy (τ CFD-gated)", kind="ok",
             new_tab=False, **_img_attrs(f"{VENT_EXPORTS}/{fn}"))
        for fn, title, desc in VENT_FIGS
        if (ROOT / "outputs/paper_figures/exports" / fn).exists()
    ]
    return section("Geometric ventilation tendencies (§5.6)", cards, anchor="ventilation")


def _render_latest_item(n, u, d, date=""):
    """One <li> for the Latest changelog: an optional date prefix, a trailing
    '— NEW' becomes a pill, and both label and gloss are HTML-escaped."""
    m = re.search(r"\s*[—-]\s*NEW\s*$", n)
    label = n[:m.start()] if m else n
    pill = (badge("info", "NEW") + " ") if m else ""
    ds = f'<span class="date">{html.escape(date)}</span>' if date else ""
    return (f'<li>{ds}{pill}<a href="{u}">{html.escape(label)}</a> '
            f'<span class="gloss">— {html.escape(d)}</span></li>')


def _latest_target_exists(url):
    """True if a Latest changelog target resolves. Hub-rendered doc pages are
    produced later in the same run, so gate them on their source markdown."""
    file = url.split("#", 1)[0]
    hub_src = {
        "/outputs/_hub/docs/technical_report.html": "docs/technical_report/technical_report.md",
        "/outputs/_hub/docs/tr_audit.html": "docs/tr_audit.md",
        "/outputs/_hub/docs/morphology_overview.html": "docs/morphology_overview.md",
        "/outputs/_hub/docs/dashboard_improvement_plan.html": "docs/dashboard_improvement_plan.md",
        "/outputs/_hub/docs/cfd_parameter_estimation_plan.html": "docs/cfd_parameter_estimation_plan.md",
        "/outputs/_hub/docs/roughness_wall_treatment_explainer.html": "docs/roughness_wall_treatment_explainer.md",
        "/outputs/_hub/health.html": "outputs/paper_figures/cross_site_stats.json",
        "/outputs/_hub/docs/planetary_health_plan.html": "docs/planetary_health_plan.md",
    }.get(file)
    return (ROOT / (hub_src or file.lstrip("/"))).exists()


def build_callout(prov):
    """Top panel: newest results (direct links) + the live work queue, so new
    figures are never hard to find."""
    if (ROOT / "docs/work_queue.md").exists():
        wq_md = ROOT / "docs/work_queue.md"
        render_doc_page(wq_md, DOCS / "work_queue.html",
                        crumb=breadcrumb([("← Project hub", "../index.html"),
                                          ("Work queue", None)]), provenance=prov,
                        base=_doc_base(wq_md), root=ROOT, mirror_dir=DOCS)
    hub = "/outputs/_hub"
    tr = f"{hub}/docs/technical_report.html"
    # A dated changelog of new/updated results. Each lands on the EXACT figure or
    # TR section it documents — never a bare on-page anchor that duplicates a
    # section below. (date, label, url, gloss); existence-gated at render.
    latest = [
        ("2026-09-16", "Maré deliverables for PI review — brief, deck, Folha de Rua refresher",
         f"{hub}/mare_review/index.html",
         "the morphology brief (6 pp), the companion slide deck, and the five refreshed "
         "Folha de Rua sheets with their interactive dashboards, each through a critic "
         "loop; drafts, with the disclosure decisions still open"),
        ("2026-07-28", "Health probe update — out-of-sample favela weakens TB × sun-deficit",
         f"{hub}/health.html#health-probe",
         "the first out-of-sample favela (Cidade de Deus) drops the rank correlation from "
         "ρ≈+0.80 (n=5) to ρ≈+0.26 (n=6, p≈0.66) — consistent with a small-sample artefact; "
         "the probe now says treat it as a hypothesis to test at larger n, not a finding. "
         "A separate ventilation-vs-TB check (ρ=+1.00) was adversarially audited and held "
         "back as fragile/collinear. See the living plan for the full cycle log"),
        ("2026-07-06", "Health outcome probe — TB × sun-deficit (Grade C, audit-gated)",
         f"{hub}/health.html#health-probe",
         "orchestrator + expert-panel loop: real Rio TB rank-tracks our winter "
         "sun-deficit (ρ≈+0.80, n=5, exact p≈0.13, NOT significant; reverses at AP "
         "scale) — an adversarial audit caught + fixed a denominator error before it "
         "shipped; vitamin-D the sourced mechanism; see the living plan for the cycle log"),
        ("2026-07-04", "Planetary-health exposure pathways (panel-graded)",
         f"{hub}/health.html",
         "a Lancet-style panel maps built form to WHO-referenced exposure — winter-"
         "sun deprivation (Grade A, AUC 0.90), equity, airborne, heat — each "
         "evidence-graded A–D; modelled exposure, not measured health"),
        ("2026-07-03", "Roughness & wall treatment — plain-language guide",
         f"{hub}/docs/roughness_wall_treatment_explainer.html",
         "review-friendly walkthrough with urban-physics schematics: z₀/z_d, the "
         "two-roles rule, the skimming-regime break, and z₀→wall-BC (k_s < y_P)"),
        ("2026-07-03", "CFD hand-off: floored z0 + provenance, pilot requested",
         f"{hub}/docs/cfd_parameter_estimation_plan.html",
         "/autoplan-reviewed, validate-first: shipped NaN-safe floored z0 + "
         "kanda_precfd_v1 stamp on patch_roughness.csv; requested the MAR-P07 pilot "
         "as the R-C anchor; F/I deferred-gated on it"),
        ("2026-07-02", "Dashboard credibility pass (council audit)",
         f"{hub}/docs/dashboard_improvement_plan.html",
         "hero headline figure, human contents, accessibility + HTML-validity fixes "
         "— see the plan and prioritised backlog"),
        ("2026-07-01", "Façade-level solar + Ladybug cross-check (§5.4.1)",
         f"{tr}#541-faade-level-extension-and-street-level-cross-check-indep",
         "Mingze's façade run (50–72 % WHO-2h deprivation) + our street cross-check: "
         "ordering agrees, Maré least deprived, Rocinha façade outlier 72 %"),
        ("2026-07-01", "MAUP resolution curve 5–30 m (§10.9)",
         f"{tr}#109-sensitivity-to-grid-resolution-maup",
         "full grid-size sweep; monotonic regime drift, cross-site ordering "
         "preserved (Spearman ρ = 0.90)"),
        ("2026-06-28", "Geometric ventilation tendencies (§5.6)",
         f"{tr}#56-geometric-ventilation-tendencies--gated-pre-cfd",
         "lateral depth · regime×depth susceptibility · wind exposure · "
         "multi-constraint index (0–3); geometry-only, τ-gated"),
        ("2026-06-27", "TR §6.6 roughness — invalidity caveat",
         f"{tr}#66-aerodynamic-roughness-z0-zd",
         "per-cell z0/zd invalid 53–75 %; the method envelope is the result"),
        ("2026-06-25", "Typology → environmental failure (headline)",
         f"{GAL}#fig-typology_failure_lookup",
         "morphotype predicts WHO-2h sun failure 14 % → 73 %, transfers "
         "leave-one-site-out"),
        ("2026-06-25", "Blind cross-site risk map (8 favelas)",
         f"{GAL}#fig-typology_blind_riskmap",
         "one continuous fabric-vector model; beats the morphotype-rate blind map"),
    ]
    # The fold belongs to results, not to a changelog: the critic measured this
    # page at 10,282 px with the entire feed inlined. Newest few here, the rest
    # one click away (2026-09-17).
    LATEST_ON_INDEX = 6
    live = [(date, n, u, d) for date, n, u, d in latest if _latest_target_exists(u)]
    shown, rest = live[:LATEST_ON_INDEX], live[LATEST_ON_INDEX:]
    items = "".join(_render_latest_item(n, u, d, date=date) for date, n, u, d in shown)
    more_latest = (f'<nav class="more"><a href="/outputs/_hub/changelog.html">'
                   f'🗓 {len(rest)} earlier entries →</a> <span class="gloss">'
                   f'the rest of the changelog.</span></nav>' if rest else "")
    wq = ('<nav class="more"><a href="/outputs/_hub/docs/work_queue.html">'
          '📋 Full work queue →</a> <span class="gloss">what is in progress, '
          'queued, and gated.</span></nav>'
          if (ROOT / "docs/work_queue.md").exists() else "")
    if rest:
        (OUT / "changelog.html").write_text(_relativize(page(
            "Changelog — earlier results",
            "Entries older than the newest few on the project hub.",
            '<section><h2 id="earlier">Earlier results</h2><div class="callout"><ul>'
            + "".join(_render_latest_item(n, u, d, date=date) for date, n, u, d in rest)
            + '</ul></div></section>',
            crumb=breadcrumb([("Project hub", "/outputs/_hub/index.html"),
                              ("Changelog", None)]),
            provenance=prov)), encoding="utf-8")
    heading = "Latest" if wq else "Latest results"
    glossary = (
        '<details class="glossary"><summary>Glossary</summary>'
        '<span class="gloss">'
        'WHO-2h ≥ 2 h direct sun/day floor · SVF sky-view factor · '
        'λf frontal-area ratio · λp plan-area ratio · σH building-height s.d. · '
        'z0/zd aerodynamic roughness length / displacement height · '
        'τ wall shear stress (CFD-gated) · LOSO leave-one-site-out'
        '</span></details>')
    return (f'<section><h2 id="latest">{heading}</h2>'
            f'<div class="callout">'
            f'<p class="lead">Newest results — click straight in:</p>'
            f'<ul>{items}</ul>{more_latest}{wq}{glossary}'
            f'</div></section>')


def _doc_card(url, name, desc, prov):
    src = ROOT / url.lstrip("/")
    back = breadcrumb([("← Project hub", "../index.html"), (src.stem, None)])
    render_doc_page(src, DOCS / f"{src.stem}.html", crumb=back, provenance=prov,
                    base=_doc_base(src), root=ROOT, mirror_dir=DOCS)
    return card(name, desc, f"/outputs/_hub/docs/{src.stem}.html", meta=url, kind="doc",
                new_tab=False)


ORG_DIAGRAM_SVG = """
<svg viewBox="0 0 980 200" role="img" aria-label="SITETERR data flow"
     style="width:100%;height:auto;font:12px system-ui,sans-serif">
  <defs>
    <marker id="siteterr-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#6b6b6b"/>
    </marker>
  </defs>
  <style>
    .box{fill:#fff;stroke:#6b6b6b;stroke-width:1}
    .lbl{fill:#1b1b1b;font-weight:600}
    .file{fill:#6b6b6b;font-family:monospace;font-size:10px}
    .edge{stroke:#6b6b6b;stroke-width:1;fill:none;marker-end:url(#siteterr-arrow)}
  </style>
  <rect class="box" x="10" y="70" width="150" height="60" rx="4"/>
  <text class="lbl" x="85" y="92" text-anchor="middle">Registry</text>
  <text class="file" x="85" y="108" text-anchor="middle">config/sites.yaml</text>
  <text class="file" x="85" y="120" text-anchor="middle">src/sites/territory.py</text>

  <path class="edge" d="M160,100 H230"/>
  <rect class="box" x="230" y="70" width="180" height="60" rx="4"/>
  <text class="lbl" x="320" y="92" text-anchor="middle">Per-site territory</text>
  <text class="file" x="320" y="108" text-anchor="middle">build_site_territory.py</text>
  <text class="file" x="320" y="120" text-anchor="middle">data/&lt;site&gt;/territory.gpkg + _provenance.json</text>

  <path class="edge" d="M410,90 C450,90 450,30 490,30"/>
  <path class="edge" d="M410,110 C450,110 450,170 490,170"/>

  <rect class="box" x="490" y="0" width="190" height="60" rx="4"/>
  <text class="lbl" x="585" y="22" text-anchor="middle">Site runs</text>
  <text class="file" x="585" y="38" text-anchor="middle">build_site_dashboard.py</text>
  <text class="file" x="585" y="50" text-anchor="middle">build_html_dashboard.py</text>

  <rect class="box" x="490" y="140" width="190" height="60" rx="4"/>
  <text class="lbl" x="585" y="162" text-anchor="middle">Citywide run</text>
  <text class="file" x="585" y="178" text-anchor="middle">wp05_full.match_favela_group</text>
  <text class="file" x="585" y="190" text-anchor="middle">(target from citywide_rule)</text>

  <path class="edge" d="M680,170 C720,170 720,110 750,110"/>
  <rect class="box" x="750" y="80" width="130" height="60" rx="4"/>
  <text class="lbl" x="815" y="102" text-anchor="middle">Ledger</text>
  <text class="file" x="815" y="118" text-anchor="middle">wp07_ledger.py</text>

  <path class="edge" d="M680,30 C900,30 900,60 900,80"/>
  <path class="edge" d="M880,110 H900"/>
  <rect class="box" x="850" y="0" width="130" height="200" rx="4" style="fill:none;stroke:none"/>
  <text class="lbl" x="915" y="14" text-anchor="middle" font-size="11">figures</text>
  <text class="lbl" x="915" y="28" text-anchor="middle" font-size="11">briefs</text>
  <text class="lbl" x="915" y="42" text-anchor="middle" font-size="11">dashboards</text>
  <text class="lbl" x="915" y="56" text-anchor="middle" font-size="11">cockpit</text>
</svg>
"""


def _territory_data(site: str):
    from src.sites.territory import load_territory
    prov_path = ROOT / "data" / site / "territory_provenance.json"
    if not prov_path.exists():
        return None
    return json.loads(prov_path.read_text(encoding="utf-8"))


def _territory_map_img(site: str) -> str | None:
    name = "mare_territory_map" if site == "maré" else f"{site}_territory_map"
    p = ROOT / "outputs" / site / "territory" / f"{name}.png"
    if not p.exists():
        return None
    return "/" + str(p.relative_to(ROOT))


def _territory_site_html(site: str, display: str, tp: dict) -> str:
    de = tp["data_extent"]["area_m2"] / 1e6
    sa = tp["study_area"]["area_m2"] / 1e6
    cw = tp["citywide"]["area_m2"] / 1e6
    bshare = tp.get("building_share_inside") or {}

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    rows = f"""
    <table>
      <thead><tr><th>Boundary</th><th>Area (km²)</th><th>Buildings inside</th></tr></thead>
      <tbody>
        <tr><td>Data extent</td><td>{de:.3f}</td><td>{pct(bshare.get('data_extent'))}</td></tr>
        <tr><td>Study area ({tp['study_area']['kind']})</td><td>{sa:.3f}</td><td>{pct(bshare.get('study_area'))}</td></tr>
        <tr><td>Citywide ({tp['citywide']['match_method']}, {tp['citywide']['n_polygons']} polygon(s))</td>
            <td>{cw:.3f}</td><td>{pct(bshare.get('citywide'))}</td></tr>
      </tbody>
    </table>"""

    overlap = tp.get("overlap_study_area_vs_citywide")
    overlap_html = ""
    if overlap:
        overlap_html = (f"<p><strong>Study area vs. citywide:</strong> IoU "
                        f"{overlap['iou']:.3f}"
                        f"{' — identical boundary' if overlap['identical_boundary'] else ' — boundaries differ'}"
                        f"</p>")

    inferred = tp.get("inferred_matches") or []
    inferred_html = ""
    if inferred:
        items = "".join(f"<li>{html.escape(m['community'])} — {html.escape(m.get('note') or 'inferred match')}</li>"
                        for m in inferred)
        inferred_html = f"<p><strong>Inferred name matches ({len(inferred)}):</strong></p><ul>{items}</ul>"

    note = tp.get("definition_note")
    note_html = f'<p class="pill terra" style="display:block;white-space:normal">{html.escape(note)}</p>' if note else ""

    candidates = tp.get("study_area_candidates") or {}
    cand_html = ""
    if candidates:
        items = "".join(
            f"<li><strong>{html.escape(c['label'])}</strong> — {c['area_m2']/1e6:.3f} km² · "
            f"{html.escape(str(c.get('status', 'candidate')))}</li>"
            for c in candidates.values())
        cand_html = f"<p><strong>Study-area candidates (not active):</strong></p><ul>{items}</ul>"

    img = _territory_map_img(site)
    img_html = ""
    if img:
        attrs = _img_attrs(img)
        dim = f' width="{attrs["width"]}" height="{attrs["height"]}"' if attrs.get("width") else ""
        img_html = (f'<img src="{attrs.get("thumb", img)}"{dim} loading="lazy" '
                    f'style="max-width:420px;width:100%;height:auto;border:1px solid #ddd" '
                    f'alt="{html.escape(display)} territory map" '
                    f'onclick="event.preventDefault();zoom(\'{img}\',\'{_pz_js_attr(display)}\')">')

    return f"""
    <section>
      <h2 id="site-{_slug_ascii(site)}">{html.escape(display)}</h2>
      <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start">
        <div style="flex:0 0 auto">{img_html}</div>
        <div style="flex:1 1 320px;min-width:280px">
          {rows}
          {overlap_html}
          {cand_html}
          {inferred_html}
          {note_html}
        </div>
      </div>
    </section>"""


def write_territory_page(prov):
    """SITETERR territory review page: the Organisation flow (registry ->
    per-site territory.gpkg -> site/citywide runs -> ledger -> figures,
    briefs, dashboards, cockpit) then one section per site with its map,
    areas, building shares and inferred-match list from
    data/<site>/territory_provenance.json. Degrades to (None, None) when no
    site has been built yet (scripts/build_site_territory.py --all)."""
    try:
        from src.sites.territory import load_sites_config
        registry = load_sites_config()
    except Exception:
        return None, None

    sections_html = []
    thumb = None
    for site, cfg in registry.items():
        tp = _territory_data(site)
        if not tp:
            continue
        sections_html.append(_territory_site_html(site, cfg["display_name"], tp))
        thumb = thumb or _territory_map_img(site)
    if not sections_html:
        return None, None

    crumb = breadcrumb([("← Project hub", "index.html"), ("Site territories", None)])
    body = (
        '<p class="lead">What "the site" means, declared once per site '
        '(<code>config/sites.yaml</code>, loader <code>src/sites/territory.py</code>) '
        'instead of six independent hardcodings. Every number below is read from '
        '<code>data/&lt;site&gt;/territory_provenance.json</code>, written by '
        '<code>scripts/build_site_territory.py</code>.</p>'
        '<h2 id="organisation">Organisation</h2>' + ORG_DIAGRAM_SVG
        + "".join(sections_html)
    )
    out = OUT / "territory.html"
    out.write_text(_relativize(page(
        "Site territories", badge("ok", f"{len(sections_html)} of {len(registry)} sites built"),
        body, crumb=crumb, provenance=prov)))
    return "/outputs/_hub/territory.html", thumb


def territory_section(prov):
    url, thumb = write_territory_page(prov)
    if not url:
        return ""
    cards = [card(
        "Site territories",
        "One declared boundary per site — data extent, study area, citywide "
        "definition and subunits, with a map and provenance per site. "
        "Replaces six places a site's boundary used to be typed.",
        url, img=thumb, kind="ok", badge_label="Start here", new_tab=False,
        **(_img_attrs(thumb) if thumb else {}))]
    return section("Territory", cards, anchor="territory")


def sites_section(prov):
    cards = []
    for s in CAMPAIGN + [s for s in SITE_NAMES if s not in CAMPAIGN]:
        idx = DASH / s / "index.html"
        if idx.exists():
            camp = s in CAMPAIGN
            tag = "campaign site" if camp else "calibration site"
            cards.append(card(SITE_NAMES[s], f"Interactive per-favela dashboard — {tag}.",
                              os.path.relpath(idx, OUT),
                              kind="ok" if camp else "info",
                              badge_label="Campaign" if camp else "Calibration"))
    if (DASH / "index.html").exists():
        cards.append(card("All sites — interactive index",
                          "Combined dashboard index for every favela.",
                          os.path.relpath(DASH / "index.html", OUT), kind="info"))
    return section("Sites", cards, anchor="sites")


# Two print scales: the whole favela as a 5 cm massing box (DSM heightfield) and
# the CFD analysis patch at 1:1000 (individual buildings — the physical ↔ digital
# twin). Cards link the downloadable STL and lightbox the axonometric preview.
PRINT_PATCHES = [
    ("rocinha", "ROC-P18"), ("maré", "MAR-P20"),
    ("riodaspedras", "RDP-P20"), ("vidigal", "VDG-P07"),
]


def _print_card(json_path: Path, preview: Path, kind: str, badge_label: str) -> str:
    st = json.loads(json_path.read_text())
    stl = "/" + str(json_path.with_suffix(".stl").relative_to(ROOT))
    w, d, h = st["model_mm"]
    if "grid" in st:  # site DSM model
        title = f"{SITE_NAMES.get(st['site'], st['site'])} — full site"
        desc = (f"Whole favela as a single watertight massing solid (draped DSM). "
                f"{st['n_buildings']:,} buildings on a {st['cell_m']:.0f} m grid.")
        meta = f"1:{st['scale_denom']} · {w:.0f}×{d:.0f}×{h:.0f} mm · relief {st['relief_mm']:.0f} mm"
    else:  # patch model
        title = f"{st['patch_id']} — CFD analysis patch"
        desc = (f"The 100 m patch at 1:1000 — individual buildings on terrain. "
                f"Print the pilot patch to pair the physical twin with its CFD run.")
        meta = f"{SITE_NAMES.get(st['site'], st['site'])} · 1:{st['scale_denom']} · {w:.0f}×{d:.0f}×{h:.0f} mm · {st['n_buildings']} buildings"
    img = "/" + str(preview.relative_to(ROOT)) if preview.exists() else None
    attrs = _img_attrs(img) if img else {}
    return card(title, desc, stl, img=img, meta=meta, kind=kind,
                badge_label=badge_label, **attrs)


# Version-A texture tiles surfaced on the Textures page: (field, site, patch, blurb).
TEXTURE_TILES = [
    ("sunlight", "vidigal", "VDG-P07",
     "Winter sun-hours (4 classes) as ground-only relief on a steep patch — "
     "the slope stress-test. Higher shade → denser texture."),
    ("ventilation", "riodaspedras", "RDP-P20",
     "Pedestrian ventilation from the RDP-P20 CFD return (8-direction simpleFoam, "
     "wind-rose-weighted |U|/U_ref, quartiles). More stagnant → denser texture. "
     "Proxy for local mean age of air; a true LMA field needs a scalar-transport run."),
]


def _tile_cards(field, site, patch):
    """Review-figure card + 3 STL download cards for one texture tile."""
    tdir = ROOT / "outputs" / site / "print" / "texture_tile"
    review = tdir / f"{patch}_texture_{field}_review.png"
    if not review.exists():
        return []
    rurl = "/" + str(review.relative_to(ROOT))
    tag = "winter sun-hours" if field == "sunlight" else "CFD ventilation |U|/U_ref"
    cards = [card(
        f"{patch} — {field} texture (review)",
        f"Three ground-only treatments compared: stippling / contour bands / "
        f"directional hatching. FDM pick: stippling. Field: {tag}.",
        rurl, img=rurl, meta=f"{field} · {patch} · 150 mm tile", kind="ok",
        badge_label="Review", **_img_attrs(rurl))]
    for variant, lbl in (("stipple", "V1 stipple"), ("contour", "V2 contour"), ("hatch", "V3 hatch")):
        js = tdir / f"{patch}_texture_{field}_{variant}.json"
        if not js.exists():
            continue
        st = json.loads(js.read_text())
        w, d, h = st["model_mm"]
        cards.append(card(
            f"{lbl} — {patch} {field} STL",
            f"Watertight variant · texture {st['texture_depth_mm']} mm deep · download.",
            "/" + str(js.with_suffix(".stl").relative_to(ROOT)),
            meta=f"{w:.0f}×{d:.0f}×{h:.0f} mm · 1:{st['scale_denom']}",
            kind="info", badge_label="STL"))
    return cards


def _print_job_section():
    """The one-plate Ender-3 print job: bed layout + sliced G-code + plate STL."""
    pdir = ROOT / "outputs/_hub/print_plate"
    js = pdir / "favelas_plate.json"
    if not js.exists():
        return ""
    st = json.loads(js.read_text())
    layout = "/outputs/_hub/print_plate/favelas_plate_layout.png"
    t = f"{st['print_time_h']} h" if st.get("print_time_h") else "slice pending"
    fil = f" · {st['filament_g']} g PLA" if st.get("filament_g") else ""
    cards = [card(
        f"Bed layout — all 5 sites on one Ender-3 plate ({t}{fil})",
        f"{st['n_sites']} sites shelf-packed on a {st['bed_mm']:.0f}×{st['bed_mm']:.0f} mm bed, "
        f"footprint {st['footprint_mm'][0]:.0f}×{st['footprint_mm'][1]:.0f} mm, "
        f"{st['layer_mm']} mm layers, {st['material']}, no supports.",
        layout, img=layout, meta=f"{st['printer']} · {st['nozzle_mm']} mm nozzle",
        kind="ok", badge_label="Print job", **_img_attrs(layout))]
    gc = pdir / "favelas_plate_ender3.gcode"
    if gc.exists():
        cards.append(card("Sliced G-code — favelas_plate_ender3.gcode",
                          f"Ready-to-print, {gc.stat().st_size/1e6:.1f} MB. Estimated {t}{fil}. "
                          "Re-slice with scripts/build_print_plate.py if you change the profile.",
                          "/outputs/_hub/print_plate/favelas_plate_ender3.gcode",
                          meta="Marlin · Ender-3 · PLA 0.2 mm", kind="info", badge_label="G-code"))
    stl = pdir / "favelas_plate_ender3.stl"
    if stl.exists():
        cards.append(card("Combined plate STL (all 5, arranged)",
                          "The origin-centred multi-object plate if you'd rather slice it "
                          "yourself in Creality Print / PrusaSlicer.",
                          "/outputs/_hub/print_plate/favelas_plate_ender3.stl",
                          meta=f"{st['triangles']:,} triangles", kind="info", badge_label="STL"))
    return section("One-plate print job (Ender-3)", cards, anchor="print-job")


def write_prints_pages(prov):
    """Write two standalone gallery pages (all sites; texture treatments) and
    return {name: (url, thumb)} so the hub can link them with a preview."""
    pages = {}

    # --- Page 1: all five site artifacts + the CFD patch prints, one page ---
    site_cards = []
    for site in CAMPAIGN:
        pdir = ROOT / "outputs" / site / "print"
        for js in sorted(pdir.glob(f"{site}_site_1to*.json")):
            site_cards.append(_print_card(js, pdir / f"{site}_site_preview.png",
                                          "ok", "Site model"))
    patch_cards = []
    for site, patch in PRINT_PATCHES:
        js = ROOT / "outputs" / site / "print" / f"{patch}_1to1000.json"
        if js.exists():
            patch_cards.append(_print_card(js, js.parent / f"{patch}_preview.png",
                                           "info", "Patch · 1:1000"))
    crumb = breadcrumb([("← Project hub", "index.html"), ("Physical twins — sites", None)])
    body = (_print_job_section()
            + section("Full-site models — 5 cm framed artifacts", site_cards, anchor="sites-print")
            + section("CFD analysis-patch prints (1:1000)", patch_cards, anchor="patch-print"))
    sub = f'{badge("ok", f"{len(site_cards)} site models")} {badge("info", "premium framed base · engraved · water")}'
    (OUT / "prints_sites.html").write_text(_relativize(page(
        "Physical twins — full-site 3D prints", sub, body, crumb=crumb, provenance=prov)))
    first = next((ROOT / "outputs" / s / "print" / f"{s}_site_preview.png"
                  for s in CAMPAIGN if (ROOT / "outputs" / s / "print" / f"{s}_site_preview.png").exists()), None)
    pages["sites"] = ("/outputs/_hub/prints_sites.html",
                      "/" + str(first.relative_to(ROOT)) if first else None)

    # --- Page 2: texture treatments (sunlight + airflow) ---
    tcards = []
    for field, site, patch, blurb in TEXTURE_TILES:
        cc = _tile_cards(field, site, patch)
        if cc:
            tcards.append(("Sunlight — winter sun-hours" if field == "sunlight"
                           else "Airflow — pedestrian ventilation (CFD)", field, cc, blurb))
    tbody = ""
    for heading, field, cc, blurb in tcards:
        tbody += f'<p class="lead">{html.escape(blurb)}</p>' + section(heading, cc, anchor=f"tex-{field}")
    if (notes := ROOT / "docs/print_texture_notes.md").exists():
        tbody = section("Notes & recommendation",
                        [_doc_card("/docs/print_texture_notes.md",
                                   "Print-risk notes + FDM recommendation",
                                   "Per-variant risk + why stippling wins on FDM.", prov)],
                        anchor="tex-notes") + tbody
    crumb2 = breadcrumb([("← Project hub", "index.html"), ("Physical twins — textures", None)])
    sub2 = f'{badge("ok", "Version A · realistic")} {badge("info", "sunlight + airflow · 3 treatments each")}'
    (OUT / "prints_textures.html").write_text(_relativize(page(
        "Physical twins — performance-texture tiles", sub2, tbody, crumb=crumb2, provenance=prov)))
    sun = ROOT / "outputs/vidigal/print/texture_tile/VDG-P07_texture_sunlight_review.png"
    pages["textures"] = ("/outputs/_hub/prints_textures.html",
                         "/" + str(sun.relative_to(ROOT)) if sun.exists() else None)
    return pages


def prints_section(prov):
    """Two nav cards → the standalone sites page and the textures page."""
    pages = write_prints_pages(prov)
    cards = []
    su, sthumb = pages.get("sites", (None, None))
    if su:
        cards.append(card(
            "Full-site 3D prints — all 5 favelas",
            "Premium framed artifacts (5 cm box): engraved nameplate, scale bar, "
            "north arrow, recessed water. Plan + true-aspect axonometric per site, "
            "then download the STL.",
            su, img=sthumb, meta="Rocinha · Maré · Rio das Pedras · Vidigal · Complexo",
            kind="ok", badge_label="Sites", new_tab=False, **(_img_attrs(sthumb) if sthumb else {})))
    tu, tthumb = pages.get("textures", (None, None))
    if tu:
        cards.append(card(
            "Performance-texture tiles — sunlight & airflow",
            "Version A: winter sun-hours (VDG-P07) and CFD pedestrian ventilation "
            "(RDP-P20) as ground-only relief; three treatments each, with print-risk "
            "notes and the FDM recommendation.",
            tu, img=tthumb, meta="sunlight + airflow · stipple / contour / hatch",
            kind="ok", badge_label="Textures", new_tab=False, **(_img_attrs(tthumb) if tthumb else {})))
    return section("Physical twins — 3D prints", cards, anchor="prints")


# ── Planetary-health section ──────────────────────────────────────────────────
# Synthesised from a Lancet Planetary Health-style expert panel: advocates for the
# heat, respiratory/infectious, healthy-housing and equity pathways, plus a
# methods-editor skeptic who fixed the A–D evidence grades and banned causal verbs
# and invented incidence numbers. Every claim here is a modelled environmental
# EXPOSURE surface, never a measured health outcome — the page leads with that
# disclaimer. Bound to real artifacts: cross_site_stats.json (4-state taxonomy),
# the sun-deficit surface, the cross-site risk map, the geometric ventilation index.

HEALTH_DISCLAIMER = (
    "These are modelled environmental <em>exposure</em> surfaces, not measured "
    "health outcomes. This pipeline holds no temperature, air-exchange, mortality, "
    "morbidity or clinical data. Every health endpoint below — heat morbidity, "
    "respiratory transmission, damp / vitamin-D / mental health — is a mechanism "
    "drawn from the published literature, not a finding of this study. What we "
    "contribute is <em>where the built fabric produces unequal, WHO-referenced "
    "exposure</em>; the health consequences are hypothesised pathways for "
    "prioritisation and would need epidemiological data to confirm."
)

HEALTH_GRADES = [
    ("A", "ok", "Modelled exposure, independently cross-checked"),
    ("B", "amber", "Modelled exposure, single model"),
    ("C", "amber", "Geometric proxy — weak / partial physical signal"),
    ("D", "warn", "Inferred from literature — no matching field in the repo"),
]

# (grade, pill-kind, grade-note, title, mechanism, evidence, caveat) — strongest first.
HEALTH_PATHWAYS = [
    ("A", "ok", "Modelled exposure, cross-checked",
     "Winter-sun deprivation → damp, mould, vitamin-D, mood",
     "Dwellings that never clear the WHO ≥2 h/day winter-sun floor cannot dry "
     "passively, so surfaces stay damp and mould and dust-mite antigen accumulate — "
     "the most direct link, tied in the literature to asthma and wheeze. Chronically "
     "shaded façades also cut the UVB reaching skin (cutaneous vitamin-D synthesis), "
     "and low winter daylight is associated with mood and sleep disturbance.",
     "The best-evidenced exposure in the project: the sun-deficit surface is "
     "physically modelled, sun_fail logistic AUC 0.90 with a coherent structure "
     "(SVF protective β −1.62; slope worsening β +0.40; north-facing protective "
     "β −0.67, correct for the southern hemisphere). Cell morphotype moves the share "
     "of street observers below the floor from 14% to 73%, transferring "
     "leave-one-site-out (morphotype 17% of variance vs site 2%). An independent "
     "Ladybug façade run agrees on ordering — 50–72% of façade area below the floor, "
     "Rocinha worst.",
     "A modelled sun-availability surface, not measured damp, serum vitamin-D or "
     "diagnosed mood disorder; the 2 h floor is a healthy-housing daylight "
     "reference, not a clinical threshold."),
    ("B", "amber", "Distributional claim modelled; link to persons inferred (D)",
     "Unequal distribution of exposure — who bears it",
     "Environmental deprivation is not shared evenly. Within one favela, street-sun "
     "access is steeply unequal, so a mean hours-per-observer figure hides the "
     "deprived tail that carries the burden. Across favelas, because failure tracks "
     "morphotype not place, the same worst-exposed fabric recurs everywhere — which "
     "turns a blind cross-site map into a screening tool that finds the most "
     "deprived fabric wherever it sits, an equity lever as heat extremes intensify.",
     "Street-sun inequality reaches Gini 0.70 (Rocinha, most unequal) — measured, "
     "not assumed. Risk is a property of the fabric type (morphotype 17% of failure "
     "variance vs site 2%), backing a blind 8-favela risk map.",
     "We measure that exposure is unequal within the built fabric; with no "
     "observer-level socio-economic microdata linked, we cannot claim the poorest "
     "individuals occupy the worst cells — that step is external inference."),
    ("B–C", "amber", "Sunlight leg modelled (B); ventilation leg geometric proxy (C)",
     "Airborne-pathogen conditions — respiratory, tuberculosis",
     "Two distinct routes converge on the same compact fabric. Direct sun is "
     "germicidal (M. tuberculosis is sunlight-sensitive) and dries the interiors "
     "where respiratory pathogens persist — the sunlight route inherits the Grade-A "
     "solar surface. Separately, low outdoor ventilation weakens street-level "
     "dilution of exhaled aerosols; by Wells-Riley logic transmission risk rises as "
     "air exchange falls.",
     "The sunlight route is as strong as the solar pathway above. The ventilation "
     "route is weak: the vent_fail logistic is only AUC 0.69 with density (λp, "
     "β +0.32) the driver and SVF non-significant; a changepoint model puts a "
     "ventilation collapse near SVF ≈ 0.12. Note the repo's per-patch CFD returns "
     "are synthetic placeholders (real OpenFOAM results pending), so the ventilation "
     "route rests on geometry plus one illustrative field, not measured airflow.",
     "Outdoor canyon ventilation is not indoor air-change rate — pedestrian "
     "|U|/U_ref describes the street, not the ACH inside dwellings where "
     "transmission happens; treat it as a fabric-level susceptibility flag."),
    ("D", "warn", "Inferred — no thermal field in the repo",
     "Heat retention — the SVF double-edge",
     "SVF cuts both ways for heat. Deep, low-SVF canyons cut the daytime shortwave "
     "dose (protective shade in a tropical heatwave — the same geometry that starves "
     "streets of winter sun), but at night a low sky view traps outgoing longwave "
     "against warm masonry and metal and suppresses the ventilation that would flush "
     "accumulated heat. Because heatwave mortality is driven disproportionately by "
     "warm nights, the trapping-and-stagnation face likely dominates.",
     "Geometry only — SVF, λp, σH and heights locate where nocturnal trapping and "
     "low ventilation co-occur; the share of cells below SVF ≈ 0.12 flags "
     "stagnation-prone fabric. The repo holds no air temperature, UTCI or PET "
     "field, so the entire thermal link is imported from external physics.",
     "We measure form, not heat; every thermal claim is a geometric inference, and "
     "low SVF is genuinely double-edged, not unidirectionally harmful."),
]

# Honest anchor surfaces (real, non-synthetic): (rel-path-under-exports, title, desc).
HEALTH_SURFACES = [
    ("fig04_diagnostic_taxonomy.png",
     "Compound environmental constraint — 4-state taxonomy",
     "Every built cell classified adequate / sun-constrained / ventilation-"
     "constrained / both. The compound (both) share reaches 72% of built cells in "
     "Rocinha. This is the section's anchor exposure surface."),
    ("fig_solar_deficit.png",
     "Winter sun-deficit surface (WHO 2 h/day floor)",
     "Physically-modelled winter direct-sun hours per street observer against the "
     "≥2 h healthy-housing floor — the Grade-A exposure map."),
    ("cross_site_riskmap.png",
     "Cross-site risk map — equity screening tool (8 favelas)",
     "One continuous fabric-vector model flags the worst-exposed morphotypes across "
     "campaign + calibration favelas; reframed here as an equity-prioritisation "
     "instrument, not a descriptive overview."),
    ("ventilation_index.png",
     "Geometric ventilation constraint (0–3, pre-CFD)",
     "Strictly geometric ventilation tendency (skimming / depth / wind-alignment "
     "flags) — independent of the synthetic CFD returns. A susceptibility flag, not "
     "an air-exchange measurement."),
]


def _health_pathways_html():
    blocks = []
    for grade, kind, note, title, mech, ev, cav in HEALTH_PATHWAYS:
        g = grade.replace("–", "").replace("-", "")[:1]  # left-border colour by lead grade
        blocks.append(
            f'<div class="hx-path hx-{g}">'
            f'{badge(kind, "Grade " + grade)} '
            f'<span class="lab">{html.escape(note)}</span>'
            f'<h3>{html.escape(title)}</h3>'
            f'<p><b>Mechanism.</b> {html.escape(mech)}</p>'
            f'<p><b>What our data show.</b> {html.escape(ev)}</p>'
            f'<p class="cav"><b>Caveat.</b> {html.escape(cav)}</p>'
            f'</div>')
    rubric = '<div class="hx-rubric">' + "".join(
        f'<span class="r">{badge(kind, "Grade " + g)} {html.escape(defn)}</span>'
        for g, kind, defn in HEALTH_GRADES) + '</div>'
    return rubric + "".join(blocks)


def _health_table_html():
    """Per-site compound-deprivation table from the real 4-state taxonomy shares."""
    js = ROOT / "outputs/paper_figures/cross_site_stats.json"
    if not js.exists():
        return ""
    per = json.loads(js.read_text()).get("per_site", [])
    rows = []
    for v in sorted(per, key=lambda r: -r.get("shares", {}).get("compound_constraint", 0)):
        s = v.get("shares", {})
        rows.append((
            SITE_NAMES.get(v.get("site"), v.get("site", "?")),
            v.get("typology", ""),
            s.get("adequate", 0), s.get("sunlight_constraint", 0),
            s.get("ventilation_constraint", 0), s.get("compound_constraint", 0),
            v.get("pct_sun_below_2h", 0)))
    if not rows:
        return ""
    body = "".join(
        f'<tr><td>{html.escape(name)}</td><td>{html.escape(typ)}</td>'
        f'<td>{adq*100:.0f}%</td><td>{sun*100:.0f}%</td><td>{vent*100:.0f}%</td>'
        f'<td class="hot">{comp*100:.0f}%</td><td>{s2*100:.0f}%</td></tr>'
        for name, typ, adq, sun, vent, comp, s2 in rows)
    return (
        '<table class="hx-tbl"><thead><tr>'
        '<th>Favela</th><th>Type</th><th>Adequate</th><th>Sun-only</th>'
        '<th>Vent-only</th><th>Both (compound)</th><th>Sun&lt;2 h</th>'
        '</tr></thead><tbody>' + body + '</tbody></table>'
        '<p class="sub">Share of built 10 m cells in each exposure state (4-state '
        'diagnostic taxonomy; ≥2 h winter-sun floor, λf&gt;0.35 ventilation flag). '
        '“Both” = sun- <em>and</em> ventilation-constrained — the compound-'
        'deprivation fabric an intervention would target first.</p>')


HEALTH_STYLE = (
    '<style>'
    '.hx-rubric{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 16px}'
    '.hx-rubric .r{font-size:12.5px;color:var(--mut);border:1px solid var(--line);'
    'border-radius:8px;padding:6px 10px;background:var(--card)}'
    '.hx-path{background:var(--card);border:1px solid var(--line);'
    'border-left:5px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}'
    '.hx-path h3{margin:6px 0;font-size:16px}.hx-path p{margin:6px 0;font-size:14px;line-height:1.6}'
    '.hx-path .lab{color:var(--mut);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.03em}'
    '.hx-path .cav{color:var(--mut)}'
    '.hx-A{border-left-color:var(--ok)}.hx-B{border-left-color:var(--amber)}'
    '.hx-C{border-left-color:var(--amber)}.hx-D{border-left-color:var(--warn)}'
    '.hx-tbl{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0 4px}'
    '.hx-tbl th,.hx-tbl td{border:1px solid var(--line);padding:6px 9px;text-align:right}'
    '.hx-tbl th:first-child,.hx-tbl td:first-child,.hx-tbl th:nth-child(2),'
    '.hx-tbl td:nth-child(2){text-align:left}'
    '.hx-tbl th{background:#eef1f4}.hx-tbl .hot{font-weight:700;color:var(--warn)}'
    '</style>')


def write_health_page(prov):
    """Standalone planetary-health page: panel-synthesised form → exposure → literature
    health pathways, evidence-graded and bound to the real exposure surfaces. Returns
    (url, thumb) so the hub can link it with a preview."""
    exports = "/outputs/paper_figures/exports"
    surf_cards = [
        card(title, desc, f"{exports}/{fn}", img=f"{exports}/{fn}",
             meta="modelled exposure surface", kind="ok", badge_label="Exposure",
             new_tab=False, **_img_attrs(f"{exports}/{fn}"))
        for fn, title, desc in HEALTH_SURFACES
        if (ROOT / exports.lstrip("/") / fn).exists()]
    site_maps = []
    for s in CAMPAIGN:
        fn = f"/outputs/{s}/paper_figures/fig_{s}_diagnostic_map.png"
        if (ROOT / fn.lstrip("/")).exists():
            site_maps.append(card(
                f"{SITE_NAMES.get(s, s)} — diagnostic map",
                "Per-cell exposure state across the settlement (adequate / sun / "
                "ventilation / compound).", fn, img=fn,
                meta="4-state taxonomy", kind="info", badge_label="Per-site",
                new_tab=False, **_img_attrs(fn)))

    # --- Outcome probe (Grade C): the exposure vs a REAL disease, with every hedge ---
    probe = ""
    pf = "/outputs/comparative/health/tb_sun_deficit_screen.png"
    if (ROOT / pf.lstrip("/")).exists():
        pc = card(
            "TB incidence vs winter sun-deficit — 5 favelas, 2015–23",
            "Exploratory ecological probe: real tuberculosis incidence (SMS-Rio SINAN, "
            "9-year mean) and our modelled winter sun-deficit rank in the same direction "
            "(Spearman ρ ≈ +0.80, n=5). Direction-only.",
            pf, img=pf, meta="ecological probe · Grade C · not significant",
            kind="amber", badge_label="Outcome probe · Grade C", new_tab=False,
            **_img_attrs(pf))
        hedges = (
            '<div class="callout" style="border-left-color:var(--amber)">'
            '<p class="lead">How to read this probe — Grade C, strictly below the Grade-A surface</p>'
            '<ul>'
            '<li>Real TB and modelled sun-deficit <b>rank in the same direction</b> across 5 '
            'favelas (Spearman ρ ≈ +0.80, n=5, exact two-tailed p ≈ 0.13 — <b>not statistically '
            'significant</b>). The bootstrap interval supports only the sign, not a magnitude.</li>'
            '<li><b>It does not survive the first out-of-sample test.</b> Onboarding a sixth favela '
            '(Cidade de Deus — high TB, low sun-deficit) drops the rank correlation from ρ ≈ +0.80 '
            'to <b>ρ ≈ +0.26 (n=6, p ≈ 0.66)</b>. A single new point breaks the gradient, which is '
            'consistent with the n=5 result being a small-sample artefact. Treat this as a '
            'hypothesis to test at larger n (≈11 needed for power), not a finding.</li>'
            '<li>It <b>cannot be separated from a generic deprivation gradient</b>: sun-deficit is '
            'collinear with density, poverty and crowding, all established TB drivers. At n=5 no '
            'adjustment is possible, so this is not evidence of a sun-specific mechanism.</li>'
            '<li>It <b>reverses sign at the coarser AP scale</b> (ρ ≈ −0.50) — a change-of-support '
            '(MAUP) effect, shown here rather than hidden.</li>'
            '<li>An <b>independent audit caught and corrected a population error</b> (Jacarezinho) '
            'before this shipped.</li>'
            '<li><b>Specificity check (powered):</b> sun-deficit tracks TB (ρ ≈ +0.80) but not '
            'dengue — a mosquito-borne disease with the same poverty gradient (ρ ≈ +0.10, 4,380 '
            'cases). This weakens a generic-deprivation explanation.</li>'
            '<li><b>Crowding check:</b> adjusting for household occupancy (partial ρ ≈ +0.76) or '
            'areal density (ρ ≈ +0.69) only modestly attenuates the ranking. But density is the '
            'fairer confound (sun-deficit is a built-form proxy collinear with it, so adjustment '
            'cannot separate it from denser morphology), and n=5 (~2 residual df) cannot establish '
            'independence. The crowding covariate and TB denominator also sit on different '
            'geographic supports. Suggestive that crowding alone does not explain it — not more.</li>'
            '<li>Ecological (bairro ≈ favela): it says nothing about whether sun-deprived '
            '<em>individuals</em> develop TB — that needs an individual-level follow-up.</li>'
            '</ul></div>')
        vitd = (
            '<div class="callout"><p class="lead">Why sun-deficit could matter for TB — '
            'the vitamin-D mechanism</p><p>Vitamin D is not measurable at favela scale, so it is '
            'the plausible <em>why</em>, narrated from real Rio priors: winter sun-deficit lowers '
            'the UVB dose and hence cutaneous 25(OH)D synthesis. In Rio’s Pró-Saúde cohort serum '
            '25(OH)D rose about +0.49 nmol/L per unit of sun-exposure and ran roughly +20 nmol/L '
            'higher in summer than winter; vitamin-D deficiency is a replicated tuberculosis '
            'susceptibility factor (pooled odds ratio 3.23). Vitamin D is the mechanism; TB is the '
            'measurable endpoint the probe above tests.</p></div>')
        probe = ('<section><h2 id="health-probe">Outcome probe — does the exposure track a '
                 'real disease?</h2><div class="grid">' + pc + '</div>' + hedges + vitd + '</section>')

    disclaimer = (
        '<div class="callout" style="border-left-color:var(--warn)">'
        f'<p class="lead">Read first — what this is (and is not)</p>'
        f'<p>{HEALTH_DISCLAIMER}</p></div>')
    panel_note = (
        '<div class="callout"><p class="lead">How this section was built — a '
        'Planetary Health panel</p><p>Four advocates argued distinct pathways '
        '(urban-heat, respiratory/infectious, healthy-housing, health-equity) '
        'against a methods-editor skeptic who fixed the A–D evidence grades, banned '
        'causal verbs and any invented incidence numbers, and required the '
        'disclaimer above. Every pathway carries its grade; the solar pathway '
        '(Grade A) is the only one strong enough to anchor the section — the rest '
        'are prioritisation hypotheses, ranked by how much of the chain we actually '
        'measured.</p></div>')

    body = (
        HEALTH_STYLE
        + disclaimer
        + '<section><h2 id="health-pathways">Four exposure → health pathways '
          '(strongest first)</h2>' + _health_pathways_html() + '</section>'
        + section("Exposure surfaces (modelled, non-synthetic)", surf_cards,
                  anchor="health-surfaces")
        + '<section><h2 id="health-table">Compound deprivation by favela</h2>'
        + _health_table_html() + '</section>'
        + probe
        + section("Per-site diagnostic maps", site_maps, anchor="health-sites")
        + panel_note)

    crumb = breadcrumb([("← Project hub", "index.html"), ("Planetary health", None)])
    sub = (f'{badge("ok", "solar pathway Grade A")} '
           f'{badge("amber", "exposure surfaces, not health outcomes")}')
    (OUT / "health.html").write_text(_relativize(page(
        "Planetary health — environmental exposure pathways", sub, body,
        crumb=crumb, provenance=prov)))
    thumb = f"{exports}/fig04_diagnostic_taxonomy.png"
    return ("/outputs/_hub/health.html",
            thumb if (ROOT / thumb.lstrip("/")).exists() else None)


def health_section(prov):
    """Nav card → the standalone planetary-health page."""
    url, thumb = write_health_page(prov)
    c = card(
        "Planetary health — exposure pathways",
        "A Lancet-style panel maps built form to WHO-referenced environmental "
        "exposure (winter-sun deprivation, ventilation, heat, equity), each "
        "evidence-graded A–D. Modelled exposure surfaces, not measured health "
        "outcomes — the solar pathway (AUC 0.90, Grade A) anchors it.",
        url, img=thumb, meta="4 pathways · compound deprivation to 72% · Gini 0.70",
        kind="ok", badge_label="Health", new_tab=False,
        **(_img_attrs(thumb) if thumb else {}))
    return section("Planetary health — exposure pathways", [c], anchor="health")


def _latest_wp07_figures_dir() -> Path | None:
    hits = sorted(ROOT.glob("runs/wp07_figures_*"))
    return hits[-1] if hits else None


# Inline pan/zoom viewer JS + CSS (no CDN — the mirror must be self-contained,
# docs/wp07_zoom_spec.md item 4). Deliberately never named `zoom`/`lb*`: every
# hub page ends with hubkit's own lightbox (function `zoom`, ids lb/lbx/lbi/
# lbcap) via page()'s _LB — a name collision would silently break one or the
# other. `.card`'s onclick (hubkit's zoom()) is unused on this page's own
# tiles; they call pzOpen directly.
_PZ_CSS = """
.pzgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px;margin:18px 0}
.pztile{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;
cursor:zoom-in;text-align:left;padding:0;font:inherit;color:inherit;display:block;width:100%}
.pztile img{width:100%;height:160px;object-fit:cover;display:block;border-bottom:1px solid #eee}
.pztile .pzcap{padding:10px 12px}
.pztile .pzcap h3{margin:0 0 3px;font-size:14px}
.pztile .pzcap p{margin:0;color:var(--mut);font-size:12px}
.pztile.missing{cursor:default;opacity:.7}
.pztile.missing .pzph{height:160px;display:flex;align-items:center;justify-content:center;
background:#f3f5f7;color:var(--mut);font-size:12px;text-align:center;padding:10px}
#pzov{display:none;position:fixed;inset:0;background:rgba(10,12,14,.94);z-index:200;overflow:hidden;touch-action:none}
#pzov:not([hidden]){display:block}
#pzstage{width:100%;height:100%;display:flex;align-items:center;justify-content:center;
cursor:grab;transform-origin:center center}
#pzstage img{max-width:92vw;max-height:80vh;width:auto;height:auto;user-select:none;-webkit-user-drag:none}
#pzclose{position:absolute;top:12px;right:18px;background:none;border:0;color:#fff;
font-size:34px;line-height:1;cursor:pointer;padding:2px 12px;z-index:2}
#pzcap{position:absolute;bottom:12px;left:0;right:0;text-align:center;color:#eee;font-size:13px}
#pzhint{position:absolute;top:14px;left:18px;color:#9fb0c0;font-size:12px}
"""

_PZ_JS = """
let pzScale=1,pzX=0,pzY=0,pzDragging=false,pzLastX=0,pzLastY=0;
function pzApply(){document.getElementById('pzstage').style.transform=
'translate('+pzX+'px,'+pzY+'px) scale('+pzScale+')';}
function pzOpen(src,label){
var img=document.getElementById('pzimg');
img.src=src;img.alt=label||'';
document.getElementById('pzcap').textContent=label||'';
pzScale=1;pzX=0;pzY=0;pzApply();
document.getElementById('pzov').hidden=false;
}
function pzClose(){document.getElementById('pzov').hidden=true;}
(function(){
var ov=document.getElementById('pzov');
if(!ov)return;
var stage=document.getElementById('pzstage');
ov.addEventListener('click',function(e){if(e.target===ov)pzClose();});
document.getElementById('pzclose').addEventListener('click',pzClose);
addEventListener('keydown',function(e){if(!ov.hidden&&e.key==='Escape')pzClose();});
stage.addEventListener('wheel',function(e){
e.preventDefault();
var delta=e.deltaY<0?1.15:(1/1.15);
pzScale=Math.min(16,Math.max(0.5,pzScale*delta));
pzApply();
},{passive:false});
stage.addEventListener('pointerdown',function(e){
pzDragging=true;pzLastX=e.clientX;pzLastY=e.clientY;
stage.setPointerCapture(e.pointerId);
});
stage.addEventListener('pointermove',function(e){
if(!pzDragging)return;
pzX+=e.clientX-pzLastX;pzY+=e.clientY-pzLastY;
pzLastX=e.clientX;pzLastY=e.clientY;pzApply();
});
stage.addEventListener('pointerup',function(){pzDragging=false;});
stage.addEventListener('pointercancel',function(){pzDragging=false;});
stage.addEventListener('dblclick',function(){pzScale=1;pzX=0;pzY=0;pzApply();});
})();
"""


def _pz_js_attr(s: str) -> str:
    """Escape a string for a JS single-quoted literal inside pzOpen('...','...')
    (same rule as hubkit._js_attr, not imported since it's a private name)."""
    return html.escape(s.replace("\\", "\\\\").replace("'", "\\'"), quote=True)


def _latest_wp07_zoom_dir() -> Path | None:
    hits = sorted(d for d in (ROOT / "runs").glob("wp07_zoom_*") if d.is_dir() and list(d.glob("*.png")))
    return hits[-1] if hits else None


def write_zoom_viewer_page(prov):
    """The WP-07Z pan/zoom viewer: docs/wp07_zoom_spec.md item 4. Discovers
    the latest runs/wp07_zoom_<UTC>/figure_manifest.json, mirrors its PNGs
    into outputs/_hub/wp07_staged/zoom/ (same L1-guard reason as the map
    family: no `runs/` segment may appear under `_hub/`), and lists the
    citywide pair, every resolved window (opens in the inline pan/zoom
    view), and every missing_boundary window as a disabled card carrying its
    stated reason — cards throughout, never a prose-only link, so the hub's
    reachability gate sees them. Returns (url, thumb)."""
    run_dir = _latest_wp07_zoom_dir()
    if run_dir is None:
        return None, None
    manifest_path = run_dir / "figure_manifest.json"
    if not manifest_path.exists():
        return None, None
    figures = json.loads(manifest_path.read_text()).get("figures", {})

    zoom_out = OUT / "wp07_staged" / "zoom"
    zoom_out.mkdir(parents=True, exist_ok=True)

    tiles, thumb = [], None
    # Citywide pair first, then one tile per window per metric (produced or
    # missing_boundary), in a stable order (sorted fig_id — never ranked).
    for fig_id in sorted(figures):
        fig = figures[fig_id]
        title = fig_id.replace("f6_", "").replace("_", " ")
        if fig["status"] != "produced":
            reason = fig.get("reason", "not produced")
            tiles.append(
                f'<div class="pztile missing"><div class="pzph">{html.escape(reason)}</div>'
                f'<div class="pzcap"><h3>{html.escape(title)}</h3>'
                f'<p>{badge("amber", "missing_boundary")}</p></div></div>'
            )
            continue
        png = fig.get("png_path")
        if not png:
            continue
        dst = zoom_out / Path(png).name
        if not dst.exists() or dst.stat().st_mtime < (run_dir / png).stat().st_mtime:
            shutil.copy2(run_dir / png, dst)
        img_url = f"/outputs/_hub/wp07_staged/zoom/{dst.name}"
        thumb = thumb or img_url
        attrs = _img_attrs(img_url)
        thumb_src = attrs.get("thumb", img_url)
        dims = f'{fig.get("png_width_px", "?")}×{fig.get("png_height_px", "?")} px'
        window = fig.get("window")
        meta = f'{dims} · {window["label"]}' if window else f'{dims} · citywide · {fig["aggregation"]["pixel_m"]:g} m/px'
        cap = _pz_js_attr(title)
        tiles.append(
            f'<button type="button" class="pztile" '
            f'onclick="pzOpen(\'{img_url}\',\'{cap}\')">'
            f'<img src="{thumb_src}" alt="{html.escape(title)}" loading="lazy">'
            f'<div class="pzcap"><h3>{html.escape(title)}</h3>'
            f'<p>{html.escape(meta)}</p></div></button>'
        )

    if not tiles:
        return None, None

    crumb = breadcrumb([("← Project hub", "../../index.html"), ("WP-07 staged figures", "../index.html"),
                        ("Zoom viewer", None)])
    body = (
        f'<style>{_PZ_CSS}</style>'
        '<p class="lead">High-resolution citywide SVF/irradiation pair (docs/wp07_zoom_spec.md) plus '
        'one native-resolution zoom extract per resolvable window from '
        '<code>config/zoom_windows.yaml</code> — click any tile to pan/zoom (drag, scroll/pinch, '
        'double-click to reset). Withheld under red line L1; nothing here is promoted.</p>'
        + section("Zoom windows", tiles, anchor="zoom-windows")
        + '<div id="pzov" hidden><button id="pzclose" aria-label="Close (Esc)">&times;</button>'
        '<div id="pzstage"><img id="pzimg" alt=""></div><p id="pzcap"></p>'
        '<p id="pzhint">drag to pan · scroll/pinch to zoom · double-click to reset · Esc to close</p></div>'
        f'<script>{_PZ_JS}</script>'
    )
    out = zoom_out / "index.html"
    out.write_text(_relativize(page(
        "WP-07Z zoom viewer", badge("terra", f"{len(tiles)} windows · withheld · L1"),
        body, crumb=crumb, provenance=prov), out.parent))
    return "/outputs/_hub/wp07_staged/zoom/index.html", thumb


_WP07_STAGED_REDIRECT_TARGET = "https://brisa.theoalessandro.com/figures#s-awaiting"


def write_staged_figures_page(prov):
    """Mirrors the staged WP-07 figure PNGs into outputs/_hub/wp07_staged/ (so
    every image_url the brisaverse register cites — /morphofavela-dash/outputs/
    _hub/wp07_staged/<name>.png — still resolves) but no longer builds a card
    listing there. Navigation council ruling, 2026-09-24, Phase 4 ("Die":
    _hub/wp07_staged/ is replaced by a stub): this used to be a third,
    unlinked staged-count view (diagnosis #2 — 4 on /figures, 6 here, 9 in the
    register). The PI's one staged count now lives at /figures#s-awaiting,
    joined from the same register this mirror only serves bytes for.
    Returns (url, thumb) — url is the stub page, thumb the first PNG copied,
    for the deliverables card."""
    import shutil
    thumb = None
    run_dir = _latest_wp07_figures_dir()
    if run_dir is not None:
        manifest_path = run_dir / "figure_manifest.json"
        figures = json.loads(manifest_path.read_text()).get("figures", {}) if manifest_path.exists() else {}
        for fig_id in sorted(figures):
            fig = figures[fig_id]
            png = fig.get("png_path")
            # Serve the copy staged into the mirror, never the run directory:
            # the hub's L1 guard forbids any "runs" path segment under _hub/.
            staged_copy = OUT / "wp07_staged" / Path(png).name if png else None
            if not png or not (staged_copy.exists() or (run_dir / png).exists()):
                continue
            if not staged_copy.exists():
                staged_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run_dir / png, staged_copy)
            img_url = f"/outputs/_hub/wp07_staged/{Path(png).name}"
            thumb = thumb or img_url
    # The WP-07M citywide maps are a second family, mirrored for the same
    # register-image_url reason (producer-declared withheld, red line L1).
    map_dirs = sorted(d for d in (ROOT / "runs").glob("wp07_map_*") if d.is_dir() and list(d.glob("*.png")))
    if map_dirs:
        map_dir = map_dirs[-1]
        mp = map_dir / "figure_manifest.json"
        map_figs = json.loads(mp.read_text()).get("figures", {}) if mp.exists() else {}
        for fig_id in sorted(map_figs):
            fig = map_figs[fig_id]; png = fig.get("png_path")
            if not png or not (map_dir / png).exists():
                continue
            dst = OUT / "wp07_staged" / Path(png).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.stat().st_mtime < (map_dir / png).stat().st_mtime:
                shutil.copy2(map_dir / png, dst)
            img_url = f"/outputs/_hub/wp07_staged/{dst.name}"
            thumb = thumb or img_url
    if thumb is None:
        return None, None

    out = OUT / "wp07_staged" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={_WP07_STAGED_REDIRECT_TARGET}">'
        '<title>Moved — WP-07 staged figures</title></head><body>'
        '<p>This page moved. The one staged count now lives in the review '
        f'folder’s "Awaiting your call" section: '
        f'<a href="{_WP07_STAGED_REDIRECT_TARGET}">{_WP07_STAGED_REDIRECT_TARGET}</a></p>'
        '</body></html>'
    )
    return "/outputs/_hub/wp07_staged/index.html", thumb


def figure_review_section(prov):
    """The PI's own figure-review folder, exploded into one card per section.

    This sits FIRST on the page deliberately. A round-B critic scored this index
    6.0 for wayfinding: the fold was spent on a chronological changelog and the
    word the PI actually searched for — "citywide" — appeared nowhere above it.
    Figures are what this project produces; they belong at the top.
    """
    roots = sorted(d for d in (ROOT / "outputs" / "_review").glob("*")
                   if (d / "MANIFEST.json").is_file())
    if not roots:
        return ""
    run = roots[-1]
    manifest = json.loads((run / "MANIFEST.json").read_text())
    base = f"/outputs/_review/{run.name}"

    counts, thumbs = {}, {}
    for e in manifest.get("files", []):
        if e.get("status") != "ok":
            continue
        counts[e["section"]] = counts.get(e["section"], 0) + 1
        if e["section"] not in thumbs and e.get("thumb"):
            thumbs[e["section"]] = f'{base}/{e["section"]}/{e["thumb"]}'

    cards = [card(
        f"Every figure, one page — {run.name}",
        "The whole review folder: curated sections first, then every other figure "
        "on disk grouped by where it came from. Each card states its release class.",
        f"{base}/index.html", kind="ok", badge_label="Start here",
        meta="generated by scripts/build_pi_review_folder.py", new_tab=False)]

    for sec in manifest.get("sections", []):
        if sec.get("group") == "other":
            continue
        n = counts.get(sec["slug"], 0)
        if not n:
            continue
        anchor_id = "s-" + sec["slug"].replace("/", "-").replace("__", "-")
        classes = [e.get("release_class") for e in manifest["files"]
                   if e["section"] == sec["slug"]]
        n_withheld = sum(1 for c in classes if c == "withheld")
        withheld = n_withheld > 0
        # "Withheld" on a card whose bar chart is publishable would be a lie in the
        # safe direction, but still a lie — say which it is.
        badge_txt = ("Withheld · L1" if n_withheld == len(classes)
                     else f"{n_withheld} of {len(classes)} withheld · L1") if withheld else None
        cards.append(card(
            sec["title"],
            f"{n} figure{'s' if n != 1 else ''}. " + _strip_tags(sec.get("blurb", "")),
            f"{base}/index.html#{anchor_id}",
            img=thumbs.get(sec["slug"]),
            kind="terra" if withheld else "info",
            badge_label=badge_txt,
            meta=sec.get("provenance", ""), new_tab=False))
    return section("Figure review", cards, anchor="figure-review")


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def deliverables_section(prov):
    cards = []
    brief = ROOT / "outputs/_hub/mare_review/mare_morphology_brief_v2.pdf"
    if brief.exists():
        cards.append(card(
            "Maré morphology brief",
            "What has already been measured in Maré — built form, sky access, sun, "
            "geometry-derived ventilation tendencies, the data inventory and the "
            "sharing terms — written for teams scoping new research on Maré. "
            "Draft for PI review, alongside its slide deck and the refreshed "
            "Folha de Rua sheets.",
            "mare_review/index.html", kind="info", badge_label="Site report",
            meta="6 pages · every number read by code from the outputs of record",
            new_tab=False))
    tr_md = ROOT / "docs/technical_report/technical_report.md"
    tr_pdf = ROOT / "docs/technical_report/technical_report.pdf"
    if tr_md.exists():
        back = breadcrumb([("← Project hub", "../index.html"), ("Technical report", None)])
        render_doc_page(tr_md, DOCS / "technical_report.html", crumb=back,
                        provenance=prov, base=_doc_base(tr_md), root=ROOT,
                        mirror_dir=DOCS)
        cards.append(card("Technical report", "Full report — fast HTML view, figures inline.",
                          "docs/technical_report.html", kind="ok",
                          badge_label="Report", new_tab=False))
    if tr_pdf.exists():
        mb = tr_pdf.stat().st_size // 1_000_000
        cards.append(card("Technical report — PDF",
                          "Canonical typeset PDF; large file, opens in a new tab.",
                          "/docs/technical_report/technical_report.pdf", kind="info",
                          badge_label=f"PDF · {mb} MB"))
    review_index = OUT / "wp07_staged" / "review" / "index.html"
    if review_index.exists():
        cards.append(card(
            "WP-07 promotion review pack",
            "The numbers behind every staged figure (the ledger), the "
            "methodology, and the validation/sensitivity results that bound "
            "the claims — nine pages, read in order. Nothing here is "
            "promoted; the decision itself is wp07_figure_promotion on /ops.",
            "wp07_staged/review/index.html", kind="info", badge_label="Review pack",
            meta="generated by scripts/build_review_pack.py", new_tab=False))
    sf_url, sf_thumb = write_staged_figures_page(prov)
    if sf_url:
        cards.append(card(
            "WP-07 staged figures",
            "The staged listing moved: the one staged count now lives in the "
            "review folder's \"Awaiting your call\" section — this card just "
            "takes you there.",
            sf_url, img=sf_thumb, kind="amber", badge_label="→ /figures",
            meta="mirrors the PNGs runs/wp07_figures_<UTC>/ and runs/wp07_map_<UTC>/ cite",
            new_tab=False, **(_img_attrs(sf_thumb) if sf_thumb else {})))
    zv_url, zv_thumb = write_zoom_viewer_page(prov)
    if zv_url:
        cards.append(card(
            "WP-07Z zoom viewer",
            "High-resolution citywide SVF/irradiation pair plus a native-resolution "
            "pan/zoom extract per study favela (Ipanema pending a bairro boundary layer) — "
            "producer-declared withheld under red line L1.",
            zv_url, img=zv_thumb, kind="terra", badge_label="Withheld · L1",
            meta="generated from the latest runs/wp07_zoom_<UTC>/figure_manifest.json",
            new_tab=False, **(_img_attrs(zv_thumb) if zv_thumb else {})))
    return section("Deliverables", cards, anchor="deliverables")


def _relativize(html_str, page_dir=None):
    """Rewrite root-absolute URLs (and lightbox zoom() targets) to paths relative
    to `page_dir` (default OUT, i.e. outputs/_hub), so every page resolves under
    any URL prefix and when opened directly via file://. A target outside
    outputs/ (e.g. under docs/) is mirrored into OUT/docs first — see
    hubkit.relativize_page."""
    return relativize_page(html_str, OUT if page_dir is None else page_dir,
                           ROOT, mirror_dir=DOCS)


def _doc_base(src: Path) -> str:
    """Root-absolute base directory for a markdown doc's relative links/images,
    e.g. docs/technical_report/technical_report.md -> '/docs/technical_report/'.
    Consumed by hubkit._rel inside md_to_html; the resulting root-absolute URLs
    are then resolved (and mirrored if needed) by _relativize."""
    rel = src.parent.relative_to(ROOT)
    return "/" if str(rel) == "." else f"/{rel}/"


def _is_withheld(rel: str) -> bool:
    """True if `rel` (a path under outputs/, no leading 'outputs/') is a
    per-cell layer the hub must never expose: <site>/morphometrics/grid,
    <site>/svf_v2/*.gpkg, runs/, <site>/cfd*. Scoped to a known site's own
    top-level subtree so an unrelated file merely named e.g.
    'cfd_parameter_estimation_plan.html' never false-positives."""
    parts = rel.split("/")
    if "runs" in parts:
        return True
    if len(parts) >= 2 and parts[0] in SITE_NAMES:
        p1 = parts[1]
        if p1 == "morphometrics" and len(parts) >= 3 and parts[2] == "grid":
            return True
        if p1 == "svf_v2" and rel.endswith(".gpkg"):
            return True
        if p1.startswith("cfd"):
            return True
    return False


def build_mirror_manifest(out_dir: Path, root: Path) -> list[str]:
    """Scan every emitted *.html page under `out_dir` and return the sorted
    list of first-segment directories under outputs/ that any href/src/zoom()
    target resolves into — computed from the written HTML, never typed by
    hand. Asserts no withheld per-cell layer is referenced."""
    segments, refs = set(), set()
    for html_file in sorted(out_dir.rglob("*.html")):
        text = html_file.read_text()
        targets = re.findall(r'(?:href|src)="([^"]+)"', text)
        targets += re.findall(r"zoom\('([^']+)'", text)
        for t in targets:
            base, _, _frag = t.partition("#")
            if not base or base.startswith(("http:", "https:", "mailto:", "//")):
                continue  # "//" = protocol-relative external (e.g. pandoc's
                          # html5shiv CDN boilerplate in the review-pack pages)
            assert not base.startswith("/"), (
                f"root-absolute target leaked into {html_file}: {base}")
            abspath = (html_file.parent / base).resolve()
            try:
                rel = abspath.relative_to(root / "outputs")
            except ValueError:
                continue  # points outside outputs/ entirely — should not happen
            if not rel.parts:
                continue
            refs.add(rel.as_posix())
            segments.add(rel.parts[0])
    for ref in refs:
        assert not _is_withheld(ref), f"withheld per-cell layer referenced by hub: {ref}"
    return sorted(segments)


def _group_for(rel_parts: tuple) -> str:
    """Which map.html group a page belongs to: top level, or its first path
    segment under outputs/_hub/ (docs, wp07_staged, prints, ...)."""
    return rel_parts[0] if len(rel_parts) > 1 else "(top level)"


def write_map_page(prov):
    """Site map, generated from the exact same walk scripts/audit_hub_graph.py
    performs (docs/hub_wp_structure_spec.md Ph.2 item 3) — imported, not
    reimplemented, so the map can never drift from what the gate actually
    checked. Must be called after every other page (incl. index.html, with
    its nav link to this one) has been written, so the walk sees the real
    tree; must itself be written before build_mirror_manifest runs."""
    result = audit_hub_graph.audit(OUT)
    groups: dict = {}
    for p, d in result.depth.items():
        rel = p.relative_to(OUT)
        groups.setdefault(_group_for(rel.parts), []).append((d, rel))

    body_parts = [
        f'<p class="lead">{len(result.depth)} pages reachable from the index, '
        f'walked breadth-first over card/nav links — the same walk '
        f'<code>scripts/audit_hub_graph.py</code> gates on. Regenerated on '
        f'every hub build, so this can never go stale the way a hand-written '
        f'map would.</p>']
    for grp in sorted(groups):
        cards = [
            card(str(rel), f"depth {d} from the index", str(rel), kind="doc",
                new_tab=False)
            for d, rel in sorted(groups[grp])]
        body_parts.append(section(f"{grp} ({len(cards)} pages)", cards,
                                  anchor=f"map-{_slug_ascii(grp)}"))

    problems = []
    if result.prose_only:
        problems.append('<p><b>PROSE-ONLY</b> (linked, but never from a card or nav): '
                        + ", ".join(str(p.relative_to(OUT)) for p in result.prose_only)
                        + "</p>")
    if result.orphan:
        problems.append('<p><b>ORPHAN</b> (no inbound link at all): '
                        + ", ".join(str(p.relative_to(OUT)) for p in result.orphan)
                        + "</p>")
    if result.dangling:
        problems.append('<p><b>DANGLING LINKS</b>: '
                        + "; ".join(f"{src.relative_to(OUT)} → {raw}"
                                    for src, raw in result.dangling)
                        + "</p>")
    if problems:
        body_parts.insert(1, '<div class="callout" style="border-left-color:var(--warn)">'
                          '<p class="lead">This build is NOT fully reachable — '
                          '`python3 scripts/audit_hub_graph.py` fails on it:</p>'
                          + "".join(problems) + "</div>")

    crumb = breadcrumb([("← Project hub", "index.html"), ("Site map", None)])
    sub = badge("ok", f"{len(result.depth)} reachable") + (
        " " + badge("warn", f"{len(result.prose_only) + len(result.orphan) + len(result.dangling)} issues")
        if problems else "")
    (OUT / "map.html").write_text(_relativize(page(
        "Site map", sub, "".join(body_parts), crumb=crumb, provenance=prov)))


def _slug_ascii(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-") or "group"


def main(root: Path | None = None):
    global ROOT, OUT, DOCS, DASH
    if root is not None:
        ROOT = Path(root).resolve()
        OUT = ROOT / "outputs" / "_hub"
        DOCS = OUT / "docs"
        DASH = ROOT / "outputs" / "_distribution" / "html_dashboards"
    DOCS.mkdir(parents=True, exist_ok=True)
    prov = git_provenance(ROOT, "scripts/build_project_hub.py")

    # Single ordered source of truth: (anchor, sidebar_label, html). The anchor
    # matches the id the section builder emits; the label is a human heading, so the
    # sidebar never shows a machine slug. Project-owned contribution (headline, TR)
    # precedes the partner façade cross-check.
    sections = [
        ("figure-review", "Figure review", figure_review_section(prov)),
        ("territory", "Site territories", territory_section(prov)),
        ("latest", "Latest & work queue", build_callout(prov)),
        ("headline", "Headline result", headline_section(prov)),
        ("work-packages", "Work packages (runs of record)", work_packages_section(prov)),
        ("deliverables", "Deliverables", deliverables_section(prov)),
        ("ventilation", "Ventilation tendencies (§5.6)", ventilation_section(prov)),
        ("maup", "Grid sensitivity (MAUP)", maup_section(prov)),
        ("facade-solar", "Solar access — façade & street", facade_solar_section(prov)),
        ("health", "Planetary health (exposure pathways)", health_section(prov)),
        ("prints", "Physical twins (3D prints)", prints_section(prov)),
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
    # The map link sits in its own <nav> (not the #-anchor toc) so it is a
    # real navigable edge to a real page, not an in-page anchor — the same
    # distinction scripts/audit_hub_graph.py's gate enforces.
    sidebar = (toc_sections([(a, lbl) for a, lbl, _ in sections])
              + '<nav class="hubmap"><a href="/outputs/_hub/map.html">🗺 Site map</a></nav>')

    n_sites = sum((DASH / s / "index.html").exists() for s in SITE_NAMES)
    # a finding-bearing stat, not a raw glob count of a gitignored dir
    sub = (f'{badge("ok", f"{n_sites} campaign favelas")} '
           f'{badge("info", "morphotype → WHO-2h winter-sun failure 14 % → 73 %")}')
    (OUT / "index.html").write_text(_relativize(
        page("MorphoFavela — project hub", sub, body,
             provenance=prov, sidebar=sidebar)))

    # Generated last, from the same walk the gate performs, over the tree as
    # it now stands (incl. index.html's own new map nav link) — the map is a
    # by-product of the check, so it cannot drift from what exists.
    write_map_page(prov)

    manifest = build_mirror_manifest(OUT, ROOT)
    (OUT / "mirror_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"hub written: {len(sections)} sections, {n_sites} site dashboards, "
          f"{len(manifest)} mirrored subtrees")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None,
                        help="repo root to build the hub for (default: this "
                             "script's own repo)")
    args = parser.parse_args()
    main(root=args.root)
