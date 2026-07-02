# Dashboard improvement plan — council audit + parallelizable execution

*Source: 6-lens design + research-management council (workflow `dashboard-council-audit`,
2026-07-02). Reviewers: information architecture, visual/graphic design,
data-viz/scicomm, UX/interaction, front-end/accessibility, research-management
(the last dropped on a structured-output retry cap; synthesis ran on 5). This doc is
the single source of truth for the backlog — the hub itself is regenerated, never
hand-edited.*

## Executive summary

The MorphoFavela hub is a genuinely well-engineered zero-dependency artifact: a clean
`hubkit` engine/consumer split, real degrade-by-existence guards, deep-linked
information scent, and an editorial doc-reading surface. But it does not yet read as an
**external-reviewer-grade deliverable**, and every lens converged on one root cause:
*the presentation buries the science and leaks dev-scratch*. The headline result
(morphotype → WHO-2h winter-sun failure, **14 %→73 %**) has no section and no image on
the landing page while a collaborator's Ladybug run leads; the flagship page ships an
HTML validity error (unescaped `&`), sub-AA contrast, enum-echo badges reading
`OK/DOC/INFO`, no image `alt` text, a mouse-only lightbox, and internal
`keep/refine` triage widgets that surface to reviewers. Nearly every fix is expressible
as edits to `build_project_hub.py` / `hubkit.py` (plus the separate gallery and
dashboard builders), with no new dependencies.

**Overall grade: B−** — structurally sound and genuinely portable, but the headline
result is hidden and the surface carries validity/a11y/dev-scratch tells that undercut
credibility until the P0 wave lands.

## Strengths (keep these)

- Clean engine/consumer split: `hubkit.py` is a small pure-function API over one
  semantic `:root` token set → the whole plan is expressible as generator edits.
- Degrade-gracefully-by-existence is real: every card/section gated on `Path.exists()`
  → the hub never renders a dead card.
- Strong information scent + honest science framing: Latest deep-links to exact TR
  anchors/figures; captions state findings-with-numbers **and** limitations-as-results
  (roughness invalid in 53–75 % of cells; ventilation tendencies persistently τ-gated).
- Genuinely portable single artifact: no build step, inlined CSS/JS, works offline.

## Top weaknesses

1. **Headline contribution is hidden** — money figure has no section/image on the hub
   (bullet #6 of a 10-item list), opens ~24th in the gallery, while the Mingze/Ladybug
   run is the first content section. A reviewer can't find the contribution in <10 s.
2. **Dev-scratch + validity tells** — enum-echo badges (`badge(kind,kind)`); hero
   callout hand-inlined with hardcoded hex bypassing tokens; unescaped `&`
   (`Typology & Signature`) is a real parse error; `keep | refine` triage widgets leak
   onto the external gallery.
3. **A11y below reviewer grade** — zero `alt` on any figure `img`; mouse-only lightbox
   with no dialog semantics/focus management; multiple sub-AA colors (`#9aa`, `#0a5`, `#888`).
4. **Navigation inconsistent + one-way** — `target=_blank` hardcoded on every card
   (tab-explosion, same-vs-new-tab divergence); doc-page crumbs become no-ops in spawned
   tabs; per-site dashboards have no link back to the hub; absolute `/outputs`/`/docs`
   links 404 under `file://`.
5. **Zero test coverage** for the two core modules despite the repo's stage-the-test rule.

## Backlog (prioritized, de-duplicated)

`parallel_group`: items in the **same** group share a file and must run serially;
different groups run concurrently. `core` = `build_project_hub.py` + `hubkit.py`.

| id | P | eff | title | group | depends | acceptance (short) |
|----|---|-----|-------|-------|---------|--------------------|
| **H1** | P0 | M | Hero section for the money figure (typology→failure, image) | core | H4 | `typology_failure_lookup.png` in an `<img>` in the first `<section>`; TOC top entry |
| **H2** | P0 | S | Strip `keep/refine` triage widget from gallery | gallery | — | no `prompt`/`keep`/`refine` widget in `figures_v2/index.html` |
| **H3** | P0 | M | Fix Latest callout: escape text, route through tokens, AA contrast | core | — | no inline `style=` in callout; `&amp;` not raw `&`; 0 validator errors; ≥4.5:1 |
| **H4** | P0 | S | Real `alt` on every figure card + lightbox img | core | — | every `<img>` has non-empty escaped `alt`; validator: no missing-alt |
| **M1** | P1 | M | Human sidebar TOC labels + single-source section bookkeeping | core | — | no `Maup`/`Facade-solar`/`Figure` slugs; label matches `<h2>` |
| **M2** | P1 | S | Reorder: project-owned contribution before Mingze cross-check | core | H1,M1 | façade section after ≥1 project section; Deliverables in top 3 |
| **M3** | P1 | M | Badges carry status meaning, not the enum token | core | H4 | no `pill x">x`; every pill is an actionable status |
| **M4** | P1 | M | Explicit new-tab behavior; hub docs open same-tab; `rel=noopener` | core | M3 | no destination opens same-tab from one entry, new-tab from another |
| **M5** | P1 | M | Hub links resolve under `file://` (relative hrefs / `<base>`) | core | — | opening `index.html` via `file://` resolves all cards/figures |
| **M6** | P1 | M | Editorial identity (serif masthead + one accent), offline-safe | core | — | hub H1 face/accent matches dashboards; no external font request |
| **M7** | P1 | S | Fix sub-AA contrast on `.cap code` + footer | core | — | meta/footer ≥4.5:1 |
| **M8** | P1 | M | Latest → non-duplicating, existence-gated dated changelog | core | H3 | no Latest href equals an on-page section anchor; gated on render |
| **M9** | P1 | S | "Return to hub" link in per-site dashboards + index | dashboards | — | every dashboard footer + index links to the hub |
| **M10** | P1 | M | Keyboard/SR-accessible lightbox (dialog, focus trap, close, caption) | core | H4 | Tab opens it; focus trapped; Esc closes+restores; caption shown |
| **M11** | P1 | M | Tests for core primitives + validity invariants | tests | H3,H4,M3 | `pytest tests/test_hubkit.py` green; fails on alt/escape/enum regressions |
| **M12** | P1 | S | One canonical money-figure number verbatim everywhere | gallery | H1 | identical failure string on hub Latest, hero card, gallery caption |
| **L1** | P2 | M | One-line glossary; replace vanity figure count with a result stat | core | H3 | τ/λf defined on-page; header sub not a raw glob count |
| **L2** | P2 | S | Reorder gallery so results lead | gallery | H2 | money figure is first card in the gallery |
| **L3** | P2 | M | Rewrite finding-free signature captions to lead with the conclusion | gallery | H2 | each targeted caption opens with a result |
| **L4** | P2 | M | Reconcile gallery skin with hubkit tokens | gallery | H2,M6 | gallery card/accent matches hub; no third accent |
| **L5** | P2 | M | Consolidate Mingze artifacts into one scoped section | core | M2 | Mingze in exactly one section; every card names its scope |
| **L6** | P2 | S | Harden slug uniqueness; remove dead blockquote CSS path | core | — | duplicate-prefix headings get unique ids (tested) |
| **L7** | P2 | L | Reduce landing-page image payload (dims + optional thumbs) | core | M3 | no `<img>` without w/h; <~1.5 MB when Pillow present; stdlib-only fallback |

## Sequencing (four lanes)

- **core** (`hubkit.py` + `build_project_hub.py`) — one dependency-ordered **serial**
  queue; this is the bulk.
- **gallery** (`build_signature_figures.py`), **dashboards** (separate per-site builder),
  **tests** (new file) — run **concurrently** with core and each other.

- **Wave 0 (P0):** core `H4 → H3 → H1`; gallery `H2` in parallel.
- **Wave 1 (P1):** core `M1 → M2 → M3 → M4`, interleaving independent `M5/M6/M7/M8/M10`;
  dashboards `M9` throughout; gallery `M12`; tests `M11` after H3/H4/M3.
- **Wave 2 (P2):** core `L1/L5/L6/L7` + gallery `L2/L3/L4` concurrently.

Regenerate the hub after every core commit, the gallery after every gallery commit.

## Execution workflow (long autonomous loop)

Per-item cycle: **(1)** implement the minimal edit to the named file(s); **(2)**
regenerate the artifact via its builder (never hand-edit generated HTML); **(3)**
self-verify with cheap deterministic invariant checks (grep the acceptance signature +
`pytest tests/`); **(4)** expert re-review gate — hand the diff + regenerated page to a
read-only sub-agent adopting the lens that raised the item; it re-checks the acceptance
criterion and returns pass / concrete defect list; **(5)** on pass, commit as a small
self-contained unit with the test staged in the same commit; on fail, loop (max ~3)
before escalating.

Because worktree isolation is unusable here (gitignored outputs live only in the main
tree) and file-mutating agents on one tree would collide, the **core lane is driven
serially**; the independent-file lanes (gallery/dashboards/tests) can run as focused
single-file agents. Re-review gates are read-only → safe to parallelize.

**Continuous-improvement loop.** After a wave, re-run the five lens sub-agents over the
regenerated hub+gallery; any new critical/high finding is appended and re-triaged.

**Stop condition.** Terminate when (a) all P0+P1 items passed lens re-review, (b) the
HTML validator reports 0 errors and `pytest` is green, (c) the money-figure first-viewport
screenshot assertion passes, and (d) two consecutive full five-lens sweeps produce no new
critical/high findings. P2 items continue opportunistically but do not gate the stop.

## Guardrails (invariants the loop must never violate)

1. **Generators only** — the hub is produced by `python scripts/build_project_hub.py`,
   gallery/dashboards by their own builders. Never hand-edit generated HTML.
2. **Idempotent regeneration** — unchanged inputs → byte-identical output (modulo the
   provenance timestamp, which is not a content change).
3. **No new deps, no build step** — stdlib + static HTML/CSS/JS only; no external webfont
   `<link>` (self-host woff2 or system-serif fallback); any `Pillow` use degrades
   gracefully when absent. `file://` portability is a feature.
4. **Respect the builder boundary** — per-site dashboards come from a separate builder;
   fixes there go to that builder and are flagged, not into `build_project_hub.py`.
5. **Preserve degrade-by-existence** — every card/section/changelog link stays gated on
   artifact/anchor existence.
6. **Commit discipline** — small self-contained commits; test staged with the behaviour;
   working tree left green (pytest + builders run) after each commit.
7. **PDF-sync** — if a change touches `technical_report.md`, rebuild the PDF and commit
   both together. (This plan touches generators, not the TR markdown, so it should rarely
   fire.)
