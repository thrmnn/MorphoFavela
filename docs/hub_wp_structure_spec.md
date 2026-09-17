# HUBWP — the MorphoFavela dashboard learns what the work packages are

Opened 2026-09-17 (cycle 3) on the PI's words: *"how can I navigate in the
morphofavela dashboard it seems its not well connected wp7 is not reachable
neither the other intermediate results … prepare an optimal dynamic workflow
for audit mapping of this part of the project and document and structure the
dashboard better then we work on the ux and nav."*
Owner: agent. Repo: MorphoFavela. Verify gate: `.claude/verify-cmd`
(pytest + cockpit check + both P1 lints) plus the new reachability audit.

## Measured 2026-09-17

`outputs/_hub/` holds 33 pages. `index.html` — the dashboard's front door —
contains the string `runs/` **0** times, `ledger` **0** times, `WP-0` **0**
times, `wp07_staged` **0** times. All nine pages of the promotion review pack
are unreachable from it. Every work package that closed this cycle (WP-01
through WP-07, ENGINE, G3, VENTAXIS) left a run of record under `runs/` that
the dashboard never mentions. The dashboard renders the June-era deliverables
and knows nothing about the C′ programme that replaced them.

The hub on the brisaverse side had the same disease and got the same cure
tonight (`brisaverse/docs/critic/nav_audit_2026-09-16.md`): the check that
claimed to test reachability tested dangling hrefs, and 15 of 29 pages were
invisible. Do it the same way here, in this order, and do not skip the order —
the PI asked for it explicitly: **reachability and function first, structure
second, UX third.**

## Phase 1 — the audit becomes a gate (functionality + robust reachability)

`scripts/audit_hub_graph.py`: breadth-first from `outputs/_hub/index.html`
over NAVIGABLE links — an anchor inside a card or a nav, not one buried in a
running `<p>`; resolve relative hrefs, treat a directory as its `index.html`.
Report every page under `outputs/_hub/` in one of: REACHABLE (with depth),
PROSE-ONLY (linked, but never from a card or nav), ORPHAN (no inbound at all).
Also verify every `href` that points inside `outputs/` resolves to a file — a
dangling link is a functional failure, reported separately. Exit 1 on any
ORPHAN, PROSE-ONLY or dangling link. Print the whole map, sorted by depth, so
the output IS the audit mapping the PI asked for. Wire it into
`.claude/verify-cmd` and into `Makefile`'s lint target.

**Prove it red** before anything else: demote one card link on a copy of the
index into a sentence, run, confirm exit 1 naming the page; restore, confirm 0.
Paste both. Then run it on the real tree and paste the map — it will be red,
and that red is Phase 2's worklist.

## Phase 2 — structure (generated, never hand-written)

Extend `scripts/build_project_hub.py` — the generator, not the HTML:

1. **A Work-packages section on the index**, one card per run of record. The
   source of truth is the ledger's own declaration:
   `runs/wp07_ledger_<latest>/ledger.json#/_meta/runs_of_record` — read the
   map, never type a run id. Each card: the WP id and one-line "what it
   computed" (from the run's manifest or `brisaverse/shared/facts/tasks.json`
   title, read not written), status, UTC, the gate it feeds, three to six
   headline numbers read from the ledger entries whose `source.run_id` is
   that run, and links to the run's own `summary.json` / `*.md` report. Order
   them in the C′ causal order (WP-01 → … → WP-07), not alphabetically.
2. **The review pack and the staged figures** get a card each in
   Deliverables, generated from `scripts/build_review_pack.py`'s output
   directory, not hand-linked.
3. **A site map page**, `outputs/_hub/map.html`, generated from the same walk
   Phase 1 performs: every page under `_hub/` grouped by section, with its
   depth from the index. Linked from the index nav. This is the durable form
   of "audit mapping": the map is a by-product of the check, so it cannot
   drift from what exists.
4. Every new card is a real card (a `.card` anchor), so Phase 1's gate sees it.

Re-run Phase 1. It must now be green on the real tree. Paste the map.

## Phase 3 — UX and navigation (only after Phases 1–2 are green)

The orchestrator runs the screenshot-and-grade critic loop on the rebuilt
dashboard: round A grades function and reachability against the map (every
WP card resolves, every number matches its ledger id, every link lands);
round B grades UX from the PI's perspective on a tablet. Your part in Phase 3
is the fixes each round names — nothing pre-emptive.

## Explicitly OUT

- Hand-editing anything under `outputs/_hub/` — it is generated.
- Touching `scripts/build_site_dashboard.py` (the folha), `src/brisa_solar/`
  (WP-07M is editing `wp07_figures.py` in a parallel worktree — do not open
  that file), or any run of record.
- Any favela-versus-formal comparison in any form — red line L1.
- Promoting anything out of `runs/` or `outputs/_hub/`.

## Never

- Never type a number, run id or path that exists in a file — read it by code.
- Never say "WHO" for the 2 h floor — Athens Charter (1943), Point 26.
- Never write the banned regime tokens (`lint_p1_tokens` runs in the gate).
- Never report Phase 1 green without having shown it red first.
- Background jobs never notify you; foreground, or wait on the PID.
