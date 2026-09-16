# FOLHA v3 — restructure the Folha de Rua around the grid

Opened 2026-09-16 (cycle 3). Owner: agent. Repo: MorphoFavela.
Verify gate: `.claude/verify-cmd`.
Critic loop: up to 3 rounds, screenshot + grading, run by the orchestrator.

## The PI's brief, verbatim

> "lets start with the structure no need to show the other sites, improve the
> structure and layout, would be nice to show grid based analysis, Terrain,
> density, and then the svf and sunlight would be nice could be interesting to
> show a zoom inlet on some key areas aswell. avoid unnecessary text prefer
> clear and readable visuals or analysis, keep working on the graph."

Five instructions, and the order of the sheet should follow the order he gave
them: **terrain → density → SVF → sunlight**. That is a causal reading of the
site — the ground, then what was built on it, then what the building does to the
sky, then what the sky delivers. Make the sheet read that way.

## What exists today

`scripts/build_site_dashboard.py` renders an A3 portrait sheet in an 8-row
gridspec: masthead · identity card · hero map (street SVF) · hero legend ·
SVF ridgeline | solar ridgeline · SVF×solar hexbin · **cross-site small
multiples** · caveat strip. `--all` builds five sites; atoms are saved
individually via `_save_atom`.

The last critic round scored it 7.0 / 8.6 / 8.0 with three standing residuals:
Maré hero-panel whitespace (flagged 3×), Rocinha/Alemão hexbin and marker
density, and text-heavy panels.

## The data you have — one file, one grid, all five sites

`outputs/<site>/geometry_indicators/per_patch_geometry.csv`, 10 m cells with
`center_x` / `center_y` (EPSG:31983) and, among 35 columns:

  - **terrain** — `slope_deg`, `sigma_h` (height roughness), `H_mean`
  - **density** — `lambda_p` (plan-area), `lambda_f_mean` (frontal-area)
  - **sky** — `svf`, `svf_c_p50`
  - **sunlight** — `kwh_m2_p50`, `sun_h_winter_p50`, `share_ge_2h_winter`
  - **constraints** — `constraint_{vertical,lateral,directional}`, `n_constraints`

Use this as the single source for the grid row: four panels on the **same grid,
same extent, same cell size**, so they are read as one analysis rather than four
charts. Verify the column names by reading the header, not from this list.

Note: `data/vidigal/raw/` has **no DTM** (checked 2026-09-16; the other four
have one). So terrain comes from `slope_deg` in the table, which exists for all
five — not from a hillshade. If you want a hillshade as context it is optional
and Vidigal must degrade gracefully, never render an empty panel.

## The structure to build

1. **Masthead** — keep, tighten.
2. **Identity card** — keep only the numbers that orient a reader who has never
   seen the site (extent, cell count, observer count). Everything else goes.
3. **The grid row — the new spine.** Four small maps, left to right, in the
   PI's order: **Terrain · Density · SVF · Sunlight**. Same extent, same
   projection, same cell size, aligned axes, one shared boundary outline. Each
   gets its own sequential colormap and a compact horizontal colorbar with units.
   No per-panel prose — a two-or-three-word title and the unit is the whole
   caption.
4. **Zoom inlets.** Two (at most three) zoom windows on key areas, **selected by
   code, never by eye**, and the selection rule printed in the sheet's provenance
   line. A defensible rule: the densest cluster of `n_constraints == 3` cells,
   and the cluster with the lowest `svf` decile — i.e. the places the analysis
   itself says are extreme. Each inlet shows the same four layers at zoom, or
   the two that matter there, with a locator rectangle drawn on the grid row so
   the reader can find it. If a site has no qualifying cluster, say so in one
   line and drop that inlet — never fabricate a window.
5. **The graph — keep working on it.** The SVF×solar relation stays and is the
   sheet's one analytical (non-map) panel. The standing residual is marker and
   hexbin density on Rocinha and Alemão. Fix it properly: choose binning from
   the data (cell count and dynamic range), not a constant that happens to suit
   Vidigal. Keep the existing `PowerNorm` treatment if it still serves.
6. **Caveat strip** — keep, but reduce to what is load-bearing.
7. **DELETE the cross-site small-multiples strip.** The PI asked for it gone.
   Remove the panel and its `draw_small_multiples` call from the layout; you may
   leave the function if another surface calls it — check first with a grep.

**Text budget.** Every panel that is not the masthead or the caveat strip gets a
title and a unit and nothing else. If an explanation feels necessary, that is a
sign the visual is not working — fix the visual. This is the instruction the
critic will grade hardest.

## Non-negotiables

- The ridgelines: keep them ONLY if they still earn their space after the grid
  row exists. They may be redundant with the four maps. Decide it, act on it,
  and say in your final message which way you went and why. Do not keep a panel
  because it is already written.
- Honesty carries forward from the last round: the observer-count tile must
  keep showing the TRUE total with the display sample as a secondary line (this
  was a real defect fixed in the last cycle — do not regress it), and the
  quadrant-fallback titles must stay conditional (Maré has no street-class
  column, so its ridgeline groups by orientation quadrant and the title must
  say so).
- A3 portrait stays (297 × 420 mm). Both `folha_<site>_A3.png` and
  `folha_<site>.pdf` keep their paths — the hub links them.
- `--all` must still build all five sites, and all five must render without an
  empty or placeholder panel. Run it and paste the tail.

## Explicitly OUT

- The interactive HTML twin (`scripts/build_html_dashboard.py`) — a follow-on
  task keeps it in step; do not touch it this round.
- Manuscript prose, any /ops card, the ledger, the P1 lints.
- Any favela-versus-formal comparison, in any panel, in any form — red line L1.
- Promoting anything out of `outputs/_distribution/`.

## Never

- Never type a number that exists in a file — read it by code.
- Never pick a zoom window by eye; the rule must be in the code.
- Never render a panel with fabricated or placeholder content; a missing input
  is a stated gap, not a drawn box.
- Never say "WHO" for the 2 h floor — Athens Charter (1943), Point 26.
- Background jobs never notify you: run the five-site build in the foreground,
  or background it and wait on its PID with
  `timeout 540 tail --pid=<PID> -f /dev/null`.
