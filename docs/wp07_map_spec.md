# WP-07M — the citywide map: gradients of ground solar access, staged and withheld

Opened 2026-09-17 (cycle 3) on the PI's words: *"the figures looked nice but I
expected to see a citywide visualization map with gradients maybe or some kind
of spatial map viz."* Owner: agent. Repo: MorphoFavela.
Verify gate: `TMPDIR=/tmp python -m pytest tests/test_wp07_figures.py -q` then `.claude/verify-cmd`.

## The gap

WP-07B produced four figures from the ledger: a distribution with favela
positions (f1), reference-day sun hours (f2), domain sensitivity (f3), the
constraint-count shares (f4). Every one is a chart. The run of record behind
them — `runs/wp05_full_20260914T215419Z/wp05_full.parquet`, 8,402,056 cells at
1 m with `svf`, `kwh_m2`, `x`, `y`, `favela_id` — has never been drawn as a
map. The headline claim is spatial and nothing shows the space.

## The boundary, stated before the method

A per-cell citywide layer is what **red line L1** withholds, and a
favela-versus-formal deficit map is its hard form. This task does NOT reopen
that. It follows the principle recorded this cycle in
`brisaverse/docs/p1_artifact_register_spec.md`: *release class governs where an
artefact may GO, never whether the PI can SEE it.* So:

- The map is produced into `runs/wp07_map_<UTC>/`, registered as `release_class:
  withheld` under L1, and appears on the P1 page's Figures tab marked as such.
  The PI sees it. Nothing promotes it.
- No favela-vs-non-favela contrast, difference, ratio or deficit is computed or
  drawn — not as a layer, not as a legend, not as an annotation. The five
  favela outlines may be drawn as boundaries so the reader can find them in the
  gradient; that is a position, not a comparison. If you find yourself writing
  a non-favela aggregate, stop.
- Whether a coarser rendition (aggregated, e.g. 100 m or 250 m cells) could be
  read as `reviewer-defence-only` rather than `withheld` is the ethics gate's
  call, then the PI's. Produce that variant too, register it as `withheld`, and
  say in the manifest that its class is a question for the gate — never
  pre-empt the verdict.

## Scope

**1. Two figures, one script.** Extend `src/brisa_solar/wp07_figures.py` (the
existing WP-07B module — same manifest shape, same ledger discipline) with:

  - **f5_citywide_svf_map** — ground SVF as a continuous gradient over the
    whole fabric frame, five favela boundaries outlined, scalebar, north
    arrow, one colorbar. Render from the parquet at the native 1 m through a
    datashader-style aggregation to the output pixel grid (mean per pixel),
    never a scatter of 8.4 M points. A second panel for annual irradiation
    (`kwh_m2`) in the same frame is welcome if it stays legible; if not, make
    it f6.
  - **f5b (aggregated variant)** — the same map at a stated coarse cell (pick
    from the G3 sensitivity grid sizes already in `config/params.yaml` so the
    number is one the project already defends), mean per cell, same styling.
    This is the candidate for a softer release class; it is still registered
    `withheld`.

**2. The manifest.** Each figure carries `release_class: "withheld"`,
`red_line: "L1"`, the ledger ids of any number printed on the figure (the
percentile positions if you annotate them — read from the ledger), the
parquet's run id and UTC, the aggregation method and pixel/cell size, and the
colormap. Register both in the WP-07 figure manifest format so
`brisaverse/shared/facts/gen_p1_artifacts.py` picks them up unchanged — check
that generator's expectations before you write the manifest.

**3. Contact sheet.** Also write a `contact.png` at 1200 px wide for the
critic loop, via `scripts/critic_sheet.py sheet`.

**4. Test.** Extend `tests/test_wp07_figures.py`: both figures produced,
manifest rows carry `release_class == "withheld"` and `red_line == "L1"`, and —
the one that matters — assert that no column, layer or label in the output
references any non-favela aggregate (grep the manifest and any sidecar JSON for
`formal`, `non_favela`, `deficit`, `difference`, `ratio`).

## Explicitly OUT

- Any favela-versus-formal quantity, in any form.
- Promoting either figure anywhere. Staging into `runs/` is the end of this task.
- Touching the four existing figures or the ledger values.
- HPC. The parquet is 8.4 M rows; aggregation to a raster is minutes on the
  laptop CPU — if you find yourself needing more, the method is wrong.

## Never

- Never type a number that exists in a file — read it by code, including the
  sky-patch count (import `P1_SKY_PATCHES`).
- Never say "WHO" for the 2 h floor — Athens Charter (1943), Point 26.
- Never write the banned regime tokens (`lint_p1_tokens` runs in the gate).
- Never pre-empt the ethics gate's class verdict in a label or caption.
- Background jobs never notify you; run in the foreground or wait on the PID.
