# WP-07Z — high-resolution citywide figures and zoom extracts, staged and withheld

Opened 2026-09-17 on the PI's words: *"make sure that we high very high
resolution for the citywide figures we will need to be able to zoom in, also
extract each favela that we study for zoomed view and other areas such as
ipanema and other relevant points."* Owner: agent. Repo: MorphoFavela.
Verify: `TMPDIR=/tmp python -m pytest tests/test_wp07_figures.py -q && python3 scripts/lint_p1_tokens.py`.

## The boundary, first

Every output of this task is a **per-cell citywide layer or an extract of one**,
so every output is `release_class: withheld`, `red_line: L1`, registered so the
PI can SEE it (principle of record: release class governs where an artefact may
GO, never whether the PI can see it). In addition:

- **No composite that places a favela extract beside a non-favela extract.**
  Each window is its own figure. Assembling them side by side is the deficit
  map the red line withholds; if a contact sheet is produced for the critic
  loop, it is favela windows only, or non-favela windows only, never mixed.
- **No difference, ratio, deficit or "vs" quantity** between any two windows.
- **Window names come from `config/zoom_windows.yaml`, which the PI owns.**
  It is seeded with the five study favelas (from the boundary layer on disk)
  and Ipanema (named by the PI). An agent never adds a named place. If a
  named place has no boundary on disk, the window is NOT drawn — the manifest
  records `status: missing_boundary` and says what input is needed. Never
  type a bounding box from memory.

## Scope

1. **`config/zoom_windows.yaml`** — one entry per window: `id`, `label`,
   `source` (`favela_boundary:<name>` resolved from the favela limits layer, or
   `bairro:<name>` resolved from a neighbourhood layer if one exists on disk,
   or `bbox_epsg31983: [xmin, ymin, xmax, ymax]` supplied by the PI), `pad_m`.
2. **Citywide at high resolution.** Extend `wp07_figures.py`'s map target
   (`--target map`) with `--pixel-m` (default from the manifest's current
   value, so nothing changes silently) and produce a `10 m`-pixel citywide
   pair (SVF, irradiation) — read the run-of-record extent to size the raster;
   expect a large PNG (state its dimensions and bytes in the manifest). Keep
   the aggregation streamed (`np.bincount` per batch), never a scatter.
3. **Windows at native resolution.** For each entry in the YAML: clip the run
   of record to the window (+ pad) and render SVF and irradiation at the
   native 1 m cell — `f6_zoom_<id>_svf.png` / `_kwh.png` — with a scalebar,
   north arrow, the window's boundary outlined, and a locator inset showing
   where the window sits citywide. Same colour ramps and limits as the
   citywide figure so a reader can move between them.
4. **A self-contained pan/zoom viewer.** `outputs/_hub/wp07_staged/zoom/index.html`
   listing the citywide high-res image and every window, each opening in a
   pan/zoom view implemented inline (no CDN — the mirror must be
   self-contained; a few dozen lines of pointer-event JS on an `<img>` is
   enough). Cards, not prose links, so the hub's reachability gate sees them.
   Register it in `build_project_hub.py` as a Deliverables card and copy the
   images into `outputs/_hub/wp07_staged/zoom/` (the L1 guard forbids a
   `runs/` segment under `_hub/`).
5. **Manifest + register.** One `figure_manifest.json` in
   `runs/wp07_zoom_<UTC>/` in the same shape the register generator consumes
   (`figures: {id: {status, png_path, release_class: "withheld", red_line: "L1",
   window, pixel_m, ...}}`). Add the family to
   `brisaverse/shared/facts/gen_p1_artifacts.py`'s `STAGED_SOURCES` and to the
   mirror allowlist and bridge sync — say in your final message whether you did
   that or left it to the orchestrator (it is a different repo; if your
   worktree cannot reach it, leave it and say so).
6. **Tests**: window resolution from the YAML (favela names resolve; an unknown
   name yields `missing_boundary`, never a guess); every output row is
   withheld/L1; no output filename or manifest field pairs a favela id with a
   non-favela id; the no-contrast grep from WP-07M's test extended to the
   new outputs.

## Explicitly OUT

- Naming any place the PI did not name. Any favela-vs-formal quantity.
  Promotion. HPC. Editing the four charts, the ledger values, or the WP-07M
  maps already produced.

## Never

- Never type a coordinate, extent or number that exists in a file.
- Never say "WHO" for the 2 h floor (Athens Charter 1943, Point 26).
- Never write the banned regime tokens (`lint_p1_tokens` runs in the gate).
- Background jobs never notify you; run in the foreground or wait on the PID.
