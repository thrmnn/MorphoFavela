# WP-03B — why the TLS-vs-ALS G2 comparison read r ≈ 0 (diagnostic) — spec (2026-09-15)

Status: implementer brief for task WP03 (second phase). The first phase
(`runs/wp03_tls_20260915T205422Z`, module `src/brisa_solar/wp03_tls.py`) is
merged and its numbers are ledgers of record, but its G2 verdict (floor =
null, r 0.02–0.15 in every alley class, ALS − TLS median Δ −0.3 to −0.6) is
NOT ACCEPTED: it is very likely an artefact of the surface construction, not
a property of the 2.5D model. Do not re-run the PDAL extraction (430 s, rasters
exist under `runs/wp03_tls_20260915T205422Z/*.tif`; if they were cleaned,
regenerate with the existing module).

## The suspected artefact (verify, do not assume)

`wp03_tls.py:415` fills every TLS cell without coverage with the ALS **DTM**
(`tls_dsm_filled = where(dsm != nodata, dsm, dtm_on_tls)`). A terrestrial
scanner standing in alleys sees façades and little of the roofs and nothing
behind the first row of buildings, so uncovered cells are overwhelmingly
BUILDING cells — filling them with terrain deletes those buildings from the
TLS surface. Observers then see far more sky on the "TLS" surface than on the
ALS 2.5D surface (Δ negative everywhere), and the per-point correlation
collapses. A second suspect is the 3-cell (3 m) planar shift the registration
residual found (r 0.979 at the shifted position; median ground Δ +1.56 m, p95
9.6 m) — either a real registration offset, a half-cell origin mismatch
between `rasterio.merge`'s grid and the ALS raster, or the `min` ground
picking sub-terrain noise.

## Deliverables (extend `wp03_tls.py`, do not fork it)

1. **Coverage diagnostic**: share of TLS-uncovered cells that lie inside a
   2019 footprint vs on ground; a coverage map PNG under `runs/` (withheld
   class, fine there). This alone confirms or kills the hypothesis — report
   it first.
2. **Three surface variants for the G2 comparison**, same engine, same
   observer sample as phase 1 (reuse its seed and class rule):
   (a) phase-1 as merged (DTM fill) — for continuity;
   (b) **ALS fill**: uncovered TLS cells take the ALS 2.5D surface value, so
       only TLS-covered cells differ between the two surfaces;
   (c) **covered-only observers**: variant (b) but keep only observers whose
       cells within `max_dist_m` of the march have ≥ 80 % TLS coverage
       (report how many survive per class).
   Per variant and class: n, r, median Δ, p95 |Δ|, share |Δ| ≤ 0.10; the G2
   floor label per variant; `g2_result_v2.json` + `report_v2.md`.
3. **Shift check**: (i) recompute the ground residual after shifting the TLS
   rasters by the detected (dx, dy) and after ±0.5-cell origin corrections —
   which one drives the residual to ~0 median; (ii) rerun variant (b) on the
   shift-corrected TLS surface as variant (d). Report the four medians side by
   side. Never adopt the shift silently: variants stay labelled.
4. **Ground definition check**: replace `min` with the 5th percentile of Z
   per cell (PDAL `writers.gdal output_type=…` has no percentile — do it in
   numpy from a `count`-weighted approach or via `filters.smrf` if it runs
   in < 10 min on the 1 m grid). Report the residual under both.
5. Tests (`tests/test_wp03_tls.py`, extend): (a) ALS-fill leaves TLS-covered
   cells untouched and uncovered cells equal to the ALS surface on a synthetic
   pair; (b) covered-only filter keeps exactly the observers whose march disc
   meets the coverage share on a synthetic coverage raster; (c) the shift
   corrector moves a synthetic raster by exactly (dx, dy) cells.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_wp03_tls.py tests/test_wp02_horizon.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py && python3 scripts/lint_p1_tokens.py
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

Commit last; paste hashes. Never the literal 145; no typed numbers; no edits
to params.yaml, solar_gates, or phase-1's run folder (write a NEW
`runs/wp03_tls_<UTC>/`). Background jobs never notify you — poll with
`timeout 540 tail --pid=<PID> -f /dev/null`.
