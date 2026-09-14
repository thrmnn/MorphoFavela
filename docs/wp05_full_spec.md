# WP-05 FULL citywide run — spec (2026-09-15)

Status: implementer brief for task WP05FULL. Runs under the PROVISIONAL default
of the `/ops` card `wp05_run_design` = **A_exhaustive_1m** (every fabric cell,
1 m obstruction surface). A PI tap for B re-executes as the 3 M sample; design
the code so both are one flag. Compute venue: LAPTOP GPU ONLY.

Pilot evidence (runs/wp05_pilot_20260914T202113Z): 8,402,056 frame cells on the
5 m grid across 325 tiles of 2 km; build ≈ 3.8 s per tile, engine marginal
0.31 ms per cell at 1 m → ≈ 1.1 h expected. Peak GPU < 0.1 GB.

## Deliverables

1. `src/brisa_solar/wp05_full.py` (or a `--exhaustive` mode of `wp05_pilot.py`
   — prefer extending, not duplicating): observers = ALL frame cells (same
   frame rules as the pilot, same removal-count logging), 1 m surface per
   tile with the 500 m halo, `march_sampling="nearest"`, checkpoint one
   Parquet per tile under `runs/wp05_full_<UTC>/tiles/`, resumable, then a
   single consolidated Parquet with: x, y, tile, stratum, favela_id (or 0),
   on_building, packed visibility, svf (cosine-weighted), kwh_m2, and the
   `sky_model = epw_weighted` label. Manifest with `sky.patches =
   P1_SKY_PATCHES`, cell_m, params hash, git sha, wall time, peak GB.
2. `runs/wp05_full_<UTC>/distribution.json`: citywide quantiles (1, 5, 10, 25,
   50, 75, 90, 95, 99) of svf and kwh_m2 for all cells and per stratum; the
   five study favelas' (Vidigal, Rocinha, Complexo do Alemão, Maré, Rio das
   Pedras — match by `nome`/`complexo` in Favelas_Limit_2019, report which
   polygons matched) median and IQR, and their percentile position within the
   citywide distribution. Every number carries `"status": "PROVISIONAL —
   wp05_run_design untapped"`.
3. `tests/test_wp05_full.py`: (a) exhaustive mode yields exactly the frame
   count on a synthetic raster; (b) per-tile checkpoints reassemble to the
   consolidated file with no duplicate (x, y); (c) distribution.json quantiles
   are monotone and inside [0, 1] for svf; (d) the five-favela match is
   explicit (test asserts the matched polygon ids are recorded, not that they
   exist).

## Release class (shared/ethics/red_lines.md)

Compute-but-withhold: the per-cell citywide layer and any favela-vs-formal
deficit MAP are `withheld` (red line L1). What P1 claims is the citywide
DISTRIBUTION and the five favelas' positions in it — those numbers are the
deliverable. Do not write any map figure into `shared/figures/` or
`papers/`; a diagnostic PNG under `runs/` is fine.

## Never

No params.yaml numbers; no ORCD; no edits to the engine; no literal 145; no
number in the report that the run did not produce.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_wp05_full.py tests/test_wp05_pilot.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py
TMPDIR=/tmp python -m pytest tests/ -q
```
