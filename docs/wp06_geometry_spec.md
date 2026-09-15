# WP-06 geometry-only ventilation-potential table — spec (2026-09-15)

Status: implementer brief for task WP06. Gated on WP-04 (done). Laptop only,
CPU is enough. Plan of record §3 W5–8; policy `docs/p1_column_allowlist.json`.

## What already exists (reuse, do not re-derive from scratch)

- `outputs/<site>/morphometrics/grid/grid_metrics.gpkg` — the 10 m built-cell
  grid with λp, λf per sector (N…NW, exactly 180°-symmetric), λf_mean/max,
  porosity, σ_h, H_mean, slope, aspect, northness/eastness, built_mask.
- `scripts/run_wind_exposure.py` — `wind_exposure = Σ_θ freq(θ)·λf(θ)` from
  `data/<site>/wind_rose.json` (measured INMET/ASOS sectors, calm excluded)
  and `exposure_ratio = wind_exposure / λf_mean`.
- `scripts/run_lateral_connectivity.open_edge_distance` — lateral depth.
- `scripts/run_ventilation_index.count_constraints` — the council-locked
  ORDINAL index (2026-06-28): three independent geometric constraints counted,
  never summed as magnitudes: vertical (λf_mean ≥ 0.65, Oke 1988 threshold),
  lateral (open-edge distance ≥ the cross-site median), directional
  (exposure_ratio ≥ 1.0). `n_constraints ∈ {0,1,2,3}`.
- WP-04 ground results at 1 m: `runs/wp04_sites_20260914T230606Z/<site>/ground.parquet`
  (svf, kwh_m2, direct-sun hours, thresholds) — the C′ solar columns.

Those scripts write into `outputs/paper_figures/` (a release surface) and use
pre-C′ vocabulary. WP-06 does NOT touch `outputs/paper_figures/`.

## Deliverables

1. `src/brisa_solar/wp06_geometry.py` — per site, from the grid + wind rose +
   lateral depth + WP-04 ground parquet, write
   `outputs/<site>/geometry_indicators/per_patch_geometry.csv` (the path the
   allowlist's `p1_output_convention` names) with ONLY allowlisted columns.
   The code must never import `src/cfd_integration` or
   `scripts/analyze_cfd_results` (the lint's "code leak" check will fail
   otherwise). Cell unit = the 10 m grid cell (`zone_id` → `patch_id`).
   Columns from the existing legal set: patch_id, center_x, center_y, svf,
   lambda_p, slope_deg, porosity, sigma_h, aspect_deg, aspect_sin, aspect_cos,
   aspect_wind_alignment, n_directions, weight_method, lambda_f_N…NW,
   lambda_f_mean, lambda_f_max, lambda_f_max_dir, H_mean.
   NEW columns to ADD to `p1_legal` in `docs/p1_column_allowlist.json` (edit
   the policy file, one commit of its own, with a `_changelog` entry):
   `svf_c_p50` (median of the WP-04 1 m ground SVF inside the cell, engine =
   raster horizon, sky = epw_weighted), `kwh_m2_p50`, `sun_h_winter_p50`,
   `share_ge_2h_winter`, `wind_exposure`, `exposure_ratio`, `open_edge_dist_m`,
   `constraint_vertical`, `constraint_lateral`, `constraint_directional`,
   `n_constraints`. The lint's schema-drift check pins the SOURCE CFD table's
   header, not the allowlist, so adding legal columns does not break it —
   verify by running it.
2. `scripts/lint_p1_tokens.py` — the plan's second rail (§2 "banned-token
   grep"): fails if any P1 pipeline source (`src/brisa_solar/**`,
   `scripts/lint_p1_*.py`, `docs/p1_column_allowlist.json`) or any
   `outputs/*/geometry_indicators/*` header/comment contains the tokens
   `flow`, `CFD`, `OpenFOAM`, `age-of-air`, `tau`/`τ`, `ACH`, `k-omega`,
   `skimming`, case-insensitive, whole-word; an explicit allow-comment
   `# p3-forward-reference` on the same line exempts a single P3 pointer.
   Wire it into `.claude/verify-cmd` and the Makefile test target next to
   `lint_p1_columns.py`. Prove it can fail by planting a token once.
   Note: this will flag `src/brisa_solar/` docstrings that currently say
   "flow" — fix the wording (describe geometry: "λf ≥ 0.65, Oke's threshold",
   never "skimming flow regime").
3. `runs/wp06_geometry_<UTC>/summary.json` — per site: n cells, share per
   `n_constraints` value, medians of the new columns, and a comparison of the
   constraint shares against `outputs/paper_figures/ventilation_index.json`
   (read-only) with the differences and their cause (e.g. svf source changed
   from clear-sky street SVF to the C′ engine). Every number PROVISIONAL where
   it depends on WP-05/G3 cards.
4. `tests/test_wp06_geometry.py`: (a) the written CSV carries only allowlisted
   columns (import the policy, do not restate it); (b) `count_constraints`
   parity on a synthetic grid against `scripts.run_ventilation_index`'s
   function; (c) wind exposure equals Σ freq·λf on a synthetic cell; (d) the
   token lint fails on a planted token and passes on the clean tree; (e) the
   module imports no CFD code (inspect `sys.modules` after import).

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_wp06_geometry.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py && python3 scripts/lint_p1_tokens.py
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

Your LAST action is `git add <files> && git commit`; paste the hashes.

## Never

No params.yaml numbers; no literal 145; no paper prose; nothing written into
`outputs/paper_figures/`, `shared/figures/` or `papers/`; no import of CFD
code; no "flow".
