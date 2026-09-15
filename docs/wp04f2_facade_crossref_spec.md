# WP-04F2 façade cross-reference against the CPU mesh raycaster — spec (2026-09-15)

Status: acceptance step owed before any façade number is used. Laptop only;
CPU raycasting on a subsample, minutes to an hour.

## Why

Ground and street results are accepted because the raster engine reproduces
the CPU mesh raycaster's street SVF at Rio das Pedras (r 0.994, median |Δ|
0.014). Façades have no such reference yet: medians of 0.03–0.12 with many
exact zeros are physically plausible for ≤ 2 m alleys but unverified.
`src/svf_v2/facades.py::compute_facade_svf` is an independent CPU
implementation on the extruded-polygon mesh (`svf_v2.scene.build_scene`) with
normal-restricted hemispheres — the reference to use.

## Deliverables

1. `scripts/run_wp04f2_facade_crossref.py`: for Rio das Pedras and Vidigal,
   draw a stratified subsample of façade points from
   `runs/wp04_sites_20260915T063553Z/<site>/facade.parquet` — 500 points per
   `height_above_ground` bin (0–3, 3–6, 6–9, > 9 m) per site, seeded from
   `sampling.random_seed` in params.yaml — and evaluate the SAME points
   (same x, y, z, normal) with `compute_facade_svf` on the site's scene mesh
   (`build_scene(dtm_extended_300m, buildings_extended_300m)`), Tregenza
   directions from `generate_tregenza_patches()`. Note the reference's SVF
   convention (count ratio over the forward hemisphere vs solid-angle vs
   cosine-weighted) and compute the raster variant that matches it, exactly
   as `wp02_horizon` did for streets.
2. `runs/wp04f2_facade_<UTC>/crossref.json` + `.md`: per site and per height
   bin: n, Pearson r, median |Δ|, p95 |Δ|, signed median, and the share of
   points where the reference is also exactly zero among the raster zeros.
   Provisional floor, as for streets: r ≥ 0.95 and median |Δ| ≤ 0.03 overall
   per site; report per-bin numbers even where the overall passes. If the
   floor fails, leave the test failing and report the bins — do not loosen.
3. `tests/test_wp04f2_facade_crossref.py`: (a) the subsample is seeded and
   stratified as specified; (b) the reference and raster hemispheres agree
   (same normal → same forward patch set); (c) the crossref floor, skipping
   cleanly if the run output is absent.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_wp04f2_facade_crossref.py tests/test_wp04f_facade.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py && python3 scripts/lint_p1_tokens.py
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

Your LAST action is `git add <files> && git commit`; paste the hashes. Never
the literal 145; no params.yaml numbers; no "flow"; no map outside runs/.
