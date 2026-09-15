# WP-04F façade fix — spec (2026-09-15)

Status: implementer brief for task WP04F. Not on the ground-level critical
path. Laptop only; minutes of GPU.

## The defect

WP-04 façade results (`runs/wp04_sites_20260914T230606Z/<site>/facade.parquet`)
have SVF medians 0.02–0.10 with 17 % exact zeros at Vidigal and 38 % below
0.01. Part is physical (a front hemisphere fully blocked across a ≤ 1 m
alley), part is suspected self-occlusion: `sample_facade_points` insets the
point ~0.5 m outside the wall, but on the 1 m raster the point's own cell —
or its neighbours along the wall — can be marked as its own building
(cell-centre rule), so the horizon march reads its own roof.

## Deliverables

1. In `src/brisa_solar/wp02_horizon.py`, an optional `exclude_building_id`
   path: `patch_visibility(..., obs_building=(n,) int, building_id_raster)`
   where surface cells whose building id equals the observer's are sampled at
   the DTM height instead of the roof during that observer's march. Requires
   `wp02_surface.build_surface` to also write a `building_id` raster (int32,
   0 = none) — add it, keep existing outputs unchanged.
2. A measured comparison on Vidigal and Rio das Pedras façades (all storeys):
   (a) baseline (as run), (b) own-building exclusion, (c) inset 1.5 m, (d) b+c.
   Report per variant: share exactly zero, share < 0.01, median, p25/p75,
   and the median by storey (height_above_ground bins 0–3, 3–6, 6–9, > 9 m).
   Write `runs/wp04f_facade_<UTC>/comparison.json` + `.md`.
3. Physics check that must hold for every variant: an unobstructed vertical
   façade reads 0.5 ± 0.04 (existing test), AND a façade facing a wall of
   height H at distance D reads the closed form for a vertical surface
   opposite an infinite parallel wall — derive it from the same
   patch-centre visibility rule the canyon test uses, and assert both
   engine variants reproduce it on a synthetic raster.
4. If (b) alone removes the exact zeros that are not explained by a real
   opposite wall within 1 m, make it the default for façade observers in
   `wp04_sites.py` and re-run the five sites' façade set only, writing a new
   `runs/wp04_sites_<UTC>/` that carries ground/street by hard link or copy
   from the accepted run and states so in the manifest. If not, leave the
   default, report, and stop.
5. `tests/test_wp04f_facade.py`: synthetic own-building exclusion (a point
   0.5 m outside a 10 m cube sees the same sky as a point 1.5 m outside);
   the opposite-wall closed form; building_id raster round trip.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_wp04f_facade.py tests/test_wp04_sites.py tests/test_wp02_horizon.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py && python3 scripts/lint_p1_tokens.py
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

Your LAST action is `git add <files> && git commit`; paste the hashes. Never
the literal 145; no params.yaml numbers; no "flow"; no map outside runs/.
