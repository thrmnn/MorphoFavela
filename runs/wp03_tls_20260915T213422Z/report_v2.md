# WP-03B TLS diagnostic -- 2026-09-15T21:38:06Z

device=cuda, sky_patches=145

## Deliverable 1 -- coverage diagnostic

share uncovered of total TLS-grid cells: 0.5603241257977938
of uncovered cells: inside a 2019 footprint = 0.700776934888731, on open ground = 0.299223065111269
n_uncovered=43633 (inside footprint=30577, on ground=13056)
coverage map: coverage_map.png (withheld class)

## Deliverable 2 -- G2 surface variants

### variant `a_dtm_fill_merged` -- floor: **None** (n_candidates=15968, n_used=15968)

| class | n | r | median delta | p95 |delta| | share<=0.10 |
|---|---|---|---|---|---|
| <1.5m | 4567 | 0.01507604543529259 | -0.5704533272448191 | 0.8157270181197441 | 0.025399605868184804 |
| 1.5-3m | 3006 | 0.10909351707877243 | -0.46423090643328047 | 0.7176532428449494 | 0.06254158349966733 |
| >3m | 8395 | 0.1527494874820359 | -0.30572663067097866 | 0.5994127340087726 | 0.1650982727814175 |

### variant `b_als_fill` -- floor: **None** (n_candidates=15968, n_used=15968)

| class | n | r | median delta | p95 |delta| | share<=0.10 |
|---|---|---|---|---|---|
| <1.5m | 4567 | 0.021677690815187448 | -0.5555140567504718 | 0.7953267250244088 | 0.028246113422377928 |
| 1.5-3m | 3006 | 0.11161136772685169 | -0.45204697798098226 | 0.7034199752308505 | 0.06520292747837658 |
| >3m | 8395 | 0.15345080329066302 | -0.3000629533657551 | 0.5861206066680671 | 0.16914830256104824 |

### variant `c_covered_only_observers` -- floor: **None** (n_candidates=15968, n_used=0)

| class | n | r | median delta | p95 |delta| | share<=0.10 |
|---|---|---|---|---|---|
| <1.5m | 0 | None | None | None | None |
| 1.5-3m | 0 | None | None | None | None |
| >3m | 0 | None | None | None | None |

note: coverage_share_disc uses max_dist_m=500 m (the same radius patch_visibility marches to), which exceeds the TLS raster's own extent (303x257 cells at 1 m) -- the disc is effectively the whole raster for nearly every observer, so the per-observer share is close to the raster's global coverage fraction everywhere, and the >=80% filter can retain very few or zero observers.

### variant `d_als_fill_shift_corrected` -- floor: **None** (n_candidates=15876, n_used=15876)

| class | n | r | median delta | p95 |delta| | share<=0.10 |
|---|---|---|---|---|---|
| <1.5m | 4450 | 0.010982683833500895 | -0.529964912141465 | 0.792235463779261 | 0.06629213483146068 |
| 1.5-3m | 3005 | 0.032214474936633965 | -0.4504762243481755 | 0.7120072471131138 | 0.07753743760399334 |
| >3m | 8421 | 0.062417502299073876 | -0.34515792859143685 | 0.6140747426271553 | 0.12136325852036575 |


## Deliverable 3 -- shift check

detected (dx, dy) cells (from this run's `registration_residual`): {'dx_cells': 3.0, 'dy_cells': 0.0}

| variant | n | median delta (m) | p95 |delta| (m) |
|---|---|---|---|
| baseline (unshifted) | 7384 | 1.561149806826549 | 9.619940685086304 |
| detected shift | 7360 | 2.3891261680767286 | 10.681997955287155 |
| origin +0.5 cell (x) | 7396 | 1.7296016402151508 | 9.59556887457532 |
| origin -0.5 cell (x) | 7427 | 1.3947452826007805 | 9.310783064010735 |

closest to zero median: **origin_minus_0p5_cell_x**

## Deliverable 4 -- ground definition check (min vs SMRF)

min ground: n=7384, median delta=1.561149806826549 m, p95=9.619940685086304 m
SMRF ground: n=6881, median delta=1.2961287054346826 m, p95=6.6538918543240015 m (elapsed 208s)
note: PDAL writers.gdal has no per-cell-percentile output_type; SMRF ground classification (the spec's own offered fallback) substitutes for a literal numpy 5th-percentile-per-cell, which would require extracting and binning ~262M raw points -- not tractable in this check's time budget. Reported as SMRF, not relabelled as a percentile.
