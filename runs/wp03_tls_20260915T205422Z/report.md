# WP-03 TLS validity floors -- 2026-09-15T21:01:49Z

device=cuda, pdal=pdal 2.2.0 (git-version: Release), sky_patches=145

## PDAL

e57 scans: 3, point counts: {'Vidigal1-registered_1m': 59747276, 'Vidigal3-registered_1m': 66165363, 'Vidigal4-registered_1m': 136602887, 'Vidigal1-registered_0p5m': 59747276, 'Vidigal3-registered_0p5m': 66165363, 'Vidigal4-registered_0p5m': 136602887}
1 m pass: 215s, total: 430s

## Registration residual (TLS ground vs ALS DTM, replaces the missing .rcp RMSE)

n=7384, clearance>=2.0 m from any footprint
median delta = 1.561149806826549
p95 |delta| = 9.619940685086304
planar shift = {'dy_cells': 0, 'dx_cells': 3, 'r': 0.9788889897395747}

## G2 -- alley-width validity floor

floor label: **None**

| class | n | r | median delta | p95 |delta| | share<=0.10 |
|---|---|---|---|---|---|
| <1.5m | 4567 | 0.01507604543529259 | -0.5704533272448191 | 0.8157270181197441 | 0.025399605868184804 |
| 1.5-3m | 3006 | 0.10909351707877243 | -0.46423090643328047 | 0.7176532428449494 | 0.06254158349966733 |
| >3m | 8395 | 0.1527494874820359 | -0.30572663067097866 | 0.5994127340087726 | 0.1650982727814175 |

### 0.5 m TLS DSM sensitivity

| class | n | median |delta| | p95 |delta| |
|---|---|---|---|
| <1.5m | 4567 | 0.48072516888685135 | 0.7879356346656786 |
| 1.5-3m | 3006 | 0.27879402807940645 | 0.657086826308876 |
| >3m | 8395 | 0.23885404542626165 | 0.5680939629562177 |

## G1 lite -- facade bias (REPORT ONLY, 2.5D facade layer NOT ACCEPTED)

n=2000 of 104394 facade points in the scanned extent
overall mean delta = 0.0748778401775419, median delta = 0.03862637433843985

| storey bin | n | mean delta | median delta | sign consistency |
|---|---|---|---|---|
| 0-3m | 837 | 0.06273144013435032 | 0.018494028003304852 | 0.7359617682198327 |
| 3-6m | 676 | 0.08414468820028555 | 0.05198170414090385 | 0.7943786982248521 |
| 6-9m | 377 | 0.08265250491050191 | 0.06825335023393361 | 0.7639257294429708 |
| >9m | 110 | 0.08370601261800288 | 0.101656893323357 | 0.7272727272727273 |
