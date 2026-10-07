# Ledger before/after — DTM no-data fix (2026-10-07)

Old: `runs/wp07_ledger_20261007T164158Z` · New: `runs/wp07_ledger_20261007T210128Z`. Only the engine fix differs (commit 38a2c49): DTM no-data enters the obstruction surface at 0 m, and the horizon march ignores NaN samples.

## Ledger entries that changed

221 of 487 entries changed

| key | old | new |
|---|---|---|
| citywide.kwh_m2.p1 | 197.2 | 249.5 |
| citywide.kwh_m2.p10 | 686.8 | 753.5 |
| citywide.kwh_m2.p25 | 1042 | 1094 |
| citywide.kwh_m2.p5 | 494.3 | 564.1 |
| citywide.kwh_m2.p50 | 1402 | 1429 |
| citywide.kwh_m2.p75 | 1601 | 1609 |
| citywide.kwh_m2.p90 | 1668 | 1670 |
| citywide.kwh_m2.p95 | 1688 | 1689 |
| citywide.kwh_m2.p99 | 1704 | 1704 |
| citywide.share_ge_1h_equinox | 0.9667 | 0.9744 |
| citywide.share_ge_1h_winter | 0.8702 | 0.8835 |
| citywide.share_ge_2h_equinox | 0.9421 | 0.9533 |
| citywide.share_ge_2h_winter | 0.8203 | 0.8367 |
| citywide.share_ge_3h_equinox | 0.9093 | 0.9243 |
| citywide.share_ge_3h_winter | 0.7655 | 0.7844 |
| citywide.share_ge_4h_equinox | 0.8657 | 0.8848 |
| citywide.share_ge_4h_winter | 0.7044 | 0.725 |
| citywide.sun_h_equinox.p10 | 3.167 | 3.5 |
| citywide.sun_h_equinox.p25 | 5.833 | 6.167 |
| citywide.sun_h_equinox.p5 | 1.667 | 2 |
| citywide.sun_h_equinox.p50 | 8.167 | 8.333 |
| citywide.sun_h_winter.p10 | 0.1667 | 0.5 |
| citywide.sun_h_winter.p25 | 3.167 | 3.5 |
| citywide.sun_h_winter.p50 | 6.333 | 6.5 |
| citywide.sun_h_winter.p75 | 8.333 | 8.5 |
| citywide.svf.p1 | 0.09934 | 0.1234 |
| citywide.svf.p10 | 0.3821 | 0.4182 |
| citywide.svf.p25 | 0.5631 | 0.5912 |
| citywide.svf.p5 | 0.2688 | 0.3068 |
| citywide.svf.p50 | 0.7548 | 0.7723 |
| citywide.svf.p75 | 0.8852 | 0.8911 |
| citywide.svf.p90 | 0.9424 | 0.9446 |
| citywide.svf.p95 | 0.9616 | 0.963 |
| citywide.svf.p99 | 0.9827 | 0.9827 |
| favela.complexo_do_alemao.kwh_m2.percentile | 36.83 | 33.94 |
| favela.complexo_do_alemao.sun_h_equinox.percentile | 38.14 | 35.76 |
| favela.complexo_do_alemao.sun_h_winter.percentile | 32.33 | 30.23 |
| favela.complexo_do_alemao.svf.percentile | 36.84 | 33.59 |
| favela.mare.kwh_m2.iqr_high | 879.1 | 1148 |
| favela.mare.kwh_m2.iqr_low | 384.8 | 521.2 |
| favela.mare.kwh_m2.median | 610.6 | 804.2 |
| favela.mare.kwh_m2.percentile | 7.737 | 11.67 |
| favela.mare.sun_h_equinox.iqr_high | 5 | 7.167 |
| favela.mare.sun_h_equinox.iqr_low | 1.167 | 2 |
| favela.mare.sun_h_equinox.median | 2.833 | 4.167 |
| favela.mare.sun_h_equinox.percentile | 8.749 | 12.75 |
| favela.mare.sun_h_winter.iqr_high | 3 | 3.833 |
| favela.mare.sun_h_winter.median | 1 | 1.333 |
| favela.mare.sun_h_winter.percentile | 13.42 | 13.55 |
| favela.mare.svf.iqr_high | 0.4896 | 0.5818 |
| favela.mare.svf.iqr_low | 0.2123 | 0.2672 |
| favela.mare.svf.median | 0.3399 | 0.4157 |
| favela.mare.svf.percentile | 7.862 | 9.855 |
| favela.riodaspedras.kwh_m2.percentile | 14.46 | 12.05 |
| favela.riodaspedras.sun_h_equinox.percentile | 14.76 | 12.75 |
| favela.riodaspedras.sun_h_winter.percentile | 18.34 | 16.68 |
| favela.riodaspedras.svf.percentile | 13 | 10.74 |
| favela.rocinha.kwh_m2.percentile | 18.5 | 15.84 |
| favela.rocinha.sun_h_equinox.percentile | 19.56 | 17.24 |
| favela.rocinha.sun_h_winter.percentile | 15.8 | 14.3 |
| favela.rocinha.svf.percentile | 18.26 | 15.55 |
| favela.vidigal.kwh_m2.iqr_high | 1264 | 1372 |
| favela.vidigal.kwh_m2.iqr_low | 650.8 | 802.4 |
| favela.vidigal.kwh_m2.median | 994.1 | 1126 |
| favela.vidigal.kwh_m2.percentile | 22.49 | 26.91 |
| favela.vidigal.sun_h_equinox.iqr_high | 8.5 | 8.667 |
| favela.vidigal.sun_h_equinox.iqr_low | 3.667 | 4 |
| favela.vidigal.sun_h_equinox.median | 6.5 | 6.667 |
| favela.vidigal.sun_h_equinox.percentile | 31.23 | 30.42 |
| favela.vidigal.sun_h_winter.percentile | 20.11 | 18.36 |
| favela.vidigal.svf.iqr_high | 0.6096 | 0.7542 |
| favela.vidigal.svf.iqr_low | 0.2972 | 0.4527 |
| favela.vidigal.svf.median | 0.4576 | 0.6247 |
| favela.vidigal.svf.percentile | 15 | 28.95 |
| g3.grid_005_05.complexo_do_alemao.svf_percentile | 39.84 | 37.09 |
| g3.grid_005_05.mare.svf_percentile | 9.083 | 11.79 |
| g3.grid_005_05.riodaspedras.svf_percentile | 13.96 | 11.84 |
| g3.grid_005_05.rocinha.svf_percentile | 18.23 | 15.81 |
| g3.grid_005_05.vidigal.svf_percentile | 16.07 | 31.11 |
| g3.grid_005_10.complexo_do_alemao.svf_percentile | 36.56 | 33.28 |
| g3.grid_005_10.mare.svf_percentile | 7.614 | 9.491 |
| g3.grid_005_10.riodaspedras.svf_percentile | 12.61 | 10.36 |
| g3.grid_005_10.rocinha.svf_percentile | 18.44 | 15.69 |
| g3.grid_005_10.vidigal.svf_percentile | 14.54 | 28.06 |
| g3.grid_005_20.complexo_do_alemao.svf_percentile | 31.77 | 27.76 |
| g3.grid_005_20.mare.svf_percentile | 6.26 | 7.362 |
| g3.grid_005_20.riodaspedras.svf_percentile | 11.85 | 9.338 |
| g3.grid_005_20.rocinha.svf_percentile | 17.4 | 14.21 |
| g3.grid_005_20.vidigal.svf_percentile | 12.72 | 23.5 |
| g3.grid_010_05.complexo_do_alemao.svf_percentile | 40.06 | 37.33 |
| g3.grid_010_05.mare.svf_percentile | 9.286 | 12.11 |
| g3.grid_010_05.riodaspedras.svf_percentile | 14.22 | 12.1 |
| g3.grid_010_05.rocinha.svf_percentile | 18.1 | 15.71 |
| g3.grid_010_05.vidigal.svf_percentile | 16.42 | 31.76 |
| g3.grid_010_10.complexo_do_alemao.svf_percentile | 36.84 | 33.59 |
| g3.grid_010_10.mare.svf_percentile | 7.862 | 9.855 |
| g3.grid_010_10.riodaspedras.svf_percentile | 13 | 10.74 |
| g3.grid_010_10.rocinha.svf_percentile | 18.26 | 15.55 |
| g3.grid_010_10.vidigal.svf_percentile | 15 | 28.95 |
| g3.grid_010_20.complexo_do_alemao.svf_percentile | 32.44 | 28.5 |
| g3.grid_010_20.mare.svf_percentile | 6.639 | 7.903 |
| g3.grid_010_20.riodaspedras.svf_percentile | 12.55 | 10.01 |
| g3.grid_010_20.rocinha.svf_percentile | 17.17 | 14.1 |
| g3.grid_010_20.vidigal.svf_percentile | 13.46 | 24.96 |
| g3.grid_020_05.complexo_do_alemao.svf_percentile | 39.95 | 37.34 |
| g3.grid_020_05.mare.svf_percentile | 9.829 | 12.99 |
| g3.grid_020_05.riodaspedras.svf_percentile | 15.05 | 13 |
| g3.grid_020_05.rocinha.svf_percentile | 18.18 | 15.92 |
| g3.grid_020_05.vidigal.svf_percentile | 17.01 | 32.82 |
| g3.grid_020_10.complexo_do_alemao.svf_percentile | 36.51 | 33.45 |
| g3.grid_020_10.mare.svf_percentile | 8.442 | 10.76 |
| g3.grid_020_10.riodaspedras.svf_percentile | 13.89 | 11.72 |
| g3.grid_020_10.rocinha.svf_percentile | 18.22 | 15.7 |
| g3.grid_020_10.vidigal.svf_percentile | 15.63 | 30.42 |
| g3.grid_020_20.complexo_do_alemao.svf_percentile | 32.7 | 29.12 |
| g3.grid_020_20.mare.svf_percentile | 7.443 | 9.223 |
| g3.grid_020_20.riodaspedras.svf_percentile | 14.04 | 11.62 |
| g3.grid_020_20.rocinha.svf_percentile | 17.42 | 14.68 |
| g3.grid_020_20.vidigal.svf_percentile | 14.45 | 27.5 |
| site.mare.ground.kwh_m2.p10 | 232.7 | 328.1 |
| site.mare.ground.kwh_m2.p25 | 394.6 | 537.7 |
| site.mare.ground.kwh_m2.p50 | 635.2 | 834.6 |
| site.mare.ground.kwh_m2.p75 | 896.6 | 1201 |
| site.mare.ground.kwh_m2.p90 | 1224 | 1512 |
| site.mare.ground.share_ge_1h_equinox | 0.7955 | 0.8834 |
| site.mare.ground.share_ge_1h_winter_solstice | 0.5377 | 0.5838 |
| site.mare.ground.share_ge_2h_equinox | 0.6542 | 0.7736 |
| site.mare.ground.share_ge_2h_winter_solstice | 0.3937 | 0.4486 |
| site.mare.ground.share_ge_3h_equinox | 0.5117 | 0.6552 |
| site.mare.ground.share_ge_3h_winter_solstice | 0.2816 | 0.3495 |
| site.mare.ground.share_ge_4h_equinox | 0.3828 | 0.5478 |
| site.mare.ground.share_ge_4h_winter_solstice | 0.1925 | 0.2789 |
| site.mare.ground.sun_h_equinox.p10 | 0 | 0.6667 |
| site.mare.ground.sun_h_equinox.p25 | 1.333 | 2.167 |
| site.mare.ground.sun_h_equinox.p50 | 3 | 4.5 |
| site.mare.ground.sun_h_equinox.p75 | 5.167 | 7.667 |
| site.mare.ground.sun_h_equinox.p90 | 7.5 | 9.667 |
| site.mare.ground.sun_h_winter.p50 | 1.167 | 1.5 |
| site.mare.ground.sun_h_winter.p75 | 3.333 | 4.5 |
| site.mare.ground.sun_h_winter.p90 | 5.5 | 7.833 |
| site.mare.ground.svf.p10 | 0.1285 | 0.1624 |
| site.mare.ground.svf.p25 | 0.2197 | 0.277 |
| site.mare.ground.svf.p50 | 0.3528 | 0.4293 |
| site.mare.ground.svf.p75 | 0.5043 | 0.6127 |
| site.mare.ground.svf.p90 | 0.6584 | 0.8108 |
| site.mare.street.kwh_m2.p10 | 352.3 | 501.8 |
| site.mare.street.kwh_m2.p25 | 607.4 | 885.5 |
| site.mare.street.kwh_m2.p50 | 952.8 | 1387 |
| site.mare.street.kwh_m2.p75 | 1465 | 1650 |
| site.mare.street.kwh_m2.p90 | 1676 | 1705 |
| site.mare.street.sun_h_equinox.p10 | 1.167 | 2.167 |
| site.mare.street.sun_h_equinox.p25 | 3 | 4.667 |
| site.mare.street.sun_h_equinox.p50 | 5.333 | 8.167 |
| site.mare.street.sun_h_equinox.p75 | 8.5 | 10.67 |
| site.mare.street.sun_h_equinox.p90 | 10.83 | 11.5 |
| site.mare.street.sun_h_winter.p25 | 1 | 1.833 |
| site.mare.street.sun_h_winter.p50 | 3.667 | 5.833 |
| site.mare.street.sun_h_winter.p75 | 6.833 | 9 |
| site.mare.street.sun_h_winter.p90 | 9.167 | 10 |
| site.mare.street.svf.p10 | 0.1926 | 0.2496 |
| site.mare.street.svf.p25 | 0.3495 | 0.4476 |
| site.mare.street.svf.p50 | 0.5567 | 0.7419 |
| site.mare.street.svf.p75 | 0.8108 | 0.9291 |
| site.mare.street.svf.p90 | 0.9575 | 0.9841 |
| site.mare.terrain_split.equinox.buildings_first.buildings_share | 0.8212 | 0.9799 |
| site.mare.terrain_split.equinox.buildings_first.terrain_share | 0.1788 | 0.02007 |
| site.mare.terrain_split.equinox.terrain_first.buildings_share | 0.6156 | 0.9713 |
| site.mare.terrain_split.equinox.terrain_first.terrain_share | 0.3844 | 0.02868 |
| site.mare.terrain_split.equinox.terrain_first.total_loss_h_mean | 8.708 | 7.281 |
| site.mare.terrain_split.winter.buildings_first.buildings_share | 0.9115 | 0.9855 |
| site.mare.terrain_split.winter.buildings_first.terrain_share | 0.08854 | 0.01448 |
| site.mare.terrain_split.winter.terrain_first.buildings_share | 0.7058 | 0.9737 |
| site.mare.terrain_split.winter.terrain_first.terrain_share | 0.2942 | 0.02625 |
| site.mare.terrain_split.winter.terrain_first.total_loss_h_mean | 8.674 | 8.024 |
| site.vidigal.ground.kwh_m2.p10 | 333.9 | 472.2 |
| site.vidigal.ground.kwh_m2.p25 | 648.8 | 791.2 |
| site.vidigal.ground.kwh_m2.p50 | 1017 | 1138 |
| site.vidigal.ground.kwh_m2.p75 | 1280 | 1405 |
| site.vidigal.ground.kwh_m2.p90 | 1481 | 1550 |
| site.vidigal.ground.share_ge_1h_equinox | 0.9044 | 0.9166 |
| site.vidigal.ground.share_ge_1h_winter_solstice | 0.6324 | 0.6358 |
| site.vidigal.ground.share_ge_2h_equinox | 0.8497 | 0.8677 |
| site.vidigal.ground.share_ge_2h_winter_solstice | 0.5493 | 0.552 |
| site.vidigal.ground.share_ge_3h_equinox | 0.7926 | 0.8137 |
| site.vidigal.ground.share_ge_3h_winter_solstice | 0.4765 | 0.4781 |
| site.vidigal.ground.share_ge_4h_equinox | 0.7298 | 0.7511 |
| site.vidigal.ground.share_ge_4h_winter_solstice | 0.4008 | 0.4017 |
| site.vidigal.ground.sun_h_equinox.p10 | 1 | 1.333 |
| site.vidigal.ground.sun_h_equinox.p25 | 3.667 | 4 |
| site.vidigal.ground.sun_h_equinox.p50 | 6.667 | 6.833 |
| site.vidigal.ground.sun_h_equinox.p75 | 8.667 | 8.833 |
| site.vidigal.ground.svf.p10 | 0.1463 | 0.2558 |
| site.vidigal.ground.svf.p25 | 0.2984 | 0.4436 |
| site.vidigal.ground.svf.p50 | 0.4619 | 0.625 |
| site.vidigal.ground.svf.p75 | 0.6175 | 0.7678 |
| site.vidigal.ground.svf.p90 | 0.788 | 0.8559 |
| site.vidigal.street.kwh_m2.p10 | 46.49 | 82.59 |
| site.vidigal.street.kwh_m2.p25 | 296.4 | 393.9 |
| site.vidigal.street.kwh_m2.p50 | 728.5 | 853 |
| site.vidigal.street.kwh_m2.p75 | 1109 | 1206 |
| site.vidigal.street.kwh_m2.p90 | 1321 | 1415 |
| site.vidigal.street.sun_h_equinox.p25 | 1.833 | 2.167 |
| site.vidigal.street.sun_h_equinox.p50 | 5.167 | 5.5 |
| site.vidigal.street.sun_h_equinox.p75 | 8 | 8.167 |
| site.vidigal.street.sun_h_equinox.p90 | 9.5 | 9.833 |
| site.vidigal.street.sun_h_winter.p90 | 7.5 | 7.667 |
| site.vidigal.street.svf.p10 | 0.0109 | 0.03476 |
| site.vidigal.street.svf.p25 | 0.1255 | 0.1967 |
| site.vidigal.street.svf.p50 | 0.3271 | 0.4515 |
| site.vidigal.street.svf.p75 | 0.5081 | 0.6391 |
| site.vidigal.street.svf.p90 | 0.6515 | 0.7552 |
| site.vidigal.terrain_split.equinox.buildings_first.buildings_share | 0.815 | 0.8385 |
| site.vidigal.terrain_split.equinox.buildings_first.terrain_share | 0.185 | 0.1615 |
| site.vidigal.terrain_split.equinox.terrain_first.buildings_share | 0.6228 | 0.6899 |
| site.vidigal.terrain_split.equinox.terrain_first.terrain_share | 0.3772 | 0.3101 |
| site.vidigal.terrain_split.equinox.terrain_first.total_loss_h_mean | 6.115 | 5.942 |
| site.vidigal.terrain_split.winter.buildings_first.buildings_share | 0.7335 | 0.7348 |
| site.vidigal.terrain_split.winter.buildings_first.terrain_share | 0.2665 | 0.2652 |
| site.vidigal.terrain_split.winter.terrain_first.buildings_share | 0.523 | 0.5358 |
| site.vidigal.terrain_split.winter.terrain_first.terrain_share | 0.477 | 0.4642 |
| site.vidigal.terrain_split.winter.terrain_first.total_loss_h_mean | 7.544 | 7.53 |

## derived

| key | old | new |
|---|---|---|
| spread.vidigal | 4.28 | 9.322 |
| spread.rocinha | 1.275 | 1.822 |
| spread.complexo_do_alemao | 8.283 | 9.573 |
| spread.mare | 3.569 | 5.63 |
| spread.riodaspedras | 3.195 | 3.658 |
| rank_under_locked_domain | complexo_do_alemao, rocinha, vidigal, riodaspedras, mare | complexo_do_alemao, vidigal, rocinha, riodaspedras, mare |
| rank_invariant_across_grid | True | False |

## Cross-tab (not a ledger entry): point deficit by constraint count

Old `runs/wp07_crosstab_20261001T125657Z` · new `runs/wp07_crosstab_20261007T201942Z` (inputs `per_patch_geometry_nodata0.csv` from `runs/wp06_geometry_20261007T201840Z`).

| site | n=0 | n=1 | n=2 | n=3 |
|---|---|---|---|---|
| vidigal | 0.479 → 0.475 | 0.623 → 0.617 | 0.705 → 0.702 | 0.650 → 0.649 |
| rocinha | 0.281 → 0.281 | 0.530 → 0.530 | 0.819 → 0.819 | 0.756 → 0.756 |
| complexo_do_alemao | 0.165 → 0.165 | 0.358 → 0.358 | 0.577 → 0.577 | 0.663 → 0.663 |
| riodaspedras | 0.160 → 0.160 | 0.289 → 0.289 | 0.642 → 0.642 | 0.840 → 0.840 |
| maré | 0.313 → 0.250 | 0.503 → 0.403 | 0.767 → 0.722 | 0.839 → 0.818 |
| pooled | 0.229 → 0.225 | 0.423 → 0.415 | 0.688 → 0.680 | 0.796 → 0.787 |

## WP-03C G2 street points (not a ledger entry)

Old `runs/wp03_tls_20260915T215720Z` · new `runs/wp03_tls_20261007T192926Z`. Variants a–d are still copied from `runs/wp03_tls_20260915T213422Z` (pre-fix) by `run_wp03c`.

| class | n | r | median Δ | share within 0.10 |
|---|---|---|---|---|
| <1.5m | 1213 | 0.504 → 0.552 | 0.027 → 0.036 | 0.674 → 0.691 |
| 1.5-3m | 321 | 0.359 → 0.443 | 0.099 → 0.133 | 0.380 → 0.433 |
| >3m | 649 | 0.262 → 0.445 | 0.253 → 0.339 | 0.156 → 0.253 |
