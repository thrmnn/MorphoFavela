# Facade audit, Mingze July CSV (aggregates only)

Run: `facade_audit_july_20261007T164627Z`. Script: `scripts/facade_audit_july.py`. Input rows: 7,631,732. Footprints: `data/RJ/buildings_RJ_2019_utm.gpkg` (EPSG:31983). Site polygons: `match_favela_group` on `Favelas_Limit_2019.shp` (Maré = definition A, 6 IPP polygons). All values below are computed by the script; no building IDs or coordinates are included.

## Q1 Basics (duplicates collapsed by mean per date on building_id, floor_id, facade_key)

Duplicates:

| site | date | rows_raw | bands | dup_extra_rows | dup_groups | dup_groups_differing |
|---|---|---|---|---|---|---|
| Complexo do Alemão | 03-20 | 472124 | 471449 | 675 | 631 | 436 |
| Complexo do Alemão | 06-21 | 472124 | 471449 | 675 | 631 | 305 |
| Complexo do Alemão | 09-22 | 472124 | 471449 | 675 | 631 | 432 |
| Complexo do Alemão | 12-21 | 472124 | 471449 | 675 | 631 | 467 |
| Maré | 03-20 | 784239 | 777416 | 6823 | 6496 | 4292 |
| Maré | 06-21 | 784239 | 777416 | 6823 | 6496 | 3364 |
| Maré | 09-22 | 784201 | 777378 | 6823 | 6496 | 4326 |
| Maré | 12-21 | 784201 | 777378 | 6823 | 6496 | 4718 |
| Rio das Pedras | 03-20 | 276015 | 275810 | 205 | 177 | 104 |
| Rio das Pedras | 06-21 | 276015 | 275810 | 205 | 177 | 70 |
| Rio das Pedras | 09-22 | 276015 | 275810 | 205 | 177 | 106 |
| Rio das Pedras | 12-21 | 276015 | 275810 | 205 | 177 | 125 |
| Rocinha | 03-20 | 375574 | 375367 | 207 | 207 | 91 |
| Rocinha | 06-21 | 375574 | 375367 | 207 | 207 | 69 |
| Rocinha | 09-22 | 375574 | 375367 | 207 | 207 | 84 |
| Rocinha | 12-21 | 375574 | 375367 | 207 | 207 | 112 |

Sun-hours granularity and flag consistency (all rows):

| pct_values_multiple_of_1h | pct_values_multiple_of_0p5h | max_sun_hours | pct_deprived_flag_consistent_with_lt2 |
|---|---|---|---|
| 100.00 | 100.00 | 13.00 | 100.00 |

Per site x date:

| site | date | n_bands | mean_h | median_h | pct_lt2h | pct_zero |
|---|---|---|---|---|---|---|
| Rio das Pedras | 03-20 | 275810 | 2.11 | 2.00 | 48.70 | 36.40 |
| Rio das Pedras | 06-21 | 275810 | 1.26 | 0.00 | 73.85 | 64.64 |
| Rio das Pedras | 09-22 | 275810 | 2.12 | 2.00 | 48.70 | 36.25 |
| Rio das Pedras | 12-21 | 275810 | 2.78 | 3.00 | 35.06 | 19.50 |
| Rocinha | 03-20 | 375367 | 1.46 | 0.00 | 64.40 | 52.04 |
| Rocinha | 06-21 | 375367 | 0.86 | 0.00 | 80.33 | 72.45 |
| Rocinha | 09-22 | 375367 | 1.41 | 0.00 | 64.80 | 52.93 |
| Rocinha | 12-21 | 375367 | 1.83 | 1.00 | 57.00 | 39.05 |
| Complexo do Alemão | 03-20 | 471449 | 2.28 | 2.00 | 47.38 | 36.61 |
| Complexo do Alemão | 06-21 | 471449 | 1.39 | 0.00 | 70.37 | 61.59 |
| Complexo do Alemão | 09-22 | 471449 | 2.28 | 2.00 | 47.45 | 36.68 |
| Complexo do Alemão | 12-21 | 471449 | 3.09 | 3.00 | 36.85 | 22.75 |
| Maré | 03-20 | 777416 | 2.24 | 1.00 | 50.15 | 39.00 |
| Maré | 06-21 | 777416 | 1.44 | 0.00 | 69.13 | 62.11 |
| Maré | 09-22 | 777378 | 2.25 | 1.00 | 50.11 | 38.88 |
| Maré | 12-21 | 777378 | 2.82 | 2.00 | 36.79 | 21.91 |

Persistence (bands present on all four dates):

| site | bands_any_date | bands_all4_dates | pct_persistent_zero | pct_lt2h_all4 | pct_zero_on_0621 | pct_persistent_zero_given_zero_0621 |
|---|---|---|---|---|---|---|
| Rio das Pedras | 275810 | 275810 | 11.22 | 24.41 | 64.64 | 17.36 |
| Rocinha | 375367 | 375367 | 28.72 | 45.56 | 72.45 | 39.64 |
| Complexo do Alemão | 471449 | 471449 | 12.45 | 24.30 | 61.59 | 20.22 |
| Maré | 777416 | 777378 | 9.77 | 22.58 | 62.10 | 15.74 |

## Q2 ID join (building_id -> footprint layer)

Share of Mingze building_id values found in each integer-like field:

| field | n_values | unique | Rio das Pedras | Rocinha | Complexo do Alemão | Maré | all_sites |
|---|---|---|---|---|---|---|---|
| OBJECTID | 2362806 | True | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| cod_projec | 2128613 | False | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cod_unico | 2362768 | False | 59.14 | 58.70 | 93.98 | 91.90 | 82.75 |
| cod_edific | 2128415 | False | 51.67 | 53.82 | 4.35 | 14.22 | 23.00 |
| cod_lote | 1991559 | False | 0.34 | 0.00 | 77.81 | 68.81 | 50.99 |

OBJECTID matches 100%, but its values are a dense 1..N range, so a hit alone proves little. Validation: band z_min equals footprint `base`, and z_max - z_min equals `altura`; the null column shifts the ID by one. Centroid location against site polygons:

| site | buildings | pct_id_in_OBJECTID | pct_zmin_eq_base_0p05 | pct_height_eq_altura_0p01 | null_idminus1_pct_zmin_eq_base | pct_centroid_in_site_polygon | pct_centroid_only_in_other_site_polygon | pct_centroid_in_no_site_polygon | pct_centroid_in_mare_E_outline |
|---|---|---|---|---|---|---|---|---|---|
| Rio das Pedras | 10720 | 100.00 | 99.65 | 99.66 | 20.78 | 93.56 | 0.00 | 6.44 | PLACEHOLDER |
| Rocinha | 13764 | 100.00 | 100.00 | 100.00 | 1.70 | 100.00 | 0.00 | 0.00 | PLACEHOLDER |
| Complexo do Alemão | 21720 | 100.00 | 100.00 | 100.00 | 1.05 | 100.00 | 0.00 | 0.00 | PLACEHOLDER |
| Maré | 37169 | 100.00 | 99.98 | 99.98 | 12.47 | 43.93 | 0.00 | 56.07 | 94.65 |

## Q3 Zero diagnosis

Side index vs footprint edge: does the number of sides equal the number of footprint vertices?

| site | buildings | pct_nsides_eq_nvertices | mean_nsides_minus_nvertices | sd |
|---|---|---|---|---|
| Complexo do Alemão | 21720 | 71.75 | -0.02 | 3.28 |
| Maré | 37156 | 17.89 | -0.00 | 4.55 |
| Rio das Pedras | 10716 | 41.38 | -0.09 | 8.16 |
| Rocinha | 13761 | 15.04 | -0.01 | 8.14 |

Orientation test on buildings where the counts match (upper floors, 06-21, 6000 buildings): correlation of the outward-normal north component with sun_hours under every orientation/start-offset/reversal hypothesis. A real mapping would give a clearly positive r under one hypothesis (June sun is in the north). Largest |r| = 0.0096.

| orientation | reversed | start_offset | r_north_component_vs_sun_hours | buildings |
|---|---|---|---|---|
| CCW | False | 0 | 0.0068 | 6000 |
| CCW | False | 1 | 0.0037 | 6000 |
| CCW | False | 2 | 0.0046 | 6000 |
| CCW | False | 3 | 0.0096 | 6000 |
| CCW | True | 0 | 0.0080 | 6000 |
| CCW | True | 1 | 0.0081 | 6000 |
| CCW | True | 2 | 0.0011 | 6000 |
| CCW | True | 3 | -0.0048 | 6000 |
| CW | False | 0 | -0.0080 | 6000 |
| CW | False | 1 | -0.0081 | 6000 |
| CW | False | 2 | -0.0011 | 6000 |
| CW | False | 3 | 0.0048 | 6000 |
| CW | True | 0 | -0.0068 | 6000 |
| CW | True | 1 | -0.0037 | 6000 |
| CW | True | 2 | -0.0046 | 6000 |
| CW | True | 3 | -0.0096 | 6000 |

Party-wall proxy at building level (share of the footprint perimeter within 0.5 m of another footprint, overlapping footprints excluded) against persistent-zero share of bands:

| site | buildings | pct_overlapping_other_footprint | pct_buildings_attached_ge25 | pearson_attached_vs_persistent_zero_share | pct_bands_persistent_zero_isolated | pct_bands_persistent_zero_all |
|---|---|---|---|---|---|---|
| Complexo do Alemão | 21720 | 55.09 | 67.14 | -0.02 | 13.16 | 12.45 |
| Maré | 37161 | 60.02 | 92.31 | 0.04 | 7.31 | 9.77 |
| Rio das Pedras | 10720 | 45.04 | 94.90 | -0.03 | 12.75 | 11.22 |
| Rocinha | 13764 | 36.07 | 91.02 | -0.03 | 30.61 | 28.72 |

| site | cls | buildings | bands | mean_attached_frac | pct_bands_persistent_zero |
|---|---|---|---|---|---|
| Rio das Pedras | isolated(<2%) | 253 | 9635 | 0.00 | 12.75 |
| Rio das Pedras | 2-25% | 294 | 8490 | 0.17 | 14.35 |
| Rio das Pedras | 25-50% | 1048 | 28286 | 0.39 | 10.54 |
| Rio das Pedras | >=50% | 9125 | 229399 | 0.78 | 11.13 |
| Rocinha | isolated(<2%) | 412 | 8926 | 0.00 | 30.61 |
| Rocinha | 2-25% | 824 | 20228 | 0.16 | 31.75 |
| Rocinha | 25-50% | 2792 | 75342 | 0.39 | 29.83 |
| Rocinha | >=50% | 9736 | 270871 | 0.76 | 28.13 |
| Complexo do Alemão | isolated(<2%) | 3324 | 55842 | 0.00 | 13.16 |
| Complexo do Alemão | 2-25% | 3813 | 83682 | 0.15 | 13.26 |
| Complexo do Alemão | 25-50% | 6807 | 153916 | 0.37 | 12.12 |
| Complexo do Alemão | >=50% | 7776 | 178009 | 0.68 | 12.14 |
| Maré | isolated(<2%) | 1515 | 32533 | 0.00 | 7.31 |
| Maré | 2-25% | 1343 | 29250 | 0.16 | 7.34 |
| Maré | 25-50% | 6726 | 136789 | 0.40 | 9.19 |
| Maré | >=50% | 27577 | 578806 | 0.75 | 10.17 |

Zero share by floor (floor 0 = ground):

| site | bands | pct_zero_0621 | pct_lt2h_0621 | floor | pct_persistent_zero |
|---|---|---|---|---|---|
| Rio das Pedras | 78156 | 64.46 | 73.80 | 0 | 11.24 |
| Rio das Pedras | 75145 | 64.44 | 73.75 | 1 | 11.41 |
| Rio das Pedras | 62802 | 64.82 | 74.10 | 2 | 11.07 |
| Rio das Pedras | 39670 | 65.27 | 74.06 | 3 | 11.41 |
| Rio das Pedras | 20037 | 64.30 | 73.24 | 4+ | 10.54 |
| Rocinha | 114259 | 72.07 | 80.20 | 0 | 28.37 |
| Rocinha | 107103 | 72.76 | 80.48 | 1 | 28.88 |
| Rocinha | 83327 | 72.46 | 80.50 | 2 | 29.33 |
| Rocinha | 46512 | 72.43 | 80.15 | 3 | 28.73 |
| Rocinha | 24166 | 72.93 | 79.99 | 4+ | 27.58 |
| Complexo do Alemão | 193531 | 61.37 | 70.33 | 0 | 12.55 |
| Complexo do Alemão | 168978 | 61.65 | 70.38 | 1 | 12.52 |
| Complexo do Alemão | 86235 | 61.94 | 70.71 | 2 | 12.30 |
| Complexo do Alemão | 20879 | 61.29 | 69.37 | 3 | 11.79 |
| Complexo do Alemão | 1826 | 64.73 | 70.15 | 4+ | 11.01 |
| Maré | 260596 | 62.10 | 69.05 | 0 | 9.35 |
| Maré | 241911 | 61.93 | 68.96 | 1 | 9.68 |
| Maré | 171505 | 62.41 | 69.46 | 2 | 10.21 |
| Maré | 80392 | 62.71 | 69.63 | 3 | 10.59 |
| Maré | 23012 | 59.59 | 67.73 | 4+ | 9.48 |

Whole-building zeros:

| site | buildings | pct_bldg_all_bands_persistent_zero | pct_bldg_no_band_persistent_zero | pct_bldg_gt50pct_bands_persistent_zero | pct_bldg_all_bands_zero_0621 | pct_bands_persistent_zero_in_wholly_zero_bldgs |
|---|---|---|---|---|---|---|
| Complexo do Alemão | 21720 | 1.83 | 56.62 | 7.84 | 24.18 | 9.71 |
| Maré | 37161 | 2.39 | 69.18 | 6.95 | 25.90 | 21.10 |
| Rio das Pedras | 10720 | 2.31 | 62.98 | 8.15 | 24.02 | 14.91 |
| Rocinha | 13764 | 10.84 | 40.63 | 25.33 | 40.20 | 31.00 |

By building height (floors = max floor_id + 1 class):

| site | fg | buildings | bands | pct_bands_persistent_zero |
|---|---|---|---|---|
| Complexo do Alemão | 1 floor | 2755 | 24530 | 13.26 |
| Complexo do Alemão | 2 floors | 9064 | 165491 | 13.00 |
| Complexo do Alemão | 3-4 floors | 9685 | 272538 | 12.14 |
| Complexo do Alemão | 5+ floors | 216 | 8890 | 9.66 |
| Maré | 1 floor | 2547 | 18608 | 6.56 |
| Maré | 2 floors | 9839 | 140715 | 8.32 |
| Maré | 3-4 floors | 21937 | 517447 | 10.27 |
| Maré | 5+ floors | 2838 | 100608 | 9.83 |
| Rio das Pedras | 1 floor | 417 | 2998 | 10.67 |
| Rio das Pedras | 2 floors | 1629 | 24619 | 11.71 |
| Rio das Pedras | 3-4 floors | 6689 | 173232 | 11.38 |
| Rio das Pedras | 5+ floors | 1985 | 74961 | 10.73 |
| Rocinha | 1 floor | 887 | 7156 | 27.57 |
| Rocinha | 2 floors | 2847 | 47542 | 27.96 |
| Rocinha | 3-4 floors | 7807 | 226467 | 29.21 |
| Rocinha | 5+ floors | 2223 | 94202 | 28.02 |

## Q4 July vs v3

| site | july_bands_0621 | v3_bands | ratio | july_buildings | v3_buildings | july_pct_persistent_direct_zero | july_pct_lt2h_all4 | v3_pct_persistent_irradiation_zero | v3_pct_persistent_irradiation_lt0p5 |
|---|---|---|---|---|---|---|---|---|---|
| Rio das Pedras | 275810 | 275810 | 1.00 | 10720 | 10720 | 11.22 | 24.41 | 54.70 | 62.30 |
| Rocinha | 375367 | 376208 | 1.00 | 13764 | 13766 | 28.72 | 45.56 | PLACEHOLDER | 50.80 |
| Complexo do Alemão | 471449 | 471449 | 1.00 | 21720 | 21720 | 12.45 | 24.30 | PLACEHOLDER | 40.30 |
| Maré | 777416 | 775224 | 1.00 | 37169 | 37165 | 9.77 | 22.58 | PLACEHOLDER | 56.20 |

v3 reports exact-zero persistence only as a range (17.2% Vidigal to 54.7% Rio das Pedras); other sites' exact-zero shares are PLACEHOLDER (not in the write-up). July bands are for 06-21 after collapse; v3 bands are collapsed across its own dates.

## Q5 Area weighting

No façade width in the CSV. Sides cannot be mapped to edges (Q3), so edge length x height is not available. Crude proxy: building exterior-ring perimeter / number of sides x (z_max - z_min); this assumes equal-width sides within a building.

| site | bands | pct_lt2h_unweighted | pct_lt2h_height_weighted | pct_lt2h_area_proxy_weighted | pct_zero_unweighted | pct_zero_area_proxy_weighted |
|---|---|---|---|---|---|---|
| Rio das Pedras | 275810 | 73.85 | 73.77 | 73.93 | 64.64 | 64.79 |
| Rocinha | 375367 | 80.33 | 80.38 | 80.49 | 72.45 | 72.62 |
| Complexo do Alemão | 471449 | 70.37 | 70.27 | 70.46 | 61.59 | 61.58 |
| Maré | 777416 | 69.13 | 69.12 | 69.07 | 62.11 | 61.88 |

## What this means for using the facade layer in P1

1. The footprint join is sound: building_id is the footprint layer OBJECTID (z_min = base and height = altura for >= 99.7% of buildings in every site, vs a one-step ID shift that fails). The spatial check against site polygons is clean for Rocinha and Alemao; Maré falls inside definition A only 43.9% (inside the E outline: 94.6%), so Mingze's Maré extent is not definition A.
2. The 06-21 winter slice (share < 2 h: 69.1% to 80.3%) is internally consistent; the exact-zero share on 06-21 (61.6% to 72.5%) is expected to include structurally unlit (poleward-facing) faces, so it is not itself an error rate.
3. The persistent-zero stratum (zero on all four dates, 9.8% to 28.7% by site; whole buildings with every band persistently zero: 1.8% to 10.8%, Rocinha highest) is the candidate geometry/join failure set. It is flat by floor and does not rise with building-level footprint attachment (Pearson r -0.03 to 0.04), so the building-level party-wall proxy does not explain it; a side-resolved test is not possible with this CSV.
4. v3 and July look like the same band set (Rio das Pedras and Alemao band counts identical), yet v3 reports 54.7% persistent exact-zero irradiation at Rio das Pedras against 11.2% persistent direct-sun zero here. Irradiation zero should be <= direct-sun zero, so one of the two runs has a failure mode; do not cite v3 zero-based medians (e.g. Rio das Pedras median 0.000) until resolved.
5. Area weighting cannot be done properly (no widths, no side-to-edge map). With the crude equal-side proxy the 06-21 share < 2 h moves by at most 0.16 percentage points, which is not evidence either way; report band-count shares and label them so.

## Items to request from Mingze

- Per band: facade centroid x, y, z and outward normal (or the two end-point coordinates of the facade segment) in a stated CRS, so sides map to geometry and a party-wall/edge test is possible. Aggregates stay internal; per-building values are withheld from any release.
- Per band: facade width (or area) to allow area-weighted shares.
- The footprint source used (layer, CRS, any simplification or subdivision) and how facade_id is numbered; the CSV side counts equal the IPP footprint vertex count for only 15% to 72% of buildings depending on site.
- Whether the context geometry (neighbouring buildings, terrain) used in the simulation included all footprints around each site, and the Maré extent actually modelled (definition A, E, or other).
- The rule that produced duplicate (building_id, floor_id, facade_key) rows (0.06% to 0.87% of rows by site-date; sun_hours differ within 33% to 74% of duplicate groups) and the Maré bands present on 06-21 but missing on 09-22 and 12-21 (band counts differ by date, see Q1).
- Per-band sun_hours and irradiation from the same run on the same band keys, so zero in irradiation can be checked against zero in direct sun band by band; plus a sample (aggregates reported only) of exact-zero bands inspected for hidden surfaces, flipped normals or result-join errors.
