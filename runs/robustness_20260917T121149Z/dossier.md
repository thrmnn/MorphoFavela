# Robustness dossier — P1 v2 (STAGED, not promoted)

Generated 2026-09-17T12:11:49Z · status: staged

Ledger used: `runs/wp07_ledger_20260917T045910Z/ledger.json` (437 entries, run 2026-09-17T04:59:10Z, status final)

L1 (favela-vs-formal) contrast check: CLEAR — no flagged ids

Reviewer robustness dossier for P1 v2 — assembly by code of every acceptance and sensitivity number already final in a run of record. Nothing recomputed; nothing new claimed. STAGED — no promotion to shared/figures, papers/, or anything the public hub serves.

## 1. Engine acceptance

Engine acceptance rests on two independent legs: (1) three analytic self-checks against closed-form geometry (no data dependency), and (2) a measured cross-reference against the CPU raycaster's street SVF at Rio das Pedras (16,905 reference points, of which 16,018 are matched by the nearest-sampling variant the cited statistics come from). Both are read here, not recomputed.

| id | value | unit | source |
|---|---|---|---|
| `engine.crossref.r` | 0.9945 | dimensionless | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/engine.crossref.r/value` |
| `engine.crossref.median_abs_delta` | 0.01372 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/engine.crossref.median_abs_delta/value` |
| `engine.crossref.p95_abs_delta` | 0.04738 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/engine.crossref.p95_abs_delta/value` |
| `engine.crossref.n_reference_points` | 16905 | points | `runs/wp02_horizon_20260914T195630Z/crossref_diagnostic.json#/n_points` |
| `engine.crossref.n` | 16018 | points | `runs/wp02_horizon_20260914T195630Z/crossref_diagnostic.json#/variants/A_nearest_sampling/n` |

Chosen variant: runs/wp02_horizon_20260914T195630Z/crossref_diagnostic.json#/variants/A_nearest_sampling — chosen because march_sampling='nearest' is the nearest-cell march named in docs/wp07_ledger_spec.md (r≈0.995), not the bilinear-march baseline in the same file (r≈0.988) nor the sibling run's single-site crossref (also bilinear-march, r≈0.988).

Analytic self-checks (closed-form, no data dependency):
- **unobstructed identity**: flat surface -> every patch visible -> sky.svf == 1 for every observer (`docs/wp02_horizon_engine_spec.md#Acceptance (item 1)`, verified by `tests/test_wp02_horizon.py::test_flat_surface_identity`)
- **infinite canyon, exact mask**: analytic closed form 'visible iff dz/|dy| > 2H/W' must equal the engine's mask patch-for-patch for H/W in {0.25, 0.5, 1, 2, 3}, allowing only patches within one step's angular quantum of the horizon (`docs/wp02_horizon_engine_spec.md#Acceptance (item 2)`, verified by `tests/test_wp02_horizon.py::test_infinite_canyon_exact_mask`)
- **isolated wall shadow**: one wall of height H at distance D: blocked patches are exactly those with alt < atan(H/D) inside the wall's azimuth span (`docs/wp02_horizon_engine_spec.md#Acceptance (item 3)`, verified by `tests/test_wp02_horizon.py::test_isolated_wall_shadow`)

## 2. Domain sensitivity (G3)

Across the 9 grid variants swept, each favela's SVF-percentile position moves — by 1.27 to 8.28 percentile points (min-max spread across the five favelas) — but the ORDERING of the five favelas does not: rank_invariant_across_grid = True (every one of the 9 variants yields the same descending order). Position moves with the grid; ordering does not.

| id | value | unit | source |
|---|---|---|---|
| `g3.spread.complexo_do_alemao.svf_percentile` | 8.283 | percentile points | `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/complexo_do_alemao/svf_percentile_spread_max_minus_min` |
| `g3.spread.mare.svf_percentile` | 3.569 | percentile points | `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/mare/svf_percentile_spread_max_minus_min` |
| `g3.spread.riodaspedras.svf_percentile` | 3.195 | percentile points | `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/riodaspedras/svf_percentile_spread_max_minus_min` |
| `g3.spread.rocinha.svf_percentile` | 1.275 | percentile points | `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/rocinha/svf_percentile_spread_max_minus_min` |
| `g3.spread.vidigal.svf_percentile` | 4.28 | percentile points | `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/vidigal/svf_percentile_spread_max_minus_min` |
| `g3.rank_under_locked_domain` | complexo_do_alemao, rocinha, vidigal, riodaspedras, mare | ordered list | `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/rank_under_locked_domain` |
| `g3.rank_invariant_across_grid` | True | boolean | `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/rank_invariant_across_grid` |
| `g3.spread.min_across_favelas` | 1.275 | percentile points | computed (min) from `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/vidigal/svf_percentile_spread_max_minus_min`; `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/rocinha/svf_percentile_spread_max_minus_min`; `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/complexo_do_alemao/svf_percentile_spread_max_minus_min`; `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/mare/svf_percentile_spread_max_minus_min`; `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/riodaspedras/svf_percentile_spread_max_minus_min` |
| `g3.spread.max_across_favelas` | 8.283 | percentile points | computed (max) from `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/vidigal/svf_percentile_spread_max_minus_min`; `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/rocinha/svf_percentile_spread_max_minus_min`; `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/complexo_do_alemao/svf_percentile_spread_max_minus_min`; `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/mare/svf_percentile_spread_max_minus_min`; `runs/wp07_ledger_20260917T045910Z/ledger.json#/derived/spread/riodaspedras/svf_percentile_spread_max_minus_min` |

## 3. Sky resolution

P1_SKY_PATCHES = 145 (imported from src/brisa_solar/constants.py, never typed as a literal in any P1 module — tests/test_p1_sky_resolution_consistency.py enforces this by AST). Scanning every run manifest under the main checkout (97 manifests, 97 carrying a sky.patches field): exactly one resolution appears — [145]. No second resolution exists in any code path feeding a pooled number.

## 4. Irradiance input

Two EPW stations feed P1: Galeão (primary, 1732 kWh/m2/yr) and Santos Dumont (1790 kWh/m2/yr). Santos Dumont reads +3.34% relative to Galeão — computed here from the two annual GHI values, never copied from epw_inventory.json's own _crosscheck field.

| id | value | unit | source |
|---|---|---|---|
| `epw.galeao.annual_ghi_kwh_m2` | 1732 | kWh/m2 | `data/epw/epw_inventory.json#/galeao/annual_ghi_kwh_m2` |
| `epw.santos_dumont.annual_ghi_kwh_m2` | 1790 | kWh/m2 | `data/epw/epw_inventory.json#/santos_dumont/annual_ghi_kwh_m2` |
| `epw.primary_station` | galeao | text | `data/epw/epw_inventory.json#/_meta/primary` |
| `epw.santos_vs_galeao_ghi_pct_diff` | 3.336 | percent | computed (pct_diff_b_over_a) from `data/epw/epw_inventory.json#/galeao/annual_ghi_kwh_m2`; `data/epw/epw_inventory.json#/santos_dumont/annual_ghi_kwh_m2` |

## 5. Second-axis validity (VENTAXIS)

Definition of record: docs/ventaxis_canonical.md (VENTAXIS). The second axis is a per-cell checklist count in {0,1,2,3} of three independent geometry predicates (vertical, lateral, directional) — never a weighted continuous index, never a measurement of air exchange itself. Shares below are the fraction of each site's cells at each count, from the wp06.* ledger entries.

| id | value | unit | source |
|---|---|---|---|
| `wp06.definition_of_record_quote` | For each built 10 m cell, `n_constraints` is a **checklist count** (an integer in {0, 1, 2, 3}) of how many of three independent geometry predicates the cell triggers. | text | `docs/ventaxis_canonical.md` (verbatim substring) |
| `wp06.vidigal.n` | 2756 | count | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.vidigal.n/value` |
| `wp06.vidigal.share_n0` | 0.2032 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.vidigal.share_n0/value` |
| `wp06.vidigal.share_n1` | 0.3988 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.vidigal.share_n1/value` |
| `wp06.vidigal.share_n2` | 0.3295 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.vidigal.share_n2/value` |
| `wp06.vidigal.share_n3` | 0.06858 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.vidigal.share_n3/value` |
| `wp06.rocinha.n` | 8031 | count | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.rocinha.n/value` |
| `wp06.rocinha.share_n0` | 0.105 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.rocinha.share_n0/value` |
| `wp06.rocinha.share_n1` | 0.2959 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.rocinha.share_n1/value` |
| `wp06.rocinha.share_n2` | 0.5301 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.rocinha.share_n2/value` |
| `wp06.rocinha.share_n3` | 0.06911 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.rocinha.share_n3/value` |
| `wp06.complexo_do_alemao.n` | 17768 | count | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.complexo_do_alemao.n/value` |
| `wp06.complexo_do_alemao.share_n0` | 0.1738 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.complexo_do_alemao.share_n0/value` |
| `wp06.complexo_do_alemao.share_n1` | 0.3791 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.complexo_do_alemao.share_n1/value` |
| `wp06.complexo_do_alemao.share_n2` | 0.3439 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.complexo_do_alemao.share_n2/value` |
| `wp06.complexo_do_alemao.share_n3` | 0.1033 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.complexo_do_alemao.share_n3/value` |
| `wp06.riodaspedras.n` | 6605 | count | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.riodaspedras.n/value` |
| `wp06.riodaspedras.share_n0` | 0.0321 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.riodaspedras.share_n0/value` |
| `wp06.riodaspedras.share_n1` | 0.1257 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.riodaspedras.share_n1/value` |
| `wp06.riodaspedras.share_n2` | 0.2883 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.riodaspedras.share_n2/value` |
| `wp06.riodaspedras.share_n3` | 0.554 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.riodaspedras.share_n3/value` |
| `wp06.mare.n` | 29229 | count | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.mare.n/value` |
| `wp06.mare.share_n0` | 0.09833 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.mare.share_n0/value` |
| `wp06.mare.share_n1` | 0.2433 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.mare.share_n1/value` |
| `wp06.mare.share_n2` | 0.3385 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.mare.share_n2/value` |
| `wp06.mare.share_n3` | 0.3199 | fraction | `runs/wp07_ledger_20260917T045910Z/ledger.json#/entries/wp06.mare.share_n3/value` |

## 6. Ground-truth comparison (G2)

What was compared: TLS-derived DSM vs the 2.5D model's ALS-DTM+footprint surface, in three vertical-clearance classes (<1.5 m, 1.5-3 m, >3 m), for shared-observer-cell SVF and for street/alley points specifically. This is a NEGATIVE result with a confirmed confound, reported as it is: the baseline comparison (variant a, no shared observer elevation) shows essentially no correlation for the <1.5 m class (r ~ 0.015); forcing a SHARED observer elevation (variants e/f) raises that to r >= 0.5 for at least one variant, confirming an observer-elevation confound. Agreement holds only for STREET points in alleys below the 1.5 m height threshold (floor class = '<1.5m', r = 0.504, median |delta SVF| = 0.0501). The open decision this bears on — the 2.5D model's validity floor — is card `g2_validity_floor`. The PI has not ruled; this dossier states the finding and names the open card, and does not resolve it.

| id | value | unit | source |
|---|---|---|---|
| `g2.registration.median_delta_m` | 1.561 | meters | `runs/wp03_tls_20260915T205422Z/registration.json#/median_delta_m` |
| `g2.registration.p95_abs_delta_m` | 9.62 | meters | `runs/wp03_tls_20260915T205422Z/registration.json#/p95_abs_delta_m` |
| `g2.registration.n` | 7384 | count | `runs/wp03_tls_20260915T205422Z/registration.json#/n` |
| `g2.street.floor_class` | <1.5m | text | `runs/wp03_tls_20260915T215720Z/g2_result_v3.json#/street/floor` |
| `g2.street.lt1p5m.r` | 0.5044 | dimensionless | `runs/wp03_tls_20260915T215720Z/g2_result_v3.json#/street/classes/0/r` |
| `g2.street.lt1p5m.median_abs_delta` | 0.05008 | fraction (SVF) | `runs/wp03_tls_20260915T215720Z/g2_result_v3.json#/street/classes/0/median_abs_delta` |
| `g2.street.lt1p5m.share_within_tol` | 0.6735 | fraction | `runs/wp03_tls_20260915T215720Z/g2_result_v3.json#/street/classes/0/share_within_tol` |
| `g2.street.lt1p5m.n` | 1213 | count | `runs/wp03_tls_20260915T215720Z/g2_result_v3.json#/street/classes/0/n` |
| `g2.confound.baseline_variant_a.lt1p5m.r` | 0.01508 | dimensionless | `runs/wp03_tls_20260915T215720Z/g2_result_v3.json#/variants/a_dtm_fill_merged/classes/0/r` |
| `g2.confound.shared_elevation_variant_e.lt1p5m.r` | 0.185 | dimensionless | `runs/wp03_tls_20260915T215720Z/g2_result_v3.json#/variants/e_shared_obs_z_als_dtm/classes/0/r` |
| `g2.confound.shared_elevation_variant_f.lt1p5m.r` | 0.528 | dimensionless | `runs/wp03_tls_20260915T215720Z/g2_result_v3.json#/variants/f_shared_obs_z_smrf_ground/classes/0/r` |
| `g2.verdict_quote` | at least one class in variant (e) or (f) reads r >= 0.5 with the shared observer elevation -- the confound was (at least partly) real; see the per-class table for which class(es) cleared the bar. | text | `runs/wp03_tls_20260915T215720Z/report_v3.md` (verbatim substring) |

Open card (not resolved here): **g2_validity_floor**

## 7. Coverage

Citywide: 8,402,056 cells at 1.0 m (WP-05 full run, exhaustive — every fabric cell, no sampling error). Per-site ground-cell counts below are WP-04's (1 m). The methods epoch table is generated from each input file's own filesystem metadata (last-modified timestamp, size), never typed — per the C' plan §1.6. This is a proxy for acquisition vintage, not a claim about it: no embedded raster datetime tag was found on the citywide DTM (confirmed via gdalinfo), consistent with the C' plan's own note that DTM_RJ.tif's true survey vintage is UNVERIFIED.

| id | value | unit | source |
|---|---|---|---|
| `coverage.citywide.n_cells` | 8402056 | count | `runs/wp05_full_20260914T215419Z/manifest.json#/n_cells_consolidated` |
| `coverage.citywide.cell_m` | 1 | meters | `runs/wp05_full_20260914T215419Z/manifest.json#/cell_m` |
| `coverage.site.vidigal.ground_n` | 141239 | count | `runs/wp04_sites_20260914T230606Z/vidigal/summary.json#/ground/n` |
| `coverage.site.rocinha.ground_n` | 310710 | count | `runs/wp04_sites_20260914T230606Z/rocinha/summary.json#/ground/n` |
| `coverage.site.complexo_do_alemao.ground_n` | 846016 | count | `runs/wp04_sites_20260914T230606Z/complexo_do_alemao/summary.json#/ground/n` |
| `coverage.site.riodaspedras.ground_n` | 167661 | count | `runs/wp04_sites_20260914T230606Z/riodaspedras/summary.json#/ground/n` |
| `coverage.site.mare.ground_n` | 258178 | count | `runs/wp04_sites_20260914T230606Z/maré/summary.json#/ground/n` |

Methods epoch table (generated from file metadata, never typed):

| scope | category | path | mtime (UTC) | size (bytes) |
|---|---|---|---|---|
| citywide | boundary | `data/RJ/Favelas_Limit_2019.shp` | 2026-02-11T14:59:50Z | 1,091,852 |
| citywide | footprints | `data/RJ/buildings_RJ_2019_utm.gpkg` | 2026-09-09T07:33:06Z | 1,197,740,032 |
| citywide | dtm | `data/RJ/DTM_RJ.tif` | 2026-02-19T14:39:00Z | 425,306,880 |
| vidigal | boundary | `data/vidigal/raw/Vidigal_Limit.shp` | 2026-03-05T18:40:08Z | 2,460 |
| vidigal | footprints | `data/vidigal/buildings_extended_300m.gpkg` | 2026-06-14T08:32:35Z | 2,539,520 |
| vidigal | dtm | `data/vidigal/dtm_extended_300m.tif` | 2026-04-09T19:49:32Z | 279,866 |
| rocinha | boundary | `data/rocinha/raw/rocinha_boundary.shp` | 2026-02-19T14:40:53Z | 9,056 |
| rocinha | footprints | `data/rocinha/buildings_extended_300m.gpkg` | 2026-06-14T08:32:27Z | 7,704,576 |
| rocinha | dtm | `data/rocinha/dtm_extended_300m.tif` | 2026-04-09T19:58:53Z | 624,042 |
| complexo_do_alemao | boundary | `data/complexo_do_alemao/raw/complexo_do_alemao_boundary.shp` | 2026-03-25T14:12:24Z | 25,756 |
| complexo_do_alemao | footprints | `data/complexo_do_alemao/buildings_extended_300m.gpkg` | 2026-04-09T19:53:56Z | 15,843,328 |
| complexo_do_alemao | dtm | `data/complexo_do_alemao/dtm_extended_300m.tif` | 2026-04-09T19:54:00Z | 1,365,701 |
| riodaspedras | boundary | `data/riodaspedras/raw/riodaspedras_boundary.shp` | 2026-03-25T14:13:43Z | 7,776 |
| riodaspedras | footprints | `data/riodaspedras/buildings_extended_300m.gpkg` | 2026-04-09T19:53:08Z | 5,664,768 |
| riodaspedras | dtm | `data/riodaspedras/dtm_extended_300m.tif` | 2026-04-09T19:53:09Z | 669,809 |
| mare | boundary | `data/maré/raw/mare_boundary.shp` | 2026-03-23T17:03:25Z | 16,172 |
| mare | footprints | `data/maré/buildings_extended_300m.gpkg` | 2026-06-14T08:32:00Z | 19,185,664 |
| mare | dtm | `data/maré/dtm_extended_300m.tif` | 2026-04-09T19:55:38Z | 2,404,915 |

## 8. Declared limitations

Vegetation/canopy: descoped 2026-09-10 (PI decision, quoted below); this dossier confirms mechanically that no DSM-named file exists under data/ in the main checkout (0 matches) and states plainly that this limitation may not be used to explain any shortfall. Façade: NOT accepted — cross-reference r = 0.884 (Rio das Pedras) / 0.644 (Vidigal), both below the floor (r >= 0.95, median |delta| <= 0.03); the ledger carries 0 façade-derived entries (must be zero — confirmed). The 2.5D model's validity floor is the open G2 question above (card g2_validity_floor, not resolved here).

| id | value | unit | source |
|---|---|---|---|
| `limitations.canopy.descope_quote` | PI dropped canopy/vegetation from P1 rather than hold the critical path on it. Solar and the ventilation index are computed on terrain + buildings only; vegetation shading is a stated limitation in the methods, and the discussion may not attribute any shortfall to trees. | text | `/home/theo/SCL/SCR/brisaverse/papers/p1-nature-cities/proposal/cprime_reframe_plan.md` (verbatim substring) |
| `limitations.facade.floor_r` | 0.95 | dimensionless | `runs/wp04f2_facade_2026-09-15T07:50:07Z/crossref.json#/floor/r` |
| `limitations.facade.floor_median_abs_delta` | 0.03 | fraction | `runs/wp04f2_facade_2026-09-15T07:50:07Z/crossref.json#/floor/median_abs_delta` |
| `limitations.facade.riodaspedras_r` | 0.8839 | dimensionless | `runs/wp04f2_facade_2026-09-15T07:50:07Z/crossref.json#/sites/riodaspedras/variants/unweighted/overall/r` |
| `limitations.facade.vidigal_r` | 0.6437 | dimensionless | `runs/wp04f2_facade_2026-09-15T07:50:07Z/crossref.json#/sites/vidigal/variants/unweighted/overall/r` |

## Sources read

- `/home/theo/SCL/SCR/brisaverse/papers/p1-nature-cities/proposal/cprime_reframe_plan.md`
- `data/epw/epw_inventory.json`
- `docs/ventaxis_canonical.md`
- `runs/wp02_horizon_20260914T195630Z/crossref_diagnostic.json`
- `runs/wp03_tls_20260915T205422Z/registration.json`
- `runs/wp03_tls_20260915T215720Z/g2_result_v3.json`
- `runs/wp03_tls_20260915T215720Z/report_v3.md`
- `runs/wp04_sites_20260914T230606Z/complexo_do_alemao/summary.json`
- `runs/wp04_sites_20260914T230606Z/maré/summary.json`
- `runs/wp04_sites_20260914T230606Z/riodaspedras/summary.json`
- `runs/wp04_sites_20260914T230606Z/rocinha/summary.json`
- `runs/wp04_sites_20260914T230606Z/vidigal/summary.json`
- `runs/wp04f2_facade_2026-09-15T07:50:07Z/crossref.json`
- `runs/wp05_full_20260914T215419Z/manifest.json`
- `runs/wp07_ledger_20260917T045910Z/ledger.json`
