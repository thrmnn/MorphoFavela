# Maré boundary impact audit — 2026-09-24

Read-only trace of every consumer of Maré's boundary/lattice, by code (grep +
direct reads of the generating scripts, run manifests, and shapefiles — not
inferred from docstrings). Companion to `docs/mare_brief_v2_spec.md` and the
open `/ops` card `mare_citywide_definition`.

## The three (really four) definitions in play

- **DATA EXTENT** = `data/maré/raw/mare_boundary.shp` (bairro polygon, 4.27 km²).
  Building footprints, DTM, the street shapefile, and the 10 m morphometrics
  grid (`outputs/maré/morphometrics/grid/grid_metrics.gpkg`) are all built
  over this extent.
- **STUDY AREA** = `src/brisa_solar/mare_study_area.py::load_study_area()`
  (union of 15 in-extent Redes da Maré communities ∩ DATA EXTENT, 2.04 km²).
  PI-approved 2026-09-23. Only 4 scripts import this module:
  `build_site_dashboard.py`, `build_html_dashboard.py`,
  `render_mare_territory_map.py`, `build_diagnostic_map.py`, plus the
  brief's `collect_numbers.py`/`render_figures.py`.
- **CITYWIDE** = 6 IPP `Favelas_Limit_2019.shp` polygons with
  `complexo == 'Maré'`, 0.84 km². Resolved by **two independent code paths**
  that implement the same rule: `wp05_full.py::match_favela_group` and
  `wp04_sites.py::site_polygon`/`match_favela_polygon`. This is the open PI
  card `mare_citywide_definition`.
- **Stale 4th number**: `docs/technical_report/technical_report.md`'s own
  Table 1 site area for Maré is **4.34 km²** — matches none of the three
  above (nearest is DATA EXTENT's 4.27, but not equal). Propagates into the
  brisaverse deck headline.

**Correction to my own initial framing**: WP-04's ground/street *observer
points* are not DATA-EXTENT-scoped, as `resolve_boundary("maré")` /
`SITE_BOUNDARY_FILE` provenance citations imply — `wp04_sites.py::run_site()`
line 556 masks `ground_grid_points()`'s observers with
`site_polygon(favelas_path, "Maré", crs)`, i.e. **CITYWIDE** (0.84 km²), not
the bairro. `resolve_boundary()`/`mare_boundary.shp` is only used to clip
*road geometries* before street-point sampling and for the dashboard's
edge-halo logic — not for which ground cells get evaluated. This is the
direct, code-confirmed root cause of the site sheet's "cov 40%" badge
(0.84/2.04 = 41%): WP-04 literally never ray-traced SVF/solar for ~60% of
the study area (the outer/southern conjuntos), because `site_polygon` only
matched 6 of the ~15 communities. Not a street-layer extent problem
(`street_mare.shp` bounds match the bairro almost exactly) and not a
plotting clip.

## Table — all traced artefacts

| Artefact | File/pointer | Current definition | Consistent? | Fix class | Cost | PI decision `mare_citywide_definition`? |
|---|---|---|---|---|---|---|
| Site sheet (folha) | `scripts/build_site_dashboard.py` | STUDY AREA | Y | — | — | N |
| Interactive dashboard | `scripts/build_html_dashboard.py` | STUDY AREA | Y | — | — | N |
| Maré brief v2 | `docs/briefs/mare/*` (commit 4860a02) | STUDY AREA | Y | — | — | N |
| Territory map | `scripts/render_mare_territory_map.py` | STUDY AREA (shows all 3 for context) | Y | — | — | N |
| **WP-07 ledger `favela.mare.*`** (svf/kwh median, IQR, percentile) | `wp07_ledger.py:_build_favelas` ← `wp05_full_20260914T215419Z/distribution.json` | CITYWIDE | **N** | RECOMPUTE | re-run WP-05's favela match+aggregation over 8.4M-cell parquet, laptop CPU minutes | **Y** |
| **WP-07 ledger `citywide.sun_h_*`/`favela.mare.sun_h_*`** | `wp07_ledger.py:_build_cityhours` ← `cityhours_full_20260917T041544Z/summary.json` | CITYWIDE | **N** | RECOMPUTE | same as above, cityhours run | **Y** |
| **WP-07 ledger `site.mare.*`** (ground/street svf, kwh, sun_h, `share_ge_*h`) | `wp07_ledger.py:_build_sites` ← `wp04_sites_20260914T230606Z/maré/summary.json` | **CITYWIDE** (corrected — `wp04_sites.py:556 site_polygon`, not DATA EXTENT) | **N** | RECOMPUTE | re-run WP-04 ground+street ray-trace for the ~1.2 km² never evaluated (≈2.4× the current 0.84 km² footprint); WP-04 already ran on GPU at this scale — laptop GPU minutes–low hours | **Y** |
| **WP-07 ledger `wp06.mare.n`/`share_n0-3`** | `wp07_ledger.py:_build_wp06` ← `wp06_geometry_20260915T052604Z/summary.json` | DATA EXTENT (pure morphometrics grid, `n=29229`, independent of WP-04's SVF mask) | **N** | FILTER | re-mask the 29,229-cell table against STUDY AREA, re-aggregate shares — seconds | N (this one's a DATA EXTENT→STUDY AREA gap, already PI-decided) |
| **G3 domain sensitivity `g3.*.mare.*`, `g3.spread.mare`** | `src/brisa_solar/g3_domain.py:201` reuses `wp05_full.match_favela_group` | CITYWIDE | **N** | RECOMPUTE | re-run the 9-variant grid's favela match/consolidation | **Y** |
| **Terrain split `t2_map_mare.png/.svg`** | `src/brisa_solar/terrain_split.py` ← WP-04's `ground.parquet` (row count 258,178 matches WP-04's `n_ground` exactly) | **CITYWIDE** (corrected — reads WP-04's already-masked ground points) | **N** | FILTER once WP-04 is fixed (re-plot from the extended parquet); RECOMPUTE is really WP-04's, not terrain_split's own | inherits WP-04's cost | **Y** |
| **Zoom window `mare` / P1 figure `f6_zoom_mare_svf/kwh`** | `config/zoom_windows.yaml:31-34 source: favela_boundary:Maré` → `wp07_figures.py:945-966 resolve_window_boundary` → `match_favela_group` | CITYWIDE | **N** | FILTER | swap matcher/boundary source, re-crop+redraw from the existing citywide parquet — CPU minutes | **Y** |
| **P1 figure `f1_citywide_position`** | `wp07_figures_spec.md` ← ledger `favela.mare.*` | CITYWIDE | **N** | RECOMPUTE | inherits the `favela.mare.*` ledger fix | **Y** |
| **P1 figure `f2_direct_sun_reference_days`** | ledger `site.mare.*` ← WP-04 `ground.parquet` | CITYWIDE | **N** | RECOMPUTE | inherits WP-04 fix (not a simple re-mask — the underlying points don't exist yet for the missing area) | **Y** |
| **P1 figure `f3_domain_sensitivity`** | ledger `g3.*` | CITYWIDE | **N** | RECOMPUTE | inherits G3 fix | **Y** |
| **P1 figure `f4_geometry_constraints`** | ledger `wp06.*` | DATA EXTENT | **N** | FILTER | inherits `wp06.mare.*` fix | N |
| **CFD patches MAR-P01..P25** (18 named in prompt; 25 actually exist) | `outputs/maré/sampling_cfd/campaign_sampling/` ← `grid_metrics.gpkg` | DATA EXTENT | **N** | 7/25 patches (P07,P10,P11,P13,P21,P22,P23) fall outside STUDY AREA — dropping them = FILTER (seconds); backfilling to target stratum counts = RECOMPUTE (sampling re-run, minutes); **CFD itself is parked** | N |
| **Roughness patches** (`outputs/cross_site/roughness/patch_roughness.csv`, 25 Maré rows) | 1:1 derivative of the CFD patches above | DATA EXTENT | **N** | same as CFD patches (FILTER to drop 7 rows; RECOMPUTE only if patches backfilled) | N |
| **Cross-site `composition_by_site.csv`** | `scripts/build_morphotype_diagrams.py::per_favela_maps()` ← `features_grid.parquet` (DATA EXTENT grid) | DATA EXTENT | **N** | FILTER (mask before `value_counts`) — but only valid once #below REFIT lands | N |
| **Morphotope / morphotope_stability** (cell-type + tissue GMMs) | `src/morphometry/signature.py`, `scripts/build_morphotope.py`, `audit_morphotope_stability.py` — pool all 5 sites' DATA-EXTENT cells as **training** rows | DATA EXTENT | **N** | **REFIT** — both pooled GMM fits (z-scoring shifts other sites' labels too) plus the LOSO/bootstrap stability audit | N |
| **Typology predictor / extra** | `scripts/analyze_typology_predictor.py::loso()` — Maré's DATA-EXTENT cells in every non-Maré LOSO fold + the pooled OLS variance decomposition | DATA EXTENT | **N** | **REFIT** | N |
| **Cross-site risk map** | `scripts/run_cross_site_riskmap.py` — pooled `[p1,p99]` envelope thresholds and the "blind" `clf_all` model include Maré's DATA-EXTENT cells | DATA EXTENT | **N** | **REFIT** | N |
| **Technical report Maré row(s)** | `docs/technical_report/technical_report.md` (v1.3, 2026-07-01) — independent WP-01/WP-03 pipeline, no ledger/study-area awareness at all | DATA EXTENT (own pipeline, ~50 mentions) + a stale **4.34 km²** area figure matching no canonical definition | **N** | RECOMPUTE base per-site tables against STUDY AREA; **REFIT** any pooled cross-site model it cites (§10 roughness, §7 MAUP curve, §5.4 façade cross-check) | N (independent axis; needs its own reconciliation, not gated on the citywide card) |
| **Robustness dossier** | `scripts/build_robustness_dossier.py` — pure pass-through of the ledger + `SITE_BOUNDARY_FILE["mare"]="raw/mare_boundary.shp"` citation | inherits whatever it cites; **provenance citation itself is misleading** (cites DATA EXTENT filename for numbers that are actually CITYWIDE) | **N** | NONE for the dossier script itself (re-run after upstream fixes, seconds); fix the boundary-file citation string separately | mixed (inherits) |
| **brisaverse deck `slides/brisa_mare_pk.pptx`** | `slides/gen_mare_pk_spec.py` ← `mare_numbers.json` ← `docs/technical_report` (area) + `figure_manifest.json` (STUDY AREA grid maps, but CITYWIDE-scoped street SVF map inside a STUDY-AREA-shaped mask) | mixed: area = stale 4.34; grid maps = STUDY AREA; street map = CITYWIDE data, overclaimed as "whole settlement" | **N** | FILTER — re-pull `mare_numbers.json` and regenerate once MorphoFavela's numbers are fixed | Y (for the street-figure part) |
| **`brisaverse/shared/facts/solar_cprime_canonical.json` `wp06.mare.n`=29229** | cites `wp06_geometry_20260915T052604Z` (pre-fix run) | DATA EXTENT (pre-STUDY-AREA rerun) | **N** | FILTER (refresh pointer + value after WP-06 rerun) | N |
| **`brisaverse/hub` staged figures** `f6_zoom_mare_*`, `t2_map_mare` | `p1_artifacts.json` ← `wp07_zoom_20260917T124118Z`, `terrain_split_full_20260917T130045Z` | CITYWIDE (both, per above) | **N** | FILTER (re-stage after MorphoFavela regen) | Y |

### Consistent / NONE (not itemised further)
Site sheet, interactive dashboard, brief v2, territory map — all STUDY AREA,
switched 2026-09-24, confirmed by direct import of `mare_study_area.py`.

## Recommended recompute order

1. **PI decision first**: resolve `mare_citywide_definition` (open `/ops`
   card). It gates the single largest cluster — `favela.mare.*`,
   `site.mare.*`, G3, cityhours, zoom-window/f6, and downstream P1 figures
   f1–f3 — because WP-04's own ground/street observer mask (not just WP-05's
   citywide comparison) turns out to be CITYWIDE-scoped. If the decision is
   "Maré's citywide comparator = STUDY AREA," WP-04 must be **re-run** (not
   just re-filtered) to cover the ~1.2 km² of the study area it never
   ray-traced (root cause of "cov 40%").
2. **RECOMPUTE WP-04 Maré** (ground+street, GPU minutes–hours) once (1) is
   decided — this single re-run fixes `site.mare.*`, f2, terrain_split, and
   unblocks the WP-06/f4 FILTER step's denominator being meaningful.
3. **FILTER WP-06** against the resulting mask → `wp06.mare.*`, f4.
4. **RECOMPUTE G3 and WP-05 favela aggregation** (or FILTER, if (1) resolves
   to STUDY AREA and the underlying citywide parquet already has full
   coverage — check before assuming a full re-tile) → `favela.mare.*`,
   `g3.*`, cityhours, f1, f3, zoom window f6.
5. **REFIT** the three cross-site pooled models (morphotope/morphotope
   stability, typology predictor, risk map) once Maré's corrected cell
   population is available — do this once, after (2)–(4), not per-model.
6. **FILTER** `composition_by_site.csv`, CFD/roughness patch tables (drop
   the 7 out-of-study-area patches; CFD stays parked otherwise), robustness
   dossier (re-run, cites already-fixed ledger).
7. **Fix orphaned numbers independent of the PI decision**: technical
   report's stale 4.34 km² area and its own WP-01/WP-03 pipeline
   (RECOMPUTE against STUDY AREA on its own schedule — it doesn't even know
   the ledger/study-area module exists); refresh `brisaverse/shared/facts/*`
   pointers and re-stage hub figures/deck (all FILTER, last, since they only
   re-pull already-fixed upstream numbers). brisaverse's own tracker already
   has this queued as cards **MAREBOUND** (done) → **MAREMORPH** (open,
   blocked) — this audit corroborates and sharpens that card's scope.
