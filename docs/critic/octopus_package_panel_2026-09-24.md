# Octopus morphology package — panel ruling on v0.1 (OM2), 2026-09-24

Chair's synthesis of six reviews: hands-on data QA, urban climatologist, geospatial data engineer, LiDAR/remote-sensing specialist, spatial statistician, and data-governance lead.
Package: `outputs/_packages/mare_om2/v0.1/`. Code: `src/om_package/`, `scripts/build_om_package.py`, `scripts/aggregate_om_points.py`.
The chair re-checked these facts against the files on 2026-09-24: 1559 OM2 points; 124 points have `plan_density_lambda_p == 1.0`; `plan_density_lambda_p` has only 168 distinct values; `manifest.json` has only 3 keys; README Use terms and How to cite both read PLACEHOLDER; OM1/, OM3/ and OM4/ are in the package directory; `features_grid.parquet` already has `lambda_f_N … lambda_f_NW` and `zone_id`. The other numbers come from the seats' own runs.

## 1. Verdict

The code behind v0.1 is sound. CRS, IDs, the join-distance caps, the dictionary completeness tests and the 1 m spacing all check out. But the package is not ready to leave the project today. About 7% of OM2 points (106/1559) lie inside house footprints, and nothing flags them. OM1/3/4 ship in a folder that is supposed to hold OM2 only. The use terms are still PLACEHOLDER.
After the must-fix list below, which needs no new data, it can go to Jingxue and Vincent as a clearly labelled **internal review draft**. It cannot yet support RQ2's temporal part, because P-05 is empty, or RQ3's between-route test, because it has one route.

## 2. Must-fix before sharing (ranked)

1. **Flag the route-through-building points.** `scripts/build_om_package.py` (per-route build) + `src/om_package/routes.py::densify_route`: add a `route_geometry_flag` column. Set it True where the point is `within` a `buildings_mare` footprint OR more than 10 m from the nearest `street_mare` centreline. Count the flagged points in `quality.py::write_quality_report` and add a row to `dictionary.py`. Rewrite the lambda_p note in README: about 31% of lambda_p = 1.0 points (38/124) are this defect. The other 69% are real fully-built 10 m cells.
2. **Make the release OM2-only.** `build_om_package.py` defaults to `ALL_ROUTES` and writes all four routes under `mare_om2/`. Build OM2 only into the shared path and move OM1/3/4 to an internal build directory. `package_docs.py::render_readme`: add a "Release scope: OM2 only" heading. State that v0.1 supports within-route spatial/temporal CV only, and that RQ3's between-route holdout needs a later release.
3. **Close the use-terms gate.** `package_docs.py::render_readme`: until Q6/Q7 are answered, put a banner at the top of README: "INTERNAL REVIEW DRAFT — Octopus LRP #2 team only, not for redistribution or citation". Never ship with PLACEHOLDER alone.
4. **Freeze the schema before anyone joins against it.**
   - `ventilation.py::compute_ventilation_proxies` (lines 68–73): also join the 8 existing `lambda_f_<dir>` columns. Rename the mean to an *omnidirectional* obstruction density and drop "windward" from its definition.
   - Add `grid_cell_id` (= `features_grid.zone_id` from the same join) so models can cluster the 10 m-grid variables.
   - `shade.py::SHADE_TABLE_COLUMNS`: reserve a nullable `tree_shade`.
   - `dictionary.py`: rows for all new columns, plus the 4 segment columns (`segment_id`, `segment_start_m`, `segment_end_m`, `n_points`).
5. **Stop asserting an unresolved timezone.** `shade.py` hardcodes `CAMPAIGN_TZ = "America/Sao_Paulo"`, but the firmware writes GPS-fix rows in UTC. Mark it UNRESOLVED (see Q1). `OCTOPUS_JOIN_EXAMPLE` must drop rows where Latitude == Longitude == 0.0 (the no-fix sentinel in `octopus_outdoor.ino`) before any spatial join.
6. **Make the known limits honest.** `package_docs.py` + `dictionary.py`:
   - SVF/shade scene = buildings + bare-earth terrain only, **no vegetation**, so SVF is an upper bound under canopy.
   - `point_id` is provisional: minted from the OSM-inferred route, so the string stays stable but the place may move when om_routes.gpkg lands.
   - Add a "Coverage vs Table 1" paragraph plus a PENDING surface-cover row.
   - Distinguish "no building in buffer" NaNs (`building_height_mean_buffer_5m`) from beyond-cap NaNs.
   - Resolve the `height_change_2024_2026` name/definition mismatch after Q3.
7. **Fix manifest and format hygiene.**
   - `build_om_package.py` lines 109–126: make paths relative and drop `root`. Add `package_version`, `crs: EPSG:31983`, `use_terms` and a sha256 per file.
   - `io_utils.py::write_table`: write the Parquet from the GeoDataFrame so it carries GeoParquet `geo` metadata (keep x/y columns for plain-pandas users).
8. **Contact sheet blank panel** (polish, trivial). In `contact_sheet.py::build_contact_sheet`, `ax_map.set_aspect("equal")` on a 10-inch-wide axis squeezes the map into the centre and leaves white space. Use `adjustable="datalim"` or a wider-than-tall map panel.

## 3. Improvements for v0.2 (ranked)

1. **P-05 building shade** once dates and timezone are known (Q1). Buildings only, with `tree_shade` explicitly unknown. It is the package's only time-varying variable.
2. **Route repair** against om_routes.gpkg (Q2), plus a published `point_id_v0.1 → v0.2` crosswalk.
3. **Runnable `scripts/join_octopus_to_points.py`:** drop 0/0 rows → reproject WGS84 to EPSG:31983 → `join_utils.nearest_join` with a cap → `point_id` + `join_dist_m`; then time `merge_asof` against P-05. Link it from README Methods.
4. **Vegetation/surface cover** from the Q3 source: canopy fraction per 5/10/20/50 m buffer, and canopy added to the SVF/shade mesh.
5. **P-07 morphology diagnostics (no temperature):**
   - the along-route autocorrelogram (SVF r ≈ 0.80 at 10 m, ≈ 0 at 70–90 m), with a CV exclusion buffer of at least the largest buffer radius (Roberts et al. 2017);
   - a correlation/VIF table across form, buffer and proxy columns, with a primary-radius recommendation;
   - the `wind_alignment` ↔ `orientation` collinearity (r = 0.79).
6. **Airborne vs terrestrial comparison** (P-07), once the 2026 cloud exists, and terrestrial SVF.
7. **Building-height provenance:** cite the real source of the cadastral `altura` field (probably IPP Cadastro Tridimensional; to be confirmed), or soften the claim to "2019-vintage cadastral layer, method not confirmed".
8. **Documentation sentences:**
   - ventilation proxies use isotropic buffers, so they say nothing about upwind fetch;
   - at low wind the canyon flow decouples from the wind recorded at Galeão;
   - sensor lag means a morphology-to-temperature join should not assume the two are simultaneous;
   - adjacent 1 m points are not independent;
   - source layers cover all of Maré (39,322 buildings, 43,419 grid cells, 84,147 SVF samples), so RQ3 can predict elsewhere;
   - CHANGELOG: change "will ever carry" to "is designed to carry".
9. **OM1/OM3/OM4 release** when the PI decides. This enables RQ3's between-route holdout.

## 4. Where the seats disagree, and the ruling

- **What comes first: geometry, campaign dates, or metadata?** QA ranks the route defect first, the climatologist ranks campaign dates first, and the engineer ranks metadata first. *Ruling:* these do not compete. The route flag, scope and schema are local, same-day work and gate sharing. Campaign dates are an external ask with the longest lead time, so they are interview Q1 and should be sent today. Metadata goes last within the must-fix list.
- **Is the package fit to hand over?** The LiDAR seat says fit for internal release; governance says not fit. *Ruling:* after the must-fix items it is fit as an internal review draft for the named Octopus team. It is not yet citable or redistributable.
- **Surface cover: derive it now, mark it PENDING, or declare it out of scope?** (climatologist / governance / LiDAR). *Ruling:* v0.1 declares it absent with a PENDING row. Whether MorphoFavela owns it at all is Q4. If yes, the source is Q3.
- **Frontal-area proxy: relabel or ship per direction?** *Ruling:* ship per direction. The 8 columns already exist in `features_grid`, so the cost is the same as relabelling.
- **Timezone: ask the team or infer it from solar-noon or diurnal consistency?** *Ruling:* ask. Inference may only corroborate the answer. The hardcoded `America/Sao_Paulo` is itself a silent assumption that must be removed now.
- **point_id: hold the IDs, mark them provisional, or promise a crosswalk?** (engineer / governance). *Ruling:* ship them as provisional and promise the crosswalk. Governance adds a point the ruling keeps: the ID string is stable, the location may not be.
- **Modelling aids (grid_cell_id, autocorrelation, VIF) vs "never temperature analysis".** *Ruling:* all three describe the morphology variables alone and need no temperature, so they are in remit. `grid_cell_id` is must-fix because it changes the schema. The diagnostics wait for v0.2. The PI can veto if this reads as analysis creep.
- **lambda_p = 1.0.** The PI's join hypothesis is only partly right. *Ruling:* no code fix beyond the route flag. The remaining 1.0 values are plausible in dense cells, and `grid_cell_id` covers the pseudo-replication.
- **Manifest depth: full Frictionless or a minimal patch?** *Ruling:* minimal patch, as in must-fix 7. Upgrade later if the team asks.

## 5. PI interview

Ordered by how much work each answer unblocks. **[FORWARD]** = only the Octopus team (Vincent/Jingxue/Simone) can answer, so the PI forwards it. **[PI]** = the PI decides.

**Q1. [FORWARD → Vincent/Jingxue] OM2 campaign dates, walk times, timestamp timezone per row, and firmware variant. Can they send them?**
Ask for: each walk's date and start/end time; whether GPS-fix rows are UTC and RTC-fallback rows local; which firmware build was flashed (does no-fix write 0.0/0.0?).
- a) Forward all three as one message now *(recommended)*. → On reply: set the timezone rule in `shade.py`, run buildings-only P-05, and write `join_octopus_to_points.py` with the 0/0 filter.
- b) The PI already knows from deployment notes. → PI writes it down; we proceed as in (a) today.
- c) Send us the raw OM2 CSVs and we infer the dates, keeping the timezone ask open. → Dates from the data; P-05 stays blocked on timezone confirmation.
- d) Defer. → v0.1 ships static-only, and README says RQ2's temporal part is unsupported.

**Q2. [PI, file held by the team] Route geometry: will you drop om_routes.gpkg into `data/maré/raw/`, or do we ship v0.1 on the inferred route with the defect flagged?**
- a) Flag now (must-fix 1), and the PI downloads om_routes.gpkg manually for v0.2 *(recommended)*. → Ship the flag this week; in v0.2 rebuild on the real route and publish the ID crosswalk.
- b) Ask Vincent for a lighter OM2-only export of the route file. → Same as (a), with the team doing the fetch.
- c) Hold v0.1 until the real route is in. → Re-mint IDs first; nothing is shared until then.
- d) Repair the inferred route ourselves by snapping it to `street_mare` centrelines. → Faster than waiting, but the ID change is still ours to own and the result stays unvalidated by the team.

**Q3. [FORWARD → Simone/Jingxue/Vincent] Where are the 2024 airborne LiDAR/DSM (with vegetation) and the 2026 terrestrial OM2 point cloud, or does either not exist?**
- a) Both exist; send access. → Height change, terrestrial SVF, tree shade, canopy in the SVF mesh, and the P-07 comparison all become buildable.
- b) There is no 2024 flight. → Rename to `height_change_2019_2026` and state that the 7-year gap mixes real construction change with survey error. Use an open canopy-height product for vegetation *(recommended fallback)*.
- c) The terrestrial cloud exists, the airborne data does not. → Terrestrial SVF plus comparison; open canopy product for tree shade.
- d) Unknown or later. → Everything stays PENDING; README says so.

**Q4. [PI, confirm with Jingxue] Which of Table 1's groups does this package own: surface structure only, or also surface cover (vegetation/impervious) and/or façade materials?**
- a) Structure only; cover and materials belong to others. → A README coverage paragraph and no further work.
- b) Structure + surface cover *(recommended: it reuses our buffers and the Q3 source)*. → A canopy/impervious fraction per buffer in v0.2.
- c) Structure + cover + façade materials from the terrestrial RGB. → Needs the Q3 terrestrial cloud and a separate work package; scope and credit to be re-agreed.

**Q5. [PI] Once dates are known, do we release a buildings-only P-05 before any canopy source exists?**
- a) Yes, with `tree_shade` explicitly unknown *(recommended)*. → Run `compute_shade()` as soon as Q1 is answered.
- b) Yes, and pull an open canopy product in parallel. → Same as (a), then a v0.2.1 with coarse tree shade.
- c) No, wait for real canopy data. → P-05 is blocked on Q3.

**Q6. [FORWARD → Simone, PI proposes] Use terms and distribution. Who may receive point-level data that resolves to individual homes, and is there an IRB or community protocol covering it?**
- a) Named Octopus team members only, no redistribution, revisit at submission *(recommended)*. → Write this into README and the manifest; must-fix 3 closes.
- b) Wider release only as segment-aggregated or generalised geometry (P-03). → Add an export mode that drops point coordinates.
- c) An open licence (e.g. CC-BY) now. → Only if Simone confirms protocol coverage; then add the licence file and move toward a DOI.
- d) Fold it into the paper's own data-availability statement. → No package-level terms; README points to the paper.

**Q7. [PI, confirm with Jingxue] Credit and How to cite.**
- a) Acknowledgment now; raise authorship when the contribution list is drafted *(recommended)*. → Fill How to cite with an acknowledgment line.
- b) Middle authorship on the paper, citation deferred to the paper. → README says "cite the paper"; the PI raises authorship with Jingxue now.
- c) A standalone data DOI (Zenodo). → Only compatible with Q6 (c); we prepare a Zenodo record.

**Q8. [PI] OM1/OM3/OM4: when, and how are they held until then?**
- a) Move them to an internal build directory now; release them when RQ3 needs a between-route holdout *(recommended)*. → Must-fix 2; README states that RQ3 is within-route only for now.
- b) Keep them in the package but marked unreleased in README and the manifest. → Cheaper, but the folder still leaks them if it is zipped.
- c) Release all four in v0.2 together with the route repair. → Q2 and Q6 must cover four routes and four communities.
