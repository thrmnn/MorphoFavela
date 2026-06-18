# Morpho-Signature Track — Plan & Literature Brief

*Draft 2026-06-18. New track after the BRISA consolidation (merge `3757caa`).
For review — nothing here is built yet.*

## Thesis

Three things we want are actually one pipeline:

1. **Favela morphological signature** — cluster cells into recurring morphotypes
   and ask what numerically *is* a favela.
2. **Diagnostic ↔ morphometrics wiring** — a clean producer→consumer path from raw
   indicators to prioritization maps.
3. **Minimum replication framework** — infer priority areas from morphology **alone**
   where we have no/few CFD runs, calibrated by the handful of anchors we do have.

All three are gated on one missing artifact: **a feature substrate** that joins the four
un-joined files (`grid_metrics.gpkg`, `svf_streets_solar.gpkg`, `canyon/hw_streets.gpkg`,
`ventilation_openness_streets.gpkg`) *correctly* — i.e. respecting that street and grid
data sit on different spatial supports (see next section). Build that substrate once and it
feeds clustering (WS-A), the index (WS-B), and every new plot (WS-C). So **WS-0 is the
keystone** — but it is two linked tables, not one naive merge.

A second framing point from the repo map: the diagnostic stack is *already* almost
pure morphometrics. The 4-state taxonomy runs on ray-cast solar hours + λf — no CFD.
CFD only enters through `annual_cfd_U_mean`/`ACH`, which are still synthetic. So the
gap to "morphometrics-only prioritization" is **not missing physics** — it is (i) no
unified table, (ii) ventilation is a single hard `λf > 0.35` cut rather than a
calibrated index, (iii) no signature/typology layer exists at all.

## Multi-level analysis — the street/grid support problem (design decision)

Grid cells and street observers are **different spatial supports answering different
questions**, and collapsing one into the other by naive averaging is a change-of-support /
MAUP error. Grid cells (10 m areal tessellation) describe the *built fabric* — density,
massing, terrain. Street observers (1.5 m points along centerlines → segments) describe the
*experienced void* — the sky, sun and canyon a pedestrian actually meets. Point-count
averaging street values into cells is biased three ways: (i) the per-cell sample count is
itself a function of street density, so an unobserved confound weights the mean; (ii) the
mean discards the distribution, yet for prioritization the worst-exposed **tail** matters
more than the centre; (iii) cells with no street get a fabricated value from nearest-k
interpolation. The current grid `svf` column is exactly this naive aggregate
(`_aggregate_svf_to_grid`) and should be treated as suspect.

**Decision — two first-class tables, explicitly linked, not merged:**

- **Fabric table (areal, per cell):** only cell-native morphometrics (λp, λf, FAR, σH,
  H_mean, building_count, porosity, slope, aspect, street_orientation_entropy). Do *not*
  fold SVF/solar/H-W means in here — they are outcomes, not fabric inputs, and they
  double-count (porosity ≈ f(λp), SVF ≈ f(λp, H/W)).
- **Experience table (network, per observer/segment):** SVF, solar_hours_*, irradiance,
  H/W, openness — the primary unit for *exposure* (where people are and what the
  ray-caster/CFD physically evaluate). Enrich each observation with the fabric covariates
  of its containing cell — areal→point is a clean point-in-polygon lookup (cell value is
  constant over its area; no averaging). This is the **legitimate** direction of transfer;
  Favelas 4D works at exactly this street-section altitude.
- **Linkage, not collapse:** when street conditions *must* appear per cell (choropleths,
  prioritization rasters), use **support-aware operators** — length-weighted-along-segment
  median, worst-decile (p10), fraction-below-threshold (sun < 2 h, SVF < deep-canyon cut) —
  and carry `street_support_m`, `n_segments`, `has_street_support`. Empty cells stay NULL in
  the modeling table; nearest-k remains a *visual* convenience only.
- **Signatures across levels:** cluster fabric and experience *separately*, then study the
  contingency — which experienced-condition types co-occur with which fabric types.
  "Fabric type X reliably produces deep-canyon, sun-starved voids" is itself a signature,
  and more defensible than a blended vector pretending the two supports are commensurable.
  Optionally formalize as a multilevel model (observations nested in cells, cell random
  effects) — the statistically correct treatment of the nesting.
- **Prioritization is decided at the void level, mapped at the cell level** — rank exposure
  where it is experienced (street/segment), then aggregate to cells by worst-case operators
  for the planning view, not the other way round.

## WS-0 · Feature substrate (keystone) — two linked tables

- Per site emit **`features_grid.parquet`** (fabric, cell-native only) and
  **`features_street.parquet`** (experience, per observation, each row carrying its
  `zone_id` + the fabric covariates of its cell via point-in-cell join). Plus a cell-level
  **support-aware street summary** merged onto the grid table (length-weighted median, p10,
  fraction-below-threshold, `street_support_m`, `n_segments`, `has_street_support`).
- Align the three street sources first (`svf_streets_solar`, `canyon/hw_streets`,
  `ventilation_openness_streets`) — confirm whether they share an observer set or need a
  spatial join; the experience table is their union keyed on observer geometry.
- `feature_dictionary.md` (name, units, **support-level**, source module, transform) +
  `run_meta.json` (git sha), mirroring `svf_v2/io.py`.
- Derived covariates: `northness=cos(aspect)`, `eastness=sin(aspect)`, flat-vs-terrain SVF
  delta (Risk 1), built-mask.
- 8 sites: 5 campaign + 3 calibration (borel, jacarezinho, morro_do_juramento).
  Pooled ≈ 82k campaign cells.

## WS-A · Favela morphometric signature

**Method (lean, deliberately *not* the full momepy taxonomy):** cluster the
standardized environmental+morphometric vector directly. Skip morphological
tessellation/plot-shape characters — informal fabric violates the one-cell-per-building
assumption (Fleischmann et al. 2020, *CEUS* 101441), and our grid already sidesteps the
plot problem. Footprint-shape characters (compactness, corner irregularity) are hostage
to segmentation quality → treat as second-class, report sensitivity.

- Pipeline: standardize → GMM, pick k by BIC elbow → Ward dendrogram of cluster
  centroids → choropleth morphotype maps. Mirror the canonical taxonomy line
  (Fleischmann, Feliciotti, Romice & Porta 2022, *EPB* 49(4)) but at our altitude.
- **Validation = cross-site recurrence** (our edge over single-city visual inspection):
  does a morphotype recur in ≥3 of 5 sites, or is it site-specific? Bootstrap cluster
  stability (esda A-DBSCAN / ClustGeo resampling). Recurrence *is* the operational
  "signature" definition.
- Weight the signature toward what transfers cleanly to informal fabric — network
  configuration, settlement-scale texture/contrast, the **formal–informal seam**
  (Taubenböck & Kraff 2014) and entrance thresholds (Ena/Beirão 2019; Netto et al.) —
  over interior plot statistics. Compute λf/SVF/porosity gradients *across the boundary*
  as primary features.
- **Cluster the two supports separately** (fabric on the grid table, experience on the
  street table) and treat the **fabric×experience contingency** as a primary signature
  output — not one blended vector (see Multi-level section).
- Anchor against the one directly-relevant quantitative favela study: **Salazar Miranda
  et al. 2022, "Favelas 4D"** (*EPB* 49(9), Rocinha) — it abandons footprint-shape for
  street-section metrics for exactly our reasons. Same altitude, validates our choice.

## WS-B · Morphometrics-only diagnostic framework

**Wiring target:** `features.parquet` → indices → prioritization map, one documented
path, provenance-stamped. Replaces today's scatter of ad-hoc joins.

- **Ventilation index (replace the hard λf cut):** map λf 8-dir bands to Oke/Grimmond
  flow regimes (isolated ≲0.15 / wake-interference 0.15–0.35 / skimming ≳0.35;
  Grimmond & Oke 1999), or Macdonald et al. 1998 closed-form z0(λf)/zd(λp). Extract
  ventilation corridors by least-cost path through the λf surface (Wong & Nichol 2010
  — the most portable GIS recipe). Cross-check with pedestrian-level porosity (FAD aloft
  and porosity below decouple).
- **Solar/heat layer:** already pure-geometry (ray-cast). Add SVF→nocturnal-cooling
  (Oke 1981) and a heat-vulnerability index in the exposure–sensitivity–adaptive-capacity
  mould (Inostroza et al. 2016).
- **CFD-anchored calibration (the methodological contribution):** fit a surrogate of the
  CFD target (velocity ratio / UTCI) on the six predictors we already have
  (λp, λf, SVF, solar, porosity, H/W) using our *own* patch campaign as anchors;
  XGBoost + SHAP to prune to load-bearing predictors; apply the calibrated index to all
  unsimulated cells. Ship a **prioritization ranking**, not absolute values.
  *Defensible precisely because we recalibrate on local favela patches* rather than
  importing a surrogate trained on flat formal fabric (Javanroodi et al. 2023 is the
  closest reduced-order template; the 2025 JBPS XGBoost+SHAP paper is the feature-ranking
  template).
- **Minimum replication kit:** the path that runs on a new favela with morphometrics
  only — `features.parquet` → frozen index weights → prioritization map. The 3 calibration
  sites are the first replication test (they have morphometrics, no CFD). The existing
  `scripts/brisa_ventilation/07_onboard_new_favela.py` is the onboarding hook.

## WS-C · New visualizations

- **Morphotype fingerprint** — radar/`Scatterpolar` of standardized characters per type.
- **Taxonomy dendrogram** + **clustergram** (Fleischmann 2023) for k diagnostics.
- **Choropleth morphotype maps**, small-multiple per-character maps, one shared palette
  across dendrogram = map = embedding = radar (colour = type identity).
- **UMAP/t-SNE embedding** of cells coloured by morphotype; datashader for ~82k cells
  without overplotting.
- **Cross-site signature recurrence matrix** (type × site heatmap) — our validation, as
  a figure.
- **Fabric×experience contingency heatmap** — which void-condition types each fabric type
  produces (the cross-level signature).
- **Boundary-gradient transects** — λf/SVF/porosity across the formal–informal seam.
- **Flat-vs-terrain SVF delta map** — published as a known-error layer (Risk 1).
- **Prioritization map** — ranked poor-ventilation × low-SVF heat risk, the WS-B payoff.

## Risks & decisions

1. **Terrain-awareness is the biggest validity risk.** λf and SVF assume a flat datum;
   on 20–30° favela slopes frontal area includes hillside, katabatic/slope-parallel flow
   overrides synoptic wind, flat SVF under-counts down-slope sun. Macdonald constants and
   the 0.35 skimming threshold were fit on flat staggered cubes → likely *over*-predict
   stagnation on stepped rows. **Decision:** slope + slope-aspect as explicit covariates
   everywhere; confirm SVF comes from a true terrain DSM (not footprint-extrusion — the
   phantom-tower / street-SVF drift class we already fixed); ship the flat-vs-terrain
   delta as a known-error layer.
2. **LCZ as a transfer claim, shown to misfit.** Map cells to LCZ (GeoClimate) but
   demonstrate Rio favelas are a 2–4-story masonry hybrid of LCZ 3 / LCZ 7 with no clean
   match, and LCZ has *no slope axis*. Contribution = a slope-augmented favela LCZ
   sub-typing (echoes the Kabul 2024 finding). Stewart & Oke 2012 is the framework.
3. **Stair/beco under-sampling.** Vidigal roads have no Escadaria/Ladeira category →
   angular integration on a vehicular-only graph overstates segregation. Prefer metric
   reach on a pedestrian-complete graph, or caveat heavily.
4. **Scope fork — recommend the lean path.** Full momepy 296-character taxonomy is
   defensible but heavy and tessellation-dependent (breaks on favelas). The lean
   environmental-vector clustering + cross-site recurrence is *stronger validation* for
   our data and far less brittle. Recommend lean; momepy characters can be added as a
   second-class texture/grain layer later (HiMoC 2025 / enclosed-tessellation are the
   fallback if we ever need cadastre-free plot proxies).

## Sequencing

1. **WS-0** unified feature table (unblocks everything).
2. **WS-A** signature clustering + cross-site recurrence (fast, high-insight, paper-worthy).
3. **WS-B** ventilation index + CFD-anchored calibration (the contribution; can start the
   pure-morphometrics index immediately, layer CFD calibration when real returns arrive).
4. **WS-C** figures grow alongside A/B, not as a separate phase.

## Key references (verify caveated entries before they enter a bibliography)

- Fleischmann, Feliciotti, Romice & Porta 2022, *EPB* 49(4) — numerical taxonomy (engine).
- Dibble et al. 2019, *EPB* 46(4) — morphometric phenetics origin.
- Arribas-Bel & Fleischmann 2022, *Habitat Int.* 128 — operational "signature" definition.
- Fleischmann & Arribas-Bel 2022, *Sci. Data* 9:546 — validated GB-scale signatures.
- Salazar Miranda et al. 2022, *EPB* 49(9) — **Favelas 4D**, the favela anchor *(verify venue)*.
- Taubenböck & Kraff 2014, *J. Housing Built Env.* 29(1) — "physical face of slums."
- Kohli et al. 2012, *CEUS* 36(2) — Generic Slum Ontology (what to measure).
- Grimmond & Oke 1999, *JAMC* 38(9) — morphometrics→roughness, flow regimes.
- Macdonald et al. 1998, *Atmos. Env.* 32(11) — z0(λf)/zd(λp) closed form.
- Stewart & Oke 2012, *BAMS* 93(12) — Local Climate Zones (transfer framework).
- Wong & Nichol 2010, *Build. Env.* 45(8) — ventilation-corridor GIS recipe.
- Javanroodi et al. 2023, *STOTEN* 829 — CFD-anchors → surrogate → transfer.
- Inostroza et al. 2016, *PLOS ONE* 11(9) — heat-vulnerability index template.
- "LCZ in informal settlements: Kabul" 2024, *Sustain. Cities Soc.* — the skeptical test.

**Tools:** momepy, libpysal, spopt.region (WardSpatial/Skater/MaxP), scikit-learn
(GMM+BIC, AgglomerativeClustering w/ connectivity), XGBoost+SHAP, clustergram, legendgram
+ mapclassify, umap-learn, datashader, GeoClimate (vector→LCZ), UMEP (SVF/SOLWEIG).
