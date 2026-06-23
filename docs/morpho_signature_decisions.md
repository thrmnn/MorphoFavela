# Morpho-Signature Track — Decision Log

Append-only record of methodological choices and the tradeoffs behind them, so
they can be justified (or revisited) later. Newest decisions at the bottom of
each workstream. Companion to `docs/morpho_signature_plan.md`.

---

## WS-0 — Feature substrate

**D1 · Two linked tables, not one merged table.**
Grid cells (areal) and street observers (network points) are different spatial
supports. Collapsing street → cell by a mean is a change-of-support / MAUP error.
*Tradeoff:* two tables are slightly more to carry than one wide frame, but a
single merged table would silently bias every downstream model. Chose correctness.

**D2 · Fabric table excludes the street-aggregated `svf`/`svf_count`.**
`run_morphometric_audit` writes a per-cell `svf` that is itself a naive average of
street points (`_aggregate_svf_to_grid`). Excluded from the fabric feature set:
SVF is an *outcome*, not a fabric input, and double-counts (porosity ≈ f(λp),
SVF ≈ f(λp, H/W)). *Tradeoff:* loses a convenient per-cell SVF column; recovered
properly as support-aware summaries (D4).

**D3 · Enrichment is areal → point only.**
Each observer gets its cell's fabric covariates by point-in-polygon lookup (the
cell value is constant over its area — no averaging). The reverse (point → area by
mean) is the move we reject. Legitimate direction; matches Favelas-4D's
street-section altitude.

**D4 · Street → cell summaries are support-aware and distributional; empty cells
are NULL.**
When street conditions must appear per cell we emit p50 + worst-decile p10 +
fraction-below-threshold + `n_street_obs` + `has_street_support`, never a bare mean,
and never nearest-k interpolation in the modeling table. *Outcome:* only ~35% of
cells have any observer (vidigal 1104/3169 ≈ 65% unsupported). Naive nearest-k
would have fabricated values for those interior/building cells. The design surfaces
the gap instead of hiding it.

**D5 · H/W joined by nearest within 3 m + flag; calibration sites lack openness/H-W.**
`hw_streets` is a coarser sampling (971 vs 6876 pts at Vidigal), so it is attached
to the primary observers by nearest-neighbour within 3 m (`has_hw` flag). The 3
calibration sites (borel, jacarezinho, morro_do_juramento) carry grid + SVF/solar
only; handled gracefully (NULL openness/H-W).

**D6 · Scripts force repo root onto `sys.path`.**
A stale `brisa-0.1.0` editable install maps top-level `src` → `/home/theo/brisa/src`
via an *appended* meta-path finder; it wins whenever the repo root is absent from
`sys.path` (so direct `python scripts/...` imported the wrong `src`). Fix:
`sys.path.insert(0, ROOT)` before importing `src`, matching the existing scripts.
*Note:* the right long-term fix is to reinstall the editable package at the renamed
path, but that touches the shared env — deferred.

---

## WS-A — Morphometric signature

**D7 · Lean 6-feature fabric vector.**
`lambda_p, H_mean, sigma_h, lambda_f_mean, lambda_f_aniso, slope_deg`.
Exclusions, each with cause (measured on 74 169 pooled built cells):

| dropped | reason |
|---------|--------|
| `porosity` | r = −0.975 with `lambda_p` — redundant density axis |
| `far` | r = 0.72 with `lambda_p`, ≈ `lambda_p`·`H_mean`/3 — collinear |
| `street_orientation_entropy` | 58.9% NaN, median 0.00, max 0.54 — too sparse / low variance |
| `building_count` | a raw count, scale-dependent, not a normalized character |
| `northness`/`eastness` | absolute slope facing is an *environmental* covariate; excluding it keeps the signature rotation-invariant |
| 8 directional `lambda_f_*` | collapsed to magnitude (`lambda_f_mean`) + anisotropy (`lambda_f_aniso = lambda_f_max − lambda_f_mean`), so type ≠ a function of absolute wind orientation |

*Tradeoff:* parsimony over completeness. FAR/porosity/texture can be re-added as a
second-class layer; the lean set keeps each axis a distinct, interpretable
morphological dimension. Revisit if a momepy texture/grain character is added.

**D8 · `sigma_h` imputed to 0 for single-building cells.**
13.9% of built cells have NaN `sigma_h`, ~all from ≤1-building cells where height
spread is genuinely 0. Impute 0 rather than drop. *Outcome:* only 34 / 74 169 cells
still carry a residual NaN and are dropped — negligible.

**D9 · Standardize on the pooled built cells (not per-site).**
A z-score over all sites means a morphotype denotes the same thing everywhere —
required for the cross-site recurrence test (D13). Per-site standardization would
make "type 3 in Maré" incomparable to "type 3 in Vidigal".

**D10 · GMM (full covariance), k chosen by BIC *elbow*, not argmin.**
GMM BIC keeps falling on large n, so argmin lands at the krange boundary; a
kneedle-style elbow on the BIC curve gives a parsimonious, reproducible k. A
silhouette / Calinski-Harabasz / Davies-Bouldin battery is reported as a
cross-check (`k_selection.csv`). `random_state=0`, `n_init=4` at the final fit.
*Outcome:* **k = 6.** *Tradeoff:* the elbow is a heuristic; the battery + the
recurrence/experience validation (below) are the real defence, not k alone.

**D11 · Labels ordered by ascending mean `lambda_p`.**
Type 0 = sparsest fabric → type 5 = densest, for stable, readable labels across
runs and figures. Cosmetic; does not affect membership.

**D12 · Cluster fabric only; experience is a downstream per-type profile.**
Preserves the WS-0 support separation — experience never enters the clustering.
Each type is profiled over its *supported* cells (`support_coverage` reported).
A separate experience clustering (full fabric×experience contingency) is deferred
to WS-A.2.

**D13 · Recurrence test = share ≥ 5% in ≥ 3 of the 5 campaign sites.**
Operational definition of a "signature": a type that recurs across sites, not a
site artefact. *Outcome:*

- Types **0, 1, 2, 4, 5 recur** (≥4 sites); **type 3 appears in only 2 sites**
  (Rio das Pedras + Maré, the two flat sites) → a flatland-specific type, not a
  universal favela signature.
- The experienced-condition gradient (which the clustering never saw) is monotonic
  and confirms the fabric types: type 5 (densest, recurs in all 5 sites, 16–39%
  share) has `frac_below_2h = 1.0`, `frac_deep_canyon = 0.89`, H/W ≈ 3.5 — a
  universal favela fabric that reliably produces sun-starved deep canyons. This is
  the prioritization signal for WS-B.

---

## WS-A.2 — Spatial refinement & stability

**D14 · Salt-and-pepper fixed by a contiguity mode filter, not regionalization.**
Per-site libpysal Queen weights → iterated majority filter (2 passes; ties keep the
current label, so it is conservative and deterministic). *Outcome:* same-type
adjacency (spatial purity) rises **0.43 → 0.80** mean across the 8 sites, ~35% of
cells relabeled. A `morphotype_smooth` column is written back alongside the raw
`morphotype`. *Tradeoff (flagged by the cartography reviewer):* mode-filtering is a
*display generalization* that **preserves the global k=6 GMM types** — so the
cross-site recurrence claim (D13) still holds. Full `spopt` regionalization
(SKATER/MaxP) would give cleaner regions but a *different, site-specific*
segmentation that breaks recurrence; reserve it for a single-site illustrative inset
only. Raw per-cell labels remain the honest datum (kept in `morphotype`); the smooth
column is for legible maps. `spopt` not installed.

**D15 · k=6 is stable under resampling.**
Bootstrap (20 GMM refits on 80% subsamples, ARI vs the reference labels):
**mean ARI 0.901, sd 0.109, min 0.729.** Promote this from a footnote to a main
figure (the domain reviewer notes a Fleischmann-lineage reviewer will demand cluster
stability regardless).

## WS-B — Morphometrics-only prioritization (geometry-first)

**D16 · Pure-geometry deprivation index, ranked, worst-decile to cell.**
Per-observer score = equal-weight mean of three [0,1] components, all ray-cast or
geometric, no CFD: `sun_deficit` (winter sun shortfall below WHO 2 h),
`sky_enclosure` (1−SVF), `wind_stagnation` (λf / 0.35 skimming threshold,
Grimmond & Oke 1999). *Choices, each per the domain-review:* (i) scored at the
**observer/void** level (the unit of exposure), aggregated to cells by **worst-
decile p90**, not a mean (preserves the WS-0 support stance; NULL where no
support); (ii) shown as **tertile rank classes** (lower/elevated/highest) on
**shared pooled breaks**, never absolute units — we have no validated absolute
scale without CFD; (iii) **equal weights are provisional** and are the explicit
hook where sparse CFD anchors will recalibrate (the methodological contribution).
*Outcome:* Rocinha skews "highest" (3357 vs 487 lower — steep dense hillside),
flat Maré skews "lower" (7303) — face-valid. `priority_p90`/`priority_class`
written back to `features_grid.parquet`; map in the gallery. **Not yet done:**
CFD-anchor overlay on the map, boundary-gradient transects, weight calibration.

**D17 · Signature scoped to the 5 campaign sites (2026-06-19, user).**
The 3 calibration sites (borel, jacarezinho, morro_do_juramento) are kept *aside* —
the morphotype signature is derived on the 5 campaign sites only, and no figure
shows the calibration sites. k stays **6** and bootstrap ARI stays **0.90** on the
5-site pool, so the scoping does not destabilise the typology; T3 remains the
flatland-specific type (now RdP + Maré). The calibration sites retain their feature
and roughness tables on disk (out-of-sample, projectable later); their stale
morphotype labels were cleared. *Why:* calibration sites were never CFD-campaign
sites; defining the signature on the study sites is the cleaner scientific scope.

## Open questions / revisit list

- **Terrain-aware λf & SVF (plan Risk 1).** Both assume a flat datum; on 20–30°
  slopes this biases the densest types. Slope is in the vector, but λf/SVF
  themselves are flat-datum — quantify the flat-vs-terrain delta before trusting
  absolute thresholds.
- **Experience coverage is type-dependent** (support_coverage 0.23–0.59). The open
  types (2, 5) are *less* observed — confirm this is sampling, not bias, before
  reading their experience medians strongly.
- **k robustness.** Re-run with bootstrap cluster stability (ClustGeo / A-DBSCAN)
  to confirm k = 6 is not fragile.
- **Separate experience clustering** for the full cross-level contingency (WS-A.2).
- **Spatial contiguity.** The per-site morphotype maps show genuine regional
  structure (e.g. Vidigal: T4 dominates the east, T1 the upper-west) but are
  salt-and-pepper at 10 m resolution — GMM has no contiguity term, so adjacent
  cells flip type. Next refinement is spatially-constrained clustering (plan
  Recipe C: `spopt.region` Ward/Skater/MaxP, or sklearn `AgglomerativeClustering`
  with a libpysal `W`) to dissolve types into coherent morphotope regions. The
  current non-spatial labels are correct per-cell; the noise is spatial, not
  categorical.
