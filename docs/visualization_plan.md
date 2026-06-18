# Visualization Track — Plan (for expert review)

*Draft 2026-06-18. Sister track to the morpho-signature work. This plan is sent
to expert reviewers BEFORE we generate the final figures; their critique is
folded back here, then figures are generated and assembled into a reviewable
HTML gallery.*

## Goal & audiences

Communicate the favela morphometric signature + morphometrics-only
prioritization to three audiences, with one coherent visual language:

1. **Paper reviewers** (urban morphometrics / urban climate) — rigor, honest
   uncertainty, comparability with the Fleischmann/LCZ conventions.
2. **Technical report** — the project's primary deliverable.
3. **Presentation / policy** (planners, architects) — the prioritization story:
   which favela fabric reliably produces sun-starved, unventilated voids.

## Design system (the spine — applies to every figure)

- **Colour = type identity.** One categorical palette for the 6 morphotypes,
  reused across radar = dendrogram = map = recurrence = embedding. Must be
  **colourblind-safe** (Okabe–Ito or a curated 6-class set, *not* matplotlib
  `tab10` default).
- **Sequential** palette for continuous/prioritization layers; **diverging** for
  deltas (flat-vs-terrain SVF).
- **NULL is shown, never imputed.** Cells with no street support render neutral
  grey (the WS-0 honesty rule), never interpolated colour.
- **Maps** carry scale bar + north arrow (reuse `add_scalebar_north` from
  `compare_mingze_vidigal.py`), a consistent projection (EPSG:31983), and a
  context basemap where it aids reading.
- **Typography / sizing** consistent; figure tracks per `feedback_figure_tracks`:
  `outputs/paper_figures/` = TR/presentations, `docs/manuscript/figures/` =
  paper candidates — don't mix.

## Figure inventory

### A · Signature (WS-A)
1. **Morphotype fingerprints** — standardized character profile per type.
   *Open issue:* radar charts are widely criticised (area distortion, axis-order
   arbitrariness). Offer **parallel-coordinates** or a **type×character heatmap**
   as the primary, radar as secondary. (expert call)
2. **Taxonomy dendrogram** — Ward on centroids; coloured by the type palette.
3. **Recurrence matrix** — site×type share heatmap; the cross-site validation as
   a figure, with the recurs/site-specific split annotated.
4. **Morphotype maps** — per-site choropleth. *Needs WS-A.2 spatial smoothing*
   (current 10 m maps are salt-and-pepper).
5. **Fabric×experience contingency** — per type, the experienced-condition
   profile (SVF, sun<2h fraction, deep-canyon fraction, H/W). Small-multiple or
   slope/heatmap. The cross-level signature.
6. **Embedding** — UMAP/PCA of cells coloured by type. *Needs `umap-learn`
   install* (PCA fallback works today).

### B · Method / honesty (WS-0)
7. **Street-support map** — `has_street_support`; shows the ~65% NULL cells so
   the support story is visible, not hidden.
8. **Flat-vs-terrain SVF delta** — known-error layer (plan Risk 1).

### C · Prioritization (WS-B, when ready)
9. **Morphometrics-only priority map** — ranked deep-canyon × low-SVF risk.
10. **Boundary-gradient transects** — λf/SVF/porosity across the formal–informal
    seam.

## Tooling & gaps

- Present: matplotlib, geopandas, libpysal, esda, scikit-learn.
- **Needs install (ask before touching shared env):** `umap-learn` (embedding),
  `mapclassify` (choropleth classification schemes), `seaborn` (clustermap),
  `contextily` (basemaps). PCA/manual classification are today's fallbacks.

## Decisions from expert review (2026-06-18)

Three reviewers (information design, thematic cartography, urban-morphology
communication) critiqued this plan + the draft figures. They converged; resolved
calls below are binding for generation.

**Palette (one system, everywhere).** Okabe–Ito 6-class categorical, tied to the
λp type order (D11), colourblind-safe; black/grey excluded so grey is reserved for
NULL. Assignment: T0 `#56B4E9`, T1 `#009E73`, T2 `#E69F00`, T3 `#F0E442`,
T4 `#D55E00`, T5 `#CC79A7` (sparse→dense; T3 the flatland-only type). NULL/no-support
`#E0E0E0` (+ hatch on maps, for greyscale). Sequential (prioritization) = `YlOrBr`,
a hue no type owns. Diverging (centroid z-scores, flat-vs-terrain delta) = `RdBu`/
`PRGn` centred hard at 0 (`TwoSlopeNorm`). Rule: **categorical = saturated mid-tone
on points/polygons/leaves; sequential & diverging pass through white inside heatmap
fills** — so the three roles never collide.

**Fingerprints — kill the radar.** Primary = **type×character diverging heatmap**
(annotated z to 0.1; rows in dendrogram leaf order; columns grouped density /
wind / terrain). Parallel-coordinates as secondary (paper only). Radar cut.

**Dendrogram.** Links neutral grey (structure), leaf swatches in the type palette
(identity), dashed "k=6 cut" line, y-axis "Ward linkage distance."

**Recurrence matrix.** Single-hue sequential (`YlGnBu`/`Blues`), **not viridis**.
Columns in dendrogram order; rows grouped flat (RdP, Maré) vs steep (Vidigal,
Rocinha, Complexo) so T3's flatland-specificity reads as a block. Annotate shares;
bold-separate the site-specific column; mark sub-5% cells.

**Experience profile.** Cleveland **dot-plot small-multiples** (one panel per
outcome; types on y in dendrogram order; dots in type palette; size/opacity ∝
support coverage — never a confident dot on 23% support). Add ≥1 **distribution**
(violin/ridgeline) per type to honour the worst-tail (Favelas-4D precedent). Not
radar, not slopegraph.

**Maps.** Pipeline: Queen mode-filter (done, WS-A.2) → dissolve same-type runs →
drop islands below MMU (8–12 cells) → thin white casing between regions. Okabe–Ito
fill; NULL grey+hatch; **site boundary outline** (separates unsupported-interior
from off-site) + the formal street grid outside as the seam story. Scale bar +
north arrow (`add_scalebar_north`). Multi-site = **constant ground-scale**
small-multiples (gridspec by bbox; Maré larger), one shared legend, one Rio locator
inset. The raw 10 m per-cell map goes to a supplementary figure (honesty).

**Three NEW figures the domain reviewer demands (recurrence is under-built):**
1. **Recurrence evidence** — per-site morphotype centroid overlay (parallel-coords,
   one panel/site, same type drawn in each) + the bootstrap-stability strip (we have
   ARI 0.90). Promotes the headline contribution from 1 figure to 2.
2. **Naive-vs-support aggregation** — one site, nearest-k mean SVF vs support-aware
   p10 with NULLs grey. Proves the change-of-support decision had visible effect.
3. **Terrain error by morphotype** — flat-vs-terrain SVF delta binned by type/slope:
   does the known error track the prioritized (steep, dense) types? Answer it before
   a reviewer asks.
   Plus a supplementary **k-selection battery** (BIC elbow + silhouette/CH/DB).

**Demote/cut.** UMAP embedding → supplementary (datashaded) or cut from the planner
track. Radar → cut from the paper track.

**Reframing (free, high-leverage).** Caption that experience variables were
**held out** of clustering → Fig 5 is *external validation*, not a correlation; lead
the experience story with the variables least collinear with λp (sun<2h fraction,
H/W), not SVF. Present the LCZ 3/7-hybrid misfit as a *contribution* echoing an
accepted LCZ finding. State SVF provenance (terrain DSM vs footprint-extrusion) in
captions. Prioritization map = **quantile classes** (not continuous physical units),
**overlay the real CFD anchor locations**, lead with the pure-geometry index and
label any synthetic-CFD calibration explicitly.

## Review protocol

1. Three expert lenses critique this plan + the current draft figures
   (`outputs/cross_site/signature/figures/`): information design, thematic
   cartography, urban-morphology communication.
2. Synthesize critique into this doc (a "Decisions from review" section).
3. Generate the agreed figure set into a self-contained **HTML gallery**
   (precedent: the Mingze report) served locally for the user to review.
4. Guided review: each figure paired with the decision it embodies and a
   yes/refine prompt, so the user can sign off or redirect figure-by-figure.
