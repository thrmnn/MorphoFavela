# MorphoFavela handoff — FINAL figure specification, ROUND 3 (2026-06-24)

For the MorphoFavela agent. This supersedes round-2 for the three **main** figures. It encodes a 3-expert council — **scientific figure designer**, **CFD/ventilation methodologist**, **Lancet Public Health epi/firewall reviewer** — convened to specify the *final* figures the paper needs. **Spec philosophy (PI):** design the FINAL target; do **not** fit the figure to current results. **Placeholders are acceptable now** — several items below depend on analysis we have not run yet (the new ventilation index, the patch-CFD validation). Where a number isn't computed, render a clearly-marked placeholder, do not fabricate.

Provenance rule unchanged (pipeline open, data not redistributable); five named sites only.

---

## 0. CROSS-CUTTING — the text-fits-in-box gate (recurring defect, now a HARD GATE)

Text overflowing its box and panel headers overlapping keep recurring (fig01 input boxes; fig04 (F)/(G)/(H) band). **Encode this as a reusable assertion pass in the figure scripts and fail the export on any violation** — do not eyeball it:

1. **No glyph crosses any boundary.** For every text artist, assert its rendered bbox (`renderer.get_window_extent`) is inside its container bbox (box/panel/axes/figure) minus the inset.
2. Box text honours a **1.5 mm inset**, wrapped to `width − 2×inset`; auto-shrink to a **5.5 pt floor** before abbreviating; never clip.
3. **Boxes are auto-height**, never fixed-height-with-clip.
4. Panel letters/titles clear neighbours by **≥2 mm**.
5. **Gutters are empty** — no bars/labels/arrows render inside an inter-panel/inter-box gutter; arrows live only in the arrow channel.
6. Colourbar labels clear the figure edge (**≥4 mm right margin**).
7. Min type **5.5 pt**, min stroke **0.3 pt** anywhere.
8. Long site names (`Rio das Pedras`) fit their column at the chosen size or are ellipsised **within** the column; check at both 90 mm and 180 mm.
9. **CVD pass:** run every categorical palette through a deuteranopia/protanopia simulator; adjacent states differ in **both hue and ≥15% lightness**.
10. **Screenshot one render at final export width** (not a zoom) and visually confirm no overflow before sign-off — the defect is only catchable at true size.

Targets: **90 mm single column** baseline, **180 mm** where panels demand (noted per figure), **600 dpi**, sans family (Helvetica/Arial). Type ramp at 90 mm: panel letters 9 pt bold · titles 8 pt · axis labels 7 pt · ticks 6 pt · annotations 5.5 pt floor.

**Palette decision:** keep the project **Okabe-Ito** categorical standard (D19) as the source, mapped to the four states with the amber↔orange pair held ≥15% lightness apart. If Okabe-Ito cannot separate them, the designer's reference hexes are the fallback: Adequate `#4C8C8C` · Sunlight-deprived `#E5B85C` · Ventilation-deprived `#D98E4A` · Compound `#9E3B4E`. **The SAME four-state palette is reused verbatim in the Fig 1 matrix and Fig 3** (one legend serves both). Per the firewall reviewer the taxonomy is **nominal, not a heat-ramp** — categorical hues, no sequential intensity, no severity diagonal.

---

## 1. THE METHODOLOGY CHANGE — ventilation axis (do this before/with the figures)

The ventilation axis is under-robust: a single relative `λf > 2.75` (pooled p75) barely discriminates because ~96% of the fabric is past the skimming-flow onset, leaving only **6.6% "ventilation-only" cells** — a knife-edge percentile artifact, not a finding. Council resolution:

**Tier 1 — geometric pre-screen at scale (whole 10 m grid, all five sites).** Replace λf-alone with a **multi-variable geometric ventilation-potential index** = f(λf, canyon aspect H:W, cluster openness/porosity). **Exclude or residualize SVF** — SVF is the dominant *solar* predictor, so using it on the ventilation axis would manufacture the compound state and make the two axes non-independent (a reviewer will check exactly this). The index re-introduces variance *inside* the skimming regime where λf is flat. Compute it **continuous underneath**.
**Tier 2 — patch/cell CFD validation (sampled subset).** OpenFOAM RANS τ on sampled patches **validates the ranking** (Spearman ρ of index vs τ) and, once AIJ-laddered, **calibrates an absolute τ floor** to replace the relative cut. Until then the ventilation axis stays **provisional/relative**.

**Display rule (firewall):** do **not** display a smooth continuous "ventilation badness" gradient over named favelas (more weaponizable). **Display the index as ordinal tertiles + provisional hatching**, scored against an external reference, aggregated to patches. **Drop "ventilation-deprived only" as a co-equal discrete state** and recast the sparse pure-ventilation deficit as a **morphological finding**: "in fabric this uniformly enclosed, ventilation deprivation almost always co-occurs with solar deprivation; pure ventilation failure is rare." That sentence belongs in the Fig 3 caption and the Discussion.

This is a **new analysis** (the index). Placeholder until computed: keep the current λf panel but label it "interim single-variable proxy — to be replaced by the multi-variable ventilation-potential index (pending)."

---

## 2. Fig 1 — study sites + pipeline schematic (target 180 mm, 3-row, 12-col grid)

**Text-overflow fixes:** input boxes overflow — fix box width 30 mm, two-line title, provenance own line 5.5 pt, **auto-height**, 1.5 mm inset, shrink-to-5.5pt-then-abbreviate. Move the floating italic "dashed = provisional…" note to a **dedicated caption strip** beneath the schematic band. 2 mm clearance on 3D thumbnail titles.

**Provenance (author correction — DTM is ALS-derived):** drop the per-box "not open" repetition. Each input box carries only its source in 5.5 pt italic: `IPP cadaster` (footprints), `ALS (MIT/SondoTécnica)` (heights), `DTM — extracted from ALS`. **The DTM box branches off the ALS box** (elbow connector) to encode "derived from ALS," not an independent input. **One global footnote** carries openness once: *"Inputs: IPP cadaster + MIT/SondoTécnica ALS (not openly redistributable). Pipeline is open."* Single lock glyph in the INPUTS header, not per box.

**Adequacy taxonomy as a 2×2 matrix** (replaces the 4 stacked chips): x = ventilation (pass→fail, `index tertile`), y = sunlight (pass→fail, `≥2 h winter sun`). Cells: Adequate (both pass) · Ventilation-deprived (sun pass/vent fail) · Sunlight-deprived (sun fail/vent pass) · Compound (both fail), in the four-state palette §0. **Nominal, not ordinal** (firewall): equal visual weight, state name in each cell, caption "four *kinds* of deficit, not tiers of risk" — **no 1–4 numbering, no severity diagonal, no intensity ramp**. Mark the ventilation axis as provisional (hatch the vent-fail column lightly).

**Two-tier pipeline depiction (methodologist):** two converging lanes — top (wide, full-fabric) = geometric pre-screen → ordinal ventilation-potential surface (provisional, relative); bottom (narrow, sampled) = patch CFD → τ. A **back-arrow from CFD up into the proxy lane** labelled "calibrate / validate ranking". Relabel the ventilation output node "ventilation-potential index (provisional, relative)" with a ghosted "→ τ-calibrated (Tier 2, pending)" box.

**3D render (panel c) — separate buildings from terrain:** (1) terrain = neutral desaturated **hillshade** (greys `#BFBFBF`→`#6E6E6E`, single light azimuth 315°/45°); (2) buildings = saturated **height ramp** (cividis/copper), distinct from grey terrain; (3) 0.3 pt dark **edge strokes** on building footprints; (4) soft **ambient-occlusion contact shadow** at the building/terrain join. Same light direction across all four sites. Height colormap stated once in the panel-c subtitle.

---

## 3. Fig 2 — winter-sun + λf maps + distributions (target 180 mm, 4-row)

Rows A (λf maps) and B (winter-sun maps) **column-aligned by site** (λf above, sun below); shared per-row colourbars; scale bar on Row A only.

**λf maps (Row A) — bring colour:** off greyscale to **`mako`** (cool teal→navy, perceptually uniform, reads "cool/secondary" against the warm solar cividis). Fallback `YlGnBu` upper-70% (avoid the near-white low end on white). Mark the threshold as a contour, not a separate legend. **Tag the panel "PROVISIONAL · pre-CFD · relative pre-screen, not a ventilation rate"** and visually demote vs the solar row. (When the multi-var index lands, this row shows the index tertiles.)

**Winter-sun maps (Row B) — fix the muddy/occluded render** (root cause = sub-pixel per-cell polygons with anti-aliased edges → strokes dominate, fill muddies to a dark mass): **rasterize the cell layer** (`set_rasterized(True)`); **remove per-cell edge strokes** (`linewidth=0, antialiased=False`); **full opacity** (never stack semi-transparent cells); **white/no basemap** (a grey basemap worsens the occluded read; if context wanted, ≤5–8% grey behind a rasterized cell layer); if cells are finer than ~0.7 px at export dpi, **aggregate to a 2–3 m raster and `imshow`** instead of plotting 10⁴+ polygons; clip the cividis bar to the populated range so high-sun cells aren't crushed.

**Panel C — replace the λf ridge (author dislikes it):** use a **horizontal box/violin strip per site** (5 rows), with the threshold as a single vertical reference line and a shaded **"skimming-flow regime (λf > ~0.6)" band** behind all rows — shows median/IQR/tails + threshold + regime in one compact strip, and motivates abandoning λf-alone (the saturation *is* the argument). Make panel D the matching **winter-sun box strip** with the 2 h marker, for symmetry. (ECDF overlay → SI; it double-encodes the deprivation fraction.)

---

## 4. Fig 3 — diagnostic taxonomy (target 180 mm; maps 3×2 + F/G/H band)

**Maps (A–E)** in a 3×2 grid; **6th cell holds the shared legend** (= the Fig 1 matrix palette). Same render fix as Fig 2-B (rasterize, no strokes, white basemap).

**The (F)/(G)/(H) overlap fix — exact reflow:** 3-column split **F = 42% · gutter 6 mm · G = 30% · gutter 6 mm · H = 28%** of 180 mm; **no content enters a gutter**. (F) bar labels inside F's column, % labels on-segment (auto-contrast). (G) header shortened to **"(G) Moran's I — compound clustering"** wholly within G's column; site labels on G's y-axis; values at bar ends with 1 mm inset. (H) fixed column widths, right-align numerics, ellipsise `Rio das Pedr…` only within H. **3 mm clearance** between the maps row and the F/G/H band. (G) bars a **single neutral hue** (site identity is on the axis).

**Taxonomy display (methodologist + firewall):** present as the nominal 2×2-consistent four states, but **hatch the provisional ventilation-bearing states** (ventilation-deprived + compound) in every map/bar/legend; footnote "ventilation axis = geometric pre-screen, provisional pending CFD-τ". The (G) clustering panel is the **provisional-axis** result — title "(G) Compound-state spatial clustering (provisional axis)". **In-panel anti-misuse banner that survives cropping**: "Investment-priority surface — geometry scored vs an external reference. NOT a habitability/risk/clearance map; must not justify relocation." Table (H): rename "max patch" → **"largest contiguous upgrading unit (m²)"**, add "patches are investment units, never dwellings; sites scored vs an external floor, not ranked". "10 m cell (=100 m²)" in legends; scale bar + north arrow per map.

**The single firewall guard every panel must carry:** this is an **ecological geometry-vs-external-reference adequacy surface aggregated to contiguous investment patches** — not a measured, individual, or between-community risk/ranking map. No dwelling-resolved colouring; no site-vs-site league ordering; no disease burden fused into the fill.

---

## 5. fig05 (Supplementary predictors) — carry the round-2 items
AUC box reports BOTH full-sample reduced-feature LOSO 0.76–0.84 and complete-case full-feature 0.87–0.93 (prevalence 46% vs 56%); separable PD lines (SVF solid / slope solid green / northness dashed blue); on-panel hemisphere sign convention; "P(fail)" → "P(below adequacy floor)"; shrink the pending-CFD placeholder (D) to a small inset.

---

## Handoff back
Export regenerated PNGs to `outputs/paper_figures/exports/` (same names); brisaverse re-promotes. **New analysis needed (placeholder until done):** the multi-variable ventilation-potential index (§1) and its patch-CFD ranking validation. Still BLOCKED (needs user): ray-caster vs Radiance/SOLWEIG cross-validation.

**PI decisions this round surfaces:**
- **Adopt the multi-variable ventilation index** (replacing λf-alone) and recast "ventilation-only" as a morphological finding? — council-recommended; it is a real methods change (new analysis + a Methods-section rewrite once computed).
- Matrix taxonomy as **nominal** (kinds of deficit) vs the current 4 stacked states — designer + firewall both favour the nominal 2×2.
- Whether to pull the SVF–sun decoupling inset into Fig 2 (else keep ED4).
