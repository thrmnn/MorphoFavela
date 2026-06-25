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

## 1. THE VENTILATION-AXIS DECISION (urban-physics council, 2026-06-24) — geometry classifies the REGIME; CFD carries ADEQUACY

The multi-variable index is **REJECTED** (unanimous: urban climatologist · CFD modeller · parsimony critic). It is unfittable without ground truth, and H/W is collinear with SVF on the vertical-openness axis — which is the *solar* axis — so it is "openness measured three times": an unfalsifiable weighted score a Lancet referee dismantles on sight. The author's instinct was right: reconsider λf itself, simplest and most elegant.

**λf saturation is the FINDING, not a liability.** ~96% of the fabric is past the Oke/Grimmond–Oke skimming-flow onset (neighbourhood median λf ≈ 1.65) → the fabric is *uniformly in the skimming-flow regime, the most ventilation-suppressed urban-canopy class*. Frontal-area density diagnoses the **regime**; it cannot finely rank ventilation **adequacy** within it (drag/sheltering saturates past the inflection — MacDonald 1998, Hagishima 2009, Zaki 2011).

**Two-tier division of labour:**
- **Tier 1 — geometry classifies the REGIME at scale** (whole grid): λf against the canonical Oke/Grimmond–Oke thresholds (isolated → wake → skimming). The at-scale statement is the saturation finding. **No fitting; no per-cell adequacy claim.**
- **Tier 2 — CFD age-of-air τ carries per-cell ventilation ADEQUACY** on a validated patch sample. τ is the only variable allowed to claim "ventilation adequacy".

**Drop the relative `λf>2.75` percentile flag and the 6.6% "ventilation-only" state** — it is the densest tail of the already-skimming, with no physical regime boundary (a percentile artifact). Report it as one sentence ("only X% of cells fall outside skimming flow"), not a co-equal taxonomy state.

**Axes stay orthogonal, single-scalar each:** SVF → **solar** axis (its dominant role); λf → **regime** label; **do NOT pair H/W+SVF for ventilation** (collinear on the vertical-openness/solar axis). The genuinely independent ventilation signal is **lateral connectivity / exposure-to-open-edge** (distance-to-open-boundary / lateral aperture) — the one quantity that still varies once λf plateaus. If a pre-CFD within-regime *tendency* layer is wanted, use that single connectivity scalar, explicitly **qualitative, superseded by τ** — never a fitted composite.

**Validation when CFD lands:** **Spearman/Kendall rank concordance** (geometry rank vs τ rank) on the patches, bootstrap CI, pre-registered interpretation. A weak ρ is itself a finding (geometry insufficient → CFD necessary). **Patch sample stratified on the JOINT geometric distribution** (+ off-diagonal disagreement cells + a random/spatial reserve), NOT on the proxy's extremes (else self-validating); report patch-vs-grid geometric coverage.

**At-scale honesty in every figure:** the at-scale ventilation layer is a screening REGIME, never "validated τ"; CFD patches are a visible **outlined SUBSET**; unsampled cells marked provisional; taxonomy cells labelled by **EVIDENCE TIER** ("adequate — CFD" vs "likely-adequate — geometry, provisional"). The design rests on one sentence: *"Geometry tells us the fabric is uniformly in the skimming-flow regime — it classifies which regime a cell is in, but not how ventilated it is; that finer adequacy is carried only by CFD, on a validated sample."*

This is a **Methods/Results reframe + new analysis** (regime classification + the patch-CFD rank validation). Placeholder until done: the manuscript carries the **interim λf regime map**, marked provisional, and **no per-cell geometric ventilation-adequacy claim**.

---

## 2. Fig 1 — study sites + pipeline schematic (target 180 mm, 3-row, 12-col grid)

**Text-overflow fixes:** input boxes overflow — fix box width 30 mm, two-line title, provenance own line 5.5 pt, **auto-height**, 1.5 mm inset, shrink-to-5.5pt-then-abbreviate. Move the floating italic "dashed = provisional…" note to a **dedicated caption strip** beneath the schematic band. 2 mm clearance on 3D thumbnail titles.

**Provenance (author correction — DTM is ALS-derived):** drop the per-box "not open" repetition. Each input box carries only its source in 5.5 pt italic: `IPP cadaster` (footprints), `ALS (MIT/SondoTécnica)` (heights), `DTM — extracted from ALS`. **The DTM box branches off the ALS box** (elbow connector) to encode "derived from ALS," not an independent input. **One global footnote** carries openness once: *"Inputs: IPP cadaster + MIT/SondoTécnica ALS (not openly redistributable). Pipeline is open."* Single lock glyph in the INPUTS header, not per box.

**Adequacy taxonomy as a 2×2 matrix** (replaces the 4 stacked chips): x = ventilation (pass→fail, `index tertile`), y = sunlight (pass→fail, `≥2 h winter sun`). Cells: Adequate (both pass) · Ventilation-deprived (sun pass/vent fail) · Sunlight-deprived (sun fail/vent pass) · Compound (both fail), in the four-state palette §0. **Nominal, not ordinal** (firewall): equal visual weight, state name in each cell, caption "four *kinds* of deficit, not tiers of risk" — **no 1–4 numbering, no severity diagonal, no intensity ramp**. Mark the ventilation axis as provisional (hatch the vent-fail column lightly).

**Two-tier pipeline depiction:** two converging lanes — top (wide, full-fabric) = geometry → **flow-regime classifier** (λf vs Oke/Grimmond–Oke thresholds → "uniformly skimming"); bottom (narrow, sampled) = **patch CFD → τ (per-cell ventilation adequacy)**. A **back-arrow from CFD up into the geometry lane** labelled "validate ranking (Spearman ρ)". The ventilation output node reads "**flow-regime: skimming (provisional)**" with a ghosted "→ per-cell τ (Tier 2, pending)" box. Geometry classifies the regime; CFD carries adequacy.

**3D render (panel c) — separate buildings from terrain:** (1) terrain = neutral desaturated **hillshade** (greys `#BFBFBF`→`#6E6E6E`, single light azimuth 315°/45°); (2) buildings = saturated **height ramp** (cividis/copper), distinct from grey terrain; (3) 0.3 pt dark **edge strokes** on building footprints; (4) soft **ambient-occlusion contact shadow** at the building/terrain join. Same light direction across all four sites. Height colormap stated once in the panel-c subtitle.

---

## 3. Fig 2 — winter-sun + λf maps + distributions (target 180 mm, 4-row)

Rows A (λf maps) and B (winter-sun maps) **column-aligned by site** (λf above, sun below); shared per-row colourbars; scale bar on Row A only.

**λf maps (Row A) — bring colour + reframe as a REGIME map:** off greyscale to **`mako`** (cool teal→navy, perceptually uniform, reads "cool/secondary" against the warm solar cividis). This row now shows the **flow-regime classification** (λf vs the Oke/Grimmond–Oke isolated/wake/**skimming** thresholds) — the visual point is that almost the whole fabric is one colour (skimming): *the saturation IS the finding*. **Tag "PROVISIONAL · regime classifier · per-cell ventilation adequacy = CFD-τ (Tier 2)"** and visually demote vs the solar row. Drop the p75 line.

**Winter-sun maps (Row B) — fix the muddy/occluded render** (root cause = sub-pixel per-cell polygons with anti-aliased edges → strokes dominate, fill muddies to a dark mass): **rasterize the cell layer** (`set_rasterized(True)`); **remove per-cell edge strokes** (`linewidth=0, antialiased=False`); **full opacity** (never stack semi-transparent cells); **white/no basemap** (a grey basemap worsens the occluded read; if context wanted, ≤5–8% grey behind a rasterized cell layer); if cells are finer than ~0.7 px at export dpi, **aggregate to a 2–3 m raster and `imshow`** instead of plotting 10⁴+ polygons; clip the cividis bar to the populated range so high-sun cells aren't crushed.

**Panel C — replace the λf ridge (author dislikes it) and make it the saturation-finding panel:** a **horizontal box/violin strip per site** (5 rows) over the **Oke/Grimmond–Oke regime bands** shaded behind all rows (isolated ≲0.15 · wake 0.15–0.65 · **skimming ≳0.65**), so the eye sees every site's λf distribution sitting deep in the skimming band (median ≈1.65). This panel *is* the at-scale ventilation result ("uniformly skimming → geometry can't grade adequacy → CFD"). Drop the p75=2.75 line. Make panel D the matching **winter-sun box strip** with the 2 h marker, for symmetry. (ECDF overlay → SI.)

---

## 4. Fig 3 — diagnostic taxonomy (target 180 mm; maps 3×2 + F/G/H band)

**Maps (A–E)** in a 3×2 grid; **6th cell holds the shared legend** (= the Fig 1 matrix palette). Same render fix as Fig 2-B (rasterize, no strokes, white basemap).

**The (F)/(G)/(H) overlap fix — exact reflow:** 3-column split **F = 42% · gutter 6 mm · G = 30% · gutter 6 mm · H = 28%** of 180 mm; **no content enters a gutter**. (F) bar labels inside F's column, % labels on-segment (auto-contrast). (G) header shortened to **"(G) Moran's I — compound clustering"** wholly within G's column; site labels on G's y-axis; values at bar ends with 1 mm inset. (H) fixed column widths, right-align numerics, ellipsise `Rio das Pedr…` only within H. **3 mm clearance** between the maps row and the F/G/H band. (G) bars a **single neutral hue** (site identity is on the axis).

**Taxonomy display (post-ventilation-decision — see §1).** The ventilation axis is now **regime (uniformly skimming, provisional) + CFD evidence tier**, NOT a fine geometric grade. Consequence to flag for the PI: at scale the ventilation flag is ~universal, so a naïve four-state map collapses toward the solar axis (almost every sunlight-deprived cell also reads "in skimming fabric"). **Recommended Fig 3:** lead with the **solar-resolved adequacy map** (adequate / sunlight-deprived, the delivered axis), overlay the **uniform skimming-regime context** as a hatched provisional layer, and mark **CFD-validated patches as outlined tiles** where τ exists ("adequate — CFD" vs "likely — geometry, provisional"). Keep the nominal 2×2 legend (Fig 1) but state in the caption that the ventilation axis is a regime, resolved per-cell only on CFD patches. Per-cell "ventilation-deprived/compound" counts await CFD; report the interim λf-flag shares only as clearly-labelled provisional placeholders. The (G) clustering panel stays as the **provisional-axis** result — title "(G) Compound-state spatial clustering (interim λf flag, provisional)". **In-panel anti-misuse banner that survives cropping**: "Investment-priority surface — geometry scored vs an external reference. NOT a habitability/risk/clearance map; must not justify relocation." Table (H): rename "max patch" → **"largest contiguous upgrading unit (m²)"**, add "patches are investment units, never dwellings; sites not ranked". "10 m cell (=100 m²)" in legends; scale bar + north arrow per map.

**The single firewall guard every panel must carry:** this is an **ecological geometry-vs-external-reference adequacy surface aggregated to contiguous investment patches** — not a measured, individual, or between-community risk/ranking map. No dwelling-resolved colouring; no site-vs-site league ordering; no disease burden fused into the fill.

---

## 5. fig05 (Supplementary predictors) — carry the round-2 items
AUC box reports BOTH full-sample reduced-feature LOSO 0.76–0.84 and complete-case full-feature 0.87–0.93 (prevalence 46% vs 56%); separable PD lines (SVF solid / slope solid green / northness dashed blue); on-panel hemisphere sign convention; "P(fail)" → "P(below adequacy floor)"; shrink the pending-CFD placeholder (D) to a small inset.

---

## Handoff back
Export regenerated PNGs to `outputs/paper_figures/exports/` (same names); brisaverse re-promotes. **New analysis needed (placeholder until done):** (1) the λf **flow-regime classification** against the Oke/Grimmond–Oke thresholds (cheap; do now — it's the saturation finding); (2) optionally a single **lateral-connectivity / exposure-to-open-edge** scalar as the qualitative within-regime tendency; (3) the **patch-CFD rank-concordance validation** (Spearman/Kendall ρ of geometry rank vs τ) — gated on the CFD campaign. Still BLOCKED (needs user): ray-caster vs Radiance/SOLWEIG cross-validation.

**PI decisions DECIDED this session (no longer open):**
- ✅ Ventilation axis = **geometry classifies the regime (λf, uniformly skimming = the finding); CFD-τ carries per-cell adequacy**. Multi-variable fitted index REJECTED; λf>2.75 percentile flag + "ventilation-only" state DROPPED (urban-physics council, unanimous).
- ✅ **Nominal 2×2 matrix** taxonomy (kinds of deficit, not tiers).
- ✅ SVF–sun decoupling **stays in ED4** (not pulled into Fig 2).

**Still needs PI input:**
- The Fig 3 structural consequence (§4): at scale the ventilation regime is ~uniform, so the four-state map collapses toward the solar axis until CFD resolves within-regime ventilation. Confirm the recommended "solar-resolved map + uniform-skimming overlay + CFD-patch tiles" framing, or specify an alternative.
