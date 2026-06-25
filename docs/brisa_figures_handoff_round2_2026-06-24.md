# MorphoFavela handoff — figure regeneration ROUND 2 (2026-06-24)

For the MorphoFavela agent. Round 1 (fig01 composite, fig03/04/05 reworks, compound-state Moran's I, full-sample LOSO, neighbourhood λf) is **promoted into brisaverse `shared/figures/` and the manuscript** — thank you; all five landed well. This round encodes a **Lancet Public Health reviewer council** (4 reviewers: epidemiology/methods, health-equity/anti-weaponization, Lancet dataviz/accessibility, built-environment/exposure-science) that reviewed the promoted PNGs at the LPH bar.

**Spec philosophy (PI directive):** do **not** fit the figure to what the current script emits — **adapt the script to the spec below**. These are the figures the paper needs; generate to them. Provenance rule unchanged: ALS heights = MIT/SondoTecnica TLS (Vidigal subset), cadaster = IPP — pipeline open, **data not redistributable**; five named sites only.

**One theme dominates the council (3 of 4 reviewers + the only MAJOR-REVISE):** the honesty firewall and the anti-weaponization/OWED framing currently live in the **captions, not the pixels**. The load-bearing change is to move *provisional-axis marking* and *anti-misuse / owed-reference framing* **into the image**, because the misuse vector is a screenshot and the firewall risk is a polished provisional panel read as a validated result.

---

## P0 — load-bearing (firewall + anti-weaponization). Do these first.

### fig04 (manuscript Fig 3 — four-state taxonomy) — the weaponizable panel; equity = MAJOR REVISE
1. **Off the alarm-red ramp.** Drop saturated red as "compound." Red on named-favela silhouettes is a screenshot-ready clearance map. Use the **project Okabe-Ito categorical palette**, severity-ordered by *luminance* (not by danger-hue). Reserve pure saturated red for nothing.
2. **Provisional states must look provisional.** The ventilation-deprived and compound states inherit the interim λf>2.75 pre-screen (no validated CFD). Render both with a **diagonal hatch / stipple overlay** in every map, bar, and legend swatch, plus a legend footnote "provisional — pending CFD τ; relative λf pre-screen, no absolute rate." The delivered solar state must read as the only fully-solid category.
3. **In-panel anti-misuse banner that survives cropping.** A persistent strip *inside* the figure bounds (not the caption): "Investment-priority surface — geometry scored against an external sunlight reference. NOT a habitability, risk, or clearance map; must not be used to justify relocation." Caption-only is insufficient.
4. **OWED legend framing.** Keep the taxonomy state names but append the owed reference so the read is *deprivation-of-a-resource-owed*, not *deficiency-of-a-place*: e.g. "Sunlight-deprived — below the ≥2 h owed winter-sun floor", "Ventilation-deprived — below the ventilation floor (provisional)", "Compound — below both floors (provisional)". (Full lexical rename to "Below … floor" is flagged for PI; do the legend-append now.)
5. **Rework table (H) away from a league/clearance table.** Rename "max patch" → **"largest contiguous upgrading unit (m²)"**; add a header note "patches are investment units, never dwellings; sites are scored against an external floor, not ranked against each other"; de-emphasise the cross-site %-comparison columns (they currently read as a league table — Vidigal "worst at 74%").
6. **Mark Panel (G) as the provisional-axis result:** title "(G) Compound-state spatial clustering (provisional axis)"; the Moran's I bars need not all be red — colour by site or one neutral.

### fig03 (manuscript Fig 2 — sunlight deficit + provisional ventilation) — firewall RED FLAG (epi = MAJOR)
7. **Asymmetric DELIVERED vs PROVISIONAL styling.** Panel B (solar) = "DELIVERED · ray-traced"; Panel A (λf) = visually **demoted** (desaturate / grey frame / hatch overlay) + on-panel tag "PROVISIONAL · pre-CFD". Right now A and B look equally authoritative — the exact mistake the firewall exists to prevent.
8. **Relabel the λf colourbar** from "λf (cell-clipped, rel. p75 2.75)" to **"relative frontal-area density λf — geometric ventilation pre-screen, not a ventilation rate (higher = deeper skimming flow, less exchange)"**, and add a small **regime strip** under it marking isolated → wake → skimming with the skimming onset (~0.65) and the neighbourhood median (~1.65) ticked, so the reader sees the whole fabric sits deep in skimming.
9. **Fix the ridge panels (C)/(D) — they don't earn their place yet.** The densities are near-flat ribbons; the claimed Maré bimodality and Rocinha/Vidigal mass-at-zero are invisible. Increase per-ridge height to overlap ~0.4–0.6 of row spacing, draw each baseline, **preserve the 0 h bin** (do not KDE-smooth across the zero boundary — the fully-occluded mass-at-zero is a real, defensible feature), and annotate the mass-at-zero explicitly.

---

## P1 — craft / legibility (Lancet dataviz)

10. **fig04 (F)/(G)/(H) header band: text-on-text.** The titles collide with the (F)/(G) site labels and bar ends. Reflow into a clean 3-row stack with explicit vertical gutters; move site labels to a fixed left axis (not over bars); titles must not sit over panel content.
11. **Colour-accessibility (deuteranopia).** The orange↔red (ventilation vs compound) pair is the failure point — they collapse to one brown. The Okabe-Ito + luminance-ordering + hatch-on-provisional from P0 fixes this; verify pairwise ΔE under a deuteranopia sim before export.
12. **Cross-figure consistency (all figures):** site order **hillside→flatland everywhere** (Vidigal, Rocinha, C. do Alemão, Rio das Pedras, Maré); append **"10 m cell (1 cell = 100 m²)"** to *every* map legend, not just a footnote; add **scale bar + north arrow** to each small map (Maré is cropped tall/tiny — give it room or an inset).
13. **Panel letters/fonts:** bold ≥9 pt panel letters, consistent corners; map titles + colourbar labels ≥7–8 pt at 90 mm column width.

---

## P2 — honesty specifics & polish

14. **fig05 (Supplementary predictors) — AUC honesty box.** Replace the bare "ROC-AUC 0.87–0.93" with both numbers + provenance: **"full-sample reduced-feature LOSO AUC 0.76–0.84; complete-case full-feature 0.87–0.93 (SVF-driven); prevalence 46% full vs 56% complete-case."** (Stats already computed: `outputs/paper_figures/fullsample_loso.json`.) Keep Panel D's honest "pending CFD" stub but **shrink it** to a small labelled inset/footnote — it currently eats a full quadrant.
15. **fig05 (B)/(C):** make the three partial-dependence lines separable (SVF solid red, slope solid green e.g. `#1B9E77`, northness dashed blue — currently northness & slope are both dashed blue and merge); annotate the **hemisphere sign convention** on-panel ("Southern Hemisphere: south-facing → less winter sun; negative northness → P(fail)↑"); rename **"P(fail)" → "P(below adequacy floor)"** (consistent OWED idiom).
16. **fig01 (manuscript Fig 1):** relabel the taxonomy chips with the OWED legend phrasing (match fig04); **remove the duplicate** provisional note (the orange italic top-right repeats the footer — keep the footer); add a small **height colorbar** to the 3-D massing excerpts so the yellow encoding is defined; add one schematic line "Output = environmental-adequacy deficit surface for in-situ upgrading."
17. **fig03 decoupling (optional, built-env):** if the SVF≠winter-sun **decoupling** is load-bearing in PH Results §5 (it is currently in ED4/fig09), add a compact SVF-vs-winter-sun scatter **inset** to Fig 2 so the "openness ≠ sun" claim doesn't float; otherwise leave in ED.

---

## Handoff back
Export the regenerated PNGs to `outputs/paper_figures/exports/` (same names); brisaverse re-promotes into `shared/figures/` and updates captions. Round-1 analytic JSONs (`compound_spatial_clustering.json`, `fullsample_loso.json`, `lambda_f_neighbourhood.json`) are already consumed in the manuscript. Still **BLOCKED** (needs user): ray-caster vs Radiance/SOLWEIG cross-validation (neither installed locally).

**PI decisions this round surfaces:**
- Full lexical rename of the taxonomy states to "Below … floor (owed)" across the *manuscript body* (not just figure legends)? — recommended by the equity reviewer; held for PI sign-off (the current names are crisp and pervade `main_ph.tex`).
- Whether to pull the SVF–sun decoupling into the Fig 2 main set vs keep in Extended Data.
