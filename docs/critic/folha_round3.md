# Folha de Rua — round 3 critic report (FINAL)

Artefact: same five A3-portrait "Folha de Rua" sheets (Vidigal, Rocinha, Complexo do
Alemão, Rio das Pedras, Maré), rebuilt again tonight (build `2026-09-16`, git
`ad6d581`, confirmed identical on every masthead checked: Vidigal, Rocinha,
Complexo do Alemão, Rio das Pedras, Maré, both at web1200 and A3), plus five
interactive dashboards rebuilt at the same time and captured in a fresh
Playwright audit (`report.json`, 12 page loads, all HTTP 200, zero console
errors, zero failed requests — unchanged from round 2).

Reviewed: contact sheet + all five sheets at web1200 (with pixel-level crops
of hero maps, ridgeline panels, hexbins, legends and scale bars), Maré and
Vidigal at A3, the dashboards contact sheet, `maré_desktop.png`,
`rocinha_tablet.png` and `rocinha_desktop.png`, `vidigal_desktop.png`,
`complexo_do_alemao_desktop.png`, `riodaspedras_desktop.png`,
`landing_desktop.png`, and `report.json`. KPI tiles were cross-checked pixel
by pixel against sheet mastheads for all five sites, not just Maré.

## Scores

| Lens | Round 1 | Round 2 | Round 3 | Evidence |
|---|---|---|---|---|
| L1 — Legibility & layout | 5.5 / 10 | 5.5 / 10 | **7.0 / 10** | Both round-2 L1 blockers are genuinely fixed: the r̄-regime badge now sits on its own row above the ridgeline panels with no collision (checked Vidigal, Rocinha, Complexo do Alemão, Rio das Pedras, Maré at web1200, and Vidigal/Maré at A3 — e.g. Maré "open-sky regime ✓ (r̄=0.62)" reads cleanly, no longer stamped over "SE n=35340"), and the 15 m edge-halo hatch now renders on the boundary of all five sheets, not just Vidigal (confirmed by zoomed crops of every sheet's polygon edge, including the middle of Maré's coastline, not just the tip). Held out of "top-notch" by two things that persist: Maré's hero map still occupies only ~19% of its panel width with the rest blank (round-1 finding 6, still untouched in round 3 — see Residual defects), and a new legibility issue where the small per-class r̄ annotations inside each ridge are low-contrast grey and get clipped at the panel's top edge on sheets with many street classes (worst on Rocinha, 12 classes: "r̄=0.13" for the top "Rua" row is cut by the frame). |
| L2 — Information design | 6.5 / 10 | 4.5 / 10 | **8.6 / 10** | The single most damaging defect in this cycle — the dashboard's "N OBSERVERS" tile undercounting the true total by 53–90% — is fixed and verified byte-exact on all five sites, not just Maré: Vidigal 6,876 = 6,876, Rocinha 35,316 = 35,316 (checked on both `rocinha_desktop.png` and `rocinha_tablet.png`), Complexo do Alemão 47,508 = 47,508, Rio das Pedras 16,905 = 16,905, Maré 84,147 = 84,147, all pixel-read directly from the KPI tile. Each tile now also carries an honest "map shows X of 8,000 sampled points" sub-line distinguishing the true total from the map's rendering sample. The density-metric mismatch (finding F) is fixed the same way: the dashboard's headline `OBSERVERS/KM²` number now has a `685 / road-km`-style sub-line that matches the sheet's `OBS/ROAD-KM` value exactly on every site checked. Landing page site cards still match sheet mastheads exactly for all five sites. Not a 9+ only because the dashboard and sheet still lead with two different primary density metrics (km² vs road-km) even though the numbers now reconcile — a stylistic, not factual, residue. |
| L3 — Cross-sheet consistency | 5.0 / 10 | 6.5 / 10 | **8.0 / 10** | The specific defect holding L3 back in round 2 — edge-halo hatch rendering on only 1 of 5 sheets — is fixed: the hatch is present and consistent (same tick style, same legend swatch) on the boundary of all five. The three round-1 L3 fixes (Vidigal axis, Maré ridgeline titles, colorbar format/position) remain intact. Badge-row placement is now uniform across sites and resolutions. Held back from "top-notch" by a real but data-driven inconsistency: the SVF×solar hexbin panel reads clearly on Vidigal (and at A3, Maré) but stays visibly washed out on Rocinha even after the round-3 normalization fix — the same panel type, same code path, reading very differently between sheets — and by the per-class r̄ label crowding/clipping that scales with each site's street-class count (5 on Vidigal vs 12 on Rocinha), which is a genuine cross-sheet legibility drift even though it is not, strictly, a bug in any one sheet. |

## Round-2 findings status

**A. [blocker → RESOLVED] Dashboard "N OBSERVERS" tile undercounted the true total by 53–90%.**
Fixed and verified on all five sites, not sampled to just Maré/Vidigal as in round 2. Pixel-read directly from each site's KPI tile: Vidigal 6,876 (sheet) = 6,876 (dashboard); Rocinha 35,316 = 35,316 (desktop and tablet both re-checked, resolving an initial low-resolution contact-sheet misread of "35,318"); Complexo do Alemão 47,508 = 47,508; Rio das Pedras 16,905 = 16,905; Maré 84,147 = 84,147. Each tile now shows a second line, "map shows N of 8,000 sampled points" (Vidigal correctly shows "no decimation needed" since its true count is below the map's sampling cap). This was the most damaging finding in round 2 and is now cleanly closed.

**B. [major → RESOLVED] Edge-halo hatch had been disabled on 4 of 5 sheets.**
Fixed. Zoomed crops of the polygon boundary confirm the ticked hatch pattern now renders on Vidigal, Rocinha, Complexo do Alemão, Rio das Pedras, and Maré (checked at the tip and at mid-coastline for Maré specifically, since its boundary is long). One region on Rocinha's hero map initially looked like an anomalously large hatched block far from the boundary; pixel-color sampling showed those "stripes" are actually SVF-colored street lines (dark-navy alleys, RGB matching the colorbar's low-SVF end, not the hatch's neutral grey/black) — a dense real alley cluster, not a rendering defect. Flagged and cleared during this review; no action needed.

**C. [major → RESOLVED] r̄-regime badge collided with the top street-class label on every sheet.**
Fixed cleanly. The aggregate regime badge ("shaded regime ✓ (r̄=…)" / "open-sky regime ✓ (r̄=…)") now sits on its own row directly under the panel title, separate from the per-ridge class-name + n= labels at the right edge and the new per-class r̄ annotations at the left edge. Checked on all five sheets at web1200 and on Vidigal/Maré at A3 — no overlap anywhere. (The fix did introduce a new, much smaller legibility issue with the per-class r̄ labels — see Residual defects.)

**D. [minor → PARTIAL] Hexbin panels read as near-empty for Vidigal and Rocinha.**
Partially fixed. Vidigal's SVF×solar panel is now clearly legible — a visible mid-grey density band along the full diagonal, a real improvement over round 2's near-blank rendering. Rocinha's panel is still functionally near-blank at normal viewing scale: even with post-hoc autocontrast stretching, the density band is barely a shade above white, in visible contrast to Vidigal's much darker rendering right next to it on the same page. The implementer's claim ("Vidigal/Rocinha panels are no longer near-blank") holds for Vidigal only.

**E. [minor → PARTIAL] Unresolved-observer marker dominates Rocinha/Alemão hero maps.**
Cannot confirm the claimed marker-size reduction directly (no round-2 image was retained for a pixel-diff), but the underlying legibility problem persists in round 3: at 8–10x zoom, Rocinha's and Complexo do Alemão's densest unresolved-observer clusters still merge into solid magenta blobs that fully occlude the street network beneath them (checked both sites). Shrinking the glyph reduces overplot only where markers are sparse; in the site's densest unresolved clusters (which is exactly where round 2 flagged the problem) the local marker density alone is enough to saturate the area regardless of individual glyph size.

**F. [nit → RESOLVED] Dashboard density KPI used a different metric than the sheet.**
Fixed via disclosure rather than unification: the dashboard's headline number is still `OBSERVERS/KM²` (a different primary metric from the sheet's `OBS/ROAD-KM`), but every site's tile now carries a second line in the same units and exact value as the sheet (e.g. Maré: dashboard shows "19712" with "685 / road-km" beneath it, matching the sheet's 685 precisely). A reader flipping between sheet and dashboard can now reconcile the numbers directly, which resolves the "same story, different numbers" concern even though the two headline metrics still differ stylistically.

**Carried from round 1 (not addressed by the round-3 claims list):**

- **[major, round-1 #6 → STILL NOT FIXED.]** Maré's hero map still occupies only ~19% of its panel's width, with wide blank margins on both sides, at both web1200 and A3 resolution. This was flagged in round 1, marked "not fixed" in round 2, and is absent from the round-3 implementer's claim list (A–F) — it was not attempted this round.
- **[nit, round-1 #11 → STILL NOT FIXED, deferred.]** No on-sheet key was added to make the r̄-regime badge states ("shaded regime ✓" / "open-sky regime ✓") interpretable without prior context. Still deferred as of round 3.

## Residual defects (ranked, smallest fix first)

1. **[minor] Per-class r̄ annotations are low-contrast and clip at the panel's top edge on sheets with many street classes.** Worst on Rocinha (12 classes: "r̄=0.13" for the top "Rua" row is cut off by the plot frame; several others render in pale grey nearly indistinguishable from the panel background), present but milder on Vidigal (5 classes, top label lightly clipped). Smallest fix: darken the label color and add a fixed top-margin to the ridgeline axes proportional to text height, independent of class count.

2. **[minor] Rocinha's SVF×solar hexbin panel remains near-blank in practice**, despite the round-3 PowerNorm fix working for Vidigal. Smallest fix: apply per-site (not global) normalization, or a minimum-opacity floor, so Rocinha gets the same treatment Vidigal received.

3. **[minor] Unresolved-observer marker still fully occludes the street network in Rocinha's and Complexo do Alemão's densest clusters.** Smallest fix: draw the "SVF==0 (unresolved)" layer beneath the street-SVF line layer (z-order change) rather than relying solely on glyph size.

4. **[major, carried] Maré's hero map wastes ~80% of its panel as blank margin**, unchanged since round 1. Smallest fix: rescale the map to fill the panel's available height for the site's actual aspect ratio instead of applying a fixed width/height box tuned for the other four (wider) sites.

5. **[nit, carried] r̄-regime badge states remain uninterpretable without prior context** — still no on-sheet key. Smallest fix: one line in the existing caveat strip, e.g. "regime check ✓ = |r̄| consistent with SVF-solar coupling within site tolerance."

## Before you read

- The blocker that mattered most in round 2 — the dashboard's headline observer count being wrong by up to 90% — is fixed and independently verified for all five sites, byte-exact against the printed sheets, not just Maré.
- Both round-2 layout blockers (badge/label collision, missing edge-halo hatch) are cleanly fixed across all five sheets; no regressions found elsewhere on the page.
- Maré's hero map still wastes most of its panel width as blank space — this was flagged in round 1, again in round 2, and was not on this round's fix list. It needs its own pass.
- Two of round 2's "minor" findings (hexbin near-blank, unresolved-marker overplot) only partially improved: both problems specifically persist on Rocinha and/or Complexo do Alemão, the same two sites round 2 called out.
- A large hatched-looking block on Rocinha's hero map that initially looked like a new rendering bug turned out, on pixel-level inspection, to be real SVF-colored alley data — flagged and cleared in this review, no action needed, but worth knowing this shape exists if it resurfaces in a future critic's first pass.
