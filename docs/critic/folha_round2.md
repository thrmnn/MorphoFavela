# Folha de Rua — round 2 critic report

Artefact: same five A3-portrait "Folha de Rua" sheets (Vidigal, Rocinha, Complexo do
Alemão, Rio das Pedras, Maré), all rebuilt tonight (build `2026-09-16`, git `5422e51`,
identical across all five — confirmed on every masthead), plus five interactive
Leaflet+Plotly dashboards for the same sites, rebuilt at the same time and captured
in a Playwright audit (`report.json`, 12 page loads, all HTTP 200, zero console
errors, zero failed requests).

Reviewed: contact sheet + all five sheets at web-1200, Maré and Rocinha at A3, the
dashboards contact sheet, `maré_desktop/tablet.png`, `vidigal_desktop.png`,
`landing_desktop.png`, `report.json`, and (to adjudicate the dashboard KPI question)
`rocinha_desktop.png`, `complexo_do_alemao_desktop.png`, `riodaspedras_desktop.png`.

## Scores

| Lens | Round 1 | Round 2 | Evidence |
|---|---|---|---|
| L1 — Legibility & layout | 5.5 / 10 | 5.5 / 10 | Unchanged, not for lack of work: finding 1's clipped right-hand labels/badge are genuinely inside the page margin now on all five sheets, and Maré's colorbar is off the map — but the label fix immediately collides the r̄-regime badge into the top street-class label on **all five** sheets (new finding C), and the edge-halo hatch that finding 10 complained about being *too much* on Rio das Pedras is now *absent* on four of five sheets even though the legend/caveat still describe it (new finding B). Net: one blocker relocated, one major inconsistency traded for another. |
| L2 — Information design | 6.5 / 10 | 4.5 / 10 | Drops below the bare-minimum line. The KPI mislabel is fixed ("R(SVF, SOL)") and Maré's ridgeline titles now say what they plot — but the round-2 brief's specific instruction to compare masthead KPIs between sheet and dashboard surfaces a severe defect: the interactive dashboard's headline "N OBSERVERS" tile undercounts the true total by 53–90% on four of five sites (Maré: 7,997 shown vs 84,147 on the sheet and the dashboard's own landing card and body text) — new finding A, the most damaging thing in this report. |
| L3 — Cross-sheet consistency | 5.0 / 10 | 6.5 / 10 | Real improvement: all three round-1 L3 defects are fixed — Vidigal is on the current 0–12 solar-hours/day axis, Maré's ridgeline titles now match the quadrant scheme, and the SVF colorbar format/position is uniform (1-decimal ticks, top-left) across all five sheets. Held back from "good" by a new cross-sheet inconsistency: the edge-halo hatch now renders on only 1 of 5 sheets (Vidigal), not all five. |

## Round-1 findings status

1. **[blocker] Right-hand ridgeline labels/badge clipped by page margin — PARTIALLY FIXED.** The clipping itself is gone: `Rua n=5716`-style labels and the r̄-regime badge sit fully inside the axes on all five sheets at both web1200 and A3 resolution (checked all five). But the fix introduced a new collision (see new finding C): the badge now overlaps the top row's class label on every sheet, e.g. Alemão "open-sky regim[e]" is smashed into "Rua n=38902", Maré "open-sky regime ✓ (r̄=0.62)" into "SE n=35340". A reader still cannot read the top row cleanly — the failure mode moved, the net legibility did not improve.

2. **[major] Maré hero colorbar drawn over the street network — FIXED.** Colorbar is now anchored top-left in empty margin space, clear of the map, on both the web1200 and A3 renders. No street network or observer markers are obscured.

3. **[major] Maré ridgeline panels titled "by street class" but plotting compass quadrants — FIXED.** Both panel titles now read "SVF distribution by orientation quadrant" / "Annual solar access by orientation quadrant", matching the SE/NW/SW/NE categories actually plotted. Confirmed at web1200 and A3.

4. **[major] Vidigal stuck on stale solar-hours/year axis — FIXED.** Vidigal's ridgeline and hexbin panels are now on the current "solar hours / day (annual mean)" 0–12 axis, matching the other four sheets. Vidigal shares the same build (`2026-09-16`) and git sha (`5422e51`) as the rest of the set, confirming a genuine full rebuild rather than a partial patch.

5. **[major] Vidigal and Rocinha both printed "Folha 02/05" — FIXED.** All five sheets now carry distinct, correct numbers: Vidigal 01/05, Rocinha 02/05, Complexo do Alemão 03/05, Rio das Pedras 04/05, Maré 05/05.

6. **[major] Maré hero panel wastes roughly half its width as blank margin — NOT FIXED (implementer's "partial" overstates it).** At both web1200 and A3 resolution the map still occupies only ~19% of the panel width, with ~40% blank on each side — visually indistinguishable from the round-1 screenshot. Freeing up the colorbar's old position (finding 2) didn't translate into a bigger map; the map was not rescaled to use the panel's available height/width for elongated sites, which is exactly the fix round 1 asked for.

7. **[minor] SVF colorbar tick precision/position inconsistent across sheets — FIXED.** All five sheets now use the same 1-decimal "0.0 / 0.5 / 1.0" tick format anchored top-left of the hero map.

8. **[minor] KPI tile "P(SVF, SOL_H)" reads as a probability — FIXED.** Relabeled "R(SVF, SOL)" on all five sheets, consistent with the Pearson-r language used in the hexbin panel.

9. **[minor] Cross-site strip site-name labels sit at different heights — FIXED.** All five site-name labels above the small-multiples strip now sit on one common baseline (checked on Maré's strip at 3x zoom; the strip layout code is shared across sheets).

10. **[minor] Rio das Pedras edge-halo hatch covers too much of the narrow polygon — FIXED, but via overcorrection that created new finding B.** The hatch no longer visually dominates Rio das Pedras — because it no longer renders there at all, nor on Rocinha, Alemão, or Maré. Only Vidigal still shows it. The original complaint (too much hatch) is technically resolved, but see new finding B for why this is not a clean win.

11. **[nit] r̄ badge states ("check" / "shaded regime ✓" / "open-sky regime ✓" / "latitude consistent ✓") uninterpretable without a key — NOT FIXED (deferred, as stated).** No on-sheet key was added. This is now compounded by new finding C: the badge text is also colliding with the class label, so it's simultaneously uninterpretable *and* harder to read than before.

## New findings (ranked by cost to reader)

**A. [blocker] Interactive dashboards — "N OBSERVERS" masthead KPI is wrong on 4 of 5 sites.**
Sheet vs. dashboard, `N OBSERVERS` tile:
- Vidigal: 6,876 (sheet) vs 6,871 (dashboard) — Δ −0.07%, negligible.
- Rocinha: 35,316 vs 7,998 — Δ −77%.
- Complexo do Alemão: 47,508 vs 7,999 — Δ −83%.
- Rio das Pedras: 16,905 vs 7,974 — Δ −53%.
- Maré: 84,147 vs 7,997 — Δ −90%.

All four wrong values cluster tightly around 7,974–7,999, and the Maré tablet page explains why, in its own map caption: *"We show 8,000 of 84,147 observers to keep the map fast — picked evenly across the SVF range so the rare dark and bright extremes are not lost."* The masthead KPI tile is reading the map's performance-sampled marker count instead of the true dataset total already used correctly elsewhere on the same page (the descriptive paragraph text, e.g. "84,147 observers on 122.9 km of road"), on the landing page's site card (which matches the sheet exactly for all five sites), and on the printed sheet itself. Everything downstream of the true total — mean SVF, edge-observer %, sky↔sun coupling — is computed correctly; only this one tile is mis-wired. This is precisely the failure mode the round-2 brief asked to check for Maré, and it turns out to affect four of five sites, at the single most prominent number on the page. Smallest fix: bind the `N OBSERVERS` tile to the same full-dataset count already used in the page's own prose, not to the map's rendering sample.

**B. [major] Edge-halo hatch has been disabled on 4 of 5 sheets, contradicting the legend and caveat text.**
The "15 m edge-halo zone" legend swatch and the `[H1] EDGE HALO` caveat text ("*Affected zone is hatched on the hero map*") appear unchanged on all five sheets — but the hatch itself only renders on Vidigal (checked at 3–4x zoom on the polygon boundary for all five: Rocinha, Complexo do Alemão, Rio das Pedras, and Maré show a plain solid boundary line with no hatch anywhere). This reads as an overcorrection of round-1 finding 10 (hatch density on Rio das Pedras) that silently killed the layer everywhere except Vidigal, rather than reducing its opacity as round 1 suggested. A reader relying on the caveat text to gauge the edge-affected zone on four of five sheets will look for a hatch pattern that isn't there. Smallest fix: restore the hatch on the four affected sheets, then address density with opacity/line-spacing rather than an on/off toggle.

**C. [major] The r̄-regime badge now collides with the top street-class label on every sheet.**
A direct side effect of fixing finding 1. On all five "Annual solar access by street class/orientation quadrant" panels, the badge text ("shaded regime ✓ (r̄=…)", "open-sky regime ✓ (r̄=…)") is drawn at the same position as the first ridge's class-name + n= label, overlapping character-for-character (confirmed on Vidigal, Rocinha, Alemão, Rio das Pedras, Maré — worst on Alemão and Maré where both texts are fully superimposed). The reader loses both pieces of information for the most common street class. Smallest fix: give the badge its own row (e.g. directly under the panel title) instead of anchoring it inline at the top ridge's y-position.

**D. [minor] Hexbin "SVF × solar" panels read as near-empty for Vidigal and Rocinha — legibility, not a bug.**
Checked: both panels are genuinely rendering (r=0.90 Vidigal, r=0.93 Rocinha, GMM 2-mode markers plotted correctly), but the shared, fixed hexbin color normalization doesn't compensate for per-site sample size or SVF range, so Vidigal (smallest N, 6,876) and especially Rocinha (largest of the two shown but has the lowest mean SVF, 0.170, so its data cluster tightly near the origin) both look close to blank against the shared [0,1]×[0,12] domain — Rocinha visibly emptier than Vidigal despite 5x the observations. A reader cannot currently distinguish "sparse but real" from "broken" at print/thumbnail scale. Smallest fix, rendering-only: a minimum-opacity floor or per-site max-count normalization for the hexbin fill.

**E. [minor] Rocinha and Complexo do Alemão hero maps are dominated by the "SVF == 0 (unresolved)" marker, not the low-confidence-offset marker the round-2 brief hypothesized.**
Checked the legend precisely: the magenta "+" swarm is `SVF == 0 (unresolved)` (Rocinha n=4,373 = 12.4% of observers; Alemão n=1,876 = 3.9%), while `offset > 2.5 m (low-confidence)` is a separate, barely-visible hollow blue circle (Rocinha n=615, Alemão n=163). Rio das Pedras and Maré have proportionally fewer unresolved observers (1.2% and 0.5%) and their street-SVF coloring stays legible. This is a legitimate site pattern (Rocinha/Alemão really do have more unresolved SVF), but the marker's glyph size/opacity means even a single-digit-to-low-teens percentage overplots and hides most of the street network underneath it. Smallest fix: shrink the unresolved-observer glyph and/or lower its opacity, or draw it beneath the street-SVF line layer.

**F. [nit] Dashboard's secondary density KPI uses a different metric than the sheet's.**
Dashboard shows `OBSERVERS / KM²` (e.g. Maré 19,712); sheet shows `OBS / ROAD-KM` (e.g. Maré 685). Not necessarily wrong on its own, but sitting next to finding A it compounds the "same story, different numbers" impression when a reader flips between the two artifacts for the same site.

## What works now

- All five sheets are a genuine same-night, same-sha rebuild (`2026-09-16` / `5422e51`) — the round-1 staleness problem (Vidigal on an old axis and an old sheet number) is fully gone, and this is now auditable at a glance from the masthead.
- The three round-1 cross-sheet consistency defects (Vidigal's stale axis, Maré's mislabeled ridgeline titles, colorbar tick format/position) are all fixed and stayed fixed under zoom.
- Maré's colorbar no longer sits on top of its own street network — the single clearest major fix in this round.
- The landing page's site cards match the printed sheets' masthead numbers exactly for all five sites (N_OBS, road-km, mean SVF) — the one part of the "same story, same numbers" requirement that fully holds up.
- The four-box caveat strip, the GMM bimodality annotation, the cross-site small-multiples strip (now with a common label baseline), and the top-down sheet narrative are all still intact and consistent across sites, as in round 1.
- The dashboards themselves load cleanly: 12/12 page loads at HTTP 200 with zero console errors and zero failed requests across desktop and tablet viewports.
