# Folha de Rua — round 1 critic report

Artefact: five A3-portrait "Folha de Rua" sheets (Vidigal, Rocinha, Complexo do Alemão,
Rio das Pedras, Maré) built by `scripts/build_site_dashboard.py`. Reviewed at web-1200
resolution for all five, plus the Maré A3 render for fine detail, plus the round-1
contact sheet.

Provenance at review time: Vidigal, Complexo do Alemão, Rio das Pedras — build
`2026-06-05`, sha `1bffc3e`. Rocinha — build `2026-07-02`, sha `eeba7e3`. Maré — build
`2026-09-16`, sha `e250d3b` (rendered minutes before this review). Source data
(`svf_streets_solar.gpkg`) changed `2026-06-13/14`.

## Scores

| Lens | Score | Evidence |
|---|---|---|
| L1 — Legibility & layout | 5.5 / 10 | The right-hand street-class labels and n= counts on the solar-ridgeline panel are clipped by the page margin on **all five** sheets (e.g. Vidigal: "Rua n", "Av n=", "Camin" truncated); Maré's SVF colorbar is drawn directly on top of the street network at the top of its own hero map (confirmed at both web1200 and A3 resolution); Maré's hero map occupies only the centre third of its panel width, leaving roughly half the panel blank on each side. |
| L2 — Information design | 6.5 / 10 | The top-down story (identity → hero map → distributions → SVF×solar consistency → cross-site strip → caveats) is intact and identical across all five sheets, and build date + git sha are printed on every masthead, which is the right instinct for a refresh where staleness matters; but the KPI tile labelled "P(SVF, SOL_H)" reads as a probability when it is the Pearson r already shown as "r = 0.88" in the hexbin panel, and the solar-ridgeline "check / shaded regime ✓ / open-sky regime ✓" badge has three possible readings with no on-sheet key. |
| L3 — Cross-sheet consistency | 5.0 / 10 | Vidigal's solar-ridgeline and hexbin panels are still on the old "solar hours / year" (0–4000) axis while the other four sheets use the current "solar hours / day (annual mean)" (0–12) axis (`draw_solar_ridgeline`/`draw_hexbin` in the current script only ever produce the day-mean axis, so this is a stale render, not a legitimate site difference); Maré's ridgeline panels are titled "by street class" but show compass quadrants (SE/NW/SW/NE) instead of Rua/Travessa/Beco/…; the SVF colorbar's tick precision and screen position differ across three variants (Vidigal/Alemão, Rio das Pedras, Maré). |

## Findings (ranked by cost to reader)

1. **[blocker] All five sheets — solar-ridgeline panel.** The right-hand street-class labels/n-counts and the r̄ status badge text ("check (r̄=0.37)" on Vidigal, "latitude consistent ✓ (r̄=0.40)" on Alemão, "shaded regime ✓" on Rocinha/Rio das Pedras) are cut off by the page's right margin on every sheet — confirmed by cropping the panel edge on all five. What "fixed" looks like: pull the right-hand label column and the badge text inside the axes bounds, or widen the panel's right margin so nothing sits past the printable area — smallest change is shrinking/repositioning the duplicate right-axis class labels, since the left-axis labels already carry the same information.

2. **[major] Maré — hero map.** The SVF colour-bar legend is drawn on top of the site's own street network/building fabric at the top of the map (a bright SVF-coloured street loop and a magenta observer marker are visible emerging from behind the colorbar box, confirmed at A3 resolution). What "fixed" looks like: anchor the colorbar to a fixed position in the panel's known-empty margin (or to the panel corner) instead of positioning it relative to the site bounding box, which collapses to a narrow column for elongated sites like Maré.

3. **[major] Maré — ridgeline panels.** Panels titled "SVF distribution by street class" / "Annual solar access by street class" show SE/NW/SW/NE (compass quadrants) instead of the Rua/Travessa/Beco/Estrada/… vocabulary used on the other four sheets — this is the documented compass-quadrant fallback in `draw_solar_ridgeline` when `tipo_logra` isn't available, but nothing on the sheet tells the reader the classification scheme changed. What "fixed" looks like: relabel the two panel titles for Maré to "… by orientation quadrant" so the heading matches what's actually plotted, and add one line to the caveat strip noting street-type classes weren't available for this site.

4. **[major] Vidigal — solar-ridgeline + hexbin panels.** Both panels are rendered on the old "solar hours / year" (0–4000) axis, leaving the hexbin looking almost empty, while the current script (`draw_solar_ridgeline`, `draw_hexbin`) always produces "solar hours / day (annual mean)" (0–12), matching the other four sheets. This is a stale pre-refresh render, not a legitimate site difference. What "fixed" looks like: rebuild Vidigal tonight with the other 2026-06-05 sheets and confirm the axis matches Alemão/Rio das Pedras/Rocinha/Maré.

5. **[major] Vidigal + Rocinha — masthead.** Both sheets print "Folha 02/05"; no sheet in the set shows "01/05". The current `SHEET_NUMBER` mapping in `src/viz/folha/sites.py` already assigns Vidigal `"01"`, so Vidigal's PNG is simply stale. What "fixed" looks like: rebuild Vidigal and confirm the masthead reads "Folha 01/05" before reprinting.

6. **[major] Maré — hero panel whitespace.** The tall N–S island shape leaves roughly half the hero panel's width empty on both sides, wasting the sheet's most prominent visual real estate (confirmed at both resolutions — the map occupies a narrow central band against a panel sized for the wider hillside sites). What "fixed" looks like: let the hero map scale to the panel's available height for elongated sites (bigger map, same panel) rather than a fixed width-driven scale — smallest change, no new panels needed.

7. **[minor] Cross-sheet — SVF colorbar.** Tick-label precision and screen position vary: Vidigal/Alemão use "0.0–1.0" (1 decimal) top-left of the hero map; Rio das Pedras uses "0.00–1.00" (2 decimals) top-right near the locator inset; Maré shows only two ticks ("0.0", "0.5") because the colorbar is squeezed against the narrow map bbox. What "fixed" looks like: fix the tick formatter (one decimal, matching Vidigal) and anchor position across all five calls to `draw_hero_map`.

8. **[minor] All sheets — KPI row.** The sixth KPI tile is labelled "P(SVF, SOL_H)" but the value is the Pearson correlation coefficient — the same number appears in the hexbin panel below as "r = 0.88". The "P(...)" notation reads as a probability to a reader who hasn't seen the hexbin yet. What "fixed" looks like: relabel the tile "r(SVF, sol)" or "Pearson r".

9. **[minor] All sheets — cross-site small-multiples strip.** Site-name labels above each thumbnail sit at different heights ("Vidigal" lower, "Rio das Pedras"/"Maré" higher) because thumbnail box heights vary with each site's aspect ratio, producing a ragged label baseline across the row. What "fixed" looks like: fix a common top-of-box y-coordinate for all five thumbnails so the title text sits on one baseline regardless of box height.

10. **[minor] Rio das Pedras — hero map.** The 15 m edge-halo hatch covers a large fraction of the site's narrow polygon, obscuring much of the street network underneath it. What "fixed" looks like: reduce hatch density/opacity when the halo covers more than some threshold share of the mapped area (Rio das Pedras is 11.5% edge share by length but the hatch reads as covering far more of the visible polygon because the site is narrow).

11. **[nit] All sheets — solar-ridgeline r̄ badge.** The badge can read "check", "shaded regime ✓", "open-sky regime ✓", or "latitude consistent ✓" in three different colours, with the logic only documented in code comments, not on the sheet. What "fixed" looks like: one short clause in the caveat strip explaining the three states (this is separate from finding 1, which is about the badge text being clipped — this is about it being uninterpretable even when whole).

## What already works

- The panel sequence and top-down narrative (identity → hero map → SVF/solar distributions → SVF×solar consistency check → cross-site strip → caveats) is identical across all five sheets — a reader who learns one sheet can navigate any of the other four.
- Build date and git sha are printed on every masthead, which is exactly the right instinct for a refresh cycle where staleness needs to be auditable at a glance (once the sheet-numbering bug in finding 5 is fixed).
- The four-box caveat strip ([H1] edge halo, [H2] observer density, [M2] solar units, [M3/L1] mean SVF · offset) is present, consistently worded, and cites the technical report section on all five sheets — good discipline for a technical audience.
- The cross-site small-multiples strip correctly highlights the current site with an orange frame on every sheet, including Maré's oddly-shaped thumbnail.
- The GMM bimodality annotation ("2 modes — canyon / open") is genuinely data-driven (it's suppressed when \|r\| > 0.9, per the code), not decorative — its absence on Rocinha/Rio das Pedras/Maré is a legitimate result, not an inconsistency to fix.
