# Folha4 council — Maré hero-map sheet — 2026-09-24

Branch `cyc4/folha4`, merged into `main` 2026-09-24 (not pushed). Covers the
Folha (press-facing) Maré sheet: SVF street-level hero map + irradiation
distribution panels (A vs E definitions), print (A3, hero / no-hero variants)
and interactive twin.

## What changed, round by round

- **Round 1** (`0a1f09f`, `deead28`) — first cut of the Maré distribution
  sheet: horizontal rotation, distribution core (panels 1–3), two hero
  variants (with/without the SVF street map), compute/draw split for reuse
  across sites.
- **Round 2 fixes** (`97a77a6`, `1b506b5`, `5569c62`) — inset-corner collision
  fixed, spacer between panels 1/2 and panel 3 made fixed-ratio (was
  variant-scaled, left a dead-space gap), SVF glossed on the hero map.
- **Round 2 → 3 wording pass** (`4a661e0`) — `build_html_dashboard.py`:
  "favelas" replaces "informal settlements" per the PI's usage; flagged the
  round-3 blocking item (no A/E spatial overlay) for the next round.
- **Round 3 locks** (`68b91fd`) — glossary and top/bottom split contracts
  locked in tests (`test_render_mare_irradiation_distributions.py`,
  `test_site_dashboard.py`).

## Round 3 — scores per seat

Average: **7.9 / 10**

| Seat | Overall | Structure | Use of space | Legibility | Restraint | Honesty | Decision value | Recommended variant |
|---|---|---|---|---|---|---|---|---|
| 1 | 8.3 | 8.5 | 8 | 8 | 8.5 | 9 | 8.5 | hero |
| 2 | 7.8 | 8 | 8 | 8 | 8 | 8.5 | 7.5 | hero |
| 3 | 7.6 | 8 | 8 | 7 | 8 | 9 | 7 | hero |

All three seats independently converge on **hero as the PI-facing default**,
no-hero kept as a secondary compact/print-efficient sheet, not co-equal.
Seat 2's `variant_verdict`: the hero SVF map is what anchors the abstract A/E
distribution curves to Maré's actual physical form (dense low-rise grid,
coastline, 15 Redes da Maré communities) — without it the sheet is numbers
with no shape. Seat 2 also verified the round-3 dead-space fix (panel 1/2 →
panel 3 transition) at full resolution in both variants: no dead space, no
title/tick-line collision.

Verified clean per the PI's 2026-09-24 ruling: "favelas" replaces "informal
settlements" on both the print sheets and the interactive twin (zero hits for
"informal settlement" in `outputs/_distribution/html_dashboards`); L1
no-deficit framing holds (panel 3's "listed north to south, not ranked"
ordering, unlabeled aggregate box).

## Blocking items (must fix before PI sign-off)

1. **No A/E spatial overlay** (seat 2). Panels 1–2 compare definition A (6 IPP
   favela polygons, blue) against E (IPP complex outline, orange) only
   statistically (irradiation KDE, decile-share bars); the hero map's SVF
   layer shows a single study-area outline, never both boundaries together.
   For a morphology read this is the one thing missing — no page shows where
   A and E actually diverge on the ground (relevant to the Marcílio Dias vs.
   bairro edge case, note [H3]). Recommended fix: a small inset map, A's
   polygons outlined in blue over E's outline in orange, same style as the
   existing hero SVF map, ideally on the same page as the hero map.

2. **Interactive twin's Leaflet basemap is broken** (seats 2 and 3, same
   defect, flagged independently). The CARTO Positron tiles render as grey
   "API KEY REQUIRED — carto.com/basemaps/apikey" watermark tiles; the
   "street, mapped" panel loses all street/building/coastline context, only
   colored observer dots float on blank tiles. Confirmed pre-existing (not
   touched by round 3's diff — round 3 touched only
   `render_mare_irradiation_distributions.py`, `build_site_dashboard.py`,
   `build_html_dashboard.py` wording, and tests). Not blocking *this round's*
   sign-off per seat 2, but seat 3 marks it blocking for showing the page to
   Redes da Maré: it reads as broken/unfinished software, not as a
   respectful community-facing artifact. Root cause: CARTO deprecated free
   anonymous basemap tiles. Fix before anyone is pointed at the live
   interactive dashboard — point at a keyless tile source (OSM raster, or
   CARTO's free anonymous endpoint if still live) or self-host tiles.

3. **No-hero print variant never spells out "SVF"** (seat 3). Round 3 closed
   the "SVF never expanded" item only on the hero variant (map title +
   colorbar caption); the no-hero sheet has no map, so the fix never reached
   it — its footer still reads "[M3/L1] MEAN SVF · OFFSET" with SVF used but
   never glossed on that sheet. Fix: expand SVF in the shared glossary
   footer line itself, independent of whether the hero map is present, so
   both variants are self-contained.

## Non-blocking improvements (flagged, not gating this round)

- Hero map has no swatch legend for its own layers (grey building
  footprints, thin purple community-boundary outlines, pale yellow-green
  unsampled reference roads vs. viridis-ramped sampled streets) — only the
  SVF colorbar is keyed. Add a compact 4–5 line swatch legend under the
  north-arrow/scale-bar block.
- Unsampled reference roads (~RGB 248,252,203) sit at the same hue as
  viridis's high-SVF end; low saturation keeps them distinct in practice,
  but an explicit "unsampled network" legend label would remove the
  residual misread risk.
- No locator/context inset placing Maré within Rio as a whole — low cost
  given round 2 already built inset-corner placement logic.
- Footer glossary/caveat block (5 columns of paragraph text at print size)
  is dense at true A3 handling distance; consider tightening line length or
  adding line-height.
- SVF colorbar caption paraphrases the abbreviation ("share of open sky
  seen from the street") rather than spelling out "Sky View Factor (SVF)" —
  do both, for a reader landing mid-page via a shared crop.
- Panel 2 is L1-compliant (descriptive only) but, because this is a
  press-facing hero panel, consider one neutral caption line preempting a
  "more shade = more deprived" misreading, e.g. "shade reflects built
  density and street width, not assessed here as advantage or
  disadvantage."
- Header/subtitle leads with internal IDs a community reader can't parse
  ("WP-05", "P1 f1"); "IPP" is never expanded anywhere on the sheet. One
  clause fixes it, e.g. "...E (the city planning institute IPP's complex
  outline...)".
- Interactive twin's plain-language callouts ("what this means for the
  reader", "what this dashboard does not see") are strong, honest,
  non-defensive writing — worth porting into the print sheet's footer boxes
  (H1–H3, M2, M3/L1), which currently stay in registry-note register.
- Panel 3's "listed north to south, not ranked" framing and the unlabeled
  grey "between communities" aggregate box are deliberate, well-executed
  L1-compliant choices — worth standardizing across the other four P1 site
  sheets when they get this same treatment.
- `_pick_inset_corner()` has only been exercised on Maré; test it against
  the other four P1 sites (hillside/canyon: Vidigal, Rocinha, Alemão; flat
  alley grid: Rio das Pedras) before reusing this hero-map layout — Maré's
  coastline geometry may not be representative.

## Hero-map verdict

**Hero is the PI-facing default**, unanimous across all three seats
(`recommended_variant` / `variant_verdict` all say hero). No-hero remains a
legitimate secondary compact/print-efficient sheet for a reader already
oriented (e.g. repeat viewing), not the sheet the PI opens first.

## Status

Merged into `main` locally 2026-09-24, not pushed. Blocking items 1–3 above
are not yet fixed in this merge — this document records the round-3 council
verdict as delivered; remediation is separate follow-up work.
