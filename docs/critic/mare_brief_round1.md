# Maré morphology brief — critic report, round 1

Reviewed: 8-page PDF draft (`sheet.png` + `page-1.png`…`page-8.png`) against
`mare_morphology_brief.src.md`, `mare_brief_spec.md`, and `mare_numbers.json`.

## Scores

| Lens | Score | Evidence |
|---|---:|---|
| L1 — Audience fit and concision | 7.5/10 | All four reader questions (what exists / resolution / what's shareable / how to engage) are answered and map cleanly onto sections (p.1 Purpose+at-a-glance, p.2 Data inventory, p.8 Availability and terms + contact), but half-empty pages (p.1, p.2, p.4, p.6) make an 8-page document feel padded to flip through, and the two different "mean building height" figures (p.1 vs p.3) cost a confused re-check. |
| L2 — Visual presentation | 5.5/10 | Four of eight pages (1, 2, 4, 6) are 45–65% blank below the last line of body text — p.6 ("Geometry-derived ventilation potential") has two paragraphs of text and nothing else on the rest of the page — which is the rubric's explicit defect; Figure Two's four histogram panels (p.4) are small with sub-print-legible axis ticks. Maps (p.3, p.5), the wind rose (p.7), and the single accent colour are handled well. |
| L3 — Honesty, traceability, disclosure | 7.2/10 | Numbers are internally consistent (dominant wind sector SE/26% matches p.1 and p.6; 43,419 cells / 67.3% built matches p.1 and p.3; morphotype shares sum to ~100% on p.8; wind rose is genuinely 8-sector, matching the "eight-sector rose" label on p.2), caveats are clearly surfaced (roughness "no absolute value reported" p.7, CFD "parked companion track, no results" p.7), and no banned wording or ranking was found — but the figure sequence silently skips "Figure Four" (Three on p.5 jumps to Five on p.7), which reads as a dropped figure or build defect to a careful reviewer. |

## Findings (ranked by cost to the reader)

1. **Missing Figure Four in the sequence.** Pages: 5 and 7. The document has captions "Figure One" (p.3), "Figure Two" (p.4), "Figure Three" (p.5), then jumps straight to "Figure Five" (p.7) — there is no "Figure Four" anywhere, confirmed in the template (`mare_morphology_brief.src.md`, figure captions at lines 62, 87, 100, 141: the fourth figure is literally labelled "Figure Five"). A reader will read this as a figure silently dropped from the build. Fixed = rename the caption at template line 141 from "Figure Five" to "Figure Four" (it is the fourth figure in the document; no in-text cross-reference to renumber). Severity: **major**.

2. **Systemic under-filled pages.** Pages: 1, 2, 4, 6. Page 6 is the worst case — two paragraphs of text occupy the top third of the page, the remaining ~65% is blank white space before the footer; pages 1, 2 and 4 each leave roughly half the page blank below the last paragraph/figure. This is the rubric's named defect ("half-empty pages are a defect") and it recurs on half the document. Fixed = stop forcing a fresh page per section; let short sections (e.g. "Geometry-derived ventilation potential") flow onto the same page as the section before or after it, so the 8-page budget is spent on content rather than on air. Severity: **major**.

3. **Figure Two histograms are too small to read at print size.** Page: 4. Four histogram panels (λp, mean height, SVF-grid, SVF-segments) are packed into one narrow strip roughly 2 inches tall with tick-label text well under readable size at A4 print resolution — this fails the rubric's explicit histogram-legibility check. Fixed = enlarge the figure (e.g. a 2×2 grid using the blank space already available below it on the same page) and increase axis tick-label size. Severity: **major**.

4. **Two different "mean building height" numbers, unreconciled.** Pages: 1 and 3. Page 1's at-a-glance table states "Mean building height: 7.1 m" (`mare_mean_H_m`); page 3 states "median mean height is 6.93 m (IQR 5.28–8.61 m)" restricted to built cells (`mare_H_mean_median`). Both are real, distinct, correctly-sourced statistics (mean vs. median), but nothing on page 1 signals that the at-a-glance figure is a mean while the body figure is a median — a reader comparing the two in the two-minute skim will see what looks like an unexplained discrepancy. Fixed = relabel the at-a-glance row, e.g. "Mean building height (grid mean, built cells)", so the two numbers are visibly different statistics rather than apparently-conflicting restatements. Severity: **major**.

5. **Spelled-out figure numbers.** Pages: 3, 4, 5, 7 ("Figure One," "Figure Two," "Figure Three," "Figure Five"). This is a direct, acknowledged artifact of the "no typed digits outside placeholders" rule (spec test 6b) applied to a structural label rather than a data value, and it reads oddly in an otherwise plain-register report. Fixed = carve out a narrow exception in the digit-scan lint for figure/table ordinals (they are not measured numbers) so captions can read "Figure 1." Severity: **minor**.

6. **Data inventory table column cramping.** Page: 2. The "Derived or restricted" header wraps across two lines while neighbouring columns don't, and cell text (e.g. "per street sample point (84,147 points)") wraps to three lines against two elsewhere in the same row — a small rough edge in the brief's single most-consulted table. Fixed = widen the "Derived or restricted" column slightly (it can borrow space from the mostly-empty page). Severity: **minor**.

7. **Morphotype table (page 8) uses under half the page width.** Not harmful on its own, but visible immediately next to the whitespace problem (finding 2) and worth fixing in the same pass if page layout is touched. Severity: **nit**.

## What already works (preserve these)

- **Consistent single-accent, no-basemap map style.** Figures One, Three and the wind rose (pages 3, 5, 7) use one blue accent, 4–5 discrete classes, scale bar + north arrow only, no coordinate ticks — exactly the spec's map constraint, applied uniformly.
- **Caveats are surfaced, not buried.** The roughness-length paragraph (p.7) explicitly states "No absolute roughness value is reported here for that reason," and the CFD status is stated plainly as "a parked companion track with no results to report at this time" — measured vs. parked is unambiguous.
- **No ranking, no banned wording.** Page 8's "this section positions Maré within that typology; it is not a ranking of the five sites" and the absence of "WHO"/"deficit"/"formal"/"flow" anywhere in the eight pages show the disclosure constraints were respected, not just checked mechanically.
- **Internal number consistency where it matters most.** Cell counts (43,419 / 67.3% built), the dominant wind sector (SE, 26%), and the morphotype shares (~100% total) agree wherever they recur across pages — the underlying `mare_numbers.json` pipeline is doing its job.
- **The interpretive sentence after the built-form numbers (p.4)** — "The low all-cell λp median alongside the low built-cell porosity reflects Maré's structure..." — turns a potentially confusing statistic (IQR spanning almost the whole 0–1 range) into a legible explanation; this pattern of number-then-interpretation should be kept wherever a raw statistic could otherwise look odd.
