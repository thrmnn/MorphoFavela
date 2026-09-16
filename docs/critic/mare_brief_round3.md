# Maré morphology brief — critic report, round 3 (FINAL)

Reviewed: the 6-page v3 PDF (`sheet.png` + `page-1.png`…`page-6.png`)
against `docs/critic/mare_brief_round2.md`, the round-3 implementer's own
account, `docs/mare_brief_spec.md`, `docs/mare_brief_v2_spec.md`, and
`docs/briefs/mare/mare_morphology_brief.src.md`. Blank-space figures are
pixel-row measurements (last non-white content row of the body, excluding
the header page-number band and the footer band, vs. the body's usable
height), reproduced with a script against each `page-N.png`, not eyeballed.

Measured blank-below-content, this round: p.1 3.0%, p.2 15.2%, p.3 26.1%,
p.4 27.0%, p.5 4.2%, p.6 71.7%.

No further fix round follows this cycle — this is the number the PI reads
alongside the PDF.

## Scores

| Lens | Round 1 | Round 2 | Round 3 | Evidence (round 3) |
|---|---:|---:|---:|---|
| L1 — Audience fit and concision | 7.5/10 | 8.0/10 | 8.2/10 | The four reader questions still map cleanly to sections, and the round-2 defect that cost a page-flip mid-read — the fabric-cluster table stranded across the p.5/p.6 break — is gone: the intro sentence ("Maré's composition:") and the table are now one unbroken block on p.5. The brief still reads start-to-finish without a forced re-check; the residual cost is page 6 opening cold with only "Availability and terms," which is a presentation issue (L2) more than a concision one. |
| L2 — Visual presentation | 5.5/10 | 6.5/10 | 6.8/10 | Real gain: the table+lead-in split (round 2's named defect #1) is fixed by construction, not just observed to look better — same page, no break between them. The wind rose (p.5) is visibly smaller: the rose block (title+plot+caption) now runs 69→432px of an ~836px used page (≈43%), down from round 2's ≈60%-of-page estimate, and p.5 itself is 4.2% blank — well filled. But pages 3 and 4 are unchanged in kind: 26.1% and 27.0% blank below their last paragraph, same root cause as round 1/2 (`page-break-inside: avoid` on the street-segment SVF figure pushes it whole to the next page, stranding the tail of the page it left). Page 6 is now 71.7% blank — worse in degree than round 2's p.6 (47%) — because the composition passage that used to anchor p.6 moved to p.5 to fix the table split, leaving only "Availability and terms" and "Methods notes and references" on the last page. Two things pull this off the floor rather than into "bare minimum": (a) v2's own spec text sets the bar as "no page less than about 60% filled **except the last**" — p.3 (73.9% filled) and p.4 (73.0% filled) clear that bar outright, and p.6's blankness is the spec's one named exemption, not a rule violation; (b) the fix that caused p.6's growth removed a worse defect (a table separated from its own lead-in). It still costs a point: a PDF whose last impression is two-thirds white space reads as unfinished to a reader who does not know the spec's exemption, and the underlying page-break mechanism for p.3/p.4 (round 1 finding 2) remains architecturally unaddressed for a fourth round running. |
| L3 — Honesty, traceability, disclosure | 7.2/10 | 7.9/10 | 7.9/10 | Unchanged in substance from round 2 — no content, numbers, or disclosure decisions moved this round, only layout. Re-verified rather than re-scored on new evidence: the letter-cluster table (A–F, shares 7.54%/3.65%/1.16%/0.298%/11.8%/75.6%) on p.5 still tracks its narrative ("dominated by one cluster... near its maximum" = F at 75.6%; "second cluster, common across all five campaign sites" = E at 11.8%); `grep -inE "deficit|formal city|\bWHO\b|\bflow\b|T[0-5]\b"` against the rendered template returns zero matches; "it is not a ranking of the five sites" (p.5) and the roughness/CFD caveats (p.5) are carried over verbatim. `${pi_contact}` on p.6 is confirmed as the spec's deliberate unfilled placeholder (`docs/mare_brief_spec.md` line 66, `docs/mare_brief_v2_spec.md` §3(c), `build_brief.py`'s `UNFILLED_ID`), not a template-rendering bug. |

## Round-2 findings status

1. **Fabric-cluster table separated from its lead-in across the p.5→p.6 break — FIXED.** p.5 now carries "The campaign's fabric-vector clustering assigns each built cell to one of six recurring fabric clusters. Maré's composition:" immediately followed by the table, both within p.5's content block (4.2% blank below); p.6 opens cold with "Availability and terms" instead, which is a section boundary, not a mid-table break. The round-2 smallest-fix suggestion (keep the intro paragraph and table in one `page-break-inside: avoid` block) matches what shipped.

2. **Wind rose (p.5) large relative to its information content — IMPROVED, not eliminated.** Measured this round: the rose block occupies ≈43% of p.5's used content height, down from round 2's ≈60% estimate — a real reduction, and p.5 is now well-filled (4.2% blank) rather than merely tolerated. Still large for 8 sector frequencies relative to Figure 1's four independently classed, 43,419-cell maps on a comparable footprint, but round 2 rated this **minor** and it is visibly smaller; call it addressed to the severity it was raised at.

3. **Round-1 finding 2 (systemic under-filled pages), round-2's "PARTIAL" — STILL PARTIAL, same mechanism, better against the stated bar.** p.3 and p.4 remain 26.1% / 27.0% blank below their last line of body text, for the same reason round 1 and round 2 both named: the street-segment SVF figure carries `page-break-inside: avoid` and moves whole to the next page when it doesn't fit the remaining gap, stranding blank space on the page it left. This is unchanged code, not a regression. New to round 3: `mare_brief_v2_spec.md` explicitly sets "no page less than about 60% filled except the last" as the bar for this cycle, and p.3 (73.9% filled) and p.4 (73.0% filled) both clear it — so this defect no longer violates the spec it is being built against, even though the underlying cause the last three rounds have all pointed at is still there, unfixed.

## Residual defects, ranked (for a future round)

1. **Page 6 is 71.7% blank below its content — major, but spec-exempted.** Root cause: moving the fabric-cluster composition passage to p.5 (the fix for round-2 finding 1) left only "Availability and terms" and "Methods notes and references" on p.6, both short sections. Smallest fix: either pull one more section onto p.6 (e.g. let "Geometry-derived ventilation potential" flow across the p.4/p.5 boundary differently so more of Figure 4 + prose lands earlier, freeing p.6 for something with real content), or explicitly accept it — the v2 spec already exempts the last page from the fill bar, so the cheapest fix is a one-line editorial call, not a layout change: confirm with the PI that a mostly-white closing page reads as "quiet, spec-compliant end" rather than "incomplete," and if not, merge pp. 5–6 content so the last page is denser.

2. **Pages 3 and 4 keep a ~26–27% blank tail — moderate, same three-round-old root cause.** `page-break-inside: avoid` on the street-segment SVF figure (`build_brief.py`, per round 1/round 2's line references) forces the whole figure to the next page rather than letting it break or shrink to fit the remaining gap. Smallest fix (unchanged advice from round 1 and round 2, still not taken): either drop `page-break-inside: avoid` for this one figure and let the renderer split it across the boundary, or add an explicit `min-height` check in the build script that shrinks the figure to the available remainder before forcing a page break. This is now the single oldest open finding across all three rounds.

3. **Wind rose still runs ≈43% of p.5 for 8 sector-frequency values.** Minor, already partially addressed (see findings status #2 above); a further ~15–20% width reduction would bring its area roughly in line with its information content, but p.5 is well-filled either way so this is cosmetic, not structural.

## Before you read

- **"MorphoFavela" appears by name** (p.1, opening sentence) — this is a disclosed project-name choice, not an oversight; confirm you're comfortable with the name surfacing to proposal teams before this leaves your hands.
- **Morphotype names are generalised to lettered clusters (A–F) by default**, not the internal T0–T5 codes — this is a reversible default per `mare_brief_v2_spec.md` §3(a) (a `--named-morphotypes` build flag restores the named variant); your call whether the default stays.
- **The contact line reads `Contact: ${pi_contact}` verbatim** on p.6 — this is deliberate (spec'd as unfilled, not a rendering bug); it needs your name/detail typed in before this goes to anyone external.
- **The last page (p.6) is about 70% blank** — it only carries "Availability and terms" and "Methods notes and references" now that the fabric-cluster table moved to p.5 to fix a worse defect (a table split from its own lead-in sentence). This satisfies the build spec's explicit exemption for the last page, but is worth a look before you decide it's fine to ship as-is.
- **Pages 3 and 4 each end with roughly a quarter of the page blank** (both still above the spec's 60%-filled floor) — a known, three-round-old layout limitation in how one figure is force-broken across pages; not a content or disclosure issue, just unfinished polish.
