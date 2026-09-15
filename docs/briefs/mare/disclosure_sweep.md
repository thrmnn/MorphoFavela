# Maré brief — disclosure sweep

Run against the rendered markdown (mare_morphology_brief.src.md, filled). Every hit needs an explicit include/drop decision — the PI decides.

## Greplist hits (ethics-gate SKILL.md v2)

| line | match | context | proposed decision |
|---:|---|---|---|
| 7 | `MorphoFavela` | MorphoFavela has already computed a detailed morphometric characterisation of | FLAG for PI (proposed INCLUDE) — names the analysis tool/pipeline that produced this brief's numbers; low sensitivity, but confirm MorphoFavela is an acceptable external-facing name before this brief leaves the repo. |
| 29 | `MorphoFavela` | Maré is the largest of MorphoFavela's five campaign sites and the flattest — | FLAG for PI (proposed INCLUDE) — names the analysis tool/pipeline that produced this brief's numbers; low sensitivity, but confirm MorphoFavela is an acceptable external-facing name before this brief leaves the repo. |
| 45 | `λf` | \| Geometry-derived ventilation tendencies \| per built cell \| derived (λf regime, lateral depth, wind exposure) \| derived aggregate \| shareable, de-georeferenced \| | INCLUDE (proposed) — standard published morphometric notation (Oke 1988; Stewart & Oke 2012), not project-internal. |
| 117 | `λf` | vertical enclosure (λf regime), lateral depth into contiguous fabric, and | INCLUDE (proposed) — standard published morphometric notation (Oke 1988; Stewart & Oke 2012), not project-internal. |
| 120 | `λf` | threshold on frontal-area density λf (vertical constraint), | INCLUDE (proposed) — standard published morphometric notation (Oke 1988; Stewart & Oke 2012), not project-internal. |
| 156 | `MorphoFavela` | MorphoFavela's five-site campaign spans hillside and flatland informal | FLAG for PI (proposed INCLUDE) — names the analysis tool/pipeline that produced this brief's numbers; low sensitivity, but confirm MorphoFavela is an acceptable external-facing name before this brief leaves the repo. |
| 199 | `Oke` | - Frontal-area density and enclosure-threshold classification: Oke (1988); | INCLUDE (proposed) — standard published morphometric notation (Oke 1988; Stewart & Oke 2012), not project-internal. |
| 200 | `Oke` | Stewart & Oke (2012), local climate zones. | INCLUDE (proposed) — standard published morphometric notation (Oke 1988; Stewart & Oke 2012), not project-internal. |

## Additional codename / method-neologism / result-parameter check

Beyond the greplist: does the text name an unpublished method, an internal codename, a venue/journal, or a result parameter?

- Lines [163, 167, 168, 169, 170, 171, 172, 174, 176, 178]: HOLD (proposed) — the six-morphotype taxonomy (T0-T5 + names) is the project's own unpublished classification scheme (see technical_report.md §5.5); naming it externally may pre-empt a paper contribution. PI to decide: keep the named taxonomy, or generalise to 'recurring fabric clusters' without the T0-T5 labels.
- Lines [120, 123, 125]: INCLUDE (proposed) — matches the already-CLEAR release class of the WP-07 f4_geometry_constraints figure in red_lines.md §5 (ordinal geometry-only constraint count, denominator = built cells, not ranked).

No venue/journal name, no collaborator name, and no other project codename (brisa/brisaverse/P1-P4/track names) appear in the rendered brief.
