# Maré brief — disclosure sweep

Run against the rendered markdown (mare_morphology_brief.src.md, filled). Every hit needs an explicit include/drop decision — the PI decides.

## Provisional disclosure default

brisaverse tasks.json `_meta.provisional_default_policy`: reversible, no external dependency, PI tap overrides. The T0–T5 morphotype taxonomy (codes and names) is the project's own unpublished classification scheme (technical_report.md §5.5); by default it does NOT appear in this brief — the composition passage in 'Maré among the five campaign sites' describes six lettered fabric clusters (A–F) by share, with a plain description of the dominant cluster and no taxonomy codes or names. Pass `--named-morphotypes` to build_brief.py to restore the named T0–T5 variant for PI review.

`--named-morphotypes`: not passed — this build rendered the default, generalised (lettered A–F) variant.

## Greplist hits (ethics-gate SKILL.md v2)

| line | match | context | proposed decision |
|---:|---|---|---|
| 7 | `MorphoFavela` | MorphoFavela has already computed a detailed morphometric characterisation of | FLAG for PI (proposed INCLUDE) — names the analysis tool/pipeline that produced this brief's numbers; low sensitivity, but confirm MorphoFavela is an acceptable external-facing name before this brief leaves the repo. |
| 38 | `MorphoFavela` | Maré is the largest of MorphoFavela's five campaign sites and the flattest — | FLAG for PI (proposed INCLUDE) — names the analysis tool/pipeline that produced this brief's numbers; low sensitivity, but confirm MorphoFavela is an acceptable external-facing name before this brief leaves the repo. |
| 54 | `λf` | \| Geometry-derived ventilation tendencies \| per built cell \| derived (λf regime, lateral depth, wind exposure) \| derived aggregate \| shareable, de-georeferenced \| | INCLUDE (proposed) — standard published morphometric notation (Oke 1988; Stewart & Oke 2012), not project-internal. |
| 119 | `λf` | vertical enclosure (λf regime), lateral depth into contiguous fabric, and | INCLUDE (proposed) — standard published morphometric notation (Oke 1988; Stewart & Oke 2012), not project-internal. |
| 122 | `λf` | threshold on frontal-area density λf (vertical constraint), | INCLUDE (proposed) — standard published morphometric notation (Oke 1988; Stewart & Oke 2012), not project-internal. |
| 156 | `MorphoFavela` | MorphoFavela's five-site campaign spans hillside and flatland informal | FLAG for PI (proposed INCLUDE) — names the analysis tool/pipeline that produced this brief's numbers; low sensitivity, but confirm MorphoFavela is an acceptable external-facing name before this brief leaves the repo. |
| 201 | `Oke` | - Frontal-area density and enclosure-threshold classification: Oke (1988); | INCLUDE (proposed) — standard published morphometric notation (Oke 1988; Stewart & Oke 2012), not project-internal. |
| 202 | `Oke` | Stewart & Oke (2012), local climate zones. | INCLUDE (proposed) — standard published morphometric notation (Oke 1988; Stewart & Oke 2012), not project-internal. |

## Additional codename / method-neologism / result-parameter check

Beyond the greplist: does the text name an unpublished method, an internal codename, a venue/journal, or a result parameter?

- Lines [122, 125, 127]: INCLUDE (proposed) — matches the already-CLEAR release class of the WP-07 f4_geometry_constraints figure in red_lines.md §5 (ordinal geometry-only constraint count, denominator = built cells, not ranked).

No venue/journal name, no collaborator name, and no other project codename (brisa/brisaverse/P1-P4/track names) appear in the rendered brief.
