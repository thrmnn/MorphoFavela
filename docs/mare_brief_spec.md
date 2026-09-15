# Maré morphology brief — spec (2026-09-15)

PI, 2026-09-15: a concise, well-presented PDF report on the morphometric work
already done for Maré, for researchers writing research proposals on Maré, so
the PI can offer them the existing analysis. The PI reviews the PDF. Audience
is EXTERNAL → this is a release-boundary artifact: it is built here, gated by
the ethics-gate + disclosure sweep afterwards, and nothing leaves until the
PI says so.

## Deliverables (all under `docs/briefs/mare/`)

1. `collect_numbers.py --outputs-root <path>` → `mare_numbers.json`: every
   number the brief prints, read by code from the outputs of record (the
   worktree has no `outputs/`; read from
   `/home/theo/SCL/SCR/MorphoFavela/outputs/maré/…` and
   `/home/theo/SCL/SCR/MorphoFavela/outputs/cross_site/…` via the flag).
   Sources: `morphometrics/grid/grid_metrics.csv` (λp, λf, H_mean, σH,
   porosity, SVF, slope distributions — medians/IQR/shares, n cells, built
   cells), `svf_v2/svf_streets_segments.gpkg` (street SVF), 
   `morphometrics/svf/svf_streets_solar.gpkg` if present (winter/annual sun
   hours, share ≥ 2 h), `geometry_indicators/per_patch_geometry.csv`
   (n_constraints shares, wind exposure), `paper_figures/diagnostic_stats.json`,
   `cross_site/signature/composition_by_site.csv` (Maré morphotype
   composition), `cross_site/roughness/patch_roughness.csv` (Maré rows,
   method envelope only), `data/maré/wind_rose.json` (sectors, calm share),
   and the site row of `docs/technical_report/technical_report.md` §1 (area,
   buildings, cells, typology — parse the table, do not retype). Each entry:
   value, unit, source path, column/expression.
2. `mare_morphology_brief.src.md` — the authored template with `${id}`
   placeholders (Python `string.Template`); `build_brief.py` fills it from
   `mare_numbers.json` (3 significant figures; integers with thousands
   separators), copies the allowed figures into `figures/`, then
   pandoc → HTML → weasyprint → `mare_morphology_brief.pdf` (reuse the
   pipeline shape of `docs/technical_report/build_pdf.py`; write a NEW, calmer
   stylesheet: A4, 10.5 pt body, one accent colour, captions, a two-column
   key-numbers block, page footer "Maré morphology brief · <date> · draft for
   PI review"). ≤ 8 pages. No typed numbers anywhere in the template.
3. Figures — copy from outputs only if in this allowed set, and state the
   class in `figure_manifest.json`: band-classed 10 m maps of λp, H_mean,
   porosity and SVF (render fresh with matplotlib from grid_metrics.gpkg:
   4–5 discrete classes, no basemap, no coordinate axes/ticks, scale bar +
   north arrow only); street-segment SVF map (segments coloured by class, no
   coordinates); distributions (histograms/ECDFs) of λp, H_mean, SVF, street
   SVF; the diagnostic map `paper_figures/fig_maré_diagnostic_map.png`
   (already band-classed, publishable family); the wind rose. EXCLUDED, never
   copied: slope/aspect/northness maps (immutable terrain, L3), any
   per-building attribute map (heights per footprint, L2-adjacent), any
   risk map (L4), any health/TB material, any favela-vs-formal panel, any
   map with UTM ticks or a basemap, anything from `outputs/maré/cfd*`.
4. Content (sections, in order; ≤ 8 pages): (1) Purpose — one paragraph:
   what exists, how a proposal team can use it, what the PI can provide;
   (2) Maré at a glance — key-numbers block; (3) Data inventory — table:
   layer · resolution/unit · provenance · derived-or-restricted · shareable
   form; (4) Built form — λp, H_mean, porosity, σH maps + distributions;
   (5) Sky access and sun — SVF (grid + street), reference-day sun hours
   with the 2 h floor labelled "Athens Charter (1943), Point 26" (never
   "WHO"); (6) Geometry-derived ventilation potential — the ordinal
   constraint count, wind exposure, wind rose, one sentence that wind
   simulation for two Maré patches is a parked companion track with no
   results to report; (7) Maré among the five campaign sites — typology
   position (flatland, planned-housing share), composition by morphotype;
   descriptive, never ranked, no "deficit"/"formal" wording; (8) Availability
   and terms — derived aggregates and the de-georeferenced 10 m grid can be
   shared; cadaster and LiDAR inputs cannot; per-cell georeferenced layers on
   request under the project's conditions; contact = PI (name only, no email
   typed by you — leave `${pi_contact}` for the PI); (9) Methods notes and
   references (Stewart & Oke 2012 LCZ, Oke 1988 for the λf threshold, Tregenza
   sky, Athens Charter 1943). Plain descriptive voice; every sentence must be
   defensible from the numbers file; where a number does not exist, write
   `PLACEHOLDER` and list it in the final message.
5. `disclosure_sweep.md` — run the ethics-gate disclosure grep from
   `/home/theo/SCL/SCR/brisaverse/.claude/skills/ethics-gate/SKILL.md` over
   the rendered markdown and list EVERY hit with a proposed include/drop
   decision (the PI decides). Also list every internal codename, venue,
   method neologism and result parameter that appears.
6. `tests/test_mare_brief.py`: (a) every `${id}` in the template resolves;
   (b) no digit sequence in the template outside placeholders (no typed
   numbers); (c) figure_manifest lists only allowed classes and no excluded
   basename patterns (`slope`, `aspect`, `height` per-building, `risk`,
   `tb`, `cfd`); (d) rendered markdown has no banned tokens (call
   `scripts/lint_p1_tokens.py`'s scan function) and none of "deficit",
   "formal city", "WHO"; (e) the PDF exists and has ≤ 8 pages (pypdf);
   skip cleanly when outputs are absent.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_mare_brief.py -q
python3 docs/briefs/mare/build_brief.py --outputs-root /home/theo/SCL/SCR/MorphoFavela/outputs
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

Commit `docs/briefs/mare/**` incl. the PDF and `figures/` (small PNGs), the
test, nothing under `outputs/`. Commit last; paste hashes; list every
PLACEHOLDER and every disclosure hit in the final message.

## Never

No email sent, no contact details typed; no health/TB content; no CFD
results; no per-building or immutable-terrain maps; no coordinates; no
ranking of favelas; no "deficit"/"formal"/"WHO"; no typed numbers; nothing
written outside `docs/briefs/mare/` and `tests/`.
