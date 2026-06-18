# Workflow patterns

A reusable workflow lives at
`/home/theo/MorphoFavela/.claude/workflows/audit-and-deliverable-per-entity.js`.
It encodes the shape we keep reinventing by hand: fan out per-entity probes,
run a panel of persona-distinct critics, synthesize into severity buckets,
optionally propose and judge a deliverable, then build it per entity.

This document explains when to reach for it, how it is wired, and how to
invoke it cleanly.

## 1. When to use the audit-and-deliverable-per-entity pattern

Use it when **all** of the following are true:

- You have a small-to-moderate set of comparable entities (3–150) that share
  a directory schema or artefact shape — sites, CFD patches, wind stations,
  species, treatments, snapshots.
- Each entity can be inspected independently and produces a structured
  discovery record (counts, ranges, NaNs, red flags), not free-form prose.
- You want the audit *signal* to come from disagreement across multiple
  expert lenses, not a single reviewer.
- The downstream output is either a written report (audit only) or a
  per-entity deliverable that follows one shared spec (figure, dossier,
  one-pager).

Anti-patterns — do not use this workflow when:

- **N = 1.** A single entity does not benefit from the parallel fan-out.
  Just write the audit directly.
- **Entities are not comparable.** If each "entity" needs a bespoke
  inspection prompt, the discovery phase collapses; you are really running
  N one-off audits and the synthesis step has nothing to dedup.
- **The deliverable is monolithic.** A single cross-entity report (one
  manuscript, one slide deck) does not map onto per-entity build. Stop at
  Phase 3 by omitting `design_angles`.
- **Discovery requires expensive compute.** Each probe is a short agent
  call; if inspection requires a 20-minute pipeline, run that pipeline
  first to produce cached JSON, then point discovery at the cache.
- **Findings must be exhaustive, not skeptical.** The `SKEPTIC_CLAUSE`
  asks lenses to omit unsubstantiated claims. For compliance or coverage
  audits that need every potential issue surfaced, this pattern under-reports.

## 2. Anatomy of the pattern

Eight phases. The first three are the audit half; the last five are the
deliverable half and can be skipped by omitting `design_angles`.

1. **Discovery (parallel, per-entity).** One agent per entity inspects its
   on-disk artefacts and returns a `DISCOVERY_SCHEMA` record. *Swap:* the
   inspection prompt — what files to open, which numerics to compute, what
   counts as a red flag. The schema itself is fixed.
2. **Expert audit (parallel, per-lens).** Each lens in `expert_lenses`
   instantiates an agent with its own persona and focus, all consuming the
   same `discoverySummary`. Each returns `FINDINGS_SCHEMA[]`. *Swap:* the
   lens triples. Three to five lenses is the sweet spot.
3. **Audit synthesis (single arbiter, disk side effect).** One agent
   collapses all findings, deduplicates, and buckets into
   `critical / fixable_in_deliverable / must_document_caveat`. Writes
   `{output_dir}/{basename}_report.md`. *Swap:* the bucket definitions if
   your domain needs different actionability axes (rarely needed).
4. **Design proposals (parallel, per-angle).** Each angle in `design_angles`
   becomes a proposal agent constrained by `{medium, constraint}`. *Swap:*
   the angles. This is where you cast the option space wide — keep angles
   genuinely different (a static PDF vs. an interactive HTML vs. a
   thumbnail-grid are three angles; three flavors of bar chart are one).
5. **Design judging (parallel, per-judge).** A small panel scores each
   proposal on fitness-to-audit, feasibility, and information density.
   *Swap:* judging rubric weights via `dashboard_spec_hint`.
6. **Spec synthesis (single writer).** Picks a winner (or hybrid),
   produces a build-ready spec at `{output_dir}/{basename}_design_spec.md`.
   *Swap:* nothing — the spec format is dictated by Phase 7's needs.
7. **Per-entity build (parallel).** One agent per entity instantiates the
   spec against that entity's data. *Swap:* the build agent's tool budget
   (it needs Write + Bash + whatever plot library you target).
8. **Completeness critic (single reviewer).** Cross-checks every built
   deliverable against the spec and the audit's `must_document_caveat`
   bucket. *Swap:* tolerance for partial deliverables via the hint.

Designers should treat phases 1, 2, and 4 as the customization surface.
3, 5, 6, 8 are plumbing. 7 is mechanical given a good spec.

## 3. Args contract

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `entities` | `string[]` | yes | — | 3–150. Drives `parallel()` width in phases 1 and 7. Stable IDs only (filesystem-safe). |
| `context` | `string` | yes | — | Shared brief, passed verbatim as preamble to every agent in every phase. Keep under ~2k chars. |
| `discovery_prompt_template` | `string` | yes | — | Must contain `{entity}`. No other placeholders. |
| `expert_lenses` | `{key, persona, focus}[]` | yes | — | 2 minimum, 3–5 recommended. `key` must be a slug. `focus` should name what the lens may NOT talk about, not just what it covers. |
| `design_angles` | `{name, medium, constraint}[]` | no | omit | Omitting stops the workflow after Phase 3. |
| `output_dir` | `string` (abs) | yes | — | Created if missing. |
| `artefact_basename` | `string` | no | `"audit"` | Stem for written files. |
| `dashboard_spec_hint` | `string` | no | `""` | Free-text steer for synthesis and judging. Use for rubric weighting (e.g., "favor static-PDF feasibility over interactivity"). |

Validation (enforced in the workflow):

- `entities.length >= 1`; warns above 50, refuses above 200.
- `discovery_prompt_template` must contain literal `{entity}`.
- `expert_lenses.length >= 2`; lens `key`s must be unique slugs
  (`^[a-z][a-z0-9-]*$`).
- `output_dir` must be absolute.
- If `design_angles` is provided, length must be >= 2 (otherwise judging is
  a no-op).
- All agent outputs are truncated to `TRUNC = 16000` chars before being
  fed into the next phase.

## 4. Three worked invocations

### 4a. MorphoFavela — the current case (Brisa per-site SVF audit + Fig 03 rebuild)

```js
Workflow({
  name: 'audit-and-deliverable-per-entity',
  args: {
    entities: ['vidigal', 'vdg-p02', 'vdg-p07', 'jacarezinho', 'mare'],
    context:
      'Per-site SVF rasters in outputs/{site}/morphometrics/svf/. ' +
      'Recent λf taxonomy fix (commit 09823d8) regenerated values; ' +
      'we need to verify SVF distributions are physically plausible ' +
      'before regenerating Fig 03/04 for the Brisa paper.',
    discovery_prompt_template:
      'Inspect outputs/{entity}/morphometrics/svf/ with geopandas+rasterio. ' +
      'Report row count, SVF min/p50/max, NaN fraction, CRS, and any pixels ' +
      'with SVF > 1.0 or < 0 as red_flags.',
    expert_lenses: [
      { key: 'urban-climatology', persona: 'Senior urban climate scientist',
        focus: 'physical plausibility of SVF distributions; ignore code style' },
      { key: 'gis-sampling', persona: 'GIS sampling specialist',
        focus: 'CRS, resolution, edge effects; ignore physics interpretation' },
      { key: 'numerical-methods', persona: 'Numerical methods reviewer',
        focus: 'NaN propagation, edge artifacts, raytrace stability' },
      { key: 'dataviz-design', persona: 'Information designer',
        focus: 'what these distributions imply for Fig 03 framing' },
    ],
    design_angles: [
      { name: 'small-multiples-violin',  medium: 'matplotlib PDF',
        constraint: 'one violin per site, shared y-axis' },
      { name: 'ridge-overlay',           medium: 'matplotlib PDF',
        constraint: 'overlapping density ridges, single panel' },
      { name: 'paired-map-histogram',    medium: 'matplotlib PDF',
        constraint: 'two-row grid: SVF map above, histogram below' },
    ],
    output_dir: '/home/theo/MorphoFavela/outputs/paper_figures/brisa_fig03_audit',
    artefact_basename: 'svf_audit',
    dashboard_spec_hint:
      'Favor reproducibility (matplotlib + commit-pinned data) over polish. ' +
      'Each site deliverable must cite its SVF percentile bounds in the caption.',
  },
});
```

Result: `svf_audit_report.md` + `svf_audit_design_spec.md` + five
per-site PNG/PDF figures in the output dir.

### 4b. Different domain — tree inventory / LAI per-plot QA (from the reuse list)

Mapping the same shape onto the LAI domain:

```js
Workflow({
  name: 'audit-and-deliverable-per-entity',
  args: {
    entities: ['plot-A1', 'plot-A2', 'plot-B1', 'plot-B2', 'plot-C1'],
    context:
      'Hemispherical-photo LAI estimates per plot, post-thresholding. ' +
      'Verify gap-fraction distributions and sun-fleck artefacts before ' +
      'aggregating to stand-level LAI for the inventory report.',
    discovery_prompt_template:
      'Load data/lai/{entity}/processed/*.csv. Report photo count, ' +
      'gap-fraction min/p50/max per zenith ring, sun-azimuth coverage, ' +
      'and flag any rings with >40% saturated pixels.',
    expert_lenses: [
      { key: 'forest-ecology',  persona: 'Forest canopy ecologist',
        focus: 'biological plausibility of LAI by species mix' },
      { key: 'optics',          persona: 'Hemispherical-photo specialist',
        focus: 'exposure, thresholding, sun-fleck contamination' },
      { key: 'field-protocol',  persona: 'Field-campaign QA reviewer',
        focus: 'sample size, azimuth coverage, time-of-day bias' },
    ],
    design_angles: [
      { name: 'per-plot-onepager', medium: 'matplotlib PDF',
        constraint: 'A4, ring-LAI bar + sample hemiphoto + caveat list' },
      { name: 'plot-grid-poster',  medium: 'matplotlib PDF',
        constraint: 'all plots, one tile each, single A2 sheet' },
    ],
    output_dir: '/home/theo/lai-inventory/reports/2026_summer',
    artefact_basename: 'lai_plot_qa',
  },
});
```

### 4c. Stress test — small (N=2) and large (N=119)

Small (N=2) — should refuse cleanly: with only two entities the parallel
fan-out is barely worth it, but the workflow does not refuse. Discovery
runs twice, the four lenses run once each, synthesis sees ~8 findings and
will likely produce a thin report. Use as a smoke test:

```js
Workflow({
  name: 'audit-and-deliverable-per-entity',
  args: {
    entities: ['vidigal', 'jacarezinho'],
    context: 'Smoke test of the workflow on two known-good sites.',
    discovery_prompt_template:
      'Quick sanity check on outputs/{entity}/morphometrics/grid.gpkg: ' +
      'row count, λp/λf ranges, CRS.',
    expert_lenses: [
      { key: 'gis-sampling',    persona: 'GIS reviewer',
        focus: 'CRS + completeness only' },
      { key: 'urban-morphology', persona: 'Morphometrics reviewer',
        focus: 'λp/λf distribution shape only' },
    ],
    output_dir: '/tmp/morphofavela-smoke',
  },
});
```

Large (N=119) — the full CFD patch campaign post-ingestion:

```js
Workflow({
  name: 'audit-and-deliverable-per-entity',
  args: {
    entities: PATCH_IDS,  // 119 IDs, e.g. ['vdg-p01', ..., 'mare-p25']
    context:
      'Post-CFD patch reports. Each patch lives at ' +
      'data/{site}/cfd_results/{patch_id}/{wind_dir}/. Validate ' +
      'convergence + physical plausibility against the design strata.',
    discovery_prompt_template:
      'For patch {entity}: load residual log, check U/p/k/omega field ' +
      'completeness for all 8 wind dirs, compute wall y+ histogram, ' +
      'flag residuals not below 1e-4 by the final iteration.',
    expert_lenses: [
      { key: 'numerical-methods', persona: 'CFD numerics reviewer',
        focus: 'residuals, mesh quality, y+ — ignore meteorology' },
      { key: 'urban-climatology', persona: 'Urban climate scientist',
        focus: 'U_z=2m plausibility against wind_rose.json' },
      { key: 'sampling',          persona: 'DOE reviewer',
        focus: 'does this patch represent its SVF×slope×λp stratum' },
    ],
    design_angles: [
      { name: 'per-patch-onepager', medium: 'matplotlib PDF',
        constraint: 'A4, residual plot + 8-wind U field + caveat box' },
    ],
    output_dir: '/home/theo/MorphoFavela/docs/cfd_reports',
    artefact_basename: 'patch_report',
    dashboard_spec_hint:
      'Phase-7 build will hit rate limits; throttle to 8 concurrent builds.',
  },
});
```

At N=119, expect the discovery phase to dominate wall-clock time. Pre-cache
inspection results to JSON if you plan to re-run.

## 5. Failure modes and how to tune

**Expert lenses thrash (every lens flags the same five things).**
Symptom: synthesis dedup produces a 3-finding report from 4×N raw findings.
Cause: lens `focus` strings overlap. Fix: rewrite each `focus` to explicitly
name what the lens may NOT talk about. The "ignore X" clause is load-bearing.

**Expert lenses disagree on severity.** Symptom: synthesis bucketing is
unstable across runs. Cause: severity is under-specified. Fix: pin the
severity definitions in `context` (e.g. "critical = blocks publication;
major = needs caveat; minor = nice-to-fix"). The workflow respects whatever
definition lives in `context`.

**Judges deadlock (no proposal scores clearly highest).** Symptom: Phase 6
spec is a hedged hybrid that is harder to build than any single proposal.
Cause: `design_angles` are too similar. Fix: enforce orthogonality — angles
should differ on `medium` (PDF vs HTML vs notebook) or `constraint`
(single-panel vs grid), not just on style. Two genuinely different angles
beat four cosmetic variants.

**Judges deadlock and angles are already orthogonal.** Cause: rubric is
under-weighted. Fix: use `dashboard_spec_hint` to tilt the rubric, e.g.
"feasibility outweighs information density 2:1 for this run".

**Build round 2 doesn't converge (completeness critic keeps flagging the
same gaps after rebuild).** Symptom: same caveat re-surfaces in
`must_document_caveat` after the per-entity build. Cause: the spec failed
to translate the caveat into a build constraint. Fix: edit
`{basename}_design_spec.md` by hand to add the caveat as an explicit
caption/annotation requirement, then re-run from Phase 7. The workflow
supports resuming at a named phase via the orchestrator; do not re-run
Phase 1 just to retry Phase 7.

**Discovery returns garbage for a subset of entities.** Symptom: red_flags
is `["could not read file"]` for some entities. Cause: those entities are
not on disk where the prompt template assumes. Fix: filter `entities` upstream
— the workflow does not validate paths before fanning out, because the
correct path varies by domain.

**Synthesis report is bland.** Symptom: every finding lands in
`must_document_caveat`. Cause: the `SKEPTIC_CLAUSE` is too strong for your
domain, or evidence requirements are too loose. Fix: tighten the discovery
prompt to compute the specific numerics lenses need to make a `critical`
call. Lenses can only be sharp if discovery gave them sharp inputs.
