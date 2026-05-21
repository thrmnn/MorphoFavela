# Manuscript figures

Reproducible scripts for figures **targeted at the final academic paper**
(journal submission). These are *paper candidates* — their primary audience
is peer reviewers and journal layout, with different communication
objectives and standards from the internal technical-report figures.

The manuscript itself is written outside this repo; this directory holds
only the figure scripts and tracked exports so they travel with the code.

> **Two-track convention.** This directory is the *paper* track.
> `outputs/paper_figures/` is the *technical-report + presentations* track.
> The two serve different audiences (peer review vs. internal stakeholders)
> and different standards (journal style vs. project narrative). Do not
> mix them — see `.claude/projects/-home-theo-MorphoFavela/memory/` for the
> convention rule. Final paper writing/layout is out of scope for this
> repo; this directory only produces *figure candidates*.

Each script is standalone and reads from `data/` and `outputs/` like every
other figure in the repo. Outputs land in `figures/exports/` (tracked, so
the PNG + SVG pair travels with the code).

## Figures

| Figure | Script | Description |
|--------|--------|-------------|
| 0.1 | `figures/fig_0_1_overview.py` | Conceptual framework + Rio map by typology + 3D excerpts of the five sites |
| 0.2 | `figures/fig_0_2_landscape.py` | Vidigal 6-panel choropleth (SVF, λp, porosity, λf, σH, slope) + cross-favela ridge plots |
| 0.3 | `figures/fig_0_3_performance.py` | Environmental performance: wind/sun maps for 4 representative patches + ACH and sun-hours distributions |
| 0.4 | `figures/fig_0_4_diagnostic.py` | ★ Headline. 4-state diagnostic taxonomy (5 site maps + 2D performance scatter + stacked bars across hillside/mixed/flatland aggregates) |
| 0.5 | `figures/fig_0_5_predictors.py` | Statistical findings: RF permutation importance, partial-dependence curves, logistic forest plot with interactions, SVF→U_mean changepoint regression |
| 0.6 | `figures/fig_0_6_climate_stress.py` | Wind-stilling stress test: 4-state shift under {U×1.00, U×0.85, U×0.70} + cell-level state-flip maps + typology vulnerability ladder |
| 0.7 | `figures/fig_0_7_proposition_clustering.py` | Spatial clustering audit: per-site Moran's I (999-perm null) + LISA cluster maps + cluster-size CCDF observed vs random null |
| 0.8 | `figures/fig_0_8_terrain_confound.py` | Slope-stratified 4-state shares + per-site slope-bin maps with compound-failure outlines + typology × slope-bin compound % grouped bars |

The 0.7 selection landed in commit `979bb40` — two sibling propositions
(`fig_0_7_proposition_interventions.py`, `fig_0_7_proposition_framework.py`)
were retired in favour of the clustering version. Recoverable from git
history if a future revision needs them.

## Regenerate

```bash
python docs/manuscript/figures/fig_0_1_overview.py
python docs/manuscript/figures/fig_0_2_landscape.py
python docs/manuscript/figures/fig_0_3_performance.py
python docs/manuscript/figures/fig_0_4_diagnostic.py

# Fig 0.5 needs the diagnostic-models artifacts first:
python scripts/run_diagnostic_models.py
python docs/manuscript/figures/fig_0_5_predictors.py

# Fig 0.6 reuses the 4-state classifier from fig_0_4_diagnostic.
python docs/manuscript/figures/fig_0_6_climate_stress.py

# Fig 0.7 uses libpysal/esda; reuses the 4-state classifier from 0.4.
python docs/manuscript/figures/fig_0_7_proposition_clustering.py

# Fig 0.8 reuses the 4-state classifier + slope bins.
python docs/manuscript/figures/fig_0_8_terrain_confound.py
```

## Style

Shared with `outputs/paper_figures/fig_style.py` (Tol muted palette,
7 pt Liberation Sans, 600 DPI PNG + SVG vector). The shared style helpers
are intentional — both tracks want consistent Rio sites colors and per-site
typology grouping — but exports paths differ (this dir vs.
`outputs/paper_figures/exports/`).

## Paper integration plan

These figures are *candidates*. Final paper layout will choose 1–2 of the
0.x series as the headline display item(s); the rest may move to extended
data or supplementary. The 4-state diagnostic (0.4) is the current
headline candidate; 0.6 (climate stress) and 0.8 (terrain confound) are
the strongest robustness companions.

A figure landing in the paper does NOT mean removing it from the technical
report's §7.5 preview block — the TR is allowed to show paper candidates
as an early-look section.

## Cross-reference

For figures **targeted at the technical report and presentations**, see
`outputs/paper_figures/README.md`.
