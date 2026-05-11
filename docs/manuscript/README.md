# Manuscript figures

Figures generated for the journal manuscript draft, kept separate from the
technical-report figures under `docs/technical_report/figures/`. The
manuscript itself lives outside this repo; this directory holds the
reproducible figure scripts only.

Each script is standalone and reads from `data/` and `outputs/` like every
other figure in the repo. Outputs land in `figures/exports/`.

## Figures

| Figure | Script | Description |
|--------|--------|-------------|
| 0.1 | `figures/fig_0_1_overview.py` | Conceptual framework + Rio map by typology + 3D excerpts of the five sites |
| 0.2 | `figures/fig_0_2_landscape.py` | Vidigal 6-panel choropleth (SVF, λp, porosity, λf, σH, slope) + cross-favela ridge plots |
| 0.3 | `figures/fig_0_3_performance.py` | Environmental performance: wind/sun maps for 4 representative patches + ACH and sun-hours distributions |
| 0.4 | `figures/fig_0_4_diagnostic.py` | ★ Headline. 4-state diagnostic taxonomy (5 site maps + 2D performance scatter + stacked bars across hillside/mixed/flatland aggregates) |
| 0.5 | `figures/fig_0_5_predictors.py` | Statistical findings: RF permutation importance, partial-dependence curves, logistic forest plot with interactions, SVF→U_mean changepoint regression |
| 0.6 A | `figures/fig_0_6_proposition_interventions.py` | **Proposition.** Counterfactual morphological levers (SVF +0.10, λp −0.10, σH ×0.5) → % of failing cells recovered, per typology + cost-effectiveness frontier |
| 0.6 B | `figures/fig_0_6_proposition_sensitivity.py` | **Proposition.** Threshold-dial sensitivity audit: 3×3 (U_vent × sun_h) small multiples + cell-level stability map + per-typology compound-failure heatmap |
| 0.6 C | `figures/fig_0_6_proposition_climate.py` | **Proposition.** Wind-stilling stress test: 4-state shift under {U×1.00, U×0.85, U×0.70} + cell-level state-flip maps + typology vulnerability ladder |

Figures with the `proposition_` infix are *candidate* journal figures —
pick one (or merge motifs) before final manuscript layout. The other
`fig_0_x` figures are committed manuscript artefacts.

## Regenerate

```bash
python docs/manuscript/figures/fig_0_1_overview.py
python docs/manuscript/figures/fig_0_2_landscape.py
python docs/manuscript/figures/fig_0_3_performance.py
python docs/manuscript/figures/fig_0_4_diagnostic.py

# Fig 0.5 needs the diagnostic-models artifacts first:
python scripts/run_diagnostic_models.py
python docs/manuscript/figures/fig_0_5_predictors.py

# Fig 0.6 candidates (each is standalone; A/B reuse build_cell_table /
# fig_0_4 infrastructure).
python docs/manuscript/figures/fig_0_6_proposition_interventions.py
python docs/manuscript/figures/fig_0_6_proposition_sensitivity.py
python docs/manuscript/figures/fig_0_6_proposition_climate.py
```

Style is shared with `outputs/paper_figures/fig_style.py` (Tol muted palette,
7 pt Liberation Sans, 600 DPI PNG + SVG vector).
