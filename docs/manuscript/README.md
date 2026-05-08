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

## Regenerate

```bash
python docs/manuscript/figures/fig_0_1_overview.py
python docs/manuscript/figures/fig_0_2_landscape.py
```

Style is shared with `outputs/paper_figures/fig_style.py` (Tol muted palette,
7 pt Liberation Sans, 600 DPI PNG + SVG vector).
