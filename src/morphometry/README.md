# `src/morphometry/`

The morphometric audit pipeline. Computes a 12-indicator 10 m grid
across each campaign site, generates publication-quality figures,
and renders a per-site PDF report. Outputs feed into both the
technical report (`docs/technical_report/`) and the Nature Cities
paper figures (`outputs/paper_figures/`).

## Indicators on the 10 m grid

For each cell that has ≥ 1 building intersecting:

| Indicator | Symbol | Definition |
|---|---|---|
| Plan area density | λp | Σ footprint area / cell area |
| Frontal area density | λf | Σ frontal area perpendicular to wind / cell area (8 directions, max + mean) |
| Sky view factor | SVF | From `src/svf_v2`; mean over cell |
| Mean height | H̄ | Population-weighted mean building height |
| Height std | σH | Std-dev of building heights in the cell |
| Slope | s | Local terrain gradient (degrees) |
| Building count | N | Number of intersecting buildings |
| Aspect | — | Local terrain aspect |
| Porosity (sectional) | Ps | 1 − built area at z = 1.5 m |
| Solar access (winter) | Sw | Hours of direct sun on Jun 21 |
| Volumetric density | ρV | Σ building volume / cell volume to H̄ |
| Plot ratio | FAR | Σ floor area / cell area (assuming 3 m storeys) |

## Public API (re-exported in `__init__`)

| Function | Purpose |
|---|---|
| `compute_grid_morphometrics(...)` | One-shot pipeline: returns the 10 m grid as a GeoDataFrame |
| `compute_lambda_f_directional(...)` | λf for one or more wind directions |
| `compute_porosity(...)` | Sectional porosity at a chosen height |
| `compute_slope_aspect(...)` | Slope + aspect from a DTM |
| `audit_svf(...)` | Sanity-check SVF outputs against a per-site target distribution |

## Submodules

- `grid.py` — orchestrator; calls `indicators.py` for each metric
- `indicators.py` — per-metric computation functions
- `audit.py` — quality checks on SVF + λp distributions
- `figures.py` — publication-grade matplotlib figures
- `report.py` — assembles the per-site PDF (matplotlib + ReportLab)

## Typical usage

```bash
python scripts/run_morphometric_audit.py --area vidigal
```

This produces `outputs/{site}/morphometrics/` with:

- `grid/grid_metrics.gpkg`     — the 10 m grid with all indicators
- `report/morphometric_report.pdf` — per-site audit
- `figures/*.png`              — individual indicator maps

## Tests

`tests/test_morphometry/` — covers `compute_grid_morphometrics` on
synthetic geometry plus golden-output regression tests on a small
subset of vidigal grid cells.

## Notes

- All metrics use the **extended building context** (300 m buffer
  via `scripts/build_extended_context.py`); cells inside the
  favela boundary still see the surrounding urban fabric.
- Grid CRS is EPSG:31983 (UTM 23S, SIRGAS 2000) for compatibility
  with the CFD sampling pipeline.
