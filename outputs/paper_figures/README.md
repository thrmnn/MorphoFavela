# Paper figures — technical-report + presentations track

Reproducible figure scripts **targeted at the internal technical report
(`docs/technical_report/`) and stakeholder presentations**. Despite the
directory name ("paper_figures"), this is the *internal* communication
track — not the academic-paper track.

> **Two-track convention.** This directory is the *technical-report +
> presentations* track. `docs/manuscript/figures/` is the *academic paper*
> track. The two serve different audiences (internal stakeholders vs.
> peer reviewers) and different standards (project narrative vs. journal
> style). Do not mix them. The directory name "paper_figures" is
> historical (this set was originally drafted for a paper before the
> two-track split was formalised); it now serves the TR.

All scripts are standalone and regenerable from pipeline outputs.

## Regenerate all figures

```bash
cd /path/to/MorphoFavela
for f in outputs/paper_figures/fig*.py; do python3 "$f"; done
```

Exports land in `exports/` (gitignored — copy the relevant PNGs into
`docs/technical_report/figures/` before rebuilding the TR PDF). All
scripts use `fig_style.save_fig` which writes both PNG and SVG.

## Main figures

| Figure | Script | Data sources | Description |
|--------|--------|-------------|-------------|
| Fig 1 | `fig01_study_sites.py` | Favela boundaries, buildings, RJ DTM | Rio overview map + 5 site insets with building footprints |
| Fig 2 | `fig02_morphometric_distributions.py` | 10m grids (all sites) | 5-panel violins: SVF, lambda_p, lambda_f, sigma_H, slope |
| Fig 3 | `fig03_svf_lambda_coupling.py` | 10m grids (pooled), campaign patches | Per-site 2D KDE contours (50%/90%) with marginals |
| Fig 4 | `fig04_sampling_design.py` | Campaign patches, grids, strata summary | Feature space + allocation bar chart + site maps |
| Fig 5 | `fig05_morphometric_maps.py` | Vidigal + Mare grids, buildings, DTM | 2x3 spatial maps: SVF, lambda_p, slope |
| Fig 6 | `fig06_terrain_aspect.py` | Per-site `svf_streets_solar.gpkg` | Cross-site SVF↔solar dissociation; per-site aspect-quadrant table |
| Fig 7 | `fig07_solar_envelope_vidigal.py` | Vidigal solar envelope | Single-site envelope deep dive |
| Fig 8 | `fig08_solar_cross_site.py` | All-site solar gpkg | Cross-site annual sun-hours summary; prints Maré N–S contrast |

## BRISA ventilation figures (manuscript Fig 0.x track)

These map to the ventilation-paper Fig 0.3–0.5 and read CFD `U_mean` +
street solar. They coexist with the same-numbered main figures above
(different `fig0X_` stems). Exports also live under
`docs/manuscript/figures/exports/` for the manuscript track.

| Figure | Script | Data sources | Description |
|--------|--------|-------------|-------------|
| Fig 0.3 | `fig03_ventilation_solar.py` | Per-site grids + `cfd_analysis/grid_with_cfd.gpkg` + solar | Cross-site environmental performance: representative patch maps + pooled distributions |
| Fig 0.4 ★ | `fig04_diagnostic_taxonomy.py` | `cfd_analysis/grid_with_cfd.gpkg`, solar | Headline four-state per-cell diagnostic (adequate / ventilation / sunlight / compound constraint) at pooled λf p75 |
| Fig 0.5 | `fig05_predictors.py` | Grids + CFD `U_mean`, solar | Predictor partial-dependence (SVF, λf, slope) + SVF→U_mean changepoint + typology contrast |

## Supplementary figures

| Figure | Script | Data sources | Description |
|--------|--------|-------------|-------------|
| Fig S1 | `figS1_correlation_matrices.py` | 10m grids (all sites) | 5 correlation heatmaps, consistent scale |
| Fig S2 | `figS2_context_extension.py` | Extended buildings, buffer analysis | Context validation: buildings, candidate pool, patch maps |
| Fig S3 | `figS3_resolution_sensitivity.py` | 10m + 20m grids | Distribution overlay: 10m vs 20m KDEs across 5 sites × 5 indicators (canonical). `--variants` flag regenerates supplementary scatter and difference-map variants. |
| Fig S4 | `figS4_patch_thumbnails.py` | Campaign patches, buildings | Per-site patch thumbnails (one page per site) |
| Fig S5 | `figS5_wind_roses.py` | Per-site wind_rose.json | Cross-site wind roses |
| Fig S (envelope) | `figS_solar_envelope.py` | Per-site `svf_streets_solar.gpkg` | Per-site seasonal solar-envelope deep dives (winter/summer/annual), non-Vidigal sites |
| Fig S (aspect) | `figS_terrain_aspect_spatial.py` | Vidigal `svf_streets_solar.gpkg` | Vidigal street-point set in four encodings: terrain aspect, SVF, solar hours, SVF−solar disagreement map |

## Style

`fig_style.py` is the **shared** style module — both this track and the
manuscript track import from it for consistent palettes and typography.
Site colors: Tol muted palette (warm = hillside, cool = flatland). White
backgrounds on all exports. Specs target TR (A4) and slide aspect ratios.

## SVF data source

The SVF column in morphometric grids is aggregated from passageway-level
sample points (1.5 m height, 145-ray Tregenza hemisphere), not grid-cell
centroids. See docstring in `fig05_morphometric_maps.py` for details.

## Cross-reference

For figures **targeted at the final academic paper**, see
`docs/manuscript/README.md`. Those have different communication
objectives, different export paths (`docs/manuscript/figures/exports/`
is tracked, this directory's `exports/` is not), and may use journal
formatting conventions when they diverge from the TR.

## Revision history

- **v4** (2026-05-21): Two-track convention codified. Directory README
  retitled from "Paper Figures — Brisa+ Nature Cities" to "technical-
  report + presentations track" to match its actual purpose; the
  paper-candidate track lives at `docs/manuscript/figures/`. Fig 6
  upgraded to cross-site (was Vidigal-only). Fig 8 (solar cross-site)
  added. Figure tables synced with current scripts on disk.
- **v3** (2026-04-09): Fig 3 redesigned as per-site 2D KDE contours with
  marginal distributions. Fig 5 SVF data source verified (passageway-
  aggregated, no centroid artifacts). All figures white backgrounds.
- **v2** (2026-04-09): Fig 3 structurally empty region emphasized. Fig 2
  porosity dropped (redundant with lambda_p). Fig 4 mini-maps enlarged.
  Global fix: transparent -> white backgrounds.
- **v1** (2026-04-09): Initial generation.
