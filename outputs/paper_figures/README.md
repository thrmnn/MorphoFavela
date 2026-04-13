# Paper Figures — Brisa+ Nature Cities

Publication figures for the morphometric analysis of informal settlements.
All scripts are standalone and regenerable from pipeline outputs.

## Regenerate all figures

```bash
cd /path/to/IVF
for f in outputs/paper_figures/fig*.py; do python3 "$f"; done
```

## Main Figures

| Figure | Script | Data sources | Description |
|--------|--------|-------------|-------------|
| Fig 1 | `fig01_study_sites.py` | Favela boundaries, buildings, RJ DTM | Rio overview map + 5 site insets with building footprints |
| Fig 2 | `fig02_morphometric_distributions.py` | 10m grids (all sites) | 5-panel violins: SVF, lambda_p, lambda_f, sigma_H, slope |
| Fig 3 | `fig03_svf_lambda_coupling.py` | 10m grids (pooled), campaign patches | Per-site 2D KDE contours (50%/90%) with marginals |
| Fig 4 | `fig04_sampling_design.py` | Campaign patches, grids, strata summary | Feature space + allocation bar chart + site maps |
| Fig 5 | `fig05_morphometric_maps.py` | Vidigal + Mare grids, buildings, DTM | 2x3 spatial maps: SVF, lambda_p, slope |

## Supplementary Figures

| Figure | Script | Data sources | Description |
|--------|--------|-------------|-------------|
| Fig S1 | `figS1_correlation_matrices.py` | 10m grids (all sites) | 5 correlation heatmaps, consistent scale |
| Fig S2 | `figS2_context_extension.py` | Extended buildings, buffer analysis | Context validation: buildings, candidate pool, patch maps |
| Fig S3 | `figS3_resolution_sensitivity.py` | 10m + 20m grids | Distribution overlay: 10m vs 20m KDEs across 5 sites × 5 indicators (canonical). `--variants` flag regenerates supplementary scatter and difference-map variants. |
| Fig S4 | `figS4_patch_thumbnails.py` | Campaign patches, buildings | Per-site patch thumbnails (one page per site) |

## Style

All figures use `fig_style.py` for consistent colors, fonts, and sizing.
Site colors: Tol muted palette (warm = hillside, cool = flatland).
White backgrounds on all exports.

## Specs

- Nature Cities single column: 88 mm | double column: 180 mm
- Font: 7 pt Liberation Sans
- Resolution: 600 DPI PNG + SVG vector
- Exports in `exports/`

## SVF data source

The SVF column in morphometric grids is aggregated from passageway-level
sample points (1.5 m height, 145-ray Tregenza hemisphere), not grid-cell
centroids. See docstring in `fig05_morphometric_maps.py` for details.

## Revision history

- **v3** (2026-04-09): Fig 3 redesigned as per-site 2D KDE contours with
  marginal distributions. Fig 5 SVF data source verified (passageway-
  aggregated, no centroid artifacts). All figures white backgrounds.
- **v2** (2026-04-09): Fig 3 structurally empty region emphasized. Fig 2
  porosity dropped (redundant with lambda_p). Fig 4 mini-maps enlarged.
  Global fix: transparent -> white backgrounds.
- **v1** (2026-04-09): Initial generation.
