# `src/visualization/`

Building-level + zone-level chart helpers used by the comparative
analysis and per-area reports. Distinct from
`outputs/paper_figures/` (paper-grade figures) and `src/morphometry/`
(audit-grade figures): this package is the small set of generic
maps + distributions used inside the comparative analysis pipeline
(`scripts/compare_areas.py`).

## What it does

Two layers, separated by what data they take:

1. **Building-level** (`building.py`) — input is a buildings
   GeoDataFrame with morphometric columns (height, area, volume,
   inter-building distance, h/w ratio). Outputs thematic maps,
   multi-panel summaries, statistical distributions, scatter plots.
2. **Zone-level / typology** (`morphology.py`) — input is a
   typology assignment + per-zone metrics. Outputs LISA cluster
   maps, Moran scatter, cluster-profile plots, elbow/silhouette
   diagnostics for the k-means clustering step.

## Public API (re-exported in `__init__`)

Building-level:

- `create_thematic_maps(...)`
- `create_metric_map(...)`
- `create_multi_panel_summary(...)`
- `create_statistical_distributions(...)`
- `create_metric_distributions(...)`
- `create_scatter_plots(...)`

Zone-level / typology:

- `plot_typology_map(...)`
- `plot_zone_metrics_panel(...)`
- `plot_lisa_clusters(...)`
- `plot_moran_scatter(...)`
- `plot_cluster_profiles(...)`
- `plot_elbow_silhouette(...)`

## Typical usage

The functions are called from `scripts/compare_areas.py` (formal vs
informal report) and `scripts/calculate_morphology_metrics.py` (per-
area summaries). Direct usage:

```python
from src.visualization import create_thematic_maps

create_thematic_maps(
    buildings_gdf,
    metrics=["height", "volume", "inter_building_distance"],
    output_dir="outputs/maps/",
)
```

## When to use which package

- Need a paper figure with the cross-site palette + Tol colours →
  `outputs/paper_figures/`
- Need a per-site audit page → `src/morphometry/figures.py`
- Need a quick map / distribution as part of a comparative or
  audit script → `src/visualization/` (this package)

Keep these distinctions clean — paper figures must remain
hand-curated for journal submission; the others can drift if a
script is refactored.
