"""Visualization package for building metrics and zone-level morphology."""

from src.visualization.building import (
    create_metric_distributions,
    create_metric_map,
    create_multi_panel_summary,
    create_scatter_plots,
    create_statistical_distributions,
    create_thematic_maps,
)
from src.visualization.morphology import (
    plot_cluster_profiles,
    plot_elbow_silhouette,
    plot_lisa_clusters,
    plot_moran_scatter,
    plot_typology_map,
    plot_zone_metrics_panel,
)

__all__ = [
    # Building-level metrics
    "create_metric_distributions",
    "create_metric_map",
    "create_multi_panel_summary",
    "create_scatter_plots",
    "create_statistical_distributions",
    "create_thematic_maps",
    # Zone-level morphology
    "plot_cluster_profiles",
    "plot_elbow_silhouette",
    "plot_lisa_clusters",
    "plot_moran_scatter",
    "plot_typology_map",
    "plot_zone_metrics_panel",
]
