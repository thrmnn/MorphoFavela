#!/usr/bin/env python3
"""Classify settlement typology across one or more study areas.

Usage:
    python scripts/classify_typology.py --areas vidigal_tls copacabana --n-clusters 4
    python scripts/classify_typology.py --areas vidigal_tls --features bcr far sigma_h
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_output_dir, get_comparative_analysis_dir
from src.typology import (
    classify_typology,
    compute_cluster_profiles,
    find_optimal_k,
    normalize_features,
)
from src.visualization import (
    plot_cluster_profiles,
    plot_elbow_silhouette,
    plot_typology_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("classify_typology")

DEFAULT_FEATURES = ["bcr", "far", "sigma_h", "lambda_f"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify settlement typology via clustering."
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        required=True,
        help="One or more area names (e.g. vidigal_tls copacabana)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters (default: auto-detect via silhouette analysis)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature columns to use for clustering",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip visualization output",
    )
    args = parser.parse_args()

    output_dir = get_comparative_analysis_dir() / "typology"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Areas: %s", args.areas)
    logger.info("Features: %s", args.features)
    logger.info("Output dir: %s", output_dir)

    # ------------------------------------------------ load zone metrics
    frames: list[gpd.GeoDataFrame] = []
    for area in args.areas:
        metrics_path = (
            get_area_output_dir(area) / "urban_morphology" / "zone_metrics.gpkg"
        )
        if not metrics_path.exists():
            logger.error(
                "Zone metrics not found for area '%s' at %s. "
                "Run compute_urban_morphology.py first.",
                area,
                metrics_path,
            )
            sys.exit(1)

        logger.info("Loading zone metrics for '%s' from %s", area, metrics_path)
        gdf = gpd.read_file(metrics_path)
        gdf["area"] = area
        frames.append(gdf)

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
    logger.info("Combined %d zones from %d area(s).", len(combined), len(args.areas))

    # Validate that requested features exist
    missing = [f for f in args.features if f not in combined.columns]
    if missing:
        logger.error(
            "Feature column(s) not found in zone metrics: %s. Available columns: %s",
            missing,
            list(combined.columns),
        )
        sys.exit(1)

    # ------------------------------------------------ normalize features
    t0 = time.time()
    normalized = normalize_features(combined, args.features)
    logger.info("Features normalized in %.1fs.", time.time() - t0)

    # ------------------------------------------------ find optimal k
    feature_matrix = normalized[args.features].to_numpy(dtype=float)

    optimal_k_result: dict | None = None
    n_clusters = args.n_clusters

    if n_clusters is None:
        logger.info("Auto-detecting optimal k via silhouette analysis...")
        t0 = time.time()
        optimal_k_result = find_optimal_k(feature_matrix)
        n_clusters = optimal_k_result["optimal_k"]
        logger.info("Optimal k=%d detected in %.1fs.", n_clusters, time.time() - t0)
    else:
        logger.info("Using user-specified n_clusters=%d.", n_clusters)

    # ------------------------------------------------ classify
    t0 = time.time()
    classified = classify_typology(normalized, args.features, n_clusters=n_clusters)
    logger.info("Typology classification completed in %.1fs.", time.time() - t0)

    # ------------------------------------------------ cluster profiles
    profiles = compute_cluster_profiles(classified, args.features)
    logger.info("Cluster profiles:\n%s", profiles.to_string())

    # ------------------------------------------------ save outputs
    # Typology clusters (GeoPackage)
    clusters_path = output_dir / "typology_clusters.gpkg"
    classified.to_file(clusters_path, driver="GPKG")
    logger.info("Saved typology clusters to %s", clusters_path)

    # Cluster profiles (CSV)
    profiles_path = output_dir / "cluster_profiles.csv"
    profiles.to_csv(profiles_path)
    logger.info("Saved cluster profiles to %s", profiles_path)

    # Optimal k analysis (JSON)
    optimal_k_path = output_dir / "optimal_k.json"
    if optimal_k_result is not None:
        with open(optimal_k_path, "w") as f:
            json.dump(optimal_k_result, f, indent=2)
        logger.info("Saved silhouette analysis to %s", optimal_k_path)
    else:
        # Save a record that k was user-specified
        with open(optimal_k_path, "w") as f:
            json.dump(
                {"optimal_k": n_clusters, "source": "user-specified"},
                f,
                indent=2,
            )
        logger.info("Saved k specification to %s", optimal_k_path)

    # -------------------------------------------------------- visualizations
    if not args.skip_plots:
        maps_dir = output_dir / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating typology map...")
        plot_typology_map(classified, maps_dir / "typology_clusters.png")

        logger.info("Generating cluster profiles chart...")
        plot_cluster_profiles(
            profiles, maps_dir / "cluster_profiles.png", features=args.features
        )

        if optimal_k_result is not None:
            logger.info("Generating elbow/silhouette plot...")
            plot_elbow_silhouette(optimal_k_result, maps_dir / "elbow_silhouette.png")
    else:
        logger.info("Skipping plots (--skip-plots).")

    logger.info("Done.")


if __name__ == "__main__":
    main()
