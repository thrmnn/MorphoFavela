#!/usr/bin/env python3
"""
Generate street SVF distribution visualization from existing data.

This script reads point-level SVF data from a GPKG file and generates
a distribution histogram showing the proportion of street network at
different SVF values.

Usage:
    python scripts/plot_street_svf_distribution.py \
        --input outputs/riodaspedras/svf_streets/street_svf_points.gpkg \
        --output outputs/riodaspedras/svf_streets/street_svf_distribution.png
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_street_svf_distribution(
    points_gdf: gpd.GeoDataFrame, output_path: Path, comfort_threshold: float = 0.3
):
    """
    Create a distribution histogram of street-level SVF values.

    Args:
        points_gdf: GeoDataFrame with point-level SVF data (must have 'svf' column)
        output_path: Path to save the plot
        comfort_threshold: SVF threshold value to mark as comfort level (default: 0.3)
    """
    if "svf" not in points_gdf.columns:
        raise ValueError("GeoDataFrame must have 'svf' column")

    # Distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate histogram - use weights to get proportions as percentages
    svf_values = points_gdf["svf"].values
    total_points = len(svf_values)

    # Create histogram with percentage on y-axis
    counts, bins, patches = ax.hist(
        svf_values,
        bins=50,
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
        weights=np.ones_like(svf_values) * (100.0 / total_points),
    )

    ax.set_xlabel("SVF Value", fontsize=12)
    ax.set_ylabel("Proportion of street network (%)", fontsize=12)
    ax.set_title(
        "Distribution of Street-Level SVF Values", fontsize=14, fontweight="bold"
    )
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add comfort SVF threshold at specified value
    ax.axvline(
        comfort_threshold,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Comfort threshold (SVF = {comfort_threshold})",
    )

    # Add statistics
    mean_svf = points_gdf["svf"].mean()
    median_svf = points_gdf["svf"].median()
    ax.axvline(
        mean_svf,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_svf:.3f}",
    )
    ax.axvline(
        median_svf,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_svf:.3f}",
    )
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved distribution plot to {output_path}")


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(
        description="Generate street SVF distribution visualization from existing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to point-level GPKG file (street_svf_points.gpkg)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: same directory as input with name street_svf_distribution.png)",
    )
    parser.add_argument(
        "--comfort-threshold",
        type=float,
        default=0.3,
        help="SVF comfort threshold value (default: 0.3)",
    )
    parser.add_argument(
        "--area",
        type=str,
        default=None,
        help="Area name (e.g., riodaspedras) for automatic path resolution",
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    elif args.area:
        from src.config import get_area_output_dir

        output_path = (
            get_area_output_dir(args.area)
            / "svf_streets"
            / "street_svf_distribution.png"
        )
    else:
        output_path = input_path.parent / "street_svf_distribution.png"

    print("=" * 60)
    print("STREET SVF DISTRIBUTION VISUALIZATION")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Comfort threshold: {args.comfort_threshold}")
    print("=" * 60)

    # Load point-level data
    print(f"Loading data from {input_path}...")
    points_gdf = gpd.read_file(input_path)
    print(f"  Loaded {len(points_gdf)} points")

    if "svf" not in points_gdf.columns:
        print("Error: Input file must have 'svf' column")
        sys.exit(1)

    # Generate visualization
    print("Generating distribution plot...")
    create_street_svf_distribution(points_gdf, output_path, args.comfort_threshold)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total points: {len(points_gdf)}")
    print(f"Mean SVF: {points_gdf['svf'].mean():.4f}")
    print(f"Median SVF: {points_gdf['svf'].median():.4f}")
    print(f"Std SVF: {points_gdf['svf'].std():.4f}")
    print(f"Min SVF: {points_gdf['svf'].min():.4f}")
    print(f"Max SVF: {points_gdf['svf'].max():.4f}")

    # Calculate proportion below comfort threshold
    below_threshold = (points_gdf["svf"] < args.comfort_threshold).sum()
    pct_below = (below_threshold / len(points_gdf)) * 100
    print(
        f"\nPoints below comfort threshold ({args.comfort_threshold}): {below_threshold} ({pct_below:.1f}%)"
    )
    print("=" * 60)
    print(f"\nPlot saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
