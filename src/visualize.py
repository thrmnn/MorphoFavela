"""Simple visualization functions."""

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging

from src.config import DPI, FIGURE_SIZE, COLORMAP_HEIGHT, COLORMAP_VOLUME

logger = logging.getLogger(__name__)


def create_thematic_maps(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """
    Create thematic maps for height and volume.
    
    Args:
        gdf: GeoDataFrame with height and volume columns
        output_path: Path to save the figure
    """
    if 'height' not in gdf.columns or 'volume' not in gdf.columns:
        raise ValueError("GeoDataFrame must have 'height' and 'volume' columns")
    
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # Height map
    ax1 = axes[0]
    gdf.plot(column='height', ax=ax1, cmap=COLORMAP_HEIGHT, 
             legend=True, legend_kwds={'label': 'Height (m)'})
    ax1.set_title('Building Height')
    ax1.set_axis_off()
    
    # Volume map
    ax2 = axes[1]
    gdf.plot(column='volume', ax=ax2, cmap=COLORMAP_VOLUME,
             legend=True, legend_kwds={'label': 'Volume (m³)'})
    ax2.set_title('Building Volume')
    ax2.set_axis_off()
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved maps to {output_path}")


def create_multi_panel_summary(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """
    Create a multi-panel summary figure with all key metrics.
    
    Args:
        gdf: GeoDataFrame with metric columns
        output_path: Path to save the figure
    """
    required_cols = ['height', 'volume', 'area', 'hw_ratio']
    missing = [col for col in required_cols if col not in gdf.columns]
    if missing:
        raise ValueError(f"GeoDataFrame must have columns: {missing}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Height map
    ax1 = axes[0, 0]
    gdf.plot(column='height', ax=ax1, cmap=COLORMAP_HEIGHT,
             legend=True, legend_kwds={'label': 'Height (m)'})
    ax1.set_title('Building Height', fontsize=12, fontweight='bold')
    ax1.set_axis_off()
    
    # Volume map
    ax2 = axes[0, 1]
    gdf.plot(column='volume', ax=ax2, cmap=COLORMAP_VOLUME,
             legend=True, legend_kwds={'label': 'Volume (m³)'})
    ax2.set_title('Building Volume', fontsize=12, fontweight='bold')
    ax2.set_axis_off()
    
    # Area map
    ax3 = axes[1, 0]
    gdf.plot(column='area', ax=ax3, cmap='YlOrRd',
             legend=True, legend_kwds={'label': 'Area (m²)'})
    ax3.set_title('Footprint Area', fontsize=12, fontweight='bold')
    ax3.set_axis_off()
    
    # Street canyon ratio (h/w) map
    ax4 = axes[1, 1]
    # Use percentile-based scaling to handle outliers
    hw_data = gdf['hw_ratio'].dropna()
    if len(hw_data) > 0:
        vmin = hw_data.quantile(0.02)  # 2nd percentile
        vmax = hw_data.quantile(0.98)  # 98th percentile
        gdf.plot(column='hw_ratio', ax=ax4, cmap='coolwarm',
                 legend=True, legend_kwds={'label': 'H/W Ratio'},
                 vmin=vmin, vmax=vmax)
        ax4.set_title(f'Street Canyon Ratio (H/W)\n(scale: {vmin:.2f} - {vmax:.2f})', 
                     fontsize=12, fontweight='bold')
    else:
        gdf.plot(column='hw_ratio', ax=ax4, cmap='coolwarm',
                 legend=True, legend_kwds={'label': 'H/W Ratio'})
        ax4.set_title('Street Canyon Ratio (Height/Width)', fontsize=12, fontweight='bold')
    ax4.set_axis_off()
    
    plt.suptitle('Morphometric Analysis Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved multi-panel summary to {output_path}")


def create_statistical_distributions(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """
    Create histograms and box plots for all metrics.
    
    Args:
        gdf: GeoDataFrame with metric columns
        output_path: Path to save the figure
    """
    metric_cols = ['height', 'area', 'volume', 'perimeter', 'hw_ratio']
    available_cols = [col for col in metric_cols if col in gdf.columns]
    
    if not available_cols:
        raise ValueError("No metric columns found in GeoDataFrame")
    
    n_metrics = len(available_cols)
    fig, axes = plt.subplots(2, n_metrics, figsize=(4*n_metrics, 10))
    
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)
    
    # Labels for axes
    labels = {
        'height': 'Height (m)',
        'area': 'Area (m²)',
        'volume': 'Volume (m³)',
        'perimeter': 'Perimeter (m)',
        'hw_ratio': 'H/W Ratio'
    }
    
    for idx, col in enumerate(available_cols):
        data = gdf[col].dropna()
        
        # Histogram
        ax_hist = axes[0, idx]
        ax_hist.hist(data, bins=50, edgecolor='black', alpha=0.7)
        ax_hist.set_xlabel(labels.get(col, col))
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title(f'{col.capitalize()} Distribution')
        ax_hist.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = data.mean()
        median_val = data.median()
        ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax_hist.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax_hist.legend(fontsize=8)
        
        # Box plot
        ax_box = axes[1, idx]
        bp = ax_box.boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax_box.set_ylabel(labels.get(col, col))
        ax_box.set_title(f'{col.capitalize()} Box Plot')
        ax_box.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Statistical Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved statistical distributions to {output_path}")


def create_scatter_plots(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """
    Create scatter plots showing relationships between metrics.
    
    Args:
        gdf: GeoDataFrame with metric columns
        output_path: Path to save the figure
    """
    required_cols = ['height', 'area', 'volume']
    missing = [col for col in required_cols if col not in gdf.columns]
    if missing:
        raise ValueError(f"GeoDataFrame must have columns: {missing}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Height vs Area
    ax1 = axes[0, 0]
    ax1.scatter(gdf['area'], gdf['height'], alpha=0.5, s=10, edgecolors='black', linewidth=0.1)
    ax1.set_xlabel('Area (m²)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Height vs Area')
    ax1.grid(True, alpha=0.3)
    
    # Volume vs Area
    ax2 = axes[0, 1]
    ax2.scatter(gdf['area'], gdf['volume'], alpha=0.5, s=10, edgecolors='black', linewidth=0.1)
    ax2.set_xlabel('Area (m²)')
    ax2.set_ylabel('Volume (m³)')
    ax2.set_title('Volume vs Area')
    ax2.grid(True, alpha=0.3)
    
    # Height vs H/W Ratio
    if 'hw_ratio' in gdf.columns:
        ax3 = axes[1, 0]
        ax3.scatter(gdf['hw_ratio'], gdf['height'], alpha=0.5, s=10, edgecolors='black', linewidth=0.1)
        ax3.set_xlabel('H/W Ratio')
        ax3.set_ylabel('Height (m)')
        ax3.set_title('Height vs H/W Ratio')
        ax3.grid(True, alpha=0.3)
    
    # Volume vs Height
    ax4 = axes[1, 1]
    ax4.scatter(gdf['height'], gdf['volume'], alpha=0.5, s=10, edgecolors='black', linewidth=0.1)
    ax4.set_xlabel('Height (m)')
    ax4.set_ylabel('Volume (m³)')
    ax4.set_title('Volume vs Height')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Metric Relationships', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved scatter plots to {output_path}")



