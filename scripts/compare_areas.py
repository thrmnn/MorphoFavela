#!/usr/bin/env python3
"""
Comparative Analysis: Formal vs Informal Settlements

Compare Vidigal (informal) and Copacabana (formal) across all metrics.

Usage:
    python scripts/compare_areas.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from pathlib import Path
import sys
import logging
from scipy import stats

# Set up clean academic design style
plt.style.use('seaborn-v0_8-whitegrid')
# Swiss design: clean, minimal colors
SWISS_COLORS = {
    'primary': '#000000',      # Black
    'secondary': '#666666',    # Dark gray
    'accent': '#0066CC',       # Blue accent
    'vidigal': '#E63946',      # Red for informal
    'copacabana': '#0066CC',   # Blue for formal
    'background': '#FFFFFF',   # White
    'light_gray': '#F5F5F5',   # Light gray for backgrounds
    'text': '#1A1A1A'          # Near black for text
}

# Set default font to clean sans-serif
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelcolor'] = SWISS_COLORS['text']
plt.rcParams['text.color'] = SWISS_COLORS['text']
plt.rcParams['axes.edgecolor'] = SWISS_COLORS['secondary']
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    get_area_analysis_dir,
    get_comparative_analysis_dir,
    SUPPORTED_AREAS,
    is_formal_area
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_area_data(area: str) -> dict:
    """Load all analysis data for an area."""
    logger.info(f"Loading data for {area}...")
    
    data = {
        'area': area,
        'area_type': 'formal' if is_formal_area(area) else 'informal'
    }
    
    # Load building metrics
    metrics_dir = get_area_analysis_dir(area, 'metrics')
    metrics_file = metrics_dir / 'buildings_with_metrics.gpkg'
    if metrics_file.exists():
        data['buildings'] = gpd.read_file(metrics_file)
        logger.info(f"  Loaded {len(data['buildings'])} buildings")
        
        # Calculate area of study region (convex hull of buildings)
        data['study_area_m2'] = data['buildings'].total_bounds
        bounds = data['buildings'].total_bounds
        data['study_area_m2'] = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        
        # Building count
        data['building_count'] = len(data['buildings'])
        
        # Building density (buildings per km²)
        area_km2 = data['study_area_m2'] / 1e6
        data['building_density'] = data['building_count'] / area_km2 if area_km2 > 0 else 0
    else:
        logger.warning(f"  Metrics file not found: {metrics_file}")
        data['buildings'] = None
    
    # Load rasters
    for raster_type in ['svf', 'solar', 'porosity', 'deprivation_raster']:
        raster_dir = get_area_analysis_dir(area, raster_type)
        raster_file = raster_dir / f"{'solar_access' if raster_type == 'solar' else 'svf' if raster_type == 'svf' else 'porosity' if raster_type == 'porosity' else 'hotspot_index'}.npy"
        
        if raster_file.exists():
            raster = np.load(raster_file)
            data[raster_type] = raster
            # Calculate valid (non-NaN) pixels
            valid = ~np.isnan(raster)
            data[f'{raster_type}_valid_pixels'] = valid.sum()
            data[f'{raster_type}_total_pixels'] = raster.size
            data[f'{raster_type}_valid_fraction'] = valid.sum() / raster.size if raster.size > 0 else 0
            logger.info(f"  Loaded {raster_type}: {raster.shape}")
        else:
            logger.warning(f"  {raster_type} file not found: {raster_file}")
            data[raster_type] = None
    
    return data


def compare_morphometric_metrics(vidigal_data: dict, copacabana_data: dict) -> pd.DataFrame:
    """Compare morphometric metrics between areas."""
    logger.info("Comparing morphometric metrics...")
    
    metrics = ['height', 'area', 'volume', 'perimeter', 'hw_ratio', 'inter_building_distance']
    
    comparison = []
    
    for metric in metrics:
        if vidigal_data['buildings'] is not None and metric in vidigal_data['buildings'].columns:
            vidigal_vals = vidigal_data['buildings'][metric].dropna()
            copa_vals = copacabana_data['buildings'][metric].dropna() if copacabana_data['buildings'] is not None and metric in copacabana_data['buildings'].columns else pd.Series()
            
            if len(vidigal_vals) > 0 and len(copa_vals) > 0:
                row = {
                    'metric': metric,
                    'vidigal_mean': vidigal_vals.mean(),
                    'vidigal_median': vidigal_vals.median(),
                    'vidigal_std': vidigal_vals.std(),
                    'vidigal_min': vidigal_vals.min(),
                    'vidigal_max': vidigal_vals.max(),
                    'copa_mean': copa_vals.mean(),
                    'copa_median': copa_vals.median(),
                    'copa_std': copa_vals.std(),
                    'copa_min': copa_vals.min(),
                    'copa_max': copa_vals.max(),
                    'difference_mean': copa_vals.mean() - vidigal_vals.mean(),
                    'ratio_mean': copa_vals.mean() / vidigal_vals.mean() if vidigal_vals.mean() > 0 else np.nan,
                }
                
                # Statistical test (Mann-Whitney U for non-parametric)
                try:
                    stat, p_value = stats.mannwhitneyu(vidigal_vals, copa_vals, alternative='two-sided')
                    row['p_value'] = p_value
                    row['significant'] = p_value < 0.05
                except:
                    row['p_value'] = np.nan
                    row['significant'] = False
                
                comparison.append(row)
    
    # Add area-level metrics
    comparison.append({
        'metric': 'building_count',
        'vidigal_mean': vidigal_data['building_count'],
        'copa_mean': copacabana_data['building_count'],
        'difference_mean': copacabana_data['building_count'] - vidigal_data['building_count'],
        'ratio_mean': copacabana_data['building_count'] / vidigal_data['building_count'] if vidigal_data['building_count'] > 0 else np.nan,
    })
    
    comparison.append({
        'metric': 'building_density_per_km2',
        'vidigal_mean': vidigal_data['building_density'],
        'copa_mean': copacabana_data['building_density'],
        'difference_mean': copacabana_data['building_density'] - vidigal_data['building_density'],
        'ratio_mean': copacabana_data['building_density'] / vidigal_data['building_density'] if vidigal_data['building_density'] > 0 else np.nan,
    })
    
    return pd.DataFrame(comparison)


def compare_raster_metrics(vidigal_data: dict, copacabana_data: dict, raster_type: str) -> dict:
    """Compare raster-based metrics between areas."""
    vid_raster = vidigal_data.get(raster_type)
    copa_raster = copacabana_data.get(raster_type)
    
    if vid_raster is None or copa_raster is None:
        return None
    
    vid_valid = vid_raster[~np.isnan(vid_raster)]
    copa_valid = copa_raster[~np.isnan(copa_raster)]
    
    if len(vid_valid) == 0 or len(copa_valid) == 0:
        return None
    
    # Area normalization: calculate statistics per unit area
    vid_area = vidigal_data['study_area_m2']
    copa_area = copacabana_data['study_area_m2']
    
    comparison = {
        'metric': raster_type,
        'vidigal_mean': vid_valid.mean(),
        'vidigal_median': np.median(vid_valid),
        'vidigal_std': vid_valid.std(),
        'vidigal_min': vid_valid.min(),
        'vidigal_max': vid_valid.max(),
        'vidigal_q10': np.percentile(vid_valid, 10),
        'vidigal_q90': np.percentile(vid_valid, 90),
        'copa_mean': copa_valid.mean(),
        'copa_median': np.median(copa_valid),
        'copa_std': copa_valid.std(),
        'copa_min': copa_valid.min(),
        'copa_max': copa_valid.max(),
        'copa_q10': np.percentile(copa_valid, 10),
        'copa_q90': np.percentile(copa_valid, 90),
        'difference_mean': copa_valid.mean() - vid_valid.mean(),
        'ratio_mean': copa_valid.mean() / vid_valid.mean() if vid_valid.mean() > 0 else np.nan,
        'vidigal_area_m2': vid_area,
        'copa_area_m2': copa_area,
        'area_ratio': copa_area / vid_area if vid_area > 0 else np.nan,
    }
    
    # Statistical test
    try:
        stat, p_value = stats.mannwhitneyu(vid_valid, copa_valid, alternative='two-sided')
        comparison['p_value'] = p_value
        comparison['significant'] = p_value < 0.05
    except:
        comparison['p_value'] = np.nan
        comparison['significant'] = False
    
    # Threshold-based comparisons (specific to metric type)
    if raster_type == 'svf':
        comparison['vidigal_low_svf_fraction'] = (vid_valid < 0.5).sum() / len(vid_valid)
        comparison['copa_low_svf_fraction'] = (copa_valid < 0.5).sum() / len(copa_valid)
        comparison['vidigal_high_svf_fraction'] = (vid_valid > 0.7).sum() / len(vid_valid)
        comparison['copa_high_svf_fraction'] = (copa_valid > 0.7).sum() / len(copa_valid)
    
    elif raster_type == 'solar':
        comparison['vidigal_solar_deficit_fraction'] = (vid_valid < 2.0).sum() / len(vid_valid)
        comparison['copa_solar_deficit_fraction'] = (copa_valid < 2.0).sum() / len(copa_valid)
        comparison['vidigal_acceptable_solar_fraction'] = (vid_valid >= 3.0).sum() / len(vid_valid)
        comparison['copa_acceptable_solar_fraction'] = (copa_valid >= 3.0).sum() / len(copa_valid)
    
    elif raster_type == 'porosity':
        comparison['vidigal_low_porosity_fraction'] = (vid_valid < 0.3).sum() / len(vid_valid)
        comparison['copa_low_porosity_fraction'] = (copa_valid < 0.3).sum() / len(copa_valid)
    
    elif raster_type == 'deprivation_raster':
        # Classify into percentiles
        vid_p90 = np.percentile(vid_valid, 90)
        vid_p80 = np.percentile(vid_valid, 80)
        copa_p90 = np.percentile(copa_valid, 90)
        copa_p80 = np.percentile(copa_valid, 80)
        
        comparison['vidigal_extreme_hotspot_fraction'] = (vid_valid >= vid_p90).sum() / len(vid_valid)
        comparison['copa_extreme_hotspot_fraction'] = (copa_valid >= copa_p90).sum() / len(copa_valid)
        comparison['vidigal_high_deprivation_fraction'] = (vid_valid >= vid_p80).sum() / len(vid_valid)
        comparison['copa_high_deprivation_fraction'] = (copa_valid >= copa_p80).sum() / len(copa_valid)
    
    return comparison


def create_comparison_visualizations(vidigal_data: dict, copacabana_data: dict, 
                                     morpho_comparison: pd.DataFrame, 
                                     raster_comparisons: dict,
                                     output_dir: Path):
    """Create comparison visualizations with clean Swiss design."""
    logger.info("Creating comparison visualizations...")
    
    # Create figure for PDF
    figs = []
    
    # 1. Morphometric metrics comparison - distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle('Morphometric Metrics: Distribution Comparison', 
                fontsize=14, fontweight='bold', color=SWISS_COLORS['text'], y=0.995)
    
    metrics = ['height', 'area', 'volume', 'perimeter', 'hw_ratio', 'inter_building_distance']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor('white')
        
        if vidigal_data['buildings'] is not None and metric in vidigal_data['buildings'].columns:
            vid_vals = vidigal_data['buildings'][metric].dropna()
            copa_vals = copacabana_data['buildings'][metric].dropna() if copacabana_data['buildings'] is not None and metric in copacabana_data['buildings'].columns else pd.Series()
            
            if len(vid_vals) > 0 and len(copa_vals) > 0:
                ax.hist(vid_vals, bins=30, alpha=0.7, label='Vidigal (Informal)', 
                       color=SWISS_COLORS['vidigal'], density=True, edgecolor='white', linewidth=0.5)
                ax.hist(copa_vals, bins=30, alpha=0.7, label='Copacabana (Formal)', 
                       color=SWISS_COLORS['copacabana'], density=True, edgecolor='white', linewidth=0.5)
                ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=9, color=SWISS_COLORS['text'])
                ax.set_ylabel('Density', fontsize=9, color=SWISS_COLORS['text'])
                ax.legend(fontsize=8, frameon=True, fancybox=False, edgecolor=SWISS_COLORS['secondary'], framealpha=0.9)
                ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(colors=SWISS_COLORS['secondary'], labelsize=8)
    
    plt.tight_layout()
    figs.append(('morphometric_distributions', fig))
    
    # 2. Morphometric metrics - box plots
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    fig.suptitle('Morphometric Metrics: Box Plot Comparison', 
                fontsize=14, fontweight='bold', color=SWISS_COLORS['text'], y=0.995)
    ax.set_facecolor('white')
    
    metrics_to_plot = [m for m in metrics if vidigal_data['buildings'] is not None and m in vidigal_data['buildings'].columns]
    data_to_plot = []
    labels = []
    
    for metric in metrics_to_plot:
        vid_vals = vidigal_data['buildings'][metric].dropna()
        copa_vals = copacabana_data['buildings'][metric].dropna() if copacabana_data['buildings'] is not None and metric in copacabana_data['buildings'].columns else pd.Series()
        
        if len(vid_vals) > 0 and len(copa_vals) > 0:
            data_to_plot.append([vid_vals.values, copa_vals.values])
            labels.append(metric.replace('_', ' ').title())
    
    if data_to_plot:
        positions = np.arange(len(metrics_to_plot)) * 3
        bp1 = ax.boxplot([d[0] for d in data_to_plot], positions=positions - 0.4, widths=0.35, 
                         patch_artist=True, tick_labels=[l + '\n(Vidigal)' for l in labels],
                         boxprops=dict(facecolor=SWISS_COLORS['vidigal'], alpha=0.8, edgecolor=SWISS_COLORS['text'], linewidth=0.8),
                         medianprops=dict(color=SWISS_COLORS['text'], linewidth=1.2),
                         whiskerprops=dict(color=SWISS_COLORS['secondary'], linewidth=0.8),
                         capprops=dict(color=SWISS_COLORS['secondary'], linewidth=0.8))
        bp2 = ax.boxplot([d[1] for d in data_to_plot], positions=positions + 0.4, widths=0.35,
                         patch_artist=True, tick_labels=[l + '\n(Copa)' for l in labels],
                         boxprops=dict(facecolor=SWISS_COLORS['copacabana'], alpha=0.8, edgecolor=SWISS_COLORS['text'], linewidth=0.8),
                         medianprops=dict(color=SWISS_COLORS['text'], linewidth=1.2),
                         whiskerprops=dict(color=SWISS_COLORS['secondary'], linewidth=0.8),
                         capprops=dict(color=SWISS_COLORS['secondary'], linewidth=0.8))
        
        ax.set_ylabel('Value', fontsize=10, color=SWISS_COLORS['text'])
        ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors=SWISS_COLORS['secondary'], labelsize=8)
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    figs.append(('morphometric_boxplots', fig))
    
    # 3. Raster comparisons - side by side
    raster_types = {'svf': 'Sky View Factor', 'solar': 'Solar Access (hours)', 
                   'porosity': 'Sectional Porosity', 'deprivation_raster': 'Deprivation Index'}
    
    # Use clean colormap (grayscale with accent)
    from matplotlib.colors import LinearSegmentedColormap
    clean_cmap = plt.cm.get_cmap('viridis').copy()
    
    for raster_type, title in raster_types.items():
        vid_raster = vidigal_data.get(raster_type)
        copa_raster = copacabana_data.get(raster_type)
        
        if vid_raster is not None and copa_raster is not None:
            # Calculate aspect ratios for each raster
            vid_shape = vid_raster.shape
            copa_shape = copa_raster.shape
            vid_aspect = vid_shape[1] / vid_shape[0]  # width/height
            copa_aspect = copa_shape[1] / copa_shape[0]
            
            # Create figure with appropriate width based on aspect ratios
            # Give each subplot enough space for its aspect ratio
            total_width = max(vid_aspect, copa_aspect) * 2 * 3  # 3 inches per unit height
            fig_height = 6
            fig = plt.figure(figsize=(total_width, fig_height))
            fig.patch.set_facecolor('white')
            fig.suptitle(f'{title}: Side-by-Side Comparison', 
                        fontsize=14, fontweight='bold', color=SWISS_COLORS['text'], y=0.98)
            
            # Create subplots with appropriate spacing
            # Use gridspec for better control
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(1, 2, figure=fig, width_ratios=[vid_aspect, copa_aspect], 
                         wspace=0.3, left=0.05, right=0.95, top=0.92, bottom=0.08)
            
            # Vidigal
            ax1 = fig.add_subplot(gs[0])
            ax1.set_facecolor('white')
            im1 = ax1.imshow(vid_raster, cmap=clean_cmap, origin='lower', aspect='equal')
            ax1.set_title(f'Vidigal (Informal)\nMean: {np.nanmean(vid_raster):.3f}', 
                         fontweight='bold', fontsize=11, color=SWISS_COLORS['text'], pad=10)
            ax1.axis('off')
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label(title, fontsize=9, color=SWISS_COLORS['text'])
            cbar1.ax.tick_params(colors=SWISS_COLORS['secondary'], labelsize=8)
            
            # Copacabana
            ax2 = fig.add_subplot(gs[1])
            ax2.set_facecolor('white')
            im2 = ax2.imshow(copa_raster, cmap=clean_cmap, origin='lower', aspect='equal')
            ax2.set_title(f'Copacabana (Formal)\nMean: {np.nanmean(copa_raster):.3f}', 
                         fontweight='bold', fontsize=11, color=SWISS_COLORS['text'], pad=10)
            ax2.axis('off')
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label(title, fontsize=9, color=SWISS_COLORS['text'])
            cbar2.ax.tick_params(colors=SWISS_COLORS['secondary'], labelsize=8)
            
            figs.append((f'{raster_type}_side_by_side', fig))
            
            # Distribution comparison
            vid_valid = vid_raster[~np.isnan(vid_raster)]
            copa_valid = copa_raster[~np.isnan(copa_raster)]
            
            if len(vid_valid) > 0 and len(copa_valid) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('white')
                fig.suptitle(f'{title}: Distribution Comparison', 
                            fontsize=14, fontweight='bold', color=SWISS_COLORS['text'], y=0.98)
                ax.set_facecolor('white')
                
                ax.hist(vid_valid, bins=50, alpha=0.7, label='Vidigal (Informal)', 
                       color=SWISS_COLORS['vidigal'], density=True, edgecolor='white', linewidth=0.5)
                ax.hist(copa_valid, bins=50, alpha=0.7, label='Copacabana (Formal)', 
                       color=SWISS_COLORS['copacabana'], density=True, edgecolor='white', linewidth=0.5)
                ax.axvline(np.nanmean(vid_valid), color=SWISS_COLORS['vidigal'], 
                          linestyle='--', linewidth=2, alpha=0.8,
                          label=f'Vidigal mean: {np.nanmean(vid_valid):.3f}')
                ax.axvline(np.nanmean(copa_valid), color=SWISS_COLORS['copacabana'], 
                          linestyle='--', linewidth=2, alpha=0.8,
                          label=f'Copacabana mean: {np.nanmean(copa_valid):.3f}')
                ax.set_xlabel(title, fontsize=10, color=SWISS_COLORS['text'])
                ax.set_ylabel('Density', fontsize=10, color=SWISS_COLORS['text'])
                ax.legend(fontsize=9, frameon=True, fancybox=False, 
                         edgecolor=SWISS_COLORS['secondary'], framealpha=0.9)
                ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(colors=SWISS_COLORS['secondary'], labelsize=8)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                figs.append((f'{raster_type}_distribution', fig))
    
    return figs


def generate_pdf_report(vidigal_data: dict, copacabana_data: dict,
                       morpho_comparison: pd.DataFrame,
                       raster_comparisons: dict,
                       figs: list,
                       output_dir: Path):
    """Generate comprehensive PDF report with Swiss design."""
    logger.info("Generating PDF report...")
    
    pdf_path = output_dir / 'comparison_report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Title page - clean Swiss design
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        # Main title
        fig.text(0.5, 0.75, 'COMPARATIVE ANALYSIS', 
                ha='center', va='center', fontsize=28, fontweight='bold',
                color=SWISS_COLORS['text'], family='sans-serif')
        
        # Subtitle line
        fig.text(0.5, 0.68, 'Formal vs Informal Settlements', 
                ha='center', va='center', fontsize=16,
                color=SWISS_COLORS['secondary'], family='sans-serif')
        
        # Study areas
        fig.text(0.5, 0.58, 'Vidigal (Informal)  ×  Copacabana (Formal)', 
                ha='center', va='center', fontsize=14,
                color=SWISS_COLORS['text'], family='sans-serif')
        
        # Decorative line
        from matplotlib.patches import Rectangle
        ax = fig.add_axes([0.25, 0.52, 0.5, 0.002])
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=SWISS_COLORS['accent']))
        ax.axis('off')
        
        # Date
        fig.text(0.5, 0.45, pd.Timestamp.now().strftime("%B %Y"), 
                ha='center', va='center', fontsize=11,
                color=SWISS_COLORS['secondary'], family='sans-serif')
        
        # Footer
        fig.text(0.5, 0.1, 'Morphometric & Environmental Performance Analysis', 
                ha='center', va='center', fontsize=10,
                color=SWISS_COLORS['secondary'], family='sans-serif', style='italic')
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # Summary statistics page - clean typography
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.97, 'EXECUTIVE SUMMARY', 
               ha='center', va='top', fontsize=18, fontweight='bold',
               color=SWISS_COLORS['text'], family='sans-serif',
               transform=ax.transAxes)
        
        # Decorative line
        line = plt.Line2D([0.1, 0.9], [0.94, 0.94], 
                         transform=ax.transAxes, 
                         color=SWISS_COLORS['accent'], linewidth=2)
        ax.add_line(line)
        
        # Format text with proper spacing
        y_start = 0.88
        line_height = 0.022
        current_y = y_start
        
        # Area characteristics section
        ax.text(0.1, current_y, 'AREA CHARACTERISTICS', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               color=SWISS_COLORS['text'], family='sans-serif',
               verticalalignment='top')
        current_y -= line_height * 2
        
        # Vidigal
        ax.text(0.1, current_y, 'Vidigal (Informal):', 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               color=SWISS_COLORS['vidigal'], family='sans-serif',
               verticalalignment='top')
        current_y -= line_height * 1.2
        ax.text(0.15, current_y, f'Buildings: {vidigal_data["building_count"]}', 
               transform=ax.transAxes, fontsize=9,
               color=SWISS_COLORS['text'], family='sans-serif',
               verticalalignment='top')
        current_y -= line_height
        ax.text(0.15, current_y, f'Study Area: {vidigal_data["study_area_m2"]/1e6:.2f} km²', 
               transform=ax.transAxes, fontsize=9,
               color=SWISS_COLORS['text'], family='sans-serif',
               verticalalignment='top')
        current_y -= line_height
        ax.text(0.15, current_y, f'Building Density: {vidigal_data["building_density"]:.1f} buildings/km²', 
               transform=ax.transAxes, fontsize=9,
               color=SWISS_COLORS['text'], family='sans-serif',
               verticalalignment='top')
        current_y -= line_height * 1.5
        
        # Copacabana
        ax.text(0.1, current_y, 'Copacabana (Formal):', 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               color=SWISS_COLORS['copacabana'], family='sans-serif',
               verticalalignment='top')
        current_y -= line_height * 1.2
        ax.text(0.15, current_y, f'Buildings: {copacabana_data["building_count"]}', 
               transform=ax.transAxes, fontsize=9,
               color=SWISS_COLORS['text'], family='sans-serif',
               verticalalignment='top')
        current_y -= line_height
        ax.text(0.15, current_y, f'Study Area: {copacabana_data["study_area_m2"]/1e6:.2f} km²', 
               transform=ax.transAxes, fontsize=9,
               color=SWISS_COLORS['text'], family='sans-serif',
               verticalalignment='top')
        current_y -= line_height
        ax.text(0.15, current_y, f'Building Density: {copacabana_data["building_density"]:.1f} buildings/km²', 
               transform=ax.transAxes, fontsize=9,
               color=SWISS_COLORS['text'], family='sans-serif',
               verticalalignment='top')
        current_y -= line_height * 2
        
        # Key findings section
        ax.text(0.1, current_y, 'KEY FINDINGS', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               color=SWISS_COLORS['text'], family='sans-serif',
               verticalalignment='top')
        current_y -= line_height * 2
        
        # Morphometric findings
        if len(morpho_comparison) > 0:
            ax.text(0.1, current_y, 'Morphometric Metrics:', 
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   color=SWISS_COLORS['text'], family='sans-serif',
                   verticalalignment='top')
            current_y -= line_height * 1.2
            
            key_metrics = ['height', 'area', 'volume', 'building_density_per_km2']
            for metric in key_metrics:
                row = morpho_comparison[morpho_comparison['metric'] == metric]
                if len(row) > 0:
                    vid_val = row['vidigal_mean'].iloc[0]
                    copa_val = row['copa_mean'].iloc[0]
                    ratio = row['ratio_mean'].iloc[0]
                    metric_name = metric.replace('_', ' ').title()
                    text = f'{metric_name}: Vidigal={vid_val:.2f}, Copacabana={copa_val:.2f} (Ratio: {ratio:.2f}x)'
                    ax.text(0.15, current_y, text, 
                           transform=ax.transAxes, fontsize=9,
                           color=SWISS_COLORS['text'], family='sans-serif',
                           verticalalignment='top')
                    current_y -= line_height
            current_y -= line_height * 0.5
        
        # Environmental findings
        for raster_type in ['svf', 'solar', 'porosity', 'deprivation_raster']:
            comp = raster_comparisons.get(raster_type)
            if comp:
                name = {'svf': 'Sky View Factor', 'solar': 'Solar Access', 
                       'porosity': 'Porosity', 'deprivation_raster': 'Deprivation Index'}[raster_type]
                ax.text(0.1, current_y, f'{name}:', 
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       color=SWISS_COLORS['text'], family='sans-serif',
                       verticalalignment='top')
                current_y -= line_height * 1.2
                ax.text(0.15, current_y, f'Mean: Vidigal={comp["vidigal_mean"]:.3f}, Copacabana={comp["copa_mean"]:.3f}', 
                       transform=ax.transAxes, fontsize=9,
                       color=SWISS_COLORS['text'], family='sans-serif',
                       verticalalignment='top')
                current_y -= line_height
                ax.text(0.15, current_y, f'Difference: {comp["difference_mean"]:.3f}', 
                       transform=ax.transAxes, fontsize=9,
                       color=SWISS_COLORS['text'], family='sans-serif',
                       verticalalignment='top')
                if comp.get('significant'):
                    current_y -= line_height
                    ax.text(0.15, current_y, f'Statistically significant (p={comp["p_value"]:.4f})', 
                           transform=ax.transAxes, fontsize=9,
                           color=SWISS_COLORS['accent'], family='sans-serif', style='italic',
                           verticalalignment='top')
                current_y -= line_height * 1.2
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # Detailed statistics tables - clean design
        if len(morpho_comparison) > 0:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            fig.patch.set_facecolor('white')
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.97, 'Morphometric Metrics Comparison', 
                   ha='center', va='top', fontsize=16, fontweight='bold',
                   color=SWISS_COLORS['text'], family='sans-serif',
                   transform=ax.transAxes)
            
            table_data = morpho_comparison[['metric', 'vidigal_mean', 'copa_mean', 
                                           'difference_mean', 'ratio_mean']].copy()
            table_data['metric'] = table_data['metric'].str.replace('_', ' ').str.title()
            table_data = table_data.round(2)
            table_data.columns = ['Metric', 'Vidigal\nMean', 'Copacabana\nMean', 'Difference', 'Ratio']
            
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                           cellLoc='center', loc='center', bbox=[0.05, 0.05, 0.9, 0.85])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.2)
            
            # Style the table
            for i in range(len(table_data.columns)):
                table[(0, i)].set_facecolor(SWISS_COLORS['light_gray'])
                table[(0, i)].set_text_props(weight='bold', color=SWISS_COLORS['text'])
                table[(0, i)].set_edgecolor(SWISS_COLORS['secondary'])
            
            for i in range(1, len(table_data) + 1):
                for j in range(len(table_data.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor(SWISS_COLORS['light_gray'])
                    table[(i, j)].set_edgecolor(SWISS_COLORS['secondary'])
                    table[(i, j)].set_text_props(color=SWISS_COLORS['text'])
            
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        # Add all visualizations
        for fig_name, fig in figs:
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        
        # Raster statistics table - clean design
        if raster_comparisons:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            fig.patch.set_facecolor('white')
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.97, 'Environmental Performance Metrics Comparison', 
                   ha='center', va='top', fontsize=16, fontweight='bold',
                   color=SWISS_COLORS['text'], family='sans-serif',
                   transform=ax.transAxes)
            
            raster_stats = []
            for raster_type, comp in raster_comparisons.items():
                if comp:
                    name = {'svf': 'Sky View Factor', 'solar': 'Solar Access (hours)',
                           'porosity': 'Porosity', 'deprivation_raster': 'Deprivation Index'}[raster_type]
                    raster_stats.append([
                        name,
                        f"{comp['vidigal_mean']:.3f}",
                        f"{comp['copa_mean']:.3f}",
                        f"{comp['difference_mean']:.3f}",
                        f"{comp['ratio_mean']:.2f}" if not np.isnan(comp['ratio_mean']) else "N/A",
                        f"{comp['p_value']:.4f}" if 'p_value' in comp and not np.isnan(comp['p_value']) else "N/A"
                    ])
            
            if raster_stats:
                cols = ['Metric', 'Vidigal\nMean', 'Copacabana\nMean', 'Difference', 'Ratio', 'P-value']
                table = ax.table(cellText=raster_stats, colLabels=cols,
                               cellLoc='center', loc='center', bbox=[0.05, 0.05, 0.9, 0.85])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2.5)
                
                # Style the table
                for i in range(len(cols)):
                    table[(0, i)].set_facecolor(SWISS_COLORS['light_gray'])
                    table[(0, i)].set_text_props(weight='bold', color=SWISS_COLORS['text'])
                    table[(0, i)].set_edgecolor(SWISS_COLORS['secondary'])
                
                for i in range(1, len(raster_stats) + 1):
                    for j in range(len(cols)):
                        if i % 2 == 0:
                            table[(i, j)].set_facecolor(SWISS_COLORS['light_gray'])
                        table[(i, j)].set_edgecolor(SWISS_COLORS['secondary'])
                        table[(i, j)].set_text_props(color=SWISS_COLORS['text'])
                
                pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            
            plt.close(fig)
    
    logger.info(f"PDF report saved to {pdf_path}")


def main():
    """Main comparison workflow."""
    logger.info("=" * 60)
    logger.info("COMPARATIVE ANALYSIS: Formal vs Informal Settlements")
    logger.info("=" * 60)
    
    # Setup output directory
    output_dir = get_comparative_analysis_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'tables').mkdir(exist_ok=True)
    (output_dir / 'metrics').mkdir(exist_ok=True)
    
    # Load data
    vidigal_data = load_area_data('vidigal')
    copacabana_data = load_area_data('copacabana')
    
    # Compare morphometric metrics
    morpho_comparison = compare_morphometric_metrics(vidigal_data, copacabana_data)
    morpho_comparison.to_csv(output_dir / 'tables' / 'comparison_morphometric_stats.csv', index=False)
    logger.info(f"Saved morphometric comparison to {output_dir / 'tables' / 'comparison_morphometric_stats.csv'}")
    
    # Compare raster metrics
    raster_comparisons = {}
    for raster_type in ['svf', 'solar', 'porosity', 'deprivation_raster']:
        comp = compare_raster_metrics(vidigal_data, copacabana_data, raster_type)
        if comp:
            raster_comparisons[raster_type] = comp
    
    # Save raster comparisons
    if raster_comparisons:
        raster_df = pd.DataFrame(list(raster_comparisons.values()))
        raster_df.to_csv(output_dir / 'tables' / 'comparison_environmental_stats.csv', index=False)
        logger.info(f"Saved environmental comparison to {output_dir / 'tables' / 'comparison_environmental_stats.csv'}")
    
    # Create visualizations
    figs = create_comparison_visualizations(vidigal_data, copacabana_data, 
                                           morpho_comparison, raster_comparisons,
                                           output_dir)
    
    # Save individual figure files
    for fig_name, fig in figs:
        fig_path = output_dir / 'visualizations' / f'{fig_name}.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved {fig_name} to {fig_path}")
    
    # Generate PDF report
    generate_pdf_report(vidigal_data, copacabana_data, morpho_comparison, 
                       raster_comparisons, figs, output_dir)
    
    logger.info("=" * 60)
    logger.info("COMPARATIVE ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

