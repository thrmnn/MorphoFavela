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
from matplotlib.image import imread
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
    
    # Load street-level data (SVF and solar along street centerlines)
    for street_type in ['svf_streets', 'solar_streets']:
        street_dir = get_area_analysis_dir(area, street_type)
        points_file = street_dir / f"street_{'svf' if 'svf' in street_type else 'solar'}_points.gpkg"
        segments_file = street_dir / f"street_{'svf' if 'svf' in street_type else 'solar'}_segments.gpkg"
        
        if points_file.exists():
            data[f'{street_type}_points'] = gpd.read_file(points_file)
            logger.info(f"  Loaded {street_type} points: {len(data[f'{street_type}_points'])}")
        else:
            data[f'{street_type}_points'] = None
        
        if segments_file.exists():
            data[f'{street_type}_segments'] = gpd.read_file(segments_file)
            logger.info(f"  Loaded {street_type} segments: {len(data[f'{street_type}_segments'])}")
        else:
            data[f'{street_type}_segments'] = None
    
    # Load street-level deprivation index (pre-computed)
    deprivation_dir = get_area_analysis_dir(area, 'deprivation_streets')
    deprivation_points_file = deprivation_dir / 'street_deprivation_points.gpkg'
    deprivation_segments_file = deprivation_dir / 'street_deprivation_segments.gpkg'
    
    if deprivation_points_file.exists():
        data['deprivation_streets_points'] = gpd.read_file(deprivation_points_file)
        logger.info(f"  Loaded deprivation points: {len(data['deprivation_streets_points'])}")
    else:
        logger.warning(f"  Deprivation points file not found: {deprivation_points_file}")
        logger.warning(f"  Run compute_deprivation_streets.py to generate this data")
        data['deprivation_streets_points'] = None
    
    if deprivation_segments_file.exists():
        data['deprivation_streets_segments'] = gpd.read_file(deprivation_segments_file)
        logger.info(f"  Loaded deprivation segments: {len(data['deprivation_streets_segments'])}")
    else:
        data['deprivation_streets_segments'] = None
    
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
    
    # Statistical test with effect size
    try:
        stat, p_value = stats.mannwhitneyu(vid_valid, copa_valid, alternative='two-sided')
        comparison['p_value'] = p_value
        comparison['significant'] = p_value < 0.05
        
        # Effect size (Cohen's d)
        effect_size = calculate_effect_size(vid_valid, copa_valid)
        comparison['effect_size'] = effect_size
        
        # Interpret effect size
        if abs(effect_size) < 0.2:
            comparison['effect_interpretation'] = 'negligible'
        elif abs(effect_size) < 0.5:
            comparison['effect_interpretation'] = 'small'
        elif abs(effect_size) < 0.8:
            comparison['effect_interpretation'] = 'medium'
        else:
            comparison['effect_interpretation'] = 'large'
    except Exception as e:
        comparison['p_value'] = np.nan
        comparison['significant'] = False
        comparison['effect_size'] = np.nan
        comparison['effect_interpretation'] = 'N/A'
    
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


def calculate_effect_size(vals1: np.ndarray, vals2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(vals1), len(vals2)
    mean1, mean2 = np.mean(vals1), np.mean(vals2)
    std1, std2 = np.std(vals1, ddof=1), np.std(vals2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean2 - mean1) / pooled_std


def compare_street_metrics(vidigal_data: dict, copacabana_data: dict, street_type: str) -> dict:
    """Compare street-level metrics between areas (SVF or solar along street centerlines)."""
    key_points = f'{street_type}_points'
    key_segments = f'{street_type}_segments'
    
    vid_points = vidigal_data.get(key_points)
    copa_points = copacabana_data.get(key_points)
    vid_segments = vidigal_data.get(key_segments)
    copa_segments = copacabana_data.get(key_segments)
    
    if vid_points is None or copa_points is None:
        return None
    
    # Determine the value column
    value_col = 'svf' if 'svf' in street_type else 'solar_hours'
    
    if value_col not in vid_points.columns or value_col not in copa_points.columns:
        return None
    
    vid_vals = vid_points[value_col].dropna().values
    copa_vals = copa_points[value_col].dropna().values
    
    if len(vid_vals) == 0 or len(copa_vals) == 0:
        return None
    
    # Calculate total street length for normalization
    vid_total_length = vid_segments['length'].sum() if vid_segments is not None and 'length' in vid_segments.columns else np.nan
    copa_total_length = copa_segments['length'].sum() if copa_segments is not None and 'length' in copa_segments.columns else np.nan
    
    comparison = {
        'metric': f'{street_type}_{value_col}',
        'vidigal_n_points': len(vid_vals),
        'vidigal_total_length_m': vid_total_length,
        'vidigal_points_per_km': (len(vid_vals) / vid_total_length * 1000) if not np.isnan(vid_total_length) and vid_total_length > 0 else np.nan,
        'vidigal_mean': vid_vals.mean(),
        'vidigal_median': np.median(vid_vals),
        'vidigal_std': vid_vals.std(),
        'vidigal_min': vid_vals.min(),
        'vidigal_max': vid_vals.max(),
        'vidigal_q10': np.percentile(vid_vals, 10),
        'vidigal_q90': np.percentile(vid_vals, 90),
        'copa_n_points': len(copa_vals),
        'copa_total_length_m': copa_total_length,
        'copa_points_per_km': (len(copa_vals) / copa_total_length * 1000) if not np.isnan(copa_total_length) and copa_total_length > 0 else np.nan,
        'copa_mean': copa_vals.mean(),
        'copa_median': np.median(copa_vals),
        'copa_std': copa_vals.std(),
        'copa_min': copa_vals.min(),
        'copa_max': copa_vals.max(),
        'copa_q10': np.percentile(copa_vals, 10),
        'copa_q90': np.percentile(copa_vals, 90),
        'difference_mean': copa_vals.mean() - vid_vals.mean(),
        'ratio_mean': copa_vals.mean() / vid_vals.mean() if vid_vals.mean() > 0 else np.nan,
    }
    
    # Statistical test with effect size
    try:
        stat, p_value = stats.mannwhitneyu(vid_vals, copa_vals, alternative='two-sided')
        comparison['p_value'] = p_value
        comparison['significant'] = p_value < 0.05
        
        # Effect size (Cohen's d)
        effect_size = calculate_effect_size(vid_vals, copa_vals)
        comparison['effect_size'] = effect_size
        
        # Interpret effect size
        if abs(effect_size) < 0.2:
            comparison['effect_interpretation'] = 'negligible'
        elif abs(effect_size) < 0.5:
            comparison['effect_interpretation'] = 'small'
        elif abs(effect_size) < 0.8:
            comparison['effect_interpretation'] = 'medium'
        else:
            comparison['effect_interpretation'] = 'large'
    except Exception as e:
        comparison['p_value'] = np.nan
        comparison['significant'] = False
        comparison['effect_size'] = np.nan
        comparison['effect_interpretation'] = 'N/A'
    
    # Threshold-based comparisons
    if 'svf' in street_type:
        comparison['vidigal_low_svf_fraction'] = (vid_vals < 0.5).sum() / len(vid_vals)
        comparison['copa_low_svf_fraction'] = (copa_vals < 0.5).sum() / len(copa_vals)
        comparison['vidigal_high_svf_fraction'] = (vid_vals > 0.7).sum() / len(vid_vals)
        comparison['copa_high_svf_fraction'] = (copa_vals > 0.7).sum() / len(copa_vals)
    elif 'solar' in street_type:
        comparison['vidigal_solar_deficit_fraction'] = (vid_vals < 2.0).sum() / len(vid_vals)
        comparison['copa_solar_deficit_fraction'] = (copa_vals < 2.0).sum() / len(copa_vals)
        comparison['vidigal_below_3h_fraction'] = (vid_vals < 3.0).sum() / len(vid_vals)
        comparison['copa_below_3h_fraction'] = (copa_vals < 3.0).sum() / len(copa_vals)
    
    return comparison


def compare_street_deprivation(vidigal_data: dict, copacabana_data: dict) -> dict:
    """Compare street-level deprivation index between areas."""
    vid_dep = vidigal_data.get('deprivation_streets_points')
    copa_dep = copacabana_data.get('deprivation_streets_points')
    
    if vid_dep is None or copa_dep is None:
        return None
    
    if 'deprivation_index' not in vid_dep.columns or 'deprivation_index' not in copa_dep.columns:
        return None
    
    vid_vals = vid_dep['deprivation_index'].dropna().values
    copa_vals = copa_dep['deprivation_index'].dropna().values
    
    if len(vid_vals) == 0 or len(copa_vals) == 0:
        return None
    
    comparison = {
        'metric': 'deprivation_streets',
        'vidigal_n_points': len(vid_vals),
        'vidigal_mean': vid_vals.mean(),
        'vidigal_median': np.median(vid_vals),
        'vidigal_std': vid_vals.std(),
        'vidigal_min': vid_vals.min(),
        'vidigal_max': vid_vals.max(),
        'vidigal_q10': np.percentile(vid_vals, 10),
        'vidigal_q90': np.percentile(vid_vals, 90),
        'copa_n_points': len(copa_vals),
        'copa_mean': copa_vals.mean(),
        'copa_median': np.median(copa_vals),
        'copa_std': copa_vals.std(),
        'copa_min': copa_vals.min(),
        'copa_max': copa_vals.max(),
        'copa_q10': np.percentile(copa_vals, 10),
        'copa_q90': np.percentile(copa_vals, 90),
        'difference_mean': copa_vals.mean() - vid_vals.mean(),
        'ratio_mean': copa_vals.mean() / vid_vals.mean() if vid_vals.mean() > 0 else np.nan,
    }
    
    # Statistical test with effect size
    try:
        stat, p_value = stats.mannwhitneyu(vid_vals, copa_vals, alternative='two-sided')
        comparison['p_value'] = p_value
        comparison['significant'] = p_value < 0.05
        
        effect_size = calculate_effect_size(vid_vals, copa_vals)
        comparison['effect_size'] = effect_size
        
        if abs(effect_size) < 0.2:
            comparison['effect_interpretation'] = 'negligible'
        elif abs(effect_size) < 0.5:
            comparison['effect_interpretation'] = 'small'
        elif abs(effect_size) < 0.8:
            comparison['effect_interpretation'] = 'medium'
        else:
            comparison['effect_interpretation'] = 'large'
    except Exception as e:
        comparison['p_value'] = np.nan
        comparison['significant'] = False
        comparison['effect_size'] = np.nan
        comparison['effect_interpretation'] = 'N/A'
    
    # Threshold-based comparisons
    comparison['vidigal_high_deprivation_fraction'] = (vid_vals >= 0.5).sum() / len(vid_vals)
    comparison['copa_high_deprivation_fraction'] = (copa_vals >= 0.5).sum() / len(copa_vals)
    comparison['vidigal_extreme_deprivation_fraction'] = (vid_vals >= 0.7).sum() / len(vid_vals)
    comparison['copa_extreme_deprivation_fraction'] = (copa_vals >= 0.7).sum() / len(copa_vals)
    
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
                # Use identical bins and x-axis range for fair comparison
                all_vals = pd.concat([vid_vals, copa_vals])
                min_val, max_val = all_vals.min(), all_vals.max()
                bins = np.linspace(min_val, max_val, 30)
                
                ax.hist(vid_vals, bins=bins, alpha=0.7, label='Vidigal (Informal)', 
                       color=SWISS_COLORS['vidigal'], density=True, edgecolor='white', linewidth=0.5)
                ax.hist(copa_vals, bins=bins, alpha=0.7, label='Copacabana (Formal)', 
                       color=SWISS_COLORS['copacabana'], density=True, edgecolor='white', linewidth=0.5)
                
                # Set identical x-axis range
                ax.set_xlim(min_val, max_val)
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
    import matplotlib
    clean_cmap = matplotlib.colormaps.get_cmap('viridis').copy()
    
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
            
            # Use identical color scale for fair comparison
            vid_valid = vid_raster[~np.isnan(vid_raster)]
            copa_valid = copa_raster[~np.isnan(copa_raster)]
            vmin = min(np.nanmin(vid_raster), np.nanmin(copa_raster))
            vmax = max(np.nanmax(vid_raster), np.nanmax(copa_raster))
            
            # Vidigal
            ax1 = fig.add_subplot(gs[0])
            ax1.set_facecolor('white')
            im1 = ax1.imshow(vid_raster, cmap=clean_cmap, origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
            ax1.set_title(f'Vidigal (Informal)\nMean: {np.nanmean(vid_raster):.3f}', 
                         fontweight='bold', fontsize=11, color=SWISS_COLORS['text'], pad=10)
            ax1.axis('off')
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label(title, fontsize=9, color=SWISS_COLORS['text'])
            cbar1.ax.tick_params(colors=SWISS_COLORS['secondary'], labelsize=8)
            
            # Copacabana
            ax2 = fig.add_subplot(gs[1])
            ax2.set_facecolor('white')
            im2 = ax2.imshow(copa_raster, cmap=clean_cmap, origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
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
                
                # Use identical bins for fair comparison
                all_vals = np.concatenate([vid_valid, copa_valid])
                min_val, max_val = np.nanmin(all_vals), np.nanmax(all_vals)
                bins = np.linspace(min_val, max_val, 50)
                
                ax.hist(vid_valid, bins=bins, alpha=0.7, label='Vidigal (Informal)', 
                       color=SWISS_COLORS['vidigal'], density=True, edgecolor='white', linewidth=0.5)
                ax.hist(copa_valid, bins=bins, alpha=0.7, label='Copacabana (Formal)', 
                       color=SWISS_COLORS['copacabana'], density=True, edgecolor='white', linewidth=0.5)
                ax.axvline(np.nanmean(vid_valid), color=SWISS_COLORS['vidigal'], 
                          linestyle='--', linewidth=2, alpha=0.8,
                          label=f'Vidigal mean: {np.nanmean(vid_valid):.3f}')
                ax.axvline(np.nanmean(copa_valid), color=SWISS_COLORS['copacabana'], 
                          linestyle='--', linewidth=2, alpha=0.8,
                          label=f'Copacabana mean: {np.nanmean(copa_valid):.3f}')
                
                # Set identical x-axis range
                ax.set_xlim(min_val, max_val)
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
    
    # Street-level side-by-side map comparisons
    # Load existing maps that already include building footprints
    
    # SVF streets - load existing maps
    vid_svf_map_path = get_area_analysis_dir('vidigal', 'svf_streets') / 'street_svf_map.png'
    copa_svf_map_path = get_area_analysis_dir('copacabana', 'svf_streets') / 'street_svf_map.png'
    
    if vid_svf_map_path.exists() and copa_svf_map_path.exists():
        vid_svf_img = imread(vid_svf_map_path)
        copa_svf_img = imread(copa_svf_map_path)
        
        # Create side-by-side map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('white')
        fig.suptitle('Street-Level Sky View Factor (SVF): Side-by-Side Comparison', 
                    fontsize=14, fontweight='bold', color=SWISS_COLORS['text'], y=0.98)
        
        # Vidigal
        ax1.imshow(vid_svf_img)
        ax1.set_title('Vidigal (Informal)', 
                     fontweight='bold', fontsize=11, color=SWISS_COLORS['text'], pad=10)
        ax1.axis('off')
        
        # Copacabana
        ax2.imshow(copa_svf_img)
        ax2.set_title('Copacabana (Formal)', 
                     fontweight='bold', fontsize=11, color=SWISS_COLORS['text'], pad=10)
        ax2.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        figs.append(('svf_streets_side_by_side', fig))
    
    # Solar streets - load existing maps
    vid_solar_map_path = get_area_analysis_dir('vidigal', 'solar_streets') / 'street_solar_map.png'
    copa_solar_map_path = get_area_analysis_dir('copacabana', 'solar_streets') / 'street_solar_map.png'
    
    if vid_solar_map_path.exists() and copa_solar_map_path.exists():
        vid_solar_img = imread(vid_solar_map_path)
        copa_solar_img = imread(copa_solar_map_path)
        
        # Create side-by-side map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('white')
        fig.suptitle('Street-Level Solar Access: Side-by-Side Comparison', 
                    fontsize=14, fontweight='bold', color=SWISS_COLORS['text'], y=0.98)
        
        # Vidigal
        ax1.imshow(vid_solar_img)
        ax1.set_title('Vidigal (Informal)', 
                     fontweight='bold', fontsize=11, color=SWISS_COLORS['text'], pad=10)
        ax1.axis('off')
        
        # Copacabana
        ax2.imshow(copa_solar_img)
        ax2.set_title('Copacabana (Formal)', 
                     fontweight='bold', fontsize=11, color=SWISS_COLORS['text'], pad=10)
        ax2.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        figs.append(('solar_streets_side_by_side', fig))
    
    # Deprivation streets - use pre-computed segments if available
    vid_dep_segments = vidigal_data.get('deprivation_streets_segments')
    copa_dep_segments = copacabana_data.get('deprivation_streets_segments')
    vid_buildings = vidigal_data.get('buildings')
    copa_buildings = copacabana_data.get('buildings')
    
    # If segments not available, try to create from points
    if vid_dep_segments is None or copa_dep_segments is None:
        vid_dep_points = vidigal_data.get('deprivation_streets_points')
        copa_dep_points = copacabana_data.get('deprivation_streets_points')
        
        if vid_dep_points is not None and copa_dep_points is not None and 'deprivation_index' in vid_dep_points.columns and 'deprivation_index' in copa_dep_points.columns:
            # Aggregate points to segments for visualization
            if 'segment_idx' in vid_dep_points.columns and 'segment_idx' in copa_dep_points.columns:
                vid_dep_segments = vid_dep_points.groupby('segment_idx')['deprivation_index'].agg(['mean', 'std', 'count']).reset_index()
                vid_dep_segments.rename(columns={'mean': 'deprivation_mean'}, inplace=True)
                vid_svf_seg = vidigal_data.get('svf_streets_segments')
                if vid_svf_seg is not None and 'segment_idx' in vid_svf_seg.columns:
                    vid_dep_segments = vid_svf_seg[['segment_idx', 'geometry']].merge(vid_dep_segments, on='segment_idx', how='inner')
                    vid_dep_segments = gpd.GeoDataFrame(vid_dep_segments, crs=vid_svf_seg.crs)
                
                copa_dep_segments = copa_dep_points.groupby('segment_idx')['deprivation_index'].agg(['mean', 'std', 'count']).reset_index()
                copa_dep_segments.rename(columns={'mean': 'deprivation_mean'}, inplace=True)
                copa_svf_seg = copacabana_data.get('svf_streets_segments')
                if copa_svf_seg is not None and 'segment_idx' in copa_svf_seg.columns:
                    copa_dep_segments = copa_svf_seg[['segment_idx', 'geometry']].merge(copa_dep_segments, on='segment_idx', how='inner')
                    copa_dep_segments = gpd.GeoDataFrame(copa_dep_segments, crs=copa_svf_seg.crs)
    
    if vid_dep_segments is not None and copa_dep_segments is not None and 'deprivation_mean' in vid_dep_segments.columns and 'deprivation_mean' in copa_dep_segments.columns:
        # Calculate consistent color scale
        vid_vals = vid_dep_segments['deprivation_mean'].dropna().values
        copa_vals = copa_dep_segments['deprivation_mean'].dropna().values
        if len(vid_vals) > 0 and len(copa_vals) > 0:
            vmin = min(np.min(vid_vals), np.min(copa_vals))
            vmax = max(np.max(vid_vals), np.max(copa_vals))
            
            # Create side-by-side map
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.patch.set_facecolor('white')
            fig.suptitle('Street-Level Deprivation Index: Side-by-Side Comparison', 
                        fontsize=14, fontweight='bold', color=SWISS_COLORS['text'], y=0.98)
            
            # Use reversed colormap (higher = more deprived = darker)
            from matplotlib.colors import ListedColormap
            import matplotlib
            dep_cmap = matplotlib.colormaps.get_cmap('RdYlGn_r').copy()
            
            # Vidigal
            ax1.set_facecolor('white')
            # Plot building footprints as background
            if vid_buildings is not None:
                vid_buildings_plot = vid_buildings.to_crs(vid_dep_segments.crs) if vid_buildings.crs != vid_dep_segments.crs else vid_buildings
                vid_buildings_plot.plot(ax=ax1, facecolor='lightgrey', edgecolor='black', 
                                       linewidth=0.3, alpha=0.4, zorder=0)
            vid_dep_segments.plot(ax=ax1, column='deprivation_mean', cmap=dep_cmap, 
                                 vmin=vmin, vmax=vmax, linewidth=2, legend=True,
                                 legend_kwds={'label': 'Deprivation Index', 'shrink': 0.8}, zorder=1)
            ax1.set_title(f'Vidigal (Informal)\nMean: {np.mean(vid_vals):.3f}', 
                         fontweight='bold', fontsize=11, color=SWISS_COLORS['text'], pad=10)
            ax1.set_aspect('equal')
            ax1.axis('off')
            
            # Copacabana
            ax2.set_facecolor('white')
            # Plot building footprints as background
            if copa_buildings is not None:
                copa_buildings_plot = copa_buildings.to_crs(copa_dep_segments.crs) if copa_buildings.crs != copa_dep_segments.crs else copa_buildings
                copa_buildings_plot.plot(ax=ax2, facecolor='lightgrey', edgecolor='black', 
                                        linewidth=0.3, alpha=0.4, zorder=0)
            copa_dep_segments.plot(ax=ax2, column='deprivation_mean', cmap=dep_cmap, 
                                  vmin=vmin, vmax=vmax, linewidth=2, legend=True,
                                  legend_kwds={'label': 'Deprivation Index', 'shrink': 0.8}, zorder=1)
            ax2.set_title(f'Copacabana (Formal)\nMean: {np.mean(copa_vals):.3f}', 
                         fontweight='bold', fontsize=11, color=SWISS_COLORS['text'], pad=10)
            ax2.set_aspect('equal')
            ax2.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            figs.append(('deprivation_streets_side_by_side', fig))
    
    # Street-level metric visualizations
    for street_type in ['svf_streets', 'solar_streets']:
        vid_points = vidigal_data.get(f'{street_type}_points')
        copa_points = copacabana_data.get(f'{street_type}_points')
        
        if vid_points is not None and copa_points is not None:
            value_col = 'svf' if 'svf' in street_type else 'solar_hours'
            title = 'SVF' if 'svf' in street_type else 'Solar Hours'
            
            if value_col in vid_points.columns and value_col in copa_points.columns:
                vid_vals = vid_points[value_col].dropna().values
                copa_vals = copa_points[value_col].dropna().values
                
                if len(vid_vals) > 0 and len(copa_vals) > 0:
                    # Distribution comparison with identical bins and scales
                    # Use landscape-friendly size (will be resized to PAGE_SIZE for PDF)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('white')
                    fig.suptitle(f'Street-Level {title}: Distribution Comparison', 
                                fontsize=14, fontweight='bold', color=SWISS_COLORS['text'], y=0.95)
                    ax.set_facecolor('white')
                    
                    # Use identical bins for fair comparison
                    all_vals = np.concatenate([vid_vals, copa_vals])
                    min_val, max_val = np.min(all_vals), np.max(all_vals)
                    bins = np.linspace(min_val, max_val, 30)
                    
                    ax.hist(vid_vals, bins=bins, alpha=0.7, label='Vidigal (Informal)', 
                           color=SWISS_COLORS['vidigal'], density=True, edgecolor='white', linewidth=0.5)
                    ax.hist(copa_vals, bins=bins, alpha=0.7, label='Copacabana (Formal)', 
                           color=SWISS_COLORS['copacabana'], density=True, edgecolor='white', linewidth=0.5)
                    ax.axvline(np.mean(vid_vals), color=SWISS_COLORS['vidigal'], 
                              linestyle='--', linewidth=2, alpha=0.8,
                              label=f'Vidigal mean: {np.mean(vid_vals):.2f}')
                    ax.axvline(np.mean(copa_vals), color=SWISS_COLORS['copacabana'], 
                              linestyle='--', linewidth=2, alpha=0.8,
                              label=f'Copacabana mean: {np.mean(copa_vals):.2f}')
                    
                    # Set identical x-axis range
                    ax.set_xlim(min_val, max_val)
                    ax.set_xlabel(f'{title} along streets', fontsize=10, color=SWISS_COLORS['text'])
                    ax.set_ylabel('Density', fontsize=10, color=SWISS_COLORS['text'])
                    ax.legend(fontsize=9, frameon=True, fancybox=False, 
                             edgecolor=SWISS_COLORS['secondary'], framealpha=0.9)
                    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(colors=SWISS_COLORS['secondary'], labelsize=8)
                    
                    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
                    figs.append((f'{street_type}_distribution', fig))
                    
                    # Box plot comparison with identical y-axis range
                    # Use landscape-friendly size (will be resized to PAGE_SIZE for PDF)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('white')
                    fig.suptitle(f'Street-Level {title}: Box Plot Comparison', 
                                fontsize=14, fontweight='bold', color=SWISS_COLORS['text'], y=0.95)
                    ax.set_facecolor('white')
                    
                    bp = ax.boxplot([vid_vals, copa_vals], tick_labels=['Vidigal\n(Informal)', 'Copacabana\n(Formal)'],
                                   patch_artist=True)
                    bp['boxes'][0].set_facecolor(SWISS_COLORS['vidigal'])
                    bp['boxes'][0].set_alpha(0.6)
                    bp['boxes'][1].set_facecolor(SWISS_COLORS['copacabana'])
                    bp['boxes'][1].set_alpha(0.6)
                    
                    # Set identical y-axis range
                    ax.set_ylim(min_val, max_val)
                    ax.set_ylabel(f'{title}', fontsize=10, color=SWISS_COLORS['text'])
                    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(colors=SWISS_COLORS['secondary'], labelsize=9)
                    
                    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
                    figs.append((f'{street_type}_boxplot', fig))
    
    # Street-level deprivation visualization
    vid_dep = vidigal_data.get('deprivation_streets_points')
    copa_dep = copacabana_data.get('deprivation_streets_points')
    
    if vid_dep is not None and copa_dep is not None and 'deprivation_index' in vid_dep.columns and 'deprivation_index' in copa_dep.columns:
        vid_vals = vid_dep['deprivation_index'].dropna().values
        copa_vals = copa_dep['deprivation_index'].dropna().values
        
        if len(vid_vals) > 0 and len(copa_vals) > 0:
            # Distribution comparison
            # Use landscape-friendly size (will be resized to PAGE_SIZE for PDF)
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('white')
            fig.suptitle('Street-Level Deprivation Index: Distribution Comparison', 
                        fontsize=14, fontweight='bold', color=SWISS_COLORS['text'], y=0.95)
            ax.set_facecolor('white')
            
            # Use identical bins for fair comparison
            all_vals = np.concatenate([vid_vals, copa_vals])
            min_val, max_val = np.min(all_vals), np.max(all_vals)
            bins = np.linspace(min_val, max_val, 30)
            
            ax.hist(vid_vals, bins=bins, alpha=0.7, label='Vidigal (Informal)', 
                   color=SWISS_COLORS['vidigal'], density=True, edgecolor='white', linewidth=0.5)
            ax.hist(copa_vals, bins=bins, alpha=0.7, label='Copacabana (Formal)', 
                   color=SWISS_COLORS['copacabana'], density=True, edgecolor='white', linewidth=0.5)
            ax.axvline(np.mean(vid_vals), color=SWISS_COLORS['vidigal'], 
                      linestyle='--', linewidth=2, alpha=0.8,
                      label=f'Vidigal mean: {np.mean(vid_vals):.3f}')
            ax.axvline(np.mean(copa_vals), color=SWISS_COLORS['copacabana'], 
                      linestyle='--', linewidth=2, alpha=0.8,
                      label=f'Copacabana mean: {np.mean(copa_vals):.3f}')
            
            # Set identical x-axis range
            ax.set_xlim(min_val, max_val)
            ax.set_xlabel('Deprivation Index (0-1, higher = more deprived)', fontsize=10, color=SWISS_COLORS['text'])
            ax.set_ylabel('Density', fontsize=10, color=SWISS_COLORS['text'])
            ax.legend(fontsize=9, frameon=True, fancybox=False, 
                     edgecolor=SWISS_COLORS['secondary'], framealpha=0.9)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(colors=SWISS_COLORS['secondary'], labelsize=8)
            
            plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
            figs.append(('deprivation_streets_distribution', fig))
    
    return figs


def generate_pdf_report(vidigal_data: dict, copacabana_data: dict,
                       morpho_comparison: pd.DataFrame,
                       raster_comparisons: dict,
                       street_comparisons: dict,
                       figs: list,
                       output_dir: Path):
    """Generate comprehensive PDF report with Swiss design."""
    logger.info("Generating PDF report...")
    
    pdf_path = output_dir / 'comparison_report.pdf'
    
    # Portrait page size (consistent for all pages)
    PAGE_SIZE = (8.5, 11)  # Portrait: width x height
    
    with PdfPages(pdf_path) as pdf:
        # Title page - clean Swiss design
        fig = plt.figure(figsize=PAGE_SIZE)
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
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='white', dpi=150)
        plt.close(fig)
        
        # Summary statistics page - clean typography
        fig, ax = plt.subplots(figsize=PAGE_SIZE)
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
        
        # Street-level findings - PRIMARY FOCUS
        if street_comparisons:
            ax.text(0.1, current_y, 'STREET-LEVEL PERFORMANCE (Primary Focus):', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   color=SWISS_COLORS['accent'], family='sans-serif',
                   verticalalignment='top')
            current_y -= line_height * 2
            
            for street_type, comp in street_comparisons.items():
                if comp:
                    if 'svf' in street_type:
                        name = 'Sky View Factor (SVF)'
                        unit = ''
                    elif 'solar' in street_type:
                        name = 'Solar Access'
                        unit = ' hours'
                    elif 'deprivation' in street_type:
                        name = 'Deprivation Index'
                        unit = ''
                    else:
                        name = street_type.replace('_', ' ').title()
                        unit = ''
                    
                    effect_str = comp.get('effect_interpretation', 'N/A')
                    sig_str = ' (Significant)' if comp.get('significant', False) else ''
                    ax.text(0.15, current_y, f'{name}:', 
                           transform=ax.transAxes, fontsize=10, fontweight='bold',
                           color=SWISS_COLORS['text'], family='sans-serif',
                           verticalalignment='top')
                    current_y -= line_height * 1.2
                    ax.text(0.2, current_y, f'Vidigal={comp["vidigal_mean"]:.3f}{unit}, Copacabana={comp["copa_mean"]:.3f}{unit}', 
                           transform=ax.transAxes, fontsize=9,
                           color=SWISS_COLORS['text'], family='sans-serif',
                           verticalalignment='top')
                    current_y -= line_height
                    ax.text(0.2, current_y, f'Effect: {effect_str} (d={comp.get("effect_size", 0):.2f}){sig_str}', 
                           transform=ax.transAxes, fontsize=9,
                           color=SWISS_COLORS['accent'] if comp.get('significant', False) else SWISS_COLORS['secondary'],
                           family='sans-serif', style='italic',
                           verticalalignment='top')
                    current_y -= line_height * 1.5
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='white', dpi=150)
        plt.close(fig)
        
        # Add key visualizations - focus on street-level, ensure portrait format
        key_figs = ['svf_streets_side_by_side', 'solar_streets_side_by_side', 
                   'svf_streets_distribution', 'solar_streets_distribution', 
                   'deprivation_streets_distribution', 'svf_streets_boxplot', 'solar_streets_boxplot']
        for fig_name, fig in figs:
            if fig_name in key_figs:
                # Always resize to portrait format for consistency
                if 'side_by_side' in fig_name:
                    # Side-by-side maps: resize to portrait, maintaining aspect ratio
                    # Original is 16x8 (landscape), need to fit in 8.5x11 (portrait)
                    # Calculate new size maintaining aspect ratio but fitting in portrait
                    current_size = fig.get_size_inches()
                    aspect_ratio = current_size[0] / current_size[1]  # width/height
                    # Fit width to page width (8.5), adjust height accordingly
                    new_width = 8.5
                    new_height = new_width / aspect_ratio
                    # If height exceeds page height (11), scale down
                    if new_height > 11:
                        new_height = 11
                        new_width = new_height * aspect_ratio
                    fig.set_size_inches(new_width, new_height)
                    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], pad=2.0)
                else:
                    fig.set_size_inches(PAGE_SIZE)
                    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], pad=2.0)
                pdf.savefig(fig, bbox_inches='tight', facecolor='white', dpi=150)
                plt.close(fig)
        
        
        # Street-level statistics table - primary focus
        if street_comparisons:
            fig, ax = plt.subplots(figsize=PAGE_SIZE)
            fig.patch.set_facecolor('white')
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.97, 'Street-Level Performance Comparison', 
                   ha='center', va='top', fontsize=18, fontweight='bold',
                   color=SWISS_COLORS['text'], family='sans-serif',
                   transform=ax.transAxes)
            
            street_stats = []
            for street_type, comp in street_comparisons.items():
                if comp:
                    if 'svf' in street_type:
                        name = 'Sky View Factor (SVF)'
                    elif 'solar' in street_type:
                        name = 'Solar Access (hours)'
                    elif 'deprivation' in street_type:
                        name = 'Deprivation Index'
                    else:
                        name = street_type.replace('_', ' ').title()
                    
                    effect_str = comp.get('effect_interpretation', 'N/A')
                    sig_str = '✓' if comp.get('significant', False) else ''
                    street_stats.append([
                        name,
                        f"{comp['vidigal_mean']:.3f}",
                        f"{comp['copa_mean']:.3f}",
                        f"{comp['difference_mean']:.3f}",
                        f"{comp.get('effect_size', np.nan):.2f}" if 'effect_size' in comp and not np.isnan(comp.get('effect_size', np.nan)) else "N/A",
                        effect_str,
                        sig_str
                    ])
            
            if street_stats:
                cols = ['Metric', 'Vidigal\nMean', 'Copacabana\nMean', 'Difference', 'Effect\nSize', 'Effect', 'Sig.']
                table = ax.table(cellText=street_stats, colLabels=cols,
                               cellLoc='center', loc='center', bbox=[0.15, 0.3, 0.7, 0.5])
                table.auto_set_font_size(False)
                table.set_fontsize(11)
                table.scale(1, 2.5)
                
                # Style the table
                for i in range(len(cols)):
                    table[(0, i)].set_facecolor(SWISS_COLORS['light_gray'])
                    table[(0, i)].set_text_props(weight='bold', color=SWISS_COLORS['text'])
                    table[(0, i)].set_edgecolor(SWISS_COLORS['secondary'])
                
                for i in range(1, len(street_stats) + 1):
                    for j in range(len(cols)):
                        if i % 2 == 0:
                            table[(i, j)].set_facecolor(SWISS_COLORS['light_gray'])
                        table[(i, j)].set_edgecolor(SWISS_COLORS['secondary'])
                        table[(i, j)].set_text_props(color=SWISS_COLORS['text'])
            
            pdf.savefig(fig, bbox_inches='tight', facecolor='white', dpi=150)
            plt.close(fig)
        
        # Raster statistics table - secondary (simplified)
        if raster_comparisons:
            fig, ax = plt.subplots(figsize=PAGE_SIZE)
            fig.patch.set_facecolor('white')
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.97, 'Environmental Performance Metrics Comparison', 
                   ha='center', va='top', fontsize=16, fontweight='bold',
                   color=SWISS_COLORS['text'], family='sans-serif',
                   transform=ax.transAxes)
            
            raster_stats = []
            # Only include key raster metrics (SVF and solar) for reference
            for raster_type in ['svf', 'solar']:
                comp = raster_comparisons.get(raster_type)
                if comp:
                    name = {'svf': 'Sky View Factor (Grid)', 'solar': 'Solar Access (Grid, hours)'}[raster_type]
                    effect_str = comp.get('effect_interpretation', 'N/A')
                    sig_str = '✓' if comp.get('significant', False) else ''
                    raster_stats.append([
                        name,
                        f"{comp['vidigal_mean']:.3f}",
                        f"{comp['copa_mean']:.3f}",
                        f"{comp['difference_mean']:.3f}",
                        f"{comp.get('effect_size', np.nan):.2f}" if 'effect_size' in comp and not np.isnan(comp.get('effect_size', np.nan)) else "N/A",
                        effect_str,
                        sig_str
                    ])
            
            if raster_stats:
                cols = ['Metric', 'Vidigal\nMean', 'Copacabana\nMean', 'Difference', 'Effect\nSize', 'Effect', 'Sig.']
                table = ax.table(cellText=raster_stats, colLabels=cols,
                               cellLoc='center', loc='center', bbox=[0.2, 0.3, 0.6, 0.5])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
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
                
                pdf.savefig(fig, bbox_inches='tight', facecolor='white', dpi=150)
            
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
    
    # Compare street-level metrics
    street_comparisons = {}
    for street_type in ['svf_streets', 'solar_streets']:
        comp = compare_street_metrics(vidigal_data, copacabana_data, street_type)
        if comp:
            street_comparisons[street_type] = comp
    
    # Compare street-level deprivation
    street_deprivation_comp = compare_street_deprivation(vidigal_data, copacabana_data)
    if street_deprivation_comp:
        street_comparisons['deprivation_streets'] = street_deprivation_comp
    
    # Save raster comparisons
    if raster_comparisons:
        raster_df = pd.DataFrame(list(raster_comparisons.values()))
        raster_df.to_csv(output_dir / 'tables' / 'comparison_environmental_stats.csv', index=False)
        logger.info(f"Saved environmental comparison to {output_dir / 'tables' / 'comparison_environmental_stats.csv'}")
    
    # Save street-level comparisons
    if street_comparisons:
        street_df = pd.DataFrame(list(street_comparisons.values()))
        street_df.to_csv(output_dir / 'tables' / 'comparison_street_level_stats.csv', index=False)
        logger.info(f"Saved street-level comparison to {output_dir / 'tables' / 'comparison_street_level_stats.csv'}")
    
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
                       raster_comparisons, street_comparisons, figs, output_dir)
    
    logger.info("=" * 60)
    logger.info("COMPARATIVE ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

