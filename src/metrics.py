"""Basic morphometric metrics calculation."""

import numpy as np
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)


def normalize_height_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normalize height column names to standard format.
    
    Handles alternative column names:
    - 'base' -> 'base_height'
    - 'altura' -> relative height, calculates 'top_height' = base + altura
    
    Args:
        gdf: GeoDataFrame with height columns (standard or alternative names)
        
    Returns:
        GeoDataFrame with normalized 'base_height' and 'top_height' columns
    """
    gdf = gdf.copy()
    
    # Check for alternative column names
    if 'base' in gdf.columns and 'altura' in gdf.columns:
        # Map base -> base_height
        gdf['base_height'] = gdf['base']
        # Calculate top_height from base + altura (relative height)
        gdf['top_height'] = gdf['base'] + gdf['altura']
        logger.info("Normalized columns: 'base' -> 'base_height', 'altura' -> calculated 'top_height'")
    elif 'base_height' not in gdf.columns or 'top_height' not in gdf.columns:
        raise ValueError(
            "Expected columns 'base_height' and 'top_height', or alternative "
            "columns 'base' and 'altura'. Found columns: " + str(list(gdf.columns))
        )
    
    return gdf


def calculate_basic_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate 6 basic morphometric metrics for building footprints.
    
    Args:
        gdf: GeoDataFrame with 'base_height' and 'top_height' columns
            (or 'base' and 'altura' which will be normalized)
        
    Returns:
        GeoDataFrame with added metric columns:
            - height: Building height (m)
            - area: Footprint area (m²)
            - volume: Building volume (m³)
            - perimeter: Footprint perimeter (m)
            - hw_ratio: Street canyon ratio (height/width)
            - inter_building_distance: Distance to nearest neighbor building (m)
    """
    # Normalize column names if needed
    gdf = normalize_height_columns(gdf)
    
    # Validate required columns
    required_cols = ['base_height', 'top_height']
    missing = [col for col in required_cols if col not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Calculate height
    gdf = gdf.copy()  # Avoid modifying original
    gdf['height'] = gdf['top_height'] - gdf['base_height']
    
    # Check for invalid heights
    invalid_mask = gdf['height'] <= 0
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        logger.warning(f"Found {n_invalid} buildings with height ≤ 0")
        # Filter out invalid buildings
        gdf = gdf[~invalid_mask].copy()
        logger.info(f"Processing {len(gdf)} valid buildings")
    
    # Calculate geometric properties
    gdf['area'] = gdf.geometry.area
    gdf['perimeter'] = gdf.geometry.length
    gdf['volume'] = gdf['area'] * gdf['height']
    
    # Calculate building width (minimum dimension of bounding box)
    # This approximates the building width for street canyon ratio
    bounds = gdf.geometry.bounds
    width = bounds['maxx'] - bounds['minx']
    length = bounds['maxy'] - bounds['miny']
    gdf['width'] = np.minimum(width, length)  # Minimum dimension as width
    
    # Street canyon ratio: height to width (h/w)
    # Avoid division by zero for zero-width buildings
    gdf['hw_ratio'] = np.where(
        gdf['width'] > 0,
        gdf['height'] / gdf['width'],
        np.nan
    )
    
    # Calculate inter-building distance (distance to nearest neighbor)
    logger.info("Calculating inter-building distances...")
    gdf['inter_building_distance'] = calculate_inter_building_distance(gdf)
    
    return gdf


def calculate_inter_building_distance(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Calculate the distance from each building to its nearest neighbor.
    
    Uses spatial indexing for efficient computation. The distance is measured
    as the minimum distance between building boundaries (not centroids).
    
    Args:
        gdf: GeoDataFrame with building geometries
        
    Returns:
        Array of distances in meters (same length as gdf)
    """
    from shapely.strtree import STRtree
    
    n_buildings = len(gdf)
    distances = np.full(n_buildings, np.nan)
    
    if n_buildings < 2:
        logger.warning("Less than 2 buildings, cannot calculate inter-building distances")
        return distances
    
    # Build spatial index for efficient nearest neighbor queries
    tree = STRtree(gdf.geometry.values)
    
    # For each building, find nearest neighbor and calculate distance
    for idx in range(n_buildings):
        building_geom = gdf.geometry.iloc[idx]
        
        # Use spatial index to find all buildings that might be neighbors
        # Query with a reasonable buffer to find candidates
        # Start with a small buffer and expand if needed
        buffer_distances = [50.0, 200.0, 500.0, 1000.0]
        candidates = []
        
        for buffer_dist in buffer_distances:
            candidates = tree.query(building_geom.buffer(buffer_dist))
            if len(candidates) > 1:  # More than just self
                break
        
        # If still no candidates, query all buildings
        if len(candidates) <= 1:
            candidates = list(range(n_buildings))
        
        # Calculate distances to all candidates and find minimum
        min_distance = np.inf
        for candidate_idx in candidates:
            if candidate_idx == idx:
                continue  # Skip self
            
            candidate_geom = gdf.geometry.iloc[candidate_idx]
            dist = building_geom.distance(candidate_geom)
            
            if dist < min_distance:
                min_distance = dist
        
        # If still no valid neighbor found, check all buildings (fallback)
        if min_distance == np.inf:
            for other_idx in range(n_buildings):
                if other_idx == idx:
                    continue
                other_geom = gdf.geometry.iloc[other_idx]
                dist = building_geom.distance(other_geom)
                if dist < min_distance:
                    min_distance = dist
        
        distances[idx] = min_distance if min_distance != np.inf else np.nan
    
    # Log statistics
    valid_distances = distances[~np.isnan(distances)]
    if len(valid_distances) > 0:
        logger.info(f"  Inter-building distance: mean={valid_distances.mean():.2f}m, "
                   f"min={valid_distances.min():.2f}m, max={valid_distances.max():.2f}m")
    
    return distances


def validate_footprints(gdf: gpd.GeoDataFrame) -> tuple[bool, list[str]]:
    """
    Validate building footprints dataset.
    
    Args:
        gdf: GeoDataFrame containing building footprints
            (accepts 'base_height'/'top_height' or 'base'/'altura')
        
    Returns:
        tuple: (is_valid: bool, issues: list of issue descriptions)
    """
    issues = []
    
    # Check for required columns (standard or alternative names)
    has_standard = 'base_height' in gdf.columns and 'top_height' in gdf.columns
    has_alternative = 'base' in gdf.columns and 'altura' in gdf.columns
    
    if not (has_standard or has_alternative):
        issues.append(
            "Missing required columns. Expected either: "
            "('base_height', 'top_height') or ('base', 'altura')"
        )
    
    # Check CRS
    if gdf.crs is None:
        issues.append("No CRS defined")
    elif not gdf.crs.is_projected:
        issues.append(f"CRS is not projected: {gdf.crs}")
    
    # Check geometry validity
    invalid_geom = ~gdf.geometry.is_valid
    if invalid_geom.any():
        n_invalid = invalid_geom.sum()
        issues.append(f"{n_invalid} invalid geometries")
    
    # Check for empty geometries
    empty_geom = gdf.geometry.is_empty
    if empty_geom.any():
        n_empty = empty_geom.sum()
        issues.append(f"{n_empty} empty geometries")
    
    # Check height values (normalize columns first if needed)
    if has_standard or has_alternative:
        # Create temporary normalized copy for validation
        try:
            gdf_norm = normalize_height_columns(gdf)
            height = gdf_norm['top_height'] - gdf_norm['base_height']
            
            if (height <= 0).any():
                n_invalid = (height <= 0).sum()
                issues.append(f"{n_invalid} buildings with height ≤ 0")
        except Exception as e:
            issues.append(f"Error validating heights: {e}")
    
    # Check for null values
    if has_standard:
        null_counts = gdf[['base_height', 'top_height']].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts.to_dict()}")
    elif has_alternative:
        null_counts = gdf[['base', 'altura']].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts.to_dict()}")
    
    is_valid = len(issues) == 0
    return is_valid, issues

