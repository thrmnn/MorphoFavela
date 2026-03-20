"""Configuration settings for the project."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw"  # Legacy path (for backward compatibility)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Area-based data organization
# Areas: vidigal (informal), vidigal_tls (informal), copacabana (formal), riodaspedras (informal),
# rocinha (informal), maré (informal), cidade_de_deus (informal), complexo_do_alemao (informal)
SUPPORTED_AREAS = [
    "vidigal",
    "vidigal_tls",
    "copacabana",
    "riodaspedras",
    "rocinha",
    "maré",
    "cidade_de_deus",
    "complexo_do_alemao",
]

# Area classification: formal vs informal
FORMAL_AREAS = ["copacabana"]  # Formal settlements (no filtering)
INFORMAL_AREAS = [
    "vidigal",
    "vidigal_tls",
    "riodaspedras",
    "rocinha",
    "maré",
    "cidade_de_deus",
    "complexo_do_alemao",
]  # Informal settlements (with filtering)


def is_formal_area(area: str) -> bool:
    """Check if an area is classified as formal (no filtering applied)."""
    return area in FORMAL_AREAS


def is_informal_area(area: str) -> bool:
    """Check if an area is classified as informal (filtering applied)."""
    return area in INFORMAL_AREAS


def get_area_data_dir(area: str) -> Path:
    """Get the data directory for a specific area."""
    if area not in SUPPORTED_AREAS:
        raise ValueError(f"Unknown area: {area}. Supported areas: {SUPPORTED_AREAS}")
    return DATA_DIR / area / "raw"


def get_area_output_dir(area: str) -> Path:
    """Get the output directory for a specific area."""
    if area not in SUPPORTED_AREAS:
        raise ValueError(f"Unknown area: {area}. Supported areas: {SUPPORTED_AREAS}")
    return OUTPUTS_DIR / area


def get_area_analysis_dir(area: str, analysis_type: str) -> Path:
    """
    Get the output directory for a specific area and analysis type.

    Args:
        area: Area name (e.g., 'vidigal_tls', 'copacabana')
        analysis_type: Type of analysis (e.g., 'svf', 'solar', 'deprivation', 'metrics')

    Returns:
        Path to the analysis-specific output directory

    Examples:
        >>> get_area_analysis_dir('vidigal_tls', 'svf')
        Path('outputs/vidigal_tls/svf')
        >>> get_area_analysis_dir('copacabana', 'deprivation_raster')
        Path('outputs/copacabana/deprivation_raster')
    """
    if area not in SUPPORTED_AREAS:
        raise ValueError(f"Unknown area: {area}. Supported areas: {SUPPORTED_AREAS}")
    return OUTPUTS_DIR / area / analysis_type


def get_comparative_analysis_dir() -> Path:
    """Get the directory for comparative analysis outputs."""
    return OUTPUTS_DIR / "comparative"


# Standard analysis type subdirectories
ANALYSIS_TYPES = {
    "metrics": "metrics",  # Basic morphometric metrics
    "morphology_metrics": "morphology_metrics",  # Extended morphology metrics
    "morphology_risk": "morphology_risk",  # Risk visualization outputs
    "svf": "svf",  # Sky View Factor
    "solar": "solar",  # Solar access
    "sky_exposure": "sky_exposure",  # Sky exposure plane exceedance
    "porosity": "porosity",  # Sectional porosity
    "density": "density",  # Occupancy density proxy
    "deprivation": "deprivation",  # Deprivation index (unit-level)
    "deprivation_raster": "deprivation_raster",  # Deprivation index (raster-based)
    "maps": "maps",  # General maps/visualizations
    "urban_morphology": "urban_morphology",  # Zone-level urban morphology metrics
    "spatial_analysis": "spatial_analysis",  # Spatial autocorrelation (Moran's I, LISA)
    "typology": "typology",  # Settlement typology classification
}

# Analysis parameters
EXPECTED_CRS = "EPSG:32723"  # UTM Zone 23S (São Paulo) - adjust as needed
MIN_BUILDING_AREA = 9.0  # m² (minimum reasonable building size)
MAX_BUILDING_HEIGHT = 100.0  # m (sanity check for validation)
# Filtering parameters (applied only to informal areas, not formal areas)
# See is_formal_area() to determine if filtering should be skipped
MAX_FILTER_HEIGHT = 20.0  # m (maximum height for filtering buildings)
MAX_FILTER_AREA = 200.0  # m² (maximum area for filtering buildings)
MAX_FILTER_VOLUME = 3000.0  # m³ (maximum volume for filtering buildings)
MAX_FILTER_HW_RATIO = 100.0  # Maximum h/w ratio for filtering buildings

# Outlier filtering (based on height vs area relationship)
# Applied only to informal areas
MAX_HEIGHT_AREA_RATIO = None  # Maximum height/area ratio (None to use percentile)
HEIGHT_AREA_PERCENTILE = 99.0  # Percentile for height/area ratio filtering

# Building clustering parameters (for filtering isolated buildings)
BUILDING_CLUSTER_BUFFER = (
    15.0  # meters - buffer distance for connected components clustering
)

# Visualization settings
DPI = 300
FIGURE_SIZE = (12, 8)
COLORMAP_HEIGHT = "viridis"
COLORMAP_VOLUME = "plasma"

# Zone-level morphology visualization
LISA_COLORS = {
    "HH": "#d7191c",
    "HL": "#fdae61",
    "LH": "#abd9e9",
    "LL": "#2c7bb6",
    "ns": "#d3d3d3",
}
COLORMAP_BCR = "YlOrRd"
COLORMAP_FAR = "YlOrRd"
COLORMAP_SIGMA_H = "viridis"
COLORMAP_LAMBDA_F = "plasma"

# SVF & cartographic visualization
SVF_CMAP = "cividis"
BUILDING_CMAP = "Greys"
CONTOUR_COLOR = "#8B7355"
CONTOUR_INTERVAL = 10  # meters
BACKGROUND_COLOR = "#ffffff"
