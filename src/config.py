"""Configuration settings for the project."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Analysis parameters
EXPECTED_CRS = "EPSG:32723"  # UTM Zone 23S (São Paulo) - adjust as needed
MIN_BUILDING_AREA = 9.0  # m² (minimum reasonable building size)
MAX_BUILDING_HEIGHT = 100.0  # m (sanity check for validation)
MAX_FILTER_HEIGHT = 20.0  # m (maximum height for filtering buildings)
MAX_FILTER_AREA = 500.0  # m² (maximum area for filtering buildings)
MAX_FILTER_VOLUME = 3000.0  # m³ (maximum volume for filtering buildings)
MAX_FILTER_HW_RATIO = 100.0  # Maximum h/w ratio for filtering buildings

# Outlier filtering (based on height vs area relationship)
MAX_HEIGHT_AREA_RATIO = None  # Maximum height/area ratio (None to use percentile)
HEIGHT_AREA_PERCENTILE = 99.0  # Percentile for height/area ratio filtering

# Visualization settings
DPI = 300
FIGURE_SIZE = (12, 8)
COLORMAP_HEIGHT = "viridis"
COLORMAP_VOLUME = "plasma"


