"""Backward compatibility shim -- all functionality moved to src.solar.

This module re-exports the public API from the new ``src.solar`` package
so that existing code continues to work without changes.
"""

import numpy as _np
import pyvista as _pv

from src.solar.sun import (
    compute_sun_positions,
    sun_position_to_direction,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
)
from src.solar.compute import (
    _build_obb_tree,  # noqa: F401 (re-export for tests)
    _ray_hits,  # noqa: F401 (re-export for tests)
    compute_solar_access_grid,
    compute_solar_access_streets,
    compute_sunlit_matrix,
)

# Keep constants that were defined here for backward compatibility
SOLAR_CMAP = "YlOrRd_r"
MAX_RAY_LENGTH = 10_000.0

# Default date re-exported from the old location
DEFAULT_DATE = "2026-06-21"


# Backward-compatible wrapper: the old _batch_solar_access returned an integer
# count array.  The new code uses compute_sunlit_matrix which returns a full
# boolean matrix.  This thin wrapper converts back to counts.


def _batch_solar_access(
    observer_points: _np.ndarray,
    sun_directions: _np.ndarray,
    scene_mesh: _pv.PolyData,
    ray_length: float = MAX_RAY_LENGTH,
) -> _np.ndarray:
    """Backward-compatible batch solar access returning unobstructed counts.

    Delegates to ``compute_sunlit_matrix`` and sums across sun positions.
    """
    sunlit = compute_sunlit_matrix(
        observer_points, sun_directions, scene_mesh, ray_length=ray_length, n_jobs=1,
    )
    return sunlit.sum(axis=1).astype(_np.int32)

# plot_solar_access will be importable once src.solar.visualize is created.
# For now, import from the original location if needed:
try:
    from src.solar.visualize import plot_solar_access  # noqa: F401
except ImportError:
    # visualize module not yet created; keep the original implementation
    # available as a fallback so existing callers don't break.
    import logging
    import matplotlib.pyplot as plt
    from pathlib import Path
    from typing import Optional, Tuple

    import geopandas as gpd

    from src.config import DPI

    _logger = logging.getLogger(__name__)

    def plot_solar_access(
        gdf: gpd.GeoDataFrame,
        output_path: Path,
        footprints_gdf: Optional[gpd.GeoDataFrame] = None,
        title: str = "Solar Access",
        cmap: str = SOLAR_CMAP,
        column: str = "solar_hours",
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = DPI,
    ) -> None:
        """Create a publication-quality solar access map."""
        from src.cartography import (
            add_north_arrow,
            add_scale_bar,
            apply_publication_style,
            format_utm_axes,
            style_buildings_by_height,
        )

        apply_publication_style()
        fig, ax = plt.subplots(figsize=figsize)

        if footprints_gdf is not None and len(footprints_gdf) > 0:
            style_buildings_by_height(ax, footprints_gdf)

        vmin = 0.0
        vmax = gdf[column].max() if len(gdf) > 0 else 1.0
        if vmax == 0:
            vmax = 1.0

        gdf.plot(
            ax=ax,
            column=column,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            markersize=4,
            legend=True,
            legend_kwds={
                "label": "Hours of Direct Sunlight",
                "shrink": 0.7,
            },
            zorder=5,
        )

        add_scale_bar(ax)
        add_north_arrow(ax)
        format_utm_axes(ax)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_aspect("equal")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        _logger.info("Saved solar access map to %s", output_path)

__all__ = [
    "compute_sun_positions",
    "sun_position_to_direction",
    "compute_solar_access_grid",
    "compute_solar_access_streets",
    "plot_solar_access",
    "DEFAULT_LATITUDE",
    "DEFAULT_LONGITUDE",
    "DEFAULT_DATE",
    "SOLAR_CMAP",
    "MAX_RAY_LENGTH",
]
