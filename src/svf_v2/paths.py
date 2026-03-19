"""
Data path resolution for SVF v2.

Explicit registry of file names per area, with a glob fallback for unknown areas.
"""

from pathlib import Path
from typing import Tuple

from src.config import get_area_data_dir

# Known file names per area (verified against data/ directory)
AREA_FILES = {
    "vidigal_tls": {
        "dtm": "vidigal_dtm_cropped.tif",
        "footprints": "vidigal_buildings.shp",
        "roads": "roads_vidigal.shp",
    },
    "vidigal": {
        "dtm": "DTM_Vidigal.tif",
        "footprints": "Vidigal_buildings.shp",
        "roads": "Vidigal_roads.shp",
    },
    "riodaspedras": {
        "dtm": "riodaspedras_dtm.tif",
        "footprints": "riodaspedras_buildings.shp",
        "roads": "roads_riodaspedras.shp",
    },
    "rocinha": {
        "dtm": "rocinha_dtm.tif",
        "footprints": "rocinha_buildings.shp",
        "roads": "roads_rocinha.shp",
    },
    "cidade_de_deus": {
        "dtm": "cidade_de_deus_dtm.tif",
        "footprints": "cidade_de_deus_buildings.shp",
        "roads": "roads_cidade_de_deus.shp",
    },
    "complexo_do_alemao": {
        "dtm": "complexo_do_alemao_dtm.tif",
        "footprints": "complexo_do_alemao_buildings.shp",
        "roads": "roads_complexo_do_alemao.shp",
    },
}


def resolve_paths(area: str) -> Tuple[Path, Path, Path]:
    """
    Resolve (dtm_path, footprints_path, roads_path) for a given area.

    Uses an explicit registry for known areas, falls back to
    case-insensitive glob for unknown areas.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    data_dir = get_area_data_dir(area)  # raises ValueError for unsupported areas

    if area in AREA_FILES:
        reg = AREA_FILES[area]
        dtm = data_dir / reg["dtm"]
        fp = data_dir / reg["footprints"]
        rd = data_dir / reg["roads"]
        for path, label in [(dtm, "DTM"), (fp, "footprints"), (rd, "roads")]:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        return dtm, fp, rd

    # Fallback: case-insensitive glob
    dtm = _find_file(data_dir, "*dtm*", ".tif", "DTM")
    fp = _find_file(data_dir, "*building*", ".shp", "footprints")
    rd = _find_file(data_dir, "*road*", ".shp", "roads")
    return dtm, fp, rd


def _find_file(directory: Path, pattern: str, suffix: str, label: str) -> Path:
    """Find a file matching pattern + suffix (case-insensitive)."""
    matches = [
        p
        for p in directory.iterdir()
        if p.suffix.lower() == suffix.lower() and _imatches(p.stem, pattern)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer shorter name (more likely the main file)
        matches.sort(key=lambda p: len(p.name))
        return matches[0]
    raise FileNotFoundError(
        f"No {label} file matching '{pattern}{suffix}' in {directory}"
    )


def _imatches(name: str, pattern: str) -> bool:
    """Case-insensitive glob-like match (supports leading/trailing *)."""
    import fnmatch

    return fnmatch.fnmatch(name.lower(), pattern.lower())
