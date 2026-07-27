"""
Data path resolution for SVF v2.

Explicit registry of file names per area, with a glob fallback for unknown areas.
"""

from pathlib import Path

from src.config import DATA_DIR, get_area_data_dir

# City-wide municipal fallback layers (data/RJ/), used when an area has no
# per-site directory. Favelas_Limit_2019.shp holds 1,074 favela polygons.
RJ_DATA_DIR = DATA_DIR / "RJ"
FAVELAS_LIMIT = RJ_DATA_DIR / "Favelas_Limit_2019.shp"
SELECTED_FAVELAS_DIR = RJ_DATA_DIR / "selected_favelas"

# Known file names per area (verified against data/ directory)
AREA_FILES = {
    "vidigal_tls": {
        "dtm": "vidigal_dtm_cropped.tif",
        "footprints": "vidigal_buildings.shp",
        "roads": "roads_vidigal.shp",
        "boundary": "Vidigal_Limit.shp",  # shares vidigal's boundary
    },
    "vidigal": {
        "dtm": "DTM_Vidigal.tif",
        "footprints": "Vidigal_buildings.shp",
        "roads": "Vidigal_roads.shp",
        "boundary": "Vidigal_Limit.shp",
    },
    "riodaspedras": {
        "dtm": "riodaspedras_dtm.tif",
        "footprints": "riodaspedras_buildings.shp",
        "roads": "roads_riodaspedras.shp",
        "boundary": "riodaspedras_boundary.shp",
    },
    "rocinha": {
        "dtm": "rocinha_dtm.tif",
        "footprints": "rocinha_buildings.shp",
        "roads": "roads_rocinha.shp",
        "boundary": "rocinha_boundary.shp",
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
        "boundary": "complexo_do_alemao_boundary.shp",
    },
    "maré": {
        "dtm": "mare_dtm.tif",
        "footprints": "buildings_mare.shp",
        "roads": "street_mare.shp",
        "boundary": "mare_boundary.shp",
    },
    "borel": {
        "dtm": "borel_dtm.tif",
        "footprints": "borel_buildings.shp",
        "roads": "roads_borel.shp",
        "boundary": "borel_boundary.shp",
    },

    "jacarezinho": {
        "dtm": "jacarezinho_dtm.tif",
        "footprints": "jacarezinho_buildings.shp",
        "roads": "roads_jacarezinho.shp",
        "boundary": "jacarezinho_boundary.shp",
    },

    "morro_do_juramento": {
        "dtm": "morro_do_juramento_dtm.tif",
        "footprints": "morro_do_juramento_buildings.shp",
        "roads": "roads_morro_do_juramento.shp",
        "boundary": "morro_do_juramento_boundary.shp",
    },

}


def resolve_paths(area: str) -> tuple[Path, Path, Path]:
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


def resolve_boundary(area: str) -> Path | None:
    """Resolve boundary shapefile path for a given area.

    For registered/globbable sites, returns the per-site boundary file
    (or None if none is registered or found). For any other identifier,
    falls back to the municipal favela-limits layer, matching ``area``
    against ``nome`` (case-insensitive) or ``cod_favela`` and materialising
    the single matched polygon so callers get a one-feature file.
    """
    try:
        data_dir = get_area_data_dir(area)
    except ValueError:
        return _resolve_municipal_boundary(area)

    if area in AREA_FILES and "boundary" in AREA_FILES[area]:
        path = data_dir / AREA_FILES[area]["boundary"]
        return path if path.exists() else None

    # Fallback: look for *limit* or *boundary* shapefiles
    for pattern in ("*limit*", "*boundary*"):
        matches = [
            p
            for p in data_dir.iterdir()
            if p.suffix.lower() == ".shp" and _imatches(p.stem, pattern)
        ]
        if matches:
            return matches[0]

    return None


def _resolve_municipal_boundary(area: str) -> Path:
    """Look up a single favela in the municipal limits layer by name or code.

    Writes the matched polygon to ``data/RJ/selected_favelas/{slug}.gpkg`` and
    returns that path, so the returned file carries exactly one feature.
    """
    import geopandas as gpd

    if not FAVELAS_LIMIT.exists():
        raise FileNotFoundError(f"Municipal favela layer not found: {FAVELAS_LIMIT}")

    gdf = gpd.read_file(FAVELAS_LIMIT)
    key = area.strip().lower()
    if key.isdigit():
        match = gdf[gdf["cod_favela"] == int(key)]
    else:
        match = gdf[gdf["nome"].str.strip().str.lower() == key]
    if match.empty:
        raise FileNotFoundError(
            f"No favela matching '{area}' (by nome or cod_favela) in {FAVELAS_LIMIT.name}"
        )

    slug = _slugify(area)
    SELECTED_FAVELAS_DIR.mkdir(parents=True, exist_ok=True)
    out = SELECTED_FAVELAS_DIR / f"{slug}.gpkg"
    match.iloc[[0]].to_file(out, driver="GPKG")
    return out


def _slugify(name: str) -> str:
    import re
    import unicodedata

    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_") or "favela"


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
    raise FileNotFoundError(f"No {label} file matching '{pattern}{suffix}' in {directory}")


def _imatches(name: str, pattern: str) -> bool:
    """Case-insensitive glob-like match (supports leading/trailing *)."""
    import fnmatch

    return fnmatch.fnmatch(name.lower(), pattern.lower())
