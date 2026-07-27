"""Municipal fallback for arbitrary polygons with no per-site data/{area}/ dir.

Covers the T7 path: resolve a boundary + DTM + footprints for an area that has
no AREA_FILES entry, sourcing everything from the city-wide data/RJ/ layers.
"""

from pathlib import Path

import geopandas as gpd
import pytest
import rasterio
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).parent.parent.parent
RJ_DIR = PROJECT_ROOT / "data" / "RJ"

# Small in-coverage box near the Vidigal centroid (EPSG:31983), sized to bound
# compute on the 742 MB municipal footprints shapefile.
_CX, _CY, _HALF = 680297.1, 7455858.8, 200.0

pytestmark = pytest.mark.integration


def _rj_available() -> bool:
    return (RJ_DIR / "buildings_RJ_2019.shp").exists() and (RJ_DIR / "DTM_RJ.tif").exists()


@pytest.fixture
def test_polygon(tmp_path) -> Path:
    poly = box(_CX - _HALF, _CY - _HALF, _CX + _HALF, _CY + _HALF)
    gdf = gpd.GeoDataFrame({"name": ["arbitrary"]}, geometry=[poly], crs="EPSG:31983")
    out = tmp_path / "arbitrary.gpkg"
    gdf.to_file(out, driver="GPKG")
    return out


@pytest.mark.skipif(not _rj_available(), reason="data/RJ/ municipal layers not present")
def test_polygon_fallback_dtm_and_buildings(test_polygon, tmp_path):
    from scripts.build_extended_context import (
        Context,
        build_extended_buildings,
        build_extended_dtm,
    )

    ctx = Context(
        name="arbitrary",
        boundary=gpd.read_file(test_polygon),
        site_dtm_path=None,
        site_fp_path=None,
        output_dir=tmp_path,
        fig_dir=tmp_path,
    )

    extended, bld_path, _ = build_extended_buildings(ctx, buffer_m=50.0)
    assert bld_path.exists()
    assert len(extended) > 0
    assert "altura" in extended.columns
    assert "tipo" in extended.columns

    dtm_path, dtm_stats = build_extended_dtm(ctx, buffer_m=50.0)
    assert dtm_path.exists()
    assert dtm_stats["res_m"] == 5.0
    with rasterio.open(dtm_path) as src:
        assert src.res == (5.0, 5.0)
        assert src.crs.to_string() == "EPSG:31983"


@pytest.mark.skipif(
    not (RJ_DIR / "Favelas_Limit_2019.shp").exists(),
    reason="municipal favela limits not present",
)
def test_resolve_boundary_municipal_by_name():
    from src.svf_v2.paths import resolve_boundary

    path = resolve_boundary("Comendador Lisboa")  # no AREA_FILES / SUPPORTED_AREAS entry
    assert path is not None and path.exists()
    gdf = gpd.read_file(path)
    assert len(gdf) == 1  # single materialised favela polygon
