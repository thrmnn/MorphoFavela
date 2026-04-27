"""Tests for src/svf_v2/paths.py -- data path resolution."""

import pytest

from src.svf_v2.paths import resolve_paths


class TestResolvePaths:
    @pytest.mark.integration
    def test_resolve_vidigal_tls(self):
        dtm, fp, rd = resolve_paths("vidigal_tls")
        assert dtm.exists(), f"DTM not found: {dtm}"
        assert fp.exists(), f"Footprints not found: {fp}"
        assert rd.exists(), f"Roads not found: {rd}"
        assert dtm.name == "vidigal_dtm_cropped.tif"
        assert fp.name == "vidigal_buildings.shp"
        assert rd.name == "roads_vidigal.shp"

    @pytest.mark.integration
    def test_resolve_vidigal(self):
        dtm, fp, rd = resolve_paths("vidigal")
        assert dtm.exists(), f"DTM not found: {dtm}"
        assert fp.exists(), f"Footprints not found: {fp}"
        assert rd.exists(), f"Roads not found: {rd}"
        assert dtm.name == "DTM_Vidigal.tif"
        assert fp.name == "Vidigal_buildings.shp"
        assert rd.name == "Vidigal_roads.shp"

    @pytest.mark.integration
    def test_resolve_riodaspedras(self):
        dtm, fp, rd = resolve_paths("riodaspedras")
        assert dtm.exists(), f"DTM not found: {dtm}"
        assert fp.exists(), f"Footprints not found: {fp}"
        assert rd.exists(), f"Roads not found: {rd}"
        assert dtm.name == "riodaspedras_dtm.tif"
        assert fp.name == "riodaspedras_buildings.shp"
        assert rd.name == "roads_riodaspedras.shp"

    def test_unknown_area_raises(self):
        with pytest.raises(ValueError, match="Unknown area"):
            resolve_paths("nonexistent_area")
