"""Tests for scripts/build_site_pages.py (figure_organization_spec.md §3,
phase O6). Pure-logic units only — no dependency on the real ~150-run repo
or the sibling brisaverse checkout; fixtures are throwaway tmp_path trees,
same pattern as tests/test_build_results_registry.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_site_pages as bsp  # noqa: E402


# --------------------------------------------------------- _is_withheld_layer

def test_withheld_layer_blocks_runs_prefixed_path():
    assert bsp._is_withheld_layer("runs/wp07_zoom_20260917T124118Z/f6_zoom_vidigal_kwh.png")


def test_withheld_layer_blocks_cfd_subtree_with_outputs_prefix():
    # Regression: the registry's node `path` is repo-root-relative, i.e.
    # "outputs/<site>/...", NOT stripped of the leading "outputs/" segment
    # the way build_project_hub.py's own _is_withheld expects its input.
    # This exact path tripped build_mirror_manifest's assertion before the
    # off-by-one "outputs/" prefix was accounted for here.
    assert bsp._is_withheld_layer("outputs/vidigal/cfd/extended_context_700m.png")


def test_withheld_layer_blocks_svf_v2_gpkg():
    assert bsp._is_withheld_layer("outputs/maré/svf_v2/patches.gpkg")


def test_withheld_layer_blocks_morphometrics_grid():
    assert bsp._is_withheld_layer("outputs/rocinha/morphometrics/grid/cell_042.png")


def test_withheld_layer_allows_ordinary_figure():
    assert not bsp._is_withheld_layer("outputs/comparative/mingze_facade/figures/fig1_seasonal_riodaspedras.png")


def test_withheld_layer_allows_morphometrics_non_grid():
    assert not bsp._is_withheld_layer("outputs/vidigal/morphometrics/summary.png")


# -------------------------------------------------------------- site variants

def test_ascii_slug_strips_accent_and_spaces():
    assert bsp._ascii_slug("maré") == "mare"
    assert bsp._ascii_slug("Complexo do Alemão") == "complexo_do_alemao"


def test_site_variants_cover_both_spellings():
    variants = bsp._site_variants("maré", "Maré")
    assert "maré" in variants
    assert "mare" in variants


# ---------------------------------------------------------- decisions_for_site

def test_decisions_for_site_matches_ascii_mention_of_accented_site():
    decisions = [{"id": "om_release_v0_1_1", "question": "q",
                  "plain_summary": "the package page mare_om2/index.html carries status"}]
    hits = bsp.decisions_for_site("maré", "Maré", decisions)
    assert [h["id"] for h in hits] == ["om_release_v0_1_1"]


def test_decisions_for_site_no_false_positive_for_unrelated_site():
    decisions = [{"id": "ipanema_boundary", "question": "Ipanema has no bairro polygon layer"}]
    assert bsp.decisions_for_site("vidigal", "Vidigal", decisions) == []


def test_decisions_for_site_matches_display_name_only():
    decisions = [{"id": "d1", "question": "Rocinha's dashboard needs a rebuild"}]
    hits = bsp.decisions_for_site("rocinha", "Rocinha", decisions)
    assert [h["id"] for h in hits] == ["d1"]


# -------------------------------------------------------------- resolve_slots

def test_resolve_slots_all_missing_on_empty_tree(tmp_path):
    slots = bsp.resolve_slots(tmp_path, "vidigal")
    assert set(slots) == set(bsp.PRODUCT_SLOTS)
    assert all(url is None for url, _note in slots.values())
    assert all(note == "not built for this site" for _url, note in slots.values())


def test_resolve_slots_finds_dashboard_and_a3(tmp_path):
    dash = tmp_path / "outputs" / "_distribution" / "html_dashboards" / "vidigal" / "index.html"
    dash.parent.mkdir(parents=True)
    dash.write_text("<html></html>")
    a3 = tmp_path / "outputs" / "_distribution" / "site_dashboards" / "vidigal" / "folha_vidigal_A3.png"
    a3.parent.mkdir(parents=True)
    a3.write_bytes(b"\x89PNG")

    slots = bsp.resolve_slots(tmp_path, "vidigal")
    assert slots["dashboard"][0] == "/outputs/_distribution/html_dashboards/vidigal/index.html"
    assert slots["a3_sheet"][0] == "/outputs/_distribution/site_dashboards/vidigal/folha_vidigal_A3.png"
    assert slots["brief"][0] is None


def test_resolve_slots_mare_territory_map_uses_legacy_ascii_filename(tmp_path):
    tmap = tmp_path / "outputs" / "maré" / "territory" / "mare_territory_map.png"
    tmap.parent.mkdir(parents=True)
    tmap.write_bytes(b"\x89PNG")
    slots = bsp.resolve_slots(tmp_path, "maré")
    assert slots["territory_map"][0] == "/outputs/maré/territory/mare_territory_map.png"


# --------------------------------------------------------------------- check

def _write_sites_yaml(config_dir: Path, sites: dict) -> None:
    import yaml
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "sites.yaml").write_text(yaml.dump({"sites": sites}))


def test_check_fails_on_site_set_mismatch(tmp_path, monkeypatch, capsys):
    _write_sites_yaml(tmp_path / "config", {"vidigal": {"display_name": "Vidigal"}})
    monkeypatch.setattr(bsp, "CONFIG", tmp_path / "config")
    rc = bsp.check(tmp_path)
    out = capsys.readouterr().out
    assert rc == 1
    assert "site set mismatch" in out


def test_check_passes_when_site_pages_exist(tmp_path, monkeypatch, capsys):
    _write_sites_yaml(tmp_path / "config", {"vidigal": {"display_name": "Vidigal"}})
    monkeypatch.setattr(bsp, "CONFIG", tmp_path / "config")
    sites_out = tmp_path / "outputs" / "_hub" / "sites"
    sites_out.mkdir(parents=True)
    (sites_out / "vidigal.html").write_text("<html></html>")
    rc = bsp.check(tmp_path)
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out
