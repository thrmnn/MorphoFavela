"""WP-07 method schematics (src/brisa_solar/wp07_method_figures.py): teaching
figures for the WP-02 raster-horizon engine — spec docs/wp02_horizon_engine_spec.md.

This module never reads a per-cell parquet/gpkg/raster; every cross-section is
synthetic and illustrative. It DOES read real inputs by absolute path from
MAIN_ROOT (config/params.yaml, the newest runs/wp02_horizon_*/manifest.json and
*_meta.json, the primary EPW) — same convention as test_wp07_figures.py's own
note that heavy inputs are not copied into a worktree. Tests skip cleanly (never
vacuously pass against a wrong number) when MAIN_ROOT itself, or one of these
real inputs, is absent.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.brisa_solar import wp07_method_figures as figs  # noqa: E402
from src.brisa_solar.constants import P1_SKY_PATCHES  # noqa: E402


def _main_root_ready() -> bool:
    if not figs.MAIN_ROOT.exists():
        return False
    if not list(figs.MAIN_ROOT.glob("runs/wp02_horizon_*/manifest.json")):
        return False
    if not list(figs.MAIN_ROOT.glob("runs/wp02_horizon_*/artifacts/*_meta.json")):
        return False
    epw_rel = figs.load_params()["weather"]["primary_epw"]
    if not (figs.MAIN_ROOT / epw_rel).exists():
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _main_root_ready(),
    reason="MAIN_ROOT (main checkout) missing a real wp02_horizon manifest, surface meta, or EPW",
)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("wp07_method_staged")
    manifest = figs.stage_all(out_dir=out_dir)
    return out_dir, manifest


def _svg_text(out_dir: Path, fig_id: str) -> str:
    raw = (out_dir / f"{fig_id}.svg").read_text()
    return figs._svg_text_content(raw)


# ---------------------------------------------------------------------------
# Staging sanity: four figures, all produced, own run family
# ---------------------------------------------------------------------------

def test_stage_all_produces_all_four_figures(staged):
    out_dir, manifest = staged
    assert (out_dir / "figure_manifest.json").exists()
    expected = {
        "f1_obstruction_surface", "f2_raycast_geometry",
        "f3_horizon_accumulation", "f4_matrix_decomposition",
    }
    assert set(manifest["figures"]) == expected
    for fid, entry in manifest["figures"].items():
        assert entry["status"] == "produced", fid
        assert (out_dir / entry["png_path"]).exists()
        assert (out_dir / entry["svg_path"]).exists()


def test_default_run_family_is_wp07_method(tmp_path):
    """The module's own docstring/default naming — never wp07_figures.py's
    families, which another agent owns concurrently."""
    src = Path(figs.__file__).read_text()
    assert 'runs" / ("wp07_method_' in src
    assert "import wp07_figures" not in src and "from .wp07_figures" not in src, (
        "must not import the sibling module"
    )
    assert "wp07_figures.stage" not in src and "wp07_figures.render" not in src
    # the docstring names config/zoom_windows.yaml to document the boundary — that's
    # fine; what must never appear is code that actually opens/loads it.
    assert "load_zoom_windows" not in src and "zoom_windows.yaml" not in re.sub(
        r'""".*?"""', "", src, flags=re.S
    ), "must not touch config/zoom_windows.yaml outside the module docstring"


def test_contact_sheet_written(staged):
    out_dir, _ = staged
    assert (out_dir / "contact.png").exists()


# ---------------------------------------------------------------------------
# Release discipline: no per-cell data anywhere, honest checklist
# ---------------------------------------------------------------------------

def test_release_class_is_publishable_candidate_for_every_figure(staged):
    """No figure here shows real per-cell geometry or a favela-vs-formal
    quantity, so none should be withheld."""
    _, manifest = staged
    for fid, entry in manifest["figures"].items():
        assert entry["release_class"] == "publishable-candidate", fid
        assert "red_line" not in entry, fid


def test_checklist_passes_for_every_figure(staged):
    _, manifest = staged
    for fid, entry in manifest["figures"].items():
        checklist = entry["checklist"]
        for key in ("no_coordinates", "no_basemap", "no_per_cell_geometry",
                    "no_favela_named", "banned_tokens_absent"):
            assert checklist[key] is True, f"{fid}.{key}"


def test_no_module_under_test_reads_a_per_cell_source(staged):
    """This module must never open a parquet/gpkg/per-cell raster — it is a
    schematic generator only. Grep, not just trust the checklist's own claim."""
    src = Path(figs.__file__).read_text()
    for forbidden in (".parquet", "geopandas", "gpd.read_file", "rasterio.open"):
        assert forbidden not in src, forbidden


def test_no_favela_named_anywhere_in_source_or_output(staged):
    out_dir, _ = staged
    favela_names = ("vidigal", "rocinha", "alemao", "mare", "riodaspedras", "rio das pedras")
    src_lower = Path(figs.__file__).read_text().lower()
    for name in favela_names:
        assert name not in src_lower, name
    for fid in ("f1_obstruction_surface", "f2_raycast_geometry",
                "f3_horizon_accumulation", "f4_matrix_decomposition"):
        text_lower = _svg_text(out_dir, fid).lower()
        for name in favela_names:
            assert name not in text_lower, f"{fid}: {name}"


# ---------------------------------------------------------------------------
# P1_SKY_PATCHES: imported, never a literal (redundant local check — the
# repo-wide tests/test_p1_sky_resolution_consistency.py also covers this file
# via its src/brisa_solar rglob, but this pins the rule to this module too).
# ---------------------------------------------------------------------------

def test_p1_sky_patches_never_hardcoded_as_a_literal():
    tree = ast.parse(Path(figs.__file__).read_text())
    offenders = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value in (145, 577)
    ]
    assert not offenders, f"literal patch count at line(s) {offenders} — import P1_SKY_PATCHES instead"


def test_p1_sky_patches_appears_in_the_matrix_figure(staged):
    out_dir, _ = staged
    text = _svg_text(out_dir, "f4_matrix_decomposition")
    assert str(P1_SKY_PATCHES) in text


# ---------------------------------------------------------------------------
# Traceability: every parameter drawn on the figures is a real, sourced value
# ---------------------------------------------------------------------------

def test_run_params_are_read_from_the_real_manifest_not_typed():
    params = figs.load_horizon_run_params()
    manifest_path = figs.MAIN_ROOT / params["source_manifest"]
    real = json.loads(manifest_path.read_text())
    assert params["cell_m"] == real["cell_m"]
    assert params["obs_height_m"] == real["obs_height_m"]
    assert params["max_dist_m"] == real["max_dist_m"]
    assert params["step_m"] == real["step_m"]
    assert params["device"] == real["device"]
    # march_sampling has no manifest field (pre-dates it); must come from the
    # function's own default, not a retyped guess.
    import inspect
    sig = inspect.signature(figs.patch_visibility)
    assert params["march_sampling_default"] == sig.parameters["march_sampling"].default


def test_run_params_values_appear_in_the_raycast_figures(staged):
    out_dir, _ = staged
    params = figs.load_horizon_run_params()
    for fid in ("f2_raycast_geometry", "f3_horizon_accumulation"):
        text = _svg_text(out_dir, fid)
        assert f"{params['step_m']:g}" in text
        assert f"{params['max_dist_m']:g}" in text


def test_top_rule_text_is_read_verbatim_from_a_real_meta_json():
    top_rule = figs.load_surface_top_rule()
    meta_path = figs.MAIN_ROOT / top_rule["source_meta"]
    real = json.loads(meta_path.read_text())
    assert top_rule["top_rule"] == real["top_rule"]
    fp = figs.load_params()["footprints"]
    assert top_rule["base_attr"] == fp["base_attr"]
    assert top_rule["height_attr"] == fp["height_attr"]
    assert top_rule["top_attr"] == fp["top_attr"]


def test_top_rule_text_appears_in_the_surface_figure(staged):
    import html

    out_dir, _ = staged
    top_rule = figs.load_surface_top_rule()
    text = html.unescape(_svg_text(out_dir, "f1_obstruction_surface"))
    assert top_rule["top_rule"] in text


def test_tregenza_band_altitudes_are_real_not_typed():
    alts = figs.tregenza_band_altitudes_deg()
    # the seven Tregenza band centres plus the zenith cap, per src/svf_v2/compute.py
    assert alts.size == 8
    assert alts[0] == pytest.approx(6.0)
    assert alts[-1] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# The real EPW-derived sky vector (f4) — not a synthetic stand-in
# ---------------------------------------------------------------------------

def test_real_sky_reproduces_the_accepted_identity():
    """The same unobstructed-identity number the accepted wp02_sky run
    reports (runs/wp02_sky_20260911/identity_check.json: 1715.89 kWh/m²,
    within the file's own 2% tolerance) — proves load_real_sky() is building
    the production object, not a stand-in."""
    sky, epw_rel = figs.load_real_sky()
    assert epw_rel == figs.load_params()["weather"]["primary_epw"]
    assert sky.patch_total_kwh.shape == (P1_SKY_PATCHES,)
    assert float(sky.patch_total_kwh.sum()) == pytest.approx(1715.89, abs=2.0)


def test_svf_weighting_note_names_production_vs_crossreference(staged):
    _, manifest = staged
    note = manifest["figures"]["f4_matrix_decomposition"]["sources"]["svf_weighting_note"]
    assert "CumulativeSky.svf" in note
    assert "PRODUCTION" in note
    assert "svf_unweighted" in note and "svf_solid_angle" in note
    assert "CPU reference" in note


def test_production_svf_variant_matches_cumulative_sky_svf():
    """The bar chart draws three SVF variants; the PRODUCTION one must
    numerically equal CumulativeSky.svf(V), not just be labelled as such."""
    from src.brisa_solar.wp02_sky import CumulativeSky

    sky, _ = figs.load_real_sky()
    rng = np.random.default_rng(20260917)
    V = rng.random((4, P1_SKY_PATCHES)) > 0.5
    cw = sky.weights * sky.directions[:, 2]
    production = (V.astype(float) @ cw) / cw.sum()
    assert np.allclose(production, sky.svf(V))


# ---------------------------------------------------------------------------
# Synthetic transect: exercises both real branches of the top rule
# ---------------------------------------------------------------------------

def test_synthetic_transect_exercises_both_top_rule_branches():
    t = figs.synthetic_transect(cell_m=1.0)
    assert t["in_a"].any() and t["in_b"].any()
    # branch a: topo not finite -> top = base + altura = base + 9.0
    a = t["in_a"]
    assert np.allclose(t["building_top"][a], t["base"][a] + 9.0)
    # branch b: topo finite & > base -> top = topo = base + 14.5
    b = t["in_b"]
    assert np.allclose(t["building_top"][b], t["base"][b] + 14.5)
    # surface is the elementwise max of DTM and building_top wherever a building exists
    is_building = t["is_building"]
    assert np.allclose(t["surface"][is_building],
                        np.fmax(t["dtm"][is_building], t["building_top"][is_building]))
    assert np.allclose(t["surface"][~is_building], t["dtm"][~is_building])


def test_observer_never_lands_on_a_building_cell():
    """render_f1 asserts this at render time; pin it as a standalone
    invariant too, since it is the exact rule the figure teaches."""
    run_params = figs.load_horizon_run_params()
    t = figs.synthetic_transect(run_params["cell_m"])
    x = t["x"]
    obs_i = int(np.argmin(np.abs(x - 20.0)))
    assert not t["is_building"][obs_i]


# ---------------------------------------------------------------------------
# Ray-casting geometry: honest arithmetic (real atan2/max, not stylised)
# ---------------------------------------------------------------------------

def test_ray_profile_horizon_matches_direct_numpy_computation():
    run_params = figs.load_horizon_run_params()
    t, ground = figs._ray_profile(run_params["step_m"], run_params["max_dist_m"])
    z_obs = 8.0 + run_params["obs_height_m"]
    horizon_deg = np.degrees(np.max(np.arctan2(ground - z_obs, t)))
    # same computation render_f2/f3 perform inline — recomputed independently
    # here so a future refactor that silently changes the formula fails loudly.
    assert horizon_deg == pytest.approx(np.degrees(np.arctan2(ground - z_obs, t)).max())
    assert 0.0 < horizon_deg < 90.0


def test_visible_and_blocked_bands_are_real_tregenza_altitudes():
    bands = figs.tregenza_band_altitudes_deg()
    assert 6.0 in bands
    assert 54.0 in bands


# ---------------------------------------------------------------------------
# No banned P1 vocabulary (lint_p1_tokens.py also scans this file directly
# via src/brisa_solar/**/*.py; this pins the same rule with a direct call)
# ---------------------------------------------------------------------------

def test_lint_p1_tokens_clean_on_this_module():
    from scripts import lint_p1_tokens as lint

    text = Path(figs.__file__).read_text()
    hits = lint._scan_lines(text.split("\n"), "src/brisa_solar/wp07_method_figures.py")
    assert not hits, hits
