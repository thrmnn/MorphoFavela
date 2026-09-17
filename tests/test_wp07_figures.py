"""WP-07B staged figure candidates: spec docs/wp07_figures_spec.md, deliverable's
five tests (a)-(e). Reads only the newest runs/wp07_ledger_*/ledger.json and skips
cleanly when it is absent (never passes vacuously against a wrong number, only
against a missing input — same convention as tests/test_wp07_ledger.py).

The real parquet inputs (`runs/wp05_full_20260914T215419Z/wp05_full.parquet`,
`runs/wp04_sites_20260914T230606Z/<site>/ground.parquet`) are gitignored and, like
tests/test_wp05_full.py notes for its own inputs, are not copied into a worktree.
f1/f2 exercise their real rendering logic here against synthetic parquet fixtures
instead (same pattern as test_wp05_full.py's synthetic-input tests); f3/f4 read
only the ledger and render for real against the actual worktree ledger.
"""
from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.brisa_solar import wp07_figures as figs
from src.brisa_solar import wp07_ledger as ledger_mod

RUNS = ROOT / "runs"


def _ledger_present() -> bool:
    return bool(list(RUNS.glob("wp07_ledger_*/ledger.json")))


pytestmark = pytest.mark.skipif(not _ledger_present(), reason="no runs/wp07_ledger_*/ledger.json present")


@pytest.fixture(scope="module")
def ledger():
    return json.loads(figs.find_latest_ledger(ROOT).read_text())


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("wp07_figures_staged")
    manifest = figs.stage_all(ROOT, out_dir=out_dir)
    return out_dir, manifest


def _produced(manifest: dict) -> dict:
    return {fid: f for fid, f in manifest["figures"].items() if f["status"] == "produced"}


# ---------------------------------------------------------------------------
# Basic staging sanity (real worktree run: f3/f4 ledger-only, always produced)
# ---------------------------------------------------------------------------

def test_staging_produces_a_manifest_and_ledger_only_figures(staged):
    out_dir, manifest = staged
    assert (out_dir / "figure_manifest.json").exists()
    produced = _produced(manifest)
    assert "f3_domain_sensitivity" in produced, "ledger-only figure must always render"
    assert "f4_geometry_constraints" in produced, "ledger-only figure must always render"
    for fid in ("f1_citywide_position", "f2_direct_sun_reference_days"):
        f = manifest["figures"][fid]
        if f["status"] == "skipped":
            assert "reason" in f and f["reason"]


def test_manifest_records_the_table_h_fallback_order(staged):
    _, manifest = staged
    assert manifest["site_order"]["order"] == list(figs.FIGURE_SITE_ORDER)
    assert manifest["site_order"]["order"] == [
        "vidigal", "rocinha", "complexo_do_alemao", "mare", "riodaspedras",
    ]
    assert "figures_table_h" in manifest["site_order"]["table_h_search"]


# ---------------------------------------------------------------------------
# (a) every ledger_ids_used exists in the ledger, and every such value's
# formatted (3-sig-fig) text appears in the figure's own SVG text content.
# ---------------------------------------------------------------------------

def test_a_ledger_ids_exist_and_their_values_appear_in_svg_text(staged, ledger):
    out_dir, manifest = staged
    produced = _produced(manifest)
    assert produced
    for fid, f in produced.items():
        raw = (out_dir / f["svg_path"]).read_text()
        text = figs._svg_text_content(raw)
        assert f["ledger_ids_used"], f"{fid}: no ledger_ids_used recorded"
        for lid in f["ledger_ids_used"]:
            value, _unit = figs.get_value(ledger, lid)  # raises KeyError if id doesn't exist
            expected = figs.fmt3(value)
            assert expected in text, f"{fid}: {lid} = {expected!r} not found in SVG text"
        for lid in f.get("ledger_ids_plotted", []):
            figs.get_value(ledger, lid)  # plotted as a marker position, never printed


# ---------------------------------------------------------------------------
# (b) no 6-7 digit integer runs (UTM coordinates) and no <image> (basemap)
# ---------------------------------------------------------------------------

def test_b_no_coordinate_leaks_no_basemap(staged):
    out_dir, manifest = staged
    for fid, f in _produced(manifest).items():
        raw = (out_dir / f["svg_path"]).read_text()
        text = figs._svg_text_content(raw)
        hit = figs._COORD_RE.search(text)
        assert hit is None, f"{fid}: looks like a UTM coordinate leaked into SVG text: {hit}"
        assert "<image" not in raw, f"{fid}: basemap <image> element present"
        assert f["checklist"]["no_coordinates"] is True
        assert f["checklist"]["no_basemap"] is True


# ---------------------------------------------------------------------------
# (c) svg_path_count < 2,000 per figure
# ---------------------------------------------------------------------------

def test_c_svg_path_count_under_2000(staged):
    out_dir, manifest = staged
    for fid, f in _produced(manifest).items():
        raw = (out_dir / f["svg_path"]).read_text()
        count = raw.count("<path ")
        assert count == f["checklist"]["svg_path_count"]
        assert count < 2000, f"{fid}: {count} paths"


# ---------------------------------------------------------------------------
# (d) banned tokens absent — call scripts/lint_p1_tokens.py's own scan function
# ---------------------------------------------------------------------------

def test_d_banned_tokens_absent_in_svg_and_manifest(staged):
    import lint_p1_tokens as lt
    importlib.reload(lt)

    out_dir, manifest = staged
    hits = lt._scan_lines(json.dumps(manifest).split("\n"), "figure_manifest.json")
    assert not hits, hits

    for fid, f in _produced(manifest).items():
        raw = (out_dir / f["svg_path"]).read_text()
        text = figs._svg_text_content(raw)
        hits = lt._scan_lines(text.split("\n"), fid)
        assert not hits, hits
        assert f["checklist"]["banned_tokens_absent"] is True


# ---------------------------------------------------------------------------
# (e) the module never imports src.cfd_integration or scripts.analyze_cfd_results
# ---------------------------------------------------------------------------

_BANNED_MODULES = ("src.cfd_integration", "cfd_integration", "scripts.analyze_cfd_results", "analyze_cfd_results")


def _is_banned(name: str) -> bool:
    return any(name == b or name.startswith(b + ".") for b in _BANNED_MODULES)


def test_e_never_imports_cfd_modules():
    source = (ROOT / "src" / "brisa_solar" / "wp07_figures.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not _is_banned(alias.name), alias.name
        elif isinstance(node, ast.ImportFrom):
            assert not _is_banned(node.module or ""), node.module

    before = set(sys.modules)
    importlib.import_module("src.brisa_solar.wp07_figures")
    new_mods = set(sys.modules) - before
    assert not any(_is_banned(m) for m in new_mods), new_mods


# ---------------------------------------------------------------------------
# Supporting unit coverage: ledger value resolution, grid discovery, the
# forbidden-column guard, and f1/f2 rendering against synthetic parquet
# fixtures shaped like the real (but locally absent) inputs.
# ---------------------------------------------------------------------------

def test_get_value_resolves_derived_spread_ids(ledger):
    val, unit = figs.get_value(ledger, "g3.spread.vidigal.svf")
    assert unit == "percentile"
    assert val == ledger["derived"]["spread"]["vidigal"]["svf_percentile_spread_max_minus_min"]

    with pytest.raises(KeyError):
        figs.get_value(ledger, "not.a.real.id")


def test_discover_grid_variants_matches_locked_domain(ledger):
    variants = figs.discover_grid_variants(ledger)
    assert len(variants) == 9
    assert any((t, d) == ledger_mod.LOCKED_VARIANT for _, t, d in variants)
    # sorted by (threshold, distance): same-threshold variants are adjacent
    thresholds = [t for _, t, _ in variants]
    assert thresholds == sorted(thresholds)


def test_read_columns_refuses_per_cell_geometry():
    for cols in (["x"], ["y"], ["row", "svf"], ["col"]):
        with pytest.raises(AssertionError):
            figs._read_columns(Path("unused.parquet"), cols)


def test_f1_skips_cleanly_when_citywide_parquet_absent(tmp_path, ledger):
    repo_root = tmp_path / "repo_empty"
    (repo_root / "runs").mkdir(parents=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = figs.render_f1(ledger, repo_root, out_dir)
    assert result["status"] == "skipped"
    assert "reason" in result


def test_f2_skips_cleanly_when_a_site_parquet_is_missing(tmp_path, ledger):
    repo_root = tmp_path / "repo_partial"
    run_dir = repo_root / "runs" / ledger_mod.RUN_OF_RECORD["wp04"]
    run_dir.mkdir(parents=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = figs.render_f2(ledger, repo_root, out_dir)
    assert result["status"] == "skipped"
    assert "reason" in result


def _write_synthetic_citywide_parquet(repo_root: Path, n: int = 4000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "svf": rng.uniform(0.0, 1.0, n).astype("float32"),
        "kwh_m2": rng.uniform(0.0, 1800.0, n).astype("float32"),
        "x": rng.uniform(0.0, 1.0, n),
        "y": rng.uniform(0.0, 1.0, n),
    })
    run_dir = repo_root / "runs" / ledger_mod.RUN_OF_RECORD["wp05"]
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(run_dir / "wp05_full.parquet")


def _write_synthetic_site_parquets(repo_root: Path, n: int = 400, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    run_dir = repo_root / "runs" / ledger_mod.RUN_OF_RECORD["wp04"]
    for slug in figs.FIGURE_SITE_ORDER:
        site_dir = run_dir / ledger_mod.SITE_DIRS[slug]
        site_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "hours_winter_solstice": rng.uniform(0.0, 10.0, n),
            "hours_equinox": rng.uniform(0.0, 12.0, n),
            "x": rng.uniform(0.0, 1.0, n),
            "y": rng.uniform(0.0, 1.0, n),
        })
        df.to_parquet(site_dir / "ground.parquet")


def test_f1_renders_from_synthetic_citywide_parquet(tmp_path, ledger):
    repo_root = tmp_path / "repo"
    _write_synthetic_citywide_parquet(repo_root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = figs.render_f1(ledger, repo_root, out_dir)
    assert result["status"] == "produced"
    assert (out_dir / result["svg_path"]).exists()
    assert (out_dir / result["png_path"]).exists()
    assert result["checklist"]["svg_path_count"] < 2000
    assert result["checklist"]["no_coordinates"] is True
    assert result["checklist"]["no_basemap"] is True
    for lid in result["ledger_ids_used"]:
        figs.get_value(ledger, lid)


def test_f2_renders_from_synthetic_site_parquets(tmp_path, ledger):
    repo_root = tmp_path / "repo"
    _write_synthetic_site_parquets(repo_root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = figs.render_f2(ledger, repo_root, out_dir)
    assert result["status"] == "produced"
    assert (out_dir / result["svg_path"]).exists()
    assert (out_dir / result["png_path"]).exists()
    assert len(result["ledger_ids_used"]) == 10  # 5 sites x 2 reference days
    for lid in result["ledger_ids_used"]:
        figs.get_value(ledger, lid)


def test_f3_and_f4_render_for_real_in_this_worktree(ledger, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    r3 = figs.render_f3(ledger, ROOT, out_dir)
    r4 = figs.render_f4(ledger, ROOT, out_dir)
    assert r3["status"] == "produced"
    assert r4["status"] == "produced"
    assert r3["source_parquets"] == []
    assert r4["source_parquets"] == []


# ---------------------------------------------------------------------------
# WP-07M (docs/wp07_map_spec.md): f5/f5b, the citywide map. Staged into its
# own runs/wp07_map_<UTC>/ (separate manifest from f1-f4 above — see
# wp07_figures.stage_map's docstring), release_class "withheld" under red
# line L1, so no promotion path exists no matter what this manifest says.
# Same skip-cleanly-on-missing-input discipline as f1/f2: both the citywide
# parquet and the favela boundary shapefile are gitignored and absent from a
# bare worktree, so real rendering is exercised here against synthetic
# fixtures shaped like the real (but locally absent) inputs.
# ---------------------------------------------------------------------------

def _write_synthetic_favela_shapefile(repo_root: Path) -> None:
    from src.config import EXPECTED_CRS

    rows = []
    for i, display in enumerate(figs.FAVELAS.values()):
        x0 = float(i)
        rows.append({"complexo": display, "nome": display, "geometry": box(x0, x0, x0 + 0.4, x0 + 0.4)})
    gdf = gpd.GeoDataFrame(rows, crs=EXPECTED_CRS)
    out_dir = repo_root / "data" / "RJ"
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_dir / "Favelas_Limit_2019.shp")


def _write_synthetic_map_parquet(repo_root: Path, n: int = 6000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "svf": rng.uniform(0.0, 1.0, n).astype("float32"),
        "kwh_m2": rng.uniform(0.0, 1800.0, n).astype("float32"),
        "x": rng.uniform(0.0, 5.0, n),
        "y": rng.uniform(0.0, 5.0, n),
        "favela_id": rng.integers(0, 2, n).astype("int32"),
    })
    run_dir = repo_root / "runs" / ledger_mod.RUN_OF_RECORD["wp05"]
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(run_dir / "wp05_full.parquet")


def test_f5_skips_cleanly_when_citywide_parquet_absent(tmp_path, ledger):
    repo_root = tmp_path / "repo_empty"
    (repo_root / "runs").mkdir(parents=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = figs.render_f5(ledger, repo_root, out_dir)
    assert result["status"] == "skipped"
    assert "reason" in result


def test_f5_skips_cleanly_when_favela_boundaries_absent(tmp_path, ledger):
    repo_root = tmp_path / "repo_no_boundaries"
    _write_synthetic_map_parquet(repo_root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = figs.render_f5(ledger, repo_root, out_dir)
    assert result["status"] == "skipped"
    assert "reason" in result


def test_f5_and_f5b_render_from_synthetic_citywide_data(tmp_path, ledger):
    repo_root = tmp_path / "repo"
    _write_synthetic_map_parquet(repo_root)
    _write_synthetic_favela_shapefile(repo_root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    r5 = figs.render_f5(ledger, repo_root, out_dir)
    r5b = figs.render_f5b(ledger, repo_root, out_dir)

    for fid, result in (("f5_citywide_svf_map", r5), ("f5b_citywide_svf_map_coarse", r5b)):
        assert result["status"] == "produced", result
        assert (out_dir / result["svg_path"]).exists()
        assert (out_dir / result["png_path"]).exists()
        assert result["release_class"] == "withheld"
        assert result["red_line"] == "L1"
        raw = (out_dir / result["svg_path"]).read_text()
        text = figs._svg_text_content(raw)
        hit = figs._COORD_RE.search(text)
        assert hit is None, f"{fid}: looks like a UTM coordinate leaked into SVG text: {hit}"
        assert result["checklist"]["no_coordinates"] is True

    from src.brisa_solar.g3_domain import FABRIC_FOOTPRINT_DISTANCE_GRID_M
    assert r5b["aggregation"]["cell_m"] == max(FABRIC_FOOTPRINT_DISTANCE_GRID_M)


def test_f5_never_computes_a_favela_vs_non_favela_quantity(tmp_path, ledger):
    """The hard boundary (docs/wp07_map_spec.md 'The boundary'): no favela-vs-
    non-favela contrast, difference, ratio or deficit — not as a layer, not
    as a legend, not as an annotation. Boundaries are positions only."""
    repo_root = tmp_path / "repo"
    _write_synthetic_map_parquet(repo_root)
    _write_synthetic_favela_shapefile(repo_root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    r5 = figs.render_f5(ledger, repo_root, out_dir)
    r5b = figs.render_f5b(ledger, repo_root, out_dir)
    assert r5["status"] == "produced" and r5b["status"] == "produced"

    banned = ("formal", "non_favela", "deficit", "difference", "ratio")
    for result in (r5, r5b):
        blob = json.dumps(result).lower()
        for token in banned:
            assert token not in blob, f"{result['id']}: manifest carries banned token {token!r}"
        raw_svg = (out_dir / result["svg_path"]).read_text().lower()
        for token in banned:
            assert token not in raw_svg, f"{result['id']}: SVG carries banned token {token!r}"


def test_stage_map_writes_its_own_manifest_and_contact_sheet(tmp_path, ledger):
    repo_root = tmp_path / "repo"
    _write_synthetic_map_parquet(repo_root)
    _write_synthetic_favela_shapefile(repo_root)
    # stage_map re-derives its own ledger via find_latest_ledger(repo_root);
    # give the synthetic repo a copy of the real worktree ledger to read.
    ledger_src = figs.find_latest_ledger(ROOT)
    ledger_dst = repo_root / "runs" / ledger_src.parent.name
    ledger_dst.mkdir(parents=True, exist_ok=True)
    (ledger_dst / "ledger.json").write_text(ledger_src.read_text())

    out_dir = tmp_path / "wp07_map_out"
    manifest = figs.stage_map(repo_root, out_dir=out_dir)

    assert (out_dir / "figure_manifest.json").exists()
    produced = {fid: f for fid, f in manifest["figures"].items() if f["status"] == "produced"}
    assert set(produced) == {"f5_citywide_svf_map", "f5b_citywide_svf_map_coarse"}
    for f in produced.values():
        assert f["release_class"] == "withheld"
        assert f["red_line"] == "L1"
    assert (out_dir / "contact.png").exists()
