"""Red/green proofs for the OM2 stage cache (src/om_package/stage_cache.py)
and the build's compute/package split (scripts/build_om_package.py).

Small tmp fixtures only: no GPU, no real Maré data. The compute stage
functions are replaced by stubs that write tiny artefacts under the same
logical names as the real ones.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.om_package import layout, p10_p11, shade, stage_route, stage_shade, stage_walks
from src.om_package.io_utils import Paths, hash_tree
from src.om_package.stage_cache import (
    HashMemo,
    StageSpec,
    StaleCacheError,
    load_or_run,
    module_closure,
    stage_key,
    verify,
)

REPO = Path(__file__).resolve().parents[1]


def _load_build_module():
    spec = importlib.util.spec_from_file_location("build_om_package", REPO / "scripts" / "build_om_package.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BOM = _load_build_module()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --- stage_cache unit proofs ---------------------------------------------------------


@pytest.fixture
def toy(tmp_path):
    """A one-module 'stage' with one input file."""
    code = tmp_path / "toy_stage.py"
    code.write_text("FACTOR = 2\n")
    data = tmp_path / "input.csv"
    data.write_text("a\n1\n2\n")
    calls = []

    def fn(work):
        calls.append(1)
        df = pd.read_csv(data)
        df.to_csv(work / "table.csv", index=False)
        return {"frame": df * 2, "summary": {"n": len(df)}, "arr": np.arange(3.0), "pair": (1, 2)}

    def spec(**params):
        return StageSpec("toy", {"input": data}, params or {"step": 5}, [code], {}, tmp_path)

    return {"code": code, "data": data, "fn": fn, "calls": calls, "spec": spec, "cache": tmp_path / "cache"}


def test_hit_returns_stored_objects_without_running(toy):
    e1 = load_or_run(toy["spec"](), toy["cache"], toy["fn"])
    e2 = load_or_run(toy["spec"](), toy["cache"], toy["fn"])
    assert len(toy["calls"]) == 1 and e1.ran and not e2.ran and e1.key == e2.key
    assert e2.obj("summary") == {"n": 2}
    assert e2.obj("pair") == (1, 2)
    np.testing.assert_array_equal(e2.obj("arr"), np.arange(3.0))
    pd.testing.assert_frame_equal(e2.obj("frame"), pd.DataFrame({"a": [2, 4]}))
    assert e2.file("table.csv").read_text() == "a\n1\n2\n"
    meta = json.loads((e2.dir / "meta.json").read_text())
    assert {"key", "components", "git_head", "dirty", "seconds", "written_at", "artefacts"} <= set(meta)


def test_package_mode_never_runs_and_names_what_changed(toy):
    with pytest.raises(StaleCacheError, match="never cached"):
        load_or_run(toy["spec"](), toy["cache"], None)
    load_or_run(toy["spec"](), toy["cache"], toy["fn"])
    with pytest.raises(StaleCacheError, match="params step changed"):
        load_or_run(toy["spec"](step=10), toy["cache"], None)
    assert len(toy["calls"]) == 1


def test_red_param_change_recomputes(toy):
    load_or_run(toy["spec"](), toy["cache"], toy["fn"])
    e = load_or_run(toy["spec"](step=10), toy["cache"], toy["fn"])
    assert e.ran and len(toy["calls"]) == 2


def test_red_one_byte_of_stage_source_recomputes(toy):
    k0 = load_or_run(toy["spec"](), toy["cache"], toy["fn"]).key
    toy["code"].write_text("FACTOR = 3\n")
    with pytest.raises(StaleCacheError, match="code toy_stage.py changed"):
        load_or_run(toy["spec"](), toy["cache"], None)
    e = load_or_run(toy["spec"](), toy["cache"], toy["fn"])
    assert e.ran and e.key != k0 and len(toy["calls"]) == 2


def test_red_input_change_recomputes_despite_hash_memo(toy):
    k0 = load_or_run(toy["spec"](), toy["cache"], toy["fn"]).key
    toy["data"].write_text("a\n1\n3\n")
    e = load_or_run(toy["spec"](), toy["cache"], toy["fn"])
    assert e.ran and e.key != k0


def test_red_edited_artefact_is_refused(toy):
    e = load_or_run(toy["spec"](), toy["cache"], toy["fn"])
    with open(e.file("table.csv"), "a") as f:
        f.write("9\n")
    with pytest.raises(StaleCacheError, match="table.csv was modified"):
        verify(e.meta, e.dir)
    with pytest.raises(StaleCacheError, match="table.csv was modified"):
        load_or_run(toy["spec"](), toy["cache"], None)


def test_upstream_key_is_part_of_the_key(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    a = stage_key("s", {}, {}, [f], {"route": "aaa"}, HashMemo(None))
    b = stage_key("s", {}, {}, [f], {"route": "bbb"}, HashMemo(None))
    assert a != b


def test_parquet_objects_are_byte_deterministic(tmp_path):
    df = pd.DataFrame({"point_id": ["a", "b"], "v": [1.5, np.nan], "t": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)})
    df.to_parquet(tmp_path / "x.parquet", index=False, compression="snappy")
    df.to_parquet(tmp_path / "y.parquet", index=False, compression="snappy")
    assert _sha(tmp_path / "x.parquet") == _sha(tmp_path / "y.parquet")


def test_module_closure_follows_relative_and_local_imports(tmp_path):
    pkg = tmp_path / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text("from .b import x\n\ndef f():\n    from src.pkg import c\n")
    (pkg / "b.py").write_text("x = 1\n")
    (pkg / "c.py").write_text("import numpy\n")
    (pkg / "d.py").write_text("")
    got = {p.name for p in module_closure([pkg / "a.py"], tmp_path)}
    assert got == {"a.py", "b.py", "c.py", "__init__.py"}
    assert "b.py" not in {p.name for p in module_closure([pkg / "a.py"], tmp_path, exclude={pkg / "b.py"})}


# --- the build's compute/package split -----------------------------------------------


def test_layout_is_outside_every_compute_stage_code_hash():
    for f in BOM.STAGE_MODULES.values():
        closure = module_closure([f], BOM.CODE_ROOT, exclude=BOM.CODE_HASH_EXCLUDE)
        assert f in closure
        assert not {"layout.py", "package_docs.py", "report.py", "figures.py", "build_om_package.py"} & {p.name for p in closure}


def _fixture_root(tmp_path) -> tuple[Paths, Path]:
    root = tmp_path / "root"
    paths = Paths(root)
    matched = root / "matched"
    matched.mkdir(parents=True)
    (matched / "OM_2_20251201_evening_27durmin.csv").write_text("t\n1\n")
    for shp in (paths.buildings_mare, paths.street_mare):
        for ext in (".shp", ".shx", ".dbf"):
            shp.with_suffix(ext).parent.mkdir(parents=True, exist_ok=True)
            shp.with_suffix(ext).write_bytes(ext.encode())
    for spec in BOM.stage_specs(paths, matched, {}).values():
        for p in spec.inputs.values():
            if not Path(p).exists():
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                Path(p).write_text(Path(p).name)
    return paths, matched


def _stub_stages(monkeypatch):
    def route_stage(work, **kw):
        pts = pd.DataFrame({"point_id": ["p0", "p1"], "distance_along_m": [0.0, 1.0]})
        for key, exts in BOM.SHIPPED_TABLES["route"].items():
            for ext in exts:
                (work / f"{key}.{ext}").write_text(f"{key} {ext}\n")
        pts.to_parquet(work / "route_points.parquet", index=False)
        return {"latlon": [-22.86, -43.24], "walk_dates": ["2025-12-01"], "horizon_deg": np.zeros((2, 3)),
                "azimuths_deg": np.arange(3.0), "walks": pd.DataFrame({"walk_id": ["w1"]}),
                "walk_fixes": pd.DataFrame({"walk_id": ["w1"]}), "season": {"campaign": {}}, "regimes": [],
                "horizon_tab": pd.DataFrame({"point_id": ["p0"], "azimuth_deg": [0.0], "horizon_deg": [1.0]})}

    def shade_stage(work, **kw):
        (work / "building_shade.parquet").write_text("shade\n")
        return {"shade_summary": {"n_rows": 1}}

    def walks_stage(work, **kw):
        for key, exts in BOM.SHIPPED_TABLES["walks"].items():
            for ext in exts:
                (work / f"{key}.{ext}").write_text(f"{key} {ext}\n")
        return {"walks_summary": {"n_walks": 1}}

    monkeypatch.setattr(stage_route, "route_stage", route_stage)
    monkeypatch.setattr(stage_shade, "shade_stage", shade_stage)
    monkeypatch.setattr(stage_walks, "walks_stage", walks_stage)


def _forbid_compute(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a compute function ran in package mode")

    for mod, name in [(p10_p11, "horizon_arrays_and_table"), (shade, "point_horizon_profiles"),
                      (shade, "compute_shade_local"), (stage_route, "route_stage"),
                      (stage_shade, "shade_stage"), (stage_walks, "walks_stage")]:
        monkeypatch.setattr(mod, name, boom)


def _package(entries, out_dir: Path, readme: str) -> dict:
    """The package stage's data and manifest steps, on the fixture cache."""
    import shutil

    shutil.rmtree(out_dir / layout.DATA_DIR, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    BOM.lay_out_tables(entries, out_dir)
    (out_dir / "README.md").write_text(readme)
    manifest = {"files": hash_tree(out_dir, exclude={"manifest.json"}),
                "cache": {n: {"key": e.key} for n, e in entries.items()}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _data_hashes_by_table(manifest: dict) -> dict[str, str]:
    out = {}
    for key in layout.TABLES:
        for ext in ("parquet", "csv", "gpkg", "json"):
            rel = layout.table(key, ext)
            if rel in manifest["files"]:
                out[f"{key}.{ext}"] = manifest["files"][rel]
    return out


def _assert_manifest_matches_disk(manifest: dict, out_dir: Path) -> None:
    on_disk = {p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file()} - {"manifest.json"}
    assert set(manifest["files"]) == on_disk
    for rel, digest in manifest["files"].items():
        assert _sha(out_dir / rel) == digest, rel


def test_green_package_only_run_never_calls_the_gpu_march(tmp_path, monkeypatch):
    paths, matched = _fixture_root(tmp_path)
    cache = tmp_path / "cache"
    _stub_stages(monkeypatch)
    computed = BOM.resolve_om2_stages(paths, matched, {}, cache, run=True)
    assert all(e.ran for e in computed.values())

    _forbid_compute(monkeypatch)
    entries = BOM.resolve_om2_stages(paths, matched, {}, cache, run=False)
    assert {n: e.key for n, e in entries.items()} == {n: e.key for n, e in computed.items()}
    assert not any(e.ran for e in entries.values())
    out = tmp_path / "pkg"
    manifest = _package(entries, out, "# README\n")
    for stage, tables in BOM.SHIPPED_TABLES.items():
        for key, exts in tables.items():
            for ext in exts:
                assert layout.table(key, ext) in manifest["files"]
    _assert_manifest_matches_disk(manifest, out)


def test_red_package_only_without_cache_fails_loudly(tmp_path, monkeypatch):
    paths, matched = _fixture_root(tmp_path)
    _forbid_compute(monkeypatch)
    with pytest.raises(StaleCacheError, match="stage route: no cache entry"):
        BOM.resolve_om2_stages(paths, matched, {}, tmp_path / "cache", run=False)


def test_red_package_only_refuses_a_stale_input(tmp_path, monkeypatch):
    paths, matched = _fixture_root(tmp_path)
    _stub_stages(monkeypatch)
    BOM.resolve_om2_stages(paths, matched, {}, tmp_path / "cache", run=True)
    (matched / "OM_2_20251201_evening_27durmin.csv").write_text("t\n2\n")
    _forbid_compute(monkeypatch)
    with pytest.raises(StaleCacheError, match="inputs matched/OM_2_20251201_evening_27durmin.csv changed"):
        BOM.resolve_om2_stages(paths, matched, {}, tmp_path / "cache", run=False)


def test_red_package_only_refuses_an_edited_cached_table(tmp_path, monkeypatch):
    paths, matched = _fixture_root(tmp_path)
    _stub_stages(monkeypatch)
    entries = BOM.resolve_om2_stages(paths, matched, {}, tmp_path / "cache", run=True)
    _package(entries, tmp_path / "pkg", "# README\n")
    with open(tmp_path / "pkg" / layout.table("walks", "csv"), "a") as f:
        f.write("edited in the package\n")
    _forbid_compute(monkeypatch)
    with pytest.raises(StaleCacheError, match="walks.csv was modified"):
        BOM.resolve_om2_stages(paths, matched, {}, tmp_path / "cache", run=False)


def test_docs_only_changes_keep_data_hashes(tmp_path, monkeypatch):
    paths, matched = _fixture_root(tmp_path)
    _stub_stages(monkeypatch)
    entries = BOM.resolve_om2_stages(paths, matched, {}, tmp_path / "cache", run=True)
    out = tmp_path / "pkg"
    before = _package(entries, out, "# README\nOne wording.\n")
    before_data = _data_hashes_by_table(before)

    _forbid_compute(monkeypatch)
    monkeypatch.setitem(layout.TABLES, "walks", f"{layout.DATA_DIR}/walk_table")
    entries2 = BOM.resolve_om2_stages(paths, matched, {}, tmp_path / "cache", run=False)
    after = _package(entries2, out, "# README\nAnother wording.\n")

    assert _data_hashes_by_table(after) == before_data and len(before_data) == 16
    assert "data/walk_table.csv" in after["files"] and "data/walks.csv" not in after["files"]
    assert after["files"]["README.md"] != before["files"]["README.md"]
    assert after["cache"] == before["cache"]
    _assert_manifest_matches_disk(after, out)
