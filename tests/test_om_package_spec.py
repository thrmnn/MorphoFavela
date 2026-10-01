"""Tests for the P-00 spec conformance module (src/om_package/spec.py).

Structural fix (PI, 2026-09-27): the P-01..P-09 package spec used to live
only in README prose — conformance was invisible. These tests prove the
mechanical check actually reacts to a broken package (RED), not just that
it runs.

Needs the real built package (outputs/_packages/mare_om2/v0.2.0/,
gitignored) for the RED/subprocess tests — same convention as
tests/test_om_package.py; those are skipped when it's absent. The pure
structural tests (SPEC shape, pending_on ids) run unconditionally.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.om_package.spec import SPEC, conformance, conformance_rows, render_conformance_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")
PACKAGE_DIR = DEFAULT_ROOT / "outputs" / "_packages" / "mare_om2" / "v0.2.0"
TASKS_JSON = Path("/home/theo/SCL/SCR/brisaverse/shared/facts/tasks.json")

pytestmark_real = pytest.mark.skipif(
    not PACKAGE_DIR.is_dir(), reason="mare_om2 v0.2.0 package not built at the default root"
)


# --- structural: SPEC itself ------------------------------------------------

def test_spec_ids_are_p01_through_p11_in_order():
    assert [item["id"] for item in SPEC] == [f"P-{i:02d}" for i in range(1, 12)]


def test_every_spec_item_has_a_verbatim_requirement_and_parts():
    for item in SPEC:
        assert item["requirement"].strip()
        assert len(item["parts"]) >= 1
        for part in item["parts"]:
            assert part["name"]
            assert callable(part["check"])


def test_spec_item_titles_are_unique():
    titles = [item["title"] for item in SPEC]
    assert len(titles) == len(set(titles))


# --- conformance() aggregation rule -----------------------------------------

def test_item_status_all_delivered_parts_is_delivered():
    from src.om_package.spec import PartResult

    fake_item = {"id": "X", "title": "t", "requirement": "r", "parts": [
        {"name": "a", "check": lambda pd_: PartResult("a", "delivered")},
        {"name": "b", "check": lambda pd_: PartResult("b", "delivered")},
    ]}
    import src.om_package.spec as spec_mod

    old = spec_mod.SPEC
    spec_mod.SPEC = [fake_item]
    try:
        conf = conformance(Path("/nonexistent"))
    finally:
        spec_mod.SPEC = old
    assert conf["items"][0]["status"] == "delivered"


def test_item_status_all_pending_parts_is_pending():
    from src.om_package.spec import PartResult

    fake_item = {"id": "X", "title": "t", "requirement": "r", "parts": [
        {"name": "a", "check": lambda pd_: PartResult("a", "pending", pending_on=["T1"])},
        {"name": "b", "check": lambda pd_: PartResult("b", "pending", pending_on=["T2"])},
    ]}
    import src.om_package.spec as spec_mod

    old = spec_mod.SPEC
    spec_mod.SPEC = [fake_item]
    try:
        conf = conformance(Path("/nonexistent"))
    finally:
        spec_mod.SPEC = old
    assert conf["items"][0]["status"] == "pending"
    assert conf["items"][0]["pending_on"] == ["T1", "T2"]


def test_item_status_mixed_parts_is_partial():
    from src.om_package.spec import PartResult

    fake_item = {"id": "X", "title": "t", "requirement": "r", "parts": [
        {"name": "a", "check": lambda pd_: PartResult("a", "delivered")},
        {"name": "b", "check": lambda pd_: PartResult("b", "pending", pending_on=["T1"])},
    ]}
    import src.om_package.spec as spec_mod

    old = spec_mod.SPEC
    spec_mod.SPEC = [fake_item]
    try:
        conf = conformance(Path("/nonexistent"))
    finally:
        spec_mod.SPEC = old
    assert conf["items"][0]["status"] == "partial"


def _item_status_of(*parts):
    from src.om_package.spec import PartResult  # noqa: F401
    import src.om_package.spec as spec_mod

    fake_item = {"id": "X", "title": "t", "requirement": "r",
                 "parts": [{"name": p.name, "check": (lambda pd_, p=p: p)} for p in parts]}
    old = spec_mod.SPEC
    spec_mod.SPEC = [fake_item]
    try:
        return conformance(Path("/nonexistent"))["items"][0]
    finally:
        spec_mod.SPEC = old


def _descoped(name):
    from src.om_package.spec import PartResult

    return PartResult(name, "descoped", evidence="e", reason="r", decision="om_v013_descope")


def test_item_status_delivered_plus_descoped_is_delivered_scoped():
    from src.om_package.spec import PartResult

    item = _item_status_of(PartResult("a", "delivered"), _descoped("b"))
    assert item["status"] == "delivered (scoped)"
    assert item["decisions"] == ["om_v013_descope"]
    assert item["pending_on"] == []


def test_item_status_descoped_plus_pending_stays_partial():
    from src.om_package.spec import PartResult

    item = _item_status_of(PartResult("a", "delivered"), _descoped("b"), PartResult("c", "pending", pending_on=["T1"]))
    assert item["status"] == "partial"
    assert item["pending_on"] == ["T1"]


def test_item_status_only_descoped_is_descoped():
    assert _item_status_of(_descoped("a"), _descoped("b"))["status"] == "descoped"


def test_descoped_part_without_decision_id_fails():
    from src.om_package.spec import PartResult

    with pytest.raises(ValueError, match="without a decision id"):
        PartResult("a", "descoped", reason="r")
    with pytest.raises(ValueError, match="without a reason"):
        PartResult("a", "descoped", decision="om_v013_descope")
    with pytest.raises(ValueError, match="pending_on"):
        PartResult("a", "descoped", reason="r", decision="om_v013_descope", pending_on=["T"])


def test_spec_descopes_exactly_the_decided_parts():
    from src.om_package.spec import PartResult

    expected = {"terrestrial_sky_view_factor", "tree_shade_column_reserved",
                "height_change_2024_2026", "airborne_vs_terrestrial_comparison"}
    got = set()
    for item in SPEC:
        for part in item["parts"]:
            res = part["check"](Path("/nonexistent"))
            if res.status == "descoped":
                got.add(part["name"])
                assert res.decision == "om_v013_descope"
    assert got == expected


# --- every pending part names a real tasks.json id --------------------------

@pytest.mark.skipif(not TASKS_JSON.exists(), reason="brisaverse shared/facts/tasks.json not found at the default root")
@pytestmark_real
def test_every_pending_part_names_an_existing_tasks_json_id():
    tasks = json.loads(TASKS_JSON.read_text(encoding="utf-8"))["tasks"]
    known_ids = {t["id"] for t in tasks}

    conf = conformance(PACKAGE_DIR)
    named = set()
    for item in conf["items"]:
        for part in item["parts"]:
            named.update(part["pending_on"])
    assert named, "expected at least one pending part with a pending_on id in the real v0.2.0 build"
    missing = named - known_ids
    assert not missing, f"pending_on names id(s) not in tasks.json: {missing}"


@pytestmark_real
def test_conformance_rows_and_markdown_render_without_error():
    conf = conformance(PACKAGE_DIR)
    rows = conformance_rows(conf)
    assert len(rows) == len(SPEC)
    md = render_conformance_markdown(conf)
    assert "P-01" in md and "P-09" in md


# --- RED: conformance reacts to a sabotaged copy -----------------------------

def _copy_package(tmp_path: Path) -> Path:
    dest = tmp_path / "mare_om2_copy" / "v0.2.0"
    shutil.copytree(PACKAGE_DIR, dest)
    return dest


@pytestmark_real
def test_sabotage_drop_ventilation_column_flips_p06_delivered_to_partial(tmp_path):
    # P-06 has no pending or descoped part, so it is plain "delivered"
    # pre-sabotage, which lets this test show a real delivered -> partial
    # transition.
    copy_dir = _copy_package(tmp_path)
    before = conformance(copy_dir)
    p06_before = next(it for it in before["items"] if it["id"] == "P-06")
    assert p06_before["status"] == "delivered"

    points_path = copy_dir / "OM2" / "points.parquet"
    df = pd.read_parquet(points_path)
    assert "ventilation_openness_proxy" in df.columns
    df = df.drop(columns=["ventilation_openness_proxy"])
    df.to_parquet(points_path, index=False)
    (copy_dir / "OM2" / "points.csv").write_text(df.to_csv(index=False))

    after = conformance(copy_dir)
    p06_after = next(it for it in after["items"] if it["id"] == "P-06")
    assert p06_after["status"] == "partial"


@pytestmark_real
def test_sabotage_drop_building_height_flips_p04_part_but_stays_partial(tmp_path):
    # P-04 starts "delivered (scoped)" (terrestrial SVF is descoped) — the
    # mechanical part that flips is airborne_building_and_canyon, from
    # delivered to pending, and a descoped cut must not mask it: partial.
    copy_dir = _copy_package(tmp_path)
    before = conformance(copy_dir)
    p04_before = next(it for it in before["items"] if it["id"] == "P-04")
    before_part = next(p for p in p04_before["parts"] if p["name"] == "airborne_building_and_canyon")
    assert before_part["status"] == "delivered"
    assert p04_before["status"] == "delivered (scoped)"

    points_path = copy_dir / "OM2" / "points.parquet"
    df = pd.read_parquet(points_path).drop(columns=["building_height_m"])
    df.to_parquet(points_path, index=False)

    after = conformance(copy_dir)
    p04_after = next(it for it in after["items"] if it["id"] == "P-04")
    after_part = next(p for p in p04_after["parts"] if p["name"] == "airborne_building_and_canyon")
    assert after_part["status"] == "pending"
    assert p04_after["status"] == "partial"


@pytestmark_real
def test_sabotage_missing_changelog_entry_flips_p09_delivered_to_pending(tmp_path):
    copy_dir = _copy_package(tmp_path)
    before = conformance(copy_dir)
    p09_before = next(it for it in before["items"] if it["id"] == "P-09")
    assert p09_before["status"] == "delivered"

    changelog = copy_dir / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8")
    assert "## v0.2.0" in text
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("## v0.2.0"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## v0.1.3"))
    stripped = "\n".join(lines[:start] + lines[end:])
    changelog.write_text(stripped, encoding="utf-8")

    after = conformance(copy_dir)
    p09_after = next(it for it in after["items"] if it["id"] == "P-09")
    assert p09_after["status"] == "pending"


# --- shipped scripts: run for real against the built package ----------------

@pytestmark_real
def test_shipped_aggregate_to_segments_runs_from_inside_package(tmp_path):
    script = PACKAGE_DIR / "OM2" / "aggregate_to_segments.py"
    assert script.exists(), "P-03 script not shipped inside the built package"
    out = tmp_path / "segments_20m.parquet"
    result = subprocess.run(
        [sys.executable, str(script), "--points", str(PACKAGE_DIR / "OM2" / "points.parquet"),
         "--segment-m", "20", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    segments = pd.read_parquet(out)
    points = pd.read_parquet(PACKAGE_DIR / "OM2" / "points.parquet")
    assert int(segments["n_points"].sum()) == len(points)


@pytestmark_real
def test_shipped_join_shade_example_runs_from_inside_package(tmp_path):
    script = PACKAGE_DIR / "OM2" / "join_shade_example.py"
    assert script.exists(), "P-05 join example not shipped inside the built package"

    shade = pd.read_parquet(PACKAGE_DIR / "p05_building_shade.parquet")
    slice_ = shade.head(20).copy()
    shade_path = tmp_path / "shade_slice.parquet"
    slice_.to_parquet(shade_path, index=False)

    device = pd.DataFrame({
        "point_id": slice_["point_id"].tolist(),
        "Timestamp": pd.to_datetime(slice_["timestamp"]).dt.tz_localize(None),
        "Temperature": [28.0] * len(slice_),
        "Humidity": [60.0] * len(slice_),
    })
    device_path = tmp_path / "device.csv"
    device.to_csv(device_path, index=False)

    out = tmp_path / "joined.csv"
    result = subprocess.run(
        [sys.executable, str(script), "--shade", str(shade_path), "--device", str(device_path), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    joined = pd.read_csv(out)
    assert len(joined) == len(slice_)
    assert "Temperature" in joined.columns


def test_shipped_aggregate_to_segments_skips_cleanly_without_a_built_package(tmp_path):
    script = REPO_ROOT / "src" / "om_package" / "shipped" / "aggregate_to_segments.py"
    if PACKAGE_DIR.is_dir():
        pytest.skip("real package present at the default root — see the real-data variant above")
    assert script.exists()


# --- the shipped copy matches the library function exactly ------------------

def _load_shipped_module(name: str):
    import importlib.util

    path = REPO_ROOT / "src" / "om_package" / "shipped" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shipped_aggregate_to_segments_matches_library_function():
    import numpy as np

    from src.om_package.segments import aggregate_to_segments as library_fn

    shipped = _load_shipped_module("aggregate_to_segments")

    df = pd.DataFrame({
        "point_id": [f"OM2-{i:06d}" for i in range(23)],
        "route_id": ["OM_2"] * 23,
        "distance_along_m": np.arange(23, dtype=float),
        "height_m": [1.5] * 23,
        "x": np.arange(23, dtype=float),
        "y": np.zeros(23),
        "lambda_p_buffer_5m": np.linspace(0, 1, 23),
        "building_height_m": np.linspace(2, 20, 23),
    })

    lib_out = library_fn(df, 7.0)
    shipped_out = shipped.aggregate_to_segments(df, 7.0)
    pd.testing.assert_frame_equal(lib_out, shipped_out)


@pytestmark_real
def test_shipped_aggregate_to_segments_matches_library_function_on_built_package():
    from src.om_package.segments import aggregate_to_segments as library_fn

    shipped = _load_shipped_module("aggregate_to_segments")
    df = pd.read_parquet(PACKAGE_DIR / "OM2" / "points.parquet")

    lib_out = library_fn(df, 20.0)
    shipped_out = shipped.aggregate_to_segments(df, 20.0)
    pd.testing.assert_frame_equal(lib_out, shipped_out)


# --- the shipped manifest verifies against the files beside it ---------------

@pytestmark_real
def test_shipped_manifest_hashes_match_files():
    import hashlib
    manifest = json.loads((PACKAGE_DIR / "manifest.json").read_text(encoding="utf-8"))
    bad = [
        rel for rel, digest in manifest["files"].items()
        if hashlib.sha256((PACKAGE_DIR / rel).read_bytes()).hexdigest() != digest
    ]
    assert bad == []


@pytestmark_real
def test_real_package_descoped_parts_render_distinctly():
    conf = conformance(PACKAGE_DIR)
    by_id = {it["id"]: it for it in conf["items"]}
    assert by_id["P-04"]["status"] == "delivered (scoped)"
    assert by_id["P-07"]["status"] == "delivered (scoped)"
    assert by_id["P-05"]["status"] == "partial"
    assert by_id["P-05"]["pending_on"] == ["OCTOPUS_CSV", "OCTOPUS_TZ"]
    md = render_conformance_markdown(conf)
    assert "descoped \u2014 om_v013_descope" in md


@pytestmark_real
def test_no_descoped_id_appears_as_pending_in_built_package():
    import re

    from src.om_package.quality import DESCOPED_ITEMS

    conf = conformance(PACKAGE_DIR)
    for item in conf["items"]:
        for part in item["parts"]:
            if part["status"] == "pending":
                assert part["name"] not in DESCOPED_ITEMS
    q = json.loads((PACKAGE_DIR / "OM2" / "p07_quality_report.json").read_text(encoding="utf-8"))
    assert not set(q["pending_items"]) & set(DESCOPED_ITEMS)
    assert set(q["descoped_items"]) == set(DESCOPED_ITEMS)

    d = pd.read_csv(PACKAGE_DIR / "p08_data_dictionary.csv")
    for _id in DESCOPED_ITEMS:
        row = d[d["id"] == _id].iloc[0]
        assert "PENDING" not in " ".join(str(row[c]) for c in d.columns).upper()

    needle = re.compile("|".join(re.escape(i) for i in DESCOPED_ITEMS) + r"|terrestrial|tree[ _]shade", re.I)
    # frozen history entries (v0.1..v0.1.3) rightly say PENDING as it was then; only the
    # README and the current v0.2.0 changelog entry must not.
    changelog = (PACKAGE_DIR / "CHANGELOG.md").read_text(encoding="utf-8")
    current = changelog.split("## v0.1.3", 1)[0]
    texts = {"README.md": (PACKAGE_DIR / "README.md").read_text(encoding="utf-8"), "CHANGELOG.md (v0.2.0)": current}
    for name, text in texts.items():
        for n, line in enumerate(text.splitlines(), 1):
            assert not (needle.search(line) and re.search(r"\bpending\b", line, re.I)
                        and "descoped" not in line.lower()), f"{name}:{n}: {line}"
