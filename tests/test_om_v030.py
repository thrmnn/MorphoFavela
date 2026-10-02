"""v0.2.0 integration: P-10 (sun exposure) and P-11 (ventilation indices with
time-matched wind) files, columns, dictionary coverage, conformance and
shipped hashes. Needs the built package (outputs/_packages/mare_om2/v0.2.0/)."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.om_package import p10_p11
from src.om_package.io_utils import DEFAULT_ROOT
from src.om_package.spec import conformance

ROOT = DEFAULT_ROOT / "outputs" / "_packages" / "mare_om2"
PKG = ROOT / "v0.2.0"
real = pytest.mark.skipif(not PKG.is_dir(), reason="mare_om2 v0.2.0 package not built at the default root")

NEW_POINT_COLUMNS = [
    "annual_sun_hours", "windward_lambda_f_prevailing", "canyon_alignment_prevailing_deg",
    "upwind_shelter_deg_prevailing", "z0_macdonald_m", "zd_macdonald_m", "open_space_fraction",
]
NEW_FILES = [
    "p10_sun_envelope.parquet", "p10_sun_envelope.csv", "p10_sun_dose.parquet", "p10_sun_dose.csv",
    "p10_horizon_profiles.parquet", "p10_clock_agreement.parquet", "p10_clock_agreement.csv", "p11_wind_observed.csv",
]
NEW_FIGURES = ["sun_envelope.png", "sun_dose.png", "map_vent_shelter.png", "profiles_vent.png", "wind_rose_compare.png"]


def test_new_point_columns_constant_matches_module():
    assert p10_p11.NEW_POINT_COLUMNS == NEW_POINT_COLUMNS


@real
@pytest.mark.parametrize("name", NEW_FILES)
def test_new_file_present(name):
    assert (PKG / name).stat().st_size > 0


@real
@pytest.mark.parametrize("name", NEW_FIGURES)
def test_new_figure_present(name):
    assert (PKG / "OM2" / name).stat().st_size > 10_000


@real
@pytest.mark.parametrize("ext", ["parquet", "csv"])
def test_new_columns_in_points(ext):
    df = pd.read_parquet(PKG / "OM2" / "points.parquet") if ext == "parquet" else pd.read_csv(PKG / "OM2" / "points.csv")
    assert set(NEW_POINT_COLUMNS) <= set(df.columns)
    assert df[NEW_POINT_COLUMNS].notna().all().all()


@real
def test_new_columns_are_consistent_with_their_inputs():
    df = pd.read_parquet(PKG / "OM2" / "points.parquet")
    assert (df["open_space_fraction"] - (1 - df["lambda_p_buffer_50m"])).abs().max() < 1e-9
    assert df["canyon_alignment_prevailing_deg"].between(0, 90).all()
    assert df["upwind_shelter_deg_prevailing"].between(-90, 90).all()
    assert (df["zd_macdonald_m"] >= 0).all() and (df["z0_macdonald_m"] >= 0).all()


@real
def test_every_shipped_table_column_has_a_dictionary_row():
    ids = set(pd.read_csv(PKG / "p08_data_dictionary.csv")["id"])
    tables = {
        "OM2/points": pd.read_parquet(PKG / "OM2" / "points.parquet"),
        "p05_building_shade": pd.read_parquet(PKG / "p05_building_shade.parquet"),
        "p10_sun_envelope": pd.read_parquet(PKG / "p10_sun_envelope.parquet"),
        "p10_sun_dose": pd.read_parquet(PKG / "p10_sun_dose.parquet"),
        "p10_horizon_profiles": pd.read_parquet(PKG / "p10_horizon_profiles.parquet"),
        "p10_clock_agreement": pd.read_parquet(PKG / "p10_clock_agreement.parquet"),
        "p11_wind_observed": pd.read_csv(PKG / "p11_wind_observed.csv"),
    }
    for name, df in tables.items():
        missing = [c for c in df.columns if c not in ids and c != "geometry"]
        assert not missing, f"{name}: no dictionary row for {missing}"


@real
def test_dictionary_rows_complete_and_ventilation_flagged_proxy():
    d = pd.read_csv(PKG / "p08_data_dictionary.csv").set_index("id")
    for col in ("definition", "unit", "source", "method", "limits", "status"):
        assert d[col].notna().all(), col
    for vid in NEW_POINT_COLUMNS[1:]:
        text = " ".join(str(d.loc[vid, c]) for c in ("definition", "limits")).upper()
        assert "PROXY" in text, vid
    for vid in ("drct", "speed_ms", "valid_utc"):
        assert "not wind at the route" in d.loc[vid, "limits"] or "not at the route" in d.loc[vid, "limits"]


@real
def test_p10_p11_conformance_delivered_and_p05_p06_untouched():
    conf = conformance(PKG)
    by = {it["id"]: it for it in conf["items"]}
    assert by["P-10"]["status"] == "delivered", by["P-10"]["evidence"]
    assert by["P-11"]["status"] == "delivered", by["P-11"]["evidence"]
    assert [p["name"] for p in by["P-10"]["parts"]] == ["sun_envelope", "sun_dose", "annual_sun_hours", "clock_agreement"]
    assert {"observed_wind", "windward_lambda_f", "canyon_alignment", "upwind_shelter", "roughness_z0_zd",
            "open_space_fraction"} <= {p["name"] for p in by["P-11"]["parts"]}
    assert by["P-05"]["status"] == "partial"  # campaign clock still unresolved
    assert by["P-06"]["status"] == "delivered"


@real
def test_no_descoped_dictionary_id_is_pending_in_conformance():
    d = pd.read_csv(PKG / "p08_data_dictionary.csv")
    descoped = set(d.loc[d["status"].str.startswith("DESCOPED"), "id"])
    assert descoped
    for it in conformance(PKG)["items"]:
        for part in it["parts"]:
            if part["name"] in descoped or any(i in part["evidence"] for i in descoped):
                assert part["status"] != "pending", (it["id"], part["name"])
    q = json.loads((PKG / "OM2" / "p07_quality_report.json").read_text())
    assert q["pending_items"] == []


@real
def test_manifest_hashes_verify_and_manifest_excludes_itself():
    m = json.loads((PKG / "manifest.json").read_text())
    assert m["package_version"] == "v0.2.0"
    assert "manifest.json" not in m["files"]
    on_disk = {p.relative_to(PKG).as_posix() for p in PKG.rglob("*") if p.is_file()} - {"manifest.json"}
    assert set(m["files"]) == on_disk
    for rel, sha in m["files"].items():
        assert hashlib.sha256((PKG / rel).read_bytes()).hexdigest() == sha, rel
    assert m["provenance"]["wind_source"]["sha256"]
    assert m["p10"]["campaign_dates"] and m["p11"]["n_obs"] == m["provenance"]["wind_source"]["n_obs"]


@real
def test_p11_wind_flags_only_campaign_dates_and_matches_source():
    w = pd.read_csv(PKG / "p11_wind_observed.csv", keep_default_na=False)
    m = json.loads((PKG / "manifest.json").read_text())
    dates = set(m["p10"]["campaign_dates"])
    for col in ("used_if_device_clock_utc", "used_if_device_clock_local"):
        assert set(w[col]) - {""} <= dates
    assert len(w) == m["provenance"]["wind_source"]["n_obs"]


@real
def test_sun_envelope_dose_shape_and_values():
    env = pd.read_parquet(PKG / "p10_sun_envelope.parquet")
    pts = pd.read_parquet(PKG / "OM2" / "points.parquet")
    assert set(env["point_id"]) == set(pts["point_id"])
    assert set(env["class"]) <= p10_p11_classes()
    shares = env["sunlit_day_share"].dropna()
    assert shares.between(0, 1).all()
    dose = pd.read_parquet(PKG / "p10_sun_dose.parquet")
    assert (dose[["dose_1h_wh_m2", "dose_2h_wh_m2", "dose_3h_wh_m2"]] >= 0).all().all()
    assert (dose["dose_3h_wh_m2"] + 0.2 >= dose["dose_1h_wh_m2"]).all()


def p10_p11_classes():
    return {"always_sunlit", "always_shaded", "date_dependent", "night"}


@real
def test_sabotage_removing_dose_flips_p10_to_partial(tmp_path):
    dest = tmp_path / "mare_om2_copy" / "v0.2.0"
    shutil.copytree(PKG, dest, ignore=shutil.ignore_patterns("p05_building_shade.csv", "p10_sun_dose.csv"))
    (dest / "p10_sun_dose.parquet").unlink()
    by = {it["id"]: it for it in conformance(dest)["items"]}
    assert by["P-10"]["status"] == "partial"


@real
def test_v013_directory_untouched():
    m = json.loads((ROOT / "v0.1.3" / "manifest.json").read_text())
    for rel, sha in m["files"].items():
        assert hashlib.sha256((ROOT / "v0.1.3" / rel).read_bytes()).hexdigest() == sha, rel


@real
def test_shipped_aggregate_default_segment_is_10m(tmp_path):
    out = tmp_path / "seg.parquet"
    r = subprocess.run([sys.executable, str(PKG / "OM2" / "aggregate_to_segments.py"),
                        "--points", str(PKG / "OM2" / "points.parquet"), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    seg = pd.read_parquet(out)
    assert (seg["segment_end_m"] - seg["segment_start_m"]).max() < 10
    r = subprocess.run([sys.executable, str(PKG / "OM2" / "aggregate_to_segments.py"),
                        "--points", str(PKG / "OM2" / "points.parquet"), "--segment-m", "25", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (pd.read_parquet(out)["segment_end_m"] - pd.read_parquet(out)["segment_start_m"]).max() < 25


@real
def test_readme_documents_segment_note_and_no_bare_project_name():
    import re

    readme = (PKG / "README.md").read_text(encoding="utf-8")
    assert "## Using the data" in readme and "time constant" in readme
    for name in ("README.md", "CHANGELOG.md", "report.md"):
        text = (PKG / name).read_text(encoding="utf-8")
        assert not re.search(r"(?<!Brisa\+ \()MorphoFavela", text), name
    hits = (PKG / "p00_disclosure_hits.txt").read_text(encoding="utf-8")
    assert "OM2/aggregate_to_segments.py" in hits.splitlines()[4]


def test_dose_long_and_drop_dark_rows():
    camp = pd.DataFrame({"point_id": ["a", "a"], "date": ["2026-01-01"] * 2, "local_slot": ["00:00", "12:00"],
                         "dose_1h_wh_m2": [0.0, 5.0], "dose_2h_wh_m2": [0.0, 6.0], "dose_3h_wh_m2": [0.0, 7.0]})
    env = pd.DataFrame({"point_id": ["a", "a"], "local_slot": ["00:00", "12:00"],
                        **{f"dose_{h}h_{k}_wh_m2": [0.0, 1.0] for h in (1, 2, 3) for k in ("min", "median", "max")}})
    long = p10_p11.dose_long({"campaign": camp, "envelope": env})
    assert set(long["scope"]) == {"2026-01-01", "envelope_min", "envelope_median", "envelope_max"}
    envelope = pd.DataFrame({"class": ["night", "date_dependent"], "local_slot": ["00:00", "12:00"]})
    kept = p10_p11.drop_dark_zero_rows(long, envelope)
    assert set(kept["local_slot"]) == {"12:00"}


def test_wind_observed_table_flags_matches_under_each_clock():
    obs = pd.DataFrame({
        "valid_utc": pd.to_datetime(["2026-01-06 12:00", "2026-01-06 15:00"], utc=True),
        "drct": [90.0, 100.0], "speed_ms": [3.0, 4.0], "calm": [False, False], "variable": [False, False],
    })
    win = pd.DataFrame({"date": ["2026-01-06"], "first_timestamp": [pd.Timestamp("2026-01-06 12:00")],
                        "last_timestamp": [pd.Timestamp("2026-01-06 12:30")]})
    t = p10_p11.wind_observed_table(obs, win)
    assert list(t["used_if_device_clock_utc"]) == ["2026-01-06", ""]
    assert list(t["used_if_device_clock_local"]) == ["", "2026-01-06"]  # local 12:00 = 15:00 UTC
    assert list(t.columns) == p10_p11.WIND_COLUMNS
