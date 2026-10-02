"""v0.3.0 integration: walks, wind regimes, local-time shade, P-10 and P-11
files, columns, dictionary coverage, conformance and shipped hashes. Needs the
built package (outputs/_packages/mare_om2/<VERSION>/)."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from src.om_package import p10_p11
from src.om_package.io_utils import DEFAULT_ROOT
from src.om_package.package_docs import VERSION
from src.om_package.spec import conformance, internal_dir_for

ROOT = DEFAULT_ROOT / "outputs" / "_packages" / "mare_om2"
PKG = ROOT / VERSION
INTERNAL = internal_dir_for(PKG)
real = pytest.mark.skipif(not PKG.is_dir(), reason=f"mare_om2 {VERSION} package not built at the default root")

NEW_FILES = [
    "p02b_walks.parquet", "p02b_walks.csv", "p05_building_shade.parquet",
    "p10_sun_envelope.parquet", "p10_sun_envelope.csv", "p10_sun_dose.parquet",
    "p10_horizon_profiles.parquet", "p11_wind_regimes.csv", "p11_regime_by_hour.csv",
    "p12_walk_points.parquet", "p12_walk_points.csv",
]
REMOVED_FILES = ["p10_clock_agreement.parquet", "p10_clock_agreement.csv", "p11_wind_observed.csv",
                 "p05b_campaign_windows.parquet", "p05b_campaign_windows.csv", "p05_building_shade.csv"]
INTERNAL_ONLY = ["CHANGELOG.md", "p00_spec_conformance.json", "p00_spec_conformance.csv", "p00_disclosure_hits.txt"]
TAUS = (5, 10, 30, 60)


def _slugs():
    reg = pd.read_csv(PKG / "p11_wind_regimes.csv")
    return reg.loc[reg["period"] == "campaign", "column_slug"].tolist()


@real
@pytest.mark.parametrize("name", NEW_FILES)
def test_new_file_present(name):
    assert (PKG / name).stat().st_size > 0


@real
@pytest.mark.parametrize("name", REMOVED_FILES + INTERNAL_ONLY)
def test_removed_and_internal_files_are_not_shipped(name):
    assert not (PKG / name).exists()


@real
@pytest.mark.parametrize("name", INTERNAL_ONLY)
def test_internal_files_are_written_beside_the_package(name):
    assert (INTERNAL / name).stat().st_size > 0


@real
def test_regime_point_columns_named_by_regime_slug():
    df = pd.read_parquet(PKG / "OM2" / "points.parquet")
    assert len(_slugs()) == 2
    assert not [c for c in df.columns if "prevailing" in c]
    for sl in _slugs():
        for stem in ("frontal_area_density_windward", "canyon_alignment_deg", "upwind_shelter_angle_deg", "z0_macdonald_m"):
            assert df[f"{stem}_{sl}"].notna().any(), (stem, sl)
        assert df[f"canyon_alignment_deg_{sl}"].dropna().between(0, 90).all()
        assert df[f"upwind_shelter_angle_deg_{sl}"].dropna().between(-90, 90).all()
    assert (df["open_space_fraction"] - (1 - df["lambda_p_buffer_50m"])).abs().max() < 1e-9
    assert (df["zd_macdonald_m"].dropna() >= 0).all()
    assert all(c in df.columns for c in ("lambda_f_N", "lambda_f_NW", "annual_sun_hours"))


@real
def test_walks_table_local_time_and_tags():
    w = pd.read_parquet(PKG / "p02b_walks.parquet")
    names = set(pd.read_csv(PKG / "p11_wind_regimes.csv").query("period == 'campaign'")["name"])
    assert w["walk_id"].is_unique and len(w) == 63
    assert w["start_local"].str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}-03:00$").all()
    assert w["start_utc"].str.endswith("Z").all()
    assert set(w["wind_regime"]) <= names | {"calm", "none"}
    assert w.loc[w["wind_regime"].isin(names), "wind_direction_deg"].notna().all()


@real
def test_walk_points_shape_values_and_no_outside_rows():
    p = pd.read_parquet(PKG / "p12_walk_points.parquet")
    walks = pd.read_parquet(PKG / "p02b_walks.parquet")
    assert (p["arrival_source"] != "outside_walk").all() and set(p["walk_id"]) <= set(walks["walk_id"])
    assert not p.duplicated(["walk_id", "point_id"]).any()
    assert p["t_arrival_local"].str.endswith("-03:00").all()
    assert (p["dose_3h_before_wh_m2"] + 0.2 >= p["dose_1h_before_wh_m2"]).all() and (p["dose_1h_before_wh_m2"] >= 0).all()
    for m in ("sky_view_factor", "shaded_at_arrival", "dose_1h_before_wh_m2",
              *(f"canyon_alignment_deg_{sl}" for sl in _slugs())):
        for t in TAUS:
            assert f"{m}_tau{t}s" in p.columns
    sh = p["shaded_at_arrival_tau5s"].dropna()
    assert sh.between(-1e-9, 1 + 1e-9).all()
    pts = pd.read_parquet(PKG / "OM2" / "points.parquet").set_index("point_id")
    assert set(p["point_id"]) <= set(pts.index)


@real
def test_shipped_aggregate_runs_by_walk_with_tau(tmp_path):
    out = tmp_path / "seg.parquet"
    r = subprocess.run([sys.executable, str(PKG / "OM2" / "aggregate_to_segments.py"),
                        "--points", str(PKG / "p12_walk_points.parquet"), "--by", "walk_id",
                        "--segment-m", "20", "--tau", "30", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    seg = pd.read_parquet(out)
    p = pd.read_parquet(PKG / "p12_walk_points.parquet")
    assert int(seg["n_points"].sum()) == len(p) and seg["walk_id"].nunique() == p["walk_id"].nunique()
    assert not [c for c in seg.columns if c.endswith("_tau5s")] and "sky_view_factor_tau30s" in seg.columns


@real
def test_regime_tables():
    reg = pd.read_csv(PKG / "p11_wind_regimes.csv")
    assert set(reg["period"]) == {"campaign", "climatology"}
    assert reg.groupby("period")["share"].sum().round(6).eq(1.0).all()
    hour = pd.read_csv(PKG / "p11_regime_by_hour.csv")
    assert sorted(hour["local_hour"].unique()) == list(range(24))
    assert hour.groupby(["period", "local_hour"])["share"].sum().round(6).eq(1.0).all()


@real
def test_shade_is_local_daylight_only():
    sh = pd.read_parquet(PKG / "p05_building_shade.parquet", columns=["timestamp_local", "timestamp_utc", "sun_altitude_deg", "date"])
    assert (sh["sun_altitude_deg"] > 0).all()
    assert str(sh["timestamp_local"].dt.tz) == "America/Sao_Paulo"
    assert (sh["timestamp_local"].dt.tz_convert("UTC") == sh["timestamp_utc"]).all()
    walk_dates = set(pd.read_parquet(PKG / "p02b_walks.parquet")["date"].astype(str))
    assert set(sh["date"].astype(str).unique()) == walk_dates


@real
def test_every_shipped_table_column_has_a_dictionary_row():
    ids = set(pd.read_csv(PKG / "p08_data_dictionary.csv")["id"])
    tables = {
        "OM2/points": pd.read_parquet(PKG / "OM2" / "points.parquet"),
        "p05_building_shade": pd.read_parquet(PKG / "p05_building_shade.parquet").head(10),
        "p02b_walks": pd.read_parquet(PKG / "p02b_walks.parquet"),
        "p10_sun_envelope": pd.read_parquet(PKG / "p10_sun_envelope.parquet").head(10),
        "p10_sun_dose": pd.read_parquet(PKG / "p10_sun_dose.parquet").head(10),
        "p10_horizon_profiles": pd.read_parquet(PKG / "p10_horizon_profiles.parquet").head(10),
        "p11_wind_regimes": pd.read_csv(PKG / "p11_wind_regimes.csv"),
        "p11_regime_by_hour": pd.read_csv(PKG / "p11_regime_by_hour.csv"),
        "p12_walk_points": pd.read_parquet(PKG / "p12_walk_points.parquet").head(10),
    }
    for name, df in tables.items():
        missing = [c for c in df.columns if c not in ids and c != "geometry"]
        assert not missing, f"{name}: no dictionary row for {missing}"


@real
def test_dictionary_rows_complete_ventilation_proxy_and_retired_rows_kept():
    d = pd.read_csv(PKG / "p08_data_dictionary.csv").set_index("id")
    for col in ("definition", "unit", "source", "method", "limits", "status"):
        assert d[col].notna().all(), col
    for sl in _slugs():
        for stem in ("frontal_area_density_windward", "canyon_alignment_deg", "upwind_shelter_angle_deg", "z0_macdonald_m"):
            text = " ".join(str(d.loc[f"{stem}_{sl}", c]) for c in ("definition", "limits")).upper()
            assert "PROXY" in text, (stem, sl)
    for vid in ("windward_lambda_f_prevailing", "z0_macdonald_m", "timestamp", "agreement_share"):
        assert d.loc[vid, "status"].startswith("RETIRED")


@real
def test_conformance_delivered_for_p10_p11_p12():
    by = {it["id"]: it for it in conformance(PKG)["items"]}
    for pid in ("P-02", "P-03", "P-06", "P-08", "P-09", "P-10", "P-11", "P-12"):
        assert by[pid]["status"] == "delivered", (pid, by[pid]["evidence"])
    assert [p["name"] for p in by["P-10"]["parts"]] == ["sun_envelope", "sun_dose", "annual_sun_hours"]
    assert by["P-05"]["status"] == "delivered (scoped)"
    assert not any(p["pending_on"] for it in by.values() for p in it["parts"])


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
def test_manifest_hashes_verify_and_cover_only_shipped_files():
    m = json.loads((PKG / "manifest.json").read_text())
    assert m["package_version"] == VERSION
    assert "manifest.json" not in m["files"]
    on_disk = {p.relative_to(PKG).as_posix() for p in PKG.rglob("*") if p.is_file()} - {"manifest.json"}
    assert set(m["files"]) == on_disk
    for rel, sha in m["files"].items():
        assert hashlib.sha256((PKG / rel).read_bytes()).hexdigest() == sha, rel
    assert m["provenance"]["wind_source"]["sha256"]
    assert m["p10"]["campaign_dates"] and m["walks"]["n_walks"] == 63
    assert m["p05_shade"]["tz"] == "America/Sao_Paulo"


@real
def test_no_clock_sensitivity_left_in_shipped_text():
    for name in ("README.md", "p08_data_dictionary.csv"):
        text = (PKG / name).read_text(encoding="utf-8")
        for needle in ("OCTOPUS_TZ", "UNRESOLVED", "device clock UNKNOWN", "clock_agreement"):
            hits = [ln for ln in text.splitlines() if needle in ln and "RETIRED" not in ln and "no longer" not in ln]
            assert not hits, (name, needle, hits[:1])


@real
def test_sensible_sizes_and_route_count():
    pts = pd.read_parquet(PKG / "OM2" / "points.parquet")
    q = json.loads((PKG / "OM2" / "p07_quality_report.json").read_text())
    assert q["n_points"] == len(pts) and q["route_geometry_flagged_points"] == int(pts["route_geometry_flag"].sum())


@real
def test_older_package_directories_untouched():
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


@real
def test_readme_documents_segment_note_and_no_bare_project_name():
    readme = (PKG / "README.md").read_text(encoding="utf-8")
    assert "## Using the data" in readme and "time constant" in readme
    for path in (PKG / "README.md", INTERNAL / "CHANGELOG.md", PKG / "report.md"):
        assert not re.search(r"(?<!Brisa\+ \()MorphoFavela", path.read_text(encoding="utf-8")), path.name


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
