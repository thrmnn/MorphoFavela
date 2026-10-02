from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from src.om_package import walks as w

X0, Y0, LEN = 680000.0, 7460000.0, 1000.0
_INV = Transformer.from_crs("EPSG:31983", "EPSG:4326", always_xy=True)
MAIN = Path("/home/theo/SCL/SCR/MorphoFavela")
REAL_ROUTE = MAIN / "data/maré/octopus/routes/OM_2_inferred_route.json"
REAL_MATCHED = MAIN / "data/maré/octopus/prerelease_v020/matched"


def _csv(path, rows):
    """rows: (utc string, distance m, edge (u, v), status)."""
    recs = []
    for ts, d, (u, v), status in rows:
        lon, lat = _INV.transform(X0 + d, Y0)
        recs.append({
            "timestamp_utc": ts, "matched_lon": lon, "matched_lat": lat,
            "matched_edge_u": u, "matched_edge_v": v, "match_status": status,
        })
    pd.DataFrame(recs).to_csv(path, index=False)


@pytest.fixture
def synth(monkeypatch):
    line = LineString([(X0, Y0), (X0 + LEN, Y0)])
    monkeypatch.setattr(w, "route_line_utm", lambda p: ("OM_2", line))
    edges = [SimpleNamespace(u=1, v=2), SimpleNamespace(u=2, v=3)]
    monkeypatch.setattr(w, "load_route", lambda p: ("OM_2", edges))
    return line


def _fixes(times, dists, wid="W"):
    return pd.DataFrame({
        "walk_id": wid,
        "t_utc": pd.to_datetime(times, utc=True),
        "distance_along_m": dists,
    })


def _pts(dists):
    return pd.DataFrame({"point_id": [f"p{i}" for i in range(len(dists))], "distance_along_m": dists})


def test_monotone_progress_and_off_route_dropped(synth, tmp_path):
    _csv(tmp_path / "OM_2_20260319_morning_20durmin.csv", [
        ("2026-03-19 11:00:00+00:00", 0, (1, 2), "matched"),
        ("2026-03-19 11:00:10+00:00", 100, (1, 2), "matched"),
        ("2026-03-19 11:00:20+00:00", 90, (2, 1), "matched"),
        ("2026-03-19 11:00:30+00:00", 500, (9, 9), "matched"),
        ("2026-03-19 11:00:40+00:00", 300, (2, 3), "interpolated"),
        ("2026-03-19 11:20:00+00:00", 1000, (2, 3), "matched"),
    ])
    walks, fixes = w.load_walks(tmp_path, Path("unused"))
    assert len(fixes) == 5 and fixes["edge_on_route"].all()
    assert (np.diff(fixes["distance_along_m"]) >= 0).all()
    assert fixes["distance_along_m"].iloc[2] == pytest.approx(100, abs=0.5)
    r = walks.iloc[0]
    assert (r.n_rows, r.n_rows_on_route) == (6, 5)
    assert r.share_on_route == pytest.approx(5 / 6)
    assert r.share_interpolated == pytest.approx(1 / 5)
    assert r.coverage_share == pytest.approx(1.0, abs=1e-3) and not r.partial
    assert r.max_gap_s == pytest.approx(1160)


def test_utc_to_local_and_period(synth, tmp_path):
    _csv(tmp_path / "OM_2_20260319_evening_10durmin.csv", [
        ("2026-03-20 01:30:00+00:00", 0, (1, 2), "matched"),
        ("2026-03-20 01:40:00+00:00", 600, (1, 2), "matched"),
    ])
    r = w.load_walks(tmp_path, Path("unused"))[0].iloc[0]
    assert r.start_local.hour == 22 and str(r.start_local.tz) == "America/Sao_Paulo"
    assert str(r.date) == "2026-03-19"
    assert r.period == "evening" and r.partial
    assert r.mid_utc == pd.Timestamp("2026-03-20 01:35:00", tz="UTC")


def test_walk_id_unique(synth, tmp_path):
    for dur in (10, 20):
        _csv(tmp_path / f"OM_2_20260319_morning_{dur}durmin.csv", [
            ("2026-03-19 11:00:00+00:00", 0, (1, 2), "matched"),
            ("2026-03-19 11:05:00+00:00", 900, (1, 2), "matched"),
        ])
    _csv(tmp_path / "OM_2_20260320_morning_10durmin.csv", [
        ("2026-03-20 11:00:00+00:00", 0, (1, 2), "matched"),
        ("2026-03-20 11:05:00+00:00", 900, (1, 2), "matched"),
    ])
    walks, _ = w.load_walks(tmp_path, Path("unused"))
    assert walks["walk_id"].is_unique
    assert set(walks["walk_id"]) == {
        "OM2_20260319_morning_10durmin", "OM2_20260319_morning_20durmin", "OM2_20260320_morning"}


def test_arrival_interpolates_and_flags_gap():
    f = _fixes(["2026-01-01 10:00:00", "2026-01-01 10:00:10", "2026-01-01 10:05:10"], [0, 100, 400])
    a = w.arrival_times(_pts([0, 50, 100, 250, 400]), f)
    t0 = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
    assert a["t_arrival_utc"].iloc[1] == t0 + pd.Timedelta(seconds=5)
    assert a["t_arrival_utc"].iloc[3] == t0 + pd.Timedelta(seconds=160)
    assert list(a["arrival_source"]) == ["gps", "gps", "gps", "gap_interpolated", "gps"]


def test_outside_walk_is_nat():
    f = _fixes(["2026-01-01 10:00:00", "2026-01-01 10:01:00"], [100, 200])
    a = w.arrival_times(_pts([50, 150, 250]), f)
    assert list(a["arrival_source"]) == ["outside_walk", "gps", "outside_walk"]
    assert a["t_arrival_utc"].isna().tolist() == [True, False, True]


def test_stall_takes_first_arrival():
    f = _fixes(
        ["2026-01-01 10:00:00", "2026-01-01 10:00:10", "2026-01-01 10:00:20",
         "2026-01-01 10:00:30", "2026-01-01 10:00:40"],
        [0, 100, 100, 100, 200],
    )
    a = w.arrival_times(_pts([100, 150]), f)
    t0 = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
    assert a["t_arrival_utc"].iloc[0] == t0 + pd.Timedelta(seconds=10)
    assert a["t_arrival_utc"].iloc[1] == t0 + pd.Timedelta(seconds=35)
    assert (a["arrival_source"] == "gps").all()


def test_walk_entirely_off_route_is_kept(synth, tmp_path):
    _csv(tmp_path / "OM_2_20260409_evening_5durmin.csv", [
        ("2026-04-09 22:00:00+00:00", 10, (8, 9), "matched"),
        ("2026-04-09 22:05:00+00:00", 20, (8, 9), "matched"),
    ])
    walks, fixes = w.load_walks(tmp_path, Path("unused"))
    assert len(walks) == 1 and fixes.empty
    assert walks.iloc[0].coverage_share == 0 and walks.iloc[0].partial
    out = w.all_arrivals(_pts([0, 1]), fixes, walks)
    assert (out["arrival_source"] == "outside_walk").all() and out["t_arrival_utc"].isna().all()


def test_all_arrivals_stacks():
    fx = pd.concat([_fixes(["2026-01-01 10:00:00", "2026-01-01 10:01:00"], [0, 100], wid) for wid in ("A", "B")])
    out = w.all_arrivals(_pts([0, 50, 100]), fx, pd.DataFrame({"walk_id": ["A", "B"]}))
    assert len(out) == 6 and set(out["walk_id"]) == {"A", "B"}


@pytest.mark.skipif(not (REAL_ROUTE.exists() and REAL_MATCHED.exists()), reason="real OM2 data not present")
def test_real_data_integration():
    from src.om_package.routes import densify_route

    walks, fixes = w.load_walks(REAL_MATCHED, REAL_ROUTE)
    assert len(walks) == 63 and walks["walk_id"].is_unique
    pts = densify_route(REAL_ROUTE)[["point_id", "distance_along_m"]]
    arr = w.all_arrivals(pts, fixes, walks)
    for _, g in arr.groupby("walk_id"):
        assert (g["t_arrival_utc"].dropna().diff().dropna() >= pd.Timedelta(0)).all()
