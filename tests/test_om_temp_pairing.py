import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.om_package import temp_pairing as tp

FACTS_EXAMPLE = Path(__file__).parent / "data" / "om_temp_facts_example.json"


def _points(n=100, classes=None):
    p = pd.DataFrame({"point_id": [f"P{i:03d}" for i in range(n)], "distance_along_m": np.arange(n, dtype=float),
                      "route_geometry_flag": False})
    if classes is not None:
        p["point_class"] = classes
    return p


def test_readings_snap_and_drop_by_class():
    cls = ["street"] * 40 + ["beco"] * 20 + ["projected"] * 40
    p = _points(classes=cls)
    t = pd.date_range("2026-01-05 12:00", periods=5, freq="5s", tz="UTC")
    fixes = pd.DataFrame({"walk_id": "OM2_20260105_morning", "t_utc": t, "distance_along_m": [10.4, 10.6, 45.0, 70.2, 99.0],
                          "match_status": ["matched", "matched", "matched", "matched", "interpolated"],
                          "edge_on_route": True})
    temps = pd.DataFrame({"walk_id": "OM2_20260105_morning", "t_utc": t, "temperature": [30.0, 30.1, 30.2, 30.3, 30.4]})
    r, f = tp.readings_table(p, fixes, temps)
    assert list(r["point_id"]) == ["P010", "P011", "P070"]
    assert f["n_matched_with_temperature"] == 4 and f["n_dropped_off_street"] == 1 and f["keep_basis"] == "point_class"
    assert str(r["t_local"].dt.tz) == tp.LOCAL_TZ and r["t_local"].iloc[0].hour == 9


def test_keep_mask_falls_back_to_flag():
    p = _points(5)
    p.loc[2, "route_geometry_flag"] = True
    keep, basis = tp.keep_mask(p)
    assert basis == "route_geometry_flag" and keep.tolist() == [True, True, False, True, True]


def test_settling_fit_and_window():
    minutes = pd.Series(np.arange(15))
    curve = pd.Series(-0.8 * np.exp(-(minutes + 0.5) / 4.0), index=minutes)
    amp, tau, win = tp.fit_settling(curve, pd.Series(100, index=minutes))
    assert abs(amp + 0.8) < 1e-3 and abs(tau - 4.0) < 1e-2
    assert win == int(np.ceil(4.0 * np.log(8.0)))
    assert tp.settle_window(0.05, 3.0) == 0
    assert tp.settle_window(-5.0, 30.0) == tp.WARMUP_MAX_MIN


def _walk_rows(walk, period, n=200, dt=5.0, background=30.0, slope=0.0, offset=0.0, covered=True, warm=None):
    t = pd.Timestamp("2026-01-05 12:00", tz="UTC") + pd.to_timedelta(np.arange(n) * dt, unit="s")
    secs = np.arange(n) * dt
    bg = background + slope * secs / 3600
    temp = bg + offset + (0 if warm is None else warm(secs))
    return pd.DataFrame({"walk_id": walk, "period": period, "t_utc": t, "distance_along_m": secs * 1.1,
                         "temperature": temp, "background": bg if covered else np.nan, "logger_covered": covered,
                         "minutes_since_start": secs / 60, "warmup": secs < 60})


def test_anomalies_remove_background_offset_and_ignore_warmup_rows():
    a = _walk_rows("w1", "morning", slope=2.0, offset=1.5, warm=lambda s: np.where(s < 60, -2.0, 0.0))
    b = _walk_rows("w2", "morning", slope=2.0, offset=-0.7, covered=False)
    r = tp.add_anomalies(pd.concat([a, b], ignore_index=True))
    kept = r[~r["warmup"]]
    assert np.allclose(kept.loc[kept["walk_id"] == "w1", "anomaly"], 0.0, atol=1e-9)
    assert np.allclose(r.loc[r["warmup"] & (r["walk_id"] == "w1"), "anomaly"], -2.0, atol=1e-9)
    assert (r.loc[r["walk_id"] == "w2", "anomaly_source"] == "walk_detrend").all()
    assert np.allclose(kept.loc[kept["walk_id"] == "w2", "anomaly"], 0.0, atol=1e-9)


def _p12_walk(walk, shade, dist_step=1.0, speed=1.0, source="gps"):
    n = len(shade)
    d = np.arange(n) * dist_step
    t = pd.Timestamp("2026-01-05 18:00", tz="UTC") + pd.to_timedelta(d / speed, unit="s")
    return pd.DataFrame({"walk_id": walk, "point_id": [f"P{i:03d}" for i in range(n)], "distance_along_m": d,
                         "t_arrival_utc": t, "arrival_source": source, "shaded_at_arrival": shade})


def test_find_events_needs_long_stable_runs():
    shade = [False] * 40 + [True] * 40 + [False] * 10 + [True] * 40
    ev = tp.find_events(_p12_walk("OM2_20260105_evening", shade))
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["direction"] == "sun_to_shade" and e["distance_m"] == 40.0 and e["period"] == "evening"
    assert e["before_s"] == 40.0 and e["after_s"] == 39.0
    fast = tp.find_events(_p12_walk("OM2_20260105_evening", [False] * 40 + [True] * 40, speed=10.0))
    assert fast.empty


def test_fit_approach_recovers_tau():
    t = np.arange(-27.5, 60, 5.0)
    curve = pd.Series(np.where(t < 0, 0, -0.3 * (1 - np.exp(-np.clip(t, 0, None) / 12.0))), index=t)
    amp, tau = tp.fit_approach(curve, pd.Series(50, index=t))
    assert abs(amp + 0.3) < 1e-3 and abs(tau - 12.0) < 0.05


def _assoc_frame(seed=0, n_walks=12, n=150, beta=-0.5):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_walks):
        shade = rng.uniform(0, 1, n)
        svf = rng.uniform(0.2, 0.8, n)
        rows.append(pd.DataFrame({"walk_id": f"w{w}", "shade_tau30s": shade, "svf_tau30s": svf,
                                  "dose_tau30s": rng.uniform(0, 600, n),
                                  "anomaly": w * 0.3 + beta * shade + 1.0 * svf + rng.normal(0, 0.05, n)}))
    return pd.concat(rows, ignore_index=True)


def test_association_model_effects_in_units():
    c, f = tp.association_model(_assoc_frame(), 30, ["shade", "svf"])
    sh = c.set_index("term").loc["shade"]
    sv = c.set_index("term").loc["svf"]
    assert sh["lo"] < -0.5 < sh["hi"] and abs(sh["effect_c"] + 0.5) < 0.02
    assert abs(sv["effect_c"] - 0.1) < 0.01  # per 0.1 of sky view factor
    assert f["cv_r2"] > 0.9 and f["n_walks"] == 12


def test_moment_r2_and_cv_match_direct_fit():
    df = _assoc_frame(seed=1)
    cols = ["shade_tau30s", "svf_tau30s"]
    keys, XX, Xy, yy = tp._walk_moments(df, cols)
    dm = tp._demean(df, ["anomaly", *cols])
    X, y = dm[cols].to_numpy(), dm["anomaly"].to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    assert abs(tp._r2_from(XX, Xy, yy) - (1 - ((y - X @ b) ** 2).sum() / (y @ y))) < 1e-10
    _, f = tp.association_model(df, 30, ["shade", "svf"])
    assert abs(tp._cv_r2_from(XX, Xy, yy) - f["cv_r2"]) < 1e-10


def test_tau_scan_finds_generating_tau():
    rng = np.random.default_rng(3)
    taus = (0, 10, 30, 90)
    rows = []
    for w in range(10):
        base = {t: rng.uniform(0, 1, 120) for t in taus}
        d = {"walk_id": f"w{w}", "period": "morning"}
        for t in taus:
            d.update({f"shade_tau{t}s": base[t], f"dose_tau{t}s": rng.uniform(0, 1, 120),
                      f"svf_tau{t}s": rng.uniform(0, 1, 120)})
        d["anomaly"] = -0.4 * base[30] + rng.normal(0, 0.05, 120)
        rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    df = pd.concat([df, df.assign(period="evening")], ignore_index=True)
    scan, f = tp.tau_scan(df, n_boot=20, rng=np.random.default_rng(0), taus=taus)
    assert f["morning"]["best_tau_s"] == 30
    assert scan.loc[(scan["period"] == "morning") & (scan["tau_s"] == 30), "cv_r2"].item() > 0.75


@pytest.fixture
def facts():
    return json.loads(FACTS_EXAMPLE.read_text(encoding="utf-8"))


def test_report_text_voice(facts):
    text = "\n".join(tp.report_paragraphs(facts)) + tp.readme_subsection(facts) + tp.team_question(facts)
    assert "—" not in text and "–" not in text
    for bad in ("SBGL", "METAR", "H/W", "λf", "significant", "robust", "novel", "PLACEHOLDER"):
        assert bad not in text
    assert "Jingxue" in text
    assert f"{facts['warmup']['evening']['window_min']} minutes" in text


def test_report_numbers_come_from_facts(facts):
    text = "\n".join(tp.report_paragraphs(facts))
    m = facts["models"]["evening_with_ratio"]["effects"]["hw"]
    assert f"{m['effect_c']:.2f}" in text.replace("−", "-")
    bumped = json.loads(json.dumps(facts))
    bumped["models"]["evening_with_ratio"]["effects"]["hw"]["effect_c"] = 9.87
    assert "9.87" in "\n".join(tp.report_paragraphs(bumped)).replace("−", "-")
