import numpy as np
import pandas as pd
import pytest

from src.om_package import sun_envelope as se

LAT, LON = -22.86, -43.24
AZ = np.arange(0, 360, 15.0)


def _h(open_deg=0.0, canyon_deg=80.0):
    rows = []
    for pid, v in (("open", open_deg), ("canyon", canyon_deg)):
        rows += [(pid, a, v) for a in AZ]
    return pd.DataFrame(rows, columns=se.HORIZON_COLUMNS)


@pytest.fixture(scope="module")
def env():
    return se.sun_envelope(_h(), lat=LAT, lon=LON, window_start="2026-01-01", window_end="2026-01-31")


def test_open_point_never_shaded_in_daylight(env):
    t, _ = env
    o = t[(t.point_id == "open") & (t["class"] != "night")]
    assert len(o) and (o["class"] == "always_sunlit").all()


def test_canyon_shaded_except_near_zenith(env):
    t, _ = env
    c = t[(t.point_id == "canyon") & (t["class"] != "night")]
    assert (c["class"] == "always_shaded").mean() > 0.8
    lit = c[c["sunlit_day_share"] > 0]
    assert lit["local_slot"].between("11:00", "14:00").all()
    assert (c["sunlit_day_share"] > 0).any()


def test_classes_partition_and_summary(env):
    t, summ = env
    assert set(t["class"]) <= set(se.CLASSES)
    assert len(t) == 2 * 288
    assert t["n_days_sun_up"].between(0, summ["n_days"]).all()
    assert sum(summ["class_share_of_daylight"].values()) == pytest.approx(1.0)
    night = t[t["class"] == "night"]
    assert (night["n_days_sun_up"] == 0).all() and night["sunlit_day_share"].isna().all()


def test_dose_zero_at_night_and_monotone_in_window():
    d = se.direct_sun_dose(_h(), ["2026-03-19"], lat=LAT, lon=LON, window_start="2026-03-01", window_end="2026-03-31")
    c = d["campaign"]
    o = c[c.point_id == "open"].set_index("local_slot")
    assert o.loc["00:00", "dose_3h_wh_m2"] == 0 and o.loc["23:55", "dose_1h_wh_m2"] == 0
    assert o["dose_1h_wh_m2"].max() > 0
    assert (o["dose_1h_wh_m2"] <= o["dose_2h_wh_m2"] + 1e-9).all()
    assert (o["dose_2h_wh_m2"] <= o["dose_3h_wh_m2"] + 1e-9).all()
    cn = c[c.point_id == "canyon"]
    assert cn["dose_3h_wh_m2"].max() < o["dose_3h_wh_m2"].max()
    e = d["envelope"]
    assert (e["dose_1h_min_wh_m2"] <= e["dose_1h_median_wh_m2"]).all()
    assert (e["dose_1h_median_wh_m2"] <= e["dose_1h_max_wh_m2"]).all()


def test_annual_sun_hours():
    a = se.annual_sun_hours(_h(), lat=LAT, lon=LON, step_min=30).set_index("point_id")["annual_sun_hours"]
    assert 3000 < a["open"] < 5000 and a["canyon"] < 0.1 * a["open"]


def test_clock_readings_differ_by_exactly_offset():
    r = se.clock_readings(["2026-03-19 12:00:00", "2025-12-05 02:30:00"])
    assert ((r["local_B"] - r["local_A"]) == pd.Timedelta(hours=3)).all()
    assert r.loc[0, "slot_A"] == "09:00" and r.loc[0, "slot_B"] == "12:00"


def test_exact_date_agreement_bounds():
    out = se.exact_date_agreement(_h(), ["2026-03-19", "2026-03-30"], lat=LAT, lon=LON)
    assert list(out["date"]) == ["2026-03-19", "2026-03-30", "all"]
    assert out["agreement_share"].between(0, 1).all()
    assert out.loc[out.date == "all", "agreement_share"].iloc[0] < 1.0


def test_tidy_collapses_repeated_azimuths():
    az = np.array([0.0, 0.0, 90.0])
    t = se._tidy(np.array(["a"]), np.array([[1.0, 3.0, 2.0]]), az)
    assert t["azimuth_deg"].tolist() == [0.0, 90.0] and t["horizon_deg"].tolist() == [3.0, 2.0]
