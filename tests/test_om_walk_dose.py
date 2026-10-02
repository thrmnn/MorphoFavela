import numpy as np
import pandas as pd
import pytest

from src.om_package import sun_envelope as se
from src.om_package.walk_dose import WALK_DOSE_HOURS, walk_dose

LAT, LON = -22.86, -43.24
AZ = np.arange(0, 360, 15.0)
TZ = "America/Sao_Paulo"


def _h(ids, deg):
    return pd.DataFrame([(p, a, deg) for p in ids for a in AZ], columns=se.HORIZON_COLUMNS)


def _arr(ids, local_times, wid="w1"):
    t = [pd.Timestamp(x, tz=TZ).tz_convert("UTC") if x else pd.NaT for x in local_times]
    return pd.DataFrame({"point_id": ids, "walk_id": wid, "t_arrival_utc": t})


NOON = "2026-01-15 12:00"


def test_constants():
    assert WALK_DOSE_HOURS == (1, 3)


def test_blocked_horizon_gives_zero():
    out = walk_dose(_h(["a"], 90.0), _arr(["a"], [NOON]), lat=LAT, lon=LON)
    assert out["dose_1h_before_wh_m2"].iloc[0] == 0.0
    assert out["dose_3h_before_wh_m2"].iloc[0] == 0.0


def test_open_noon_positive_and_3h_exceeds_1h():
    out = walk_dose(_h(["a"], 0.0), _arr(["a"], [NOON]), lat=LAT, lon=LON)
    d1, d3 = out["dose_1h_before_wh_m2"].iloc[0], out["dose_3h_before_wh_m2"].iloc[0]
    assert d1 > 400 and d3 > d1
    assert list(out.columns) == ["point_id", "walk_id", "dose_1h_before_wh_m2", "dose_3h_before_wh_m2"]


def test_later_morning_arrival_has_larger_1h_dose():
    out = walk_dose(_h(["a", "b"], 0.0), _arr(["a", "b"], ["2026-01-15 07:00", "2026-01-15 10:00"]), lat=LAT, lon=LON)
    assert out["dose_1h_before_wh_m2"].iloc[1] > out["dose_1h_before_wh_m2"].iloc[0]


def test_nat_and_unknown_points_are_nan_and_walks_are_separate():
    arr = pd.concat([_arr(["a", "b", "zz"], [NOON, None, NOON], "w1"), _arr(["a"], ["2026-01-15 10:00"], "w2")], ignore_index=True)
    out = walk_dose(_h(["a", "b"], 0.0), arr, lat=LAT, lon=LON)
    assert out["dose_1h_before_wh_m2"].iloc[[1, 2]].isna().all()
    assert out["dose_1h_before_wh_m2"].iloc[0] > out["dose_1h_before_wh_m2"].iloc[3] > 0


def test_agrees_with_direct_sun_dose_on_slot_boundary():
    hz = _h(["a"], 0.0)
    ref = se.direct_sun_dose(
        hz, ["2026-01-15"], lat=LAT, lon=LON, hours=(1, 3), window_start="2026-01-15", window_end="2026-01-15"
    )["campaign"]
    ref = ref[ref["local_slot"] == "12:00"].iloc[0]
    out = walk_dose(hz, _arr(["a"], [NOON]), lat=LAT, lon=LON).iloc[0]
    assert out["dose_1h_before_wh_m2"] == pytest.approx(ref["dose_1h_wh_m2"], rel=0.02)
    assert out["dose_3h_before_wh_m2"] == pytest.approx(ref["dose_3h_wh_m2"], rel=0.02)
