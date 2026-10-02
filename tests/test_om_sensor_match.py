import numpy as np
import pandas as pd
import pytest

from src.om_package.sensor_match import DEFAULT_TAUS_S, sensor_matched, tau_from_t90

T0 = pd.Timestamp("2026-01-15 15:00", tz="UTC")


def _arr(secs, wid="w1"):
    return pd.DataFrame(
        {
            "point_id": [f"p{i}" for i in range(len(secs))],
            "walk_id": wid,
            "t_arrival_utc": [pd.NaT if np.isnan(s) else T0 + pd.Timedelta(seconds=float(s)) for s in secs],
        }
    )


def _vals(x, name="X"):
    return pd.DataFrame({"point_id": [f"p{i}" for i in range(len(x))], name: x})


def _brute(t, x, tau):
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        if np.isnan(t[i]):
            continue
        num = den = 0.0
        for j in range(len(t)):
            if np.isnan(t[j]) or np.isnan(x[j]) or t[j] > t[i]:
                continue
            dt = t[i] - t[j]
            if dt <= 5 * tau:
                w = np.exp(-dt / tau)
                num += w * x[j]
                den += w
        if den > 0:
            out[i] = num / den
    return out


def test_t90_conversion():
    assert tau_from_t90(10 * np.log(10)) == pytest.approx(10.0, rel=1e-12)
    assert tau_from_t90(23.0) == pytest.approx(23.0 / 2.302585, rel=1e-6)


def test_constant_returns_constant():
    s = np.arange(0, 600.0)
    out = sensor_matched(_arr(s), _vals(np.full(600, 27.3)), ["X"])
    for tau in DEFAULT_TAUS_S:
        assert np.allclose(out[f"X_tau{tau}s"], 27.3, atol=1e-12)


def test_step_response_at_tau_and_cutoff_beyond_5tau():
    tau, h = 30.0, 0.1
    s = np.arange(0, 6000.0) * h
    x = np.where(s >= 300, 1.0, 0.0)
    out = sensor_matched(_arr(s), _vals(x), ["X"], taus=[tau])["X_tau30s"].to_numpy()
    # mass of the new air in the (renormalised, truncated) window; continuous limit of
    # (1 - e^-1) / (1 - e^-5) at dt = tau after the step
    expected = (1 - np.exp(-1)) / (1 - np.exp(-5))
    assert out[int(round((300 + tau) / h))] == pytest.approx(expected, abs=5e-3)
    assert out[int(round(300 / h)) - 1] == 0.0
    assert out[int(round((300 + 5 * tau) / h)) + 2] == pytest.approx(1.0, abs=1e-12)


def test_nan_skipped_and_renormalised():
    s = np.arange(0, 20.0)
    x = np.arange(20.0)
    x[5] = np.nan
    out = sensor_matched(_arr(s), _vals(x), ["X"], taus=[10])["X_tau10s"].to_numpy()
    assert np.isfinite(out[5])
    assert np.allclose(out, _brute(s, x, 10), atol=1e-9, equal_nan=True)
    allnan = sensor_matched(_arr(s), _vals(np.full(20, np.nan)), ["X"], taus=[10])
    assert allnan["X_tau10s"].isna().all()


def test_nan_value_with_nothing_in_window_is_nan():
    s = np.array([0.0, 100.0])
    out = sensor_matched(_arr(s), _vals([1.0, np.nan]), ["X"], taus=[5])["X_tau5s"].to_numpy()
    assert out[0] == 1.0 and np.isnan(out[1])


def test_missing_arrival_gives_nan_and_is_ignored():
    s = np.array([0.0, 1.0, np.nan, 2.0])
    x = np.array([1.0, 2.0, 99.0, 3.0])
    out = sensor_matched(_arr(s), _vals(x), ["X"], taus=[10])["X_tau10s"].to_numpy()
    assert np.isnan(out[2])
    assert np.allclose(out[[0, 1, 3]], _brute(s, x, 10)[[0, 1, 3]])


@pytest.mark.parametrize("tau", [5, 10, 30, 60])
def test_matches_bruteforce_irregular_ties_unsorted(tau):
    rng = np.random.default_rng(1)
    n = 400
    dt = rng.exponential(2.0, n)
    dt[rng.random(n) < 0.15] = 0.0
    dt[rng.random(n) < 0.03] = 400.0
    t = np.cumsum(dt)
    x = rng.normal(size=n) * 5 + 30
    x[rng.random(n) < 0.1] = np.nan
    perm = rng.permutation(n)
    out = sensor_matched(_arr(t[perm]), _vals(x[perm]), ["X"], taus=[tau])[f"X_tau{tau}s"].to_numpy()
    assert np.allclose(out, _brute(t[perm], x[perm], tau), atol=1e-9, equal_nan=True)


def test_multiple_columns_and_labels():
    s = np.arange(0, 50.0)
    v = _vals(np.arange(50.0), "A")
    v["B"] = 2.0
    out = sensor_matched(_arr(s), v, ["A", "B"], taus=[2.5, 10])
    assert {"A_tau2.5s", "B_tau2.5s", "A_tau10s", "B_tau10s", "point_id", "walk_id"} <= set(out.columns)
    assert np.allclose(out["B_tau10s"], 2.0)
