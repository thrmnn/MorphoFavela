import numpy as np
import pandas as pd

from src.om_package import fixed_loggers as fl


def _mins(rows):
    return pd.DataFrame(rows, columns=["t_utc", "device", "kind", "temperature", "humidity"]).assign(
        t_utc=lambda d: pd.to_datetime(d["t_utc"], utc=True)
    )


def _csv(path, rows):
    pd.DataFrame(rows, columns=["Timestamp", "Latitude", "Longitude", "Temperature", "Humidity"]).to_csv(path, index=False)


def test_loader_drops_junk_and_locates(tmp_path):
    _csv(tmp_path / "O_3_20260101_01durhrs.csv", [
        ["2000-01-01 00:00:00", 0, 0, 30.0, 50],
        ["2026-01-01 10:00:05", -22.855, -43.24, 30.0, 50],
        ["2026-01-01 10:00:35", -22.855, -43.24, 32.0, 50],
        ["2026-01-1 10:01:05", -22.855, -43.24, 99.0, 50],
        ["2026-01-1 10:02:05", 0.0, 0.0, 31.0, 50],
    ])
    m, loc = fl.load_loggers(tmp_path, tz="UTC")
    assert list(m["device"].unique()) == ["O_3"] and (m["kind"] == "outdoor").all()
    assert m["t_utc"].min() >= fl.MIN_VALID_TIME
    assert m.loc[m["t_utc"] == pd.Timestamp("2026-01-01 10:00", tz="UTC"), "temperature"].item() == 31.0
    assert len(m) == 2 and loc.attrs["n_dropped"] == 2
    assert loc["lat"].item() == -22.855


def test_offset_removed_so_dropout_does_not_step():
    t = pd.date_range("2026-01-01", periods=60, freq="min", tz="UTC")
    base = 30 + np.arange(60) * 0.05
    rows = [[x, "O_3", "outdoor", b, 0] for x, b in zip(t, base)]
    rows += [[x, "O_4", "outdoor", b + 2.0, 0] for x, b in zip(t[:30], base[:30])]
    m = _mins(rows)
    off = fl.device_offsets(m)
    assert abs(off["O_4"] - off["O_3"] - 2.0) < 1e-9
    ref = fl.reference_series(m)
    assert np.allclose(np.diff(ref.to_numpy()), 0.05)


def test_background_nan_beyond_ten_minutes():
    t = pd.date_range("2026-01-01 10:00", periods=10, freq="min", tz="UTC")
    m = _mins([[x, "O_3", "outdoor", 30.0 + i, 0] for i, x in enumerate(t)])
    q = pd.to_datetime(["2026-01-01 10:04:30", "2026-01-01 10:15:00", "2026-01-01 10:30:00", "2026-01-01 09:00:00"], utc=True)
    out = fl.background_temperature(q, m)
    assert abs(out[0] - 34.5) < 1e-9
    assert np.isnan(out[2]) and np.isnan(out[3])


def test_indoor_not_used_for_outdoor_and_coverage():
    t = pd.date_range("2026-01-01 10:00", periods=20, freq="min", tz="UTC")
    m = _mins([[x, "I_1", "indoor", 28.0, 0] for x in t] + [[x, "O_3", "outdoor", 31.0, 0] for x in t[:10]])
    assert fl.coverage_share(t[0], t[19], m, "indoor") == 1.0
    assert abs(fl.coverage_share(t[0], t[19], m, "outdoor") - 0.5) < 1e-9
    assert np.allclose(fl.background_temperature([t[2]], m, "indoor"), 28.0)


def test_tz_shifts_to_utc(tmp_path):
    _csv(tmp_path / "O_3_20260101_01durhrs.csv", [["2026-01-01 10:00:05", -22.855, -43.24, 30.0, 50]])
    m, _ = fl.load_loggers(tmp_path, tz="America/Sao_Paulo")
    assert m["t_utc"].item() == pd.Timestamp("2026-01-01 13:00", tz="UTC")


def test_default_clock_is_rio_local_time(tmp_path):
    _csv(tmp_path / "O_3_20260101_01durhrs.csv", [["2026-01-01 10:00:05", -22.855, -43.24, 30.0, 50]])
    m, _ = fl.load_loggers(tmp_path)
    assert m["t_utc"].item() == pd.Timestamp("2026-01-01 13:00", tz="UTC")
