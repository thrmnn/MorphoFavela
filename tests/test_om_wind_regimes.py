import numpy as np
import pandas as pd
import pytest

from src.om_package import wind_regimes as W


def synth(modes, n=2000, kappa=30.0, seed=0, calm_n=0):
    rng = np.random.default_rng(seed)
    parts = []
    for (mu, frac) in modes:
        k = int(n * frac)
        parts.append(np.rad2deg(rng.vonmises(np.deg2rad(mu), kappa, k)) % 360.0)
    d = np.concatenate(parts)
    t = pd.date_range("2025-12-01", periods=len(d) + calm_n, freq="h", tz="UTC")
    spd = np.concatenate([rng.uniform(1, 6, len(d)), np.zeros(calm_n)])
    drct = np.concatenate([d, np.zeros(calm_n)])
    o = pd.DataFrame({"valid_utc": t, "drct": drct, "speed_ms": spd})
    o["calm"] = o["speed_ms"] < W.CALM_MS
    o["variable"] = False
    return o


def mean_dirs(res):
    return sorted(g["mean_direction_deg"] for g in res["regimes"])


@pytest.mark.parametrize("modes", [[(90, 0.6), (200, 0.4)], [(350, 0.55), (100, 0.45)]])
def test_recovers_modes_incl_wrap(modes):
    res = W.find_regimes(synth(modes))
    for mu in (m[0] for m in modes):
        assert min(float(W.circ_dist(g["mean_direction_deg"], mu)) for g in res["regimes"]) < 10
    assert res["mixture"]["max_difference_deg"] < 10
    assert res["regimes"][0]["share_of_reports"] >= res["regimes"][1]["share_of_reports"]
    assert res["regimes"][0]["key"] == "reg1"


def test_names_from_data():
    res = W.find_regimes(synth([(350, 0.55), (100, 0.45)]))
    assert {g["name"] for g in res["regimes"]} == {"north", "east"}


def test_calm_and_variable_dropped():
    o = synth([(90, 0.6), (200, 0.4)], n=500, calm_n=50)
    o.loc[0:9, "drct"] = np.nan
    o["variable"] = o["drct"].isna() & ~o["calm"]
    res = W.find_regimes(o)
    c = res["counts"]
    assert c["n_calm"] == 50 and c["n_variable_or_missing_direction"] == 10
    assert c["n_used"] == 490
    assert sum(g["n_reports"] for g in res["regimes"]) == 490
    assert res["calm_share"] == pytest.approx(50 / 550)


def test_hourly_sums_to_one_with_calm():
    o = synth([(90, 0.6), (200, 0.4)], n=480, calm_n=48)
    res = W.find_regimes(o)
    h = W.hourly_frequency(o, res)
    assert list(h.index) == list(range(24))
    assert list(h.columns) == ["reg1", "reg2", "calm"]
    assert np.allclose(h.sum(axis=1), 1.0)
    assert h["calm"].sum() > 0


def fixture_obs():
    t = pd.to_datetime(["2026-01-10 12:00", "2026-01-10 13:00", "2026-01-10 14:00",
                        "2026-01-10 20:00"], utc=True)
    return pd.DataFrame({"valid_utc": t, "drct": [90.0, 200.0, 0.0, 90.0],
                         "speed_ms": [4.0, 3.0, 0.0, 4.0],
                         "calm": [False, False, True, False], "variable": False})


def test_tag_walks_nearest_and_cutoff():
    res = W.find_regimes(synth([(90, 0.6), (200, 0.4)]))
    walks = pd.DataFrame({
        "walk_id": ["a", "b", "c", "d", "e"],
        "mid_utc": pd.to_datetime(["2026-01-10 12:20", "2026-01-10 12:40", "2026-01-10 14:10",
                                   "2026-01-10 16:00", "2026-01-10 20:59"], utc=True)})
    walks["start_utc"] = walks["mid_utc"] - pd.Timedelta(minutes=20)
    walks["end_utc"] = walks["mid_utc"] + pd.Timedelta(minutes=20)
    out = W.tag_walks(walks, fixture_obs(), res).set_index("walk_id")
    assert out.loc["a", "regime"] == "east" and out.loc["a", "minutes_from_mid"] == -20
    assert out.loc["a", "report_time_utc"] == pd.Timestamp("2026-01-10 12:00", tz="UTC")
    assert out.loc["b", "regime"] == "south-southwest"
    assert out.loc["b", "report_time_utc"] == pd.Timestamp("2026-01-10 13:00", tz="UTC")
    assert out.loc["c", "regime"] == "calm" and np.isnan(out.loc["c", "direction_deg"])
    assert out.loc["d", "regime"] == "none"
    assert out.loc["e", "regime"] == "east" and out.loc["e", "minutes_from_mid"] == -59


def test_colours_distinct():
    assert set(W.REGIME_COLOURS) == {"reg1", "reg2"}
    assert len(set(W.REGIME_COLOURS.values())) == 2


def test_mixture_with_uniform_background():
    rng = np.random.default_rng(1)
    o = synth([(60, 0.35), (250, 0.35)], n=3000, kappa=40.0)
    bg = rng.uniform(0, 360, 900)
    d = np.concatenate([o["drct"].to_numpy(), bg])
    o = pd.DataFrame({"valid_utc": pd.date_range("2025-12-01", periods=len(d), freq="h", tz="UTC"),
                      "drct": (np.round(d, -1)) % 360, "speed_ms": 3.0, "calm": False, "variable": False})
    res = W.find_regimes(o)
    mix = res["mixture"]
    for mu in (60, 250):
        assert min(float(W.circ_dist(m, mu)) for m in mix["mean_direction_deg"]) < 10
    assert max(mix["difference_deg"]) < 10
    assert 0.15 < mix["background_weight"] < 0.45
