"""Unit tests for the T3 terrain-vs-morphology sun-deficit decomposition.

Pins the physical invariants of the terrain solar-horizon model and the
decomposition arithmetic (shares in [0,1], sum to 1, terrain ≤ observed)."""

import numpy as np
import pandas as pd
import pytest

from scripts.health.terrain_morphology_split import (
    decompose,
    winter_sun_positions,
)


def test_winter_sun_is_northern_and_low():
    # Rio latitude: June-solstice noon sun is in the north (az≈0/360) and its
    # peak altitude is roughly 90-|lat|-23.45 ≈ 43.5°.
    alt, az, daylight = winter_sun_positions(-22.95, n_steps=25)
    peak = np.argmax(alt)
    assert np.degrees(alt[peak]) == pytest.approx(43.6, abs=1.0)
    assert min(az[peak] % 360, 360 - az[peak] % 360) < 1.0  # ~due north
    assert 10.0 < daylight < 11.0  # winter daylight ~10.6 h at -23°


def _toy_obs():
    # Two observers: one where terrain already loses most sun, one open.
    day = 10.6
    return pd.DataFrame({
        "site": ["s", "s"],
        "obs_sun_hours": [0.0, 6.0],
        "terrain_sun_hours": [3.0, 9.0],
        "daylight_h": [day, day],
        "svf": [0.2, 0.6],
        "slope_deg": [25.0, 3.0],
        "aspect_deg": [180.0, 0.0],
        "tif": [0.8, 1.0],
        "deficit": [1, 0],
        "terrain_deficit": [0, 0],
        "terrain_sun_lost": [day - 3.0, day - 9.0],
        "morph_sun_lost": [3.0 - 0.0, 9.0 - 6.0],
        "total_sun_lost": [day - 0.0, day - 6.0],
    })


def test_shares_are_valid_probabilities():
    summary, meta = decompose(_toy_obs())
    r = summary.iloc[0]
    for col in ("terrain_share_hours", "morph_share_hours"):
        assert 0.0 <= r[col] <= 1.0
    assert r["terrain_share_hours"] + r["morph_share_hours"] == pytest.approx(1.0)
    assert r["terrain_share_floor"] + r["morph_share_floor"] == pytest.approx(1.0)


def test_terrain_never_exceeds_observed_deficit():
    # buildings only subtract sun ⇒ terrain deficit ≤ observed deficit
    summary, _ = decompose(_toy_obs())
    r = summary.iloc[0]
    assert r["terrain_deficit_pct"] <= r["obs_deficit_pct"]
    assert 0.0 <= r["terrain_share_floor"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
