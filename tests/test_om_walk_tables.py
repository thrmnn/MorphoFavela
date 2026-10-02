import numpy as np
import pandas as pd
import pytest

from src.om_package import p10_p11, sun_envelope as se, walk_tables as wt
from src.om_package.segments import aggregate_to_segments
from src.om_package.shade import compute_shade_local
from src.om_package.spec import internal_dir_for

LAT, LON = -22.86, -43.24
AZ = np.arange(0, 360, 15.0)
TZ = "America/Sao_Paulo"


def test_regime_slug_from_name():
    assert p10_p11.regime_slug("east-southeast") == "east_southeast"
    assert p10_p11.regime_slug("North Northwest") == "north_northwest"


def test_regime_column_names_and_duplicate_slug():
    cols = p10_p11.regime_column_names(["a_b"])
    assert cols == ["frontal_area_density_windward_a_b", "canyon_alignment_deg_a_b",
                    "upwind_shelter_angle_deg_a_b", "z0_macdonald_m_a_b"]
    season = {"campaign": {"regimes": [{"key": "reg1", "name": "east", "mean_direction_deg": 90.0},
                                       {"key": "reg2", "name": "east", "mean_direction_deg": 95.0}]}}
    with pytest.raises(ValueError):
        p10_p11.campaign_regime_list(season)


def test_new_point_columns_one_set_per_regime():
    lf = {f"lambda_f_{d}": [0.1 * i] for i, d in enumerate(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])}
    pts = pd.DataFrame({"point_id": ["p"], "street_orientation_deg": [0.0], "lambda_p_buffer_50m": [0.4],
                        "building_height_mean_buffer_50m": [6.0], **lf})
    H = np.full((1, len(AZ)), 20.0)
    hor = se._tidy(["p"], H, AZ)
    regimes = [{"slug": "east", "mean_direction_deg": 90.0}, {"slug": "north", "mean_direction_deg": 0.0}]
    out = p10_p11.new_point_columns(pts, H, AZ, hor, regimes=regimes, lat=LAT, lon=LON)
    assert out["canyon_alignment_deg_east"].iloc[0] == 90.0 and out["canyon_alignment_deg_north"].iloc[0] == 0.0
    assert out["upwind_shelter_angle_deg_east"].iloc[0] == 20.0
    assert out["frontal_area_density_windward_east"].iloc[0] == pytest.approx(0.2)
    assert {"annual_sun_hours", "zd_macdonald_m", "open_space_fraction"} <= set(out.columns)


def test_iso_local_has_offset_and_utc_twin():
    t = pd.Series(pd.to_datetime(["2026-01-15 15:00:30"], utc=True))
    assert wt.iso_local(t).iloc[0] == "2026-01-15T12:00:30-03:00"
    assert wt.iso_utc(t).iloc[0] == "2026-01-15T15:00:30Z"


def test_shaded_at_arrival_follows_horizon():
    H = np.vstack([np.full(len(AZ), 0.0), np.full(len(AZ), 89.0)])
    arr = pd.DataFrame({"point_id": ["open", "closed", "open"],
                        "t_arrival_utc": pd.to_datetime(["2026-01-15 15:00", "2026-01-15 15:00", None], utc=True)})
    out = wt.shaded_at_arrival(arr, H, AZ, np.array(["open", "closed"]), lat=LAT, lon=LON)
    assert out[0] == 0.0 and out[1] == 1.0 and np.isnan(out[2])


def test_compute_shade_local_daylight_only_local_time(tmp_path):
    H = np.vstack([np.full(len(AZ), 0.0), np.full(len(AZ), 89.0)])
    summary, sub = compute_shade_local(["open", "closed"], ["2026-01-15"], 5, LAT, LON, TZ, H, AZ, tmp_path / "s.parquet")
    df = pd.read_parquet(tmp_path / "s.parquet")
    assert len(df) == summary["n_rows"] and (df["sun_altitude_deg"] > 0).all()
    assert df["timestamp_local"].dt.hour.min() >= 4 and df["timestamp_local"].dt.hour.max() <= 19
    assert (df["timestamp_local"].dt.tz_convert("UTC") == df["timestamp_utc"]).all()
    assert df.loc[df.point_id == "open", "shaded"].sum() == 0 and df.loc[df.point_id == "closed", "shaded"].all()
    assert (sub["timestamp_local"].dt.minute % 15 == 0).all()


def test_aggregate_by_walk_conserves_points():
    df = pd.DataFrame({"walk_id": ["a"] * 4 + ["b"] * 4, "point_id": range(8),
                       "distance_along_m": [0, 1, 10, 11] * 2, "x_tau5s": range(8)})
    seg = aggregate_to_segments(df, 10, by="walk_id")
    assert list(seg["walk_id"]) == ["a", "a", "b", "b"] and int(seg["n_points"].sum()) == 8
    assert seg.loc[0, "x_tau5s"] == 0.5


def test_internal_dir_is_a_sibling_of_the_packages(tmp_path):
    pkg = tmp_path / "_packages" / "mare_om2" / "v9.9.9"
    assert internal_dir_for(pkg) == tmp_path / "_packages" / "_internal" / "mare_om2" / "v9.9.9"
