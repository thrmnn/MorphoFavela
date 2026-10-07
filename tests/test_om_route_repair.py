import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import LineString, box

from src.om_package import route_repair as rr

CRS = "EPSG:31983"


def _points(xy):
    xy = np.asarray(xy, float)
    return gpd.GeoDataFrame(
        {"point_id": [f"p{i}" for i in range(len(xy))], "distance_along_m": np.arange(len(xy), dtype=float)},
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=CRS)


def _buildings(boxes, h=9.0):
    return gpd.GeoDataFrame({"altura": [h] * len(boxes)}, geometry=[box(*b) for b in boxes], crs=CRS)


def _fixes(xy_walks):
    rows = [(f"w{k}", x, y) for k, arr in enumerate(xy_walks) for x, y in arr]
    return pd.DataFrame(rows, columns=["walk_id", "x", "y"])


STREETS = gpd.GeoDataFrame(geometry=[LineString([(-100, 0), (100, 0)])], crs=CRS)
NO_FIXES = pd.DataFrame({"walk_id": ["w0"], "x": [1e6], "y": [1e6]})


def _classify(pts_xy, boxes, fixes=NO_FIXES, streets=STREETS):
    pts = _points(pts_xy)
    b = _buildings(boxes)
    return rr.classify_points(pts, b, streets, fixes)


def test_street_point_unchanged():
    r = _classify([(0, 1), (1, 1)], [(30, 30, 40, 40)])
    assert (r["point_class"] == "street").all()
    assert (r["shift_m"] == 0).all()


def test_projected_to_open_ground_with_setback():
    r = _classify([(10, 2.0)], [(0, 1.0, 20, 6.0)])
    row = r.iloc[0]
    assert row["point_class"] == "projected"
    assert row["dist_open_ground_m"] == np.float64(1.0)
    assert abs(row["y_repaired"] - (1.0 - rr.SETBACK_M)) < 1e-6
    assert abs(row["shift_m"] - (1.0 + rr.SETBACK_M)) < 1e-6
    union = shapely.union_all(_buildings([(0, 1.0, 20, 6.0)]).geometry.to_numpy())
    assert not shapely.contains_xy(union, row["x_repaired"], row["y_repaired"])
    assert abs(union.distance(shapely.Point(row["x_repaired"], row["y_repaired"])) - rr.SETBACK_M) < 1e-6


def test_beco_far_from_street_kept_in_open_ground():
    r = _classify([(0, 25)], [(30, 30, 40, 40)])
    assert r.iloc[0]["point_class"] == "beco"
    assert r.iloc[0]["shift_m"] == 0


def test_deep_in_footprint_without_gps_unresolved_with_gps_covered():
    box_ = [(-20, 5, 20, 45)]
    r = _classify([(0, 25)], box_)
    assert r.iloc[0]["point_class"] == "unresolved"
    walks = [[(0.5 * k, 25.0 + 0.1 * k)] for k in range(rr.COVERED_MIN_WALKS)]
    r = _classify([(0, 25)], box_, fixes=_fixes(walks))
    row = r.iloc[0]
    assert row["point_class"] == "covered_passage"
    assert row["n_walks_gps_nearby"] == rr.COVERED_MIN_WALKS
    assert row["shift_m"] == 0


def test_project_threshold_is_deep_side():
    box_ = [(-20, 5, 20, 45)]
    r = _classify([(0, 5 + rr.PROJECT_MAX_M - 0.1), (0, 5 + rr.PROJECT_MAX_M + 0.1)], box_)
    assert list(r["point_class"]) == ["projected", "unresolved"]


def test_consensus_median_robust_to_outlier():
    n = 40
    pts = _points([(i, 0.0) for i in range(n)])
    walks = []
    for k in range(7):
        walks.append([(i, 3.0 + (0.2 if k % 2 else -0.2)) for i in range(0, n, 2)])
    walks.append([(i, 300.0) for i in range(0, n, 2)])
    c = rr.gps_consensus(pts, _fixes(walks))
    mid = c.iloc[15:25]
    assert (mid["n_walks"] >= rr.CONSENSUS_MIN_WALKS).all()
    assert np.allclose(mid["cx"], np.arange(15, 25), atol=1.5)
    assert np.allclose(mid["cy"], 3.0, atol=0.4)


def test_consensus_needs_enough_walks():
    pts = _points([(i, 0.0) for i in range(30)])
    walks = [[(i, 2.0) for i in range(0, 30, 2)] for _ in range(rr.CONSENSUS_MIN_WALKS - 1)]
    c = rr.gps_consensus(pts, _fixes(walks))
    assert c["cx"].isna().all()


def test_consensus_must_lie_in_open_ground():
    pts = _points([(i, 0.0) for i in range(30)])
    walks = [[(i, 2.0) for i in range(0, 30, 2)] for _ in range(6)]
    c = rr.gps_consensus(pts, _fixes(walks), open_ground=lambda x, y: np.zeros(len(x), bool))
    assert c["cx"].isna().all()


def test_street_width_rays():
    xy = [(i, 0.0) for i in range(20)]
    pts = _points(xy)
    b = gpd.GeoDataFrame({"altura": [6.0, 10.0]}, geometry=[box(-5, 3, 30, 9), box(-5, -9, 30, -5)], crs=CRS)
    sw = rr.street_width_m(pts, b, tangent_bearing_deg=np.full(20, 90.0))
    assert np.allclose(sw["street_width_m"], 8.0)
    assert np.allclose(sw["d_left_m"], 5.0) or np.allclose(sw["d_left_m"], 3.0)
    assert set(np.round(sw[["d_left_m", "d_right_m"]].iloc[0], 6)) == {3.0, 5.0}
    hh = rr.building_height_hw(sw)
    assert np.allclose(hh["building_height_m"], 8.0)
    assert np.allclose(hh["height_width_ratio"], 1.0)
    assert not sw["width_capped"].any()


def test_street_width_cap_and_inside():
    pts = _points([(0.0, 0.0), (50.0, 50.0)])
    b = gpd.GeoDataFrame({"altura": [8.0]}, geometry=[box(40, 40, 60, 60)], crs=CRS)
    sw = rr.street_width_m(pts, b, tangent_bearing_deg=np.array([90.0, 90.0]))
    assert sw["street_width_m"].iloc[0] == 2 * rr.WIDTH_CAP_M
    assert sw["width_capped"].iloc[0]
    assert np.isnan(sw["street_width_m"].iloc[1])


def test_medial_axis_runs_down_the_middle_of_a_lane():
    pts = _points([(i, 0.0) for i in range(0, 60)])
    b = _buildings([(-10, 3, 70, 20), (-10, -20, 70, -3)])
    _, blocked = rr._blocked(b, rr.SETBACK_M)
    cells = rr.medial_axis_cells(pts, blocked)
    near = cells[(cells[:, 0] > 10) & (cells[:, 0] < 50) & (np.abs(cells[:, 1]) < 4)]
    assert len(near) > 20
    assert np.abs(near[:, 1]).max() < 0.8


def test_horizon_svf_limits():
    az = np.tile(np.arange(0, 360, 30.0), 2)
    assert np.allclose(rr.horizon_svf(np.zeros((1, len(az))), az), 1.0)
    assert np.allclose(rr.horizon_svf(np.full((1, len(az)), 90.0), az), 0.0)


def test_figure_smoke(tmp_path):
    from src.om_package.fig_flags import build_fig_flags

    xy = [(i, 1.0) for i in range(0, 40)] + [(40 + i, 2.5) for i in range(20)]
    boxes = [(30, 1.5, 60, 8.0)]
    pts = _points(xy)
    b = _buildings(boxes)
    walks = [[(i + 0.3, 1.0 + 0.1 * k) for i in range(0, 60, 3)] for k in range(8)]
    fx = _fixes(walks)
    res = rr.classify_points(pts, b, STREETS, fx)
    out = build_fig_flags(res, b, fx, tmp_path / "fig_flags.png", insets=[(30.0, 59.0)])
    assert out.exists() and out.stat().st_size > 5000


def test_medial_axis_is_deterministic():
    import numpy as np
    import shapely

    from src.om_package.route_repair import medial_axis_cells

    pts = gpd.GeoDataFrame(geometry=[shapely.Point(x, 0) for x in range(0, 60, 2)], crs="EPSG:31983")
    blocked = shapely.unary_union([shapely.box(10, 3, 30, 12), shapely.box(10, -12, 30, -3),
                                   shapely.box(35, 2.5, 50, 9), shapely.box(35, -9, 50, -2.5)])
    a, b = medial_axis_cells(pts, blocked), medial_axis_cells(pts, blocked)
    assert a.shape == b.shape and np.array_equal(a, b)
