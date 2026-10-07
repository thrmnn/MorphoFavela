"""Classify and repair route points that the street map cannot explain.

The route trace comes from a street graph (OSM style) and is only as precise as
the 2019 footprints it is overlaid on. The old binary route_geometry_flag
lumped four different situations. Each point now gets one class, a repaired
position (x_repaired, y_repaired) and the evidence behind the decision.

Classes, applied in this order (all distances in metres, EPSG:31983):

  street           not inside a footprint and at most STREET_MAX_M from a mapped
                   street centreline. Position unchanged.
  projected        inside a footprint, at most PROJECT_MAX_M from open ground.
                   Moved to the nearest open ground, set back SETBACK_M from the
                   wall (open ground = outside every footprint grown by
                   SETBACK_M, so slivers narrower than twice the set back do not
                   count as walkable).
  beco             in open ground, more than STREET_MAX_M from any mapped street:
                   an alley missing from the street map. Position kept, unless
                   the line reconstruction below shows the walked line elsewhere.
  covered_passage  inside a footprint, deeper than PROJECT_MAX_M from open
                   ground, and at least COVERED_MIN_WALKS walks have a GPS fix
                   within GPS_NEAR_M. Position and measures kept (real shade
                   under a building); flagged for resident confirmation.
  unresolved       anything else: deep inside a footprint with no GPS support.
                   Position kept, measures not trusted.

Line reconstruction (becos only; projected and street points are never
re-routed by it):
  1. GPS consensus. Every raw fix (all walks; 0/0 and standing fixes dropped) is assigned
     to its nearest route point if that is within CONSENSUS_CORRIDOR_M. A route
     point collects fixes assigned within CONSENSUS_WINDOW_M of it along the
     route; its consensus position is the coordinate-wise median of those fixes
     (robust to multipath), then median-smoothed along the route over
     SMOOTH_HALF_PTS points either side. Used only where at least
     CONSENSUS_MIN_WALKS distinct walks contribute and the position lies in open
     ground.
  2. Medial axis. The free space between footprints within MEDIAL_RADIUS_M of
     the route is rasterised (MEDIAL_CELL_M cells) and skeletonised; skeleton
     cells with less than MEDIAL_MIN_CLEARANCE_M clearance are dropped.
  3. Pick. Where both exist and the nearest medial axis cell is within
     MEDIAL_MATCH_M of the consensus, the point moves to that cell (position
     source medial_axis); if the medial axis is farther, it moves to the
     consensus itself (gps_consensus). It moves only when the walked line is more
     than BECO_MOVE_MIN_M from the route trace; otherwise the trace is kept.
  Agreement per stretch of non street points is the distance from each
  consensus position to the nearest medial axis cell (stretch_agreement).

street_width_m measures facade to facade width from footprints alone, so it
exists where the street layer has no street.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.strtree import STRtree

from .routes import UTM23S, WGS84

STREET_MAX_M = 10.0
#: Route trace against 2019 footprint precision: the pilot found 93 percent of
#: in-footprint points within 4 m of open ground, i.e. a trace offset, not a
#: passage. Deeper than this a trace offset is no longer a believable story.
PROJECT_MAX_M = 4.0
SETBACK_M = 0.5
GPS_NEAR_M = 8.0
#: A fix closer than this to its previous or next fix (5 s apart, so slower than
#: 0.4 m/s) is a walker standing still. Dwell clusters (the start of each walk,
#: stops) are not the walked line and would pull the consensus into a blob.
DWELL_STEP_M = 2.0
COVERED_MIN_WALKS = 10
CONSENSUS_CORRIDOR_M = 15.0
CONSENSUS_WINDOW_M = 5.0
CONSENSUS_MIN_WALKS = 5
SMOOTH_HALF_PTS = 5
MEDIAL_RADIUS_M = 25.0
MEDIAL_CELL_M = 0.5
MEDIAL_MIN_CLEARANCE_M = 1.0
MEDIAL_MATCH_M = 6.0
BECO_MOVE_MIN_M = 5.0
WIDTH_CAP_M = 40.0

CLASSES = ["street", "projected", "beco", "covered_passage", "unresolved"]

_TO_UTM = Transformer.from_crs(WGS84, UTM23S, always_xy=True)


def load_gps_fixes(matched_dir: Path) -> pd.DataFrame:
    """Raw (unmatched) fixes of every walk: walk_id, x, y in EPSG:31983.
    Latitude and Longitude of 0 mean no fix and are dropped, and so are fixes of
    a standing walker (DWELL_STEP_M)."""
    frames = []
    for f in sorted(Path(matched_dir).glob("OM_2_*.csv")):
        d = pd.read_csv(f, usecols=["Latitude", "Longitude"])
        d = d[(d["Latitude"] != 0) & (d["Longitude"] != 0)].dropna()
        x, y = _TO_UTM.transform(d["Longitude"].to_numpy(), d["Latitude"].to_numpy())
        step = np.hypot(np.diff(x), np.diff(y))
        prev = np.r_[np.inf, step]
        nxt = np.r_[step, np.inf]
        moving = np.minimum(prev, nxt) >= DWELL_STEP_M
        frames.append(pd.DataFrame({"walk_id": f.stem, "x": x[moving], "y": y[moving]}))
    return pd.concat(frames, ignore_index=True)


def _xy(points) -> np.ndarray:
    return np.column_stack([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])


def _blocked(buildings: gpd.GeoDataFrame, margin: float):
    union = shapely.union_all(buildings.geometry.to_numpy())
    return union, union.buffer(margin)


def n_walks_nearby(xy: np.ndarray, fixes: pd.DataFrame, radius: float = GPS_NEAR_M) -> np.ndarray:
    """Distinct walks with at least one fix within radius of each point."""
    tree = cKDTree(fixes[["x", "y"]].to_numpy())
    walk = pd.factorize(fixes["walk_id"])[0]
    out = np.zeros(len(xy), dtype=int)
    for i, idx in enumerate(tree.query_ball_point(xy, radius)):
        out[i] = len(set(walk[idx])) if idx else 0
    return out


def gps_consensus(points, fixes: pd.DataFrame, open_ground=None, *,
                  corridor_m: float = CONSENSUS_CORRIDOR_M, window_m: float = CONSENSUS_WINDOW_M,
                  min_walks: int = CONSENSUS_MIN_WALKS, smooth_half: int = SMOOTH_HALF_PTS) -> pd.DataFrame:
    """Consensus walked line, one row per route point (points ordered along the
    route): cx, cy (NaN where unsupported), n_walks. open_ground, if given, is a
    function (x, y) -> bool array of positions that lie in open ground."""
    xy = _xy(points)
    d_along = points["distance_along_m"].to_numpy(float)
    dist, idx = cKDTree(xy).query(fixes[["x", "y"]].to_numpy())
    keep = dist <= corridor_m
    fx = fixes.loc[keep, ["x", "y"]].to_numpy()
    fwalk = pd.factorize(fixes.loc[keep, "walk_id"])[0]
    f_along = d_along[idx[keep]]
    order = np.argsort(f_along)
    fx, fwalk, f_along = fx[order], fwalk[order], f_along[order]
    lo = np.searchsorted(f_along, d_along - window_m, "left")
    hi = np.searchsorted(f_along, d_along + window_m, "right")
    n = len(xy)
    raw = np.full((n, 2), np.nan)
    nw = np.zeros(n, dtype=int)
    for i in range(n):
        if hi[i] > lo[i]:
            sl = slice(lo[i], hi[i])
            nw[i] = len(set(fwalk[sl]))
            if nw[i] >= min_walks:
                raw[i] = np.median(fx[sl], axis=0)
    sm = raw.copy()
    for i in range(n):
        if np.isnan(raw[i, 0]):
            continue
        w = raw[max(0, i - smooth_half): i + smooth_half + 1]
        sm[i] = np.nanmedian(w, axis=0)
    ok = ~np.isnan(sm[:, 0])
    if open_ground is not None and ok.any():
        ok[ok] = open_ground(sm[ok, 0], sm[ok, 1])
        sm[~ok] = np.nan
    return pd.DataFrame({"cx": sm[:, 0], "cy": sm[:, 1], "n_walks": nw})


def medial_axis_cells(points, blocked, *, radius_m: float = MEDIAL_RADIUS_M, cell_m: float = MEDIAL_CELL_M,
                      min_clearance_m: float = MEDIAL_MIN_CLEARANCE_M) -> np.ndarray:
    """Skeleton of the free space between footprints within radius_m of the
    route, as an (n, 3) array x, y, clearance (m)."""
    from rasterio import features
    from rasterio.transform import from_origin
    from skimage.morphology import medial_axis

    xy = _xy(points)
    line = shapely.LineString(xy) if len(xy) > 1 else shapely.Point(xy[0])
    zone = line.buffer(radius_m)
    free = zone.difference(blocked)
    minx, miny, maxx, maxy = zone.bounds
    w = int(np.ceil((maxx - minx) / cell_m)) + 1
    h = int(np.ceil((maxy - miny) / cell_m)) + 1
    tf = from_origin(minx, maxy, cell_m, cell_m)
    mask = features.rasterize([(free, 1)], out_shape=(h, w), transform=tf, fill=0, dtype="uint8").astype(bool)
    # medial_axis breaks ties at random unless seeded; unseeded, every build drew a
    # different skeleton and the beco shift statistics in the report changed between builds
    skel, dist = medial_axis(mask, return_distance=True, rng=0)
    rows, cols = np.nonzero(skel)
    clear = dist[rows, cols] * cell_m
    keep = clear >= min_clearance_m
    x = minx + (cols[keep] + 0.5) * cell_m
    y = maxy - (rows[keep] + 0.5) * cell_m
    return np.column_stack([x, y, clear[keep]])


def classify_points(points, buildings, streets, gps_fixes, *, medial_cells: np.ndarray | None = None) -> pd.DataFrame:
    """One row per point: point_id, distance_along_m, point_class, x/y original,
    x_repaired, y_repaired, shift_m, position_source and the evidence
    (dist_open_ground_m, dist_street_m, n_walks_gps_nearby, consensus_x/y,
    n_walks_consensus, medial_x/y, consensus_medial_gap_m).
    points need point_id, distance_along_m, geometry, ordered along the route."""
    xy = _xy(points)
    x, y = xy[:, 0], xy[:, 1]
    n = len(xy)
    union, blocked = _blocked(buildings, SETBACK_M)

    inside = shapely.contains_xy(union, x, y)
    pts = shapely.points(x, y)
    st_tree = STRtree(streets.geometry.to_numpy())
    _, d_st = st_tree.query_nearest(pts, return_distance=True, all_matches=False)
    dist_street = np.asarray(d_st, float)
    dist_open = np.zeros(n)
    dist_open[inside] = shapely.distance(pts[inside], union.boundary)

    xr, yr = x.copy(), y.copy()
    for i in np.nonzero(inside & (dist_open <= PROJECT_MAX_M))[0]:
        line = shapely.shortest_line(pts[i], blocked.boundary)
        xr[i], yr[i] = line.coords[1]

    n_gps = n_walks_nearby(xy, gps_fixes)
    cls = np.full(n, "unresolved", dtype=object)
    cls[~inside & (dist_street <= STREET_MAX_M)] = "street"
    cls[~inside & (dist_street > STREET_MAX_M)] = "beco"
    cls[inside & (dist_open <= PROJECT_MAX_M)] = "projected"
    cls[inside & (dist_open > PROJECT_MAX_M) & (n_gps >= COVERED_MIN_WALKS)] = "covered_passage"

    def open_ground(px, py):
        return ~shapely.contains_xy(blocked, px, py)

    cons = gps_consensus(points, gps_fixes, open_ground)
    if medial_cells is None:
        medial_cells = medial_axis_cells(points, blocked)
    m_tree = cKDTree(medial_cells[:, :2])
    cx, cy = cons["cx"].to_numpy(), cons["cy"].to_numpy()
    has_c = ~np.isnan(cx)
    mx = np.full(n, np.nan)
    my = np.full(n, np.nan)
    gap = np.full(n, np.nan)
    if has_c.any():
        d, j = m_tree.query(np.column_stack([cx[has_c], cy[has_c]]))
        gap[has_c] = d
        mx[has_c], my[has_c] = medial_cells[j, 0], medial_cells[j, 1]

    source = np.where(cls == "projected", "projected", "route").astype(object)
    for i in np.nonzero((cls == "beco") & has_c)[0]:
        if np.hypot(cx[i] - x[i], cy[i] - y[i]) <= BECO_MOVE_MIN_M:
            continue
        if gap[i] <= MEDIAL_MATCH_M:
            xr[i], yr[i], source[i] = mx[i], my[i], "medial_axis"
        else:
            xr[i], yr[i], source[i] = cx[i], cy[i], "gps_consensus"

    return pd.DataFrame({
        "point_id": points["point_id"].to_numpy(),
        "distance_along_m": points["distance_along_m"].to_numpy(),
        "point_class": cls,
        "x_original": x, "y_original": y,
        "x_repaired": xr, "y_repaired": yr,
        "shift_m": np.hypot(xr - x, yr - y),
        "position_source": source,
        "dist_open_ground_m": dist_open,
        "dist_street_m": dist_street,
        "n_walks_gps_nearby": n_gps,
        "consensus_x": cx, "consensus_y": cy, "n_walks_consensus": cons["n_walks"].to_numpy(),
        "medial_x": mx, "medial_y": my, "consensus_medial_gap_m": gap,
    })


def stretch_agreement(result: pd.DataFrame, merge_gap_m: float = 5.0, agree_m: float = 5.0) -> pd.DataFrame:
    """Runs of points that are not on a mapped street (gaps up to merge_gap_m
    bridged): length, class mix, points with a GPS consensus, and the distance
    from consensus to the nearest medial axis cell (median, share within agree_m)."""
    r = result.sort_values("distance_along_m").reset_index(drop=True)
    off = (r["point_class"] != "street").to_numpy()
    d = r["distance_along_m"].to_numpy()
    runs, start = [], None
    last = None
    for i in np.nonzero(off)[0]:
        if start is None:
            start = last = i
        elif d[i] - d[last] <= merge_gap_m:
            last = i
        else:
            runs.append((start, last))
            start = last = i
    if start is not None:
        runs.append((start, last))
    rows = []
    for a, b in runs:
        s = r.iloc[a: b + 1]
        s = s[s["point_class"] != "street"]
        g = s["consensus_medial_gap_m"].dropna()
        rows.append({
            "start_m": float(d[a]), "end_m": float(d[b]), "length_m": float(d[b] - d[a] + 1),
            "n_points": int(len(s)), "n_with_consensus": int(len(g)),
            "dominant_class": s["point_class"].mode().iloc[0],
            "gap_median_m": float(g.median()) if len(g) else np.nan,
            "share_within_agree_m": float((g <= agree_m).mean()) if len(g) else np.nan,
        })
    return pd.DataFrame(rows)


def tangent_deg(points) -> np.ndarray:
    """Route tangent bearing at each point, central difference over 3 points
    either side so a repaired jump of a metre or two does not tilt it."""
    xy = _xy(points)
    n = len(xy)
    lo = np.clip(np.arange(n) - 3, 0, n - 1)
    hi = np.clip(np.arange(n) + 3, 0, n - 1)
    return np.degrees(np.arctan2(xy[hi, 0] - xy[lo, 0], xy[hi, 1] - xy[lo, 1])) % 360.0


def street_width_m(points_repaired, buildings, *, tangent_bearing_deg: np.ndarray | None = None,
                   cap_m: float = WIDTH_CAP_M) -> pd.DataFrame:
    """Facade to facade width from footprints: rays perpendicular to the route
    direction, left and right of each point, to the nearest footprint edge
    (cap_m per side). Columns: street_width_m (left + right; NaN for a point
    inside a footprint), d_left_m, d_right_m, h_left_m, h_right_m (height of the
    footprint hit), width_capped (a side hit nothing within cap_m).
    tangent_bearing_deg defaults to tangent_deg(points_repaired)."""
    xy = _xy(points_repaired)
    n = len(xy)
    bearing = np.radians(tangent_deg(points_repaired) if tangent_bearing_deg is None else tangent_bearing_deg)
    ux, uy = np.sin(bearing), np.cos(bearing)
    geoms = buildings.geometry.to_numpy()
    heights = buildings["altura"].to_numpy(float) if "altura" in buildings else np.full(len(geoms), np.nan)
    tree = STRtree(geoms)
    union = shapely.union_all(geoms)
    inside = shapely.contains_xy(union, xy[:, 0], xy[:, 1])
    out = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        nx, ny = -uy * sign, ux * sign
        ends = np.column_stack([xy[:, 0] + nx * cap_m, xy[:, 1] + ny * cap_m])
        rays = shapely.linestrings(np.stack([xy, ends], axis=1))
        origin = shapely.points(xy)
        ri, gi = tree.query(rays, predicate="intersects")
        dist = np.full(n, cap_m)
        h = np.full(n, np.nan)
        if len(ri):
            inter = shapely.intersection(rays[ri], geoms[gi])
            d = shapely.distance(origin[ri], inter)
            df = pd.DataFrame({"r": ri, "d": d, "h": heights[gi]})
            df = df[np.isfinite(df["h"]) & (df["h"] > 0)]
            best = df.sort_values("d").groupby("r").first()
            dist[best.index.to_numpy()] = best["d"].to_numpy()
            h[best.index.to_numpy()] = best["h"].to_numpy()
        out[side] = (dist, h)
    dl, hl = out["left"]
    dr, hr = out["right"]
    width = dl + dr
    width[inside] = np.nan
    return pd.DataFrame({
        "street_width_m": width, "d_left_m": dl, "d_right_m": dr, "h_left_m": hl, "h_right_m": hr,
        "width_capped": (dl >= cap_m) | (dr >= cap_m),
    })


def horizon_svf(horizon_deg: np.ndarray, azimuths_deg: np.ndarray) -> np.ndarray:
    """Sky view factor from a horizon profile: 1 - mean(sin^2 h) over azimuth
    (horizon angles below zero count as zero). The Tregenza patch azimuths repeat,
    so the profile is first reduced to one value per distinct azimuth."""
    az, inv = np.unique(np.round(azimuths_deg, 3), return_inverse=True)
    h = np.zeros((horizon_deg.shape[0], len(az)))
    for k in range(len(az)):
        h[:, k] = horizon_deg[:, inv == k].mean(axis=1)
    h = np.clip(np.radians(h), 0, np.pi / 2)
    return 1.0 - np.mean(np.sin(h) ** 2, axis=1)


def building_height_hw(sw: pd.DataFrame) -> pd.DataFrame:
    """Canyon height as the mean of the two flanking footprint heights (one side
    if the other hit nothing) and height-to-width ratio with the ray width."""
    h = sw[["h_left_m", "h_right_m"]].mean(axis=1, skipna=True)
    w = sw["street_width_m"]
    return pd.DataFrame({"building_height_m": h, "height_width_ratio": h / w.where(w > 0.5)})


def repair_route(paths, matched_dir: Path, stored_points: Path, out_dir: Path, *, device: str | None = "cuda") -> dict:
    """Run the whole repair on OM2: classify, measure widths, re-march the
    horizon for moved points only, write <out_dir>/route_repair.parquet and
    <out_dir>/flags_facts.json. stored_points is the shipped points table
    (the before values). Returns the facts dict."""
    import json

    from .p10_p11 import horizon_arrays_and_table
    from .routes import densify_route

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = densify_route(paths.route_json("OM_2"))
    buildings = gpd.read_file(paths.buildings_mare)
    streets = gpd.read_file(paths.street_mare)
    fixes = load_gps_fixes(matched_dir)
    res = classify_points(pts, buildings, streets, fixes)

    rep = gpd.GeoDataFrame(res[["point_id"]].assign(distance_along_m=res["distance_along_m"]),
                           geometry=gpd.points_from_xy(res["x_repaired"], res["y_repaired"]), crs=UTM23S)
    bearing = tangent_deg(pts)
    sw = street_width_m(rep, buildings, tangent_bearing_deg=bearing)
    hh = building_height_hw(sw)
    res = pd.concat([res, sw, hh.add_suffix("_new")], axis=1)

    before = pd.read_parquet(stored_points, columns=[
        "point_id", "street_width_m", "building_height_m", "height_width_ratio", "sky_view_factor"])
    before.columns = ["point_id", "street_width_m_before", "building_height_m_before",
                      "height_width_ratio_before", "sky_view_factor_before"]
    res = res.merge(before, on="point_id", how="left")

    moved = res["shift_m"].to_numpy() > 0.01
    res["svf_horizon_before"] = np.nan
    res["svf_horizon_after"] = np.nan
    m = res[moved]
    # Moved points are marched at both positions; every other point once, at its
    # (unchanged) position, as the like-for-like reference for the horizon method.
    both = gpd.GeoDataFrame(
        {"point_id": ["o_" + p for p in m["point_id"]] + ["r_" + p for p in res["point_id"]]},
        geometry=gpd.points_from_xy(np.r_[m["x_original"], res["x_repaired"]], np.r_[m["y_original"], res["y_repaired"]]),
        crs=UTM23S)
    h, az, _ = horizon_arrays_and_table(both, paths, device=device)
    svf = horizon_svf(h, az)
    k = int(moved.sum())
    res.loc[moved, "svf_horizon_before"] = svf[:k]
    res["svf_horizon_after"] = svf[k:]
    res["moved"] = moved
    res.to_parquet(out_dir / "route_repair.parquet", index=False)
    facts = repair_facts(res)
    (out_dir / "flags_facts.json").write_text(json.dumps(facts, indent=2))
    return facts


def repair_facts(res: pd.DataFrame) -> dict:
    """Numbers for the report, all computed from the repair table."""
    def med(s):
        s = pd.Series(s).dropna()
        return float(s.median()) if len(s) else None

    cls = res["point_class"]
    proj = res[cls == "projected"]
    beco = res[cls == "beco"]
    st = res[cls == "street"]
    stretches = stretch_agreement(res)
    gap = res["consensus_medial_gap_m"].dropna()
    moved = res[res["moved"]]
    both = st.dropna(subset=["street_width_m", "street_width_m_before"])
    facts = {
        "n_points": int(len(res)),
        "counts": {c: int((cls == c).sum()) for c in CLASSES},
        "projected_shift_median_m": med(proj["shift_m"]),
        "projected_shift_max_m": float(proj["shift_m"].max()) if len(proj) else None,
        "beco_points": int(len(beco)),
        "beco_length_m": float(len(beco)),
        "beco_moved_points": int((beco["shift_m"] > 0.01).sum()),
        "beco_moved_shift_median_m": med(beco.loc[beco["shift_m"] > 0.01, "shift_m"]),
        "covered_passage_length_m": float((cls == "covered_passage").sum()),
        "gps_medial_gap_median_m": med(gap),
        "gps_medial_share_within_5m": float((gap <= 5).mean()) if len(gap) else None,
        "n_with_gps_consensus": int(len(gap)),
        "stretches": stretches.to_dict(orient="records"),
        "width_street_points": {
            "n": int(len(both)),
            "median_new_m": med(both["street_width_m"]),
            "median_old_m": med(both["street_width_m_before"]),
            "median_difference_m": med(both["street_width_m"] - both["street_width_m_before"]),
            "median_abs_difference_m": med((both["street_width_m"] - both["street_width_m_before"]).abs()),
            "capped_share": float(st["width_capped"].mean()),
            "median_difference_uncapped_m": med((both["street_width_m"] - both["street_width_m_before"])[~both["width_capped"]]),
            "n_uncapped": int((~both["width_capped"]).sum()),
        },
        "moved_points": {
            "n": int(len(moved)),
            "svf_horizon_before_median": med(moved["svf_horizon_before"]),
            "svf_horizon_after_median": med(moved["svf_horizon_after"]),
            "svf_stored_before_median": med(moved["sky_view_factor_before"]),
            "svf_horizon_street_points_median": med(st["svf_horizon_after"]),
            "svf_stored_street_points_median": med(st["sky_view_factor_before"]),
            "height_width_before_median": med(moved["height_width_ratio_before"]),
            "height_width_after_median": med(moved["height_width_ratio_new"]),
            "building_height_before_median_m": med(moved["building_height_m_before"]),
            "building_height_after_median_m": med(moved["building_height_m_new"]),
            "street_width_before_median_m": med(moved["street_width_m_before"]),
            "street_width_after_median_m": med(moved["street_width_m"]),
        },
    }
    return facts
