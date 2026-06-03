"""Prong B — H/W canyon aspect-ratio proxy along the pedestrian network.

For each ~10 m segment of the street network, estimate the local canyon
aspect ratio H/W where:
    W = sum of perpendicular distances to the nearest flanking building
        facade on each side of the centerline (with a search radius cap)
    H = mean height of the two flanking buildings.

Flag skimming-flow likely at H/W > 0.65 (Oke 1988).

Outputs (per site):
    outputs/<site>/morphometrics/canyon/hw_streets.gpkg
    outputs/brisa_ventilation_fix/hw_streets_stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.strtree import STRtree

SITES = {
    "vidigal": ("Vidigal", "data/vidigal/raw/Vidigal_roads.shp"),
    "rocinha": ("Rocinha", "data/rocinha/raw/roads_rocinha.shp"),
    "complexo_do_alemao": (
        "Complexo do Alemão",
        "data/complexo_do_alemao/raw/roads_complexo_do_alemao.shp",
    ),
    "riodaspedras": ("Rio das Pedras", "data/riodaspedras/raw/roads_riodaspedras.shp"),
    "maré": ("Maré", "data/maré/raw/street_mare.shp"),
}

SEGMENT_LEN = 10.0          # m, sampling interval along centerlines
SEARCH_RADIUS = 40.0        # m, max half-width to consider a building "flanking"
PERP_OFFSET = 0.5           # m, lateral start offset to break the centerline tie
HW_SKIMMING = 0.65          # Oke 1988
HW_WAKE_LO = 0.35           # Oke 1988 isolated→wake-interference boundary


def _iter_segments(line: LineString, step: float):
    """Yield (midpoint, direction unit vector) every ``step`` along the line."""
    length = line.length
    if length < step:
        coords = list(line.coords)
        x0, y0 = coords[0][:2]
        x1, y1 = coords[-1][:2]
        dx, dy = x1 - x0, y1 - y0
        n = float(np.hypot(dx, dy))
        if n == 0:
            return
        ux, uy = dx / n, dy / n
        mid = line.interpolate(length / 2)
        yield mid.x, mid.y, ux, uy
        return
    d = step / 2
    while d < length:
        p0 = line.interpolate(max(0.0, d - step / 4))
        p1 = line.interpolate(min(length, d + step / 4))
        dx, dy = p1.x - p0.x, p1.y - p0.y
        n = float(np.hypot(dx, dy))
        if n == 0:
            d += step
            continue
        mid = line.interpolate(d)
        yield mid.x, mid.y, dx / n, dy / n
        d += step


def _hw_at(
    mx: float,
    my: float,
    ux: float,
    uy: float,
    tree: STRtree,
    geoms: list,
    heights: np.ndarray,
    radius: float,
):
    """Return (W, H, d_left, d_right, h_left, h_right) or None."""
    # Perpendicular to direction (ux,uy): nx=-uy, ny=ux
    nx, ny = -uy, ux
    p_left = Point(mx + PERP_OFFSET * nx, my + PERP_OFFSET * ny)
    p_right = Point(mx - PERP_OFFSET * nx, my - PERP_OFFSET * ny)

    query_box = Point(mx, my).buffer(radius).bounds
    cand_idx = tree.query(Point(mx, my).buffer(radius))
    if len(cand_idx) == 0:
        return None

    best_left = (np.inf, np.nan)
    best_right = (np.inf, np.nan)
    for i in cand_idx:
        g = geoms[i]
        # Project building centroid onto the perpendicular axis to assign side
        c = g.centroid
        side = (c.x - mx) * nx + (c.y - my) * ny
        dist = g.distance(Point(mx, my))
        if dist > radius:
            continue
        h = heights[i]
        if not np.isfinite(h) or h <= 0:
            continue
        if side >= 0:
            if dist < best_left[0]:
                best_left = (dist, h)
        else:
            if dist < best_right[0]:
                best_right = (dist, h)

    d_left, h_left = best_left
    d_right, h_right = best_right
    if not (np.isfinite(d_left) and np.isfinite(d_right)):
        return None
    W = d_left + d_right
    if W <= 0.5:
        return None
    H = 0.5 * (h_left + h_right)
    return W, H, d_left, d_right, h_left, h_right


def process_site(site: str, label: str, roads_path: str) -> dict:
    roads = gpd.read_file(_ROOT / roads_path)
    buildings = gpd.read_file(_ROOT / "data" / site / "buildings_extended_300m.gpkg")
    if "height" not in buildings.columns and "altura" in buildings.columns:
        buildings["height"] = buildings["altura"]
    invalid = ~buildings.geometry.is_valid
    if invalid.any():
        buildings.loc[invalid, "geometry"] = buildings.loc[invalid, "geometry"].buffer(0)
    if buildings.crs != roads.crs:
        buildings = buildings.to_crs(roads.crs)

    heights = buildings["height"].astype(float).values
    geoms = list(buildings.geometry.values)
    tree = STRtree(geoms)

    records = []
    for ridx, geom in enumerate(roads.geometry):
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        elif isinstance(geom, LineString):
            lines = [geom]
        else:
            continue
        for line in lines:
            for mx, my, ux, uy in _iter_segments(line, SEGMENT_LEN):
                res = _hw_at(mx, my, ux, uy, tree, geoms, heights, SEARCH_RADIUS)
                if res is None:
                    continue
                W, H, dl, dr, hl, hr = res
                records.append(
                    {
                        "road_idx": ridx,
                        "x": mx,
                        "y": my,
                        "dx": ux,
                        "dy": uy,
                        "W": W,
                        "H": H,
                        "d_left": dl,
                        "d_right": dr,
                        "h_left": hl,
                        "h_right": hr,
                        "HW": H / W if W > 0 else np.nan,
                    }
                )

    if not records:
        return {"site": site, "label": label, "error": "no segments produced"}

    df = pd.DataFrame.from_records(records)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.x, df.y), crs=roads.crs
    )

    out_dir = _ROOT / "outputs" / site / "morphometrics" / "canyon"
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = out_dir / "hw_streets.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")

    s = df["HW"].dropna()
    stats = {
        "n_segments": int(len(df)),
        "n_valid_HW": int(s.size),
        "HW": {
            "min": float(s.min()),
            "p5": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "mean": float(s.mean()),
        },
        "W_m": {
            "p5": float(df["W"].quantile(0.05)),
            "p50": float(df["W"].quantile(0.50)),
            "p95": float(df["W"].quantile(0.95)),
        },
        "H_m": {
            "p5": float(df["H"].quantile(0.05)),
            "p50": float(df["H"].quantile(0.50)),
            "p95": float(df["H"].quantile(0.95)),
        },
        "exceedance": {
            "wake_interference (HW>0.35)": float((s > HW_WAKE_LO).mean()),
            "skimming (HW>0.65)": float((s > HW_SKIMMING).mean()),
            "deep_skimming (HW>1.0)": float((s > 1.0).mean()),
            "deep_skimming (HW>1.5)": float((s > 1.5).mean()),
        },
        "hw_streets_gpkg": str(gpkg_path.relative_to(_ROOT)),
    }
    return {"site": site, "label": label, **stats}


def main() -> None:
    out_root = _ROOT / "outputs" / "brisa_ventilation_fix"
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for site, (label, roads_path) in SITES.items():
        print(f"[+] {label}")
        try:
            r = process_site(site, label, roads_path)
            results.append(r)
            if "HW" in r:
                hw = r["HW"]
                ex = r["exceedance"]
                print(
                    f"    n={r['n_valid_HW']}  HW p5/p50/p95 = "
                    f"{hw['p5']:.2f}/{hw['p50']:.2f}/{hw['p95']:.2f}  "
                    f"%skim={100*ex['skimming (HW>0.65)']:.1f}  "
                    f"%wake={100*ex['wake_interference (HW>0.35)']:.1f}"
                )
            else:
                print(f"    ERROR: {r.get('error')}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({"site": site, "label": label, "error": str(e)})

    payload = {
        "segment_length_m": SEGMENT_LEN,
        "search_radius_m": SEARCH_RADIUS,
        "thresholds": {
            "isolated_to_wake_interference (HW=0.35)": "Oke 1988",
            "skimming_onset (HW=0.65)": "Oke 1988",
        },
        "sites": results,
    }
    (out_root / "hw_streets_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
    print(f"\nWrote {out_root / 'hw_streets_stats.json'}")


if __name__ == "__main__":
    main()
