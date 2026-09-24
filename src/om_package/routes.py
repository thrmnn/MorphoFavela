"""Load an Octopus OM route (OM_1..OM_4_inferred_route.json), chain its
edges into one ordered centreline, and densify it into stable 1 m points.

Route file format (Google Drive, owner = the PI):
    {"route_id", "edges": [{u, v, key, edge_order}, ...],
     "edge_geometry": {"(u,v,key)": {"geometry": WKT LINESTRING lon/lat WGS84,
                                      "length": m}}}

Edges are chained by ``edge_order``. A stored LINESTRING may run in either
direction relative to (u, v) — this module never trusts (u, v) node
direction (the route file supplies no node coordinates to check it against)
and instead chains geometrically: each edge is oriented so its start point
matches the previous edge's end point (nearest-endpoint chaining), which is
robust regardless of how (u, v) was assigned upstream.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, Point
from shapely.ops import transform as shapely_transform

WGS84 = "EPSG:4326"
UTM23S = "EPSG:31983"  # SIRGAS 2000 / UTM 23S

#: Pedestrian observer height (m). Matches this repo's own SVF/ray-casting
#: default (src/svf_v2/sampling.py `pedestrian_height=1.5`) so OM2 points
#: join cleanly against the airborne SVF/street outputs that were sampled
#: at the same height.
PEDESTRIAN_HEIGHT_M = 1.5

#: OM2 is sampled every 1 m along the chained route (P-02).
POINT_SPACING_M = 1.0

_TO_UTM = Transformer.from_crs(WGS84, UTM23S, always_xy=True)


def _key_to_tuple(key: str) -> tuple[int, int, int]:
    """``"(u,v,key)"`` -> ``(u, v, key)``. The route file's edge_geometry
    keys are Python tuple reprs stringified, not JSON arrays."""
    return ast.literal_eval(key)


@dataclass(frozen=True)
class RouteEdge:
    u: int
    v: int
    key: int
    edge_order: int
    geometry_wgs84: LineString  # as stored, orientation not yet resolved


def load_route(path: Path) -> tuple[str, list[RouteEdge]]:
    """Parse a route JSON file. Returns (route_id, edges sorted by edge_order)."""
    data = json.loads(Path(path).read_text())
    route_id = data["route_id"]
    edge_geom = data["edge_geometry"]
    edges = []
    for e in data["edges"]:
        gkey = f"({e['u']},{e['v']},{e['key']})"
        entry = edge_geom.get(gkey)
        if entry is None:
            # Fall back to scanning keys (defensive: formatting drift, e.g. spaces)
            for k, v in edge_geom.items():
                if _key_to_tuple(k) == (e["u"], e["v"], e["key"]):
                    entry = v
                    break
        if entry is None:
            raise KeyError(f"{route_id}: no edge_geometry for edge {e}")
        geom = shapely_wkt.loads(entry["geometry"])
        edges.append(
            RouteEdge(u=e["u"], v=e["v"], key=e["key"], edge_order=e["edge_order"], geometry_wgs84=geom)
        )
    edges.sort(key=lambda e: e.edge_order)
    return route_id, edges


def _endpoints(line: LineString) -> tuple[Point, Point]:
    coords = list(line.coords)
    return Point(coords[0]), Point(coords[-1])


def chain_edges(edges: list[RouteEdge]) -> LineString:
    """Orient each edge geometrically so consecutive edges connect head-to-tail,
    then merge into one LineString (WGS84 lon/lat, as stored)."""
    if not edges:
        raise ValueError("no edges to chain")

    oriented: list[list[tuple[float, float]]] = []

    if len(edges) == 1:
        oriented.append(list(edges[0].geometry_wgs84.coords))
    else:
        s0, e0 = _endpoints(edges[0].geometry_wgs84)
        s1, e1 = _endpoints(edges[1].geometry_wgs84)
        # Whichever endpoint of edge0 is closest to either endpoint of edge1
        # is the shared node; start the chain from edge0's OTHER endpoint.
        dists = {
            "s0-s1": s0.distance(s1),
            "s0-e1": s0.distance(e1),
            "e0-s1": e0.distance(s1),
            "e0-e1": e0.distance(e1),
        }
        best = min(dists, key=dists.get)
        first_coords = list(edges[0].geometry_wgs84.coords)
        if best in ("s0-s1", "s0-e1"):
            # s0 is shared -> chain starts at e0
            first_coords = list(reversed(first_coords))
        oriented.append(first_coords)

        chain_end = Point(oriented[-1][-1])
        for edge in edges[1:]:
            coords = list(edge.geometry_wgs84.coords)
            start, end = Point(coords[0]), Point(coords[-1])
            if start.distance(chain_end) <= end.distance(chain_end):
                pass  # already forward
            else:
                coords = list(reversed(coords))
            oriented.append(coords)
            chain_end = Point(coords[-1])

    merged: list[tuple[float, float]] = []
    for coords in oriented:
        if merged and merged[-1] == coords[0]:
            merged.extend(coords[1:])
        else:
            merged.extend(coords)
    return LineString(merged)


def route_line_utm(path: Path) -> tuple[str, LineString]:
    """route_id, chained route LineString reprojected to EPSG:31983."""
    route_id, edges = load_route(path)
    line_wgs84 = chain_edges(edges)
    line_utm = shapely_transform(_TO_UTM.transform, line_wgs84)
    return route_id, line_utm


def stable_point_id(route_id: str, index: int) -> str:
    """e.g. OM2-000000. index = integer count of metres from route start
    (== round(distance_along_m) at POINT_SPACING_M = 1 m), so IDs are
    deterministic from (route_id, distance_along) and stable across builds."""
    slug = route_id.replace("OM_", "OM")
    return f"{slug}-{index:06d}"


def densify_route(path: Path, spacing_m: float = POINT_SPACING_M) -> gpd.GeoDataFrame:
    """OM route -> 1-point-per-metre GeoDataFrame, EPSG:31983, pedestrian height.

    Columns: point_id, route_id, seq, distance_along_m, height_m, geometry.
    No segments are imposed — that's P-03's job on top of these points.
    """
    route_id, line = route_line_utm(path)
    total_len = line.length
    n_points = int(np.floor(total_len / spacing_m)) + 1
    distances = np.arange(n_points) * spacing_m
    # Always include the final vertex exactly (route end), even if it falls
    # short of the next whole-metre step.
    if not np.isclose(distances[-1], total_len) and total_len - distances[-1] > 1e-9:
        distances = np.append(distances, total_len)

    points = [line.interpolate(d) for d in distances]
    slug = route_id.replace("OM_", "OM")
    ids = [f"{slug}-{i:06d}" for i in range(len(distances))]

    gdf = gpd.GeoDataFrame(
        {
            "point_id": ids,
            "route_id": route_id,
            "seq": np.arange(len(distances)),
            "distance_along_m": distances,
            "height_m": PEDESTRIAN_HEIGHT_M,
        },
        geometry=points,
        crs=UTM23S,
    )
    return gdf


def route_length_m(path: Path) -> float:
    _, line = route_line_utm(path)
    return line.length
