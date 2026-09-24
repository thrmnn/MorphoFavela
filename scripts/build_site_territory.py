#!/usr/bin/env python3
"""SITETERR: write data/<site>/territory.gpkg + territory_provenance.json
from config/sites.yaml, via src.sites.territory.load_territory.

    python scripts/build_site_territory.py --site maré
    python scripts/build_site_territory.py --all
    python scripts/build_site_territory.py --all --root /home/theo/SCL/SCR/MorphoFavela

territory.gpkg layers: data_extent, study_area, citywide, subunits (when the
site has any) — one feature per polygon in citywide/subunits, one feature
(the dissolved boundary) in data_extent/study_area.

territory_provenance.json adds what load_territory() itself does not compute
(it never loads a site's buildings layer): the share of buildings that fall
inside each of the four boundaries, and the overlap between study_area and
citywide (so a silent divergence like Maré's shows up as a number, not a
guess).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sites.territory import ROOT_DEFAULT, Territory, load_sites_config, load_territory  # noqa: E402
from src.svf_v2.paths import AREA_FILES  # noqa: E402


def _buildings_gdf(site: str, root: Path) -> gpd.GeoDataFrame | None:
    reg = AREA_FILES.get(site)
    if not reg or "footprints" not in reg:
        return None
    p = root / "data" / site / "raw" / reg["footprints"]
    if not p.exists():
        return None
    gdf = gpd.read_file(p)
    if gdf.crs is None:
        gdf = gdf.set_crs(31983)
    else:
        gdf = gdf.to_crs(31983)
    return gdf


def _share_inside(points, geom) -> float | None:
    if points is None or len(points) == 0:
        return None
    from src.sites.territory import within_mask
    xs = points.geometry.x.to_numpy()
    ys = points.geometry.y.to_numpy()
    mask = within_mask(xs, ys, geom)
    return float(mask.mean())


def _inferred_matches(territory: Territory) -> list[dict]:
    """The provenance's own crosswalk 'inferred' entries (today, only Maré's
    neighbourhoods_provenance.json carries a per-community `match` field)."""
    crosswalk = territory.provenance.get("subunits", {}).get("provenance", {}).get("crosswalk", [])
    return [c for c in crosswalk if c.get("match") == "inferred"]


def build_one(site: str, root: Path) -> dict:
    territory = load_territory(site, root)
    buildings = _buildings_gdf(territory.site, root)
    centroids = None
    if buildings is not None:
        centroids = gpd.GeoDataFrame(geometry=buildings.geometry.centroid, crs=buildings.crs)

    prov = json.loads(json.dumps(territory.provenance))  # plain-dict deep copy

    prov["building_share_inside"] = {
        "data_extent": _share_inside(centroids, territory.data_extent),
        "study_area": _share_inside(centroids, territory.study_area),
        "citywide": (_share_inside(centroids, territory.citywide.geometry.union_all())
                     if len(territory.citywide) else None),
        "n_buildings": int(len(buildings)) if buildings is not None else None,
    }
    for cand_id, cand in territory.study_area_candidates.items():
        prov["building_share_inside"][f"candidate:{cand_id}"] = _share_inside(centroids, cand["geometry"])

    study_area_geom = territory.study_area
    citywide_union = territory.citywide.geometry.union_all() if len(territory.citywide) else None
    if citywide_union is not None and not study_area_geom.is_empty:
        inter = study_area_geom.intersection(citywide_union).area
        union = study_area_geom.union(citywide_union).area
        prov["overlap_study_area_vs_citywide"] = {
            "intersection_m2": float(inter),
            "union_m2": float(union),
            "iou": float(inter / union) if union else None,
            "identical_boundary": bool(
                study_area_geom.symmetric_difference(citywide_union).area < 1.0),
        }
    else:
        prov["overlap_study_area_vs_citywide"] = None

    prov["inferred_matches"] = _inferred_matches(territory)

    out_dir = root / "data" / territory.site
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = out_dir / "territory.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()

    gpd.GeoDataFrame(geometry=[territory.data_extent], crs=31983).to_file(
        gpkg_path, layer="data_extent", driver="GPKG")
    gpd.GeoDataFrame(geometry=[territory.study_area], crs=31983).to_file(
        gpkg_path, layer="study_area", driver="GPKG")
    if len(territory.citywide):
        territory.citywide.to_file(gpkg_path, layer="citywide", driver="GPKG")
    if territory.subunits is not None and len(territory.subunits):
        territory.subunits.to_file(gpkg_path, layer="subunits", driver="GPKG")
    for cand_id, cand in territory.study_area_candidates.items():
        gpd.GeoDataFrame(geometry=[cand["geometry"]], crs=31983).to_file(
            gpkg_path, layer=f"study_area_candidate_{cand_id}", driver="GPKG")

    prov_path = out_dir / "territory_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[{territory.site}] wrote {gpkg_path} + {prov_path}")
    return prov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", help="one registry key from config/sites.yaml")
    ap.add_argument("--all", action="store_true", help="build every registered site")
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT,
                    help="repo root to read data/ from and write data/<site>/territory.* to "
                         "(default: this checkout's own root — pass the main checkout's "
                         "absolute path when running from a worktree)")
    args = ap.parse_args()
    root = args.root.resolve()

    if not args.site and not args.all:
        ap.error("pass --site X or --all")

    sites = list(load_sites_config()) if args.all else [args.site]
    for site in sites:
        build_one(site, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
