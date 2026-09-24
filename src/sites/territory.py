"""SITETERR: one declared, reviewable territory per campaign site.

Registry: config/sites.yaml. Build (writes data/<site>/territory.gpkg +
territory_provenance.json): scripts/build_site_territory.py.

Before this module, a site's boundary was defined in six places (see
config/sites.yaml's header comment) and Maré's study area silently differed
from its citywide definition for months. This module is the one place that
reads config/sites.yaml and turns it into geometry; every consumer (the two
dashboard builders, the Maré brief, the territory map renderer, WP-05's
citywide matching) is expected to call `load_territory` rather than
re-deriving a boundary from a hardcoded path or `if site == "maré"` branch.

`load_territory` is intentionally light: it loads the boundary files and
returns geometry, but does not load a site's (large) buildings layer. The
richer provenance (building shares per boundary, inferred-match list,
overlaps between definitions) is a `scripts/build_site_territory.py` build
artifact, written to data/<site>/territory_provenance.json — read that file
if you need those numbers rather than recomputing them.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import shapely
import yaml

#: Where src/sites/territory.py itself lives — config/sites.yaml is checked
#: into git, so it is always read from THIS checkout, never from the `root`
#: argument below (which points at a possibly-different checkout's data/).
_CODE_ROOT = Path(__file__).resolve().parents[2]

#: Default for the `root` argument every data-loading function below takes:
#: the repo root that data/ hangs off. Callers running from a worktree with
#: no data/ of its own must pass the main checkout's absolute path.
ROOT_DEFAULT = _CODE_ROOT
SITES_YAML_REL = "config/sites.yaml"
FAVELAS_LIMIT_REL = "RJ/Favelas_Limit_2019.shp"

#: Accepted aliases for a registry key, so callers that historically wrote
#: the ASCII slug (src.brisa_solar.wp05_full / wp07_ledger's "mare") and
#: callers that write the accented site directory name ("maré") both land
#: on the one config/sites.yaml entry.
_SITE_ALIASES = {"mare": "maré"}


def normalize_site_key(site: str) -> str:
    return _SITE_ALIASES.get(site, site)


def _union(geoseries) -> "shapely.Geometry":
    return geoseries.union_all() if hasattr(geoseries, "union_all") else geoseries.unary_union


def _norm(s) -> str:
    return str(s).strip().lower()


def _to_31983(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(31983)
    return gdf.to_crs(31983)


def within_mask(x: np.ndarray, y: np.ndarray, geom) -> np.ndarray:
    """Vectorised point-in-polygon test against `geom`, bbox-prefiltered
    first. Moved here from src.brisa_solar.mare_study_area (re-exported
    there for backward compatibility) so every site can use it, not only
    Maré."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0, y0, x1, y1 = geom.bounds
    in_box = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    out = np.zeros(len(x), dtype=bool)
    if in_box.any():
        out[in_box] = shapely.contains_xy(geom, x[in_box], y[in_box])
    return out


def rotation_deg(geom) -> int:
    """Whole-degree rotation that puts the long axis of `geom`'s minimum
    rotated rectangle horizontal. Derived from geometry alone — never typed.
    Perturbing the geometry changes the returned angle (see
    tests/test_site_territory.py)."""
    mrr = shapely.minimum_rotated_rectangle(geom)
    coords = list(mrr.exterior.coords)[:-1]
    if len(coords) != 4:
        return 0
    edges = [(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in edges]
    i = int(np.argmax(lengths))
    (x0, y0), (x1, y1) = edges[i]
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    # Normalise to (-90, 90] so "rotate the map by -angle" always puts the
    # long edge horizontal, never upside down relative to it.
    angle = ((angle + 90) % 180) - 90
    return int(round(angle))


def rotate_for_display(gdf: gpd.GeoDataFrame, rotation_deg_value: float,
                       origin="center") -> gpd.GeoDataFrame:
    """Rotate `gdf` by -rotation_deg_value about `origin` so a study area
    whose display_rotation_deg is d renders with its long axis horizontal.
    Does not mutate the input. Works for a raster's extent too — wrap its
    bounds in a single-row GeoDataFrame (e.g. `gpd.GeoDataFrame(geometry=
    [shapely.geometry.box(*rasterio_dataset.bounds)], crs=...)`) and rotate
    that; this module never touches raster pixel data itself."""
    from shapely.affinity import rotate as _rotate
    out = gdf.copy()
    out["geometry"] = out.geometry.apply(lambda g: _rotate(g, -rotation_deg_value, origin=origin))
    return out


def north_arrow_angle(rotation_deg_value: float) -> float:
    """Degrees clockwise from straight-up that a north arrow must be drawn
    at on a display rotated by `rotate_for_display`'s convention (map
    content rotated by -rotation_deg_value)."""
    return float(rotation_deg_value)


def load_sites_config() -> dict:
    """config/sites.yaml from THIS checkout (see _CODE_ROOT) — always, even
    when `load_territory` is asked to read data/ from a different checkout's
    absolute path."""
    with open(_CODE_ROOT / SITES_YAML_REL, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc["sites"]


def _load_favelas(root: Path) -> gpd.GeoDataFrame:
    return _to_31983(gpd.read_file(root / "data" / FAVELAS_LIMIT_REL))


def match_citywide(favelas: gpd.GeoDataFrame, target: str) -> tuple[gpd.GeoDataFrame, str]:
    """Thin re-export of src.brisa_solar.wp05_full.match_favela_group —
    imported lazily (wp05_full pulls in torch/rasterio) so callers that only
    need geometry (tests, the hub, the dashboards) don't pay that import
    cost. Calling the *same* function WP-05/07 uses, rather than
    reimplementing its two-tier rule, is what makes the matched polygons
    identical to today's by construction."""
    from src.brisa_solar.wp05_full import match_favela_group
    return match_favela_group(favelas, target)


def load_data_extent(site_cfg: dict, root: Path) -> gpd.GeoDataFrame:
    return _to_31983(gpd.read_file(root / "data" / site_cfg["data_extent"]))


def _load_neighbourhoods_gpkg(sub_cfg: dict, root: Path) -> tuple[gpd.GeoDataFrame, dict]:
    gdf = _to_31983(gpd.read_file(root / "data" / sub_cfg["file"], layer=sub_cfg["layer"]))
    prov = {}
    prov_path = root / "data" / sub_cfg.get("provenance", "")
    if sub_cfg.get("provenance") and prov_path.exists():
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
    return gdf, {"kind": "neighbourhoods_gpkg", "file": sub_cfg["file"],
                 "provenance_file": sub_cfg.get("provenance"), "provenance": prov}


def _load_favelas_by_complexo(site_cfg: dict, root: Path) -> tuple[gpd.GeoDataFrame, dict]:
    favelas = _load_favelas(root)
    target = _norm(site_cfg["citywide_rule"]["target"])
    gdf = favelas[favelas["complexo"].map(_norm) == target].reset_index(drop=True)
    return gdf, {"kind": "favelas_limit_2019_by_complexo",
                 "target": site_cfg["citywide_rule"]["target"]}


def load_subunits(site_cfg: dict, root: Path) -> tuple[Optional[gpd.GeoDataFrame], dict]:
    sub_cfg = site_cfg.get("subunits")
    if not sub_cfg:
        return None, {}
    if sub_cfg["source"] == "neighbourhoods_gpkg":
        gdf, prov = _load_neighbourhoods_gpkg(sub_cfg, root)
    elif sub_cfg["source"] == "favelas_limit_2019_by_complexo":
        gdf, prov = _load_favelas_by_complexo(site_cfg, root)
    else:
        raise ValueError(f"unknown subunits source: {sub_cfg['source']!r}")
    name_field = sub_cfg.get("name_field")
    if name_field and name_field in gdf.columns and "name" not in gdf.columns:
        gdf = gdf.assign(name=gdf[name_field])
    return gdf, prov


def load_polygon_file(path_rel: str, root: Path) -> gpd.GeoDataFrame:
    return _to_31983(gpd.read_file(root / "data" / path_rel))


def build_study_area(kind_cfg: dict, data_extent_geom, citywide_gdf: gpd.GeoDataFrame,
                     subunits: Optional[gpd.GeoDataFrame], root: Path) -> tuple["shapely.Geometry", dict, Optional[gpd.GeoDataFrame]]:
    """Returns (geometry, provenance, subunits_with_in_extent).
    `subunits_with_in_extent` is `subunits` with an added boolean `in_extent`
    column when kind is subunits_union_in_extent, else the input unchanged."""
    kind = kind_cfg["kind"]
    if kind == "citywide_polygons":
        geom = _union(citywide_gdf.geometry)
        return geom, {"kind": kind}, subunits
    if kind == "subunits_union_in_extent":
        if subunits is None:
            raise ValueError("subunits_union_in_extent requires `subunits` in config/sites.yaml")
        threshold = kind_cfg["share_threshold"]
        share = (subunits.geometry.intersection(data_extent_geom).area
                / subunits.geometry.area)
        in_extent = share.to_numpy() >= threshold
        subunits = subunits.assign(in_extent=in_extent, share_in_data_extent=share.to_numpy())
        included = subunits[subunits["in_extent"]]
        geom = _union(included.geometry).intersection(data_extent_geom)
        return geom, {
            "kind": kind, "share_threshold": threshold,
            "n_subunits_total": int(len(subunits)),
            "n_included": int(in_extent.sum()), "n_excluded": int((~in_extent).sum()),
        }, subunits
    if kind == "polygon_file":
        gdf = load_polygon_file(kind_cfg["path"], root)
        geom = _union(gdf.geometry)
        return geom, {"kind": kind, "path": kind_cfg["path"]}, subunits
    raise ValueError(f"unknown study_area kind: {kind!r}")


@dataclass
class Territory:
    site: str
    root: Path
    display_name: str
    data_extent: "shapely.Geometry"
    study_area: "shapely.Geometry"
    citywide: gpd.GeoDataFrame
    citywide_method: str
    subunits: Optional[gpd.GeoDataFrame]
    study_area_candidates: dict = field(default_factory=dict)
    display_rotation_deg: int = 0
    provenance: dict = field(default_factory=dict)

    @property
    def subunits_included(self) -> Optional[gpd.GeoDataFrame]:
        if self.subunits is None or "in_extent" not in self.subunits.columns:
            return None
        return self.subunits[self.subunits["in_extent"]].reset_index(drop=True)

    @property
    def subunits_excluded(self) -> Optional[gpd.GeoDataFrame]:
        if self.subunits is None or "in_extent" not in self.subunits.columns:
            return None
        return self.subunits[~self.subunits["in_extent"]].reset_index(drop=True)

    @property
    def has_subunit_study_area(self) -> bool:
        return self.provenance.get("study_area", {}).get("kind") == "subunits_union_in_extent"


def load_territory(site: str, root: Path = ROOT_DEFAULT) -> Territory:
    """Build a site's Territory from config/sites.yaml. Loads only boundary
    files (never a site's buildings layer) — see module docstring."""
    site = normalize_site_key(site)
    sites_cfg = load_sites_config()
    if site not in sites_cfg:
        raise KeyError(f"unknown site {site!r} — config/sites.yaml has: {sorted(sites_cfg)}")
    cfg = sites_cfg[site]

    data_extent_gdf = load_data_extent(cfg, root)
    data_extent_geom = _union(data_extent_gdf.geometry)

    favelas = _load_favelas(root)
    citywide_gdf, citywide_method = match_citywide(favelas, cfg["citywide_rule"]["target"])
    citywide_gdf = citywide_gdf.reset_index(drop=True)
    expected_method = cfg["citywide_rule"]["method"]
    matched_tier = "complexo_exact" if citywide_method == "complexo_exact" else (
        "nome_exact" if citywide_method.startswith("nome_exact") else citywide_method)
    if matched_tier != expected_method:
        raise ValueError(
            f"{site}: config/sites.yaml declares citywide_rule.method="
            f"{expected_method!r} but match_favela_group actually used "
            f"{citywide_method!r} — the registry is stale, fix it")

    subunits, subunits_prov = load_subunits(cfg, root)
    study_area_geom, study_area_prov, subunits = build_study_area(
        cfg["study_area"], data_extent_geom, citywide_gdf, subunits, root)

    candidates = {}
    for cand_cfg in cfg.get("study_area_candidates") or []:
        cand_geom, cand_prov, _ = build_study_area(cand_cfg, data_extent_geom, citywide_gdf, subunits, root)
        cand_prov_file = cand_cfg.get("provenance_file")
        cand_source_prov = {}
        if cand_prov_file and (root / "data" / cand_prov_file).exists():
            cand_source_prov = json.loads((root / "data" / cand_prov_file).read_text(encoding="utf-8"))
        candidates[cand_cfg["id"]] = {
            "geometry": cand_geom,
            "label": cand_cfg.get("label", cand_cfg["id"]),
            "status": cand_cfg.get("status", "candidate"),
            "area_m2": float(cand_geom.area),
            "build": cand_prov,
            "source_provenance": cand_source_prov,
        }

    rot_raw = cfg.get("display_rotation_deg", "auto")
    rot = rotation_deg(study_area_geom) if rot_raw == "auto" else int(rot_raw)

    provenance = {
        "site": site,
        "display_name": cfg["display_name"],
        "data_extent": {"file": cfg["data_extent"], "area_m2": float(data_extent_geom.area)},
        "citywide_rule": cfg["citywide_rule"],
        "citywide": {
            "n_polygons": int(len(citywide_gdf)),
            "area_m2": float(_union(citywide_gdf.geometry).area) if len(citywide_gdf) else 0.0,
            "match_method": citywide_method,
        },
        "study_area": {**study_area_prov, "area_m2": float(study_area_geom.area)},
        "study_area_candidates": {
            k: {"label": v["label"], "status": v["status"], "area_m2": v["area_m2"], "build": v["build"]}
            for k, v in candidates.items()
        },
        "subunits": subunits_prov,
        "display_rotation_deg": rot,
        "definition_note": cfg.get("definition_note"),
    }

    return Territory(
        site=site, root=root, display_name=cfg["display_name"],
        data_extent=data_extent_geom, study_area=study_area_geom,
        citywide=citywide_gdf, citywide_method=citywide_method,
        subunits=subunits, study_area_candidates=candidates,
        display_rotation_deg=rot, provenance=provenance,
    )


def has_subunit_study_area(site: str) -> bool:
    """Registry-driven replacement for `if site == "maré"`: true exactly
    for sites whose study area is a subunit union restricted to the data
    extent (today, only Maré) — the condition every `if site in ("maré",
    "mare")` branch in the dashboard builders actually meant."""
    site = normalize_site_key(site)
    cfg = load_sites_config()
    if site not in cfg:
        return False
    return cfg[site]["study_area"]["kind"] == "subunits_union_in_extent"
