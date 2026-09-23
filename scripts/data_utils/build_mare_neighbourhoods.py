"""Build the Maré neighbourhood layer: the 16 communities Redes da Maré recognises,
each assembled from the Prefeitura's own SABREN polygons.

No single published layer carries all 16. The IPP favela limits omit the
conjuntos habitacionais, and IBGE 2022 derives from the same cadastre (survey:
docs/research/mare_neighbourhood_sources.md). SABREN publishes the conjuntos as
a separate service, so the union of the two covers every community. The
name→polygon crosswalk below is the only hand-made part; each row states
whether the match is exact or inferred so the PI can check the inferred ones.

    python scripts/data_utils/build_mare_neighbourhoods.py [--refresh]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARE = PROJECT_ROOT / "data" / "maré"
BAIRRO = MARE / "raw" / "mare_boundary.shp"
OUT = MARE / "neighbourhoods.gpkg"
PROVENANCE = MARE / "neighbourhoods_provenance.json"

SABREN = "https://pgeo3.rio.rj.gov.br/arcgis/rest/services/SABREN"
SOURCES = {
    "favelas_2022": {
        "url": f"{SABREN}/Limites_de_Favelas/FeatureServer/13",
        "title": "Limites de Favelas 2022 (SABREN)",
        "id_field": "cod_favela", "name_field": "nome",
        "cache": MARE / "raw" / "sabren_favelas_2022.geojson",
    },
    "conjuntos": {
        "url": f"{SABREN}/Conjuntos_Habitacionais/FeatureServer/0",
        "title": "Conjuntos Habitacionais (SABREN)",
        "id_field": "codigo_con", "name_field": "conjunto",
        "cache": MARE / "raw" / "sabren_conjuntos.geojson",
    },
}
PUBLISHER = "Instituto Pereira Passos / Secretaria Municipal de Habitação, Prefeitura do Rio de Janeiro"
COMMUNITY_LIST_SOURCE = "Redes da Maré, 'Sobre a Maré' (https://www.redesdamare.org.br) and Guia de Ruas da Maré (2014)"

# (community, [(source, id)], match, note). "inferred" = the SABREN name differs
# from the community name; the PI confirms these.
CROSSWALK = [
    ("Morro do Timbau", [("favelas_2022", 108)], "inferred", "SABREN name 'Timbau'"),
    ("Baixa do Sapateiro", [("favelas_2022", 84)], "exact", ""),
    ("Marcílio Dias", [("favelas_2022", 115)], "inferred", "SABREN name 'Centro Social Marcílio Dias'; lies in bairro Penha Circular, outside the official Maré bairro"),
    ("Parque Maré", [("favelas_2022", 90)], "exact", ""),
    ("Parque Rubens Vaz", [("favelas_2022", 104)], "exact", ""),
    ("Parque Roquete Pinto", [("favelas_2022", 103)], "exact", ""),
    ("Parque União", [("favelas_2022", 105)], "exact", ""),
    ("Nova Holanda", [("favelas_2022", 96)], "exact", ""),
    ("Praia de Ramos", [("favelas_2022", 106)], "inferred", "SABREN name 'Ramos', complexo Parque Roquete Pinto"),
    ("Conjunto Esperança", [("conjuntos", 73)], "exact", ""),
    ("Vila do João", [("conjuntos", 72)], "exact", "SABREN name 'Conjunto Vila do João'"),
    ("Vila do Pinheiro", [("conjuntos", 71), ("favelas_2022", 1075)], "inferred", "conjunto 'Conjunto Vila Pinheiros' + favela 'Comunidade Vila do Pinheiro'"),
    ("Conjunto Pinheiros", [("conjuntos", 10)], "exact", ""),
    ("Bento Ribeiro Dantas", [("conjuntos", 12)], "exact", "SABREN name 'Conjunto Bento Ribeiro Dantas'"),
    ("Nova Maré", [("conjuntos", 40)], "exact", "SABREN name 'Conjunto Nova Maré'"),
    ("Salsa e Merengue / Novo Pinheiro", [("conjuntos", 13), ("favelas_2022", 1160)], "inferred", "conjunto 'Conjunto Salsa e Merengue' + favela 'Salsa e Merengue'"),
]
SEARCH_PAD_DEG = 0.01


def _query(url: str, **params) -> bytes:
    q = urllib.parse.urlencode(dict(outFields="*", outSR=31983, f="geojson", **params))
    with urllib.request.urlopen(f"{url}/query?{q}", timeout=120) as r:
        raw = r.read()
    if b'"error"' in raw[:200]:
        raise RuntimeError(f"{url}: {raw[:300]!r}")
    return raw


def fetch(source: str, bbox4326, refresh: bool) -> gpd.GeoDataFrame:
    """Every polygon in the padded bairro envelope, plus the crosswalk's own ids
    wherever they lie (Marcílio Dias sits outside the envelope)."""
    spec = SOURCES[source]
    by_id = spec["cache"].with_name(spec["cache"].stem + "_crosswalk_ids.geojson")
    if refresh or not spec["cache"].exists() or not by_id.exists():
        xmin, ymin, xmax, ymax = bbox4326
        spec["cache"].write_bytes(_query(
            spec["url"], where="1=1", geometry=f"{xmin},{ymin},{xmax},{ymax}",
            geometryType="esriGeometryEnvelope", inSR=4326, spatialRel="esriSpatialRelIntersects"))
        ids = sorted({i for _, parts, _, _ in CROSSWALK for s, i in parts if s == source})
        by_id.write_bytes(_query(spec["url"], where=f"{spec['id_field']} IN ({','.join(map(str, ids))})"))
    # ArcGIS labels outSR geojson as 4326 regardless of the requested SR.
    frames = [gpd.read_file(p).set_crs(31983, allow_override=True) for p in (spec["cache"], by_id)]
    both = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(both.drop_duplicates(subset="objectid"), crs=31983)


def build(refresh: bool = False) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    bairro = gpd.read_file(BAIRRO).to_crs(31983)
    bairro_geom = bairro.union_all()
    x0, y0, x1, y1 = bairro.to_crs(4326).total_bounds
    bbox = (x0 - SEARCH_PAD_DEG, y0 - SEARCH_PAD_DEG, x1 + SEARCH_PAD_DEG, y1 + SEARCH_PAD_DEG)
    layers = {s: fetch(s, bbox, refresh) for s in SOURCES}

    rows, used = [], {s: set() for s in SOURCES}
    for community, parts, match, note in CROSSWALK:
        geoms, names = [], []
        for source, pid in parts:
            spec, lay = SOURCES[source], layers[source]
            hit = lay[lay[spec["id_field"]].astype(int) == pid]
            if len(hit) != 1:
                raise ValueError(f"{community}: {source} id {pid} matched {len(hit)} polygons")
            geoms.append(hit.geometry.iloc[0])
            names.append(f"{source}:{pid} '{hit[spec['name_field']].iloc[0]}'")
            used[source].add(pid)
        geom = gpd.GeoSeries(geoms, crs=31983).union_all()
        rows.append({
            "community": community, "match": match, "note": note,
            "source_parts": "; ".join(names), "area_m2": geom.area,
            "share_in_official_bairro": geom.intersection(bairro_geom).area / geom.area,
            "geometry": geom,
        })
    communities = gpd.GeoDataFrame(rows, crs=31983)

    fav = layers["favelas_2022"]
    rest = fav[fav.intersects(bairro_geom) & ~fav["cod_favela"].astype(int).isin(used["favelas_2022"])]
    rest = rest[(rest.intersection(bairro_geom).area / rest.area) > 0.5]
    other = gpd.GeoDataFrame({
        "sabren_name": rest["nome"].to_numpy(),
        "source_parts": [f"favelas_2022:{int(i)}" for i in rest["cod_favela"]],
        "area_m2": rest.area.to_numpy(),
    }, geometry=rest.geometry.to_numpy(), crs=31983)

    union = communities.union_all()
    overlaps = [
        (a.community, b.community, round(a.geometry.intersection(b.geometry).area, 1))
        for i, a in communities.iterrows() for j, b in communities.iterrows()
        if i < j and a.geometry.intersection(b.geometry).area > 1.0
    ]
    qa = {
        "n_communities": len(communities),
        "n_inferred": int((communities["match"] == "inferred").sum()),
        "union_area_m2": union.area,
        "official_bairro_area_m2": bairro_geom.area,
        "union_share_of_bairro": union.intersection(bairro_geom).area / bairro_geom.area,
        "union_area_outside_bairro_m2": union.difference(bairro_geom).area,
        "pairwise_overlaps_m2": overlaps,
        "n_other_sabren_polygons_in_bairro": len(other),
    }
    return communities, other, qa


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-query SABREN instead of the cached responses")
    args = ap.parse_args()
    communities, other, qa = build(args.refresh)
    OUT.unlink(missing_ok=True)
    communities.to_file(OUT, layer="communities", driver="GPKG")
    other.to_file(OUT, layer="other_sabren_polygons", driver="GPKG")
    PROVENANCE.write_text(json.dumps({
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "publisher": PUBLISHER,
        "sources": {k: {"title": v["title"], "url": v["url"], "cache": str(v["cache"].relative_to(PROJECT_ROOT))}
                    for k, v in SOURCES.items()},
        "community_list": COMMUNITY_LIST_SOURCE,
        "crosswalk": [dict(community=c, parts=[f"{s}:{i}" for s, i in p], match=m, note=n) for c, p, m, n in CROSSWALK],
        "qa": qa,
    }, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(qa, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
