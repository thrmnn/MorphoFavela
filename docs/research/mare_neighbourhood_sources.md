# Sources for the 16-community delineation of Complexo da Maré

Researched 2026-09-23. Read-only web research; no data committed to the repo.

## 0. The 16 communities — confirmed list

Cross-checked against Redes da Maré's own "Sobre a Maré" page (community-produced,
authoritative for naming) and Wikipédia PT (which cites the same chronology):

1. Morro do Timbau (1940)
2. Baixa do Sapateiro (1947)
3. Marcílio Dias (1948)
4. Parque Maré (1953)
5. Parque Rubens Vaz (1954/1961 — sources disagree by a few years)
6. Parque Roquete Pinto (1955)
7. Parque União (1961)
8. Nova Holanda (1962)
9. Praia de Ramos (1962)
10. Conjunto Esperança (1982)
11. Vila do João (1982)
12. Vila do Pinheiro / Vila dos Pinheiros (1983/1984)
13. Conjunto Pinheiros (1989)
14. Conjunto (Bento Ribeiro) Dantas (1992)
15. Nova Maré (1996)
16. Novo Pinheiro / Salsa e Merengue (2000)

This matches the list given in the task. The user's list is correct; Redes da Maré's
own page does not print it as a single clean list of 16 names — it narrates them
chronologically with founding years — so any citation of "the 16" should cite this
chronological passage, not a bullet list (none exists on the site).

## 1. Redes da Maré — "Sobre a Maré" / "Censo Maré"

- Publisher: Redes da Maré (community-based NGO, est. 2007, partnered with
  Observatório de Favelas)
- URLs: `https://www.redesdamare.org.br/br/quemsomos/sobre` (VERIFIED, fetched,
  contains the chronological 16-community list quoted above),
  `https://www.redesdamare.org.br/br/info/12/censo-mare` (VERIFIED, fetched)
- Geometry: **none downloadable.** Only narrative HTML and PDF reports (Censo
  Populacional da Maré 2019 PDF, Censo de Empreendimentos 2014 PDF). No shapefile,
  GeoJSON or KML linked anywhere on these pages.
- Coverage: names/narrative for all 16 communities.
- Reliability: community-produced, the closest thing to a canonical source for the
  16-community naming and boundaries-by-consensus (built with the 16 local
  residents' associations), but not a GIS product.

## 2. Guia de Ruas da Maré (2012, 2014 editions)

- Publisher: Redes da Maré + Observatório de Favelas, with the 16 residents'
  associations
- URL: `https://www.redesdamare.org.br/media/downloads/arquivos/Guia_RuasMare2014.pdf`
  — VERIFIED (HTTP 200, `Content-Type: application/pdf`, ~9 MB by header)
- Also mirrored: `https://observatoriodefavelas.org.br/wp-content/uploads/2014/01/GuiaMare_Web.pdf`
  (2012 first edition) — UNVERIFIED (not HEAD-tested this session)
- Geometry: **printed/PDF map only**, no vector layer. Per-community street maps +
  street index with CEPs; first cartographic mapping of the whole 16-community area
  (815 logradouros, most previously absent from official bases).
- Reliability: community-produced, high on-the-ground accuracy for street layout and
  community boundaries as locally understood; not machine-readable.

## 3. Data.Rio / IPP — SABREN "Limite Favelas" (2019 and 2022 editions)

- Publisher: Instituto Pereira Passos (IPP), Prefeitura do Rio de Janeiro (SABREN =
  Sistema de Assentamentos de Baixa Renda)
- Feature services (both VERIFIED live via ArcGIS REST queries this session):
  - 2019: `https://pgeo3.rio.rj.gov.br/arcgis/rest/services/Habitacao/Favelas_Urbanizacao/FeatureServer/0`
    — 1,072 features citywide
  - 2022: `https://pgeo3.rio.rj.gov.br/arcgis/rest/services/SABREN/Limites_de_Favelas/FeatureServer/13`
    — also 1,072 features citywide (same base geometry, updated attribute year);
    licence stated on the ArcGIS item as CC-BY 4.0
  - Hub pages: `https://www.data.rio/datasets/limite-favelas-2019`,
    `https://www.data.rio/datasets/PCRJ::limite-favelas-2022/about` (both VERIFIED
    HTTP 200)
- Geometry: downloadable (Shapefile/GeoJSON/CSV/KML via the ArcGIS Hub "Download"
  button; REST API also supports arbitrary format export), CRS is Web Mercator
  (EPSG:3857) on the service, reprojectable on export.
- **Coverage confirmed by direct query** (`bairro='Maré'`, 2022 layer): only 20
  polygons, all favela-type. Present: Timbau, Baixa do Sapateiro, Nova Holanda,
  Parque Maré, Parque Rubens Vaz, Parque União, Parque Roquete Pinto, Ramos, Vila do
  Pinheiro, + several micro-parcels (Avenida Canal, Pata Choca, Suave, etc.).
  **Absent:** Vila do João, Conjunto Esperança, Conjunto Pinheiros, Bento Ribeiro
  Dantas, Nova Maré, Salsa e Merengue, Marcílio Dias (as a named settlement). This
  reproduces exactly the ~26%-of-bairro gap already known from the 2019 layer — the
  2022 edition does **not** fix it.
- Reliability: official/authoritative for what it covers, but confirmed structurally
  incomplete for Maré's conjuntos habitacionais.

## 4. IBGE Censo Demográfico 2022 — "Favelas e Comunidades Urbanas" (FCU) polygons

- Publisher: IBGE (national statistics institute)
- URL tested and downloaded: `https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Favelas_e_comunidades_urbanas_Resultados_do_universo/arquivos_vetoriais/poligonos_FCUs_shp.zip`
  — VERIFIED (HTTP 200, 9,090,709 bytes, `Content-Type: application/zip`,
  `Last-Modified: 2024-11-08`)
- Downloaded and inspected with `ogrinfo` (file `qg_2022_670_fcu_agreg.shp`):
  - CRS: SIRGAS 2000 geographic, EPSG:4674
  - 12,348 polygon features nationwide (matches IBGE's press-release total),
    813 within `nm_mun='Rio de Janeiro'`
  - Licence: IBGE open data (CC-BY per IBGE's standard geociências terms;
    not independently re-verified this session)
- **Coverage check (important, new finding):** filtering Rio features by name found
  Nova Holanda, Parque Maré, Parque Roquete Pinto, Parque Rubens Vaz, Parque União,
  Ramos, Timbau, Baixa do Sapateiro, Comunidade Vila do Pinheiro — but **zero**
  matches for "Vila do João", "Conjunto Esperança", "Bento Ribeiro Dantas", "Nova
  Maré", "Salsa e Merengue", "Novo Pinheiro", "Conjunto Pinheiros", or "Marcílio
  Dias" (only a same-name social facility, "Centro Social Marcílio Dias", exists).
  **IBGE's 2022 census polygon layer has the same conjuntos-habitacionais gap as
  SABREN** — it appears to reuse/derive from the municipal favela cadastre rather
  than an independent field delineation. This is a genuinely new and important
  result for the project: IBGE 2022 does not solve the problem either.
- Not checked this session (flagged, not ruled out): `FCUs_nao_setorizadas_shp_20260410.zip`
  and `concentracoes_urbanas_shp.zip` in the same FTP directory — different products,
  worth a follow-up look before concluding no IBGE product covers the conjuntos.

## 5. OpenStreetMap

- Overpass API: **UNVERIFIED / unreachable this session** — both
  `overpass-api.de/api/interpreter` and `overpass.kumi.systems` /
  `overpass.private.coffee` mirrors returned `406 Not Acceptable` or timed out from
  this sandbox for every query tried (including a trivial one-node test), so no
  Overpass query could be completed.
- Fell back to the Nominatim search API (`nominatim.openstreetmap.org`, backed by
  the same OSM database), which partially answers the same question — VERIFIED live
  responses for each name queried:
  - Maré bairro itself: `relation/5520335`, `boundary=administrative` — a real
    polygon, downloadable via the OSM API/Overpass/any OSM extract.
  - Of the 16 communities, spot-checked 9: **most exist only as point nodes**
    tagged `place=neighbourhood` (Vila do João, Salsa e Merengue, Conjunto
    Esperança, Morro do Timbau, Bento Ribeiro Dantas — all `osm_type: node`, no
    polygon). A minority exist as polygon **ways**: Parque União (`way/87102030`),
    Nova Maré (`way/837979119`), and Parque Roquete Pinto as a
    `landuse=residential` way (`way/87102033`, tagged as a street block, not a
    neighbourhood boundary).
- Coverage: names for all/most of the 16 exist as OSM place points; polygon
  (boundary) geometry exists for only a handful, inconsistently tagged.
- Reliability: crowd-sourced, editable, currently incomplete and tag-inconsistent
  for Maré's internal communities — usable as a scaffold (all names are at least
  geocoded) but not as-is a complete polygon source. Could be improved by manual
  OSM editing keyed off the Guia de Ruas Maré maps, but that is original work, not
  an existing citable source.

## 6. Casa Fluminense / Observatório de Favelas / academic theses

- Searched; did not find a specific, confirmed-downloadable shapefile of the
  16-community delineation from either organization's site in this session.
- One academic lead surfaced but not verified for open geometry: "Mapeamento do
  processo de evolução urbana do Complexo da Maré, Rio de Janeiro," Revista de
  Morfologia Urbana (`revistademorfologiaurbana.org/index.php/rmu/article/view/336`)
  — UNVERIFIED, not fetched this session; likely contains maps/figures rather than
  an open vector layer, would need a direct check.
- Not ruled out as a fruitful direction, just not confirmed.

---

## Ranked recommendation (≤25 lines)

1. **Best authoritative naming/definition:** Redes da Maré "Sobre a Maré" page +
   Guia de Ruas da Maré 2014 PDF. Community-produced with the 16 residents'
   associations, the closest thing to ground truth for which 16 communities exist
   and their boundaries-by-local-consensus. No GIS geometry — PDF/HTML only.
2. **Best downloadable geometry (with a caveat):** Data.Rio/SABREN "Limite Favelas
   2022" ArcGIS FeatureServer (`SABREN/Limites_de_Favelas/13`) — official,
   CC-BY 4.0, live REST endpoint, exportable to Shapefile/GeoJSON. But confirmed by
   direct query to cover only 20 favela-type polygons in Maré (~26% of the bairro),
   identical gap to the 2019 layer already in hand. Does not add anything new.
3. **New finding, same gap:** IBGE Censo 2022 "Favelas e Comunidades Urbanas"
   national polygon shapefile (SIRGAS2000/EPSG:4674, verified download+inspect) also
   **omits every one of Maré's conjuntos habitacionais** (Vila do João, Conjunto
   Esperança, Bento Ribeiro Dantas, Nova Maré, Conjunto Pinheiros, Salsa e Merengue,
   Marcílio Dias). It is not an independent fix for the SABREN gap — treat as
   confirming, not solving, the known problem.
4. **Fallback / only path to full 16-community polygons found so far:** manually
   digitize the missing conjuntos from the Guia de Ruas da Maré 2014 PDF maps
   (verified, real, downloadable PDF with per-community street maps), using OSM
   building footprints as a modern basemap and the Redes da Maré community list as
   the naming authority. No ready-made complete polygon source exists in any
   candidate checked this session (SABREN, IBGE, OSM, Redes da Maré, Casa
   Fluminense/Observatório de Favelas).
5. Not fully explored, worth a follow-up: IBGE's other two zips in the same FTP
   folder (`FCUs_nao_setorizadas`, `concentracoes_urbanas`), and the academic
   Revista de Morfologia Urbana mapping paper.

## Addendum 2026-09-23 — the gap closes with SABREN's conjuntos layer

The survey above missed one service in the same SABREN folder:
`https://pgeo3.rio.rj.gov.br/arcgis/rest/services/SABREN/Conjuntos_Habitacionais/FeatureServer/0`
(IPP / Secretaria Municipal de Habitação; VERIFIED by query on 2026-09-23). It carries 7 polygons
in the Maré bairro: Conjunto Pinheiros, Bento Ribeiro Dantas, Salsa e Merengue, Nova Maré,
Vila Pinheiros, Vila do João, Esperança. Together with the 2022 favela limits (layer 13) they
cover all 16 communities, with Marcílio Dias taken from the favela layer's "Centro Social
Marcílio Dias" (bairro Penha Circular, outside the official Maré bairro).
`scripts/data_utils/build_mare_neighbourhoods.py` assembles the layer; the crosswalk and QA are
in `data/maré/neighbourhoods_provenance.json`. Manual digitising from the Guia de Ruas is not needed.
