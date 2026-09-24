# Boundary check: our 16-community Maré layer vs. Redes da Maré's own delineation

Researched 2026-09-24. Read-only web research; compares
`data/maré/neighbourhoods.gpkg` (layer `communities`, 16 polygons built from
SABREN favela-limits-2022 + conjuntos-habitacionais, see
`data/maré/neighbourhoods_provenance.json`) against maps published by Redes
da Maré and its printed street guide. No coordinates are claimed anywhere
below — printed/photo maps give position and adjacency, not geometry; every
"agree/differs" call is a visual read, flagged where uncertain.

## Sources

| # | Source | URL | Status | Local file |
|---|---|---|---|---|
| 1 | Redes da Maré, "Sobre a Maré" | `https://www.redesdamare.org.br/br/quemsomos/sobre` | VERIFIED (fetched) | — (HTML; chronological 16-name list, no map embedded as text) |
| 2 | "Mapa da Maré" aerial-photo map, embedded on the Sobre page | `https://www.redesdamare.org.br/media/textos/mapa-mare-bairros-sobre.jpg` | VERIFIED (HTTP 200, downloaded, viewed) | `mapa-mare-bairros-sobre.jpg` |
| 3 | Museu da Maré / Arquivo, "Favelas da Maré (2022)" cartographic map, by Luiz Augusto Ferreira Lourenço | `https://arquivomuseudamare.org/acervo-cartografico/mapa-das-favelas-da-mare/` → image `https://arquivomuseudamare.org/wp-content/uploads/tainacan-items/8849/40066/adov_car_0145-scaled.jpg` | VERIFIED (HTTP 200, downloaded, viewed) | `museu_mare_2022_map.jpg` |
| 4 | Guia de Ruas da Maré 2014 (Redes da Maré + Observatório de Favelas, w/ the 16 residents' associations) | `https://www.redesdamare.org.br/media/downloads/arquivos/Guia_RuasMare2014.pdf` | VERIFIED (HTTP 200, 5.9 MB, downloaded, 133 pages, viewed pp. 1–10, 16–21, 30–31, 36–37, 66–67, 70–71, 74–77, 95–97) | `Guia_RuasMare2014.pdf` |
| 5 | Guia de Ruas p.19 — "Comunidades da Maré" index map (17 numbered, boundary-outlined polygons over an aerial photo) | same PDF, printed page 19 | VERIFIED (viewed directly) | (within `Guia_RuasMare2014.pdf`) |
| 6 | Observatório de Favelas mirror of the Guia (2012 first edition) | `https://observatoriodefavelas.org.br/wp-content/uploads/2014/01/GuiaMare_Web.pdf` | UNVERIFIED (not fetched this session; not needed once #4/#5 were in hand) | — |
| 7 | RioOnWatch, "Um Mergulho na História: O Nascimento e Formação do Complexo da Maré" | `https://rioonwatch.org.br/?p=23997` | UNVERIFIED (found via search, not fetched) | — |
| 8 | Wikifavelas, "Complexo da Maré" | `https://wikifavelas.com.br/index.php/Complexo_da_Mar%C3%A9` | UNVERIFIED (found via search, not fetched) | — |

Source #5 (Guia de Ruas p.19) is the single most authoritative item found: an
official, boundary-outlined map produced by Redes da Maré + Observatório de
Favelas with the 16 residents' associations, numbering 17 zones (the 16
communities plus Marcílio Dias shown separately). It is the primary basis
for the table below; #2 and #3 corroborate it independently.

## Per-community table

"Position/adjacency" judged against our rendered map
(`outputs/maré/territory/mare_territory_map.png`) and the Guia p.19 index
map, both oriented with Praia de Ramos at the NW tip and Conjunto Esperança
at the SE tip.

| Our community | Redes name(s) seen | List membership | Position/adjacency | Extent |
|---|---|---|---|---|
| Morro do Timbau | "Morro do Timbau" (all sources) | agree | agree — between Baixa do Sapateiro (N) and Bento Ribeiro Dantas/Vila do Pinheiro (S/E) in both our map and p.19 (#08) | cannot tell (no vector source) |
| Baixa do Sapateiro | "Baixa do Sapateiro" (all sources) | agree | agree — between Parque Maré (N) and Morro do Timbau (S), matches p.19 (#09) | cannot tell |
| Marcílio Dias | "Marcílio Dias" (all sources) | agree | **differs in kind, as expected**: p.19 draws it as a detached inset box, separate from the other 16 polygons, with no connecting geometry; guide p.74 states in prose it "fica geograficamente distante das demais comunidades da Maré – separada por uma área militar" | cannot tell shape; separateness from the main bairro is textually and cartographically confirmed |
| Parque Maré | "Parque Maré" (all sources) | agree | agree — between Nova Holanda (N) and Baixa do Sapateiro (S), matches p.19 (#11) | cannot tell |
| Parque Rubens Vaz | "Parque Rubens Vaz" (all sources) | agree | agree — small sliver between Parque União and Nova Holanda in both our map and p.19 (#13) | cannot tell |
| Parque Roquete Pinto | "Roquete Pinto" (Redes always drops "Parque") | agree, minor name variant | agree — NW tip, adjacent to Praia de Ramos, matches p.19 (#15) | cannot tell |
| Parque União | "Parque União" (all sources) | agree | agree — matches p.19 (#14), west edge of the main mass | cannot tell |
| Nova Holanda | "Nova Holanda" (all sources) | agree | agree — matches p.19 (#12) | cannot tell |
| Praia de Ramos | "Praia de Ramos" (all sources) | agree | agree — northernmost, adjacent to Roquete Pinto, matches p.19 (#16). (The aerial "Mapa da Maré" image alone was ambiguous about whether Praia de Ramos has its own polygon distinct from Roquete Pinto; p.19 resolves this — they are two separately drawn, adjacent polygons.) | cannot tell |
| Conjunto Esperança | "Conjunto Esperança" (all sources) | agree | agree — southernmost tip, adjacent to Vila do João, matches p.19 (#01 next to #02) | cannot tell |
| Vila do João | "Vila do João" (all sources) | agree | agree — matches p.19 (#02), between Conjunto Esperança and Novo Pinheiro/Vila do Pinheiro | cannot tell |
| Vila do Pinheiro | "Vila Pinheiro" / "Vila Pinheiro (Parque Ecológico)" (p.19 splits it into two numbered zones, #04 and #05) | agree on name | agree on general location (east side, largest area) but p.19 **subdivides** it | **differs: our single merged polygon corresponds to two zones on Redes' own map** — a main "Vila Pinheiro" area and a "Vila Pinheiro (Parque Ecológico)" area built on the former Ilha dos Pinheiros (guide p.30: "É a única favela carioca que possui um parque ecológico em seu interior"). Cannot tell from the picture whether our two SABREN source parts (`conjuntos:71`, `favelas_2022:1075`) line up 1:1 with Redes' main/ecológico split |
| Conjunto Pinheiros | "Conjunto Pinheiro" (all sources) | agree | agree — matches p.19 (#03), between Bento Ribeiro Dantas and Vila do Pinheiro | cannot tell |
| Bento Ribeiro Dantas | "Bento Ribeiro Dantas" (all sources) | agree | agree — matches p.19 (#07), east of Timbau | cannot tell |
| Nova Maré | "Nova Maré" (all sources) | agree | agree — matches p.19 (#10), small area between Baixa do Sapateiro/Parque Maré and Timbau, near Avenida Brasil | cannot tell |
| Salsa e Merengue / Novo Pinheiro | "Novo Pinheiro" (official/map label); "Salsa e Merengue" (popular name, explained explicitly in guide p.36) | agree, exact | agree — matches p.19 (#06), adjacent to Vila do Pinheiro | cannot tell |

**Community count / list:** all 16 of our communities are present as named
entities across the Redes sources; the Guia's own "16 favelas" list (p.17)
and 17-zone index map (p.19, 16 + a detached Marcílio Dias inset) name
exactly the same 16 places our layer uses, with only the minor spelling
variants noted above ("Roquete Pinto" not "Parque Roquete Pinto"; "Novo
Pinheiro"/"Vila Pinheiro" not "Vila do Pinheiro" — Redes is inconsistent
between "do"/no-article forms across its own documents).

## The one finding that matters most: coverage, not naming or order

Every name, and essentially every adjacency, in our layer agrees with
Redes da Maré's own map. The real difference is **extent**. Our provenance
QA already recorded that the union of our 16 polygons covers only ~47.8%
of the official Maré bairro area (`neighbourhoods_provenance.json`,
`qa.union_share_of_bairro`), because SABREN's favela/conjunto limits are
tight building-footprint polygons with real gaps between them (visible as
white space in `mare_territory_map.png`, e.g. between the Ramos/Roquete
Pinto pair and the Parque União cluster).

The Guia de Ruas 2014 p.19 index map — Redes da Maré's own official,
boundary-outlined territorial map — draws the 16 communities (plus the
detached Marcílio Dias) as one essentially continuous, wall-to-wall
partition of the whole urbanized strip from Praia de Ramos to Conjunto
Esperança: the yellow community-boundary lines run edge to edge across the
full aerial photo, with no unassigned area visible between neighbours. That
is a materially different conception of "the 16 communities" than our
building-footprint layer: Redes treats every part of the bairro as
belonging to one of the 16; our layer treats large parts of the bairro as
belonging to none of them.

## Verdict on the 5 inferred matches

1. **Timbau → Morro do Timbau** — CONFIRMED. Identical name in every Redes
   source (aerial map, Museu map, guide index and narrative).
2. **Ramos → Praia de Ramos** — CONFIRMED. "Praia de Ramos" is Redes' own
   name for this community everywhere it appears; the Guia p.19 index map
   shows it as a polygon (#16) distinct from but adjacent to Roquete Pinto
   (#15), matching our map's placement exactly.
3. **Centro Social Marcílio Dias → Marcílio Dias** — CONFIRMED as a named
   community, and its geographic separateness — the reason it can't be
   pinned down spatially relative to the other 15 — is independently and
   explicitly corroborated: guide p.74 states it in prose ("separada por
   uma área militar"), and the p.19 index map draws it as a detached inset,
   not connected to the other 16 polygons. This is the best-supported of
   the five, even though (like all five) exact polygon congruence can't be
   confirmed from a picture.
4. **Vila do Pinheiro = conjunto Vila Pinheiros + favela Comunidade Vila do
   Pinheiro** — PARTIALLY CONFIRMED, with a real open question. The name
   match is solid, but Redes' own p.19 map splits this territory into two
   numbered zones (main "Vila Pinheiro" + "Vila Pinheiro (Parque
   Ecológico)", the latter tied to the former Ilha dos Pinheiros). Whether
   our two SABREN source parts (`conjuntos:71` conjunto + `favelas_2022:1075`
   favela) correspond to that same main/ecológico split, or to some other
   division, cannot be told from a photo — worth a follow-up check against
   the actual SABREN polygon shapes.
5. **Salsa e Merengue = conjunto + favela polygon** — CONFIRMED, the
   best-documented of the five. Guia p.36 states explicitly, in prose, that
   the official name is "Novo Pinheiro" and that residents call it "Salsa e
   Merengue" because of the house paint colours — an exact match to our
   dual-name community, drawn as a single polygon (#06) on the p.19 index
   map.

## Ranked boundary edits the evidence supports

1. **[High] Coverage gap, not a positional error.** Evidence: Guia de Ruas
   2014 p.19 index map (wall-to-wall 16-community partition) vs.
   `neighbourhoods_provenance.json` (`union_share_of_bairro` ≈ 0.478) and
   the visible gaps in `mare_territory_map.png`. If the layer's intended
   use is "which community is this part of Maré in," the current
   SABREN-footprint polygons under-represent every community's true
   extent, not just one. Two options, not mutually exclusive: (a) relabel
   the current layer as a footprint/extent layer rather than a community
   boundary layer, or (b) digitize a second, wall-to-wall layer from Guia
   p.19 (georeferencing the aerial photo — no vector source exists) to use
   wherever "which community" coverage of the whole bairro is required.
2. **[Medium] Vila do Pinheiro's internal split.** Evidence: Guia p.19
   (zones #04/#05) and p.30 (Ilha dos Pinheiros / Parque Ecológico text).
   Check whether SABREN `favelas_2022:1075` ("Comunidade Vila do Pinheiro")
   is the ecológico sub-area; if so, consider exposing it as a sub-polygon
   or attribute rather than folding it silently into the merged community.
3. **[Low] Display-name variants.** Evidence: every Redes source drops
   "Parque" from "Parque Roquete Pinto" → "Roquete Pinto", and prefers
   "Novo Pinheiro"/"Vila Pinheiro" over "Vila do Pinheiro" in some places.
   No geometry change; consider adding these as alternate/short display
   names if the technical report cites Redes as the naming authority.
4. **[Informational, no edit] Marcílio Dias.** Upgrade the provenance
   note's confidence: it is not just "inferred from the SABREN name" but
   now also textually and cartographically corroborated by Redes da Maré
   itself (Guia p.19, p.74) as geographically detached from the rest of
   Maré. Worth citing in `neighbourhoods_provenance.json`'s note field.

No evidence in any source examined contradicts an adjacency, a name, or a
north–south ordering in our current layer. The Roquete Pinto/Avenida Brasil
edge areas were not separately resolvable at print resolution beyond what's
stated above (all "cannot tell" extent cells).
