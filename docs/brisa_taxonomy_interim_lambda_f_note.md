# Definitive interim λf ventilation taxonomy — one-paragraph note

**Status:** interim / relative. Pending CFD-ACH calibration + LMA matching;
this remains a geometric pre-screen, not a calibrated absolute flag.

Under the corrected (cell-clipped) λ\_f product, the pooled five-site
distribution of built-cell λ\_f sits in 0.0–10+ with p50 = 1.62 and p95 = 6.30,
so the legacy 0.35 and Macdonald 0.40 thresholds — calibrated on
regular-array experiments — both saturate at < 15 % adequate when applied
absolutely to Rio favelas. We therefore re-pin λ\_f to a **single pooled
relative threshold of λ\_f = 2.75**, which is **p75 of the pooled
corrected-λ\_f distribution** and yields a pooled adequate share of
**47.4 %** — centred in the target 40–55 % band.
Under this threshold the four-state shares pool at **adequate 47.4 % ·
sunlight constraint 27.6 % · ventilation constraint 6.6 % · compound
constraint 18.4 %** across 56,657 classified cells.
The hillside / flatland compound contrast is **14.8 % (hillside) vs
21.6 % (flatland), gap −6.8 pp**: under the corrected λ\_f, flatland
favelas (Maré, Rio das Pedras) carry slightly *more* compound constraint
than hillsides — the opposite of what the broken-λ\_f map suggested.
That is because the broken λ\_f was effectively a building-count proxy
(r = 0.82 vs building\_count, paper §1 forensic finding), so hillsides
appeared vent-saturated by accident; once λ\_f is clipped to cells,
the high-vent flag selects for the densest geometry, which is the flat
Maré-type fabric, while hillsides remain dominated by the sun signal
(sunlight constraint 37.5 % vs flatland 19.0 %).
The previously-cited **56,657 built cells** is the **canonical
denominator** going forward; the 64,389 figure that briefly appeared
in `taxonomy_shares.json` was the raw built mask before dropping cells
with no street-solar observation within 25 m. The 7,732-cell gap is
dominated by Maré (4,881 cells, 17 % of its built mask) — large-block
interiors and low-density edges that the pedestrian-network solar
sampler cannot reach — followed by Alemão (1,380, 8 %), Rio das Pedras
(895, 14 %), Rocinha (368, 5 %) and Vidigal (208, 8 %).

| Site | Terrain | n\_built | n\_classified | Adequate | Sunlight | Ventilation | Compound |
|---|---|---:|---:|---:|---:|---:|---:|
| Vidigal | hillside | 2 756 | 2 548 | 42.3 % | 45.3 % | 3.0 % | 9.3 % |
| Rocinha | hillside | 8 031 | 7 663 | 23.0 % | 39.4 % | 2.8 % | 34.8 % |
| Complexo do Alemão | hillside | 17 768 | 16 388 | 56.1 % | 35.4 % | 2.4 % | 6.2 % |
| Rio das Pedras | flatland | 6 605 | 5 710 | 25.6 % | 25.1 % | 8.2 % | 41.1 % |
| Maré | flatland | 29 229 | 24 348 | 54.8 % | 17.5 % | 10.7 % | 17.0 % |
| **ALL (pooled)** | — | **64 389** | **56 657** | **47.4 %** | **27.6 %** | **6.6 %** | **18.4 %** |
| HILLSIDE | — | 28 555 | 26 599 | 45.2 % | 37.5 % | 2.6 % | 14.8 % |
| FLATLAND | — | 35 834 | 30 058 | 49.2 % | 19.0 % | 10.2 % | 21.6 % |

Threshold: **λ\_f > 2.75** (p75 of pooled corrected distribution) and
**winter direct sun < 2 h** (WHO floor, unchanged). Both labelled
INTERIM and RELATIVE pending CFD-ACH + LMA. Code:
`scripts/brisa_ventilation/08_interim_lambda_f_taxonomy.py`. Full
sweep + reconciliation: `outputs/brisa_ventilation_fix/taxonomy_interim_lambda_f.json`.
