# G3 / HD-02 domain sensitivity — spec (2026-09-15)

Status: implementer brief for task G3CARD. Produces the evidence the PI needs
to sign off the analysis domain (config/params.yaml `domain`, status default).
Laptop only; minutes of GPU at most.

## Why

The five favelas' percentile positions depend on which ground cells count as
"the city": the WP-05 frame (fabric coverage ≥ 0.10 in a 100 m window AND a
footprint within 10 m) gives Rocinha p18.3 / Vidigal p15.0 / Alemão p36.8;
WP-04's all-polygon-interior universe gives p25.7 / p16.6 / p43.8. The headline
claim needs ONE definition, chosen with the sensitivity in view.

## Deliverable

`runs/g3_domain_<UTC>/sensitivity.json` + `sensitivity.md` (a table, no prose
beyond captions) with, for each domain variant:

- variants = fabric_coverage ∈ {0.05, 0.10, 0.20} × footprint_distance_m ∈
  {5, 10, 20} (9 cells; the plan's sensitivity grid) plus the two universes
  already computed (WP-05 frame, WP-04 polygon interior);
- frame size (cells), favela share, citywide SVF and kWh/m² medians, and the
  five favelas' percentile-of-median for SVF and kWh/m²;
- the max–min spread of each favela's percentile across variants (the number
  the card will show).

Reuse, do not recompute what exists: `runs/wp05_full_20260914T215419Z/
wp05_full.parquet` holds every cell of the 0.10/10 m frame with svf and
kwh_m2; `frame_cells.parquet`/`frame_diagnostics.json` hold the frame
bookkeeping. Tighter variants (0.20, 5 m) are SUBSETS — filter. Looser
variants (0.05, 20 m) add cells that were never evaluated: rebuild only the
added cells with `wp05_full.py`'s tile pass (same engine, 1 m, nearest march)
and append; report how many were added and their wall time. Keep the
per-cell layers under `runs/` (withheld).

## Tests — tests/test_g3_domain.py

1. Subset variants are exact subsets of the 0.10/10 m frame (cell ids).
2. Percentile-of-median is computed on the variant's own distribution
   (synthetic check with a known answer).
3. The WP-05 frame variant reproduces distribution.json's five percentiles
   to 0.1 point.

## Then

Draft the `/ops` card as DATA ONLY (a JSON snippet in `sensitivity.json`
named `card_draft`: id `g3_domain`, question, three options with the measured
numbers in `detail`, recommended = the variant closest to the grid centre
unless the table says otherwise, and say why in one sentence). The
orchestrator places it in brisaverse; you do not touch brisaverse.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_g3_domain.py tests/test_wp05_full.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

Your LAST action is `git add <files> && git commit` on your branch; paste the
hashes. Never a number in params.yaml; never the literal 145; no map outside
`runs/`.
