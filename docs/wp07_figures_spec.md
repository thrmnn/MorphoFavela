# WP-07B — staged C′ figure candidates — spec (2026-09-15)

Status: implementer brief for task WP07B. Depends on WP07A (ledger merged).
Laptop, CPU, minutes. STAGING ONLY: everything lands in
`runs/wp07_figures_<UTC>/`. The orchestrator runs the ethics gate on each
figure afterwards and the PI alone promotes anything into `shared/figures/`
or `papers/`. Paper prose (captions beyond a data label) is NOT in scope.

## Inputs

- The ledger: newest `runs/wp07_ledger_*/ledger.json` — every printed number
  (medians, percentiles, shares, the 2 h floor label) comes from a ledger id.
- Distributions only (never re-derive a headline number here):
  `runs/wp05_full_20260914T215419Z/wp05_full.parquet` (svf, kwh_m2 columns;
  bin on read — 8.4 M rows, never hold per-cell geometry),
  `runs/wp04_sites_20260914T230606Z/<site>/ground.parquet` (sun-hours columns).
- `brisaverse/shared/ethics/red_lines.md` §2 + §5 — read before drawing.

## Figures (PNG 300 dpi + SVG each, one script, `src/brisa_solar/wp07_figures.py`)

| id | content | reads | release class proposed |
|---|---|---|---|
| `f1_citywide_position` | A: histogram (100 bins) of citywide ground SVF; the five favela medians as vertical markers labelled "<name> · p<percentile>" from the ledger. B: same for kWh/m². No favela-vs-non-favela split is drawn, ever (L1). | parquet + ledger | publishable-candidate |
| `f2_direct_sun_reference_days` | per-site ECDF (or violin) of ground direct-sun hours, winter solstice (A) and equinox (B); a horizontal reference line at the 2 h floor labelled "Athens Charter (1943), Point 26"; the ≥2 h share per site annotated from the ledger. | parquet + ledger | publishable-candidate |
| `f3_domain_sensitivity` | five favelas' SVF percentile-of-median across the nine grid variants (x = variant, grouped by coverage threshold), the locked domain marked; spread per favela from the ledger `g3.spread.*`. Methods / extended-data figure. | ledger only | publishable-candidate |
| `f4_geometry_constraints` | stacked bars of `n_constraints` shares (0/1/2/3) per site from the ledger `wp06.*`. | ledger only | publishable-candidate |

Site order is FIXED, never ranked by value: locate the order used by Table H
(`rg -l figures_table_h` across `~/SCL/SCR`); if it cannot be found, use
hillside then flatland — Vidigal, Rocinha, Complexo do Alemão, Maré, Rio das
Pedras — and say so in the manifest.

Style: matplotlib only; reuse `outputs/paper_figures/fig_style.py` ONLY if it
imports without pulling any CFD module (check `sys.modules` after import);
otherwise a 30-line local style. Colour-blind-safe palette; no titles or
labels containing "deficit", "formal", "shortfall"; axis labels carry units.

## Guardian-readiness (built in, tested)

`figure_manifest.json` per figure: `ledger_ids_used` (every number printed),
`source_parquets`, `release_class_proposed`, and a `checklist`:
`no_coordinates`, `no_basemap`, `no_per_cell_geometry`, `sites_fixed_order`,
`svg_path_count`, `banned_tokens_absent`. Tests
(`tests/test_wp07_figures.py`): (a) every `ledger_ids_used` exists in the
ledger and every number printed in the SVG text (regex on `<text>` content)
matches a ledger value at the stated rounding; (b) SVG contains no
6–7-digit integer runs (UTM coordinates) and no `<image>` element (no
basemap); (c) `svg_path_count` < 2,000 per figure (the registry's per-format
reads hold figures with thousands of per-building paths); (d) banned tokens
absent in SVG text (call the lint function); (e) the script never imports
`src.cfd_integration` or `scripts.analyze_cfd_results`. Skip cleanly when
inputs are absent.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_wp07_figures.py tests/test_wp07_ledger.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py && python3 scripts/lint_p1_tokens.py
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

Your LAST action is `git add <files> && git commit` on your branch; paste the
hashes. Track `figure_manifest.json`; PNG/SVG stay untracked (gitignored) —
list their paths in the final message.

## Never

Nothing written to `shared/figures/`, `papers/`, `outputs/paper_figures/`,
`docs/manuscript/`; no map of any kind; no per-cell scatter; no favela-vs-
formal or favela-vs-non-favela panel; never the literal 145; no "flow"/CFD
tokens; no typed numbers — every printed value is a ledger read.
