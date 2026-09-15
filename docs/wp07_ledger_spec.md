# WP-07A — C′ numbers ledger — spec (2026-09-15)

Status: implementer brief for task WP07A (brisaverse tasks.json). First phase of
the WP-07 numbers side. Laptop, CPU, seconds. Paper prose is NOT in scope —
the ledger is numbers + provenance only; the PI writes every sentence.

## Why

Both PI decisions landed on 2026-09-15 (`wp05_run_design` = A_exhaustive_1m,
`g3_domain` = B_centre_010_10), so every C′ number produced by WP-04/05/06 and
G3 is FINAL. Those numbers currently live in five run folders whose JSON still
carries the stale string "PROVISIONAL — … untapped". Figures and (later) the
manuscript must read ONE ledger whose every value is copied by code from a run
of record and can be re-verified mechanically — never typed.

## Runs of record (read-only; never modify a run folder)

| purpose | path |
|---|---|
| citywide distribution + five-favela positions | `runs/wp05_full_20260914T215419Z/distribution.json` (+ `manifest.json`) |
| five sites, ground + street ONLY (façade NOT accepted — exclude) | `runs/wp04_sites_20260914T230606Z/<site>/summary.json` |
| domain sensitivity (9 grid variants + 2 universes) | `runs/g3_domain_20260915T042927Z/sensitivity.json` |
| geometry constraint shares | `runs/wp06_geometry_20260915T052604Z/summary.json` |
| engine acceptance vs CPU street SVF (nearest-cell march) | the `runs/wp02_horizon_20260914T19*/crossref*.json` whose sampling is nearest-cell (r ≈ 0.995) — state in the ledger which file you used and why |
| decision provenance | `config/params.yaml` `domain.status`, `sampling.run_design`, `sampling.cell_m_status` (pointers only — copy no parameter values into code) |

## Deliverables

1. `src/brisa_solar/wp07_ledger.py` — `build_ledger(repo_root) -> dict` and a
   `__main__` that writes `runs/wp07_ledger_<UTC>/ledger.json` + `ledger.md`
   (+ `manifest.json` via the existing `write_run_manifest` pattern, or a
   minimal manifest with git sha and source run ids). Entry schema:
   ```
   {id, value, unit, source:{file, json_pointer, run_id, run_utc}, status:"final",
    decided_by:["wp05_run_design 2026-09-15","g3_domain 2026-09-15"],
    release_class:"publishable-candidate"|"reviewer-defence-only"|"withheld"}
   ```
   Ids are stable dotted slugs, e.g. `citywide.svf.p50`, `citywide.kwh_m2.p90`,
   `favela.vidigal.svf.median`, `favela.vidigal.svf.percentile`,
   `site.rocinha.ground.sun_h_winter.p50`, `site.rocinha.ground.share_ge_2h_winter`,
   `site.rocinha.street.svf.p50`, `g3.spread.complexo_do_alemao.svf`,
   `wp06.riodaspedras.share_n3`, `engine.crossref.r`, `engine.crossref.median_abs_delta`.
   Cover: citywide p1…p99 for svf + kwh_m2; per favela median/IQR/percentile
   for both; per site ground + street quantiles (p10/p25/p50/p75/p90), both
   reference days' sun-hours quantiles and all threshold shares; the five
   percentiles under each of the 9 grid variants; WP-06 n and the four shares
   per site; the engine acceptance triplet (r, median |Δ|, p95 |Δ|).
   `_meta` states once: the supersession of the sources' PROVISIONAL strings
   (decision ids + dates), the façade exclusion, the sky constant (import
   `P1_SKY_PATCHES`; never the literal), and "values unrounded; ledger.md
   rounds to 3 significant figures for reading only".
2. **Derived block** (computed here, formula stated in `_meta.derived`):
   per favela the max–min spread of its SVF percentile across the NINE grid
   variants (exclude the WP-04 polygon-interior universe), and the rank order
   of the five favelas by SVF percentile under the locked domain, with a flag
   `rank_invariant_across_grid: bool`.
3. **Release classes** — from `brisaverse/shared/ethics/red_lines.md` §2/§5 (read
   it): citywide and per-site AGGREGATES → `publishable-candidate` (the gate
   decides later); any favela-vs-non-favela or favela-vs-formal contrast (the
   all-favela vs non-favela medians in distribution.json, if present) →
   `reviewer-defence-only` (L1); anything per-cell is NOT a ledger entry at all
   (no arrays longer than 12 elements anywhere in ledger.json).
4. Extend `scripts/lint_p1_tokens.py` scan set with `runs/wp07_*/**/*.md` and
   `runs/wp07_*/**/*.json` (the ledger is a P1 source now). Prove it still
   passes and still fails on a planted token.
5. `tests/test_wp07_ledger.py`: (a) round-trip — every entry's value equals
   the source re-read at `json_pointer` (exact float equality); (b) ids unique,
   every source file is one of the runs of record above; (c) no list longer
   than 12 in the document; (d) every entry whose id contains `non_favela`,
   `formal` or `contrast` is `reviewer-defence-only`; (e) `derived` spread
   equals a direct recomputation from sensitivity.json; (f) `ledger.md` carries
   no banned token (call the lint's function, don't restate the list); (g) the
   5-favela SVF percentiles equal the WP-05 distribution's to 1e-9 and the
   `g3.*` base-variant percentiles equal them to 0.1 point.
   Skip cleanly (pytest.skip) where a run folder is absent.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_wp07_ledger.py tests/test_wp06_geometry.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py && python3 scripts/lint_p1_tokens.py
python3 scripts/emit_cockpit.py --check
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

Your LAST action is `git add <files> && git commit` on your branch; paste the
hashes. Track `runs/wp07_ledger_<UTC>/*.json|*.md` (ledgers are tracked).

## Never

No paper prose; no rounding in ledger.json; no typed numbers anywhere in
code or tests (every expected value is re-read from a source); never the
literal 145; no "flow"/CFD tokens; no per-cell data; nothing written outside
`src/brisa_solar/`, `scripts/lint_p1_tokens.py`, `tests/`, `runs/wp07_ledger_*/`.
