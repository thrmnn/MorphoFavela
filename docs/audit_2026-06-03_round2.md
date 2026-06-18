# MorphoFavela repo audit — 2026-06-03 (Round 2)

> Companion to docs/audit_2026-06-03.md. Round 2 covers the 8 surface gaps the Round 1 completeness critic flagged.

## Executive summary

- **194 findings** across 8 surfaces (tests, brisa-deck, src-viz, governance, docs-root, claude-md, notebooks-vendor-patches, executability). Severity breakdown: ~26 high, ~110 medium, ~58 low.
- **Top theme: untracked load-bearing code.** `src/viz/` (imported by 6 tracked files including a paper figure and the Vidigal diagnostic map) and the entire `scripts/brisa_deck/` directory (22 figure scripts) sit at `??` in git status. A fresh clone of `feat/brisa-paper` would `ImportError` on the modified `outputs/paper_figures/fig05_predictors.py`.
- **Top theme: Makefile + docs reference deleted scripts.** `make morphology` and `make pipeline` invoke `scripts/calculate_morphology_metrics.py` (deleted in CHANGELOG Pass 2). `docs/methodology/street_svf.md`, `docs/GPU_SVF_EXACT_VALIDATION.md`, `docs/FAVELA_EXTRACTION_WORKFLOW.md`, `docs/methodology/sky_exposure.md`, and the TLS notebook all call scripts that no longer exist.
- **Top theme: rename + env drift.** environment.yml advertises a `morphofavela` env that does not exist on disk; the IVF/brisa envs that actually drive work are 1+ majors behind nearly every pin (numpy 2.2.6 vs 1.24.4, geopandas 1.1.3 vs 0.13.2). CHANGELOG / ROADMAP still cite `ivf-*` console scripts but pyproject ships only `mf-*`. Notebook hard-codes `/home/theo/IVF`.
- **Executability surprise: CI lint is RED on this branch.** Running the exact CI command (`ruff check src/ scripts/ tests/ outputs/paper_figures/`) fails with 2 I001 errors in the two `MM`-status files at the top of `git status` (build_vidigal_diagnostic_map.py, fig05_predictors.py); `ruff format --check` reports 40 files would be reformatted. Pre-commit pins ruff `v0.6.9` while CI/pyproject pin `0.15.15` — a six-major gap. Vendored UMEP is 11 commits behind upstream main with an unpinned `--depth 1` clone.

## Top 10 priorities (composite-ranked)

1. **src/viz/ is untracked but load-bearing for tracked figure scripts** [high, src-viz/untracked-status, 9.0/10] — fig05_predictors and the Vidigal diagnostic map will `ImportError` on any fresh clone. Action: `git add src/viz/` and commit alongside the brisa_deck scripts. Evidence: `git status` shows `?? src/viz/`; six tracked files import `from src.viz import presentation_style`.

2. **scripts/brisa_deck/ is untracked despite being a 22-file BRISA-paper deck producer** [high, brisa-deck/untracked-status, 8.7/10] — every BRISA slide asset is produced here and the branch is `feat/brisa-paper`. Action: decide via README first whether this belongs in the repo, then `git add scripts/brisa_deck/*.py`. Evidence: `git status` shows `?? scripts/brisa_deck/`; no `.gitignore` rule covers it.

3. **scripts/brisa_deck/ depends on the same untracked src/viz/** [high, src-viz/coupling, 8.3/10] — the two untracked surfaces must be committed together. Action: single changeset for `src/viz/` + `scripts/brisa_deck/`. Evidence: 4 brisa_deck modules `import from src.viz`.

4. **Makefile `morphology` and `pipeline` targets invoke deleted `calculate_morphology_metrics.py`** [high, Makefile/code-drift, 7.3/10] — `make morphology` and `make pipeline` both fail with FileNotFoundError. Action: repoint to `scripts/run_morphometric_audit.py` (the `mf-morphometry` entry). Evidence: Makefile:26,30; CHANGELOG:25 records the deletion.

5. **environment.yml pins are 1+ majors ahead of the env that actually produces results** [high, environment.yml/version-drift, 7.0/10] — environment.yml advertises numpy 2.2.6, geopandas 1.1.3, scipy 1.17.1 while the active IVF/brisa env runs numpy 1.24.4, geopandas 0.13.2, scipy 1.10.0. Action: rebuild morphofavela env from environment.yml and verify pipeline runs, or downgrade pins to match the env that produced May 2026 results. Evidence: environment.yml:11 ("Versions are pinned to those used to produce the paper results").

6. **README + CONTRIBUTING tell users to `conda activate morphofavela`; no such env exists locally** [high, environment.yml/env-name, 7.0/10] — the documented reproduction recipe cannot be exercised on the author's own machine. Action: create the env locally to validate the recipe, or rename to the working env name. Evidence: README:68,195; CONTRIBUTING:13; `conda env list` shows only IVF, brisa.

7. **patches/ artifacts are gitignored but still consumed by tracked scripts** [high, patches/untracked, 6.7/10] — `scripts/generate_synthetic_cfd_results.py`, `scripts/cfd_overlay/build_vidigal_patch_viz.py`, and `scripts/cfd_overlay/build_qgis_project.py` read paths under `patches/VDG-P02/inputs/` and `patches/VDG-P07/inputs/` that no public clone can populate. Action: document the data-availability path in README/data/README.md (Zenodo DOI, on-request, or regenerator script). Evidence: `.gitignore` line 72; commit 18a649b.

8. **Vendored UMEP-processing is unpinned and 11 commits behind upstream main** [high, vendor/supply-chain, 6.3/10] — `scripts/validate_svf_against_umep.py` clones with `--depth 1` no `--branch`, so the SVF cross-validation numbers in §10.3 are non-reproducible. Action: pin UMEP_REPO to a specific commit/tag (e.g. 28b3fea) via post-clone `git checkout <sha>` and record the SHA in `docs/onboarding/local_setup.md`. Evidence: validate_svf_against_umep.py:59; local HEAD `28b3fea` vs upstream `ea9694f`.

9. **scripts/brisa_deck/ has no README, manifest, or output-path documentation** [high, brisa-deck/doc-coverage, 6.0/10] — 22 figure scripts hard-code `OUT_DIR = /home/theo/brisa_paper/artifacts/slides/assets/` (a sibling-repo path) with no env-var fallback, no README, and no row in `scripts/README.md`. Action: add `scripts/brisa_deck/README.md` modeled on `outputs/paper_figures/README.md` (track positioning, figure-by-figure table, regen command) and replace hard-coded paths with `BRISA_PAPER_ASSETS` env var. Evidence: `ls scripts/brisa_deck/` shows 22 `.py` files, no `.md`; every script has `Output: /home/theo/brisa_paper/...`.

10. **docs/methodology/{street_svf,sky_exposure}.md + docs/GPU_SVF_EXACT_VALIDATION.md + docs/FAVELA_EXTRACTION_WORKFLOW.md invoke deleted scripts** [high, docs/code-drift, 5.7/10] — every Quick Start / Usage / Repro block points at scripts that no longer exist (`compute_svf_streets.py`, `compute_svf_streets_gpu.py`, `analyze_sky_exposure_streets.py`, `extract_favela_data.py` at the old flat path). Action: rewrite the repro blocks to the current entry points (`scripts/run_svf_v2.py --mode streets`, `scripts/data_utils/extract_favela_data.py`), or archive the docs. Evidence: street_svf.md:12,22,35; GPU_SVF_EXACT_VALIDATION.md:51,69,87; FAVELA_EXTRACTION_WORKFLOW.md:35,207; sky_exposure.md:112.

## All findings by surface

### tests/

**flaky-or-skipped**
- [medium] vidigal_tls integration tests skip silently due to boundary lookup mismatch — `AREA_FILES['vidigal_tls']['boundary'] = 'Vidigal_Limit.shp'` lives only under `data/vidigal/raw/`, so `_area_data_available` always returns False. Every parameterized integration test silently skips that area.
- [medium] Integration parametrization covers 3 of 10 onboarded areas — `AREAS = ['vidigal_tls','vidigal','riodaspedras']` hard-coded; 7 other AREA_FILES entries are never exercised.
- [low] `test_svf_street_10pts` and `test_svf_facade_10pts` skip on empty samplers — a 0-row result is a data-drift regression for onboarded sites, not a "lacks feature" condition.
- [low] `skipif(not HAS_RASTERIO)` is dead — rasterio is a hard pin.
- [low] `pytest.importorskip('pyviewfactor')` is stale in 3 tests — pyviewfactor is a required dep.
- [low] Defensive "No sun positions computed" skip is unreachable in Rio June solstice test.

**orphan-test / import-graph**
- [medium] `src.data_alignment_utils` exercised only by `tests/test_data_alignment.py` — 798 lines of production code kept alive by its test.
- [medium] `src.morphology_metrics` exercised only by `tests/test_morphology_metrics.py` — the live morphometric stack lives under `src/morphometry/` and `src/metrics.py`.
- [low] `tests/test_cfd_overlay/test_build_vidigal_patch_viz.py` is the sole importer of that CLI script.

**stale-test / doc-coverage**
- [high] `tests/README.md` describes a non-existent SVF-only test layout — lists 5 top-level files (`test_svf_unit.py`, etc.) that do not exist. Real tree has ~24 modules across solar, CFD integration, cartography, typology, exposure.
- [medium] Real marker taxonomy (`fast` / `integration`) is undocumented in tests/README.md.
- [medium] Area-data skip behaviour in `test_svf_v2/conftest.py` is undocumented.
- [medium] `tests/README.md` references `requirements.txt` that does not exist.
- [medium] Documented `-m cuda` / `-m "not cuda"` marker is not registered or used anywhere.
- [medium] Conftest fixture set is undocumented; subdirectory conftests collide silently.
- [low] `tests/test_solar_access.py` duplicates coverage already in `tests/test_solar/`.
- [low] `tests/conftest.py` exposes SVF mesh fixtures (empty_mesh, single_building_mesh, two_buildings_mesh) that no current test consumes.
- [low] `tests/run_tests.py` is an orphaned SVF/CUDA runner — gates on pytorch3d (not a dep) and a `cuda` marker no test uses.
- [low] `tests/test_utils.py` (191 lines) is a helper module with no tests, never imported.
- [low] `tests/utils/test_helpers.py` (274 lines) is unreferenced helper code.
- [low] `_make_box_building` defined twice in test_svf_v2/ — conftest version vs inline copy in test_tregenza.py.

### scripts/brisa_deck/

**untracked-status-assessment**
- [high] Entire directory is untracked despite being a 22-file BRISA-paper deck producer (no `.gitignore` rule).
- [high] All 4 brisa_deck/*.py files that import `from src.viz` depend on the same untracked `src/viz/`.

**doc-coverage**
- [high] No README, manifest, or index — sibling `outputs/paper_figures/` ships a detailed README; brisa_deck has zero markdown.
- [high] Output paths point to an external sibling repo `/home/theo/brisa_paper/artifacts/slides/assets/` with no documentation of the dependency.
- [high] No directory-level README documents brisa_deck purpose, output target, or relationship to outputs/paper_figures/.
- [medium] `scripts/README.md` index does not mention `brisa_deck/`.
- [medium] Cross-directory import from `outputs/paper_figures/fig_style` via `sys.path.insert` is undocumented.
- [low] Per-script docstrings do not declare data-source provenance for most figures (no "Data sources" column).

**output-trace**
- [medium] All 22 scripts write to a hardcoded `~/brisa_paper/` path that is not portable across hosts.
- [medium] `fig_patch_sampling_terrain.png` is produced by two scripts that overwrite each other (`fig_patch_sampling_terrain.py` shadowed by `fig_patch_sampling_terrain_context.py`).
- [low] `presentation_figures` dual-write branch has no downstream consumer in 4 scripts.
- [low] `fig_morpho_distributions.png` has no downstream consumer (superseded by 3-row variant).
- [low] `fig_morpho_typology_scatter.png` has no downstream consumer.
- [low] `fig_patch_sampling_vidigal_v2.png` has no downstream consumer.
- [low] `fig_svf_cross_site_large.png` has no downstream consumer.
- [low] `fig_svf_cross_site_v2_2row.png` has no downstream consumer.
- [low] `fig_svf_streets_cross_site{,_color}.png` have no downstream consumers.
- [low] `fig_vidigal_aspect_slope_curves.png` has no downstream consumer.

**duplicate-with-paper-figures**
- [medium] brisa_deck duplicates `fig02_morphometric_distributions` in 5 variants (distributions, distributions_3row, boxplots, violins, typology_scatter) — slide-themed reskins of the canonical paper figure.
- [medium] Ridges/violins/boxplots scaffolding repeated across 4 files inside brisa_deck/ without consolidation.
- [low] brisa_deck outputs are asymmetric duplicates of `outputs/paper_figures/exports/`.

### src/viz/

**untracked-status-assessment**
- [high] Package is untracked; `git check-ignore` returns nothing — forgotten, not intentional.
- [low] Not actually a duplicate of `outputs/paper_figures/fig_style.py` — fig_style targets paper-print sizes, presentation_style targets projector sizes, and `apply('paper')` is deliberately a no-op so they compose. Commit but keep both.

**duplicate-with-fig-style / duplicate-with-cartography**
- [medium] Third `add_scale_bar` implementation duplicates `src/cartography.add_scale_bar` (the canonical one, imported by 9 modules).
- [medium] Two `add_scalebar` helpers with divergent defaults (height, padding, loc, color, fontsize).
- [medium] Name collision: two `add_scale_bar` functions with different signatures (`loc` vs `location`).
- [medium] Duplicated "clean ticks/spines from map axes" helper (`apply_to_map_axes` vs `clean_map_axes`).
- [medium] North-arrow logic duplicated and entangled with scale bar.
- [low] Two parallel rcParams bundles (`apply` vs `apply_style`) with no shared base — docstring even warns about load-order foot-gun.
- [low] `_SHORT_LABELS` map shadows colorbar labels also hard-coded in fig_style consumers.

**doc-coverage / import-graph**
- [medium] `src/viz/` has no README despite being a distinct sibling package to `src/visualization/`.
- [medium] `src/viz/` presence and contract undocumented in repo-level docs (zero grep hits in README, ROADMAP, CLAUDE.md, technical_report).
- [low] Conftest assumptions for the preset API are undocumented (no tests, no fixture).
- [low] `_SHORT_LABELS` lookup table has no documented coverage policy.
- [low] `src/viz` uses absolute `src.viz.*` self-imports instead of relative.

### governance (pyproject / environment / vendor / CITATION)

**pinned-version-drift**
- [high] environment.yml pins are far ahead of the actual working IVF/brisa env (numpy, scipy, geopandas, pandas, sklearn, pyarrow, pvlib, pyvista, matplotlib, libpysal, esda, numba all 1+ majors out of step).
- [high] environment.yml `name: morphofavela` env does not exist; legacy IVF env drives all work.
- [low] README + CONTRIBUTING `conda activate morphofavela` instruction is unexercised.
- [medium] pyogrio and fiona declared in pyproject but not installed in the active env.
- [medium] ruff dev-dep pin `0.15.15` does not match installed ruff `0.15.0`.
- [medium] pyproject `[project.scripts]` require IVF env; brisa env is missing pyvista/pvlib (5 of 13 entry points fail there).
- [low] CHANGELOG `numpy < 2.3` narrative contradicts the active env (numpy 1.24).
- [low] Python version drift between environment.yml (3.11) and brisa env (3.12).

**supply-chain-pin**
- [high] Vendored UMEP-processing has no pin (`--depth 1` clone, unpinned).
- [high] Vendored UMEP is 11 commits behind upstream main; drift includes SQL-injection and import fixes.
- [high] Vendor edits to `__init__.py` files are uncommitted and undocumented — the validator silently rewrites three vendored `__init__.py` files at every run.
- [medium] vendor/umep_processing carries uncommitted local edits to `__init__.py` files.
- [medium] vendor/ tree is gitignored, so the validator's reproducibility claim has no checked-in artefact.
- [medium] No `vendor/PIN.md` documents the verified upstream commit.

**doc-coverage**
- [medium] `vendor/` has no top-level README explaining the vendoring policy.

**code-drift**
- [medium] ROADMAP/CITATION still scope the project to 5 sites; data/ now has borel, jacarezinho, morro_do_juramento.
- [low] CITATION.cff version 0.1.0 / date 2026-04-13 contradicts CHANGELOG activity through May 2026.

### docs-root / ROADMAP / CHANGELOG / README

**code-drift / broken-references**
- [high] `docs/methodology/street_svf.md` references the deleted `scripts/compute_svf_streets.py` (3 invocations) — 2 separate findings.
- [high] `docs/GPU_SVF_EXACT_VALIDATION.md` repros call deleted `scripts/compute_svf_streets_gpu.py` — 3 separate findings.
- [high] `docs/methodology/sky_exposure.md` Usage block calls non-existent `scripts/analyze_sky_exposure_streets.py` — 2 findings.
- [high] `docs/FAVELA_EXTRACTION_WORKFLOW.md` cites `scripts/extract_favela_data.py` at deleted flat path (4 references) — 3 findings.
- [high] `docs/PRODUCTION_READINESS_PLAN.md` cleanup table is overwhelmingly stale — all A-section + most B-section targets already executed.
- [medium] `brisa_lambdaf_ventilation_fix_plan.md` is OPEN despite all three prongs marked VALIDATED in sibling docs.
- [medium] ROADMAP cites deleted `scripts/compute_svf_streets.py`, `scripts/analyze_sky_exposure_streets.py`, `scripts/calculate_metrics.py` — 5 separate findings.
- [medium] ROADMAP `ivf-*` console entry points renamed to `mf-*` — 4 separate findings.
- [medium] ROADMAP §Phase 4 module list cites non-existent `src/visualize_morphology.py`, `src/exposure.py` — 3 findings.
- [medium] CHANGELOG retains `ivf-*` console aliases that were renamed to `mf-*`.
- [medium] Makefile default `AREA=vidigal_tls` is a non-campaign site documented as deleted.
- [medium] README contradicts itself on the size of the morphometric indicator set (12 vs 20+ vs 25).
- [medium] README §Project Structure tree advertises `src/exposure.py` (now a package).
- [low] ROADMAP has two distinct Phase 5 sections (Future Environmental Performance and CFD Sampling Campaign).
- [low] ROADMAP status banner declares "5 sites onboarded" but BRISA work added 3 more.
- [low] ROADMAP version history skips v5.5.0 chronologically and omits v6 / post-May 2026 work.
- [low] ROADMAP tests list mentions tests that pair with deleted modules.
- [low] README walkthrough cites unused INMET station `A602`.
- [low] CONTRIBUTING claims `tests/test_cfd_integration` has 46 tests; actual is 71.

**doc-coverage / staleness**
- [medium] `docs/README.md` Layout block omits engineering_review.md and every BRISA / audit / production doc.
- [medium] `engineering_review.md` references TR-numbers that drifted (per the same-day audit doc).
- [medium] `PRODUCTION_READINESS_PLAN.md` is a 2026-05-02 audit whose work is now done; reads as live state.
- [medium] Onboarding docs cite stale smoke-test count (508) — current collection yields 553.
- [medium] `morphometric_indicators.md` retains raw `[cite_start]` PDF-export markers and contradicts its own count.
- [medium] `local_setup.md` lists 5 campaign sites but data/ has 3 more BRISA sites unmentioned.
- [medium] `FAVELA_EXTRACTION_WORKFLOW.md` (last touched 2026-03-06) reads as a pre-implementation design doc with unchecked TODO boxes.
- [medium] Two-track figure convention has no documented third track for `brisa_deck/`.
- [medium] Three BRISA ventilation docs duplicate the same forensic-findings narrative (plan / fix_report / handoff).
- [medium] Per-site broken-λf statistics table is duplicated between fix_plan and fix_report.
- [medium] Three-prong A/B/C structure is restated in plan, report, and handoff.
- [medium] `brisa_ventilation_handoff.md` and `brisa_ventilation_fix_report.md` substantially overlap.
- [low] CHANGELOG `577 tests / 508 non-integration` no longer matches collection (actual 622).
- [low] CHANGELOG `ivf-*` console scripts after rename.
- [low] `brisa_lambdaf_ventilation_fix_plan.md` still marked OPEN; companion report says VALIDATED.
- [low] `docs/README.md` inventory omits brisa_*.md and PRODUCTION_READINESS_PLAN.md (3 findings on overlapping omissions).
- [low] `engineering_review.md` and `local_setup.md` cite stale smoke-test counts.
- [low] `engineering_review.md` references missing manuscript and broken TR §10.3 number.
- [low] `PRODUCTION_READINESS_PLAN.md` E-row claims `engineering_review.md` is MISSING — it now exists.
- [low] `PRODUCTION_READINESS_PLAN.md` D-row claims `docs/README.md` is MISSING — it now exists.
- [low] `PRODUCTION_READINESS_PLAN.md` D6 §8 line pointer (781) is off by ~500 lines (actual 1272).
- [low] File index for `scripts/brisa_ventilation/` is duplicated in handoff and fix_report.
- [low] `street_svf.md` --area help advertises slugs no longer in `SUPPORTED_AREAS`.
- [low] ROADMAP §Phase 2.5 still describes solar access as winter-solstice-only; pipeline now seasonal.
- [low] Citations pinning block duplicated between fix_plan and fix_report.
- [low] Four-state taxonomy per-site shares overlap between fix_report §4 and taxonomy_interim note.
- [low] `brisa_ventilation_fix_report.md` cites stale `compute_frontal_area_ratio` line range.

### CLAUDE.md / .claude/

- [medium] Memory directory path `.claude/projects/-home-theo-MorphoFavela/memory/` does not exist; only `-home-theo-IVF/memory/` is on disk.
- [medium] CLAUDE.md claims six project subagents but seven exist (the file under-counts; numerical-claims-auditor is missing from the list).
- [medium] numerical-claims-auditor agent unmentioned in CLAUDE.md subagent inventory.
- [low] `.claude/hooks/check_report_sync.py` docstring claims 15-test FP-rate floor; suite has 29 tests.

### notebooks-vendor-patches

**notebooks/**
- [high] TLS comparison notebook hard-codes pre-rename `/home/theo/IVF` project root.
- [high] TLS notebook invokes `scripts/compute_svf.py` which no longer exists.
- [high] Notebook hardcodes pre-rename project path (`PROJECT_ROOT = Path("/home/theo/IVF")`).
- [medium] `notebooks/README.md` does not document `compare_vidigal_tls_lod2_solar_svf.ipynb` — 2 findings.
- [medium] `explore_favelas.ipynb` cells contain stale IVF paths in committed outputs.
- [medium] Notebooks last touched Feb/Mar 2026, predate rectangular_domain_v1 migration, rename, and open-source release.
- [low] `notebooks/README.md` does not reflect the rename or new data layout.

**vendor/** — see governance section above.

**patches/**
- [high] `patches/` artifacts are gitignored after the open-source release commit but still consumed by tracked scripts — 2 findings.
- [medium] VDG-P07 dossier cites unverified SLURM jobs and a snapshot that has aged out (mesh `14092625` RUNNING, solver `14092626` PENDING from 2026-05-18) — 2 findings.
- [medium] VDG-P02 dossier blocked on "VDG-P07 smoke verdict" with no follow-up — 2 findings.
- [medium] VDG-P07 dossier promises §6 deliverables (`vti2geotiff.py`, overlay/) that never landed.
- [medium] `patches/` has no top-level README explaining the dossier format.
- [low] `patches/.../inputs/SHA256SUMS` provenance is unverifiable without the gitignored bytes.
- [low] `patches/VDG-P0{2,7}/inputs/` have SHA256SUMS but no schema doc for `patch_meta.json` / `preflight_report.json`.

### executability (runs-clean lens)

- [high] Makefile `morphology` and `pipeline` targets invoke deleted script `scripts/calculate_morphology_metrics.py` — 7 separate findings converged on this.
- [high] Makefile `pipeline` target chains the same missing script after `run_svf_v2.py`.
- [low] Makefile declares a phony `cross-cluster` target that has no recipe — 2 findings.

## Executability results

**[project.scripts] entry-point import check (under IVF env, python 3.11.14):**

| Entry point | Module | IVF result |
|-------------|--------|------------|
| mf-context | `scripts.run_context` | imports OK |
| mf-svf | `scripts.run_svf_v2` | imports OK |
| mf-solar | `scripts.compute_solar_access` | imports OK |
| mf-facade-solar | `scripts.run_facade_solar` | imports OK |
| mf-morphometry | `scripts.run_morphometric_audit` | imports OK |
| mf-typology | `scripts.run_typology_analysis` | imports OK |
| mf-deprivation | `scripts.run_deprivation_raster` | imports OK |
| mf-wind-rose | `scripts.build_wind_rose` | imports OK |
| mf-pilot-sampling | `scripts.run_pilot_sampling` | imports OK |
| mf-campaign-sampling | `scripts.run_campaign_sampling` | imports OK |
| mf-synthetic-cfd | `scripts.generate_synthetic_cfd_results` | imports OK |
| mf-analyze-cfd | `scripts.analyze_cfd_results` | imports OK |
| mf-validate-svf | `scripts.validate_svf_against_umep` | imports OK |

Under the `brisa` env (python 3.10), 5 of 13 fail (mf-context, mf-svf, mf-solar, mf-facade-solar, mf-morphometry) — pyvista/pvlib missing; pyproject `requires-python = ">=3.11"` is the documented contract.

**Makefile dry-run (`make -n <target>` then check script existence):**

| Target | Script invoked | Status |
|--------|----------------|--------|
| `help` | (echo only) | OK |
| `test` | `pytest tests/` | OK |
| `test-fast` | `pytest tests/ -m fast` | OK |
| `lint` | `ruff check src/ scripts/ tests/` | FAILS (see ruff lint below) |
| `format` | `ruff format src/ scripts/ tests/` | FAILS (40 files would be reformatted) |
| `svf` | `python scripts/run_svf_v2.py` | OK |
| `morphology` | `python scripts/calculate_morphology_metrics.py` | **BROKEN** — script deleted |
| `pipeline` | `run_svf_v2.py` + `calculate_morphology_metrics.py` | **BROKEN** — second step missing |
| `report` | `python scripts/generate_report.py` | OK |
| `clean` | `rm -rf outputs/`* | OK |
| `cross-cluster` | (phony, no recipe) | **PHANTOM** |

**CI lint gate (actual, not declared):**

- `ruff check src/ scripts/ tests/ outputs/paper_figures/` → **FAIL** with 2 I001 errors:
  - `scripts/build_vidigal_diagnostic_map.py:18` (import block unsorted)
  - `outputs/paper_figures/fig05_predictors.py:45` (import block unsorted)
- `ruff format --check` on the same scope → **FAIL**: 40 files would be reformatted.
- Both files above appear as `MM` in `git status` — local edits have not yet been re-linted.
- Pre-commit pins ruff `rev: v0.6.9` (.pre-commit-config.yaml); CI / pyproject pin `ruff==0.15.15` — a **six-major-version gap**. Pre-commit will pass commits that CI rejects.

**Vendored UMEP state:**

- Local HEAD: `28b3fea` (2026-04-28).
- Upstream `origin/main`: `ea9694f` (2026-06-03) — 11 commits ahead, includes SQL-injection fix, Datetime import fix, and "Avoid shadowing QGIS bundled Python packages".
- Clone is `--depth 1`, no tag, no SHA recorded anywhere in tracked source.
- Validator unconditionally rewrites three `__init__.py` files to empty on every run; `git status` inside `vendor/umep_processing/` shows `M __init__.py`, `M util/__init__.py`, `?? functions/__init__.py` permanently dirty.

## Completeness gaps (Round 2)

- **Unaudited subsystem: `docs/manuscript/figures/`** — third figure track (8 paper-candidate scripts `fig_0_1` through `fig_0_8`, ~150 KB, tracked, own README) entirely unaudited in Round 2. CI lint scope (`src/ scripts/ tests/ outputs/paper_figures/`) explicitly excludes it; no automated quality enforcement. `fig_0_5_predictors.py` likely overlaps `outputs/paper_figures/fig05_predictors.py`.
- **Unverified claim: CI lint is RED right now.** Round 2 has findings around ruff version drift and Makefile breakage but no finding actually ran the CI command and recorded that 2 ruff errors + 40 format diffs are live on `feat/brisa-paper` HEAD. Add a high-severity executability finding and run `ruff check --fix && ruff format`.
- **Unverified claim: pre-commit ruff `v0.6.9` vs CI ruff `0.15.15` is a 6-major gap** — much larger governance bug than the existing "0.15.15 vs 0.15.0" finding. Pre-commit will pass commits CI rejects.
- **Unaudited subsystem: `scripts/brisa_ventilation/`** — the 8-script producer codebase for the entire BRISA ventilation forensic story. Round 2 has 4 findings about the prose docs but ZERO about the scripts. Scripts 07/08 are also flagged by `ruff format --check`.
- **Missing lens: gitignored-but-load-bearing files in `outputs/paper_figures/`** — `cross_site_stats.json`, `rf_pd_curves.json`, `rf_pooled_data.parquet`, `rf_predictor_stats.json` sit in a tracked directory but are themselves gitignored. `fig05_predictors.py` (MM) consumes `rf_pooled_data.parquet`; a fresh clone cannot regenerate fig05 without first running `scripts/run_predictor_analysis.py`. Same hidden-input problem the patches/ findings flag, at the figure tier.
- **Unverified claim: agent count is 7, not 6 or 8.** `.claude/agents/` contains 7 agent `.md` files (data-contract-checker, cfd-results-ingestor, numerical-claims-auditor, report-sync-auditor, sampling-auditor, site-onboarder, wind-ingestion) plus README.md. CLAUDE.md says six, one Round 2 finding says eight — both wrong. Also `.claude/settings.local.json` `allow` list hard-codes `/home/theo/miniconda3/envs/morphofavela/bin/python` for 7 entries pointing to a non-existent env path.

## Methodology notes

- Surfaces: 8 (the gaps from Round 1) — tests, brisa-deck, src-viz, governance, docs-root, claude-md, notebooks-vendor-patches, executability.
- 31 (surface, lens) finder agents fan-out.
- Same adversarial verify (3 refuters per finding, claim survives unless ≥2 refute) and composite-rank (median of 3 priority votes, ties broken by severity) as Round 1.
- Round 2 total: 194 findings (vs 118 in Round 1). Surface gaps from Round 1's completeness critic are closed; Round 2's completeness critic surfaced 6 new gaps (above) for a hypothetical Round 3.
