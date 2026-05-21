# Production-Readiness Plan

**Owner:** thrmnn  ·  **Drafted:** 2026-05-02  ·  **Target review window:** 5 working days

## North star

**An engineering reviewer should be able to clone the repo, set up the
environment, run the pipeline on one site, and read the technical report
end-to-end in less than one working day** — and emerge with concrete,
non-trivial questions about methodology rather than confusion about
mechanics.

If the reviewer asks "where does this number come from?", the answer must
always be: a section reference + a script + a committed output path. No
hand-waving, no folkloric knowledge in chat history.

---

## Audit findings (state as of 2026-05-02)

### A. Cleanup targets (high-confidence — these can just be deleted/moved)

| # | Target | Status | Action |
|---|---|---|---|
| 1 | `feature/gpu-svf-acceleration/` (orphan branch leftover, 1 stub README, tracked in git) | Tracked | **Delete.** Branch was never merged; readme is a planning stub. |
| 2 | `tests/*.md` — 8 ad-hoc planning/design docs (DATA_ALIGNMENT, EDGE_CASE_FAILURES, TEST_*, etc., totaling ~42 KB) | Tracked | **Delete.** Pre-AI-era design docs that the actual tests subsumed. Keep `tests/README.md`. |
| 3 | `SKY_EXPOSURE_METHODOLOGY.md`, `STREET_SVF_USAGE.md` (top-level, 302 lines combined) | Tracked | **Move to `docs/methodology/`.** They're real methodology, just misfiled at the top level — clutter for a first-time visitor. |
| 4 | `docs/archive/` — 31 historical planning docs | Tracked | **Audit + cull.** Keep at most 3 that have lasting value (e.g. `MIGRATION_GUIDE.md`); move the rest to a `pre-2026-04` git tag and delete from working tree. |
| 5 | `logs/` — 11 stale run logs, gitignored but on disk | Untracked | **`rm -rf logs/`** locally; the gitignore stays. |
| 6 | `ivf.egg-info/` — pip install artifact | Untracked | Already gitignored as `*.egg-info/`. **No action.** |
| 7 | `requirements.txt` (alongside `pyproject.toml`) | Tracked | **Decision needed.** Either delete (pyproject is canonical) or document why both exist (e.g. for non-pip users). Default: delete. |

### B. Cleanup targets that need a decision (medium-confidence)

| # | Target | Risk | Question |
|---|---|---|---|
| B1 | `scripts/` redundancies: `calculate_morphology_metrics.py` (legacy) vs `compute_urban_morphology.py` (current); `compute_deprivation_index.py` vs `compute_deprivation_index_raster.py`; `analyze_morphology_risk.py` vs `run_morphometric_audit.py` | Low — but new readers will pick the wrong script | **Audit each pair.** Mark legacy ones with a deprecation banner or delete. |
| B2 | `src/` flat-modules-vs-packages mix: `cartography.py`, `metrics.py`, `morphology_metrics.py`, `spatial_analysis.py`, `typology.py`, `urban_morphology.py` (flat) coexist with `svf_v2/`, `morphometry/`, `solar/`, `cfd_integration/`, `exposure/`, `visualization/` (packages) | Medium — implies which is "real" code | **One pass.** Either fold flat modules into packages or document why they stay flat. |
| B3 | `tests/test_patch_selection/` — memory says `src/patch_selection/` was deleted April 2026 | High — tests for deleted code | **Verify + delete or rewrite.** If the directory still has importable tests, they're either skipped or testing something that moved. |
| B4 | `notebooks/compare_vidigal_tls_lod2_solar_svf.ipynb` + `notebooks/explore_favelas.ipynb` — exploratory, not part of pipeline | Low — but `*.ipynb` is in gitignore and these are tracked exceptions | Either: graduate the canonical analysis to a script, or move to `notebooks/exploratory/` with clear non-canonical labelling. |
| B5 | `scripts/run_area_analyses.py` — orchestrator script that may be the entry point for a long-superseded "run everything for one area" flow | Low | Verify it still runs against current pipeline. If broken, delete or rewrite. |

### C. Top-level documentation gaps

| Doc | Current line count | Gap |
|---|---:|---|
| `README.md` | 289 | Solid pipeline walkthrough but missing: (1) one-paragraph problem framing for a first-time visitor, (2) explicit "What this repo is NOT" (CFD execution, manuscript), (3) Citation block, (4) license badge. |
| `ROADMAP.md` | 603 | Just resynced. Length is fine; structure is fine. |
| `CHANGELOG.md` | 417 | Already comprehensive. Verify it covers v5.5.0 once committed. |
| `CONTRIBUTING.md` | 159 | Needs verification: does it describe the *current* policy (commit on main, no PR-required, hook gates)? |
| `CITATION.cff` | 41 | Verify accurate authorship + a DOI placeholder if relevant. |
| `LICENSE` | (untracked-by-eye) | Read once, confirm it's the right one. |
| **`docs/README.md`** | **MISSING** | A `docs/` index telling readers what each subdir is. Currently `docs/` has `archive/`, `guides/`, `technical_report/`, `FAVELA_EXTRACTION_WORKFLOW.md`, `GPU_SVF_EXACT_VALIDATION.md`, `cfd_sampling_overrides.yaml` with no map. |

### D. Technical report gaps (highest priority — this is the deliverable)

The TR is **1069 lines, 11 sections + 3 appendices**, well-structured.
Recent fixes (5-site UMEP cross-val, §6.5 Blocken correction) brought it
to current. Remaining gaps for an external engineering review:

| # | Gap | Why it matters for engineers |
|---|---|---|
| D1 | **No author / contact / commit-hash / build-date metadata** at top | Engineers need to know who to ping with questions and what version they're reviewing |
| D2 | **Executive summary is project-internal**, not engineer-friendly | Lacks a 3-sentence statement of (1) what an engineer would do with this data, (2) what they should NOT trust yet, (3) what's pending |
| D3 | **No §0 Glossary / Nomenclature** | SVF, λp, λf, σh, Blocken radius, Tregenza-145, neutral log-law, ACH — these are domain shorthand and several have multiple definitions across the field |
| D4 | **No "Reproducibility" section** with explicit per-figure / per-table reproduction commands | "Run X to reproduce Fig 4" is the engineer's first instinct |
| D5 | **No "Failure modes & observability"** | An engineer reviewing for production use wants: how do we know if it broke? what does correct output look like? |
| D6 | **§8 Repository Structure** (line 781) — needs validation against actual current tree (e.g. mentions `src/patch_selection/` in older drafts?) | Stale repo-structure prose is a credibility tax |
| D7 | **No numerical-claims sweep** since the §6.5 Blocken bug | Every "X% of Y" or "all N patches satisfy Z" deserves an audit. The Blocken miss showed prose can drift even when constraint checks are right. |
| D8 | **Figure caption audit** — every figure should have axis labels with units, a caption that names the script that produced it, and an output path | Engineers will ask "where does this come from"; prevent that question |
| D9 | **Cross-reference integrity** — every "(see §X.Y)" must resolve | One broken link = "what else is wrong" |
| D10 | **PDF polish** — TOC, page breaks not splitting tables, running header with section name | Visual quality signals technical quality |
| D11 | **Appendix D: Engineering review checklist** | Tell reviewers explicitly what kind of feedback is most useful (methodological? code? pipeline?) |

### E. Onboarding gaps

| Path | Status | Gap |
|---|---|---|
| Fresh clone → working pipeline | README §"Pipeline walkthrough" covers this | Hasn't been re-tested on a fresh machine in months |
| Site onboarding | `site-onboarder` agent + `data/README.md` | Both are operator-facing; nothing for an *engineering reviewer* who isn't onboarding a site |
| **Engineering review path** | **MISSING** | No `docs/onboarding/engineering_review.md` saying "you have 1 day; here's the order to read things in" |
| Local setup troubleshooting (GDAL on Mac, GPU SVF, conda quirks) | Scattered through README | Concentrated troubleshooting doc would help |

### F. Tests / CI gaps

- 577 tests, single ruff + pytest CI workflow — solid baseline.
- No coverage reporting in CI.
- No PDF-build job in CI (so `technical_report.pdf` could silently bitrot).
- No link-check on `*.md` (broken refs slip in).
- `tests/test_patch_selection/` — verify against deleted source (see B3).

---

## Workstreams (4 parallel tracks, ~5 days total)

### Track A — Cleanup & repo structure (1–1.5 days)

Goal: a fresh `git clone` shows only files that earn their place.

**Day 1 morning** — high-confidence, parallel:

- A1. Delete `feature/gpu-svf-acceleration/` (`git rm -rf feature/`)
- A2. Delete `tests/*.md` design docs (keep `tests/README.md`)
- A3. Move `SKY_EXPOSURE_METHODOLOGY.md` → `docs/methodology/sky_exposure.md`
- A4. Move `STREET_SVF_USAGE.md` → `docs/methodology/street_svf.md`
- A5. `rm -rf logs/` locally
- A6. Decide on `requirements.txt`: delete unless something depends on it

**Day 1 afternoon** — needs decisions:

- A7. Audit `scripts/` redundancies (B1 list above) — for each pair, run both, decide which is current, deprecate or delete the other. **Commit one decision per pair.**
- A8. Audit `tests/test_patch_selection/` (B3) — verify what each test actually imports
- A9. Audit `src/` flat-modules-vs-packages (B2) — recommend folding into existing packages where natural

**Day 1 evening** — long tail:

- A10. `docs/archive/` cull — keep ≤3 retroactively useful docs, delete or move-to-tag the rest
- A11. `notebooks/` — graduate or label exploratory (B4)

**Validation:** `git status` after this track shows only intentional files. New entry in CHANGELOG.md.

### Track B — Top-level documentation (0.5 day)

Goal: someone landing on the GitHub front page knows in 30 seconds what
this is, who it's for, and how to start.

- B1. README rewrite — opening paragraph for a first-time visitor; "What this is / What this isn't" block; citation; license badge; link to TR; link to onboarding doc.
- B2. CONTRIBUTING.md verification — does it describe current policy (work on `main`, no PRs, hook gates)? If not, rewrite to match reality.
- B3. CITATION.cff — verify authorship; add a DOI / ORCID placeholder if relevant.
- B4. New `docs/README.md` — table of contents for `docs/` subtree.
- B5. CHANGELOG.md — confirm v5.5.0 entry covers the April–May arc.

**Validation:** README answers "what / who / how to start" in the first screen.

### Track C — Technical report hardening (2–3 days, **highest priority**)

Goal: a senior engineer can review the report and have actionable
methodological / pipeline questions, not "what is this acronym" questions.

**Day 2 morning** — surface fixes:

- C1. Add metadata block at top: authors, contact, version (commit hash from `git rev-parse --short HEAD`), build date.
- C2. Add §0 Glossary / Nomenclature — every domain term defined once, with units. (SVF, λp, λf, σh, H/W ratio, Tregenza-145, Blocken radius, neutral log-law, ACH, fetch, calm fraction, etc.)
- C3. Strengthen Executive Summary — three sentences for an engineering reader: what to use, what's pending, what the open risks are.

**Day 2 afternoon** — content audits:

- C4. Numerical-claims sweep — grep every percentage, count, and comparator; verify against the source data file. Use the §6.5 Blocken miss as the canonical class of bug to look for.
- C5. Cross-reference integrity — every "(see §X)" resolves.
- C6. §8 Repository Structure validation — run a script that diffs the prose against the actual `src/`, `scripts/`, `data/`, `outputs/` trees.

**Day 3 morning** — figure + caption audit:

- C7. For each of the 19 PNGs in `figures/`: caption names the producer script + output path; axes have units; colorbars labelled.

**Day 3 afternoon** — net-new sections:

- C8. Add §12 Reproducibility — for every numbered figure and table, the exact command to reproduce it. Group by site / by phase.
- C9. Add §13 Failure modes & observability — what each pipeline stage does on success vs failure; what the validators (`data-contract-checker`, `sampling-auditor`, `cfd-results-ingestor`) catch; how to interpret their output.
- C10. Add Appendix D — Engineering review checklist (what kind of feedback is most valuable; code review vs methodology review vs pipeline review).

**Day 3 evening** — polish:

- C11. PDF rebuild + visual review (TOC, page breaks, header/footer, table breaks).
- C12. Spell-check pass (`aspell` or equivalent) — once.

**Validation:** every "(see §X)" resolves; every numerical claim links to source; every figure has units and producer script; PDF rebuilds clean.

### Track D — Onboarding (0.5 day)

Goal: new engineering reviewer has an explicit, time-boxed reading path.

- D1. New `docs/onboarding/engineering_review.md` — 1-day reading plan: README (10 min) → TR Executive Summary + §1 + §10–13 (1 hr) → walk through one site's outputs end-to-end (2 hr) → focused deep-read of §3–§7 (3 hr) → review checklist (Appendix D).
- D2. New `docs/onboarding/local_setup.md` — concentrated troubleshooting: GDAL on macOS/Linux/Windows, conda env quirks, GPU optional path, common errors.
- D3. Pre-run smoke test — explicit "this command should succeed in <2 min on a fresh clone" loop. Likely: `pytest tests/ -m "not integration" -q --tb=short`.

**Validation:** a colleague who has never seen the repo can reach "first figure rendered locally" within 1 hour using only the onboarding docs.

---

## Sequencing (compressed 5-day plan)

| Day | Tracks | Output |
|---|---|---|
| **1** | A (cleanup) + B1 (README polish) | Clean working tree; README front-loaded for engineering audience |
| **2** | C1–C6 (TR metadata, glossary, exec summary, audits) | TR fixed at the surface + content-validated |
| **3** | C7–C12 (figures, new §12/§13, App D, PDF polish) | TR ready for external eyes |
| **4** | B2–B5 + Track D | Top-level docs done; onboarding path explicit |
| **5** | Validation gates (below) + buffer for revisions | Tag a pre-review commit; send to engineering team |

This compresses with parallelism: A1–A6 can be a single morning's work
(they're independent), and Track C is the only one that genuinely
requires multiple sequential days.

---

## Validation gates (must pass before sending to engineering)

- [ ] **Fresh clone test** — wipe `~/MorphoFavela`, `git clone`, follow README, `pytest tests/ -m "not integration"` passes; first figure renders locally
- [ ] **PDF rebuilds clean** — `python docs/technical_report/build_pdf.py` succeeds; PDF size sane; no missing-figure boxes; TOC populated
- [ ] **Cross-reference scan** — script (or grep + manual) confirms every "(see §X)" in TR resolves
- [ ] **Numerical-claims scan** — every percentage / count / comparator in TR has a verifiable source (script + output file path)
- [ ] **Figure audit** — every figure in `docs/technical_report/figures/` has a caption that names its producer script and an axis with units
- [ ] **No tracked junk** — `git ls-files | grep -E "(feature/|logs/|\.egg-info/|tests/.*\.md$)"` returns empty (except `tests/README.md`)
- [ ] **CONTRIBUTING.md describes actual policy** — branch model, commit format, test/PR requirements, hook gates, who to ping
- [ ] **CI passes on the pre-review commit** — including ruff and pytest jobs
- [ ] **Engineering review checklist** (Appendix D) is concrete enough that a reviewer can produce useful feedback in 1 day

---

## Out of scope (deliberately deferred)

- VDG-P07 ingestion — calendar-gated on MIT ORCD, separate workstream
- Nature Cities manuscript draft — separate deliverable; the TR is for engineering review
- Solar irradiance cross-validation — deferred per ROADMAP, not on critical path
- 5 m / 2 m grid resolution sensitivity — deferred per §10.4
- Re-onboarding Cidade de Deus — gated on upstream building-data fix
- Hook false-positive characterisation — separate 2-week soak window
- Graduating `src/cfd_integration/io.py` to versioned-stable — gated on VDG-P07

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Numerical-claims sweep finds another §6.5-class bug | **Medium** | That's the point of the sweep; build in 0.5-day buffer for fixes |
| Fresh-clone test fails on macOS/Windows due to GDAL | Medium | Onboarding doc concentrates troubleshooting; include a Docker/Devcontainer fallback |
| Engineering reviewers focus on code style, not methodology | Low–Medium | Appendix D guides feedback toward methodology / pipeline; CONTRIBUTING.md links code-style questions to ruff config |
| Track C balloons (TR is 1069 lines, lots of surface) | Medium | Day 3 evening is buffer; if §12/§13 aren't done, ship without them and add as a follow-up |
| Cleanup deletes something used by an undocumented downstream | Low | Each deletion is a separate commit; revert is trivial |

---

## Suggested first commit

After this plan is approved, the natural first commit is:

```
chore(repo): cleanup pass 1 — remove orphan feature/, tests/*.md, top-level methodology drift
```

…containing items A1–A4 and A6, with the ROADMAP.md update from earlier
today rolled into the same push. That gives a clean working tree to
start Track C against.
