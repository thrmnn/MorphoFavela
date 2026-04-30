# Project instructions for Claude sessions

Short, action-oriented rules for working in this repo. Code structure and
design are documented in `README.md`, `ROADMAP.md`, and module docstrings —
this file only contains workflow rules Claude should follow but that aren't
obvious from the code.

## Technical report

**The technical report (`docs/technical_report/`) is the project's primary
deliverable document. Keep it in sync with the code.**

When you make changes that affect the report, update `technical_report.md`
and rebuild the PDF in the same commit:

```bash
python docs/technical_report/build_pdf.py
```

Triggers that require updating the report:

- **New or modified pipeline script** in `scripts/` that changes outputs
  → update §3–§7 as relevant
- **New morphometric indicator or grid column** → update §4.2
- **Sampling allocation changes** (patch counts, strata rules, spacing) →
  update §6 and regenerate allocation figures first
- **New site added / site removed** from the campaign → update §1, the
  summary tables, and regenerate all figures
- **CFD results ingested** → update §7.4, §11, add results chapter
- **Regenerated paper figures** → copy the relevant PNGs into
  `docs/technical_report/figures/` before rebuilding the PDF

Do NOT update the PDF for:

- Local WIP edits to the markdown that aren't being committed
- Fixing typos in code comments (unrelated to the report's content)
- Changes inside `outputs/` (ignored by git anyway)

**Commit both files (`technical_report.md` and `technical_report.pdf`)
together.** A stale PDF is worse than no PDF because readers trust the
rendered artefact.

## Pre-commit reflex

Three muscle habits that are enforced at runtime by
`.claude/hooks/check_report_sync.py` (advisory or blocking) but should
also be internalised so the hook is a backstop, not the primary gate:

1. **Rebuild the PDF in the same commit as a `technical_report.md`
   edit.** `python docs/technical_report/build_pdf.py`. The hook
   blocks commits where the .md is staged without the .pdf
   (`exit 2`), so a forgotten rebuild is caught — but reflex beats
   recovery.
2. **Commit predecessor work before launching anything > 10 min.**
   Background jobs (UMEP shadow-cast, CFD runs, large-grid
   regenerations) die with the agent process. If a partial result
   was uncommitted when the session ended, it is gone. The pattern
   is: every successful local result gets committed before the next
   experiment kicks off, even if the diff is tiny. See
   `feedback_long_running_jobs.md` in memory for the full rule.
3. **Stage the test in the same commit as the new behaviour.**
   `feat(...)` and `fix(...)` commits should land their tests at the
   same time as the code path. The hook surfaces an advisory when
   `feat/fix` touches `src/` or `scripts/` without staging a `tests/`
   file. Override is implicit (advisory does not block); the right
   correction is to stage the test, not to ignore the advisory.

## Memory system

Project-specific facts about pipeline state, site data quirks, and
outstanding work live in `.claude/projects/-home-theo-IVF/memory/`. Update
those memories when you learn something durable — not in this CLAUDE.md.

## Project subagents

Six project-scoped subagents under `.claude/agents/` codify this
project's contracts. Reach for them before re-implementing the
checks they encode:

- **Validators** (read-only): `data-contract-checker` (per-site
  inputs against `data/README.md`), `sampling-auditor` (CFD patch
  campaign integrity), `report-sync-auditor` (diff vs.
  `technical_report.md`/`.pdf`/`figures/` per the triggers above).
- **Workflow accelerators**: `site-onboarder` (new site through
  the 7-step `data/README.md` checklist; halts at the manual DTM
  clip), `wind-ingestion` (INMET/ASOS → `wind_rose.json`,
  encoding the 3 known INMET quirks), `cfd-results-ingestor`
  (validate `data/{site}/cfd_results/` returns from `~/Airflow`
  against the `src/cfd_integration/` schema; flags producer
  drift).

Design rules and how to add a new agent live in
`.claude/agents/README.md`. Agents are loaded at session startup
— after pulling new ones, restart Claude Code to register them.

## Incremental commits

This repo's main branch is protected in practice by the fact that work
happens directly on `main`. Commit and push in small, self-contained units
after each logical step; don't accumulate sprawling changesets. Every
commit should leave the working tree in a state where `pytest tests/` and
the pipeline scripts still run.

## Pipeline outputs are not tracked

`data/` and `outputs/` are gitignored. The exception is
`outputs/paper_figures/*.py` and its README. When you need a figure to
travel with the repo (e.g., for the technical report), copy it into
`docs/technical_report/figures/`.

## CFD campaign boundary

Simulation execution happens in a separate repository. This repo:

- Produces the sampling (`outputs/{site}/sampling_cfd/campaign_sampling/patches/`)
- Specifies the CFD output contract (`src/cfd_integration/README.md`)
- Ingests results via `src/cfd_integration/` when they arrive at
  `data/{site}/cfd_results/{patch_id}/{wind_direction}/`

Do not implement OpenFOAM case setup or mesh generation here. If asked,
point to the CFD repo.

## CFD patch sampling lives in scripts/

The 119-patch CFD campaign (stratified on SVF × slope × λp, 100 m-
diameter circular analysis patches) is produced by two scripts:

- `scripts/run_pilot_sampling.py` — 12–15 patches per site, stratum
  coverage + greedy maximin spacing
- `scripts/run_campaign_sampling.py` — incremental top-up to 22–25
  per site with SVF-priority weighting

Outputs land in `outputs/{site}/sampling_cfd/`. CFD runtime consumers
(aggregation, metrics, weighting) are in `src/cfd_integration/`. An
earlier clustering-based approach in `src/patch_selection/` was
deleted in April 2026 — don't recreate it.
