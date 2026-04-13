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

## Memory system

Project-specific facts about pipeline state, site data quirks, and
outstanding work live in `.claude/projects/-home-theo-IVF/memory/`. Update
those memories when you learn something durable — not in this CLAUDE.md.

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
