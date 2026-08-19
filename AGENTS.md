# MorphoFavela — agent roster & guidelines

Six read-mostly specialized subagents (`.claude/agents/`), all oriented around
data/claims integrity rather than pipeline execution — this repo's tool
surface is morphometrics + data contracts, not HPC/CFD. Written 2026-08-19 as
documentation of an existing convention, not a new process.

## Roster

| Agent | Triggers on | Boundary |
|---|---|---|
| `site-onboarder` | adding a new favela site to the dataset | read/edit/write, stops at manual DTM-clip (deliberate non-automation) |
| `wind-ingestion` | (re)building `data/{site}/wind_rose.json` | read/bash — INMET/ASOS ingestion, 3 known quirks encoded |
| `data-contract-checker` | before running pipeline scripts / after pulling data | read-only — validates against `data/README.md` |
| `sampling-auditor` | after `run_campaign_sampling.py`, before CFD submission | read-only — stratification/spacing/count contracts |
| `cfd-results-ingestor` | CFD results arrive from `~/SCL/SCR/Airflow` | read-only by default; aggregation gated behind an explicit flag |
| `report-sync-auditor` | reviewing a diff for technical-report drift | read-only — the CLAUDE.md sync triggers, mechanized |
| `numerical-claims-auditor` | before external review of the technical report | read-only — every numerical claim vs. its traceable source |

`numerical-claims-auditor` is the newest, and its own frontmatter names the
incident that justified it (the §6.5 Blocken miss: claimed "≥ 150 m", actual
114 m). That's the model for when a new agent earns a slot — not a hypothetical
risk, a real one that already happened once.

## When a task earns a new named agent

- **New agent** when it's a distinct failure mode with its own recovery recipe,
  or needs a genuinely different read/write boundary than any sibling.
- **Reuse** when the trigger and boundary match an existing agent even if the
  surface symptom differs.
- **Inline** for anything that doesn't recur.

## Flagged, not yet built

No agent audits *agent staleness itself* — a trigger description that no
longer matches the repo's current file layout. One layer removed from the
derive-vs-hardcode bug class (`scripts/checks/check_derived_constants.py`,
landed 2026-08-19) but the same shape. Track D territory — periodic self-check,
not a new named worker, until there's a first real instance to design against.

See `~/SCL/SCR/docs/ops_council_synthesis_2026-08-19.md` §3 for the full reasoning.
