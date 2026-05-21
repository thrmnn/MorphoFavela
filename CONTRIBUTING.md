# Contributing

This is a research codebase under active development for a Nature Cities
submission. The bar is reproducibility + clean diffs over throughput.

## Development setup

```bash
git clone https://github.com/thrmnn/MorphoFavela.git
cd MorphoFavela

# Either Conda (recommended for GDAL / GEOS native deps)
conda create -n IVF python=3.11
conda activate IVF
pip install -e ".[dev]"

# or a plain venv (you'll need GDAL + GEOS installed via apt/brew)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional GPU stack for SVF v2 ray-casting:

```bash
pip install -e ".[gpu]"  # adds torch + pytorch3d
```

The full pipeline assumes input rasters + footprints under
`data/{site}/` — see [`data/README.md`](data/README.md) for the input
contract.

## Running tests

```bash
make test          # full suite, fail-fast
make test-fast     # only the synthetic-geometry tests (≤ 5 s total)
python -m pytest tests/test_cfd_integration -v   # single-module deep dive
```

Tests are organised by module:

- `tests/test_cfd_integration/` — 46 tests (schema, I/O, aggregation,
  metrics, weighting). Reference standard for what a clean test suite
  in this repo looks like.
- `tests/test_svf_v2/` — 120+ tests across compute, scene, ray-casting.
- `tests/` (root) — module-level smoke + correctness tests.

## Code style + quality

All code is formatted and linted with [ruff](https://docs.astral.sh/ruff/).

```bash
make format        # ruff format
make lint          # ruff check
```

Pre-commit hooks (configured in `.pre-commit-config.yaml`) run ruff
format + check on every commit. Install once after cloning:

```bash
pip install pre-commit
pre-commit install
```

CI (`.github/workflows/ci.yml`) enforces `ruff check`, `ruff format
--check`, and `pytest -m "not integration"` on every push and PR to
`main`.

## Commit conventions

[Conventional Commits](https://www.conventionalcommits.org/) with the
following types:

- `feat:` — new capability (script, module, figure)
- `fix:` — bug fix (always cite the symptom in the body)
- `refactor:` — code reorganisation without behaviour change
- `docs:` — documentation only
- `test:` — test additions / improvements
- `chore:` — tooling, dependencies, gitignore, CI
- `style:` — formatting only

Scope examples (parenthetical): `(wind-rose)`, `(svf)`, `(cfd-integration)`,
`(sampling)`, `(report)`. Subject ≤ 70 chars, imperative mood. Body
explains *why* (constraint, prior incident, design tradeoff) — the *what*
is in the diff.

Each commit should leave the working tree in a state where `make test`
and the pipeline scripts still run.

## The technical report is part of the deliverable

`docs/technical_report/technical_report.md` is the canonical project
description, distributed alongside the code. **Update it in the same
commit as any code change that affects what it documents.** The PDF is
rebuilt with:

```bash
python docs/technical_report/build_pdf.py
```

Triggers requiring a report update are spelled out in
[`CLAUDE.md`](CLAUDE.md#technical-report) — read those before
committing pipeline-affecting work.

## CFD integration boundary

Simulation execution lives in a separate repo at `~/Airflow` (not in
this codebase). Do not implement OpenFOAM case generation, mesh
preparation, or HPC submission here — point work toward that repo.

This repo:

- Produces the per-patch sampling under
  `outputs/{site}/sampling_cfd/campaign_sampling/patches/`.
- Specifies the CFD I/O contract in
  [`src/cfd_integration/README.md`](src/cfd_integration/README.md).
- Ingests CFD outputs that arrive at
  `data/{site}/cfd_results/{patch_id}/{wind_direction}/` via
  `src/cfd_integration/`.

## Project subagents

The repo ships seven Claude Code subagents under
[`.claude/agents/`](.claude/agents/) — four **validators** (read-only
contract checks) and three **workflow accelerators** (multi-step
orchestration with explicit hand-offs at manual steps).

### Validators (read-only; never modify files)

| Agent | Use it when |
|---|---|
| [`data-contract-checker`](.claude/agents/data-contract-checker.md) | Pulled new data, added a site, edited `data/README.md`, or about to run a pipeline script on `data/{site}/` |
| [`sampling-auditor`](.claude/agents/sampling-auditor.md) | Re-ran `run_campaign_sampling.py`, before submitting patches to CFD execution, or to confirm campaign integrity |
| [`report-sync-auditor`](.claude/agents/report-sync-auditor.md) | Before committing pipeline / figure / sampling changes — flags drift between code and `docs/technical_report/` |
| [`numerical-claims-auditor`](.claude/agents/numerical-claims-auditor.md) | Before sending the technical report for external review, after any sampling/grid regeneration, or after a TR edit that touches numbers — extracts every numerical claim and verifies it against a traceable source. Targets the §6.5-class prose-drift bug. |

### Workflow accelerators (modify code, run scripts, hand off at manual steps)

| Agent | Use it when |
|---|---|
| [`site-onboarder`](.claude/agents/site-onboarder.md) | Adding a new favela site — walks the 7-step `data/README.md` checklist, stops loudly at the manual DTM-clip step |
| [`wind-ingestion`](.claude/agents/wind-ingestion.md) | (Re)building `wind_rose.json` from INMET or ASOS — encodes the 3 known INMET quirks |
| [`cfd-results-ingestor`](.claude/agents/cfd-results-ingestor.md) | CFD outputs returned from `~/Airflow` to `data/{site}/cfd_results/` — validates schema and flags producer drift |

See [`.claude/agents/README.md`](.claude/agents/README.md) for design
rules and how to add a new one. Agents are loaded at session
startup — restart Claude Code after pulling new agents.

## Reporting issues

Open a GitHub issue with:

- A minimal reproducer (script invocation + inputs).
- Expected vs actual behaviour.
- Environment (`python -V`, OS, conda vs venv, GPU stack present).

For data quality questions specific to a site, check
`data/{site}/PROVENANCE.md` (when present) before opening — many
"bugs" are upstream raster gaps documented there.
