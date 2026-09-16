# ROBUST — the reviewer robustness dossier for P1 v2

Opened 2026-09-16 (cycle 3). Owner: agent. Repo: MorphoFavela.
Depends on: VENTAXIS (for the second axis's definition of record).
Verify gate: `TMPDIR=/tmp python -m pytest tests/test_robustness_dossier.py -q`
then `.claude/verify-cmd`.

## What this is

Every acceptance and sensitivity number P1 v2 stands on is already final and
sitting in a run of record. None of them is assembled anywhere a reviewer — or
the PI, writing the Methods — could read in one pass. This task is **assembly by
code**, not new analysis. Nothing is recomputed; nothing new is claimed.

**This is a STAGED artifact.** It is written into `runs/robustness_<UTC>/` and
goes no further. Promotion into `shared/figures`, `papers/`, or anything the
public hub serves is a release-boundary crossing and belongs to the PI. Stage
only.

## Scope

**1. `scripts/build_robustness_dossier.py`** — reads, never types, and writes
`runs/robustness_<UTC>/{dossier.json,dossier.md,manifest.json}`. Every row it
prints carries the ledger id (or the run-file JSON pointer) it came from, so any
row can be re-derived from the artifact alone. Source of record:
`runs/wp07_ledger_20260915T184619Z/ledger.json` — 371 entries, all `status:
final` — unless VENTAXIS has regenerated it, in which case use the newest
`runs/wp07_ledger_*/ledger.json` and record which one in the manifest.

**2. The sections.** Read every value; the pointers below tell you where to look,
not what the value is.

  - **Engine acceptance.** The `engine.crossref.*` entries (r, median |Δ|, p95
    |Δ|) plus the analytic self-checks the engine was accepted against —
    the unobstructed identity, the infinite-canyon case, the isolated-wall
    shadow. The canyon reference value and the check outcomes are in the
    WP-02 horizon run and `docs/wp02_horizon_engine_spec.md`; read them.
    State which cross-reference variant the ledger chose and why — the ledger's
    `_meta.engine_acceptance_source` says so in one sentence; quote it.
  - **Domain sensitivity (G3).** The 9 grid variants, the per-favela percentile
    spread (`derived.spread.*`), and — the load-bearing one —
    `derived.rank_invariant_across_grid`. The honest framing is that the
    *position* moves with the grid while the *ordering* does not; say exactly
    that, with both numbers.
  - **Sky resolution.** One resolution end-to-end. Read the count from
    `src/brisa_solar/constants.py` (import it; do not type the integer — it is a
    literal this project forbids typing) and state that no second resolution
    exists in any code path feeding a pooled number.
  - **Irradiance input.** The two EPW stations, their annual GHI, and the
    percentage difference between them — all in `data/epw/epw_inventory.json`.
    Compute the percentage from the two values; do not copy a percentage.
  - **Second-axis validity.** Cite `docs/ventaxis_canonical.md` (VENTAXIS) for
    the definition, and give the constraint-count shares per site from the
    `wp06.*` ledger entries.
  - **Ground-truth comparison (G2).** The WP-03 TLS result: what was compared,
    the measured registration residual, and the alley-class outcome. It is a
    *negative* result with a confound — an observer-elevation confound was
    confirmed, and agreement holds only for street points in alleys below a
    height threshold. Report it as it is; do not soften it. The card
    `g2_validity_floor` is open and the PI has not ruled, so the dossier states
    the finding and names the open decision rather than resolving it.
  - **Coverage.** Citywide cell count and resolution, per-site n, and the
    boundary/footprint vintages. The methods epoch table is generated from file
    metadata (C′ plan §1.6) — generate it, never type a vintage.
  - **Declared limitations.** At minimum: vegetation/canopy is out of scope
    (descoped 2026-09-10; no DSM exists on disk) and may not be used to explain
    any shortfall; the façade surface is **NOT accepted** (WP-04 cross-reference
    r 0.88/0.64) and no façade number appears anywhere in the ledger; the
    2.5D model's validity floor is the open G2 question above.

**3. `tests/test_robustness_dossier.py`** — asserts the dossier builds, that
every numeric row carries a resolvable ledger id or JSON pointer, and that no
row's value differs from the source it cites. That last assertion is the whole
point of the artifact: make it a real comparison, not a smoke test.

**4. Honesty rails.** The dossier is a P1 artifact, so write it in C′ vocabulary
and put its output globs inside `scripts/lint_p1_tokens.py`'s scan (VENTAXIS is
widening those globs in parallel — coordinate by reading the file, and if the
glob you need is already there, say so rather than adding a duplicate). Run
`python3 scripts/lint_p1_tokens.py` and `python3 scripts/lint_p1_columns.py` and
paste both exit codes.

## Explicitly OUT

- Recomputing anything. If a number you need does not exist, that is a finding
  to report, not a run to launch.
- Promoting the dossier anywhere outside `runs/`.
- Manuscript prose (paper voice is the PI's, always-ask).
- Resolving `g2_validity_floor` or any other open card.
- The favela-vs-formal comparison in any form — it is a red line (L1) and no
  such field exists in any run of record. If you find one, stop and report.

## Never

- Never type a number that exists in a file — read it by code.
- Never type the citywide sky-patch count as a literal; import it from
  `src/brisa_solar/constants.py`.
- Never say "WHO" for the 2 h floor — Athens Charter (1943), Point 26.
- Never present a negative or unresolved result as settled.
