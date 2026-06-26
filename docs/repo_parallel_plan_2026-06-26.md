# Full-repo audit → parallel execution plan (2026-06-26)

Output of a 7-auditor repo council (tests · code-quality · doc/TR-sync · data-pipeline ·
scientific-rigor · figures · hygiene) + chair synthesis. Drives autonomous execution.

## Engineering contract (unchanged, non-negotiable)
Additive suffixed columns only · never overwrite the canonical λf
(`lambda_f_canonical.json`, test-pinned, bit-for-bit) or the canonical morphotype fit ·
every downstream shift is a documented A/B sensitivity, never a silent re-baseline ·
spatially-blocked CV for predictor claims · local compute only · reversible.

## Waves

**WAVE 1 — 6 fully file-disjoint streams, run concurrently (worktree-isolated agents):**
- **S2 MAUP sensitivity (#7)** — `run_maup_sensitivity.py` + test; regime shares + σH/λf at
  10 m vs 20 m as an explicit A/B; TR methods appendix. Reads the canonical lock, never writes.
- **S3 test-guard hardening** — golden λf/regime fixture (always-on, bit-for-bit), predictor
  FLIP synthetic regression pin, CONT_FEATURES schema fixture, version fast-fail in conftest.
- **S4 CFD-provenance** — `synthetic`/`provenance` fields through `cfd_integration` schema+io;
  enforce cardinal `wind_direction`; resolves the standing synthetic-data MEMORY warning.
- **S5 figure-gate + provenance** — wire the text-overflow gate into every figure script,
  shared provenance-caption helper, fix the `fig_0_7` stem, prune stale PNGs. (Serialize
  internally on `fig_style.py`; ONE agent owns all of S5.)
- **S6 nav-doc resync** — README / ROADMAP / docs/README (~10 commits stale). Doc-only; does
  NOT touch `technical_report.md`.
- **S7 code-hygiene + deps** — pyproject deps, strip 142 dead `noqa`, PEP-585/604 typing,
  de-hardcode brisa_deck output paths, stale-branch REPORT only.

**WAVE 2 — S1 canonical-morphometry, ALONE (after Wave 1 quiesces):**
1. Unify built-cell + phantom-tower invariants behind shared tested helpers (contract test
   pins n=64,389).
2. `SIGNATURE_FEATURES` literal-tuple pin (pre-req guard).
3. Shape/grain morphotype re-fit (#3) — aggregate existing per-building shape/grain to grid,
   VIF-screen vs λp, re-fit GMM k=6 into an **additive `morphotype_shape` column** (canonical
   `morphotype` untouched), report LOSO-ARI A/B at morphotype + morphotope level.
4. Bootstrap-ARI + machine-readable `k_selection_summary.json`.

## Integration rules (the only serialization points)
- **PDF rebuilt ONCE on main** after Wave-1 merge — S2/S5 edit `.md`/figures but do NOT
  rebuild the 30 MB PDF in their worktrees (binary-merge hazard).
- `technical_report.md` owned by **S2 only** in Wave 1; S5 leaves any TR ref fix as a note.
- S1 runs strictly after Wave 1, so S7's sweep of the shared scripts lands first and S1 edits
  the cleaned versions — no collision.

## Excluded (hard-blocked) — not in any stream
CFD-τ · ray-caster vs Radiance/SOLWEIG · Mingze upload · BRISA manuscript edits · git
force-push/history-rewrite/branch-deletion (S7 emits a REPORT only).
