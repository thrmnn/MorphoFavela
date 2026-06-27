# Paper-integration track — standing plan (2026-06-26)

Branch `track/paper-integration`, off a clean 660-green `main` (5a8aedc). All council-roadmap
(7 items) and repo-audit (7 streams) work is shipped. This track closes the gap between **what
the code/outputs already prove** and **what the technical report (the primary local deliverable)
actually says**.

## The gap (grounded, 2026-06-26)

Shipped in code + `outputs/` + `outputs/paper_figures/exports/`, but **NOT in the TR**:
- **Three geometric scalars** — lateral-connectivity (`run_lateral_connectivity.py`, ρ=+0.49),
  2D ventilation-susceptibility (`run_ventilation_susceptibility.py`, 41.8% skimming∩deep),
  wind-exposure (`run_wind_exposure.py`, near-isotropic). PNGs exist; not copied into
  `docs/technical_report/figures/`; zero prose in `technical_report.md`.
- **`morphotype_shape`** additive separable axis (LOSO-ARI 0.18 cell / 0.44 morphotope) —
  decision log D21–D23, but no TR §5.5 mention.
- **Predictor-flip hardening** — spatially-blocked CV, block-bootstrap CIs (CI-overlapping),
  VIF audit, blind-map external validation (AUC-PR 0.76). §5.5 has the headline flip numbers
  but not the rigor that makes them defensible.

Already in: MAUP §10.9 (`maup_regime_shares.png` travelled); dissolved-λf re-baseline; regime taxonomy.

## Engineering contract (unchanged, non-negotiable)
Additive suffixed columns only · never overwrite canonical λf / morphotype (bit-for-bit) ·
every shift is a documented A/B, never a silent re-baseline · spatially-blocked CV for predictor
claims · local compute only · reversible · **rebuild the PDF in the same commit as a TR.md edit.**

## Priorities (rank = deliverable-value × feasibility, all fully local)

1. **[P1] TR integration of the three geometric scalars** — new results subsection (geometric
   ventilation *tendencies*, τ-gated, NOT adequacy). Copy 3 PNGs into TR figures/. Frame +
   provenance captions. Touches `technical_report.md` + PDF.
2. **[P1] TR integration of `morphotype_shape` + predictor hardening** — §5.5: separable shape
   axis (ARI A/B), spatial-CV / block-bootstrap CIs / VIF / blind-map. Touches §5.5 + PDF.
3. **[P2] 20 m-grid re-baseline + Fig S3** (task #24) — the resolution variant §10.4/§10.9
   already flag as "queued". Pure `outputs/`, no TR-prose collision (hands a figure + ref note
   back to the spine).
4. **[P2] Mingze HTML local refresh with re-baselined numbers** (task #41) — local file only;
   the *upload* stays blocked (user-driven).
5. **[P3] Verification gates** — numerical-claims-auditor + report-sync-auditor over the final
   TR; reconcile §12.2 test count.

## The serialization constraint (why "parallel" is bounded)

`technical_report.md` is one file and the PDF is one 30 MB binary → **TR-prose work (#1, #2) is a
serial spine, PDF rebuilt ONCE at the end** (same rule that governed the last repo-audit run).
What genuinely parallelises around that spine (file-disjoint):
- **Side-stream A** — figure travel + hub/gallery/work_queue refresh (figures dir + hub, no .md prose)
- **Side-stream B** — 20 m re-baseline + Fig S3 (#3, pure outputs/)
- **Side-stream C** — Mingze HTML refresh (#4, separate tree)
- **Side-stream D** — read-only auditors (#5) as gates after the spine lands

Autonomous shape: spine #1→#2→single PDF rebuild→#5; A/B/C concurrent; per-section drafting +
adversarial numerical verification fan out inside each spine step.

## Hard blockers (NOT startable autonomously)
- **CFD-τ** — separate repo, no execution here. Gates per-cell adequacy + results chapter §7.4/§11.
- **Ray-caster vs Radiance/SOLWEIG x-val** — external tooling/install; TR §10.3 placeholder.
- **Mingze WeTransfer upload** — the local HTML refresh is fine; the *send* is the user's.
- **git-history rewrite (#39)** — explicitly excluded; no force-push/history-rewrite.
- **brisaverse manuscript edits** — external repo. The TR here is the in-scope local proxy.

## EXECUTION OUTCOME (2026-06-27) — scope 1–5 ✅ COMPLETE, pushed, 710 green

Multi-agent workflow (4 parallel drafters/side-streams → serial integrator → 2 read-only
auditors) + a follow-on correctness fix. Commits `daeeca8`→`aa63446` on `track/paper-integration`.

- **#1 TR §5.6** (`08f1743`) — three τ-gated geometric ventilation scalars integrated
  (lateral-connectivity, regime×depth susceptibility 41.8 % skimming∩deep, near-isotropic
  wind-exposure); 3 PNGs travelled; numerical audit 54/54, report-sync PASS.
- **#2 TR §5.5** (`08f1743`) — `morphotype_shape` separable axis (ARI 0.18 cell / 0.44 tissue,
  canonical frozen) + predictor-flip hardening (spatially-blocked CV, overlapping block-bootstrap
  CIs → direction not separated gap, VIF 1.1–3.0, external blind map AUC-PR 0.76, Fig 5.5e).
- **#3 20 m grid + Fig S3** (`e314ade` fig script, `aa63446` fix) — **uncovered a confound**: the
  20 m grids were still on the pre-dissolve *summed* λf, so the published "+18.4 pp skimming at
  20 m" MAUP claim was an artifact. Parametrized `migrate_lambda_f_dissolve.py` by `--grid-suffix`,
  dissolve-migrated the 20 m grids; the like-for-like effect REVERSES — coarsening lowers λf
  median −40.7 % and shifts skimming **−34.1 pp** (65→31 %) into wake (+28.8 pp), the correct
  frontal/plan geometric scaling. §10.9 + §4.6 corrected, both MAUP figures regenerated, guard
  test `tests/test_migrate_lambda_f_dissolve.py` pins the suffix→path mapping. Closed task #24.
- **#4 Mingze HTML** — refreshed locally vs current canonical Vidigal solar (MAE 1.77→1.70 h,
  Pearson 0.565→0.589, etc.); local-only, upload remains user-driven. Closed task #41.
- **#5 Verification** — workflow auditors PASS + a fresh post-correction numerical audit on the
  reversed §10.9 (16/16 traced, signs correct). Canonical 10 m λf bit-for-bit untouched throughout.

Watch-item (advisory, from the final audit): the §10.9 bracket strings "25–39 pp" (true
25.0–38.6) and "2.7–4.4×" (true 2.68–4.41) are rounded; a future regeneration nudging Maré's
swing past 39 pp or a site's over-count outside 2.65–4.45 would warrant re-checking those two.
