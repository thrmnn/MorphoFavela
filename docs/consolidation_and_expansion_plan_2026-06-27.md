# Consolidation + expansion plan (2026-06-27)

Drafted at the close of the paper-integration track to (a) audit and consolidate
everything shipped, and (b) stage the next body of analytical work with a concrete
agent-team / dynamic-workflow design. Nothing here executes yet — this is the map
for the next session.

## 1. Grounded state (verified, not recalled)

**Git.** Work happens on `track/*` branches; `main` is the single integration line.
- `track/paper-integration` — **5 ahead / 0 behind** `main` → a clean fast-forward.
  Commits `daeeca8`→`8b8d888`. This is the only track with unmerged work.
- `track/roughness` — **0 ahead / 101 behind** → already merged into `main`.
- `track/morpho-signature` — **0 ahead / 112 behind** → already merged.
- `origin/track/morphotope`, `origin/track/viz` — **0 ahead** → stale.
- `origin/feature/publication-report` (archived, PR #9 closed), `origin/feature/pyviewfactor` — survivors.

**Tests.** 710 passed, 22 skipped (`406 s`). Suite green.

**Technical report.** §5.5 (morphotype_shape + predictor hardening), §5.6 (three
τ-gated geometric scalars), §10.9 (MAUP, corrected) integrated; PDF rebuilt.
BUT `docs/tr_audit.md` still has open items (all local, no external dependency):
- ☐ §5.5 circularity caveat (SVF ≈ f(λp, H/W); winter-sun is the cleaner held-out signal)
- ☐ §5.5 support caveat (~35% of cells carry an observer; per-type 0.23–0.59)
- ☐ §5.5 conditional-type caveat (T2/T3 flatland-conditional, not universal)
- ☐ Embed validation figures (recurrence + experience dotplots → §5.5; roughness_validity → §6.6, currently figureless)
- ☐ §12 reproducibility — add Stage 7 (signature/morphotope) + Stage 8 (roughness)
- ☐ Heading hierarchy — §5.5 is `##` while §5.1–5.4 are `###`
- ☐ Version/date header stale (v1.0, 2026-05-03) vs 2026-06 content
- ☐ §6.6 cross-references — 8 "§6.5" refs now ambiguous after the renumber

**Visualizations.** Hub `outputs/_hub/index.html` (work_queue surfaced on top);
`outputs/cross_site/signature/figures_v2/index.html`; 5 per-site dashboards under
`outputs/_distribution/html_dashboards/`; 48 figures in `docs/technical_report/figures/`.

**Blocked / external (do NOT start autonomously).** CFD-τ (separate repo) · ray-caster
vs Radiance/SOLWEIG x-val (external tooling) · Mingze WeTransfer upload (user-driven) ·
git-history rewrite #39 (excluded, no force-push) · brisaverse manuscript (external repo).

## 2. Consolidation (close out the current cycle) — all local

**Phase C0 — branch hygiene. [USER-AUTHORIZED 2026-06-27: auto-merge + prune, no further OK needed.]**
Fast-forward `main` ← `track/paper-integration`; push. Prune the merged tracks local+remote:
`track/roughness`, `track/morpho-signature`, `origin/track/morphotope`, `origin/track/viz`.
Leave #39 (history rewrite) as the only open task, untouched — no force-push/history-rewrite.
Branch a fresh `track/*` off the new `main` for §3 work.

**Phase C1 — TR coherence close-out (serial spine).** Clear the 8 open `tr_audit.md`
☐ items. `technical_report.md` is one file + one PDF binary → this is a **serial spine,
PDF rebuilt ONCE at the end**. Per-item drafting can fan out to read-only drafters, but
only the integrator edits the .md. Embed the two figure sets, add the three §5.5 caveats,
the two reproducibility stages, and the hygiene fixes (heading level, version header,
§6.6 xrefs).

**Phase C2 — full re-audit (read-only gates, parallel).** After C1 lands:
`numerical-claims-auditor` over the whole TR, `report-sync-auditor` over the consolidation
diff, plus a re-check of the two rounded watch-item brackets in §10.9 ("25–39 pp",
"2.7–4.4×"). Reconcile §12.2 test count to the live 710.

**Phase C3 — visualization refresh.** Rebuild the hub (`build_project_hub.py`), regen the
`figures_v2` index, verify hub links resolve, refresh the work_queue "Latest" entry.

Phases C0→C1→C2 are a chain; C3 runs concurrent with C2.

## 3. Expansion menu (the new science — pick 1–2 next session)

All candidates are fully local, additive (suffixed columns, canonical λf/morphotype
bit-for-bit), and reversible — same engineering contract as the last two tracks.

| # | Track | What it adds | Cost | Gate |
|---|-------|--------------|------|------|
| E1 | **Roughness method-ensemble envelope** | z0/zd across Lettau/Macdonald/Kanda/Millward-Hopkins as a *spread*, not one value; honest uncertainty band on the inlet BC. The morphometric inputs already exist per cell. | M | none (local) |
| E2 | **Composite ventilation-tendency index** | Fuse the three §5.6 scalars (lateral-connectivity, regime×depth, wind-exposure) into one ranked geometric tendency layer + cross-site map. | S–M | none |
| E3 | **Full MAUP resolution curve** | Now the 20 m confound is fixed, run 5 m/15 m/30 m and report the regime-share scaling *curve* (not just a 10/20 A/B) — turns a caveat into a result. | M | none |
| E4 | **Seasonal solar envelope** | Extend winter-only WHO-2h to equinox/summer; seasonal sun-access deficit per morphotype. | M–L | none |
| E5 | **Productionized cross-site risk map** | Apply the continuous fabric-vector predictor across all 5 campaign + 3 calibration sites; blind external validation expansion. | M | none |
| E6 | **Morphotope tissue-transition analysis** | Spatial adjacency / transitions between the k=3 tissues; Moran's I already wired in fig04. | M | none |

**USER-SELECTED 2026-06-27 (locked for next session): E2 + E1 + E5.** Suggested order —
**E2 first** (shortest, fuses scalars already on disk), then **E1** (roughness ensemble,
inputs per-cell ready), then **E5** (cross-site risk map, the heaviest; depends on the
predictor vector that's already canonical). E3/E4/E6 deferred. All three are local,
additive, canonical-safe.

## 4. Agent teams + dynamic workflow (reusable shape for §3)

**Hard constraint (from the repo-audit run).** `outputs/` and `data/` are gitignored
and live ONLY in the main working tree → **worktree isolation is unusable** for any agent
that reads/writes pipeline outputs. Expansion agents run serial-on-main against shared
outputs; only pure-source/test edits could use a worktree.

**Standing project subagents (reuse, don't reinvent):**
- Read-only gates: `numerical-claims-auditor`, `report-sync-auditor`, `sampling-auditor`,
  `data-contract-checker`.
- Workflow accelerators: `site-onboarder`, `wind-ingestion`, `cfd-results-ingestor`.
- Generic: `Explore` (recon), `Plan` (design), `general-purpose` (build).

**The dynamic workflow (one `Workflow` call per phase, user stays in the loop between):**

1. **Understand** — fan out `Explore` over the chosen expansion's existing substrate
   (which scripts/outputs/columns already exist) → structured map. No edits.
2. **Design** — `Plan` panel proposes 2–3 approaches; a judge stage scores on
   deliverable-value × feasibility × contract-safety; synthesize the winner.
3. **Build** — serial spine for any `technical_report.md` prose (single PDF rebuild at
   end) + parallel side-streams for file-disjoint `outputs/` work. Each new metric: draft
   → adversarial numerical verify (refute-by-default) before it's accepted.
4. **Verify** — read-only gates as a barrier: numerical-claims-auditor + report-sync-auditor
   (+ sampling/data-contract where relevant). Confirmed findings only.
5. **Consolidate** — hub refresh + work_queue "Latest" + memory update; commit per logical
   step, push, FF-merge to main on user OK.

Pattern notes: pipeline() by default (a metric verifies as soon as its draft lands);
barrier only where a stage needs all prior results (e.g. dedup before the audit, or the
single PDF rebuild). Adversarial verify on every numerical claim — the §6.5 Blocken miss
and the MAUP confound are the bug class this catches.

## 5. Compaction-safe summary

Everything shipped is committed + pushed on `track/paper-integration` (5 ahead of main,
clean FF). Next session: run §2 (consolidate + merge to main + close tr_audit) then §3
(pick an expansion from the menu) using the §4 workflow. Canonical λf/morphotype stay
bit-for-bit throughout.
