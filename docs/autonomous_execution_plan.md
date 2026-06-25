# Autonomous execution plan — removing the user as a blocker (2026-06-25)

Operating charter for long unattended runs. Goal: maximise the work done without
you in the loop, surfacing to you ONLY genuine forks, external actions, or hard
blocks. Supersedes the loop notes in `docs/autonomous_loop_plan.md` for sequencing;
the λf/figure specifics live in `docs/lambda_f_fix_and_figure_regen_plan.md` and the
round-2/round-3 handoffs.

## 1. Operating principle

Default to ACTION. Each unit of work: implement → test (`pytest` + `ruff`) →
view any figure → commit → push → refresh hub. Commit per logical step so nothing
is lost across sessions and every state is `pytest`-green. Surface to you only at a
real decision boundary (see §5). Run multi-part work as **Workflow fan-outs** or
parallel **subagents**; verify findings adversarially before trusting them.

## 2. Standing pre-authorizations (decide myself, no ask)

- **Merge/commit/push** green branches to `main` (tests + ruff pass).
- **Scientific re-baselines** that are reversible and non-destructive — preserve the
  prior canonical (e.g. `lambda_f_mean_summed`), keep a stability number (ARI), and
  document the call. (The dissolved-λf re-baseline is the template.)
- **Naming / palette / figure-design** calls — pick, document, flag for post-hoc
  review; do not block on them.
- **Local compute** of any size (with `nice -n 10`, never grabbing all cores).
- **Docs / memory / hub** updates; TR edits **with** the PDF rebuilt in the same commit.
- **Pipeline regen** of gitignored outputs.

## 3. Still gated — surface, do not self-authorize

- Force-push, history rewrite, branch deletion, tag moves.
- **External sends**: Mingze WeTransfer upload, emails, anything leaving the machine.
- Destructive deletes of data I did not create and cannot regenerate.
- **CFD execution** (separate repo, out of scope) and **HPC** submission.
- Installing system tools that need your credentials (Radiance/SOLWEIG, auth flows).

## 4. Parallel tracks (the backlog, runnable concurrently)

Each track is a Workflow or a chain of subagents; they touch disjoint files so they
parallelise. Priority order top-to-bottom.

- **A — λf re-baseline cascade (in flight).** Done: features regen, k=6 dissolved
  re-fit, type re-naming, gallery. Remaining: morphotope re-fit (`build_morphotope`),
  typology→failure predictor regen, TR §5.5 + §6.6 rewrite + PDF, brisa P4/E2 figures
  (`type_site_fingerprint`, recurrence), numerical-claims audit of the TR.
- **B — Figure regen rounds 2/3.** Firewall-in-pixels (hatch provisional, anti-misuse
  banner, nominal 2×2 taxonomy), the **text-overflow HARD GATE** assertion pass,
  fig01/fig03/fig04/fig05 reworks. Subagent per figure; I review each PNG.
- **C — Roughness z0/zd on dissolved λf** + TR §6.6 numbers + roughness figures.
- **D — Regime classification integration** into fig03 panel C + the manuscript
  "uniformly skimming → CFD-necessary" sentence.
- **E — Lateral-connectivity scalar** (distance-to-open-edge) — the one independent
  pre-CFD ventilation-tendency signal the council flagged; qualitative, τ-superseded.
- **F — Data-quality sweeps** — DTM spikes (RDP-P03 etc.), phantom-tower residuals,
  numerical-claim drift; each a read-only auditor agent → fix → commit.
- **G — TR sync discipline** — after any A/C/D change, run `report-sync-auditor` +
  `numerical-claims-auditor`, rebuild the PDF.

Blocked (need you / external): ray-caster × Radiance/SOLWEIG; CFD τ; Mingze upload.

## 5. Decision routing — when I stop for you

Stop and ask ONLY when:
1. A **scientific fork** changes published conclusions and is not reversible by
   preserving the prior (the dissolved-λf morphotype re-baseline was such a fork —
   I asked; now decided).
2. An **external/irreversible action** is required (send, delete, force-push).
3. A **hard block** (missing tool/credential/data) halts a track.
Everything else: make the reasonable call, document it, continue. Batch the
"flag-for-review" items into the work-queue rather than interrupting.

## 6. Execution mechanism

- **Workflow tool** for fan-out with deterministic control flow (per-figure, per-site,
  per-finding) and structured returns; **subagents** for independent multi-step units.
- **Verify before trusting**: adversarial/independent re-derivation for any new finding
  (the λf over-count and the morphotype ARI were both caught by re-checking).
- **Self-pacing**: for unattended stretches use `/loop` (or `ScheduleWakeup`) with a
  fallback heartbeat; otherwise run tracks back-to-back, committing each.
- **Heavy jobs** (migrations, re-fits, CFD-scale grids) run in the background after the
  predecessor is committed; a notification resumes me.

## 7. Verification gate (every commit)

`pytest tests/` green · `ruff check` clean · figure viewed at true size · TR edits carry
the rebuilt PDF · new behaviour ships its test. Hub refreshed so results are findable.

## 8. Kickoff order after the current task

A (finish the cascade) → C + D (roughness + regime, share the λf work) → B (figure regen,
largest, parallelisable by subagent) → E → F. G runs continuously. I'll post a one-line
progress note per track completion and a consolidated checkpoint per session, and only
interrupt per §5.
