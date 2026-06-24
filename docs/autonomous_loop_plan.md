# Plan — a multi-hour autonomous multi-agent loop

*What it would take to run MorphoFavela autonomously for hours, with several agents
in parallel and heavy computation in the background — and the **top blockers you can
remove** to enable it, ranked by payoff. Drafted 2026-06-19.*

## The goal

A long-running loop that: pulls the next item from `docs/work_queue.md`, fans out
parallel agents (build / analyse / review / audit), launches heavy computation that
benefits from wall-clock (CFD post-processing, large-grid regen, bootstrap at scale,
terrain-following recompute), commits green results on track branches, refreshes the
hub, and keeps going — stopping only on the gates you reserve.

## Top blockers you can remove (ranked by payoff)

### Tier 1 — unlocks the most science + lets every script run unattended

1. **One consolidated conda env.** Today work is split across three envs — `IVF`
   (geopandas, no statsmodels/weasyprint), `miniforge3` base (statsmodels +
   weasyprint + geopandas), `brisa`. Scripts need different interpreters, so an
   unattended loop keeps guessing the right Python and some steps silently can't run
   (the variance decomposition needed `miniforge3`; the PDF needs weasyprint; the
   queued embedding/regionalization need `umap-learn`/`spopt`). **Action:** create one
   `morphofavela` env from a pinned `environment.yml` (I'll generate it) with:
   geopandas, shapely, libpysal, esda, **spopt**, scikit-learn, scipy, **statsmodels**,
   **umap-learn**, **seaborn**, **mapclassify**, **contextily**, matplotlib, pyarrow,
   weasyprint + pandoc, trimesh/pyvista. *Payoff:* every script + every queued analysis
   runs from one interpreter, unattended. **Biggest single throughput unblock.**

2. **Real OpenFOAM CFD results (or a path to run them).** The largest *scientific*
   blocker. Gated on it: R-C roughness validation (resolve the rougher-vs-smoother
   question), ventilation/ACH, the flagship CFD axis, and the ventilation half of the
   typology→failure predictor. Synthetic placeholders cannot be honestly used.
   **Action:** launch the CFD campaign (Airflow/HPC) and drop returns at the contract
   path `data/{site}/cfd_results/{patch}/{dir}/`; the ingestion + R-C extraction are
   already specified (`src/cfd_integration/README.md`). *Payoff:* unblocks 4 gated
   workstreams at once and turns the roughness "envelope" into a validated number.

3. **Pre-authorized HPC (MIT ORCD) job submission.** The loop's "leverage the long
   time" depends on somewhere to put hours-long compute. **Action:** confirm I may
   submit SLURM jobs (credentials + an allowed submission pattern) for: CFD
   post-processing, large-grid / 20 m SVF regen, terrain-following morphometry
   recompute, and large bootstrap/stability runs. *Payoff:* heavy compute runs in the
   background while agents do analysis — the core of "leverage the long time."

### Tier 2 — unlocks autonomy throughput (fewer stalls)

4. **Pre-authorize merges of green track branches to `main`** (tests pass + ruff
   clean + report-sync ok). Today every merge waits for you. **Action:** "auto-merge a
   track branch when its suite is green and the hub builds." Reserve force-push,
   history rewrite, branch deletes, and external sends as still-gated. *Payoff:* the
   loop consolidates instead of piling unmerged branches.

5. **Pre-decide a batch of taste forks** so the loop doesn't stall on questions:
   (a) **finalize the names** — morphotypes T0–T5 + morphotopes M0–M4 (current
   proposals, or your edits); (b) **palette** — keep Okabe-Ito categorical vs switch to
   an ordered density ramp; (c) **figure keep/refine** sign-off on the gallery.
   *Payoff:* removes the most frequent loop-stall (AskUserQuestion).

6. **Pre-authorize specific heavy regenerations** to run overnight: the **20 m
   re-baseline** (#24) and the **terrain-following morphometry** recompute (fixes the
   roughness datum confound, council option C). Both are gated heavy compute. *Payoff:*
   the loop uses idle wall-clock instead of waiting for a green light.

7. **A persistence mechanism for hours-long unattended runs.** Background jobs and the
   hub server die when the session drops. **Action:** pick one — (i) run a detached
   driver on the laptop/VPS that re-invokes the loop; (ii) authorize a `CronCreate`
   routine (cloud agents on a schedule); or (iii) a single long `Workflow` run. Set a
   **token budget** (e.g. "+2M tokens") so the loop sizes its fan-out. *Payoff:* work
   survives disconnects and genuinely runs for hours.

### Tier 3 — data + infra polish

8. **Street-network / beco data quality** — Vidigal's roads lack the Escadaria/Ladeira
   category, so stair circulation is under-sampled, which limits the planned
   street-network/beco configuration metric. **Action:** a corrected roads layer (or
   accept the caveat).

9. **Always-on hub host.** The review hub dies with the session. **Action:** authorize
   a persistent host (a small service on the VPS, or Tailscale Funnel / the brisaverse
   Vercel-mirror pattern) so you can review anytime without me restarting the server.

10. **Stale `brisa-0.1.0` editable install** shadows top-level `src` (handled today by
    `sys.path.insert`), a latent foot-gun. **Action:** `pip uninstall brisa` (or
    reinstall it at the renamed path) in the consolidated env.

## What the loop does once unblocked (architecture)

- **Driver:** read `work_queue.md` → take the top unblocked item → spawn the right
  pattern.
- **Parallel patterns** (one Workflow per phase, agents fan out):
  *understand* (parallel readers) · *build* (one agent per sub-task, worktree-isolated
  when they mutate files) · *adversarial review/audit* (the TR-audit + numerical-claims
  + report-sync lenses, run every few commits) · *synthesize*.
- **Heavy compute:** submit to HPC / run detached; the loop polls and ingests on
  completion rather than blocking.
- **Self-auditing:** after each batch, run the report-sync + numerical-claims auditors
  so the TR/figures never drift (the standing weakness this session surfaced).
- **Cadence + gates:** commit green on track branches; auto-merge when authorized;
  stop+report only for the reserved gates; keep the hub + queue current so you can
  drop in anytime.

## The single highest-leverage action

**Create the one consolidated env (Blocker 1)** — it alone makes every script and
every queued analysis runnable unattended from a single interpreter, which is the
precondition for *any* multi-hour loop. Then Blocker 2 (real CFD) is what turns the
roughness/ventilation work from "honest placeholder" into "validated result."

## Env status (2026-06-24) — consolidated env DEFERRED; work on IVF

The consolidated `morphofavela` env was attempted two ways and both failed:
1. Fresh `conda env create -f environment.yml` — **unsolvable** (`gdal=3.12.4` needs
   Python >=3.13, but the file pins 3.11; aspirational pins never fresh-tested).
2. Clone IVF + `mamba install` the extras — the extras **upgraded numpy to 2.4.6 and
   broke scipy's ABI** (`scipy._cyutility ... slice_memviewslice`); unrecoverable by
   patching. The broken clone was removed.

**Working env remains `IVF`** (geopandas/sklearn/esda/libpysal/matplotlib/trimesh/
manifold3d/triangle/seaborn — enough for the brisaverse figures, prints, and most
analysis). Gaps: `statsmodels` (variance ANOVA — already done), `spopt`/`umap-learn`
(Loop Batch 2 — still env-gated).

**Correct recipe for a clean retry (not yet run):** clone IVF, then install extras with
**numpy pinned to IVF's version** so nothing upgrades the ABI:
`mamba install -p <env> -c conda-forge statsmodels weasyprint spopt umap-learn "numpy=<IVF numpy>"`
then `pip install manifold3d triangle`. OR a fresh conda-forge solve with `gdal>=3.11`
unpinned. Verify with `pytest tests/` before removing IVF.
