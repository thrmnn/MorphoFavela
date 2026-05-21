# Engineering Review — 1-day reading path

You're reviewing this codebase. You have a working day. Here's the
order to read things in, with rough time budgets.

## Before you start

```bash
git clone https://github.com/thrmnn/MorphoFavela.git && cd MorphoFavela
conda create -n morphofavela python=3.11 && conda activate morphofavela
pip install -e ".[dev]"
pytest tests/ -m "not integration" -q --tb=short
```

If `pytest` reports 508 tests passing in ≤ 2 min, the codebase is
loaded correctly and you can stop verifying mechanics. If the smoke
test fails, see [`local_setup.md`](local_setup.md).

## Reading path

| Time | What to read |
|---|---|
| **10 min** | [`README.md`](../../README.md) — the operational guide; what this is, what this is *not*, how to run the pipeline |
| **20 min** | Technical report metadata block + Executive Summary + §0 Glossary ([`docs/technical_report/technical_report.md`](../technical_report/technical_report.md)). Skim the glossary so you know what SVF, λp, λf, σH, Tregenza-145, Blocken margin, neutral log-law mean before you encounter them in context. |
| **30 min** | TR §1–§3 (sites, data sources, preparation). Confirm the sites and data inputs match what you'd expect for a Rio favela morphometric study. |
| **45 min** | TR §4 (morphometric grid). The 20+ indicators per cell are the building blocks of everything downstream. §4.2 names every indicator with units. §4.3–§4.6 show distributions, spatial maps, correlations, and resolution sensitivity. |
| **30 min** | TR §5 (cross-site morphology). The §5.2 typology summary table is the most-cited cross-site comparison. |
| **90 min** | TR §6 (CFD patch sampling) — **the central methodological claim**. Read §6.1 (design), §6.2 (eligibility filter), §6.3 (selection algorithm), §6.4 (allocation results). Critical follow-up: §6.5 Blocken compliance — every patch satisfies `5 × H_max ≤ 250 m`, but margins range 114–215 m with 11/119 under 150 m. |
| **45 min** | TR §7 (CFD integration pipeline) **alongside** [`src/cfd_integration/README.md`](../../src/cfd_integration/README.md). The CFD repo (a separate project at `~/Airflow`) consumes the contract specified there; engineering reviewers should evaluate whether the contract is tight enough that an independent OpenFOAM team could produce conforming output. |
| **30 min** | TR §9 (validation summary), §10 (known limitations), §11 (next steps). §10 is the honest list of what's NOT trustworthy yet. |
| **30 min** | TR §12 (reproducibility) and §13 (failure modes & observability). §12.4 maps every figure to a producer command. §13 explains how the four read-only validators surface drift. |
| **60 min** | **Walk one site end-to-end against §12.3.** Pick `vidigal` — it's the smallest valid grid (n = 1,503 in the §10.3 cross-validation), so commands run fast. Verify a number from the report against the source CSV/JSON the prose cites. The §12.5 per-table regeneration index tells you which file backs which table. |
| **Closing** | Read TR Appendix D (engineering review checklist) — what kind of feedback is most valuable, what's not useful at this stage, how to file. |

## What you're being asked to evaluate

In rough priority order:

1. **Methodology.** Is the SVF–UMEP cross-validation in §10.3 a strong-enough benchmark? Should the σH↔H_mean correlation in §4.5 be reported pooled rather than per-site? Is the 12-stratum SVF × slope × λp grid the right axis set for *health-relevant* wind regimes, or is wind direction a missing axis?
2. **CFD contract.** Is `src/cfd_integration/README.md` tight enough that an independent OpenFOAM team could produce conforming output without back-and-forth?
3. **Sampling design.** Is 119 patches the right total for 5 sites? Are SVF-priority weights (×2.0 / ×1.0 / ×0.8) defensible?
4. **Reproducibility.** Does §12 give you enough to actually reproduce a figure on a fresh clone?
5. **Numerical integrity.** A `numerical-claims-auditor` sweep ran on 2026-05-03 and caught the §6.5-class propagation miss in §9. Other §6.5-class bugs may exist; spot-checks against `outputs/` are welcome.

## Things the report deliberately defers

These are out of scope for now and don't need feedback:

- Real CFD results — VDG-P07 is in flight at MIT ORCD; the result-side pipeline is synthetic-validated only.
- The Nature Cities manuscript draft — that lives separately and cites this report.
- Solar irradiance cross-validation — explicitly deferred per ROADMAP.
- 5 m / 2 m grid resolution sensitivity — explicitly deferred per §10.4.
- Cidade de Deus re-onboarding — gated on upstream building-data fix.

## How to file feedback

| Severity | Channel |
|---|---|
| Methodological / numerical errors | GitHub issue with section reference + a one-line repro |
| Suggested next experiments | GitHub issue tagged `discussion` |
| Pipeline-contract questions | Email the author (`thermann.ai@gmail.com`) — answers usually need design context |
| Editorial / typographical | Direct message; bulk fixes are batched |

## If you have less than a day

Compressed 2-hour path: README (10 min) → TR Exec Summary + §0 Glossary
(20 min) → §6 Sampling (45 min) → §10 Limitations + §13 Failure modes
(30 min) → Appendix D (15 min). This catches the central methodological
claim plus the honest assessment of what's not yet trustworthy.

If you have less than two hours: read TR Exec Summary (the
"engineering reviewer in three sentences" block) + §0 Glossary
+ Appendix D, then file questions.
