# Technical-report audit — coherence & bulletproofing punch list

*From an expert coherence audit (2026-06-19), cross-checked against the decision
logs and the actual `outputs/cross_site/signature/` artifacts. Worked top-down; each
item marked ☐ open / ☑ done.*

## Critical (credibility / structural)

- ☑ **§6.5 collision** — two §6.5 sections (`### 6.5 Rectangular domain compliance` +
  `## 6.5 Aerodynamic Roughness`). Renumber roughness → **§6.6**.
- ☑ **§6.6 roughness omits its own headline finding** — add the RC-1 invalidity
  (per-cell z0/zd physically invalid in 53–75% of cells: zd > H_max, z0 → 0; model
  extrapolation, not measurement; envelope is the result, absolute z0 CFD-gated) and
  the RC-3 180°-symmetry constraint. *The single most important fix.*
- ☑ **Recurrence reconciliation** — ground truth (5-site, `recurrence_flags.csv`):
  T0,T1,T4,T5 recur; **T2,T3 are both conditional**. TR "4 of 6 morphotypes" is
  CORRECT; decision-log D13 ("0,1,2,4,5 recur") is the stale 8-site version → note it.
- ☑ **Morphotype vs morphotope disambiguation** — distinct tissue vocabulary (M0–M4
  "… Tissue", no Fringe/Consolidated/Core reuse); state the T-cell / M-tissue rule.

## High

- ☐ **§5.5 circularity caveat** — SVF ≈ f(λp, H/W) (excluded from the fabric set for
  exactly this reason, D2), so the SVF gradient is partly mechanical; the **winter-sun
  / WHO-failure** outcomes are the cleaner held-out signal.
- ☐ **§5.5 support caveat** — experience medians are on supported cells only (~35% of
  cells carry an observer; per-type support 0.23–0.59; open types less observed).
- ☐ **§5.5 conditional-type caveat** — the cell typology is not fully universal (T2/T3
  flatland-conditional) before pivoting to morphotopes.
- ☑ **Predictor work marked forthcoming** — WHO-2h prediction / LOSO / parsimony is
  referenced as "basis" but is a separate sub-study → flag "(forthcoming)".
- ☐ **Embed the validation figures** — recurrence + experience-dotplots into §5.5;
  roughness_validity into §6.6 (currently §6.6 shows zero figures).

## Medium

- ☑ **Exec Summary** silent on the signature and roughness — add one bullet each (with
  caveats).
- ☑ **§10 Known Limitations** has nothing on roughness validity — add a subsection.
- ☐ **§12 Reproducibility** — add Stage 7 (signature/morphotope/configuration) and
  Stage 8 (roughness) pipelines.
- ☐ **Heading hierarchy** — §5.5 is `##` while §5.1–5.4 are `###`; normalize.
- ☐ **Version/date** header stale (v1.0, 2026-05-03) vs the 2026-06 content → bump.
- ☐ **§6.6 cross-references** — eight "§6.5" refs now ambiguous; disambiguate after
  renumber.

## Cell-count note (resolved)

64,355 built cells is correct for the **5-site** signature (D17 scoping); the decision
log's 74,169 is the superseded 8-site pool. Overview + TR "~64k" are right.
