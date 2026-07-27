# Adversarial audit — ventilation-deficit ρ = +1.00 vs TB (Track D / T4)

**Date:** 2026-07-28 · **Auditor role:** G6 adversarial verification skeptic ·
**Status:** GATE DECISION — see verdict.

**Claim under audit.** Across the 5 TB screen sites, a morphometric
ventilation-deficit exposure (built-cell share with `lambda_f_mean > 0.35`, the P1
skimming/4-state threshold) rank-matches 9-yr TB incidence perfectly:
Spearman **ρ = +1.00**, exact two-tailed permutation **p = 0.017**. Sun-alone
ρ = +0.80 (p = 0.133); compound (sun ∩ ventilation) ρ = +0.90.
Source: `scripts/health/compound_exposure.py`,
`outputs/comparative/health/compound_exposure.json`,
`scripts/health/tb_sun_deficit_screen.py`.

All numbers below were recomputed independently from the JSON per-site values.

---

## 1. Saturation / fragility — the rank order hangs on a 0.78 pp gap

The five ventilation-deficit shares (λf > 0.35), sorted, against TB rank:

| site | vent-deficit % | gap to next | TB /100k | TB rank |
|---|---|---|---|---|
| Complexo do Alemão | 81.92 | — | 93.7 | 1 |
| Maré | 86.35 | **+4.43** | 161.9 | 2 |
| Vidigal | 87.13 | **+0.78** | 257.3 | 3 |
| Rocinha | 91.51 | +4.39 | 400.5 | 4 |
| Jacarezinho | 96.73 | +5.21 | 427.0 | 5 |

The claim that the share is "saturated at 82–97%" is **partly** overstated: the
range is **14.8 pp**, a real ordering, not a flat band. BUT the perfect rank match
is decided at one crux — the **Maré → Vidigal step of 0.78 pp**. That pair is the
only adjacent gap under ~4 pp, and it separates TB rank 2 from rank 3.

**Rank-flip test.** Swap the ventilation values of Maré and Vidigal (they sit
0.78 pp apart — well inside the method noise of a frontal-area share computed over
2.5k–24k grid cells with differing DTM/DSM provenance):

- ρ(vent, TB): **+1.00 → +0.90**
- exact perm p: **0.017 → 0.083**

A sub-1-percentage-point perturbation at a single pair destroys both the "perfect"
match and the "significant" floor. **The +1.00 is a knife-edge, not a plateau.**

## 2. Leave-one-out — VACUOUS, not reassuring

Dropping each site and recomputing:

| dropped | n | ρ(vent, TB) |
|---|---|---|
| Rocinha | 4 | +1.00 |
| Vidigal | 4 | +1.00 |
| Maré | 4 | +1.00 |
| Alemão | 4 | +1.00 |
| Jacarezinho | 4 | +1.00 |

This looks maximally robust and **is worthless as evidence.** When the full n=5 set
is *perfectly monotone*, every 4-point subset is *necessarily* monotone too, so
LOO is algebraically pinned at +1.00 and **cannot** detect fragility. LOO robustness
here is a mathematical guarantee of the perfect match, not an independent test of it.
The informative sensitivity is the rank-flip in §1, which LOO structurally hides.

## 3. Independence / collinearity

Pairwise Spearman among the exposures and covariates:

| pair | ρ |
|---|---|
| ventilation × sun-deficit | **+0.80** |
| ventilation × density (pop/km²) | +0.60 |
| ventilation × crowding (persons/hh) | −0.40 |
| ventilation × TB | +1.00 |
| sun × TB | +0.80 |
| density × TB | +0.60 |

Ventilation-deficit is **strongly collinear with the sun-deficit (+0.80)** and
moderately with generic density (+0.60). It is not a degenerate copy of either, but
it carries no clearly *independent* structure: it ranks the sites in the same order
as the sun-deficit would, only with the two middle sites (Maré/Vidigal) nudged into
exact agreement with TB by 0.78 pp.

## 4. Partial correlation — ALSO VACUOUS by construction

Running the same first-order partial the T2 confound test used,
ρ(vent, TB | z):

| control z | partial ρ | ρ(vent,z) | ρ(TB,z) |
|---|---|---|---|
| density | **+1.00** | +0.60 | +0.60 |
| crowding | **+1.00** | −0.40 | −0.40 |
| sun | **+1.00** | +0.80 | +0.80 |

The partial stays at +1.00 for **every** covariate — and this is **not** evidence
that ventilation survives adjustment. Because ventilation and TB have *identical
rank vectors*, any covariate correlates *equally* with both (note ρ(vent,z) =
ρ(TB,z) in every row). Plug rxz = ryz into the first-order partial formula
`(rxy − rxz·ryz)/√((1−rxz²)(1−ryz²))` with rxy = 1 and it collapses to exactly 1.0
regardless of the covariate. **The partial is mathematically incapable of dropping
below 1.0 here** — it cannot refute confounding, so it provides zero reassurance.
(Contrast the sun-deficit, whose partial ρ *does* move — +0.69 given density, +0.76
given crowding — because sun and TB do *not* share identical ranks. That partial is
informative; this one is not.)

## 5. Dengue specificity — not contradicted, but tautological

ρ(ventilation-deficit, dengue incidence) = **−0.40**. Ventilation does not track the
mosquito-borne placebo, which on its face *supports* specificity (unlike a generic
deprivation axis, it doesn't pick up dengue). But note ρ(TB, dengue) = **−0.40** as
well — identical, because ventilation and TB share the same ranks. So this is not an
independent check of the *ventilation metric*; it merely restates TB's own
(negative) relationship with dengue. It does not hurt the claim, but it adds no
information beyond the sun-deficit's already-reported dengue placebo.

## 6. p-value meaning

At n=5 the smallest attainable exact two-tailed permutation p is 2/120 ≈ **0.017**
(the observed statistic and its mirror image out of 5! = 120 relabellings).
**p = 0.017 is the arithmetic floor** — it means "the ranks matched perfectly,"
nothing more. It is not a measure of strength; a perfect match at n=5 *always*
prints 0.017. Reporting it as "p = 0.017" invites reading it as strong evidence
when it is definitionally the weakest possible perfect match.

---

## Verdict: (b) direction-suggestive but fragile and collinear — with the strong caveat that its confound-controls are algebraically vacuous

Not (a): a "reasonably robust" signal would survive a sub-1 pp perturbation and would
have *informative* independence checks. This has neither. Not purely (c) either: the
14.8 pp spread is a genuine ordering and the direction is consistent with the
independently-corroborating sun-deficit (+0.80), so it is not a saturation artifact
in the "all values equal" sense.

But the "**ρ = +1.00, p = 0.017**" headline is **inadmissible** as stated, for three
compounding reasons:

1. **Fragility.** The perfect match rests on a 0.78 pp gap. A perturbation smaller
   than the metric's own cross-site method noise takes it to +0.90 / p = 0.083.
2. **Vacuous verification.** Its two robustness checks — LOO and the partial
   correlation — are *both mathematically forced* (LOO by monotonicity, the partial
   by the identical rank vectors) and therefore cannot refute confounding. The
   ventilation number is *less* verified than the sun-deficit, whose partial
   actually moves. A ρ = 1.00 whose confound-controls are incapable of failing is
   weaker evidence than a ρ = 0.80 whose controls could have failed and didn't.
3. **Collinearity + floor p.** Collinear with sun (+0.80) and density (+0.60);
   p = 0.017 is the n=5 floor, i.e. "a perfect match," not a strength.

**May it reach `health.html`? Not as a standalone number, and never as "+1.00 /
p = 0.017."** The perfect rank match is exactly the "too good to be true" artifact the
G6 gate exists to stop. It may be surfaced *only* as a subordinate, hedged line
inside the existing Grade-C ecological probe — never as a headline, never with the
bare +1.00, and never before n is lifted (T7 / new-site onboarding). The honest
finding is the *sun-deficit* probe (+0.80, informative partial, powered dengue
placebo); ventilation is at most a corroborating direction.

### Exact permitted wording

For `docs/planetary_health_plan.md` (keep the existing GATED tone at lines 380–388;
this audit *resolves* the G6 gate to CONDITIONAL, not GO):

> A geometric ventilation-deficit share (built cells with λf > 0.35) ranks the five
> sites in the same order as 9-yr TB incidence. We do **not** report this as ρ = +1.00
> / p = 0.017: at n = 5 that p is the arithmetic floor (2/120) and the match is
> fragile — a 0.78 pp shift in the Maré/Vidigal pair (below the metric's cross-site
> noise) drops it to ρ = 0.90. Its leave-one-out and partial-correlation checks are
> algebraically pinned at +1.00 (a consequence of the perfect monotonicity and of
> ventilation sharing TB's exact rank vector) and so cannot test confounding.
> Ventilation-deficit is collinear with the winter sun-deficit (ρ = +0.80) and with
> density (ρ = +0.60). Read: **direction-consistent corroboration of the sun-deficit
> probe, not an independent ventilation signal.** Gate status: CONDITIONAL — stays off
> `health.html` as a standalone number; may appear only as a hedged sub-line of the
> Grade-C probe once n ≥ 8 (T7).

For `health.html` (if surfaced at all, inside `#health-probe`, below the sun-deficit
line, no number in the headline):

> Ventilation-deficit (built-form frontal-area share) orders the five favelas the same
> way TB incidence does — consistent with the sun-deficit direction, but at n = 5 this
> is a fragile, collinear corroboration, not independent evidence of a ventilation
> mechanism.

### What would upgrade it

Only more sites (T7 / onboarding → n ≥ 8). At n ≥ 8 the rank-flip fragility, the LOO
degeneracy, and the partial-correlation degeneracy all dissolve (the partial can move
because ventilation and TB will no longer share an identical rank vector), and a
non-floor permutation p becomes attainable. Nothing computable at n = 5 can rescue
the +1.00.
