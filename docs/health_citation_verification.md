# Health-track external-citation re-verification (G13)

**Verified:** 2026-07-27 · against SciELO, PMC, and the publishing journals via web tools.
**Scope:** the five external sources cited by the planetary-health track in
`docs/planetary_health_plan.md` and `scripts/build_project_hub.py` (health section).

**Prime directive of this pass:** no author byline, DOI, or PMC ID is asserted here
unless it was read off the primary source. Anything that could not be confirmed is
marked UNVERIFIED rather than repaired from memory.

## Verdict table

| # | Cited as | Verdict | Confirmed identity |
|---|----------|---------|--------------------|
| 1 | Pró-Saúde — *Cad. Saúde Pública* 2022;38(1):e00287820, n=491 | **VERIFIED** | Bezerra FF, Normando P, Fonseca ACP, Zembrzuski V, Campos-Junior M, Cabello-Acero PH, Faerstein E. DOI 10.1590/0102-311X00287820 |
| 2 | Leão et al. — *Clinics* 2021;76:e2571, n=24,074 | **VERIFIED** | Leão LMCSM, Rodrigues BC, Dias PTP, Gehrke B, de Souza TDSP, Hirose CK, Freire MDC. DOI 10.6061/clinics/2021/e2571 |
| 3 | Vit-D ↔ TB OR 3.23 (1.91–5.45) — *Cureus* 2021;13(9):e17883 | **VERIFIED (with a framing caveat)** | Kafle S, Basnet AK, Karki et al., "Association of Vitamin D Deficiency With Pulmonary Tuberculosis: A Systematic Review and Meta-Analysis." DOI 10.7759/cureus.17883 |
| 4 | PMC4544397 — "setor-level" TB spatial analysis, Rocinha/Vidigal | **VERIFIED content / DISCREPANCY on spatial unit** | Pereira AGL, Medronho RA, Escosteguy CC, Ortiz Valencia LI, Magalhães MAFM, "Spatial distribution and socioeconomic context of tuberculosis in Rio de Janeiro." *Rev Saúde Pública* 2015. DOI 10.1590/S0034-8910.2015049005470 |
| 5 | PMC8009065 — "2018 Rio clinical-lab 25(OH)D cross-section", n≈24,074 | **VERIFIED / DUPLICATE of #2** | Resolves to the *same paper* as #2: Leão LMCSM et al., *Clinics* 2021;76:e2571 |

## Per-citation detail

### 1 — Pró-Saúde Study · VERIFIED
- **Journal / year / ID:** *Cadernos de Saúde Pública* 2022;38(1), article e00287820, DOI 10.1590/0102-311X00287820. Confirmed on SciELO.
- **Title:** "Genetic, sociodemographic and lifestyle factors associated with serum 25-hydroxyvitamin D concentrations in Brazilian adults: the Pró-Saúde Study."
- **Byline (previously flagged as "to verify"):** Bezerra FF (lead), Normando P, Fonseca ACP, Zembrzuski V, Campos-Junior M, Cabello-Acero PH, Faerstein E (senior; Pró-Saúde PI). **The open byline flag in the plan is now resolved.**
- **Sample size:** n=491 (251 women; 34–79 y), cross-section nested in the Pró-Saúde cohort. Matches.
- **Coefficients (all confirmed verbatim):**
  - Sun-exposure index: **β = 0.49 nmol/L per unit (95%CI 0.22; 0.75).** Matches.
  - Summer vs winter: **β = 20.14 nmol/L (95%CI 14.38; 25.90).** Matches.
  - Deficiency: **55%** of the population below 50 nmol/L. Matches.
- **Verdict:** every claimed value is real and correctly attributed.

### 2 — Leão et al., *Clinics* 2021 · VERIFIED
- **Journal / year / ID:** *Clinics* (São Paulo) 2021;76:e2571, DOI 10.6061/clinics/2021/e2571.
- **Byline:** Leão LMCSM, Rodrigues BC, Dias PTP, Gehrke B, de Souza TDSP, Hirose CK, Freire MDC. The plan's "Leão et al." attribution is correct.
- **Sample size:** 24,074 individuals (ages 1–95; 64.7% female), from ~80,000 consecutive Rio lab measurements Feb–May 2018, supplement users excluded. Matches.
- **Senior deficiency:** among those ≥60 y with the <30 ng/mL cutoff — women 53.2%, men 50.6%. Matches the cited "50.6–53.2% of seniors <30 ng/mL."
- **Verdict:** confirmed.

### 3 — Cureus vit-D↔TB OR 3.23 · VERIFIED (framing caveat)
- **Journal / ID / byline:** *Cureus* 2021;13(9):e17883, DOI 10.7759/cureus.17883. Kafle S, Basnet AK, Karki et al.
- **Effect size:** pooled **OR = 3.23 (95%CI 1.91–5.45, p<0.0001).** Matches exactly.
- **Caveat to surface before it prints:** this is a **systematic review / meta-analysis of pulmonary TB** (26 studies qualitative, 12 pooled), not a Rio primary study; and the OR is stated in the direction *"odds of vitamin-D deficiency were 3.23× higher in TB patients vs. healthy controls,"* i.e. deficiency conditioned on TB — not "TB risk given deficiency." The magnitude is what the plan uses as a mechanistic bridge, which is fine, but the wording should not imply a prospective TB-incidence odds ratio or a local study.

### 4 — PMC4544397 · VERIFIED content, DISCREPANCY on spatial unit
- **Identity:** Pereira AGL, Medronho RA, Escosteguy CC, Ortiz Valencia LI, Magalhães MAFM, "Spatial distribution and socioeconomic context of tuberculosis in Rio de Janeiro, Brazil," *Rev Saúde Pública* 2015, DOI 10.1590/S0034-8910.2015049005470. PMC ID resolves correctly.
- **Content confirmed:** explicitly names Rocinha and Vidigal (plus Cidade de Deus) as high-incidence areas; Rocinha crude rate **447.3/100k** vs city mean 95.9/100k. This backs the plan's "447/100k … (PMC4544397)" line.
- **DISCREPANCY:** the analysis is at the **bairro / neighbourhood level (158 neighbourhoods)**, *not* census **setor** level. The plan (line ~252 "setor-level TB spatial analysis") and the memory index overstate its granularity. Correct to "neighbourhood/bairro-level." This matters because the whole CEP-CONEP protocol below is justified precisely by the *absence* of a setor/address-level linked source — so mislabelling this paper as setor-level undercuts that justification.

### 5 — PMC8009065 · VERIFIED, but it is the SAME paper as #2
- **Identity:** PMC8009065 resolves to Leão LMCSM et al., *Clinics* 2021;76:e2571 — **the identical study cited in #2** (same n=24,074, same 2018 Rio lab cross-section).
- **DISCREPANCY:** the plan lists this as an apparently *separate* fifth source ("2018 Rio clinical-lab cross-section (n≈24,074; PMC8009065)" at line ~270) alongside "Leão et al. … e2571" (line ~327). A reader would count two independent corroborating datasets where there is **one**. De-duplicate: PMC8009065 = the PMC mirror of Clinics e2571.

## Corrections required before this prints publicly

1. **PMC4544397 is neighbourhood/bairro-level, not setor-level.** Fix `docs/planetary_health_plan.md` (the "setor-level TB spatial analysis" row, ~line 252, and the Rocinha row ~line 304) and the memory index. Do not describe it as census-tract/setor granularity.
2. **PMC8009065 and *Clinics* 2021;76:e2571 (Leão et al.) are one and the same study.** Merge the two plan entries (~line 270 and ~line 327) so it is not read as two independent sources.
3. **Cureus OR 3.23 wording:** label it as a pulmonary-TB meta-analysis and state the OR direction (vit-D deficiency given TB). Avoid phrasing that implies a local or prospective TB-incidence OR.
4. **Pró-Saúde byline is now resolved** — Bezerra FF et al. (senior author Faerstein E). The "one author byline to verify before publishing" note in the plan can be closed.

**No fabrications found.** All five DOIs/PMC IDs resolve to real articles whose headline numbers match what the track cites; the only defects are two granularity/duplication mislabels and one framing caveat — all correctable in text, none requiring a number to change.
