# CEP-CONEP research-ethics protocol — address/CEP-level SINAN-Tuberculose + SIM linkage in five Rio favelas

> **DRAFT — for the user to review and file; NOT submitted.**
> This is a working draft to be edited, completed with institutional details, and
> submitted through Plataforma Brasil by the responsible researcher. Placeholders
> in ⟨angle brackets⟩ must be filled before submission. Nothing here has been sent
> to any committee.

**Protocol working title:** Morphology of solar/ventilation exposure and the intra-urban
distribution of tuberculosis and cause-of-death in five Rio de Janeiro favelas: an
individual-level georeferenced record-linkage study.

**Modality:** Secondary use of identified administrative health microdata with
address/CEP-level geocoding. Observational, retrospective, ecological-to-individual
record linkage. No contact with human participants.

**Regulatory frame:** CNS Resolution 466/2012 and 510/2016 (research with human
subjects / social-science and secondary-data provisions), CONEP guidance on the
use of secondary databases and record linkage, and Lei Geral de Proteção de Dados
(LGPD, Lei 13.709/2018), in particular the treatment of sensitive health data for
scientific-research purposes.

---

## 1. Justification and background

The MorphoFavela project has produced high-resolution morphometric surfaces (sky-view
factor, building-facade solar access, ventilation/roughness envelopes) for five
consolidated Rio favelas — **Rocinha, Vidigal, Maré (Complexo da Maré), Complexo do
Alemão, and Jacarezinho**. These surfaces quantify, per building and per street segment,
how much direct sun and through-ventilation the built fabric admits.

A biologically plausible pathway links this morphology to tuberculosis (TB) burden:
dense, low-sky-view fabric reduces the UVB dose reaching skin and hence cutaneous
25-hydroxyvitamin-D [25(OH)D] synthesis, and vitamin-D status is associated with TB.
The supporting literature has been independently verified for this protocol
(see `docs/health_citation_verification.md`):

- In a Rio adult cohort (Pró-Saúde Study; Bezerra FF et al., *Cad. Saúde Pública*
  2022;38(1):e00287820, DOI 10.1590/0102-311X00287820, n=491), serum 25(OH)D rose
  **+0.49 nmol/L per unit of sun-exposure index (95%CI 0.22–0.75)** and was
  **+20.14 nmol/L higher in summer than winter (95%CI 14.38–25.90)**; 55% of the
  sample was deficient.
- In a 24,074-person Rio clinical-laboratory cross-section (Leão LMCSM et al.,
  *Clinics* 2021;76:e2571, DOI 10.6061/clinics/2021/e2571; PMC8009065 is the same
  study), 50.6–53.2% of adults ≥60 y had inadequate 25(OH)D.
- A meta-analysis (Kafle S et al., *Cureus* 2021;13(9):e17883, DOI 10.7759/cureus.17883)
  found the odds of vitamin-D deficiency were **3.23× higher in pulmonary-TB patients
  than controls (95%CI 1.91–5.45)**.
- At the intra-urban scale, a neighbourhood-level spatial analysis of Rio TB
  (Pereira AGL et al., *Rev Saúde Pública* 2015, DOI 10.1590/S0034-8910.2015049005470;
  PMC4544397) already flags **Rocinha and Vidigal** as high-incidence areas
  (Rocinha ≈447/100k vs. city mean ≈96/100k).

**The gap this study fills.** The existing evidence is ecological: it links
*neighbourhood* TB rates to *neighbourhood* conditions, and links vitamin-D to TB in
*separate* clinical samples. No source links an *individual* TB case (or death) to the
*morphological sun/ventilation exposure at that person's own address*. The change of
spatial support — from bairro/setor aggregates to the residential point — is exactly
what the current MorphoFavela screen cannot do, because SINAN and SIM public-use
extracts are de-identified and truncated above the setor. Reaching the residential
point requires identified microdata under ethics approval. Without this linkage the
morphology→TB hypothesis cannot be tested at the level at which the morphology is
measured, and any published intra-favela exposure gradient would remain an ecological
inference vulnerable to the ecological fallacy and to MAUP.

## 2. Objectives

**Primary.** Estimate the association between address-level morphometric solar/ventilation
exposure and individual TB notification, within the census setores of the five study
favelas, adjusting for individual and setor-level confounders (age, sex, crowding,
income proxy, HIV co-infection where recorded).

**Secondary.**
1. Repeat the association for TB-attributable and all-cause mortality using SIM.
2. Characterise residual spatial clustering after adjustment for morphology.
3. Quantify how much of the previously reported neighbourhood-level Rocinha/Vidigal
   excess is explained by within-favela morphological heterogeneity.

## 3. Data sources and fields requested

Two identified administrative databases, restricted to residents whose address geocodes
inside the census setores that intersect the five favela boundaries, for notification/
death years ⟨YYYY–YYYY⟩:

**A. SINAN-Tuberculose (notifiable-disease information system, TB module).**
Requested fields: notification and diagnosis dates; residential address / CEP;
municipality and setor of residence; age; sex; race/colour; clinical form; HIV status;
treatment outcome; case ID for de-duplication. Requested via ⟨SES-RJ / SMS-Rio
vigilância epidemiológica⟩ as the data custodian.

**B. SIM (mortality information system).**
Requested fields: date of death; residential address / CEP; setor; underlying and
contributing causes (ICD-10); age; sex; race/colour. Requested via ⟨SMS-Rio / DATASUS
identified-microdata request⟩.

**C. Project-side (already held, non-personal).** MorphoFavela morphometric rasters and
per-building/street exposure surfaces; IBGE census-setor geometry and aggregate
socioeconomic indicators. These contain no personal data.

Only the fields above are requested; no fields beyond those needed for geocoding,
linkage, confounder adjustment, and de-duplication are sought (data minimisation, LGPD
Art. 6).

## 4. Linkage and geocoding method

1. **Geocoding** of residential address/CEP to coordinates is performed **inside the
   custodian's secure environment** wherever the custodian can support it; where it must
   be done by the research team, it occurs only on the isolated secure workstation
   (§6) and the address strings are destroyed immediately after coordinates are
   assigned.
2. Each geocoded point is spatially joined to the MorphoFavela exposure surfaces to
   attach the address-level sun/ventilation exposure values and the containing setor.
3. **Immediately after the join, direct identifiers (name, full address, CEP, mother's
   name, any document numbers) are stripped** and replaced by a random study ID. The
   analytic dataset retains only: study ID, exposure values, setor, and the confounder/
   outcome fields. Coordinates are reduced to the attached exposure values and setor and
   are **not** retained at full precision in the analytic file.
4. Record de-duplication (same person, multiple notifications) uses a deterministic/
   probabilistic key computed *before* de-identification and discarded afterwards.
5. The identifier→study-ID crosswalk, if retained at all, is held only for the minimum
   period needed to resolve linkage queries (§9) and is stored separately from the
   analytic data under separate access control.

No attempt is made to re-contact, re-identify, or clinically follow any individual.

## 5. Confidentiality and analysis constraints

- Analysis and all outputs are at aggregate/model level. **No cell, map, or table that
  could identify an individual is produced or published.** Small-count suppression
  (minimum cell size ⟨e.g. 5⟩) is applied to every tabulation and to any map.
- Published maps show modelled exposure gradients and setor-level summaries only, never
  case point locations.
- The five favelas are named (they are already named as high-TB areas in the public
  literature), but no output narrows below the setor for a specific case.

## 6. Data-security plan

- Identified data are held **only** on ⟨institution⟩'s access-controlled, encrypted
  server / an offline encrypted workstation with full-disk encryption; no cloud storage,
  no personal laptops, no removable media except for the encrypted transfer authorised
  by the custodian.
- Access is limited to named team members (§ responsible researcher + ⟨named
  collaborators⟩), each under an individual confidentiality commitment.
- Transfer from the custodian uses the custodian's approved encrypted channel.
- Access is logged; the analytic (de-identified) dataset and the identified staging
  dataset are physically/logically separated with distinct credentials.
- The geocoding/linkage step runs on the isolated workstation with networking disabled.

## 7. Risk and benefit

**Risks.** The only material risk is a confidentiality breach on the identified staging
data (TB and cause-of-death are sensitive). It is mitigated by processing within the
custodian's environment where possible, prompt de-identification after linkage (§4),
small-cell suppression, restricted named access, encryption, and no publication below
setor. There is no physical, psychological, or clinical risk, as there is no participant
contact.

**Benefits.** No direct benefit to individuals. Societal benefit: the first
address-resolved test of whether built-form solar/ventilation deprivation contributes to
the well-documented favela TB excess, which could inform housing, upgrading, and
vitamin-D public-health responses in exactly the communities (Rocinha, Vidigal, Maré,
Alemão, Jacarezinho) already identified as high-burden.

The benefit/risk balance is favourable given the exclusive use of already-collected
administrative records and the layered de-identification.

## 8. Informed-consent waiver rationale (secondary data)

A waiver of individual informed consent is requested under CNS Res. 466/2012 (IV.8 —
waiver where consent is impracticable and risk is minimal) and Res. 510/2016, and under
LGPD Art. 11 §2 (treatment of sensitive data for scientific research), because:

1. The study uses **only pre-existing administrative records** collected for mandatory
   notification (SINAN) and civil registration (SIM); no new data are generated from
   participants.
2. Retrospective individual consent is **impracticable** — the cohort includes deceased
   persons (SIM) and years of past TB notifications across five large communities;
   attempting re-contact would itself be intrusive and infeasible.
3. **Risk is minimal** and confined to confidentiality, which is controlled by the
   security and de-identification plan (§4–§6).
4. The research **could not be carried out without the waiver**, and its results serve a
   public-health interest for the same populations.

Requesting consent would defeat the purpose and is not proportionate to the minimal
residual risk.

## 9. Data-use agreement and retention/disposal

- A formal **data-use / data-transfer agreement (Termo de compromisso / convênio)** will
  be signed with each custodian (⟨SES-RJ, SMS-Rio, DATASUS⟩) specifying permitted use,
  the security plan, no re-identification, no onward sharing, and the disposal schedule.
  Identified data are used solely for this protocol's objectives.
- **Retention:** identifiers and any linkage crosswalk are destroyed as soon as linkage
  and de-duplication are validated (target ⟨within N months⟩ of receipt). The
  de-identified analytic dataset is retained for the minimum period required by CNS norms
  for research-data custody (⟨5 years⟩) and then destroyed.
- **Disposal:** cryptographic erasure / secure wipe of identified files and any staging
  copies; documented destruction certificate filed with the ethics record. No identified
  data are archived or reused for any other study without a new protocol.

## 10. Team, custodians, and institutional data

⟨To complete before submission: responsible researcher and CV/Lattes; host institution
and CEP of record; co-investigators; custodian contacts and their authorisation letters
(anuência) for SES-RJ / SMS-Rio / DATASUS; Plataforma Brasil CAAE once generated;
funding; conflict-of-interest declaration; timeline.⟩

---

### Notes for the filer

- Every ⟨placeholder⟩ must be resolved before Plataforma Brasil submission; the custodian
  anuência letters are typically the rate-limiting step.
- Only the four external sources verified in `docs/health_citation_verification.md` are
  cited here. Note that PMC4544397 is a **neighbourhood/bairro-level** analysis, not a
  setor-level one — the protocol's justification (§1) deliberately rests on that gap, so
  do not upgrade its description to "setor-level."
- SINAN/SIM field availability and the exact custodian route (state SES-RJ vs municipal
  SMS-Rio vs DATASUS) should be confirmed with the vigilância epidemiológica before
  finalising §3.
