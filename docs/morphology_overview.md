# Favela Morphology — what we measure, and why

A guided read of the morphology analysis: the question it answers, the figures
that answer it, and an honest, decision-oriented account of where the method holds
and where it breaks. Every figure is clickable in the
[signature & roughness gallery](/outputs/cross_site/signature/figures_v2/index.html).

## The goal

Favelas are not formless. Beneath the apparent chaos there is a **recurrent built
fabric** — and that fabric *governs the environment people live in*: how much sky
and winter sun a passage gets, how deep its canyons are, how wind moves through it.
Two questions:

1. **Is there a measurable favela morphological signature** — a few fabric *types*
   that recur across independent settlements?
2. **Does that fabric predict environmental deprivation** (sun, ventilation), so we
   can prioritize from morphology *alone* where simulation is scarce?

We work per **10 m grid cell** across the 5 campaign favelas (~64,000 built cells;
the 3 calibration sites are kept aside). Each cell carries a **fabric vector** — plan
density λp, mean height, height variability σH, frontal-area density λf and its
anisotropy, and slope — deliberately kept separate from the *experienced* conditions
(sky-view, sun, canyon depth) so the latter can serve as out-of-sample validation.

## The six morphotypes, at a glance

We cluster the standardized fabric vector (GMM, k=6). Here is an idealized 40 m
street section of each type — the quickest way to see the difference:

![Idealized morphotype sections](/outputs/cross_site/signature/figures_v2/morphotype_schematics.png)

*Density rises T0→T5 along a* **Fringe → Consolidated → Core** *ladder; T1 and T4 are
the same densities tipped onto ~19° hillside. Names below are council-proposed.*

| type | name | λp | H (m) | H/W | what it is |
|------|------|----|----|----|-----------|
| T0 | **Open Fringe** | 0.22 | 4.4 | 0.35 | sparse flat single-storey edge — the matrix the favela grows into |
| T1 | **Hillside Fringe** | 0.51 | 4.6 | 0.86 | low-rise on ~19° slope; canyon from terrain, not mass |
| T2 | **Open Consolidated** | 0.68 | 6.8 | 0.79 | flat, dense, sky still intact *(conditional — flat sites only)* |
| T3 | **Shaded Consolidated** | 0.69 | 7.9 | 1.49 | flat, dense, daylight lost to frontal density *(conditional)* |
| T4 | **Hillside Core** | 0.78 | 7.8 | 2.58 | steep dense hillside, fully sun-starved |
| T5 | **Saturated Core** | 1.00 | 7.5 | 3.47 | plan-area maxed flat interior, deep-canyon labyrinth |

## Where each type lives

The mix is not the same across favelas — and the pattern is topographic:

![Morphotype composition per favela](/outputs/cross_site/signature/figures_v2/composition_by_site.png)

*Hillside favelas (Vidigal, Rocinha, Complexo do Alemão) are dominated by the
**Hillside** types (T1/T4); the flatter Rio das Pedras and Maré by the flat-dense
types (T3/T5). Per-favela maps: Vidigal, Rocinha, Rio das Pedras, Complexo do Alemão,
Maré are in the gallery (`map_<site>.png`).*

![Vidigal morphotype map](/outputs/cross_site/signature/figures_v2/map_vidigal.png)

*Vidigal — the spatial pattern: Hillside Core (orange) on the steep upper slopes,
Hillside Fringe (green) at the edges. Grey = no street observer.*

## Validation 1 — the types recur across cities

A cluster is only a *signature* if it reappears in independent settlements:

![Cross-site recurrence](/outputs/cross_site/signature/figures_v2/recurrence.png)

*T0, T1, T4, T5 recur across ≥4 sites — genuine favela signatures. T2/T3 appear only
in the flat sites: **conditional** morphotypes, present where flat buildable land
exists. k=6 is stable (bootstrap ARI 0.90).*

## Validation 2 — fabric predicts the lived environment

The decisive test: the *experienced* conditions **never entered the clustering**. If
they nonetheless worsen monotonically along the fabric spine, the typology is real:

![Fabric × experience](/outputs/cross_site/signature/figures_v2/experience_dotplots.png)

*They do. Sky-view falls 0.65 → 0.10 and the fraction below the WHO 2 h winter-sun
floor climbs 0 → 1.0 from T0 to T5. **Saturated Core reliably produces sun-starved
deep canyons** and recurs in all 5 sites: the prioritization signal, from geometry
alone.*

## Validation 3 — the signature at tissue scale (morphotopes)

A single 10 m cell is the right unit to *measure*, but a "favela signature" is really
a **tissue** — a block-scale mix of cell-types. So we take each cell's morphotype
*composition* over a 50 m window and cluster those into **morphotopes**:

![Morphotope maps](/outputs/cross_site/signature/figures_v2/morphotope_maps.png)

*Five tissues, as coherent regions (no salt-and-pepper): Vidigal is fringe tissue,
Rocinha a dense hillside-core, the flat sites carry dark flat-core tissue in a
mixed matrix. **4 of 5 tissues recur across ≥3 favelas** — a stronger claim than at
the cell scale.*

![What each tissue is made of](/outputs/cross_site/signature/figures_v2/morphotope_profile.png)

*The cell-type mix of each tissue — and this **answers the "are T2/T3 real?"
critique**: the Shaded-Consolidated cell-type (T3) is not noise, it concentrates in
the flat dense-core tissue M4, a coherent tissue that recurs in the flat sites. The
"conditional" cell-types are real tissue states.*

---

# Aerodynamic roughness — and a finding you need for decisions

For wind/CFD we need the surface roughness z0 and displacement height zd. We estimate
them morphometrically (UMEP; Kanda 2013 primary). **A 3-expert council review found
the per-cell estimate is physically invalid across most favela fabric.** This section
is written to inform what you do next.

![Physical validity](/outputs/cross_site/signature/figures_v2/roughness_validity.png)

*Per site, the fraction of built cells where the morphometric z0/zd is physically
valid (green) vs impossible (red/brown).*

### What "invalid" means, concretely

Two failure signatures, in **53–75% of built cells**:

- **zd > H_max** — the computed displacement height sits *above the tallest building
  in the cell*. Displacement height is, by definition, the level at which the mean
  drag acts; it cannot exceed the canopy top. When the model returns this, it is
  reporting something physically impossible.
- **z0 → 0** — the roughness length collapses toward zero (literally `0.0` in the
  densest cells), i.e. the model calls some of the **roughest fabric on Earth
  "aerodynamically smooth."**

### Why it happens (so you can judge how fixable it is)

It is **not a coding bug** — it is the *method* used far outside its support:

- Every morphometric method (Lettau, Macdonald, Raupach, Millward-Hopkins, Kanda) was
  fit on regular obstacle arrays with **λp ≲ 0.4, λf ≲ 0.4**. Favela cells run
  **λp > 0.5 (up to 1.0) and λf ≈ 2** — one to two orders of magnitude past the fit.
- In that regime the drag-partition term **saturates**: as packing → solid block, the
  formula sends form drag → 0 (the "skimming" asymptote) and pushes displacement
  toward (and past) the canopy top. We verified the invalid cells have **median
  λp = 1.0 and Kanda's height predictor X saturated at ~0.97** — exactly the corner
  where the algebra breaks.
- So in these cells z0 is being set by the **σH/H_max ratio, not by the fabric** you
  think you are measuring. It is an extrapolation, not a measurement.

### What this does and does NOT invalidate

- **Does NOT touch the morphotype signature.** The signature uses λp, λf, σH, slope as
  *raw fabric descriptors* (no drag model); recurrence + the experience validation
  stand. Roughness is a separate downstream product.
- **Does invalidate the per-cell z0/zd map** as a physical roughness field. Treat
  `roughness_map.png` as illustrative only.
- **The patch scale is partly rescued.** Aggregating to the 100 m CFD patch averages
  the per-cell collapse (per-patch z0 came out 0.10–0.61 m, physically plausible) —
  but it inherits the same out-of-envelope uncertainty.

### The honest result, and your options

With ~20× disagreement between methods and **zero favela validation**, no single z0 is
defensible. The **method-spread envelope is the result**: morphometry cannot pin
favela z0 to better than ~1.5 orders of magnitude. Decision paths:

| option | what it buys | cost / risk |
|--------|-------------|-------------|
| **A. Report the envelope + validity flags, lead with CFD** *(recommended)* | Honest, publishable as a gap result; positions the CFD campaign as the validation | Needs the OpenFOAM run for any *absolute* z0 |
| **B. Use morphometric z0 only at patch scale, flagged** | A usable CFD-inlet prior now (per-patch, plausible band) | Still out-of-envelope; must carry the uncertainty band |
| **C. Terrain-following morphometry first** | Removes the slope confound in σH/λf; may shrink the impossible-value rate | A pipeline change (recompute λf/σH on a local datum); won't fix the λp→1 saturation |
| **D. Add a sheltering/porosity correction** (Millward-Hopkins has one) | Physically caps drag as λf→ high; could rescue some cells | Still unvalidated without CFD; a research step |

The recommendation is **A now, B for the CFD hand-off, and C+D as the research moves**
the council flagged — *none* of which substitutes for the CFD anchor (R-C). A further
hard limit: morphometric z0(θ) is **180°-symmetric by construction** (frontal area is
identical for opposite winds), so it can never represent channelling — only the CFD
breaks that symmetry.

## How this feeds the CFD pipeline

Roughness has *two decoupled roles*: the morphometric z0(θ) of the **upstream
settlement** sets each patch's **inlet ABL + turbulence target**; the **ground z0
inside the resolved patch stays small** (meshed buildings supply the drag). Per-patch
values are in `patch_roughness.csv`; the full contract — inlet-from-upstream-fetch,
two-zone ground, z0 floor for λp>0.5, drag-integral extraction on slopes,
homogeneity/GCI gates — is in `src/cfd_integration/README.md`. The CFD then *validates
and recalibrates* the morphometric estimate (R-C) and resolves the rougher-vs-smoother
question the geometry cannot.

## What we're still building (council's biggest moves)

- **Configuration metrics** — *party-wall adjacency added* (council's top "what's
  missing"): the fraction of each building's perimeter fused to a neighbour, a
  relational trait the intensity vector never saw. **Favela fabric is highly fused
  everywhere — 0.6–1.0 vs ~0.1 for detached formal blocks** — and it reveals a *new*
  axis: the **flat** types (T2/T3/T5) are near-fully party-walled while the
  **hillside** types (T1/T4) are more stepped/detached (`party_wall_by_type.png`).
  Street-network / *beco* width is the next configuration feature.
- **Terrain-following morphometry** — to separate hillside from fabric in σH/λf
  (option C above).
- **Block-scale morphotope** — a "favela signature" is arguably a ~50–100 m tissue
  (type composition + adjacency), not a single cell.

The honesty is the point: we report where the method holds (the signature, the
experience link) and where it breaks (per-cell roughness at extreme density), rather
than letting a reviewer find it first.
