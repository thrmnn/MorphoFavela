# Favela Morphology — what we measure, and why

A guided read of the morphology analysis: the question it answers, the figures
that answer it, and an honest account of where the method holds and where it
breaks. Each figure is clickable in the [signature & roughness gallery](/outputs/cross_site/signature/figures_v2/index.html).

## The goal

Favelas are not formless. Beneath the apparent chaos there is a **recurrent
built fabric** — and that fabric *governs the environment people live in*: how
much sky and winter sun a passage gets, how deep its canyons are, how the wind
moves through it. Our question is twofold:

1. **Is there a measurable favela morphological signature** — a small set of
   fabric *types* that recur across independent settlements?
2. **Does that fabric predict environmental deprivation** (sun, ventilation),
   so we can prioritize from morphology *alone* where simulation is scarce?

We work per **10 m grid cell** across the 5 campaign favelas (Vidigal, Rocinha,
Rio das Pedras, Complexo do Alemão, Maré — ~64,000 built cells). The 3 calibration
sites are kept aside. Every cell carries a **fabric vector** — plan-area density
λp, mean height, height variability σH, frontal-area density λf, its directional
anisotropy, and terrain slope — deliberately separated from the *experienced*
conditions (sky-view, sun hours, canyon depth) so the latter can serve as
out-of-sample validation.

## The signature — six morphotypes

We cluster the standardized fabric vector (GMM, k=6). The fingerprint heatmap is
each type's standardized profile — red above average, blue below:

![Morphotype fingerprints](/outputs/cross_site/signature/figures_v2/fingerprint_heatmap.png)

*Reading it: the types run from **T0 Open Footing** (sparse, flat, single-storey
fringe) up a densification spine to **T5 Saturated Core** (plan-area saturated,
deep-canyon interior). Two types — T1, T4 — are the same densities tipped onto
~19° hillside. The names encode a* density → enclosure *spine crossed with a*
flat/steep *switch.*

| type | name | what it is |
|------|------|-----------|
| T0 | **Open Footing** | sparse, flat, single-storey fringe — the matrix the favela grows into |
| T1 | **Stepped Footing** | low-rise on ~19° slope; canyon from terrain, not mass |
| T2 | **Massing Plateau** | flat consolidated mid-rise, sky still intact (conditional) |
| T3 | **Shaded Plateau** | flat dense; daylight lost to frontal density (conditional) |
| T4 | **Cliff Stack** | steep dense hillside, H/W 2.6, fully sun-starved |
| T5 | **Saturated Core** | λp=1 maxed flat interior, deep-canyon 0.89, H/W 3.5 |

*(Names are council-proposed, pending sign-off.)*

## Validation 1 — the types recur across cities

A cluster is only a *signature* if it reappears in independent settlements. The
recurrence matrix shows each type's share per site:

![Cross-site recurrence](/outputs/cross_site/signature/figures_v2/recurrence.png)

*T0, T1, T4, T5 recur across ≥4 sites — genuine favela signatures. T2/T3 appear
in only the flat sites: **conditional** morphotypes, present where there is flat
buildable land. k=6 is stable (bootstrap ARI 0.90).*

## Validation 2 — fabric predicts the lived environment

The decisive test: the *experienced* conditions (sky-view, winter sun, canyon
depth) **never entered the clustering**. If they nonetheless worsen monotonically
along the fabric spine, the typology is environmentally real:

![Fabric × experience](/outputs/cross_site/signature/figures_v2/experience_dotplots.png)

*They do. Sky-view falls 0.65 → 0.10 and the fraction of observers below the WHO
2 h winter-sun floor climbs 0 → 1.0 from T0 to T5 — an out-of-sample confirmation.
**Saturated Core reliably produces sun-starved deep canyons**, and recurs in all 5
sites: the prioritization signal, from geometry alone.*

## Aerodynamic roughness — and an honest limit

For wind/CFD we need the surface roughness z0 and displacement zd. We estimate
them morphometrically (UMEP; Kanda 2013 primary). **But an expert council review
found the per-cell estimate physically invalid across most of the favela fabric:**

![Physical validity](/outputs/cross_site/signature/figures_v2/roughness_validity.png)

*In 53–75% of cells the model returns the impossible — displacement above the
tallest building, or z0 collapsing to ~zero (the skimming asymptote) for some of
the roughest fabric on Earth. λp>0.5 and λf≈2 are 1–2 orders past every method's
calibration; the drag formula saturates. **This is model extrapolation, not a
measurement.** With ~20× disagreement between methods and no favela validation,
the* method-spread envelope *is the result: morphometry cannot pin favela z0 to
better than ~1.5 orders — CFD is required.* A further constraint: morphometric
z0(θ) is **180°-symmetric by construction** (frontal area is the same for wind
from opposite directions), so it can never represent channelling — only the CFD
can break that symmetry.

## How this feeds the CFD pipeline

The roughness has *two decoupled roles*. The morphometric z0(θ) of the **upstream
settlement** sets each CFD patch's **inlet ABL profile + turbulence target** (the
approach flow it should see); the **ground z0 inside the resolved patch stays
small** because the meshed buildings already supply the drag. The per-patch values
are written to `patch_roughness.csv`; the full contract (inlet-from-upstream-fetch,
two-zone ground, z0 floor, drag-integral extraction on slopes, homogeneity/GCI
gates) is in `src/cfd_integration/README.md`. The CFD then *validates and
recalibrates* the morphometric estimate (R-C, pending the OpenFOAM run) and
resolves the rougher-vs-smoother question the geometry can't.

## What we're still missing (council)

- **Configuration**, not just intensity: party-wall adjacency and street-network /
  *beco* width — the favela's defining relational traits — are not yet in the vector.
- **Scale**: a "favela signature" is arguably a block-scale *morphotope* (type
  composition + adjacency over ~50–100 m), not a single cell; the cell is the right
  *measurement* unit but a coarser unit may be the right *reporting* one.
- **Terrain-following morphometry** to separate the hillside from the fabric in σH/λf.

These are the next moves. The honesty is the point: we report where the method
holds (the signature, the experience link) and where it breaks (per-cell roughness
at extreme density), rather than letting a reviewer find it first.
