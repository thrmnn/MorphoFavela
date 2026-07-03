# Roughness & wall treatment — a plain-language guide

*A review-friendly walkthrough of the concepts behind the roughness track and the
CFD hand-off. No prior wind-engineering needed. The four schematics are concept art
(not data); the real numbers live in `patch_roughness.csv` and the plan
(`cfd_parameter_estimation_plan.md`).*

---

## Why this matters

To simulate wind and ventilation in a favela, the CFD model needs to know **how rough
the ground is to the wind** — a city of tightly-packed buildings drags on the airflow
very differently from an open field. That "how rough" is captured by two numbers,
**z₀** and **z_d**, that we estimate from building geometry alone. This page explains
what they mean, the one trap that ruins them if you're not careful, why favela density
pushes the estimate to its breaking point, and how the number becomes an actual
boundary condition in the simulation.

---

## 1. The log wind profile — what z₀ and z_d mean

![The log wind profile over a building canopy, marking z0, zd and H](/docs/roughness_explainer/log_profile.png)

Above any rough surface the wind speed grows with height along a **logarithmic curve**.
Two numbers pin that curve down:

- **z₀ (roughness length)** — *how bumpy the surface is to the wind.* Big z₀ = rough =
  the wind is slowed over a deeper layer. A lawn has z₀ ≈ a few millimetres; a city, a
  metre or more.
- **z_d (displacement height)** — *how far up the wind's effective "floor" is lifted.*
  Dense buildings hold the flow up off the real ground; the wind behaves as if the
  ground were at height z_d. In tightly-packed favela fabric **z_d can sit above the
  mean roof height** — the tall roofs take most of the drag. That looks wrong but it's
  physically correct.

Everything downstream (the inlet wind profile, the turbulence) is built from these two
numbers plus the friction velocity u\* via `U(z) = (u*/κ)·ln((z − z_d)/z₀)`.

---

## 2. The two roles of z₀ — never double-count

![Two roles of z0: upstream fetch inlet vs resolved-patch ground](/docs/roughness_explainer/two_roles.png)

This is the single most important idea in the coupling, and the easiest to get wrong.
z₀ plays **two different roles** and they must be kept apart:

1. **The upstream fetch (inlet).** Before the wind reaches the patch we care about, it
   crosses a lot of city. *That* fabric sets how rough and turbulent the **arriving**
   wind is — it parameterises the simulation's inlet.
2. **The ground inside the patch.** Inside the analysis patch the buildings are **drawn
   explicitly** (meshed). They already create drag just by being there. So the ground
   *underneath* them must stay almost smooth (z₀ ≈ 1–3 cm).

The trap (Blocken 2007): if you also make the in-patch ground rough, you count the same
buildings **twice** — once as drawn geometry, once as ground roughness. The flow comes
out wrong in a way that looks plausible. Keep the two roles separate.

---

## 3. Roughness regimes — and why favela density breaks the estimate

![Roughness regimes: isolated, wake interference, skimming, and the favela out-of-envelope zone](/docs/roughness_explainer/regimes.png)

As you pack more buildings onto the ground (rising **λ_p**, the built fraction),
roughness does **not** rise forever. It follows three regimes:

- **Isolated roughness** (sparse) — each building snags the wind on its own.
- **Wake interference** (medium) — buildings shelter each other; roughness **peaks**.
- **Skimming flow** (dense) — the wind skates *over* the tops as if over a rough lid;
  the roughness **collapses back toward zero**.

Here's the problem: **favelas sit at λ_p > 0.5**, deep in the skimming regime and
**outside the calibration range of every published method** (all fitted below ~0.5). So
the estimate collapses toward z₀ ≈ 0 — which is useless as the denominator of that
`ln(z/z₀)` profile. Our honest position: the four methods disagree by **4×–148×** here,
so the *spread is the result*, and the absolute number is **CFD-gated** (only a real
simulation can settle it). To keep the pipeline runnable in the meantime we **floor z₀
to 0.03 m** and flag every floored patch.

---

## 4. Wall treatment — turning z₀ into a boundary condition

![Wall treatment: ks = 9.793 z0 / Cs and the ks < yP rule](/docs/roughness_explainer/wall_treatment.png)

A CFD solver doesn't take "z₀" directly on a wall. It uses a **rough-wall function**
that needs an **equivalent sand-grain roughness height k_s** — literally, how tall the
bumps would be if the surface were sandpaper. The conversion (OpenFOAM/Fluent) is:

> **k_s = 9.793 · z₀ / C_s**  (with C_s ≈ 0.5)

with one hard rule: **k_s must be smaller than the first mesh cell (k_s < y_P)** — the
roughness has to fit *inside* the near-wall cell or the wall model is invalid. This is
why wall treatment is genuinely the CFD side's call: y_P is a property of *their* mesh,
and C_s of *their* solver. From our side we supply the floored z₀ and flag which patches
are floored; the k_s conversion is documented, not owned.

Put together with §2: the **approach floor** gets a rough wall (z₀ = the upstream-fetch
value); the ground **under the resolved buildings** gets a small, mesh-valid z₀. Two
zones, never one.

---

## What we shipped, and the honest caveat

- **Shipped:** the floored z₀ + flag + a provenance stamp on `patch_roughness.csv`, so a
  CFD case can read a usable, non-zero roughness and know exactly which morphometric
  baseline produced it.
- **Requested:** a single real pilot simulation (MAR-P07) to *measure* z₀ from the flow
  and check our morphometric estimate — see the plan and
  `src/cfd_integration/README.md` §Pilot request.
- **The caveat, stated plainly:** at favela density the morphometric z₀ is a **screening
  estimate, not a measurement**. It is invalid in 53–75 % of cells and CFD-gated. Do not
  read the absolute number as truth until the pilot anchors it.

*Full method + decisions: `docs/roughness_plan.md`, `docs/roughness_decisions.md`,
`docs/cfd_parameter_estimation_plan.md`. Schematics regenerate via
`scripts/build_roughness_schematics.py`.*
