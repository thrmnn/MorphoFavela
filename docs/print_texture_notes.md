# Version A — performance-texture tiles: variants, print risk, recommendation

A 150 × 150 mm realistic tile of one CFD analysis patch: true terrain, buildings
extruded at real height with **smooth roofs**, and a performance field classified
into four classes encoded as a **ground-only** relief texture. Three treatments
are produced as separate watertight STL variants from one shared base. Two fields
are wired:

- **Sunlight** (`--field sunlight`, default **Vidigal VDG-P07**) — street-level
  **winter sun-hours**. VDG-P07 is the only patch with a balanced spread across all
  four sun-hour classes *and* it is steep, so it stress-tests texture on slopes.
- **Airflow** (`--field ventilation`, **Rio das Pedras RDP-P20**) — pedestrian
  ventilation from the RDP-P20 CFD return (8-direction steady simpleFoam / kΩSST,
  ~15 k points/direction), wind-rose-weighted **|U|/U_ref**, split into quartiles.
  This is a **proxy** for local mean age of air (LMA): low speed = stagnant = old
  air. **LMA proper is not in this steady set** — it needs a passive-scalar
  transient run — so the tile is labelled ventilation, not LMA. The 8-direction
  mean is spatially flat in absolute terms (all ~0.28–0.39 |U|/U_ref), so classes
  are *relative quartiles* within the patch, not absolute comfort bands.

Build: `python scripts/build_texture_tile.py [--field sunlight|ventilation --site … --patch …]`
→ `outputs/{site}/print/texture_tile/`.

## Classes (smooth → densest)

| Class | Sunlight (winter sun-hours) | Airflow (U/U_ref quartile) | Texture |
|-------|-----------------------------|------------------------------|---------|
| 1 | > 6 h (full sun) | Q1 · best-ventilated | smooth |
| 2 | 4–6 h | Q2 | lightest |
| 3 | 2–4 h | Q3 | medium |
| 4 | < 2 h (deep shade) | Q4 · most stagnant | densest |

Sunlight: each ground cell takes the nearest street-observer's
`solar_hours_winter`. Airflow: nearest CFD sample-point |U|/U_ref, wind-rose
averaged over the 8 directions, then quartile-binned. Building cells are never
textured in either field.

## Shared design rule

The whole tile is a **single heightfield solid** (terrain + building heights baked
into one draped surface), so every variant is a single watertight manifold with
no boolean CSG. Texture is a **vertical z-displacement** applied to ground cells
only — not along the true surface normal. On a slope a vertical dimple is the
printable choice (no overhangs for FDM), and at these depths the difference from a
normal-aligned cut is sub-0.1 mm, so the steep-slope legibility the brief asks for
is preserved. This is the one deliberate deviation from "follow the surface
normal", made for printability.

## V1 — Stippling (recommended for FDM)

1.0 mm⌀ × 0.5 mm hemispherical pits on a jittered grid; spacing 4.0 / 2.5 / 1.5 mm
for Class 2 / 3 / 4 (Class 1 smooth). At a 0.4 mm nozzle a 1 mm pit is ~2.5
nozzle-widths and 0.5 mm is 2–3 layers — it registers but rounds off; **the
density gradient survives even when individual pits blur**. No thin standing walls
or single-layer floors to lose. Resin renders each pit crisply.
**Risk: low.** Est. FDM print (0.2 mm, rough, slicer-confirmed): ~4–6 h.

## V2 — Contour bands (isohel lines)

Continuous 0.6 mm × 0.4 mm grooves on the class boundaries, plus a subtle
0.2 mm-per-class step-down (Class 1 highest → Class 4 lowest). 0.6 mm ≈ 1.5
nozzle-widths, so grooves print as a shallow valley rather than a crisp line, and
the 0.2 mm steps are ~1 layer — near the FDM floor. **Flagged risk:** where a
class boundary runs parallel to a steep terrain break the groove competes with the
slope and reads ambiguously; on VDG-P07 several boundaries sit on > 20° ground.
Best on flatter fabric (e.g. a Maré patch); resin holds the line width.
**Risk: medium (FDM), low (resin).** Est. ~4–6 h.

## V3 — Directional hatching

Parallel 0.5 mm × 0.3 mm grooves oriented along the dominant shadow azimuth;
spacing 4.0 / 2.5 / 1.5 mm by class (Class 1 smooth). 0.5 mm is ~1.25
nozzle-widths and 0.3 mm is the stated FDM engraving floor — the **shallowest of
the three**, most at risk of being smeared by a 0.4 mm nozzle. On slopes > 20° the
fixed-azimuth grooves alias against the draped grid. Directionality is legible;
absolute depth is marginal on FDM, fine on resin.
**Risk: medium–high (FDM), low (resin).** Est. ~4–6 h.

## Recommendation

**For FDM at this scale, print V1 (stippling).** Isolated concave pits carry no
thin walls or single-layer floors, so the shade gradient survives a 0.4 mm nozzle
where V2's 0.2 mm steps and V3's 0.3 mm grooves sit at or under the FDM floor.
Print **V2 / V3 in resin** if you want the crisp line work. All three are
watertight; the vertical-displacement texture is printable on the tile's steep
ground without supports.
