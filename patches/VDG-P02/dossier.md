# VDG-P02 — Vidigal pilot candidate (TLS-validatable)

**Status:** PREPARED, not launched. CFD scaffold committed `bf163be` on
`feature/rectangular-ach-pipeline`; preflight 7/7 PASS. Gated on the VDG-P07
smoke recipe verdict + operator decision.

## Why VDG-P02
Of 22 Vidigal candidate patches, VDG-P02 is the **only one with substantial
terrestrial-LiDAR coverage**: 66.6%
of its Ø100 m analysis disk lies inside the TLS LoD2
scan hull (VDG-P04 grazes it at 9.5%; VDG-P07 and the rest = 0%). This enables
direct validation of the CFD building geometry against the terrestrial scan —
the strategic reason it is the next candidate.

## Geometry (extreme — read before launch)
| metric | VDG-P02 | VDG-P07 (ref) |
|---|---|---|
| slope_deg | 26.6 | 8.8 |
| lambda_p | 0.960 | 0.285 |
| svf | 0.074 | 0.65 |
| H_mean (m) | 8.46 | 5.31 |
| buildings | 1825 | 1033 |

VDG-P02 is the hardest patch in the catalog (near-wall-to-wall packing, steep,
deep-canyon). The dense+steep mesh recipe applies; it may still hit the
`checkmesh-underdetermined-cells` trap beyond the recipe's reach.

## Inputs (vendored, sha256 in inputs/SHA256SUMS)
buildings.gpkg, terrain.tif, patch_meta.json, preflight_report.json — generated
by `scaffold_from_ivf.py VDG-P02` from the IVF vidigal envelope
(per_patch_indicators + buildings/dtm_extended_300m).

_generated 2026-05-18 15:42_

## Launch readiness — council + premortem verdict (2026-05-18, LOCKED)

**Do NOT 8-dir launch.** 3-lens council + premortem-auditor: the dense+steep
mesh recipe holds <10% (fails ~85%) on VDG-P02 — categorically outside the
band VDG-P07 (λp 0.285) defines. Plan = serial gated ladder:
**G0** wait for VDG-P07 smoke#2 verdict → **G1** ONE instrumented mesh-only
smoke (worst-incidence, no solver) as a *failure-boundary-characterization*
experiment → **G2** 1-dir full → **G3** 8-dir. Deliverable this cycle = a
parameterized failure boundary, not 8 results.

Escalation pre-authorized (mesh-doctor, merge-before-refine): ① STL pre-merge
of sub-cell/party-wall gaps (favela buildings share walls — physically
truthful) ② planar-terrain transform (rotate 26.6°→flat; zero physical cost)
③ drop prism layers (fine/better for ACH) ④ surgical post-merge refinement.
No blind retries (operator directive). Full register: Airflow
`tracking/premortem_VDG-P02_2026-05-18.md`.
