# Patch dossier — VDG-P07 (Vidigal, Rio de Janeiro)

> IVF↔CFD contract-v1 traceability record. Authored on the patch-overlay-prep
> branch `feat/vdg-p07-overlay-prep`. This file is the single provenance anchor
> for everything under `patches/VDG-P07/`.

## 1. Identity & provenance

| Field | Value |
|---|---|
| `patch_id` | **VDG-P07** |
| Site | vidigal (Vidigal favela, Rio de Janeiro) |
| Stratum | `SVF3_SLP1_LP1` |
| Contract | `rectangular_domain_v1` (IVF↔CFD, v1.0, 2026-05-08) |
| CRS | **EPSG:31983** — SIRGAS 2000 / UTM zone 23S (verified on buildings + terrain) |
| Inputs origin | CFD repo `Airflow`, branch `feature/rectangular-ach-pipeline`, commit `78af4241a5e597d200c12a89be58eeb593ff27ec` ("feat(cfd): scaffold VDG-P07 Vidigal pilot patch (rectangular_domain_v1)") |
| Snapshot taken | by the patch-overlay-prep agent (tablet→laptop over ssh), from `~/Airflow/cases/VDG-P07/` |

**Lineage caveat (honest):** an earlier *cylindrical* VDG-P07 pilot was broken
and is superseded — see CFD `tracking/decisions_log.md:27-28` ("Supersedes:
VDG-P07 + MAR-P07 cylindrical pilots (broken / superseded)") and the recorded
SIGFPE incident `Airflow/docs/incidents/2026-05-08_vdg-p07_simplefoam_sigfpe/`
(simpleFoam SIGFPE on the cylindrical scheme). The artifacts vendored here are
from the **rebuilt rectangular_domain_v1** scaffold only. There is **no
`hypothesis_log.md` entry for VDG-P07** as of this snapshot — do not cite one.

## 2. patch_meta.json — full field set (as vendored)

| Key | Value | Key | Value |
|---|---|---|---|
| center_x | 680016.7399997711 | center_y | 7455860.080299854 |
| analysis_patch_shape | circle | analysis_patch_diameter | 100.0 m |
| cfd_domain_radius | 250.0 m | n_buildings_in_domain | 1033 |
| H_mean | 5.3062 m | H_max_analysis | 13.76 m |
| slope_deg | 8.763490940845845 | svf | 0.6495098039215687 |
| lambda_p | 0.2849881354948173 | porosity | 0.8085194226635466 |
| sigma_h | 1.9118487806309368 | building_coverage | 0.3081374884205929 |
| domain_upstream_m | 118.8 | domain_downstream_m | 256.4 |
| domain_lateral_m | 500.0 | domain_top_m | 68.8 |
| domain_blockage_ratio | 0.02 | source_data_required_m | 585.0 |
| domain_data_coverage | 1.0 | schema_version | 1 |

Domain extents satisfy contract v1 formulas (H_max=13.76): upstream
`5·H_max+R = 118.8` ✓, downstream `15·H_max+R = 256.4` ✓, lateral
`max(5·H_max+R, 5·W) = 500` ✓, top `5·H_max = 68.8` ✓. The baked CFD
domain is therefore **asymmetric along-wind** (upstream 118.8 ≠ downstream
256.4) — the chirality lever the Phase-3 gates exploit.

## 3. Vendored input artifacts (`inputs/`) + checksums

SHA-256, computed at source and re-verified in place after snapshot:

| File | bytes | sha256 |
|---|---|---|
| buildings.gpkg | 634880 | `af2844329ff240ecd563a655fbb7b5be084848ed4f2648cf488558bbfe36e4b7` |
| terrain.tif | 42057 | `b39e17ade43616d138b2dfdcc6f516524614f6de60ec67f7613e1431279a5c58` |
| patch_meta.json | 792 | `b09c1709dcbeda14e5948fd927d67206a8473f919f7e52150c8d72bb797b0843` |
| preflight_report.json | 2510 | `49ee72932c902f8d920fb10d855ae7aa0c393201afbc8f13697e2c584c23ca25` |

`inputs/SHA256SUMS` is the authoritative manifest (use `sha256sum -c`).

### buildings.gpkg
1033 features, EPSG:31983, bounds `[679765.56, 7455607.86, 680277.30, 7455983.08]`.
Height columns: `height` ≡ `altura` (m, 0–14.57), `base_height` (ground MASL,
135.36–279.24). **`height == 0.0` for exactly 5 of 1033 features** (and <0.5 m
for 25). This — *not* a NULL — is the "5 nulls" referenced upstream; Phase 2
flags these 5 degenerate footprints explicitly as a styled QA sublayer.
Median height 5.32 m, P95 9.48 m, max 14.57 m (consistent with
H_max_analysis 13.76).

### terrain.tif
EPSG:31983, 102×102, float32, **5.0 m** pixel, `AREA_OR_POINT=Area`,
nodata `3.3999999521443642e+38`, bounds `[679763.724, 7455602.608,
680273.724, 7456112.608]`, transform origin `(679763.7241, 7456112.6084)`
px `(5, -5)`. Elevation 115.6–352.5 m (203 nodata cells at the data-poor
corners). The Ø100 m analysis disk is fully covered. **This lattice is the
snap target for every Phase-2/3 raster** (verdict §5).

### patch_meta.json / preflight_report.json
preflight `pass_all: true` (files_present, metadata_schema, crs_consistency
31983/31983, building_data 1028/1033 valid≡height>0, domain_coverage,
z0_blocken ok, no_collision). Vendored verbatim.

## 4. CFD linkage

- Contract: IVF `src/cfd_integration/rectangular_domain_v1.json` (domain
  sizing/blockage) + `src/cfd_integration/README.md` (result format).
- Per-direction georef params are emitted by the CFD case scaffold into
  `tmp_<slug>_dir<NNN>/case_meta.json`:
  `rotation_rad_local_ccw`, `local_offset_utm{dx,dy,dz}`, `wind_direction_deg`,
  `domain_extent_m`, `patch_meta_contract`. Phase 3 **consumes and
  checksums** these — it does not re-derive rotation from wind direction.
- VTI producer: `Airflow/scripts/postprocess/lma_postprocess.py` →
  `canopy_tau_field.vti` (ImageData, local frame, origin = patch centre,
  ±50 m, 1 m, node-sampled, z absolute 0..2·H_mean; τ = age-of-air [s];
  Sandberg `ACH = 3600/mean(τ)`).
- Existing direction-composite `Airflow/scripts/postprocess/lma_aggregate_directions.py`
  is **superseded for the overlay deliverable** (it composites arithmetic-mean
  τ with default uniform 1/N and emits no CRS/GeoTIFF — see §6).
- Submission chain: `Airflow/scripts/hpc/patch_{smoke,mesh,solver,postprocess,
  campaign}.sbatch`.
- **Handoff-reported, UNVERIFIED from tablet (no live orcd master at snapshot):**
  smoke chain — mesh job `14092625` (RUNNING), solver `14092626`
  (PENDING, `afterok`). Recorded as provenance claims, not verified facts.

## 5. Wind rose (composite weights)

`IVF/data/vidigal/wind_rose.json` — **`quality_flag: "measured"`** (NOT a
placeholder): INMET Forte de Copacabana **A652**, 2015–2024, n=85,103
(1,214 calm <0.5 m/s excluded). Strongly bimodal sea-breeze regime —
E `f=0.303`, W `f=0.228` (E+W ≈ 53 %), SE highest mean speed 3.37 m/s.
Phase-3 composite uses `freq × mean_speed` per the contract's
`weighted_by_wind_rose(weight_by="freq_speed")` convention. Equal-weighting
is scientifically inadmissible for this rose (verdict §2).

## 6. Expected deliverables (this patch)

| Path | Phase | Status |
|---|---|---|
| `patches/VDG-P07/dossier.md` | 1 | this file |
| `patches/VDG-P07/inputs/{buildings.gpkg,terrain.tif,patch_meta.json,preflight_report.json,SHA256SUMS}` | 1 | vendored |
| `patches/VDG-P07/provenance.json` | 1 | machine-readable mirror of §1–5 |
| `patches/VDG-P07/qgis/VDG-P07.qgz` (+ `.qml` styles, `make_qgz.py`) | 2 | pending |
| `scripts/cfd_overlay/vti2geotiff.py` (+ tests) | 3 | pending |
| `patches/VDG-P07/overlay/{per-dir,composite}.tif` + `.georef.json` | 3 (run) | gated on smoke COMPLETED + real VTIs |

## 7. Phase-3 georef contract (council verdict, binding)

vti2geotiff conforms to the 9-point verdict: compose in EPSG:31983
(one bilinear resample/dir, invert bake by `−rotation_rad_local_ccw` +
`local_offset_utm`, no double interp); composite in **ventilation space**
`ACH_comp=Σ fᵢ·ACHᵢ`, `τ_comp=1/ACH_comp`, `ACHᵢ=3600/τᵢ` per cell, `fᵢ`
= freq×speed wind rose, never equal-weight; **terrain-following** z
(pedestrian 1.5–2 m AGL primary; canopy [z0,H_mean] vol-mean secondary;
z vs local terrain, 8.76° slope); solid cells → **NaN before** interp;
Ø100 m disk clipped **last** with a resample halo; grid lattice-locked to
terrain.tif (1 m, integer-nested in the 5 m terrain lattice, EPSG:31983
GeoKeys, `AREA_OR_POINT=Area`, explicit nodata sentinel); georef pinned in
GeoTIFF tags + `.georef.json` sidecar (CRS+axis, per-dir
rotation/pivot/units, dx/dy derivation, z-def+datum, composite rule, units,
nodata, all versions + per-VTI sha256 + patch_id); fail-closed gates
(synthetic asymmetric round-trip, corner/centre coord assertion vs
patch_meta, building-footprint RMS, axis-order, outside-disk==nodata) and
`--dry-run` golden-diff, dry-run on the smoke VTI before the 8-dir run.

**Deliberate reading of verdict §5 (surfaced):** "same pixel size, origin
an integer #pixels from terrain origin" is implemented as *lattice-phase
locking at 1 m* (terrain's 5 m = 5×1 m, exactly nested) rather than
coarsening the 1 m CFD field to 5 m — preserves CFD resolution while
keeping exact co-registration. Documented in the sidecar.
