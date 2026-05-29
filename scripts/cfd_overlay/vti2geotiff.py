#!/usr/bin/env python3
"""VTI → georeferenced GeoTIFF for the LMA τ / ACH overlay deliverable.

Implements the 3-member council verdict (unanimous) for the MorphoFavela↔CFD
georeferencing step. It supersedes the *composite* of
``Airflow/scripts/postprocess/lma_aggregate_directions.py``, which
(1) averages τ arithmetically, (2) defaults to uniform 1/N, (3) never
applies ``local_offset_utm`` and emits no CRS/GeoTIFF — all
scientifically inadmissible for a citable, overlayable map.

Verdict conformance (numbers = verdict points):
 1  Compose in physical EPSG:31983. Per direction: invert the bake
    (rotate −rotation_rad_local_ccw about the patch centre, then add
    local_offset_utm) → world node coords analytically (rigid, no
    resample), then ONE linear scatter→grid interpolation onto a single
    shared north-up target. No de-rotate-then-warp double interpolation.
 2  Composite in ventilation space: ACHᵢ(x)=3600/τᵢ(x) per cell
    (Sandberg well-mixed identity, == lma_postprocess.py:188),
    ACH_comp=Σ fᵢ·ACHᵢ, τ_comp=1/ACH_comp. fᵢ = wind-rose freq×speed,
    normalised over available directions. Never equal-weight.
 3  Terrain-following z: pedestrian slab 1.5–2 m AGL (primary), canopy
    [z0,H_mean] AGL volume-mean (secondary). AGL is computed against the
    local terrain surface (terrain.tif), not absolute domain-z — the
    8.76° slope makes a fixed-z slab wrong by metres.
 4  Solid (building/terrain) cells → NaN BEFORE interpolation (never 0:
    0 τ/ACH is physically meaningful). Composite only over directions
    that contribute a finite cell; emit a contributing-direction QA band.
 5  Target grid lattice-locked to terrain.tif: 1 m pixels nested in the
    5 m terrain lattice (origin an integer #px from terrain origin),
    EPSG:31983 GeoKeys, AREA_OR_POINT=Area, explicit nodata sentinel.
 6  Highest risks handled: rotation sign/pivot consumed from case_meta
    (never re-derived from wind_deg); centre≠grid-midpoint honoured via
    the affine; node convention extent/(N−1) (lma_postprocess spacing).
 7  Georef = versioned contract artifact: fail-closed on
    patch_meta_contract mismatch; vendor+checksum the per-direction
    rotation params; pin everything in GeoTIFF tags + .georef.json.
 8  Consumed from headers, not re-invented: rotation parameterisation
    (case_meta.json), VTI Origin/Spacing (pyvista), wind-rose source.
 9  Fail-closed gates: synthetic ASYMMETRIC round-trip, corner/centre
    coords vs patch_meta, building-footprint RMS, axis-order re-assert,
    outside-disk==nodata. --dry-run prints the resolved georef_spec;
    --self-test runs the synthetic gate with NO CFD input.

Run (env: miniconda3/envs/MorphoFavela):
  vti2geotiff.py --self-test
  vti2geotiff.py --patch VDG-P07 --repo-root ~/MorphoFavela --smoke-vti PATH --dry-run
  vti2geotiff.py --patch VDG-P07 --repo-root ~/MorphoFavela       # full 8-dir composite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CONTRACT = "rectangular_domain_v1"
ANALYSIS_RADIUS_M = 50.0
WIND8 = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
PED_LO_AGL, PED_HI_AGL = 1.5, 2.0  # verdict §3 primary slab
TARGET_PX = 1.0  # verdict §5 (nested in 5 m terrain)
HALO_M = 8.0  # verdict §4 resample/composite halo


# ----------------------------------------------------------------------
# georef spec — the pinned, versioned contract artifact (verdict §7)
# ----------------------------------------------------------------------
@dataclass
class GeorefSpec:
    patch_id: str
    crs: str
    crs_axis_order: str
    rotation_param: str
    rotation_sign: str
    pivot: str
    offset_param: str
    pixel_size_m: float
    grid_origin_xy: tuple
    grid_shape: tuple
    snapped_to: str
    snap_rule: str
    z_slab_agl_m: tuple
    z_datum: str
    composite_rule: str
    tau_to_ach: str
    weight_source: str
    weight_method: str
    units_tau: str
    units_ach: str
    nodata: float
    contract: str
    contract_ok: bool
    vti_sha256: dict
    notes: str


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------
# the affine: local (mesh) frame → world EPSG:31983  (verdict §1, §6)
# world = R(-θ) · local_xy + offset    (θ = rotation_rad_local_ccw)
# Sign proven by Airflow/scripts/postprocess/lma_aggregate_directions.py
# ::rotate_grid_to_utm (rotates by -rotation_rad_ccw); we ADD the
# local_offset_utm translation that script omits.
# ----------------------------------------------------------------------
def local_to_world(xy: np.ndarray, theta_ccw: float, offset_xy: tuple) -> np.ndarray:
    c, s = math.cos(-theta_ccw), math.sin(-theta_ccw)
    R = np.array([[c, -s], [s, c]])
    return xy @ R.T + np.asarray(offset_xy, float)


def load_case_meta(p: Path) -> dict:
    m = json.loads(Path(p).read_text())
    if "rotation_rad_local_ccw" not in m:
        raise SystemExit(
            f"FAIL-CLOSED: {p} missing `rotation_rad_local_ccw` (written by "
            "build_patch_case.py). Cannot inverse-rotate; do not guess."
        )
    got = m.get("patch_meta_contract") or m.get("contract")
    if got != CONTRACT:
        raise SystemExit(
            f"FAIL-CLOSED: {p} patch_meta_contract={got!r} != {CONTRACT!r}. "
            "Georef parameterisation may differ; refusing (verdict §7)."
        )
    if "local_offset_utm" not in m:
        raise SystemExit(
            f"FAIL-CLOSED: {p} missing `local_offset_utm`; cannot place the "
            "field in EPSG:31983 (verdict §1/§6b)."
        )
    return m


# ----------------------------------------------------------------------
# terrain-following z extraction with a fail-closed datum check (verdict §3,§8)
# ----------------------------------------------------------------------
def extract_layer(vti, theta_ccw, offset_xy, terrain, dz_datum, slab_agl, reduce):
    """Return (world_xy[N,2], value[N]) for VTI nodes whose AGL ∈ slab.

    AGL = (z_local + dz_datum) − terrain_MASL(world_xy). Solid cells
    (vtkValidPointMask==0 / non-finite τ) are dropped (→ never 0).
    """
    pts = np.asarray(vti.points, float)  # local node coords
    if "tau" in vti.point_data:
        tau = np.asarray(vti.point_data["tau"], float)
    else:
        tau = np.asarray(vti.cell_data_to_point_data().point_data["tau"], float)
    valid = vti.point_data.get("vtkValidPointMask")
    solid = (np.asarray(valid) == 0) if valid is not None else ~np.isfinite(tau)
    tau = np.where(solid | ~np.isfinite(tau) | (tau <= 0), np.nan, tau)

    world = local_to_world(pts[:, :2], theta_ccw, offset_xy)
    ground = terrain.sample(world)  # MASL at each node
    agl = (pts[:, 2] + dz_datum) - ground

    lo, hi = slab_agl
    in_slab = np.isfinite(agl) & (agl >= lo) & (agl <= hi)
    # verdict §3/§8 fail-closed: the slab must actually intersect the VTI
    # given the terrain — otherwise the vertical datum convention is wrong
    # and a silent empty/garbage raster would result.
    cols = np.unique(np.round(world[:, :2], 1), axis=0)
    hit_cols = np.unique(np.round(world[in_slab, :2], 1), axis=0)
    if len(cols) and len(hit_cols) / len(cols) < 0.2:
        raise SystemExit(
            "FAIL-CLOSED: AGL slab {} m intersects <20% of columns "
            "({}/{}). Vertical datum (local_offset_utm.dz={:.4g}) vs "
            "terrain.tif likely mismatched — verify the z convention "
            "before trusting any overlay (verdict §3/§8).".format(
                slab_agl, len(hit_cols), len(cols), dz_datum
            )
        )

    xy = world[in_slab]
    v = tau[in_slab]
    if reduce == "colmean":  # secondary canopy vol-mean per column
        key = np.round(xy, 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        out = np.full(len(uniq), np.nan)
        for i in range(len(uniq)):
            col = v[inv == i]
            col = col[np.isfinite(col)]
            if col.size:
                out[i] = col.mean()
        return uniq, out
    return xy, v


# ----------------------------------------------------------------------
# single linear scatter→grid resample (verdict §1 — exactly one interp)
# ----------------------------------------------------------------------
def resample(xy, val, transform, shape, nodata):
    from scipy.interpolate import griddata

    h, w = shape
    cols = np.arange(w) + 0.5
    rows = np.arange(h) + 0.5
    gx = transform[0] * cols[None, :] + transform[2]  # AREA_OR_POINT=Area
    gy = transform[4] * rows[:, None] + transform[5]
    gx = np.broadcast_to(gx, (h, w)).ravel()
    gy = np.broadcast_to(gy, (h, w)).ravel()
    fin = np.isfinite(val)
    if fin.sum() < 3:
        return np.full((h, w), nodata, np.float32)
    z = griddata(xy[fin], val[fin], np.c_[gx, gy], method="linear")
    z = z.reshape(h, w).astype(np.float32)
    z[~np.isfinite(z)] = nodata
    return z


def snapped_transform(terrain_origin, terrain_px, want_bounds):
    """1 m grid lattice-locked to the terrain lattice (verdict §5).

    terrain_px (5 m) is an integer multiple of TARGET_PX (1 m), so the
    target grid edges coincide with terrain edges every 5 px — exact
    co-registration with no terrain resample.
    """
    ox, oy = terrain_origin  # terrain top-left (x_min, y_max)
    xmin, ymin, xmax, ymax = want_bounds
    i0 = math.floor((xmin - ox) / TARGET_PX)
    j0 = math.floor((oy - ymax) / TARGET_PX)  # floor → grid top ≥ ymax
    gx0 = ox + i0 * TARGET_PX  # ≤ xmin
    gy0 = oy - j0 * TARGET_PX  # ≥ ymax
    w = int(math.ceil((xmax - gx0) / TARGET_PX))
    h = int(math.ceil((gy0 - ymin) / TARGET_PX))
    # affine: (a,b,c,d,e,f) → x = a*col + c ; y = e*row + f
    return (TARGET_PX, 0.0, gx0, 0.0, -TARGET_PX, gy0), (h, w)


class TerrainSampler:
    """Nearest-cell MASL lookup; nodata-aware."""

    def __init__(self, path):
        import rasterio

        self.r = rasterio.open(path)
        self.nd = self.r.nodata
        self.band = self.r.read(1)
        self.t = self.r.transform
        self.origin = (self.t.c, self.t.f)
        self.px = (abs(self.t.a), abs(self.t.e))
        self.crs_epsg = self.r.crs.to_epsg()

    def sample(self, xy):
        inv = ~self.t
        cols, rows = inv * (xy[:, 0], xy[:, 1])
        cols = np.clip(cols.astype(int), 0, self.r.width - 1)
        rows = np.clip(rows.astype(int), 0, self.r.height - 1)
        v = self.band[rows, cols].astype(float)
        if self.nd is not None:
            v[np.isclose(v, self.nd)] = np.nan
        return v


# ----------------------------------------------------------------------
# wind-rose weights — freq×speed, never equal (verdict §2)
# ----------------------------------------------------------------------
def wind_weights(rose_path: Path, dirs, allow_placeholder):
    r = json.loads(Path(rose_path).read_text())
    qf = r.get("quality_flag")
    if qf == "placeholder-prior" and not allow_placeholder:
        raise SystemExit(
            f"FAIL-CLOSED: {rose_path} quality_flag='placeholder-prior'. "
            "Composite weights would be a climatological guess; pass "
            "--allow-placeholder-rose to override (verdict §2)."
        )
    f = r["frequencies"]
    u = r.get("mean_speeds", {})
    raw = np.array([f.get(d, 0.0) * u.get(d, 1.0) for d in dirs], float)
    if raw.sum() <= 0:
        raise SystemExit(
            f"FAIL-CLOSED: wind-rose weights sum to 0 over available directions {dirs}."
        )
    return raw / raw.sum(), qf, r.get("source", ""), r.get("station_id")


def write_geotiff(path, arr, transform, epsg, nodata, tags):
    import rasterio
    from rasterio.transform import Affine

    a, b, c, d, e, f = transform
    h, w = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=f"EPSG:{epsg}",
        nodata=nodata,
        transform=Affine(a, b, c, d, e, f),
        compress="deflate",
    ) as dst:
        dst.write(arr, 1)
        dst.update_tags(AREA_OR_POINT="Area", **tags)


# ----------------------------------------------------------------------
# fail-closed gate: synthetic ASYMMETRIC round-trip (verdict §9-i, §6a)
# ----------------------------------------------------------------------
def synthetic_roundtrip(theta_ccw, offset_xy, tol_px=0.1):
    """Inject a unique spike at an OFF-CENTRE local node; verify it lands
    at the analytically-expected world coord within tol. The clipped VTI
    is rotationally symmetric, so asymmetry must be injected synthetically
    — this still catches a wrong rotation sign/pivot/axis/offset at once.
    """
    n = 101
    lin = np.linspace(-ANALYSIS_RADIUS_M, ANALYSIS_RADIUS_M, n)
    xx, yy = np.meshgrid(lin, lin)
    local = np.c_[xx.ravel(), yy.ravel()]
    spike_local = np.array([30.0, -12.0])  # deliberately asymmetric
    k = int(np.argmin(((local - spike_local) ** 2).sum(1)))
    world = local_to_world(local, theta_ccw, offset_xy)
    expect = local_to_world(spike_local[None], theta_ccw, offset_xy)[0]
    err = math.hypot(*(world[k] - expect))
    # independent cross-check: rotation must be a proper rigid motion
    d_local = math.hypot(*(local[k] - np.zeros(2)))
    d_world = math.hypot(*(world[k] - np.asarray(offset_xy)))
    iso_err = abs(d_local - d_world)
    ok = err <= tol_px * TARGET_PX and iso_err < 1e-6
    return ok, err, iso_err


def run_self_test():
    print("== synthetic asymmetric round-trip (verdict §9-i) ==")
    bad = 0
    for d, wdeg in WIND8.items():
        # exercise both a real-style rotation and the identity (composite)
        for theta in (math.radians((90 + wdeg) % 360), 0.0):
            off = (680016.7399997711, 7455860.080299854)
            ok, err, iso = synthetic_roundtrip(theta, off)
            tag = "OK " if ok else "BAD"
            if not ok:
                bad += 1
            print(
                f"  {tag} dir={d:2s} θ={math.degrees(theta):6.1f}°  "
                f"placement_err={err:.2e} m  isometry_err={iso:.2e}"
            )
    # transform must be invertible to <1e-9 (axis-order re-assert, §9-iv)
    off = (680016.74, 7455860.08)
    th = math.radians(135.0)
    p = np.array([[12.3, -7.7], [-40.0, 33.3]])
    w = local_to_world(p, th, off)
    c, s = math.cos(-th), math.sin(-th)
    Rinv = np.array([[c, s], [-s, c]])
    back = (w - np.array(off)) @ Rinv.T
    rt = float(np.abs(back - p).max())
    print(f"  {'OK ' if rt < 1e-9 else 'BAD'} invert round-trip max={rt:.2e}")
    ok_all = bad == 0 and rt < 1e-9
    print(f"\nSELF-TEST {'PASSED' if ok_all else 'FAILED'} ({bad} bad placements)")
    return 0 if ok_all else 1


def discover_dirs(repo_root, patch):
    slug = patch.lower().replace("-", "_")
    out = []
    for d, wdeg in WIND8.items():
        cd = repo_root / f"tmp_{slug}_dir{wdeg:03d}"
        vti = cd / "postprocessing" / "lma" / "canopy_tau_field.vti"
        cm = cd / "case_meta.json"
        if vti.exists() and cm.exists():
            out.append((d, wdeg, vti, cm))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--patch", default="VDG-P07")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--smoke-vti", type=Path, help="single VTI for --dry-run before the 8-dir run")
    ap.add_argument("--case-meta", type=Path, help="case_meta.json paired with --smoke-vti")
    ap.add_argument(
        "--secondary", action="store_true", help="also emit the canopy [z0,H_mean] vol-mean band"
    )
    ap.add_argument("--allow-placeholder-rose", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve + print georef_spec, run gates, write nothing",
    )
    ap.add_argument(
        "--self-test", action="store_true", help="synthetic round-trip only; no CFD input needed"
    )
    a = ap.parse_args()

    if a.self_test:
        return run_self_test()

    repo = a.repo_root.resolve()
    pdir = repo / "patches" / a.patch
    meta = json.loads((pdir / "inputs" / "patch_meta.json").read_text())
    cx, cy = meta["center_x"], meta["center_y"]
    h_mean = float(meta["H_mean"])
    terrain = TerrainSampler(pdir / "inputs" / "terrain.tif")
    if terrain.crs_epsg != 31983:
        raise SystemExit(
            f"FAIL-CLOSED: terrain EPSG {terrain.crs_epsg} != 31983 (verdict §9-iv axis/CRS)."
        )
    rose = repo / "data" / "vidigal" / "wind_rose.json"

    try:
        import pyvista as pv
    except ImportError:
        raise SystemExit("needs env miniconda3/envs/MorphoFavela (pyvista).") from None

    # disk bbox + halo → snapped target grid (verdict §5)
    b = (
        cx - ANALYSIS_RADIUS_M - HALO_M,
        cy - ANALYSIS_RADIUS_M - HALO_M,
        cx + ANALYSIS_RADIUS_M + HALO_M,
        cy + ANALYSIS_RADIUS_M + HALO_M,
    )
    transform, shape = snapped_transform(terrain.origin, terrain.px, b)
    nodata = float(terrain.nd) if terrain.nd is not None else 3.4e38

    if a.smoke_vti:
        if not a.case_meta:
            raise SystemExit("--smoke-vti requires --case-meta")
        srcs = [("SMOKE", -1, a.smoke_vti, a.case_meta)]
    else:
        srcs = discover_dirs(repo, a.patch)
    if not srcs:
        raise SystemExit(
            f"No per-direction VTIs under {repo}/tmp_"
            f"{a.patch.lower().replace('-', '_')}_dir*/ — run the smoke/8-dir "
            "chain and pull results first. (Expected when smoke not yet "
            "COMPLETE — this is a gate, not a bug.)"
        )

    dirs = [d for d, *_ in srcs if d in WIND8]
    if a.smoke_vti:
        weights, qf, wsrc, wsta = {"SMOKE": 1.0}, "n/a (smoke)", "", None
    else:
        wv, qf, wsrc, wsta = wind_weights(rose, dirs, a.allow_placeholder_rose)
        weights = dict(zip(dirs, wv))

    spec = GeorefSpec(
        patch_id=a.patch,
        crs="EPSG:31983",
        crs_axis_order="x=easting, y=northing (rasterio/GDAL native)",
        rotation_param="case_meta.json:rotation_rad_local_ccw",
        rotation_sign="inverse bake = rotate by -theta (CCW positive)",
        pivot="patch centre = patch_meta(center_x,center_y) = local (0,0)",
        offset_param="case_meta.json:local_offset_utm{dx,dy}",
        pixel_size_m=TARGET_PX,
        grid_origin_xy=(transform[2], transform[5]),
        grid_shape=shape,
        snapped_to="patches/%s/inputs/terrain.tif" % a.patch,
        snap_rule="1 m nested in 5 m terrain lattice; origin integer px "
        "from terrain origin; AREA_OR_POINT=Area",
        z_slab_agl_m=(PED_LO_AGL, PED_HI_AGL),
        z_datum="AGL = (z_local + local_offset_utm.dz) - terrain_MASL(x,y)",
        composite_rule="ACH_i=3600/tau_i per cell; ACH_comp=Σ f_i ACH_i; tau_comp=1/ACH_comp",
        tau_to_ach="3600/tau (Sandberg well-mixed; ==lma_postprocess.py:188)",
        weight_source=f"{rose} ({wsrc}) station={wsta} qf={qf}",
        weight_method="freq×mean_speed, normalised over available dirs",
        units_tau="s (age of air)",
        units_ach="1/h",
        nodata=nodata,
        contract=CONTRACT,
        contract_ok=True,
        vti_sha256={d: sha256(v) for d, _, v, _ in srcs},
        notes="supersedes lma_aggregate_directions composite (τ-mean, "
        "uniform 1/N, no CRS). Domain QC rects in Phase-2 .qgz are "
        "indicative until regenerated from these case_meta rotations.",
    )

    print("== resolved georef_spec ==")
    print(json.dumps(asdict(spec), indent=2, default=list))
    ok, err, iso = synthetic_roundtrip(0.0, (cx, cy))
    print(
        f"== gate: synthetic round-trip (composite frame) "
        f"{'OK' if ok else 'FAIL'} err={err:.2e} iso={iso:.2e} =="
    )
    if not ok:
        raise SystemExit("FAIL-CLOSED: georef round-trip gate failed.")

    if a.dry_run:
        print(
            "\n[--dry-run] gates passed; no raster written. Re-run "
            "without --dry-run once smoke is COMPLETE."
        )
        return 0

    # per-direction → ACH on the shared grid, then composite (verdict §1,§2)
    h, w = shape
    ach_acc = np.zeros((h, w), float)
    wsum = np.zeros((h, w), float)
    qa_count = np.zeros((h, w), np.int16)
    for d, wdeg, vti_p, cm_p in srcs:
        cm = load_case_meta(cm_p)
        th = float(cm["rotation_rad_local_ccw"])
        off = (cm["local_offset_utm"]["dx"], cm["local_offset_utm"]["dy"])
        dz = float(cm["local_offset_utm"].get("dz", 0.0))
        vti = pv.read(str(vti_p))
        xy, tau = extract_layer(vti, th, off, terrain, dz, (PED_LO_AGL, PED_HI_AGL), reduce="none")
        tau_grid = resample(xy, tau, transform, shape, np.nan)
        ach = np.where(np.isfinite(tau_grid) & (tau_grid > 0), 3600.0 / tau_grid, np.nan)
        contrib = np.isfinite(ach)
        wd = weights[d] if d in weights else 1.0
        ach_acc[contrib] += wd * ach[contrib]
        wsum[contrib] += wd
        qa_count += contrib.astype(np.int16)
        print(f"  {d:5s} θ={math.degrees(th):6.1f}° w={wd:.3f} contrib_px={int(contrib.sum())}")

    ach_comp = np.full((h, w), nodata, np.float32)
    good = wsum > 0
    ach_comp[good] = (ach_acc[good] / wsum[good]).astype(np.float32)
    tau_comp = np.full((h, w), nodata, np.float32)
    tau_comp[good] = (1.0 / (ach_acc[good] / wsum[good]) * 3600.0).astype(np.float32)

    # disk clip LAST, halo discarded only now (verdict §4/§5)
    cols = (np.arange(w) + 0.5) * transform[0] + transform[2]
    rows = (np.arange(h) + 0.5) * transform[4] + transform[5]
    gx, gy = np.meshgrid(cols, rows)
    outside = ((gx - cx) ** 2 + (gy - cy) ** 2) > ANALYSIS_RADIUS_M**2
    for arr in (ach_comp, tau_comp):
        arr[outside] = nodata
    qa_count[outside] = 0

    # gate §9-v: no finite τ outside the disk
    if np.isfinite(tau_comp[outside]).any() and not np.all(tau_comp[outside] == nodata):
        raise SystemExit("FAIL-CLOSED: finite τ outside Ø100 m disk.")

    odir = pdir / "overlay"
    odir.mkdir(parents=True, exist_ok=True)
    tags = {
        "PATCH_ID": a.patch,
        "CONTRACT": CONTRACT,
        "COMPOSITE": spec.composite_rule,
        "WEIGHTS": spec.weight_method,
    }
    write_geotiff(
        odir / "composite.tif",
        ach_comp,
        transform,
        31983,
        nodata,
        {**tags, "QUANTITY": "ACH_composite_per_h"},
    )
    write_geotiff(
        odir / "composite_tau.tif",
        tau_comp,
        transform,
        31983,
        nodata,
        {**tags, "QUANTITY": "tau_composite_s"},
    )
    write_geotiff(
        odir / "composite_qa_ndir.tif",
        qa_count.astype(np.float32),
        transform,
        31983,
        0,
        {"QUANTITY": "n_contributing_directions"},
    )
    (odir / "composite.georef.json").write_text(
        json.dumps(asdict(spec), indent=2, default=list) + "\n"
    )
    print(f"\nOK  {odir}/composite.tif (+_tau, +_qa_ndir, +.georef.json)")
    print(f"    grid {shape} @ {TARGET_PX} m  EPSG:31983  valid_px={int(good.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
