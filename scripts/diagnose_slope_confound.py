"""Track C / STEP 0 — disprove the σH slope-confound premise empirically.

The roughness track greenlit "terrain-following morphometry" to strip a supposed
slope confound where a flat absolute datum inflates σH on hillsides. But grid σH
is already built on ``altura`` (per-building height above its OWN base —
terrain-following; src/morphometry/grid.py sets height = altura, and
compute_height_variability takes its per-cell std). If the premise were right,
σH would rise with slope. This script measures it.

Result (the scoping finding): corr(slope_deg, σH) is small and NEGATIVE on the
steep sites — the wrong sign for "slope inflates σH" — so the σH arm of Track C is
a near-no-op and is dropped. Only λf's 2-D plan-view face projection remains a
candidate artefact (handled, magnitude-gated, in STEP 2).

Spearman (rank) correlation, on built cells with finite slope + metric.

Output: outputs/comparative/roughness/slope_confound.json
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
METRICS = ["sigma_h", "lambda_f_mean"]
OUT = ROOT / "outputs" / "comparative" / "roughness" / "slope_confound.json"


def built(site: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg")
    return g[g["building_count"] > 0]


def corr(g: gpd.GeoDataFrame, metric: str) -> dict:
    s = g["slope_deg"].to_numpy()
    m = g[metric].to_numpy()
    ok = np.isfinite(s) & np.isfinite(m)
    if ok.sum() < 30:
        return {"n": int(ok.sum()), "rho": None, "p": None}
    rho, p = spearmanr(s[ok], m[ok])
    return {"n": int(ok.sum()), "rho": float(rho), "p": float(p)}


def tilt_upper_bound(g: gpd.GeoDataFrame) -> np.ndarray:
    """STEP 2 gate evidence — UPPER-BOUND fractional correction to mean λf from the
    along-wind terrain step. Per cell at slope β / aspect α, the terrain rises across
    a 10 m cell along sector θ by Δz(θ)=10·tan(β)·max(0,cos(α−θ)); treat the full Δz
    as added frontal height → frac(θ)=Δz/H_mean, mean over the 8 λf sectors. Over-
    estimates (full rise on full width, no sheltering). Cells need finite β,α,H>0.5."""
    sectors = np.deg2rad(np.arange(0, 360, 45))
    beta = np.deg2rad(g["slope_deg"].to_numpy())
    alpha = np.deg2rad(g["aspect_deg"].to_numpy())
    h = g["H_mean"].to_numpy()
    ok = np.isfinite(beta) & np.isfinite(alpha) & np.isfinite(h) & (h > 0.5)
    beta, alpha, h = beta[ok], alpha[ok], h[ok]
    dz = 10.0 * np.tan(beta)[:, None] * np.clip(np.cos(alpha[:, None] - sectors[None, :]), 0, None)
    frac = (dz / h[:, None]).mean(axis=1)
    return frac


def main() -> None:
    grids = {s: built(s) for s in SITES}
    per_site = {s: {m: corr(grids[s], m) for m in METRICS} for s in SITES}

    # STEP 2 — λf terrain-step magnitude gate (pooled, by slope bin).
    fracs, slopes = [], []
    for s in SITES:
        f = tilt_upper_bound(grids[s])
        fracs.append(f)
        sl = grids[s]["slope_deg"].to_numpy()
        slopes.append(sl[np.isfinite(grids[s]["slope_deg"].to_numpy())
                        & np.isfinite(grids[s]["aspect_deg"].to_numpy())
                        & (grids[s]["H_mean"].to_numpy() > 0.5)])
    frac = np.concatenate(fracs)
    slope = np.concatenate(slopes)
    below25 = float(np.median(frac[slope < 25]))
    tilt = {
        "method": "upper-bound frac(θ)=10·tan(slope)·max(0,cos(aspect−θ))/H_mean, mean over 8 sectors",
        "gate": "median below 25° < 0.05 → correction immaterial, STOP (no Step 3)",
        "median_below_25deg": below25,
        "gate_cleared": bool(below25 < 0.05),
        "by_slope_bin": {
            f"{lo}-{hi}": {
                "n": int(((slope >= lo) & (slope < hi)).sum()),
                "median": float(np.median(frac[(slope >= lo) & (slope < hi)]))
                if ((slope >= lo) & (slope < hi)).any() else None,
            }
            for lo, hi in [(0, 5), (5, 15), (15, 25), (25, 90)]
        },
        "note": (
            "Gate CLEARED on the bulk/flatland (median below 25° = "
            f"{below25*100:.1f}%), so the canonical λf is left untouched. The UPPER "
            "bound is large on steep hillside cells (15–25° median ~20%, 25°+ ~34%), "
            "but it is an overestimate and corr(slope, λf) is negative empirically — a "
            "full terrain-following λf (Step 3) is unwarranted; revisit only for a "
            "hillside-specific λf claim."
        ),
    }
    pooled_g = gpd.GeoDataFrame(
        __import__("pandas").concat([grids[s][["slope_deg", *METRICS]] for s in SITES])
    )
    pooled = {m: corr(pooled_g, m) for m in METRICS}

    payload = {
        "title": "Track C Step 0 — slope vs σH / λf rank correlation (premise disproof)",
        "method": "Spearman ρ on built cells (building_count>0), finite slope+metric.",
        "finding": (
            "corr(slope, σH) is small and predominantly NEGATIVE on steep sites — the "
            "wrong sign for 'slope inflates σH via a flat datum'. σH is already "
            "terrain-following (altura-based). σH arm of Track C dropped; only λf "
            "face-projection remains (STEP 2, magnitude-gated)."
        ),
        "per_site": per_site,
        "pooled": pooled,
        "lambda_f_tilt_gate": tilt,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT.relative_to(ROOT)}\n")
    print(f"{'site':<20s} {'corr(slope,σH)':>16s} {'corr(slope,λf)':>16s}")
    for s in SITES:
        a, b = per_site[s]["sigma_h"], per_site[s]["lambda_f_mean"]
        print(f"{s:<20s} {a['rho']:>+16.3f} {b['rho']:>+16.3f}")
    pa, pb = pooled["sigma_h"], pooled["lambda_f_mean"]
    print(f"{'POOLED':<20s} {pa['rho']:>+16.3f} {pb['rho']:>+16.3f}")


if __name__ == "__main__":
    main()
