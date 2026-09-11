"""WP-02: turn an EPW year into a cumulative sky on the Tregenza hemisphere.

The existing engine (src/svf_v2) answers "which sky patches can this cell see".
This module answers "how much energy does each patch deliver over a year", so
their product is annual irradiation. Keeping them separate means the expensive
geometry is computed once and can be re-weighted for any sky.

Two things worth stating plainly:

1. **The discretization is normalized out.** Summing the 145 patch weights
   against their centre-point cosines gives 3.158803, not pi (3.141593) — the
   scheme over-counts the cosine projection by 0.548%. Diffuse energy is
   therefore distributed using that measured sum, not pi, so an unobstructed
   cell reproduces the EPW's own annual diffuse exactly instead of inheriting a
   half-percent bias.

2. **Direct beam is binned to the nearest patch.** That is an approximation:
   the sun is a ~0.5-degree disc and a Tregenza patch spans ~11 degrees, so a
   cell near a shadow edge is resolved no finer than the patch. It is
   appropriate for annual totals and for the citywide percentile, and it is NOT
   appropriate for a sun-hours-on-a-given-day claim — compute those from sun
   positions directly (src/solar), never from this binning.

Run: python3 -m src.brisa_solar.wp02_sky <epw_path>
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pvlib

from .constants import P1_SKY_PATCHES, REPO_ROOT


@dataclass
class CumulativeSky:
    """Annual irradiation delivered by each sky patch, on a horizontal plane."""
    patch_diffuse_kwh: np.ndarray      # (145,) kWh/m2, sums to annual DHI
    patch_direct_kwh: np.ndarray       # (145,) kWh/m2, sums to annual beam-horizontal
    directions: np.ndarray             # (145,3) unit vectors, z up
    weights: np.ndarray                # (145,) solid angle, sr
    annual_ghi_kwh: float              # EPW's own annual global horizontal
    cosine_sum: float                  # measured sum(w*cos); pi in the continuum
    epw: str

    @property
    def patch_total_kwh(self) -> np.ndarray:
        return self.patch_diffuse_kwh + self.patch_direct_kwh

    def irradiation(self, visibility: np.ndarray) -> np.ndarray:
        """Annual irradiation for cells given per-patch visibility (n_cells, 145)."""
        return np.asarray(visibility, float) @ self.patch_total_kwh

    def svf(self, visibility: np.ndarray) -> np.ndarray:
        """Cosine-weighted sky view factor from the same visibility array."""
        cw = self.weights * self.directions[:, 2]
        return (np.asarray(visibility, float) @ cw) / cw.sum()


def build(epw_path: str | Path) -> CumulativeSky:
    from src.svf_v2.compute import generate_tregenza_patches

    directions, weights = generate_tregenza_patches()
    assert len(directions) == P1_SKY_PATCHES, "sky resolution must be the one locked constant"

    df, meta = pvlib.iotools.read_epw(str(epw_path), coerce_year=2021)
    solpos = pvlib.solarposition.get_solarposition(
        df.index, meta["latitude"], meta["longitude"], altitude=meta["altitude"]
    )
    zen = np.radians(solpos["apparent_zenith"].to_numpy())
    azi = np.radians(solpos["azimuth"].to_numpy())
    ghi = df["ghi"].to_numpy(float)
    dni = df["dni"].to_numpy(float)
    dhi = df["dhi"].to_numpy(float)

    cos_z = np.cos(zen)
    up = cos_z > 0.0                      # sun above the horizon

    # ── diffuse: isotropic over the hemisphere ────────────────────────────────
    # A patch's share of energy reaching a horizontal plane is w*cos / sum(w*cos).
    # Using the MEASURED sum removes the scheme's 0.548% cosine-projection bias.
    cos_patch = directions[:, 2]
    cw = weights * cos_patch
    cosine_sum = float(cw.sum())
    share = cw / cosine_sum
    annual_dhi = float(dhi.sum()) / 1000.0
    patch_diffuse = share * annual_dhi

    # ── direct: bin each hour's beam to the nearest patch ─────────────────────
    # Sun vector in the same frame as the patches: z up, and EPW azimuth is
    # measured clockwise from north, so x=east, y=north.
    sx = np.sin(zen) * np.sin(azi)
    sy = np.sin(zen) * np.cos(azi)
    sz = cos_z
    sun = np.column_stack([sx, sy, sz])[up]
    beam_horizontal = (dni * cos_z)[up] / 1000.0      # kWh/m2 on the horizontal

    nearest = np.argmax(sun @ directions.T, axis=1)
    patch_direct = np.zeros(len(directions))
    np.add.at(patch_direct, nearest, beam_horizontal)

    return CumulativeSky(
        patch_diffuse_kwh=patch_diffuse,
        patch_direct_kwh=patch_direct,
        directions=directions,
        weights=weights,
        annual_ghi_kwh=float(ghi.sum()) / 1000.0,
        cosine_sum=cosine_sum,
        epw=str(epw_path),
    )


def main() -> int:
    import sys
    params_epw = None
    if len(sys.argv) > 1:
        params_epw = sys.argv[1]
    else:
        from .constants import load_params
        params_epw = str(REPO_ROOT / load_params()["weather"]["primary_epw"])

    sky = build(params_epw)
    unobstructed = float(sky.patch_total_kwh.sum())
    err = (unobstructed - sky.annual_ghi_kwh) / sky.annual_ghi_kwh * 100.0
    report = {
        "epw": Path(sky.epw).name,
        "annual_ghi_kwh_m2_epw": round(sky.annual_ghi_kwh, 2),
        "unobstructed_from_cumulative_sky_kwh_m2": round(unobstructed, 2),
        "identity_error_pct": round(err, 4),
        "tolerance_pct": 2.0,
        "identity_verdict": "PASS" if abs(err) < 2.0 else "FAIL",
        "annual_diffuse_kwh_m2": round(float(sky.patch_diffuse_kwh.sum()), 2),
        "annual_direct_horizontal_kwh_m2": round(float(sky.patch_direct_kwh.sum()), 2),
        "cosine_sum_measured": round(sky.cosine_sum, 6),
        "cosine_sum_continuum_pi": round(float(np.pi), 6),
        "discretization_bias_pct": round((sky.cosine_sum - np.pi) / np.pi * 100, 4),
        "sky_patches": int(P1_SKY_PATCHES),
    }
    print(json.dumps(report, indent=1))
    return 0 if report["identity_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
