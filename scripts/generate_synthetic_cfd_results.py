#!/usr/bin/env python3
"""Generate synthetic CFD results for one site.

Produces a complete `data/{site}/cfd_results/` tree matching the contract
in `src/cfd_integration/README.md` so the result-side analysis pipeline
can be exercised end-to-end before real OpenFOAM results return from
~/Airflow.

The synthetic field is deliberately simple: per-patch mean U_mag is
modulated by the patch's SVF and λp (lower SVF + higher λp → lower
ventilation), with per-direction noise. This is enough to make the
predictor regression produce non-degenerate coefficients without
committing to any physics we haven't validated.

Two sampling modes:

* **Legacy circular** (default, `grid_spacing=None`): a random point
  cloud in a 250 m circle. Kept for the analyze-pipeline tests that
  pass `n_samples_per_direction`.
* **Rectangular v1** (`--grid-spacing`, opt-in): a regular grid over
  the per-patch, per-direction `rectangular_domain_v1` footprint
  (`domain_{upstream,downstream,lateral}_m` from `campaign_patches.csv`),
  rotated to the wind bearing, with an in-patch deficit + downstream
  wake. This is the dataviz-grade mode.

Every `summary.json` carries `"synthetic": true` plus a `provenance`
block so these results can never be mistaken for a real OpenFOAM run.
Write them to a separate root (e.g. `--out
data/{site}/cfd_results_synthetic`), not the ingestor-default
`cfd_results/`.

Usage:
    python scripts/generate_synthetic_cfd_results.py --site vidigal
    python scripts/generate_synthetic_cfd_results.py --site vidigal --layout parquet
    python scripts/generate_synthetic_cfd_results.py --site vidigal --n-patches 5
    # rectangular dataviz-grade, full site coarse + one patch fine:
    python scripts/generate_synthetic_cfd_results.py --site vidigal \\
        --grid-spacing 4 --out data/vidigal/cfd_results_synthetic
    python scripts/generate_synthetic_cfd_results.py --site vidigal \\
        --patch-id VDG-P02 --grid-spacing 2 --out data/vidigal/cfd_results_synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cfd_integration.schema import WIND_DIRECTIONS_8  # noqa: E402

logger = logging.getLogger(__name__)

_DIRECTION_DEGREES = {
    "N": 0,
    "NE": 45,
    "E": 90,
    "SE": 135,
    "S": 180,
    "SW": 225,
    "W": 270,
    "NW": 315,
}
_DEGREE_DIRNAME = {v: f"wind_{v:03d}" for v in _DIRECTION_DEGREES.values()}


def _patch_mean_u_mag(svf: float, lambda_p: float, u_ref: float) -> float:
    """Plausible mean U_mag inside the analysis patch.

    Modulated by patch morphology: lower SVF (deeper canyons) and higher
    λp (more obstructions) reduce in-canopy wind speed. Calibrated to
    produce values in the 0.5–4 m/s range for typical favela morphology.
    """
    blockage = 0.4 * (1.0 - float(svf)) + 0.6 * float(lambda_p)
    blockage = float(np.clip(blockage, 0.0, 1.0))
    return float(u_ref * (0.85 - 0.65 * blockage))


def _generate_samples(
    cx: float,
    cy: float,
    domain_radius: float,
    u_mag_mean: float,
    direction: str,
    rng: np.random.Generator,
    n_target: int = 5000,
) -> pd.DataFrame:
    """Generate sample_points for one patch × one direction.

    Points are uniformly scattered in the 250m-radius CFD domain. U_mag
    has a soft radial gradient (lower in centre, mimicking blockage) plus
    Gaussian noise. U/V/W components are projected from the wind-direction
    unit vector.
    """
    theta = rng.uniform(0, 2 * np.pi, n_target)
    r = domain_radius * np.sqrt(rng.uniform(0, 1, n_target))
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z = np.full(n_target, 1.5)

    radial_factor = 0.6 + 0.4 * (r / domain_radius)
    u_mag_target = u_mag_mean * radial_factor + rng.normal(0, 0.15, n_target)
    u_mag_target = np.clip(u_mag_target, 0.05, None)

    bearing_rad = np.deg2rad(_DIRECTION_DEGREES[direction])
    flow_x = -np.sin(bearing_rad)
    flow_y = -np.cos(bearing_rad)
    u_comp = u_mag_target * flow_x + rng.normal(0, 0.05, n_target)
    v_comp = u_mag_target * flow_y + rng.normal(0, 0.05, n_target)
    w_comp = rng.normal(0, 0.02, n_target)
    # U_mag derived from the perturbed components so it self-validates against
    # the validator's `|U_mag - sqrt(U^2+V^2+W^2)| < tol` check.
    u_mag = np.sqrt(u_comp**2 + v_comp**2 + w_comp**2)
    tke = 0.5 * (u_mag * 0.15) ** 2 + rng.uniform(0, 0.01, n_target)

    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "z": z,
            "U": u_comp,
            "V": v_comp,
            "W": w_comp,
            "U_mag": u_mag,
            "TKE": tke,
        }
    )


def _generate_samples_rect(
    cx: float,
    cy: float,
    dom_up: float,
    dom_down: float,
    dom_lat: float,
    u_mag_mean: float,
    direction: str,
    rng: np.random.Generator,
    grid_spacing: float,
    patch_radius: float = 50.0,
) -> pd.DataFrame:
    """Regular grid over the rectangular_domain_v1 footprint for one direction.

    The domain rectangle is built in a flow-aligned local frame and
    rotated into world coordinates: ``s`` runs along the flow from
    ``-dom_up`` (inlet/upwind) to ``+dom_down`` (outlet), ``t`` runs
    cross-flow over ``±dom_lat``. The flow vector follows the existing
    meteorological convention (wind *from* ``direction``), so the sign
    of ``(flow_x, flow_y)`` must match ``_generate_samples`` — getting
    it wrong mirrors the wake upstream.

    The field is plausible, not physical: free-stream away from the
    fabric, a velocity bowl inside the central 100 m analysis patch,
    and a wake that recovers downstream. Flagged synthetic everywhere.
    """
    bearing_rad = np.deg2rad(_DIRECTION_DEGREES[direction])
    flow_x = -np.sin(bearing_rad)
    flow_y = -np.cos(bearing_rad)
    perp_x = -flow_y  # 90° CCW from flow → cross-flow axis
    perp_y = flow_x

    s = np.arange(-dom_up, dom_down + grid_spacing, grid_spacing)
    t = np.arange(-dom_lat, dom_lat + grid_spacing, grid_spacing)
    S, T = np.meshgrid(s, t)
    S = S.ravel()
    T = T.ravel()
    n = S.size

    x = cx + S * flow_x + T * perp_x
    y = cy + S * flow_y + T * perp_y
    z = np.full(n, 1.5)

    d_center = np.hypot(S, T)
    in_patch = d_center < patch_radius
    near_fabric = d_center < patch_radius * 3.0
    wake = (S > 0) & (np.abs(T) < patch_radius * 2.0)

    # Bowl: 0.5·mean at patch centre → 1.0·mean at the patch edge.
    bowl = 0.5 + 0.5 * np.clip(d_center / patch_radius, 0.0, 1.0)
    # Wake deficit strongest just behind the patch, recovering over ~220 m.
    wake_factor = 1.0 - 0.45 * np.exp(-S / 220.0)

    u_field = np.full(n, u_mag_mean * 1.7)  # toward (sub-u_ref) free-stream
    u_field = np.where(near_fabric, u_mag_mean, u_field)
    u_field = np.where(in_patch, u_mag_mean * bowl, u_field)
    u_field = np.where(wake, np.minimum(u_field, u_mag_mean) * wake_factor, u_field)
    u_field = np.clip(u_field + rng.normal(0, 0.12, n), 0.05, None)

    u_comp = u_field * flow_x + rng.normal(0, 0.05, n)
    v_comp = u_field * flow_y + rng.normal(0, 0.05, n)
    w_comp = rng.normal(0, 0.02, n)
    u_mag = np.sqrt(u_comp**2 + v_comp**2 + w_comp**2)
    tke = 0.5 * (u_mag * 0.15) ** 2 + np.where(wake, 0.05, 0.0) + rng.uniform(0, 0.01, n)

    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "z": z,
            "U": u_comp,
            "V": v_comp,
            "W": w_comp,
            "U_mag": u_mag,
            "TKE": tke,
        }
    )


def _git_short_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _provenance(seed: int, grid_spacing: float | None, u_ref: float) -> dict:
    """Provenance block embedded in every summary.json so synthetic
    results are self-identifying and reproducible."""
    return {
        "synthetic": True,
        "generator": "scripts/generate_synthetic_cfd_results.py",
        "git_commit": _git_short_commit(),
        "seed": seed,
        "model": "blockage_v1" if grid_spacing is None else "blockage_v1+rect_wake",
        "sampling": "circular_250m" if grid_spacing is None else "rectangular_domain_v1",
        "grid_spacing_m": grid_spacing,
        "u_ref_ms": u_ref,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": "SYNTHETIC — not a CFD solve. Do not mix with real cfd_results/.",
    }


def _write_direction(
    out_dir: Path,
    direction: str,
    samples: pd.DataFrame,
    metadata: dict,
    layout: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if layout == "csv":
        samples.to_csv(out_dir / "sample_points.csv", index=False)
    elif layout == "parquet":
        samples.to_parquet(out_dir / "sample_points.parquet", index=False)
    else:
        raise ValueError(f"unknown layout: {layout}")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(metadata, f, indent=2)


def generate_site(
    site: str,
    out_root: Path,
    layout: str = "csv",
    n_patches: int | None = None,
    n_samples_per_direction: int = 15000,
    seed: int = 42,
    u_ref: float = 6.0,
    domain_radius: float = 250.0,
    directions: list[str] | None = None,
    grid_spacing: float | None = None,
    patch_id: str | None = None,
) -> dict:
    """Generate synthetic results for a site. Returns coverage summary.

    ``grid_spacing=None`` keeps the legacy circular random cloud
    (``n_samples_per_direction`` points). Set ``grid_spacing`` to use
    the rectangular_domain_v1 grid sampler instead. ``patch_id``
    restricts generation to a single patch (used for the fine-grid
    refinement pass).
    """
    if directions is None:
        directions = list(WIND_DIRECTIONS_8)

    patches_csv = (
        PROJECT_ROOT
        / "outputs"
        / site
        / "sampling_cfd"
        / "campaign_sampling"
        / "campaign_patches.csv"
    )
    if not patches_csv.exists():
        raise FileNotFoundError(f"campaign_patches.csv not found for {site}: {patches_csv}")
    patches_df = pd.read_csv(patches_csv)
    if n_patches is not None:
        patches_df = patches_df.head(n_patches)
    if patch_id is not None:
        patches_df = patches_df[patches_df["patch_id"] == patch_id]
        if patches_df.empty:
            raise ValueError(f"patch_id {patch_id!r} not found in {patches_csv}")

    rng = np.random.default_rng(seed)
    out_root.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(seed, grid_spacing, u_ref)

    coverage = {"site": site, "n_patches": len(patches_df), "patches": {}}
    for _, row in patches_df.iterrows():
        pid = row["patch_id"]
        cx = float(row["center_x"])
        cy = float(row["center_y"])
        u_mag_mean = _patch_mean_u_mag(row["svf"], row["lambda_p"], u_ref)
        patch_dir = out_root / pid
        coverage["patches"][pid] = {"u_mag_mean": u_mag_mean, "directions": []}

        for direction in directions:
            if grid_spacing is None:
                samples = _generate_samples(
                    cx,
                    cy,
                    domain_radius,
                    u_mag_mean,
                    direction,
                    rng,
                    n_target=n_samples_per_direction,
                )
            else:
                samples = _generate_samples_rect(
                    cx,
                    cy,
                    float(row["domain_upstream_m"]),
                    float(row["domain_downstream_m"]),
                    float(row["domain_lateral_m"]),
                    u_mag_mean,
                    direction,
                    rng,
                    grid_spacing,
                )
            metadata = {
                "patch_id": pid,
                "site": site,
                "wind_direction": direction,
                "wind_speed_ref": u_ref,
                "converged": True,
                "residual_final": 1.2e-5,
                "solver": "simpleFoam",
                "turbulence_model": "kOmegaSST",
                "n_iterations": 2000,
                "wall_clock_s": 3600.0,
                "synthetic": True,
                "provenance": provenance,
            }
            if layout == "parquet":
                dir_dir = patch_dir / _DEGREE_DIRNAME[_DIRECTION_DEGREES[direction]]
            else:
                dir_dir = patch_dir / direction
            _write_direction(dir_dir, direction, samples, metadata, layout)
            coverage["patches"][pid]["directions"].append(direction)
            last_n = len(samples)

        logger.info(
            "Generated %s: %d directions × %d samples (mean U_mag=%.2f m/s)",
            pid,
            len(directions),
            last_n,
            u_mag_mean,
        )

    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic CFD results for one site.")
    parser.add_argument("--site", required=True, help="Site name (e.g., 'vidigal').")
    parser.add_argument(
        "--layout",
        choices=("csv", "parquet"),
        default="csv",
        help="On-disk format. csv = IVF-native cardinal dirs, "
        "parquet = Airflow-native wind_NNN dirs. Default csv.",
    )
    parser.add_argument(
        "--n-patches",
        type=int,
        default=None,
        help="Limit to first N patches (default: all in campaign_patches.csv).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=15000,
        help="Legacy-only: points/direction for the circular random "
        "cloud (Python API with grid_spacing=None). Unused by the CLI, "
        "which always uses the rectangular grid.",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=2.0,
        help="Rectangular_domain_v1 grid spacing in metres (default 2.0, "
        "the contract pedestrian-grid resolution). Larger = fewer rows.",
    )
    parser.add_argument(
        "--patch-id",
        default=None,
        help="Restrict to one patch (e.g. VDG-P02), for the fine-grid "
        "refinement pass over a coarse full-site run.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--u-ref", type=float, default=6.0, help="Reference inflow wind speed (m/s)."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output root (default: data/{site}/cfd_results/).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_root = (
        Path(args.out)
        if args.out is not None
        else PROJECT_ROOT / "data" / args.site / "cfd_results"
    )

    coverage = generate_site(
        site=args.site,
        out_root=out_root,
        layout=args.layout,
        n_patches=args.n_patches,
        n_samples_per_direction=args.n_samples,
        seed=args.seed,
        u_ref=args.u_ref,
        grid_spacing=args.grid_spacing,
        patch_id=args.patch_id,
    )
    print(
        f"Generated {coverage['n_patches']} patches × "
        f"{len(coverage['patches'][next(iter(coverage['patches']))]['directions'])} directions "
        f"in {out_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
