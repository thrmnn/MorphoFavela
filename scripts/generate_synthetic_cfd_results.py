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

Usage:
    python scripts/generate_synthetic_cfd_results.py --site vidigal
    python scripts/generate_synthetic_cfd_results.py --site vidigal --layout parquet
    python scripts/generate_synthetic_cfd_results.py --site vidigal --n-patches 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
) -> dict:
    """Generate synthetic results for a site. Returns coverage summary."""
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

    rng = np.random.default_rng(seed)
    out_root.mkdir(parents=True, exist_ok=True)

    coverage = {"site": site, "n_patches": len(patches_df), "patches": {}}
    for _, row in patches_df.iterrows():
        patch_id = row["patch_id"]
        cx = float(row["center_x"])
        cy = float(row["center_y"])
        u_mag_mean = _patch_mean_u_mag(row["svf"], row["lambda_p"], u_ref)
        patch_dir = out_root / patch_id
        coverage["patches"][patch_id] = {"u_mag_mean": u_mag_mean, "directions": []}

        for direction in directions:
            samples = _generate_samples(
                cx,
                cy,
                domain_radius,
                u_mag_mean,
                direction,
                rng,
                n_target=n_samples_per_direction,
            )
            metadata = {
                "patch_id": patch_id,
                "site": site,
                "wind_direction": direction,
                "wind_speed_ref": u_ref,
                "converged": True,
                "residual_final": 1.2e-5,
                "solver": "simpleFoam",
                "turbulence_model": "kOmegaSST",
                "n_iterations": 2000,
                "wall_clock_s": 3600.0,
            }
            if layout == "parquet":
                dir_dir = patch_dir / _DEGREE_DIRNAME[_DIRECTION_DEGREES[direction]]
            else:
                dir_dir = patch_dir / direction
            _write_direction(dir_dir, direction, samples, metadata, layout)
            coverage["patches"][patch_id]["directions"].append(direction)

        logger.info(
            "Generated %s: %d directions × %d samples (mean U_mag=%.2f m/s)",
            patch_id,
            len(directions),
            n_samples_per_direction,
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
        help="Sample points per direction (default 15000 — matches the "
        "validator's expected ~15k/patch for a 250m domain at 2m grid).",
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
    )
    print(
        f"Generated {coverage['n_patches']} patches × "
        f"{len(coverage['patches'][next(iter(coverage['patches']))]['directions'])} directions "
        f"in {out_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
