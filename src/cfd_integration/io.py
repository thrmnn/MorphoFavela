"""Load CFD results from disk.

Expected layout (see `src/cfd_integration/README.md`):

    data/{site}/cfd_results/{patch_id}/{wind_direction}/
        sample_points.csv     # primary data (REQUIRED)
        summary.json          # metadata (REQUIRED)
        field.vtu             # full 3D field (OPTIONAL)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.cfd_integration.schema import (
    WIND_DIRECTIONS_8,
    CFDCampaignResult,
    CFDPatchResult,
    PatchSimulationMetadata,
    WindRose,
)

logger = logging.getLogger(__name__)

# Default root for CFD results within this repo
CFD_RESULTS_ROOT = "data/{site}/cfd_results"


def load_patch_csv(
    csv_path: Path, metadata_path: Optional[Path] = None
) -> CFDPatchResult:
    """Load one patch × one wind-direction result from CSV + JSON.

    CSV schema (required columns): x, y, z, U, V, W, U_mag, TKE
    Optional column: p (pressure)

    JSON schema (required keys): patch_id, site, wind_direction, wind_speed_ref
    Optional keys: converged, residual_final, solver, turbulence_model,
                   n_iterations, wall_clock_s
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CFD CSV not found: {csv_path}")

    samples = pd.read_csv(csv_path)

    # Auto-compute U_mag if missing but components present
    if "U_mag" not in samples.columns and {"U", "V", "W"}.issubset(samples.columns):
        import numpy as np

        samples["U_mag"] = np.sqrt(
            samples["U"] ** 2 + samples["V"] ** 2 + samples["W"] ** 2
        )

    # Load metadata
    if metadata_path is None:
        metadata_path = csv_path.parent / "summary.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"CFD metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        meta_dict = json.load(f)

    metadata = PatchSimulationMetadata(
        patch_id=meta_dict["patch_id"],
        site=meta_dict["site"],
        wind_direction=meta_dict["wind_direction"],
        wind_speed_ref=meta_dict["wind_speed_ref"],
        converged=meta_dict.get("converged", True),
        residual_final=meta_dict.get("residual_final"),
        solver=meta_dict.get("solver", "simpleFoam"),
        turbulence_model=meta_dict.get("turbulence_model", "kOmegaSST"),
        n_iterations=meta_dict.get("n_iterations", 0),
        wall_clock_s=meta_dict.get("wall_clock_s", 0.0),
    )

    return CFDPatchResult(metadata=metadata, samples=samples)


def load_patch_vtu(vtu_path: Path) -> pd.DataFrame:
    """Load a full 3D field from a `.vtu` file.

    Returns a DataFrame of sample points at the given height (typically extracted
    via a horizontal slice at z = 1.5m in post-processing).

    Requires pyvista (optional dependency). Falls back to an error if missing.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for .vtu loading. "
            "Install with: pip install pyvista. "
            "Alternatively, have the OpenFOAM agent export sample_points.csv directly."
        ) from exc

    mesh = pv.read(str(vtu_path))
    points = mesh.points
    # Extract common OpenFOAM field names
    data = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}
    for field_name, col in [("U", ["U", "U:0"]), ("k", ["k", "TKE"])]:
        for candidate in col:
            if candidate in mesh.point_data:
                arr = mesh.point_data[candidate]
                if arr.ndim == 2:  # vector field
                    data["U"] = arr[:, 0]
                    data["V"] = arr[:, 1]
                    data["W"] = arr[:, 2]
                else:
                    data[field_name] = arr
                break

    return pd.DataFrame(data)


def load_campaign_results(
    site: str,
    results_root: Optional[Path] = None,
    require_all_directions: bool = False,
) -> CFDCampaignResult:
    """Load all CFD results for one site.

    Traverses `data/{site}/cfd_results/{patch_id}/{wind_dir}/` and loads every
    patch × direction combination that has `sample_points.csv` + `summary.json`.

    Parameters
    ----------
    site : str
        Site name (e.g., 'vidigal').
    results_root : Path, optional
        Override the default path `data/{site}/cfd_results/`.
    require_all_directions : bool
        If True, raise if any patch is missing any of the 8 directions.

    Returns
    -------
    CFDCampaignResult
    """
    if results_root is None:
        from src.config import PROJECT_ROOT

        results_root = PROJECT_ROOT / CFD_RESULTS_ROOT.format(site=site)
    results_root = Path(results_root)

    if not results_root.exists():
        raise FileNotFoundError(f"CFD results root not found: {results_root}")

    patches: dict[str, dict[str, CFDPatchResult]] = {}
    for patch_dir in sorted(results_root.iterdir()):
        if not patch_dir.is_dir():
            continue
        patch_id = patch_dir.name
        patches[patch_id] = {}

        for dir_dir in sorted(patch_dir.iterdir()):
            if not dir_dir.is_dir():
                continue
            wind_dir = dir_dir.name
            if wind_dir not in WIND_DIRECTIONS_8:
                logger.warning("Skipping unknown wind direction: %s", wind_dir)
                continue

            csv_path = dir_dir / "sample_points.csv"
            json_path = dir_dir / "summary.json"
            if not (csv_path.exists() and json_path.exists()):
                continue

            try:
                patches[patch_id][wind_dir] = load_patch_csv(csv_path, json_path)
            except Exception as e:
                logger.error("Failed to load %s/%s: %s", patch_id, wind_dir, e)

        if require_all_directions and len(patches[patch_id]) < 8:
            missing = set(WIND_DIRECTIONS_8) - set(patches[patch_id])
            raise ValueError(f"{patch_id}: missing directions {sorted(missing)}")

    # Wind rose (optional)
    wind_rose = None
    rose_path = results_root.parent / "wind_rose.json"
    if rose_path.exists():
        with open(rose_path) as f:
            rose_data = json.load(f)
        wind_rose = WindRose(
            site=site,
            frequencies=rose_data["frequencies"],
            mean_speeds=rose_data.get("mean_speeds", {}),
            source=rose_data.get("source", ""),
        )

    logger.info(
        "Loaded %d patches × directions = %d simulations for %s",
        len(patches),
        sum(len(v) for v in patches.values()),
        site,
    )
    return CFDCampaignResult(site=site, patches=patches, wind_rose=wind_rose)
