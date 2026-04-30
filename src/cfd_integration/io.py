"""Load CFD results from disk.

Two on-disk layouts are supported transparently:

1. **IVF native** — cardinal direction directory + CSV samples:

       data/{site}/cfd_results/{patch_id}/{N|NE|E|SE|S|SW|W|NW}/
           sample_points.csv     # primary data (REQUIRED)
           summary.json          # metadata (REQUIRED)
           field.vtu             # full 3D field (OPTIONAL)

2. **Airflow native** — numeric-degree directory + parquet samples:

       data/{site}/cfd_results/{patch_id}/wind_{NNN}/
           *.parquet             # primary data (REQUIRED, one or more files)
           summary.json          # metadata (REQUIRED)

   `wind_000` → `N`, `wind_045` → `NE`, …, `wind_315` → `NW`. Multiple
   parquet files in the same directory are concatenated row-wise (used
   when the OpenFOAM agent emits one parquet per processor decomposition).

`load_campaign_results` auto-detects which layout each patch / direction
uses and dispatches to the appropriate loader; users do not need to
choose.

Stable API
----------
The public surface enumerated in ``__all__`` below is the contract that
the result-side analysis pipeline (`scripts/analyze_cfd_results.py`,
`src/cfd_integration/aggregate.py`, `src/cfd_integration/weighting.py`)
and the `cfd-results-ingestor` subagent depend on. Treat any change to
the names, signatures, or return types of these symbols as a breaking
change. Internal helpers (leading underscore) and the private
`_DEGREE_TO_CARDINAL` mapping are not part of this contract and may
change without notice.
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

__all__ = [
    "CFD_RESULTS_ROOT",
    "load_campaign_results",
    "load_patch_csv",
    "load_patch_parquet",
    "load_patch_vtu",
]

logger = logging.getLogger(__name__)

# Default root for CFD results within this repo. Format string with one
# placeholder (``{site}``); used by ``load_campaign_results`` when no
# ``results_root`` override is passed.
CFD_RESULTS_ROOT = "data/{site}/cfd_results"

# Map the Airflow producer's wind_NNN directory name to canonical cardinal.
# 0 N, 45 NE, 90 E, 135 SE, 180 S, 225 SW, 270 W, 315 NW (meteorological bearing).
_DEGREE_TO_CARDINAL = {
    "000": "N",
    "045": "NE",
    "090": "E",
    "135": "SE",
    "180": "S",
    "225": "SW",
    "270": "W",
    "315": "NW",
}


def _normalize_wind_direction(name: str) -> Optional[str]:
    """Return the canonical cardinal name for a wind-direction directory.

    Accepts both layouts:
      - IVF native: ``N``, ``NE``, …, ``NW`` → returned unchanged.
      - Airflow native: ``wind_000``, ``wind_045``, …, ``wind_315`` → mapped.

    Returns None for anything else (callers should warn-and-skip).
    """
    if name in WIND_DIRECTIONS_8:
        return name
    if name.startswith("wind_") and len(name) == 8 and name[5:].isdigit():
        return _DEGREE_TO_CARDINAL.get(name[5:])
    return None


def _make_metadata(meta_dict: dict) -> PatchSimulationMetadata:
    """Build a PatchSimulationMetadata from a parsed summary.json dict."""
    return PatchSimulationMetadata(
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


def _ensure_u_mag(samples: pd.DataFrame) -> pd.DataFrame:
    """If U_mag is missing but U, V, W are present, compute it in place."""
    if "U_mag" not in samples.columns and {"U", "V", "W"}.issubset(samples.columns):
        import numpy as np

        samples["U_mag"] = np.sqrt(samples["U"] ** 2 + samples["V"] ** 2 + samples["W"] ** 2)
    return samples


def load_patch_csv(csv_path: Path, metadata_path: Optional[Path] = None) -> CFDPatchResult:
    """Load one patch × one wind-direction result from CSV + JSON.

    The IVF-native loader. Use `load_patch_parquet` for the Airflow-native
    parquet form, or `load_campaign_results` to auto-dispatch on layout.

    Parameters
    ----------
    csv_path : Path
        Path to ``sample_points.csv``.
    metadata_path : Path, optional
        Path to ``summary.json``. If omitted, looked up next to ``csv_path``.

    Returns
    -------
    CFDPatchResult
        Parsed metadata + samples DataFrame. ``U_mag`` is auto-computed
        from ``U/V/W`` if missing from the CSV.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` or the resolved ``metadata_path`` does not exist.

    Notes
    -----
    CSV schema (required columns): ``x, y, z, U, V, W, U_mag, TKE``.
    Optional: ``p`` (pressure).

    JSON schema (required keys): ``patch_id, site, wind_direction,
    wind_speed_ref``. Optional: ``converged, residual_final, solver,
    turbulence_model, n_iterations, wall_clock_s``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CFD CSV not found: {csv_path}")

    samples = _ensure_u_mag(pd.read_csv(csv_path))

    if metadata_path is None:
        metadata_path = csv_path.parent / "summary.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"CFD metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        meta_dict = json.load(f)

    return CFDPatchResult(metadata=_make_metadata(meta_dict), samples=samples)


def load_patch_parquet(
    parquet_paths: Path | list[Path],
    metadata_path: Optional[Path] = None,
) -> CFDPatchResult:
    """Load one patch × one wind-direction result from parquet + JSON.

    The Airflow-native loader. Mirrors `load_patch_csv` exactly except for
    the on-disk format. Use `load_campaign_results` to auto-dispatch.

    Parameters
    ----------
    parquet_paths : Path or list[Path]
        Single ``Path`` for the common case. Pass a list to concatenate
        row-wise — used when OpenFOAM emits one parquet per processor
        decomposition. Rows are concatenated in the order given.
    metadata_path : Path, optional
        Path to ``summary.json``. If omitted, looked up next to the first
        parquet file.

    Returns
    -------
    CFDPatchResult
        Same shape as `load_patch_csv` returns. ``U_mag`` auto-computed
        from ``U/V/W`` if missing.

    Raises
    ------
    ValueError
        If ``parquet_paths`` is an empty list.
    FileNotFoundError
        If any parquet path or the resolved ``metadata_path`` does not exist.

    Notes
    -----
    Column schema and metadata schema are identical to `load_patch_csv` —
    see that function's docstring.
    """
    if isinstance(parquet_paths, (str, Path)):
        parquet_paths = [Path(parquet_paths)]
    else:
        parquet_paths = [Path(p) for p in parquet_paths]

    if not parquet_paths:
        raise ValueError("load_patch_parquet requires at least one parquet path")
    for p in parquet_paths:
        if not p.exists():
            raise FileNotFoundError(f"CFD parquet not found: {p}")

    if len(parquet_paths) == 1:
        samples = pd.read_parquet(parquet_paths[0])
    else:
        samples = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    samples = _ensure_u_mag(samples)

    if metadata_path is None:
        metadata_path = parquet_paths[0].parent / "summary.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"CFD metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        meta_dict = json.load(f)

    return CFDPatchResult(metadata=_make_metadata(meta_dict), samples=samples)


def load_patch_vtu(vtu_path: Path) -> pd.DataFrame:
    """Load a full 3D field from a `.vtu` file.

    Optional path; the primary supported inputs are CSV/parquet via
    `load_patch_csv` / `load_patch_parquet`. Use this only when the
    OpenFOAM agent could not export the pre-sampled CSV/parquet and a
    full 3D field has to be ingested instead.

    Parameters
    ----------
    vtu_path : Path
        Path to a ``.vtu`` file.

    Returns
    -------
    DataFrame
        Sample points with columns ``x, y, z`` plus whatever vector
        (``U, V, W``) and scalar (``k``/``TKE``) fields exist on the
        mesh. Typically subset to a horizontal slice at z = 1.5 m in
        post-processing.

    Raises
    ------
    ImportError
        If ``pyvista`` is not installed (it is an optional dependency).
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
    """Load all CFD results for one site, auto-dispatching by layout.

    The primary entry point. Traverses
    ``data/{site}/cfd_results/{patch_id}/{wind_dir}/`` and loads every
    patch × direction combination that has ``summary.json`` plus either
    ``sample_points.csv`` (IVF native) or one or more ``*.parquet``
    files (Airflow native). Directories that match neither layout are
    logged as warnings and skipped.

    Parameters
    ----------
    site : str
        Site name (e.g., ``'vidigal'``, ``'maré'``). Used both as the
        path-template fill and in the returned ``CFDCampaignResult.site``.
    results_root : Path, optional
        Override the default path ``data/{site}/cfd_results/``. The
        wind rose, if any, is loaded from ``results_root.parent /
        wind_rose.json``.
    require_all_directions : bool
        If True, raise if any patch is missing any of the 8 cardinal
        directions. Default False (partial coverage is logged but
        accepted, since real CFD jobs sometimes fail to converge on a
        subset of directions).

    Returns
    -------
    CFDCampaignResult
        Patches keyed by patch_id, then by cardinal direction. Wind rose
        attached if a sibling ``wind_rose.json`` exists; ``None`` otherwise.

    Raises
    ------
    FileNotFoundError
        If ``results_root`` does not exist.
    ValueError
        If ``require_all_directions=True`` and any patch is missing
        directions.
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
            cardinal = _normalize_wind_direction(dir_dir.name)
            if cardinal is None:
                logger.warning(
                    "Skipping unknown wind direction directory: %s/%s",
                    patch_id,
                    dir_dir.name,
                )
                continue

            json_path = dir_dir / "summary.json"
            if not json_path.exists():
                logger.warning("Skipping %s/%s: no summary.json", patch_id, dir_dir.name)
                continue

            csv_path = dir_dir / "sample_points.csv"
            parquet_files = sorted(dir_dir.glob("*.parquet"))

            try:
                if csv_path.exists():
                    patches[patch_id][cardinal] = load_patch_csv(csv_path, json_path)
                elif parquet_files:
                    patches[patch_id][cardinal] = load_patch_parquet(parquet_files, json_path)
                else:
                    logger.warning(
                        "Skipping %s/%s: no sample_points.csv or *.parquet",
                        patch_id,
                        dir_dir.name,
                    )
                    continue
            except Exception as e:
                logger.error("Failed to load %s/%s: %s", patch_id, dir_dir.name, e)

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
