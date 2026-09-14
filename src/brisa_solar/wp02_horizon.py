"""WP-02: per-patch visibility via raster horizon-angle marching.

Spec: docs/wp02_horizon_engine_spec.md §2. For each observer and each Tregenza
sky-patch direction, march outward over the obstruction surface
(wp02_surface.py) and compare the patch's altitude against the highest
elevation angle encountered along the ray (the classic horizon-angle method).
Runs identical code on cuda and cpu via torch; chunked over observers so a
chunk's working set is (chunk, P) at any instant, not (chunk, P, n_steps) —
the running max over steps is accumulated in a loop instead of materialised,
which keeps memory flat regardless of step count.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .constants import P1_SKY_PATCHES, REPO_ROOT, load_params


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _inv_coeffs(transform):
    inv = ~transform
    return inv.a, inv.b, inv.c, inv.d, inv.e, inv.f


def _bilinear_sample(surface_t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, inv_coeffs) -> torch.Tensor:
    """Bilinear surface sample at world (x, y); clamps to the raster edge outside its bounds."""
    a, b, c, d, e, f = inv_coeffs
    colf = a * x + b * y + c
    rowf = d * x + e * y + f
    u = colf - 0.5
    v = rowf - 0.5
    h, w = surface_t.shape[-2], surface_t.shape[-1]
    u0 = torch.floor(u)
    v0 = torch.floor(v)
    fu = (u - u0).clamp(0.0, 1.0)
    fv = (v - v0).clamp(0.0, 1.0)
    u0c = u0.long().clamp(0, w - 1)
    u1c = (u0 + 1).long().clamp(0, w - 1)
    v0c = v0.long().clamp(0, h - 1)
    v1c = (v0 + 1).long().clamp(0, h - 1)
    z00 = surface_t[v0c, u0c]
    z01 = surface_t[v0c, u1c]
    z10 = surface_t[v1c, u0c]
    z11 = surface_t[v1c, u1c]
    z0 = z00 * (1 - fu) + z01 * fu
    z1 = z10 * (1 - fu) + z11 * fu
    return z0 * (1 - fv) + z1 * fv


def _nearest_sample(surface_t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, inv_coeffs):
    """Nearest-cell sample plus the (row, col) indices used — the observer's own cell."""
    a, b, c, d, e, f = inv_coeffs
    colf = a * x + b * y + c
    rowf = d * x + e * y + f
    h, w = surface_t.shape[-2], surface_t.shape[-1]
    col = torch.floor(colf).long().clamp(0, w - 1)
    row = torch.floor(rowf).long().clamp(0, h - 1)
    return surface_t[row, col], row, col


def patch_visibility(
    surface: np.ndarray,
    transform,
    obs_xy: np.ndarray,
    *,
    directions: np.ndarray,
    is_building: np.ndarray | None = None,
    obs_height_m: float = 1.5,
    max_dist_m: float = 500.0,
    step_m: float | None = None,
    march_sampling: str = "nearest",
    device: str | None = None,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-(observer, patch) visibility by horizon-angle raster marching.

    Returns ``(visible, on_building)``: ``visible`` is ``(n, P)`` bool with
    ``P == P1_SKY_PATCHES``; ``on_building`` is ``(n,)`` bool. An observer
    whose cell is a building cell (per ``is_building``) gets an all-False
    visibility row and ``on_building[i] = True`` instead of being silently
    sampled from the roof (spec §2).

    Sampling: the observer's own elevation is always read at its NEAREST
    cell (it sits at a specific cell, not an interpolated point).
    ``march_sampling`` controls how each horizon-march step reads the
    surface: ``"nearest"`` (default) or ``"bilinear"``, both clamped to the
    surface's edge outside its bounds. NEAREST is the default because it is
    the measured winner against the CPU (exact-polygon) reference on Rio das
    Pedras (2026-09-14): bilinear interpolation blends a wall cell with its
    shorter ground neighbour, softening every building silhouette and
    letting extra sky "leak" past real edges (median |Delta| 0.042 -> 0.014,
    r 0.988 -> 0.995; see runs/wp02_horizon_*/crossref_diagnostic.json).
    """
    if march_sampling not in ("nearest", "bilinear"):
        raise ValueError(f"march_sampling must be 'nearest' or 'bilinear', got {march_sampling!r}")
    march_sample_fn = _nearest_sample if march_sampling == "nearest" else _bilinear_sample
    if directions.shape[0] != P1_SKY_PATCHES:
        raise ValueError(f"expected {P1_SKY_PATCHES} directions, got {directions.shape[0]}")
    n_patches = directions.shape[0]

    dev = device or default_device()
    torch_dev = torch.device(dev)
    dtype = torch.float64

    surface_t = torch.as_tensor(np.ascontiguousarray(surface), dtype=dtype, device=torch_dev)
    ib_t = None
    if is_building is not None:
        ib_t = torch.as_tensor(np.ascontiguousarray(is_building), dtype=torch.bool, device=torch_dev)

    obs_xy = np.asarray(obs_xy, dtype=np.float64)
    n = obs_xy.shape[0]

    step = float(step_m) if step_m is not None else abs(transform.a)
    n_steps = max(1, int(round(max_dist_m / step)))
    ts = (torch.arange(1, n_steps + 1, device=torch_dev, dtype=dtype)) * step

    dxv = torch.as_tensor(directions[:, 0], dtype=dtype, device=torch_dev)
    dyv = torch.as_tensor(directions[:, 1], dtype=dtype, device=torch_dev)
    dzv = torch.as_tensor(directions[:, 2], dtype=dtype, device=torch_dev)
    horiz_norm = torch.sqrt(dxv * dxv + dyv * dyv)
    zenith_patch = horiz_norm <= 1e-9
    safe_norm = torch.where(zenith_patch, torch.ones_like(horiz_norm), horiz_norm)
    hx = torch.where(zenith_patch, torch.zeros_like(dxv), dxv / safe_norm)
    hy = torch.where(zenith_patch, torch.zeros_like(dyv), dyv / safe_norm)
    alt = torch.asin(dzv.clamp(-1.0, 1.0))

    inv_coeffs = _inv_coeffs(transform)

    out_visible = np.zeros((n, n_patches), dtype=bool)
    out_on_building = np.zeros(n, dtype=bool)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        ox = torch.as_tensor(obs_xy[start:end, 0], dtype=dtype, device=torch_dev)
        oy = torch.as_tensor(obs_xy[start:end, 1], dtype=dtype, device=torch_dev)
        m = ox.shape[0]

        z_ground, row, col = _nearest_sample(surface_t, ox, oy, inv_coeffs)
        if ib_t is not None:
            on_building = ib_t[row, col]
        else:
            on_building = torch.zeros(m, dtype=torch.bool, device=torch_dev)
        z_obs = z_ground + obs_height_m

        horizon = torch.full((m, n_patches), float("-inf"), dtype=dtype, device=torch_dev)
        for t in ts:
            xs = ox[:, None] + t * hx[None, :]
            ys = oy[:, None] + t * hy[None, :]
            zs = march_sample_fn(surface_t, xs, ys, inv_coeffs)
            if march_sampling == "nearest":
                zs = zs[0]   # (_nearest_sample also returns row, col; unused for the march)
            ang = torch.atan2(zs - z_obs[:, None], t)
            horizon = torch.maximum(horizon, ang)

        vis = alt[None, :] > horizon
        vis = vis | zenith_patch[None, :]   # zenith cap: no horizontal march exists to obstruct it
        vis = vis & ~on_building[:, None]

        out_visible[start:end] = vis.cpu().numpy()
        out_on_building[start:end] = on_building.cpu().numpy()

    return out_visible, out_on_building


def svf_unweighted(visibility: np.ndarray) -> np.ndarray:
    """Count-ratio SVF: visible patches / total patches, no solid-angle weighting."""
    v = np.asarray(visibility, dtype=float)
    return v.mean(axis=-1)


def svf_solid_angle(visibility: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Solid-angle-weighted SVF: sum(visible weights) / sum(all weights), no cosine term."""
    v = np.asarray(visibility, dtype=float)
    w = np.asarray(weights, dtype=float)
    return (v @ w) / w.sum()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def write_run_manifest(
    run_dir,
    *,
    cell_m: float,
    obs_height_m: float,
    max_dist_m: float,
    step_m: float,
    sampling_rule: str,
    device: str,
) -> Path:
    """Write runs/<run_id>/manifest.json (spec §4); read by test_all_run_manifests_used_the_same_sky."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    params = load_params()
    sky_section = json.dumps(params["sky"], sort_keys=True)
    manifest = {
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sky": {"patches": int(P1_SKY_PATCHES)},
        "cell_m": cell_m,
        "obs_height_m": obs_height_m,
        "max_dist_m": max_dist_m,
        "step_m": step_m,
        "sampling_rule": sampling_rule,
        "device": device,
        "torch_version": torch.__version__,
        "git_sha": _git_sha(),
        "params_sky_section_sha256": hashlib.sha256(sky_section.encode()).hexdigest()[:16],
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path
