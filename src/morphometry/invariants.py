"""Shared morphometry invariants — built-cell mask + phantom-tower filter.

Two contracts that several pipeline stages used to inline (and could drift apart):

1. ``built_mask`` — what counts as a *built* grid cell. Canonical definition is
   ``building_count > 0`` (the pooled population the signature/λf invariants are
   pinned to, n=64,389 across the 5 campaign sites). A lenient variant
   ``(lambda_p > 0.01) | (building_count > 0)`` exists for consumers that screened
   on coverage; on the canonical grids the two coincide.

2. ``drop_phantom_buildings`` — removes the Rio edificações ``topo == 0``
   corruption (``altura`` mis-derived as the base elevation → phantom towers up to
   232 m) before footprints reach height-dependent metrics or the SVF scene. The
   detector is the union of the two historically-used arms; the second arm
   (``altura ≈ base`` with an implausible > 40 m favela height) is a defensive
   catch that, post-cleanup, never fires beyond ``topo == 0`` on real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Implausible favela building height (m) above which an altura≈base footprint is
# treated as a phantom tower rather than a genuine block.
PHANTOM_HEIGHT_M = 40.0
# Built-cell lenient λp floor (a cell with this little coverage but no counted
# building is still "built" for the lenient variant).
LENIENT_LAMBDA_P = 0.01


def phantom_mask(buildings: pd.DataFrame, extruding_only: bool = False) -> np.ndarray:
    """Boolean mask of phantom-tower footprints (see module docstring).

    Returns an all-False mask when the height columns are absent, so the filter
    is a safe no-op on schemas without ``topo``/``altura``. With
    ``extruding_only=True`` the ``topo == 0`` arm is restricted to footprints
    that would actually extrude (``altura > 0``); zero-height ``topo == 0`` rows
    are then left in place. This matches ``build_extended_context``'s historical
    ``(topo == 0) & (altura > 0)`` predicate, so migrating it to this helper is
    bit-for-bit behaviour-preserving on the canonical grids.
    """
    if "topo" not in buildings.columns or "altura" not in buildings.columns:
        return np.zeros(len(buildings), dtype=bool)
    topo0 = (buildings["topo"] == 0)
    if extruding_only:
        topo0 = topo0 & (buildings["altura"] > 0)
    phantom = topo0.to_numpy()
    if "base" in buildings.columns:
        phantom = phantom | (
            ((buildings["altura"] - buildings["base"]).abs() < 0.01)
            & (buildings["altura"] > PHANTOM_HEIGHT_M)
        ).to_numpy()
    return phantom


def drop_phantom_buildings(buildings, verbose: bool = True, extruding_only: bool = False):
    """Drop phantom-tower footprints before they poison height-dependent metrics.

    The ``topo == 0`` corruption (source copied ``base`` into ``altura`` and
    zeroed the rooftop) produces absurd extruded heights — 83–232 m slivers in
    Rocinha — that wreck H_mean / σ_H / λf and the roughness z0/zd. No-op on
    schemas without the height columns. See ``phantom_mask`` for
    ``extruding_only``.
    """
    mask = phantom_mask(buildings, extruding_only=extruding_only)
    n = int(mask.sum())
    if n and verbose:
        print(f"  dropped {n} phantom-tower footprint(s) (topo==0 / altura==base >40 m)")
    return buildings[~mask].copy()


def built_mask(grid: pd.DataFrame, lenient: bool = False) -> np.ndarray:
    """Canonical built-cell boolean mask.

    Default (``lenient=False``) is ``building_count > 0`` — the pooled
    population the signature/λf invariants pin to (n=64,389 over the 5 campaign
    sites). ``lenient=True`` adds cells with ``lambda_p > 0.01`` but no counted
    building; on the canonical grids the two are identical.
    """
    bc = grid.get("building_count")
    bc_mask = (bc > 0).to_numpy() if bc is not None else np.zeros(len(grid), dtype=bool)
    if not lenient:
        return bc_mask
    lp = grid.get("lambda_p")
    lp_mask = (
        (lp.fillna(0) > LENIENT_LAMBDA_P).to_numpy()
        if lp is not None
        else np.zeros(len(grid), dtype=bool)
    )
    return lp_mask | bc_mask
