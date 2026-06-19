"""Morphometrics-only environmental prioritization (WS-B, geometry-first).

A pure-geometry deprivation score — no CFD. Three WHO/Grimmond-anchored,
ray-cast/geometric components, each scaled to [0,1] (higher = worse), combined
with provisional equal weights (the hook where sparse CFD anchors will later
recalibrate, per docs/morpho_signature_decisions.md):

- sun_deficit    : winter direct-sun shortfall below the WHO 2 h floor
- sky_enclosure  : 1 − SVF (sky obstruction)
- wind_stagnation: λf vs the 0.35 skimming-flow threshold (Grimmond & Oke 1999)

Scored at the observer (void) level — the unit of exposure — then aggregated to
cells by the worst-decile (p90), never a mean, and reported as quantile classes
rather than absolute units (we have no validated absolute scale without CFD).
Cells with no street support stay NULL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUN_FLOOR_H = 2.0
LF_SKIM = 0.35  # Grimmond & Oke 1999 skimming-flow threshold
WEIGHTS = {"sun_deficit": 1 / 3, "sky_enclosure": 1 / 3, "wind_stagnation": 1 / 3}
CLASS_LABELS = ("lower", "elevated", "highest")


def priority_components(obs: pd.DataFrame) -> pd.DataFrame:
    """The three [0,1] deprivation components per observer (higher = worse)."""
    sun = ((SUN_FLOOR_H - obs["solar_hours_winter"]) / SUN_FLOOR_H).clip(0, 1)
    sky = (1 - obs["svf"]).clip(0, 1)
    wind = (obs["lambda_f_mean"] / LF_SKIM).clip(0, 1)
    return pd.DataFrame({"sun_deficit": sun.to_numpy(),
                         "sky_enclosure": sky.to_numpy(),
                         "wind_stagnation": wind.to_numpy()},
                        index=obs.index)


def priority_score(obs: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Weighted composite priority in [0,1] per observer."""
    w = weights or WEIGHTS
    comp = priority_components(obs)
    return sum(comp[k] * w[k] for k in w)


def quantile_classes(score: pd.Series, edges=(1 / 3, 2 / 3)) -> pd.Series:
    """Tertile rank labels (lower / elevated / highest) — a rank, not a value."""
    q = score.rank(pct=True)
    out = np.where(q <= edges[0], CLASS_LABELS[0],
                   np.where(q <= edges[1], CLASS_LABELS[1], CLASS_LABELS[2]))
    return pd.Series(out, index=score.index)


def aggregate_priority_to_cells(
    obs: pd.DataFrame, grid: pd.DataFrame, score: pd.Series
) -> pd.DataFrame:
    """Worst-decile (p90) observer priority per cell; NULL where unsupported."""
    d = pd.DataFrame({"zone_id": obs["zone_id"].to_numpy(), "score": score.to_numpy()})
    d = d.dropna(subset=["zone_id"])
    agg = d.groupby("zone_id")["score"].agg(
        priority_p90=lambda s: s.quantile(0.90),
        priority_p50="median",
        n_obs="size",
    )
    out = agg.reindex(grid["zone_id"].to_numpy())
    out["has_priority"] = out["n_obs"].notna() & (out["n_obs"] > 0)
    out["n_obs"] = out["n_obs"].fillna(0).astype(int)
    out.index.name = "zone_id"
    return out.reset_index()
