"""Per-CFD-patch morphometric roughness z0(θ)/zd(θ) — the CFD-inlet hand-off.

For each campaign patch, take the patch-scale morphometry already in patch_meta.json
(λp, H_mean, σH, H_max) and the patch-aggregated frontal area λf(θ) from the grid
cells inside the analysis disk, then run UMEP/Kanda to get the **morphometric**
z0(θ)/zd(θ) that sets the CFD inlet ABL profile + k_eq for that patch. This is the
upstream-settlement roughness (one of the two decoupled z0 roles); the ground z0
inside the resolved patch stays small. NOT the CFD-extracted z0 (that is R-C, gated
on real OpenFOAM). Decisions: docs/roughness_decisions.md.

    python scripts/build_patch_roughness.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.morphometry.roughness import (
    DIRS,
    patch_mean_lambda_f,
    roughness,
)

OUT = ROOT / "outputs" / "cross_site" / "roughness"


def main():
    rows = []
    for gridp in sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet"))):
        site = Path(gridp).parents[1].name
        cells = gpd.read_parquet(gridp)
        if "centroid_x" not in cells:
            cells["centroid_x"] = cells.geometry.centroid.x
            cells["centroid_y"] = cells.geometry.centroid.y
        metas = sorted(glob.glob(str(ROOT / "outputs" / site /
                       "sampling_cfd" / "campaign_sampling" / "patches" / "*" / "patch_meta.json")))
        for mp in metas:
            m = json.loads(Path(mp).read_text())
            cx, cy = m.get("center_x"), m.get("center_y")
            radius = m.get("analysis_patch_diameter", 100) / 2
            zH, pai = m.get("H_mean"), m.get("lambda_p")
            zMax = m.get("H_max_analysis", (zH or 0) + 2.5 * (m.get("sigma_h") or 0))
            zSdev = m.get("sigma_h", 0) or 0
            laf = patch_mean_lambda_f(cells, cx, cy, radius)
            fai_mean = float(np.nanmean([laf[d] for d in DIRS]))
            zd_kan, z0_kan = roughness("Kan", zH, fai_mean, pai, zMax, zSdev)
            row = {"site": site, "patch_id": m.get("patch_id"),
                   "z0_kan": z0_kan, "zd_kan": zd_kan,
                   "lambda_p": pai, "H_mean": zH, "sigma_h": zSdev, "H_max": zMax,
                   "slope_deg": m.get("slope_deg"), "n_cells": laf["n_cells"],
                   "flag_pai_over_envelope": bool(pai is not None and pai > 0.5)}
            for d in DIRS:
                row[f"z0_kan_{d}"] = roughness("Kan", zH, laf[d], pai, zMax, zSdev)[1]
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no campaign patches found")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "patch_roughness.csv", index=False)
    for site, sub in df.groupby("site"):
        dst = (ROOT / "outputs" / site / "sampling_cfd" / "campaign_sampling"
               / "patch_roughness.csv")
        if dst.parent.exists():
            sub.to_csv(dst, index=False)
    ok = df["z0_kan"].notna()
    print(f"{len(df)} patches, {int(ok.sum())} with z0; "
          f"{int(df['flag_pai_over_envelope'].sum())} flagged λp>0.5")
    print(df.groupby("site")["z0_kan"].agg(["count", "median"]).round(3).to_string())


if __name__ == "__main__":
    main()
