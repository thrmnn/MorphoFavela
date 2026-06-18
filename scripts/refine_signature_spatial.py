"""WS-A.2 — spatial refinement + stability for the morphotype signature.

Two checks on the WS-A clustering:
1. **Bootstrap stability** of k=6 (ARI of refits vs the reference labels).
2. **Contiguity mode-filter** per site that dissolves the salt-and-pepper 10 m
   morphotype maps into coherent regions, keeping the 6 global types. Writes a
   ``morphotype_smooth`` column back and reports the same-type adjacency
   (spatial purity) before/after. Decisions in docs/morpho_signature_decisions.md.

    python scripts/refine_signature_spatial.py
"""

from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # win over the stale brisa-0.1.0 editable install

from src.morphometry.signature import (  # noqa: E402
    assemble_signature_matrix,
    bootstrap_stability,
    fit_morphotypes,
    queen_neighbors,
    spatial_mode_smooth,
    spatial_purity,
    standardize,
)
from src.svf_v2.io import _git_sha  # noqa: E402

OUT = ROOT / "outputs" / "cross_site" / "signature"
RANDOM_STATE = 0
PASSES = 2


def main() -> None:
    # 1. stability on the pooled standardized matrix
    frames = []
    for p in sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet"))):
        g = gpd.read_parquet(p)
        g["site"] = Path(p).parents[1].name
        frames.append(pd.DataFrame(g.drop(columns="geometry")))
    df = pd.concat(frames, ignore_index=True)
    mat = assemble_signature_matrix(df)
    Xz, _ = standardize(mat)
    k = int(json.loads((OUT / "run_meta.json").read_text())["k"])
    ref = fit_morphotypes(Xz, k, random_state=RANDOM_STATE)
    aris = bootstrap_stability(Xz, ref, k, random_state=RANDOM_STATE)
    print(f"k={k} bootstrap ARI: mean={aris.mean():.3f} sd={aris.std():.3f} "
          f"min={aris.min():.3f} (n={len(aris)})")

    # 2. per-site spatial mode filter
    rows = []
    for p in sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet"))):
        site = Path(p).parents[1].name
        g = gpd.read_parquet(p).reset_index(drop=True)
        if "morphotype" not in g or g["morphotype"].isna().all():
            continue
        labels = g["morphotype"].astype(float).to_numpy()
        neigh = queen_neighbors(g)
        before = spatial_purity(labels, neigh)
        smooth = spatial_mode_smooth(labels, neigh, passes=PASSES)
        after = spatial_purity(smooth, neigh)
        valid = ~np.isnan(labels)
        changed = float(np.mean(smooth[valid] != labels[valid]))
        g["morphotype_smooth"] = pd.array(
            np.where(np.isnan(smooth), np.nan, smooth), dtype="Int64"
        )
        g.to_parquet(p)
        rows.append({"site": site, "purity_before": before,
                     "purity_after": after, "frac_changed": changed})
        print(f"{site:20s} purity {before:.2f} -> {after:.2f}  "
              f"(changed {changed:.1%})")

    report = pd.DataFrame(rows)
    report.to_csv(OUT / "spatial_refinement.csv", index=False)
    meta = {
        "git_sha": _git_sha(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passes": PASSES,
        "bootstrap_ari_mean": float(aris.mean()),
        "bootstrap_ari_sd": float(aris.std()),
        "bootstrap_ari_min": float(aris.min()),
        "purity_before_mean": float(report["purity_before"].mean()),
        "purity_after_mean": float(report["purity_after"].mean()),
    }
    (OUT / "stability_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
