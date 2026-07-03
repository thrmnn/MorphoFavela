"""Build full-favela 3D-print STLs (a 5 cm box) + preview renders.

    python scripts/build_site_model.py                 # all 5 campaign sites
    python scripts/build_site_model.py --site vidigal   # one site

The whole favela is draped as a single watertight digital-surface-model solid
(building heights rasterised onto a coarsened terrain grid), scaled so the
longest side is --box mm. Output lands in outputs/{site}/print/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.print3d.model import ROOT, build_site_model, save_site_model
from src.print3d.render import render_site

CAMPAIGN_SITES = ["rocinha", "maré", "riodaspedras", "vidigal", "complexo_do_alemao"]
PRETTY = {
    "rocinha": "Rocinha",
    "maré": "Maré",
    "riodaspedras": "Rio das Pedras",
    "vidigal": "Vidigal",
    "complexo_do_alemao": "Complexo do Alemão",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", choices=CAMPAIGN_SITES, help="default: all campaign sites")
    ap.add_argument("--box", type=float, default=50.0, help="longest model side, mm")
    ap.add_argument("--cell", type=float, default=0.25, help="print grid cell, mm")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    sites = [args.site] if args.site else CAMPAIGN_SITES
    for site in sites:
        mesh, stats, dsm = build_site_model(site, target_mm=args.box, model_cell_mm=args.cell)
        out = save_site_model(mesh, stats, ROOT / "outputs" / site / "print")
        w, d, ht = stats.model_mm
        print(f"\n  {PRETTY[site]}  1:{stats.scale_denom}")
        print(f"  model size : {w} × {d} × {ht} mm   (relief {stats.relief_mm} mm)")
        print(f"  grid       : {stats.grid[0]}×{stats.grid[1]} @ {stats.cell_m} m/cell  ({stats.n_buildings:,} buildings)")
        print(f"  triangles  : {stats.triangles:,}   watertight: {stats.watertight}")
        print(f"  -> {out.relative_to(ROOT)}")
        if not args.no_preview:
            png = render_site(
                dsm,
                f"{PRETTY[site]} — full site",
                f"1:{stats.scale_denom} · {w:.0f}×{d:.0f} mm box · {stats.n_buildings:,} buildings · {stats.cell_m:.0f} m grid",
                ROOT / "outputs" / site / "print" / f"{site}_site_preview.png",
            )
            print(f"  -> {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
