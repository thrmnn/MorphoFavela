"""Build a resin-printable STL from a CFD analysis patch.

    python scripts/build_print_model.py --site rocinha --patch ROC-P18 --scale 1000

Default ROC-P18 (Rocinha, 35.7° slope, lambda_p 0.77) — a steep, dense hillside
core whose stacked relief showcases resin detail. Output lands in
outputs/{site}/print/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.print3d.model import ROOT, build_print_model, save_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="rocinha")
    ap.add_argument("--patch", default="ROC-P18")
    ap.add_argument("--scale", type=int, default=1000,
                    help="scale denominator; 1000 → 100 m patch = 10 cm model, 500 → 20 cm")
    ap.add_argument("--embed", type=float, default=2.0,
                    help="metres each building is sunk into the plinth so it fuses")
    ap.add_argument("--base-thickness", type=float, default=3.0,
                    help="solid plinth thickness in model mm")
    ap.add_argument("--sampling", default="campaign_sampling")
    args = ap.parse_args()

    mesh, stats = build_print_model(
        args.site, args.patch, scale_denom=args.scale,
        embed=args.embed, base_thickness=args.base_thickness, sampling=args.sampling,
    )
    out = save_model(mesh, stats, ROOT / "outputs" / args.site / "print")
    w, d, h = stats.model_mm
    print(f"  {args.site}/{args.patch}  1:{args.scale}")
    print(f"  model size   : {w} × {d} × {h} mm  (relief {stats.relief_mm} mm)")
    print(f"  buildings    : {stats.n_buildings}  (tallest {stats.tallest_building_mm} mm)")
    print(f"  triangles    : {stats.triangles:,}")
    print(f"  manifold     : {stats.single_manifold} (single watertight solid: {stats.watertight})")
    print(f"  -> {out.relative_to(ROOT)}")
    if not stats.single_manifold:
        print("  NOTE: manifold3d unavailable — exported as a union of individually closed "
              "solids. Resin slicers (Chitubox/Lychee) handle this; install manifold3d for "
              "a single watertight solid.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
