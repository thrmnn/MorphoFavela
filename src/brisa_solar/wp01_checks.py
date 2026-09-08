"""WP-01 data-foundation checks that need no DSM.

The plan's A1.2 (footprint height vs DSM-DTM on 20 random buildings) is not
computable: no DSM exists on disk (verified 2026-09-08). These two checks test
what the obstruction surface actually rests on instead.

  base_vs_dtm  — the obstruction surface places a building at DTM + height, so
                 the footprint layer's own `base` elevation must agree with the
                 DTM under it. Disagreement means buildings float or sink.
  height_invariant — `topo` must equal `base + altura`. A violation means the
                 height attributes are internally inconsistent and neither can
                 be trusted.

Sampling is SYSTEMATIC (every Nth feature), not random: the shapefile has
2.36M features and a strided pass avoids loading them all. Reported as such.

Run: python3 -m src.brisa_solar.wp01_checks [--stride N] [--run-id ID]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import fiona
import numpy as np
import rasterio
from shapely.geometry import shape

from .constants import REPO_ROOT, load_params


def _summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "median_abs": round(float(np.median(np.abs(values))), 4),
        "mean": round(float(np.mean(values)), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "max_abs": round(float(np.max(np.abs(values))), 4),
    }


def run(stride: int, run_id: str) -> dict:
    params = load_params()
    fp = REPO_ROOT / params["footprints"]["path"]
    dtm_path = REPO_ROOT / params["terrain"]["dtm_city"]
    h_attr = params["footprints"]["height_attr"]
    b_attr = params["footprints"]["base_attr"]
    t_attr = params["footprints"]["top_attr"]

    xs, ys, bases, alturas, topos = [], [], [], [], []
    with fiona.open(fp) as src:
        total = len(src)
        for i, feat in enumerate(src):
            if i % stride:
                continue
            props = feat["properties"]
            b, a, t = props.get(b_attr), props.get(h_attr), props.get(t_attr)
            if b is None or a is None or t is None:
                continue
            c = shape(feat["geometry"]).centroid
            xs.append(c.x); ys.append(c.y)
            bases.append(b); alturas.append(a); topos.append(t)

    bases = np.asarray(bases, float); alturas = np.asarray(alturas, float)
    topos = np.asarray(topos, float)

    with rasterio.open(dtm_path) as dtm:
        sampled = np.array([v[0] for v in dtm.sample(zip(xs, ys))], dtype=float)
        nodata = dtm.nodata

    valid = np.isfinite(sampled) & (sampled < nodata / 2)
    diff = bases[valid] - sampled[valid]
    invariant = topos - (bases + alturas)

    result = {
        "_run_id": run_id,
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_check": "WP-01 base_vs_dtm + height_invariant (A1.2 substitute — no DSM on disk)",
        "sampling": {
            "method": "systematic (every Nth feature), not random",
            "stride": stride,
            "population": total,
            "sampled": len(bases),
        },
        "inputs": {"footprints": str(fp.relative_to(REPO_ROOT)),
                   "dtm": str(dtm_path.relative_to(REPO_ROOT)),
                   "height_attrs": [b_attr, h_attr, t_attr]},
        "base_vs_dtm_m": {
            **_summary(diff),
            "outside_dtm_coverage": int((~valid).sum()),
            "frac_gt_1m": round(float(np.mean(np.abs(diff) > 1.0)), 4) if diff.size else None,
            "frac_gt_5m": round(float(np.mean(np.abs(diff) > 5.0)), 4) if diff.size else None,
        },
        "height_invariant_topo_minus_base_plus_altura": {
            **_summary(invariant),
            "n_violations_gt_1mm": int(np.sum(np.abs(invariant) > 1e-3)),
        },
        "altura_m": _summary(alturas),
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1000)
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("wp01_%Y%m%dT%H%M%SZ"))
    args = ap.parse_args()

    res = run(args.stride, args.run_id)
    out_dir = REPO_ROOT / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wp01_base_vs_dtm.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"\nwrote {out_dir.relative_to(REPO_ROOT)}/wp01_base_vs_dtm.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
