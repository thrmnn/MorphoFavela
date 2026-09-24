#!/usr/bin/env python3
"""P-03 — re-aggregate an OM package's point table to any segment length
the team chooses. Point count is conserved: every point lands in exactly
one segment.

Run:
    python scripts/aggregate_om_points.py \\
        --points outputs/_packages/mare_om2/v0.1/OM2/points.parquet \\
        --segment-length-m 20 \\
        --out outputs/_packages/mare_om2/v0.1/OM2/segments_20m.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.om_package.segments import aggregate_to_segments


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="path to a points parquet/csv (from build_om_package.py)")
    ap.add_argument("--segment-length-m", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    points_path = Path(args.points)
    df = pd.read_parquet(points_path) if points_path.suffix == ".parquet" else pd.read_csv(points_path)

    segments = aggregate_to_segments(df, args.segment_length_m)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        segments.to_csv(out_path, index=False)
    else:
        segments.to_parquet(out_path, index=False)

    n_points_in = len(df)
    n_points_out = int(segments["n_points"].sum())
    print(f"[aggregate_om_points] {n_points_in} points -> {len(segments)} segments of {args.segment_length_m} m")
    print(f"[aggregate_om_points] point count conserved: {n_points_in == n_points_out} ({n_points_in} == {n_points_out})")
    print(f"[aggregate_om_points] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
