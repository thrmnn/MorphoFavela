#!/usr/bin/env python3
"""P-05 — join example: match this package's ``p05_building_shade`` table
(``point_id``, ``timestamp_utc`` and ``timestamp_local`` at 5-minute steps, on
the walk dates, daylight only) against a real Octopus device CSV, by
``point_id`` and the device ``Timestamp`` (UTC) floored to 5 minutes.

Time: the loggers record UTC; Rio local time is America/Sao_Paulo (UTC-3, no
DST). The shade table carries both, so the join uses ``timestamp_utc``.

SPATIAL-JOIN CAVEAT: this join is by ``point_id``, not by GPS coordinate.
The device CSV must already carry the ``point_id`` its rows belong to
(e.g. assigned during data collection, or by a prior nearest-OM2-point
spatial join done with the full Brisa+ (MorphoFavela) repo's geopandas/KDTree
tooling). This standalone script (pandas + pyarrow only, no Brisa+ (MorphoFavela)
import) deliberately does not perform that spatial join itself — see
the nearest-OM2-point version of a spatial join.

Device CSV columns: ``Timestamp,Latitude,Longitude,Temperature,Humidity,
PM1.0,PM2.5,PM4.0,PM10.0`` (GPS-track schema). Latitude == Longitude == 0.0
is the firmware's no-fix sentinel and is dropped before joining, when those
columns are present.

Run (from inside the package directory):
    python OM2/join_shade_example.py --shade p05_building_shade.parquet \\
        --device path/to/octopus_log_with_point_id.csv --out joined_example.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def drop_nofix_rows(df: pd.DataFrame, lat_col: str = "Latitude", lon_col: str = "Longitude") -> pd.DataFrame:
    """Drop GPS no-fix sentinel rows (Latitude == Longitude == 0.0) before
    joining, when the device CSV carries those columns at all. Mirrors
    src/om_package/shade.py's function of the same name (kept in sync by
    hand — this file has no repo import to share it from)."""
    if lat_col not in df.columns or lon_col not in df.columns:
        return df
    no_fix = (df[lat_col].to_numpy() == 0.0) & (df[lon_col].to_numpy() == 0.0)
    return df.loc[~no_fix].reset_index(drop=True)


def join_shade_to_device(shade_df: pd.DataFrame, device_df: pd.DataFrame) -> pd.DataFrame:
    """Inner join on (point_id, timestamp_utc = device Timestamp (UTC)
    floored to 5 minutes)."""
    if "point_id" not in device_df.columns:
        raise ValueError(
            "device CSV has no 'point_id' column — this example joins by point_id "
            "(see module docstring's SPATIAL-JOIN CAVEAT); assign point_id first "
            "(e.g. via a nearest-OM2-point spatial join in the full repo) before calling this."
        )
    device_df = drop_nofix_rows(device_df)

    shade = shade_df.copy()
    shade["timestamp_utc"] = pd.to_datetime(shade["timestamp_utc"], utc=True)

    device = device_df.copy()
    device["timestamp_5min_utc"] = pd.to_datetime(device["Timestamp"], utc=True).dt.floor("5min")

    merged = shade.merge(
        device,
        left_on=["point_id", "timestamp_utc"],
        right_on=["point_id", "timestamp_5min_utc"],
        how="inner",
        suffixes=("_shade", "_device"),
    )
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shade", required=True, help="path to this package's p05_building_shade.parquet/.csv")
    ap.add_argument("--device", required=True, help="path to an Octopus device CSV carrying a point_id column")
    ap.add_argument("--out", required=True, help="output path (.parquet or .csv)")
    args = ap.parse_args()

    shade_df = _read_table(Path(args.shade))
    device_df = _read_table(Path(args.device))
    merged = join_shade_to_device(shade_df, device_df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        merged.to_csv(out_path, index=False)
    else:
        merged.to_parquet(out_path, index=False)

    print(f"[join_shade_example] shade rows={len(shade_df)} device rows={len(device_df)} -> matched rows={len(merged)}")
    print(f"[join_shade_example] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
