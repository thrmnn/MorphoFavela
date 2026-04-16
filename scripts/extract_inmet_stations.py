#!/usr/bin/env python3
"""Extract + concatenate INMET BDMEP CSVs across years for a fixed
list of stations.

INMET publishes one ZIP per calendar year (nationwide). Each ZIP
contains one CSV per automatic station. For the wind rose work we
only need four RJ stations (A652, A636, A621, A602) across the
2015–2024 window; this script pulls just those CSVs out of each
yearly ZIP and concatenates each station's records into a single
CSV ready for `scripts/build_wind_rose.py --inmet-csv`.

Usage:
    python scripts/extract_inmet_stations.py \\
        --zips-dir data/inmet/raw \\
        --out-dir data/inmet/processed \\
        --stations A652 A636 A621 A602
"""

from __future__ import annotations

import argparse
import logging
import re
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


STATION_PATTERN = re.compile(r"_(A\d{3})_")


def extract_one_zip(
    zip_path: Path,
    wanted: set[str],
    out_root: Path,
) -> dict[str, Path]:
    """Pull CSVs for `wanted` station codes out of one yearly ZIP.

    Returns {station_code: extracted_csv_path}.
    """
    year = zip_path.stem
    out_dir = out_root / "per_year" / year
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if not name.endswith((".CSV", ".csv")):
                continue
            m = STATION_PATTERN.search(name)
            if not m:
                continue
            code = m.group(1)
            if code not in wanted:
                continue
            target = out_dir / name
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted[code] = target
            logger.info("  %s/%s -> %s", year, code, target.relative_to(out_root))

    missing = wanted - set(extracted)
    if missing:
        logger.warning("%s: missing stations %s", year, sorted(missing))
    return extracted


def concat_station_across_years(
    station: str,
    per_year_files: list[Path],
    out_path: Path,
) -> Path:
    """Concatenate one station's per-year CSVs into a single file.

    INMET CSVs have an 8-row metadata header per file. When concatenating
    across years we keep the first file's header intact and drop the
    header block from subsequent files so pandas can read the combined
    file with skiprows=8 as usual.
    """
    if not per_year_files:
        raise ValueError(f"No files to concat for {station}")

    per_year_files = sorted(per_year_files, key=lambda p: p.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as out:
        for i, f in enumerate(per_year_files):
            with open(f, "rb") as src:
                data = src.read()
            if i == 0:
                out.write(data)
            else:
                # Skip 9 lines (8 metadata + 1 column header)
                lines = data.splitlines(keepends=True)
                out.write(b"".join(lines[9:]))
    logger.info("concatenated %s from %d years -> %s",
                station, len(per_year_files), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract + concatenate INMET BDMEP CSVs for named stations",
    )
    parser.add_argument("--zips-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--stations", nargs="+", required=True,
        help="INMET station codes (e.g., A652 A636 A621 A602)",
    )
    args = parser.parse_args()

    wanted = set(args.stations)
    zip_paths = sorted(args.zips_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No ZIPs in {args.zips_dir}")

    logger.info("Processing %d yearly ZIPs × %d stations",
                len(zip_paths), len(wanted))

    per_station: dict[str, list[Path]] = {s: [] for s in wanted}
    for zp in zip_paths:
        try:
            extracted = extract_one_zip(zp, wanted, args.out_dir)
        except zipfile.BadZipFile as e:
            logger.error("Bad ZIP %s: %s (skipping)", zp, e)
            continue
        for code, path in extracted.items():
            per_station[code].append(path)

    concat_dir = args.out_dir / "concat"
    for code, files in per_station.items():
        if not files:
            logger.warning("No files extracted for %s; skipping concat", code)
            continue
        out_path = concat_dir / f"{code}_2015_2024.csv"
        concat_station_across_years(code, files, out_path)

    logger.info("done")


if __name__ == "__main__":
    main()
