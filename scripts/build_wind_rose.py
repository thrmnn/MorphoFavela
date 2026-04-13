#!/usr/bin/env python3
"""Build wind rose JSON files for CFD simulations.

Each site needs a wind_rose.json in data/{site}/ that documents the
frequency and mean speed of each of the 8 cardinal wind directions.
This is used to weight CFD results into annualised metrics.

Two modes:

    # From an INMET CSV (preferred — real measured data)
    python scripts/build_wind_rose.py --inmet-csv data/inmet/boa_vista.csv \
        --site vidigal --station "INMET Alto da Boa Vista" \
        --year-start 2010 --year-end 2023

    # From manual frequencies (placeholder or hand-curated)
    python scripts/build_wind_rose.py --from-template \
        --site vidigal

The ingestion assumes INMET CSV schema with columns:
    date, time, wind_speed_ms, wind_direction_deg

Direction in degrees (meteorological: 0 = from north, 90 = from east).
Outputs wind_rose.json matching src/cfd_integration/schema.py:WindRose.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cfd_integration.schema import WIND_DIRECTIONS_8

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Template based on published Rio de Janeiro climate summaries
# Coastal Rio is dominated by the SE trade-wind regime with NE return
# during winter. These are approximate priors — REPLACE with station data.
TEMPLATE_FREQUENCIES = {
    "N":  0.06, "NE": 0.14, "E":  0.16, "SE": 0.22,
    "S":  0.14, "SW": 0.10, "W":  0.08, "NW": 0.10,
}
TEMPLATE_MEAN_SPEEDS = {
    "N":  2.0, "NE": 3.2, "E":  3.5, "SE": 3.8,
    "S":  2.8, "SW": 2.4, "W":  2.2, "NW": 2.0,
}
TEMPLATE_SOURCE = (
    "PLACEHOLDER — climatological prior for Rio de Janeiro coastal zone. "
    "REPLACE with site-specific INMET station data before running CFD campaign."
)


# Recommended INMET station per site (proximity + terrain similarity)
RECOMMENDED_STATIONS = {
    "vidigal":            "Alto da Boa Vista (A652) — nearest mountainous site",
    "rocinha":            "Alto da Boa Vista (A652)",
    "riodaspedras":       "Jacarepaguá (A636) — west zone plain",
    "complexo_do_alemao": "Marambaia (A602) or Guaratiba — north plain",
    "maré":               "Ilha do Governador / Galeão — bayside",
}


def deg_to_cardinal(deg: float) -> str:
    """Convert direction in degrees to one of the 8 cardinal bins.

    Bins are 45° wide, centred on each cardinal direction.
    e.g., N covers [337.5, 22.5), NE covers [22.5, 67.5), etc.
    """
    if pd.isna(deg):
        return ""
    deg = deg % 360
    bins = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    # Each bin centred at 0, 45, 90, ... Shift by 22.5 for boundary.
    idx = int(((deg + 22.5) % 360) // 45)
    return bins[idx]


def from_inmet_csv(
    csv_path: Path,
    station: str,
    year_start: int | None = None,
    year_end: int | None = None,
) -> dict:
    """Build wind rose from an INMET observations CSV.

    Expects columns (case-insensitive, some tolerance for spacing):
        date (YYYY-MM-DD or similar), time, wind_speed_ms, wind_direction_deg

    Missing or invalid rows are dropped. Calm periods (wind < 0.5 m/s) are
    excluded from direction frequency but logged in the source note.
    """
    df = pd.read_csv(csv_path)
    # Normalise column names
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Identify required columns with fuzzy matching
    def _find_col(candidates):
        for cand in candidates:
            for c in df.columns:
                if cand in c:
                    return c
        return None

    date_col = _find_col(["data", "date"])
    spd_col = _find_col(["wind_speed", "velocidade", "vento_velocidade"])
    dir_col = _find_col(["wind_direction", "direcao", "vento_direcao"])

    if not spd_col or not dir_col:
        raise ValueError(
            f"Could not find wind speed / direction columns in {csv_path}. "
            f"Columns found: {list(df.columns)}"
        )

    # Optional year filter
    if date_col and (year_start or year_end):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if year_start:
            df = df[df[date_col].dt.year >= year_start]
        if year_end:
            df = df[df[date_col].dt.year <= year_end]

    # Drop invalid
    df = df.dropna(subset=[spd_col, dir_col])
    df[spd_col] = pd.to_numeric(df[spd_col], errors="coerce")
    df[dir_col] = pd.to_numeric(df[dir_col], errors="coerce")
    df = df.dropna(subset=[spd_col, dir_col])

    n_total = len(df)
    n_calm = (df[spd_col] < 0.5).sum()
    active = df[df[spd_col] >= 0.5].copy()

    active["cardinal"] = active[dir_col].apply(deg_to_cardinal)
    freq = active["cardinal"].value_counts(normalize=True).to_dict()
    speed = active.groupby("cardinal")[spd_col].mean().to_dict()

    # Ensure all 8 directions present (zero fill)
    frequencies = {d: float(freq.get(d, 0.0)) for d in WIND_DIRECTIONS_8}
    mean_speeds = {d: float(speed.get(d, 0.0)) for d in WIND_DIRECTIONS_8}

    period = ""
    if year_start or year_end:
        period = f" {year_start or '?'}-{year_end or '?'}"

    source = (
        f"{station}{period} — "
        f"n={n_total:,} observations ({n_calm:,} calm, "
        f"{n_total - n_calm:,} active, excluded from direction frequencies)"
    )

    return {
        "frequencies": frequencies,
        "mean_speeds": mean_speeds,
        "source": source,
    }


def from_template() -> dict:
    """Return the placeholder climatological prior."""
    return {
        "frequencies": dict(TEMPLATE_FREQUENCIES),
        "mean_speeds": dict(TEMPLATE_MEAN_SPEEDS),
        "source": TEMPLATE_SOURCE,
    }


def write_wind_rose(site: str, rose_data: dict, output_dir: Path | None = None) -> Path:
    """Write a wind_rose.json for a site."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / site
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "wind_rose.json"

    payload = {
        "site": site,
        "source": rose_data["source"],
        "frequencies": rose_data["frequencies"],
        "mean_speeds": rose_data["mean_speeds"],
        "recommended_station": RECOMMENDED_STATIONS.get(site, ""),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Wrote %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Build wind rose JSON for CFD simulations")
    parser.add_argument("--site", required=True,
                        choices=list(RECOMMENDED_STATIONS.keys()) + ["all"])
    parser.add_argument("--inmet-csv", type=Path, default=None,
                        help="INMET observations CSV (preferred)")
    parser.add_argument("--station", default="",
                        help="INMET station name (for source attribution)")
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument("--from-template", action="store_true",
                        help="Use the placeholder climatological prior (flagged as PLACEHOLDER)")
    args = parser.parse_args()

    if not args.inmet_csv and not args.from_template:
        parser.error("Must provide either --inmet-csv <path> or --from-template")

    sites = list(RECOMMENDED_STATIONS.keys()) if args.site == "all" else [args.site]

    for site in sites:
        if args.inmet_csv:
            rose = from_inmet_csv(args.inmet_csv, args.station or site,
                                  args.year_start, args.year_end)
        else:
            rose = from_template()
        write_wind_rose(site, rose)


if __name__ == "__main__":
    main()
