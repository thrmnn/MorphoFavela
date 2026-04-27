#!/usr/bin/env python3
"""Build wind rose JSON files for CFD simulations.

Each site needs a wind_rose.json in data/{site}/ that documents the
frequency and mean speed of each of the 8 cardinal wind directions
at a defined anemometer height (nominally z = 10 m). This is used
to weight CFD results into annualised metrics.

Two modes:

    # From an INMET CSV (preferred — real measured data)
    python scripts/build_wind_rose.py --inmet-csv data/inmet/alto_boa_vista.csv \
        --site vidigal --station "Alto da Boa Vista (A652)" \
        --year-start 2015 --year-end 2024

    # From the site-specific climatological prior (PLACEHOLDER)
    python scripts/build_wind_rose.py --from-template --site all

The ingestion assumes INMET CSV schema with columns (case-insensitive,
fuzzy-matched):

    date (or "data"), time, wind_speed_ms (or "velocidade"),
    wind_direction_deg (or "direcao")

Direction in degrees (meteorological: 0 = from north, 90 = from east).

Output wind_rose.json matches src/cfd_integration/schema.py:WindRose,
including provenance metadata (station id + coords, time window,
n observations, calm fraction, quality flag).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cfd_integration.schema import WIND_DIRECTIONS_8

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Climatological prior for Rio de Janeiro coastal zone.
#
# Based on published climate summaries: SE trade-wind regime dominates,
# NE return in winter. These values are an approximate Rio-wide prior and
# do NOT capture per-site terrain effects (blocking, channelling, bay
# funnelling). REPLACE with site-specific INMET station data before
# running the CFD campaign — measured observations will differ materially
# from this prior, especially at Rocinha (valley channelling), Vidigal
# (mountain lee-side blocking of SE flow), and Maré (Guanabara Bay
# funnelling).
# ─────────────────────────────────────────────────────────────────────────────

PRIOR_FREQUENCIES = {
    "N":  0.06, "NE": 0.14, "E":  0.16, "SE": 0.22,
    "S":  0.14, "SW": 0.10, "W":  0.08, "NW": 0.10,
}
PRIOR_MEAN_SPEEDS = {
    "N":  2.0, "NE": 3.2, "E":  3.5, "SE": 3.8,
    "S":  2.8, "SW": 2.4, "W":  2.2, "NW": 2.0,
}
PRIOR_SOURCE = (
    "PLACEHOLDER — Rio de Janeiro coastal-zone climatological prior. "
    "REPLACE with site-specific INMET station data before running CFD "
    "campaign; see scripts/build_wind_rose.py --inmet-csv."
)


@dataclass
class SiteProfile:
    """Per-site metadata used by the template writer."""
    exposure_class: str                # coastal / valley / plain / urban / bayside
    recommended_station_id: str        # INMET code (e.g., "A652")
    recommended_station_name: str
    recommended_station_coords: Optional[tuple[float, float]] = None  # (lat, lon)
    expected_adjustment: str = ""      # how real data is likely to differ from prior


# Per-site profiles. Station recommendations were validated by the INMET
# research agent (April 2026) against the INMET catalogue, daily-graph
# URLs, and a published TMY paper for A652. Coordinates for A621 and A602
# are from cross-referenced secondary sources; verify against the
# catalogue CSV before citing in the technical report.
#
# NOTE: the earlier placeholder attributed A652 to "Alto da Boa Vista".
# A652 is actually "Forte de Copacabana" (INMET code); the Alto da Boa
# Vista station belongs to the municipal Alerta Rio network, not INMET.
# Corrected below.
SITE_PROFILES: dict[str, SiteProfile] = {
    "vidigal": SiteProfile(
        exposure_class="coastal hillside (mountain lee-side)",
        recommended_station_id="A652",
        recommended_station_name="Forte de Copacabana",
        recommended_station_coords=(-22.988, -43.190),
        expected_adjustment=(
            "A652 is the nearest unobstructed coastal reference (~5 km "
            "east). Dois Irmãos blocks direct SE flow onto the Vidigal "
            "slope, so expect the site-scale rose to show reduced SE and "
            "increased N/NW from lee-side eddies relative to A652's "
            "open-coast prior. CFD simulations capture this locally; the "
            "inflow rose should remain A652."
        ),
    ),
    "rocinha": SiteProfile(
        exposure_class="valley (channelled NE-SW)",
        recommended_station_id="A652",
        recommended_station_name="Forte de Copacabana",
        recommended_station_coords=(-22.988, -43.190),
        expected_adjustment=(
            "A652 provides the unobstructed coastal driver for the "
            "SE→NE regime that ventilates the Rocinha valley. The "
            "valley between Dois Irmãos and Pedra da Gávea channels "
            "flow along the NE–SW axis — CFD will resolve the in-valley "
            "bimodal pattern; the inflow rose should stay at A652."
        ),
    ),
    "riodaspedras": SiteProfile(
        exposure_class="western plain (Jacarepaguá lowland)",
        recommended_station_id="A636",
        recommended_station_name="Jacarepaguá",
        recommended_station_coords=(-22.99, -43.37),
        expected_adjustment=(
            "A636 is colocated with the Jacarepaguá lagoon basin (~2–4 "
            "km). Prior is reasonable; expect SE dominance to hold with "
            "slightly reduced speeds vs A652 due to inland position."
        ),
    ),
    "complexo_do_alemao": SiteProfile(
        exposure_class="urban interior (north zone)",
        recommended_station_id="A621",
        recommended_station_name="Vila Militar",
        recommended_station_coords=(-22.86, -43.41),
        expected_adjustment=(
            "Closest urban-interior north-zone station (~8 km W). "
            "Previous placeholder recommended A602 Marambaia which is "
            "geographically mismatched (southwest coast, not north "
            "zone) — corrected. Expect lower mean speeds than A652 "
            "(~20–30%) and flatter directional distribution."
        ),
    ),
    "maré": SiteProfile(
        exposure_class="bayside (Guanabara Bay)",
        recommended_station_id="A652",   # INMET fallback
        recommended_station_name="Forte de Copacabana (INMET fallback; SBGL Galeão METAR preferred)",
        recommended_station_coords=(-22.988, -43.190),
        expected_adjustment=(
            "Best match is SBGL (Galeão airport) METAR via Iowa ASOS "
            "archive (~3 km N, same bay-wind regime, WMO-compliant, "
            ">20 yr continuous). INMET A652 is the fallback if staying "
            "within BDMEP. Guanabara Bay acts as a NE–SW channel; "
            "expect stronger NE/SE peaks and higher mean speeds than "
            "the prior, with suppressed W/NW. METAR ingestion not yet "
            "implemented in this script — see from_inmet_csv adaptation."
        ),
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# INMET ingestion
# ─────────────────────────────────────────────────────────────────────────────


def deg_to_cardinal(deg: float) -> str:
    """Convert direction in degrees to one of the 8 cardinal bins.

    Bins are 45° wide, centred on each cardinal direction.
    e.g., N covers [337.5, 22.5), NE covers [22.5, 67.5), etc.
    """
    if pd.isna(deg):
        return ""
    deg = deg % 360
    bins = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((deg + 22.5) % 360) // 45)
    return bins[idx]


def _read_inmet_csv(csv_path: Path) -> pd.DataFrame:
    """Read one INMET BDMEP automatic-station CSV.

    BDMEP format (verified April 2026):
      - `;` delimiter, `,` decimal, latin-1 encoding
      - 8-row metadata header (station code, name, lat, lon, alt, start)
      - Data columns in Portuguese, e.g.
          'DATA (YYYY-MM-DD)', 'HORA (UTC)',
          'VENTO, VELOCIDADE HORARIA (m/s)',
          'VENTO, DIRECAO HORARIA (gr) (° (gr))'
      - Missing values encoded as -9999
      - Anemometer height: 10 m (WMO standard)
    """
    return pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        encoding="latin-1",
        skiprows=8,
        na_values=["-9999", "-9999.0", ""],
    )


def from_inmet_csv(
    csv_path: Path | list[Path],
    station_id: str,
    station_name: str,
    station_coords: Optional[tuple[float, float]] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> dict:
    """Build a wind rose from one or more INMET BDMEP CSVs.

    Accepts a single CSV path or a list of per-year CSV paths (one
    INMET yearly ZIP unpacks into many per-station CSVs — concatenate
    the per-year CSVs for the same station to get a multi-year window).

    Calm periods (|U| < 0.5 m/s OR direction = NaN) are excluded from
    direction binning but their count is recorded in ``calm_fraction``.
    """
    paths = [csv_path] if isinstance(csv_path, Path) else list(csv_path)
    frames = [_read_inmet_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Normalise column names: lowercase, strip whitespace, strip accents,
    # replace spaces with underscores. INMET headers contain "direção",
    # "pressão", etc., which would otherwise miss our ASCII candidates.
    import unicodedata
    def _norm(s: str) -> str:
        nf = unicodedata.normalize("NFKD", s)
        return "".join(c for c in nf if not unicodedata.combining(c)).lower().strip().replace(" ", "_")
    df.columns = [_norm(c) for c in df.columns]

    def _find_col(candidates):
        for cand in candidates:
            for c in df.columns:
                if cand in c:
                    return c
        return None

    date_col = _find_col(["data_(yyyy-mm-dd)", "data", "date"])
    spd_col = _find_col(["velocidade_horaria", "wind_speed", "velocidade"])
    dir_col = _find_col(["direcao_horaria", "wind_direction", "direcao"])

    if not spd_col or not dir_col:
        raise ValueError(
            f"Could not find wind speed / direction columns in {csv_path}. "
            f"Columns found: {list(df.columns)}"
        )

    # INMET changed the date format around 2019: pre-2019 uses
    # ISO "YYYY-MM-DD"; 2019+ uses Portuguese "YYYY/MM/DD". Normalise
    # before parsing so concat'd files spanning the change still work.
    def _parse_dates(s):
        s = s.astype(str).str.replace("/", "-", regex=False)
        return pd.to_datetime(s, errors="coerce")

    if date_col and (year_start or year_end):
        df[date_col] = _parse_dates(df[date_col])
        if year_start:
            df = df[df[date_col].dt.year >= year_start]
        if year_end:
            df = df[df[date_col].dt.year <= year_end]

    tw_start = tw_end = None
    if date_col and len(df) > 0:
        dates = _parse_dates(df[date_col]).dropna()
        if len(dates) > 0:
            tw_start = dates.min().date().isoformat()
            tw_end = dates.max().date().isoformat()

    df[spd_col] = pd.to_numeric(df[spd_col], errors="coerce")
    df[dir_col] = pd.to_numeric(df[dir_col], errors="coerce")

    # Drop rows with no speed observation at all (different from calm)
    df = df.dropna(subset=[spd_col])
    n_total = len(df)

    # Calm: speed < 0.5 m/s OR direction NaN (INMET convention)
    calm_mask = (df[spd_col] < 0.5) | df[dir_col].isna()
    n_calm = int(calm_mask.sum())
    active = df[~calm_mask].copy()

    active["cardinal"] = active[dir_col].apply(deg_to_cardinal)
    freq = active["cardinal"].value_counts(normalize=True).to_dict()
    speed = active.groupby("cardinal")[spd_col].mean().to_dict()

    frequencies = {d: float(freq.get(d, 0.0)) for d in WIND_DIRECTIONS_8}
    mean_speeds = {d: float(speed.get(d, 0.0)) for d in WIND_DIRECTIONS_8}

    period_str = ""
    if tw_start and tw_end:
        period_str = f" {tw_start[:4]}–{tw_end[:4]}"

    source = (
        f"INMET {station_name}{period_str}; n={n_total:,} obs "
        f"({n_calm:,} calm below 0.5 m/s, excluded from direction "
        f"frequencies)"
    )

    return {
        "frequencies": frequencies,
        "mean_speeds": mean_speeds,
        "source": source,
        "reference_height_m": 10.0,   # INMET anemometer standard
        "station_id": station_id,
        "station_name": station_name,
        "station_coords": list(station_coords) if station_coords else None,
        "time_window_start": tw_start,
        "time_window_end": tw_end,
        "n_observations": n_total,
        "calm_fraction": float(n_calm / n_total) if n_total else None,
        "quality_flag": "measured",
    }


def from_iowa_asos_csv(
    csv_path: Path,
    station_id: str,
    station_name: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> dict:
    """Build a wind rose from an Iowa State ASOS archive CSV (METAR).

    Download URL pattern:
        https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
            ?station={ID}&data=drct&data=sknt
            &year1=2015&month1=1&day1=1
            &year2=2025&month2=1&day2=1
            &tz=Etc/UTC&format=onlycomma&latlon=yes&missing=M&trace=T

    Columns produced by Iowa ASOS:
        station, valid (UTC), lon, lat, drct (degrees), sknt (knots)

    Missing values encoded as 'M'. Speed in knots → convert to m/s.
    Anemometer height is the METAR standard 10 m for airport stations.
    """
    df = pd.read_csv(csv_path, na_values=["M"])
    required = {"valid", "drct", "sknt"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Iowa ASOS CSV missing expected columns {required}. "
            f"Got: {sorted(df.columns)}"
        )

    # Station coords carried per-row in the ASOS schema
    lat = float(df["lat"].iloc[0]) if "lat" in df.columns and len(df) else None
    lon = float(df["lon"].iloc[0]) if "lon" in df.columns and len(df) else None
    station_coords = (lat, lon) if lat is not None and lon is not None else None

    df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
    if year_start:
        df = df[df["valid"].dt.year >= year_start]
    if year_end:
        df = df[df["valid"].dt.year <= year_end]

    tw_start = tw_end = None
    if len(df) > 0:
        tw_start = df["valid"].min().date().isoformat()
        tw_end = df["valid"].max().date().isoformat()

    df["sknt"] = pd.to_numeric(df["sknt"], errors="coerce")
    df["drct"] = pd.to_numeric(df["drct"], errors="coerce")
    df["u_ms"] = df["sknt"] * 0.514444  # knots → m/s

    df = df.dropna(subset=["u_ms"])
    n_total = len(df)

    # Calm: |U| < 0.5 m/s OR direction missing
    calm_mask = (df["u_ms"] < 0.5) | df["drct"].isna()
    n_calm = int(calm_mask.sum())
    active = df[~calm_mask].copy()

    active["cardinal"] = active["drct"].apply(deg_to_cardinal)
    freq = active["cardinal"].value_counts(normalize=True).to_dict()
    speed = active.groupby("cardinal")["u_ms"].mean().to_dict()

    frequencies = {d: float(freq.get(d, 0.0)) for d in WIND_DIRECTIONS_8}
    mean_speeds = {d: float(speed.get(d, 0.0)) for d in WIND_DIRECTIONS_8}

    period_str = ""
    if tw_start and tw_end:
        period_str = f" {tw_start[:4]}–{tw_end[:4]}"

    source = (
        f"Iowa ASOS {station_name}{period_str}; n={n_total:,} obs "
        f"({n_calm:,} calm below 0.5 m/s or null direction, excluded)"
    )

    return {
        "frequencies": frequencies,
        "mean_speeds": mean_speeds,
        "source": source,
        "reference_height_m": 10.0,
        "station_id": station_id,
        "station_name": station_name,
        "station_coords": list(station_coords) if station_coords else None,
        "time_window_start": tw_start,
        "time_window_end": tw_end,
        "n_observations": n_total,
        "calm_fraction": float(n_calm / n_total) if n_total else None,
        "quality_flag": "measured",
    }


def from_template(site: str) -> dict:
    """Return the Rio-coastal climatological prior, enriched with the
    site-specific station recommendation + exposure class + the expected
    adjustment when real data arrives.

    All frequency/speed values are the prior; the differentiation is in
    the metadata. Quality flag = 'placeholder-prior' to block accidental
    use in annualised metrics.
    """
    profile = SITE_PROFILES.get(site)
    if profile is None:
        raise ValueError(
            f"Unknown site: {site}. Known sites: {list(SITE_PROFILES)}"
        )
    source = (
        f"{PRIOR_SOURCE} Site profile: {profile.exposure_class}. "
        f"Recommended station: {profile.recommended_station_name} "
        f"({profile.recommended_station_id}). "
        f"Expected adjustment on real data: {profile.expected_adjustment}"
    )
    return {
        "frequencies": dict(PRIOR_FREQUENCIES),
        "mean_speeds": dict(PRIOR_MEAN_SPEEDS),
        "source": source,
        "reference_height_m": 10.0,
        "station_id": profile.recommended_station_id,
        "station_name": profile.recommended_station_name,
        "station_coords": (
            list(profile.recommended_station_coords)
            if profile.recommended_station_coords else None
        ),
        "time_window_start": None,
        "time_window_end": None,
        "n_observations": None,
        "calm_fraction": None,
        "quality_flag": "placeholder-prior",
    }


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────


def _validate_rose(payload: dict) -> None:
    """Sanity-check a rose payload before writing."""
    f = payload["frequencies"]
    assert set(f) == set(WIND_DIRECTIONS_8), (
        f"frequencies must cover all 8 directions; got {sorted(f)}"
    )
    total = sum(f.values())
    assert abs(total - 1.0) < 1e-6, (
        f"frequencies must sum to 1.0; got {total:.6f}"
    )
    assert payload["reference_height_m"] is not None, (
        "reference_height_m is required (nominally 10 m for INMET)"
    )
    assert payload["quality_flag"] in (
        "measured", "gap-filled", "placeholder-prior",
    ), f"invalid quality_flag: {payload['quality_flag']}"


def write_wind_rose(
    site: str,
    rose_data: dict,
    output_dir: Optional[Path] = None,
) -> Path:
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
        "reference_height_m": rose_data.get("reference_height_m"),
        "station_id": rose_data.get("station_id"),
        "station_name": rose_data.get("station_name"),
        "station_coords": rose_data.get("station_coords"),
        "time_window_start": rose_data.get("time_window_start"),
        "time_window_end": rose_data.get("time_window_end"),
        "n_observations": rose_data.get("n_observations"),
        "calm_fraction": rose_data.get("calm_fraction"),
        "quality_flag": rose_data.get("quality_flag"),
    }
    _validate_rose(payload)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(
        "Wrote %s [%s, station=%s]",
        path, payload["quality_flag"], payload["station_id"],
    )
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build wind rose JSON for CFD simulations",
    )
    parser.add_argument(
        "--site", required=True,
        choices=list(SITE_PROFILES) + ["all"],
    )
    parser.add_argument(
        "--inmet-csv", type=Path, default=None,
        help="INMET observations CSV (preferred — real measured data)",
    )
    parser.add_argument(
        "--asos-csv", type=Path, default=None,
        help="Iowa ASOS METAR CSV (for Maré / SBGL Galeão).",
    )
    parser.add_argument(
        "--station-id", default=None,
        help="INMET station code (e.g., A652). Defaults to the site's "
             "recommended station from SITE_PROFILES.",
    )
    parser.add_argument(
        "--station-name", default=None,
        help="INMET station name. Defaults to the site's recommendation.",
    )
    parser.add_argument(
        "--station-lat", type=float, default=None,
        help="Station latitude (decimal degrees).",
    )
    parser.add_argument(
        "--station-lon", type=float, default=None,
        help="Station longitude (decimal degrees).",
    )
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument(
        "--from-template", action="store_true",
        help="Use the Rio climatological prior (flagged placeholder-prior).",
    )
    args = parser.parse_args()

    if not any([args.inmet_csv, args.asos_csv, args.from_template]):
        parser.error(
            "Must provide --inmet-csv <path>, --asos-csv <path>, or --from-template"
        )

    sites = list(SITE_PROFILES) if args.site == "all" else [args.site]

    for site in sites:
        profile = SITE_PROFILES[site]
        station_id = args.station_id or profile.recommended_station_id
        station_name = args.station_name or profile.recommended_station_name
        if args.inmet_csv:
            coords = None
            if args.station_lat is not None and args.station_lon is not None:
                coords = (args.station_lat, args.station_lon)
            elif profile.recommended_station_coords is not None:
                coords = profile.recommended_station_coords
            rose = from_inmet_csv(
                args.inmet_csv,
                station_id=station_id,
                station_name=station_name,
                station_coords=coords,
                year_start=args.year_start,
                year_end=args.year_end,
            )
        elif args.asos_csv:
            rose = from_iowa_asos_csv(
                args.asos_csv,
                station_id=station_id,
                station_name=station_name,
                year_start=args.year_start,
                year_end=args.year_end,
            )
        else:
            rose = from_template(site)
        write_wind_rose(site, rose)


if __name__ == "__main__":
    main()
