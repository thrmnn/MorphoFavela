"""Observed wind at Galeão (SBGL) for the OM2 campaign window.

SBGL is ~3 km from Maré; it is a regional reference, NOT wind measured at
the route. Everything here is airport METAR at 10 m, never an on-site
measurement. Source: Iowa Environmental Mesonet ASOS archive (the same
service and column schema as scripts/build_wind_rose.py
from_iowa_asos_csv, which produced data/maré/wind_rose.json).

The fetched CSV is cached under <root>/data/maré/octopus/wind/ with a
manifest.json (url, fetched UTC, sha256, counts). If the fetch fails,
fetch_sbgl raises: wind is never synthesised.

Clock caveat: METAR times are UTC. Whether the Octopus device clocks log
UTC or Rio local time (fixed UTC-3, no DST since 2019) is UNKNOWN, so
callers of wind_at must pass UTC; ``device_to_utc`` converts once the
clock is known.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import DEFAULT_ROOT

STATION = "SBGL"
WINDOW_START = "2025-12-01"
WINDOW_END = "2026-04-30"
CACHE_STEM = "sbgl_metar_20251201_20260430"
#: knots -> m/s, same factor as scripts/build_wind_rose.py.
KNOT_MS = 0.514444
#: Calm below this speed, as in the climatology (build_wind_rose.py).
CALM_MS = 0.5
MAX_GAP_MIN = 60
LOCAL_UTC_OFFSET_H = -3
ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def wind_dir() -> Path:
    return Path("data") / "maré" / "octopus" / "wind"


def cache_paths(root: Path | str = DEFAULT_ROOT) -> tuple[Path, Path]:
    d = Path(root) / wind_dir()
    return d / f"{CACHE_STEM}.csv", d / "manifest.json"


def _request_url(start: str, end: str) -> str:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end) + pd.Timedelta(days=1)  # end date inclusive
    return (
        f"{ASOS_URL}?station={STATION}&data=drct&data=sknt"
        f"&year1={s.year}&month1={s.month}&day1={s.day}"
        f"&year2={e.year}&month2={e.month}&day2={e.day}"
        "&tz=Etc/UTC&format=onlycomma&latlon=yes&missing=M&trace=T"
        "&report_type=3&report_type=4"
    )


def summarise(df: pd.DataFrame) -> dict:
    spd = pd.to_numeric(df["sknt"], errors="coerce") * KNOT_MS
    drct = pd.to_numeric(df["drct"], errors="coerce")
    calm = spd < CALM_MS
    variable = ~calm & drct.isna() & spd.notna()
    return {
        "n_obs": int(len(df)),
        "n_calm": int(calm.sum()),
        "n_variable_or_missing_direction": int(variable.sum()),
        "n_missing_speed": int(spd.isna().sum()),
    }


def fetch_sbgl(root: Path | str = DEFAULT_ROOT, start: str = WINDOW_START, end: str = WINDOW_END,
               timeout_s: int = 120) -> dict:
    """Download the window from Iowa ASOS and cache CSV + manifest. Raises
    on any network/parse failure; writes nothing in that case."""
    import urllib.request

    url = _request_url(start, end)
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        raw = r.read()
    text = raw.decode("utf-8")
    if not text.startswith("station,valid,lon,lat,drct,sknt"):
        raise RuntimeError(f"unexpected ASOS response header: {text[:80]!r}")
    csv_path, manifest_path = cache_paths(root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(raw)
    df = pd.read_csv(csv_path, na_values=["M"])
    manifest = {
        "station": STATION,
        "window_utc": [start, end],
        "url": url,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "csv": csv_path.name,
        "calm_threshold_ms": CALM_MS,
        **summarise(df),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def load_obs(root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """Cached observations: valid_utc (tz-aware), drct, speed_ms, calm, variable."""
    csv_path, _ = cache_paths(root)
    df = pd.read_csv(csv_path, na_values=["M"])
    out = pd.DataFrame({
        "valid_utc": pd.to_datetime(df["valid"], utc=True),
        "drct": pd.to_numeric(df["drct"], errors="coerce"),
        "speed_ms": pd.to_numeric(df["sknt"], errors="coerce") * KNOT_MS,
    })
    out["calm"] = out["speed_ms"] < CALM_MS
    out["variable"] = ~out["calm"] & out["drct"].isna() & out["speed_ms"].notna()
    return out.sort_values("valid_utc").reset_index(drop=True)


def device_to_utc(ts, clock: str) -> pd.Timestamp:
    """Device-clock timestamp -> UTC. clock is 'utc' or 'local' (UTC-3)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        return t.tz_convert("UTC")
    if clock == "utc":
        return t.tz_localize("UTC")
    if clock == "local":
        return (t - pd.Timedelta(hours=LOCAL_UTC_OFFSET_H)).tz_localize("UTC")
    raise ValueError("clock must be 'utc' or 'local'")


def wind_at(timestamp_utc, obs: pd.DataFrame | None = None, root: Path | str = DEFAULT_ROOT,
            max_gap_min: float = MAX_GAP_MIN) -> dict | None:
    """Nearest SBGL observation with a usable direction within max_gap_min
    of timestamp_utc, else None. Calm and variable-direction reports have no
    direction and are skipped, so the result may be older than a calm report
    that is nearer in time. Returns {valid_utc, gap_min, drct, speed_ms}."""
    obs = load_obs(root) if obs is None else obs
    usable = obs[obs["drct"].notna() & ~obs["calm"]]
    if usable.empty:
        return None
    t = pd.Timestamp(timestamp_utc)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    gaps = (usable["valid_utc"] - t).abs()
    i = gaps.idxmin()
    gap_min = gaps[i].total_seconds() / 60.0
    if gap_min > max_gap_min:
        return None
    row = usable.loc[i]
    return {"valid_utc": row["valid_utc"], "gap_min": gap_min,
            "drct": float(row["drct"]), "speed_ms": float(row["speed_ms"])}


def rose(drct: np.ndarray, speed_ms: np.ndarray, n_sectors: int = 16) -> dict:
    """Sector frequencies (sum 1 over non-calm, directional reports) and mean
    speed. Sectors are centred on k*360/n_sectors (N = sector 0)."""
    width = 360.0 / n_sectors
    idx = (((np.asarray(drct, float) % 360.0) + width / 2) // width).astype(int) % n_sectors
    counts = np.bincount(idx, minlength=n_sectors).astype(float)
    sums = np.bincount(idx, weights=np.asarray(speed_ms, float), minlength=n_sectors)
    n = counts.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / counts, np.nan)
    return {
        "centres_deg": (np.arange(n_sectors) * width).tolist(),
        "frequencies": (counts / n if n else counts).tolist(),
        "mean_speed_ms": mean.tolist(),
        "n_directional": int(n),
    }


def campaign_window_rose(obs: pd.DataFrame | None = None, root: Path | str = DEFAULT_ROOT,
                         start=None, end=None, n_sectors: int = 16) -> dict:
    """16-sector rose of the cached window (or [start, end] UTC), same
    exclusions as the climatology: calm (< 0.5 m/s) and missing direction."""
    obs = load_obs(root) if obs is None else obs
    if start is not None:
        obs = obs[obs["valid_utc"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        obs = obs[obs["valid_utc"] <= pd.Timestamp(end, tz="UTC")]
    active = obs[obs["drct"].notna() & ~obs["calm"]]
    out = rose(active["drct"].to_numpy(), active["speed_ms"].to_numpy(), n_sectors)
    out.update({
        "n_obs": int(len(obs)),
        "calm_fraction": float(1 - len(active) / len(obs)) if len(obs) else None,
        "window_utc": [str(obs["valid_utc"].min()), str(obs["valid_utc"].max())] if len(obs) else None,
        "label": "SBGL METAR (observed), 10 m",
    })
    return out


def climatology_rose(root: Path | str = DEFAULT_ROOT, n_sectors: int = 16,
                     year_start: int = 2015, year_end: int = 2024) -> dict:
    """The 2015-2024 SBGL climatology at the same sector count, re-binned
    from the raw ASOS CSV (data/asos/SBGL_2015_2024.csv) that built
    data/maré/wind_rose.json, with identical exclusions."""
    df = pd.read_csv(Path(root) / "data" / "asos" / "SBGL_2015_2024.csv", na_values=["M"])
    t = pd.to_datetime(df["valid"], errors="coerce")
    df = df[(t.dt.year >= year_start) & (t.dt.year <= year_end)]
    spd = pd.to_numeric(df["sknt"], errors="coerce") * KNOT_MS
    drct = pd.to_numeric(df["drct"], errors="coerce")
    ok = spd.notna()
    spd, drct = spd[ok], drct[ok]
    active = (spd >= CALM_MS) & drct.notna()
    out = rose(drct[active].to_numpy(), spd[active].to_numpy(), n_sectors)
    out.update({"n_obs": int(ok.sum()), "calm_fraction": float((~active).mean()),
                "label": f"SBGL {year_start}-{year_end} climatology, 10 m"})
    return out
