"""Fixed-site temperature loggers as a background reference for the OM2 walks.

Every walk runs the same direction, so removing a per-walk linear time trend
from the walking temperatures would also remove the along-route gradient. A
fixed logger sees only the time axis, so subtracting its trend removes the
common temporal drift and leaves the spatial signal.

Files are named <I|O>_<device>_<YYYYMMDD>_<hh>durhrs.csv (I indoor, O outdoor)
with columns Timestamp (Rio local time, see load_loggers), Latitude, Longitude, Temperature, Humidity, PM*.
Loading reduces each device to one-minute means: walks last about 27 minutes,
so finer time resolution adds size and no information.

Combining outdoor devices: each device carries a constant offset from its own
site and sensor calibration. The offset is estimated as the median, over the
minutes where at least two outdoor devices run, of the device minus the mean of
the running outdoor devices. It is subtracted before the cross-device median.
Without this step the series would jump whenever a device starts or stops.
The result is on the common outdoor level, not on any one device's absolute
scale, which is all a drift reference needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Timestamps before this are logger clock resets to the 2000-01-01 epoch.
MIN_VALID_TIME = pd.Timestamp("2020-01-01", tz="UTC")
#: Plausible air temperature range for Rio de Janeiro street loggers, deg C.
TEMP_RANGE = (5.0, 60.0)
#: Latitude/longitude box around Rio; (0, 0) means no GPS fix.
LAT_RANGE = (-23.1, -22.7)
LON_RANGE = (-43.5, -43.0)
#: A query time has no background when no logger minute lies within this window.
MAX_GAP = pd.Timedelta(minutes=10)

COLUMNS = ["t_utc", "device", "kind", "temperature", "humidity"]


def _parse_time(raw: pd.Series, tz: str) -> pd.Series:
    t = pd.to_datetime(raw, format="ISO8601", errors="coerce")
    if t.isna().any():
        t = t.fillna(pd.to_datetime(raw[t.isna()], format="mixed", errors="coerce"))
    return t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def parse_name(path: Path) -> tuple[str, str]:
    """('I_1', 'indoor') from I_1_20251205_11durhrs.csv."""
    parts = Path(path).stem.split("_")
    kind = {"I": "indoor", "O": "outdoor"}[parts[0]]
    return f"{parts[0]}_{parts[1]}", kind


def _read_one(path: Path, tz: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    device, kind = parse_name(path)
    raw = pd.read_csv(path, usecols=["Timestamp", "Latitude", "Longitude", "Temperature", "Humidity"])
    n_raw = len(raw)
    raw["t"] = _parse_time(raw["Timestamp"], tz)
    ok = (
        raw["t"].notna()
        & (raw["t"] >= MIN_VALID_TIME)
        & raw["Temperature"].between(*TEMP_RANGE)
    )
    raw = raw[ok]
    minute = raw["t"].dt.floor("min")
    g = raw.groupby(minute).agg(temperature=("Temperature", "mean"), humidity=("Humidity", "mean"))
    g = g.rename_axis("t_utc").reset_index()
    g["device"], g["kind"] = device, kind
    fix = raw[raw["Latitude"].between(*LAT_RANGE) & raw["Longitude"].between(*LON_RANGE)]
    return g[COLUMNS], fix[["Latitude", "Longitude"]].assign(device=device, kind=kind), n_raw - len(raw)


def load_loggers(folder: Path, tz: str = "UTC") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(minutes, locations).

    minutes: one row per device and UTC minute, columns COLUMNS, junk dropped
    (epoch resets, unparsable times, temperatures outside TEMP_RANGE).
    locations: per device, median GPS fix over valid fixes (the median shrugs
    off the occasional wild fix) plus the 5th to 95th percentile spread in
    metres, and the number of dropped junk rows in attrs["n_dropped"].

    tz is the clock the Timestamp column is written in. The fixed loggers write
    Rio local time (America/Sao_Paulo): cross-correlation with Galeão airport
    temperature peaks at a 3 h shift and their daily peak (13:00 file clock)
    matches the airport's 16:00 UTC peak. The walk files, unlike these, are UTC.
    """
    paths = sorted(Path(folder).glob("[IO]_*_*durhrs.csv"))
    if not paths:
        raise FileNotFoundError(f"no logger files in {folder}")
    mins, fixes, dropped = [], [], 0
    for p in paths:
        m, f, d = _read_one(p, tz)
        mins.append(m)
        fixes.append(f)
        dropped += d
    minutes = pd.concat(mins, ignore_index=True)
    minutes = (
        minutes.groupby(["device", "kind", "t_utc"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["device", "t_utc"], ignore_index=True)
    )
    fx = pd.concat(fixes, ignore_index=True)
    rows = []
    for (dev, kind), s in fx.groupby(["device", "kind"]):
        lat0, lon0 = s["Latitude"].median(), s["Longitude"].median()
        dy = (s["Latitude"] - lat0) * 111_320.0
        dx = (s["Longitude"] - lon0) * 111_320.0 * np.cos(np.radians(lat0))
        r = np.hypot(dx, dy)
        rows.append({"device": dev, "kind": kind, "lat": lat0, "lon": lon0,
                     "spread_p95_m": float(r.quantile(0.95)), "n_fixes": len(s)})
    loc = pd.DataFrame(rows)
    loc.attrs["n_dropped"] = dropped
    return minutes, loc


def _wide(minutes: pd.DataFrame, kind: str) -> pd.DataFrame:
    sub = minutes[minutes["kind"] == kind]
    return sub.pivot(index="t_utc", columns="device", values="temperature").sort_index()


def device_offsets(minutes: pd.DataFrame, kind: str = "outdoor") -> pd.Series:
    """Constant offset of each device from the mean of the running devices of its kind.

    Needs minutes with at least two devices running; a device with no overlap
    gets offset 0 (nothing to compare it with).
    """
    w = _wide(minutes, kind)
    multi = w.notna().sum(axis=1) >= 2
    ref = w[multi].mean(axis=1)
    off = (w[multi].sub(ref, axis=0)).median()
    return off.reindex(w.columns).fillna(0.0)


def reference_series(minutes: pd.DataFrame, kind: str = "outdoor") -> pd.Series:
    """Offset-corrected cross-device median per UTC minute (index t_utc)."""
    w = _wide(minutes, kind)
    return w.sub(device_offsets(minutes, kind), axis=1).median(axis=1).dropna()


def background_temperature(t_utc, minutes: pd.DataFrame, kind: str = "outdoor") -> np.ndarray:
    """Background temperature at each time, linearly interpolated between minutes.

    NaN where no logger minute of that kind lies within MAX_GAP on both sides
    of the time (so an interpolation never bridges a long outage), or where the
    time falls outside the logger record.
    """
    ref = reference_series(minutes, kind)
    t = pd.to_datetime(pd.Series(t_utc), utc=True)
    q = t.astype("int64").to_numpy().astype(float)
    x = ref.index.astype("int64").to_numpy().astype(float)
    out = np.interp(q, x, ref.to_numpy(), left=np.nan, right=np.nan)
    gap = MAX_GAP.value
    i = np.searchsorted(x, q)
    lo = x[np.clip(i - 1, 0, len(x) - 1)]
    hi = x[np.clip(i, 0, len(x) - 1)]
    bridged = ((q - lo) <= gap) & ((hi - q) <= gap)
    return np.where(bridged, out, np.nan)


def coverage_share(start_utc, end_utc, minutes: pd.DataFrame, kind: str = "outdoor") -> float:
    """Share of the minutes in [start, end] with at least one logger minute of that kind."""
    grid = pd.date_range(pd.Timestamp(start_utc).floor("min"), pd.Timestamp(end_utc).floor("min"), freq="min", tz="UTC")
    have = pd.DatetimeIndex(minutes.loc[minutes["kind"] == kind, "t_utc"].unique())
    return float(grid.isin(have).mean())
