"""Two wind regimes found from airport (Galeão) reports, replacing the single
prevailing direction. Regional reference at 10 m, NOT wind at the route.

Primary method: peaks of the circularly smoothed 16-sector rose; each report
goes to the nearer peak. Check method: two-component von Mises mixture (EM).
Times are UTC; local hour = America/Sao_Paulo (UTC-3, no DST).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import i0e

from scripts.build_wind_rose import CALM_MS, KNOT_MS

from .io_utils import DEFAULT_ROOT

N_SECTORS = 16
SECTOR_W = 360.0 / N_SECTORS
COMPASS = ["north", "north-northeast", "northeast", "east-northeast", "east", "east-southeast",
           "southeast", "south-southeast", "south", "south-southwest", "southwest",
           "west-southwest", "west", "west-northwest", "northwest", "north-northwest"]
REGIME_KEYS = ("reg1", "reg2")
REGIME_COLOURS = {"reg1": "#E69F00", "reg2": "#0072B2"}  # Okabe-Ito orange, blue
LOCAL_TZ = "America/Sao_Paulo"
TAG_MAX_GAP_MIN = 60


def compass_name(deg: float) -> str:
    return COMPASS[int(((deg % 360.0) + SECTOR_W / 2) // SECTOR_W) % N_SECTORS]


def circ_dist(a, b):
    return np.abs((np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0)


def circ_mean(deg) -> float:
    r = np.deg2rad(np.asarray(deg, float))
    return float(np.rad2deg(np.arctan2(np.sin(r).sum(), np.cos(r).sum())) % 360.0)


def load_csv(path: Path | str) -> pd.DataFrame:
    """Raw ASOS CSV -> valid_utc, drct, speed_ms, calm, variable (same schema as wind_obs.load_obs)."""
    df = pd.read_csv(path, na_values=["M"])
    out = pd.DataFrame({
        "valid_utc": pd.to_datetime(df["valid"], utc=True),
        "drct": pd.to_numeric(df["drct"], errors="coerce"),
        "speed_ms": pd.to_numeric(df["sknt"], errors="coerce") * KNOT_MS,
    })
    out["calm"] = out["speed_ms"] < CALM_MS
    out["variable"] = ~out["calm"] & out["drct"].isna() & out["speed_ms"].notna()
    return out.sort_values("valid_utc").reset_index(drop=True)


def load_campaign(root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    return load_csv(Path(root) / "data" / "maré" / "octopus" / "wind" / "sbgl_metar_20251201_20260430.csv")


def load_climatology(root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    return load_csv(Path(root) / "data" / "asos" / "SBGL_2015_2024.csv")


def clean(obs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep directional, non-calm reports. Calm = speed < CALM_MS (covers
    sknt == 0 and direction 0 with calm). Missing speed / direction dropped."""
    speed = obs["speed_ms"]
    calm = speed < CALM_MS
    missing_speed = speed.isna()
    variable = ~calm & ~missing_speed & obs["drct"].isna()
    keep = ~calm & ~missing_speed & obs["drct"].notna()
    counts = {"n_reports": int(len(obs)), "n_calm": int(calm.sum()),
              "n_variable_or_missing_direction": int(variable.sum()),
              "n_missing_speed": int(missing_speed.sum()), "n_used": int(keep.sum())}
    return obs[keep].copy(), counts


def rose_peaks(drct: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Sector indices of the two highest local maxima of the 1-2-1 smoothed
    16-sector rose that are >= 2 sectors apart (circularly), plus that rose."""
    d = np.asarray(drct, float)
    idx = (((d % 360.0) + SECTOR_W / 2) // SECTOR_W).astype(int) % N_SECTORS
    f = np.bincount(idx, minlength=N_SECTORS).astype(float)
    f /= f.sum()
    s = 0.25 * np.roll(f, 1) + 0.5 * f + 0.25 * np.roll(f, -1)
    cand = [i for i in range(N_SECTORS)
            if s[i] >= s[(i - 1) % N_SECTORS] and s[i] >= s[(i + 1) % N_SECTORS]]
    cand.sort(key=lambda i: -s[i])
    first = cand[0]
    for j in cand[1:]:
        if min((j - first) % N_SECTORS, (first - j) % N_SECTORS) >= 2:
            return first, j, s
    raise ValueError("fewer than two separated peaks in the smoothed rose")


def _vm_logpdf(x, mu, kappa):
    return kappa * (np.cos(x - mu) - 1.0) - np.log(2 * np.pi * i0e(kappa))


def vonmises_mixture(drct: np.ndarray, init_deg: tuple[float, float], n_iter: int = 500,
                     tol: float = 1e-9) -> dict:
    """Two-component von Mises mixture by EM (kappa via the Banerjee approximation)."""
    x = np.deg2rad(np.asarray(drct, float))
    mu = np.deg2rad(np.array(init_deg, float))
    kappa = np.array([2.0, 2.0])
    w = np.array([0.5, 0.5])
    prev = -np.inf
    for _ in range(n_iter):
        lp = np.stack([np.log(w[k]) + _vm_logpdf(x, mu[k], kappa[k]) for k in range(2)])
        m = lp.max(axis=0)
        ll = float((m + np.log(np.exp(lp - m).sum(axis=0))).sum())
        r = np.exp(lp - m)
        r /= r.sum(axis=0)
        for k in range(2):
            nk = r[k].sum()
            w[k] = nk / len(x)
            c, s = (r[k] * np.cos(x)).sum(), (r[k] * np.sin(x)).sum()
            mu[k] = np.arctan2(s, c)
            R = min(np.hypot(c, s) / nk, 0.999)
            kappa[k] = R * (2 - R ** 2) / (1 - R ** 2)
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll
    return {"mean_direction_deg": (np.rad2deg(mu) % 360.0).tolist(), "weights": w.tolist(),
            "kappa": kappa.tolist(), "log_likelihood": ll}


def find_regimes(obs: pd.DataFrame) -> dict:
    """Two regimes from the rose peaks, with the von Mises mixture as a check.
    Regimes are ordered by share; keys are reg1/reg2 in that order (use
    assign_keys to impose the campaign-season keys on another period)."""
    act, counts = clean(obs)
    d = act["drct"].to_numpy(float)
    spd = act["speed_ms"].to_numpy(float)
    p1, p2, _ = rose_peaks(d)
    peaks = np.array([p1 * SECTOR_W, p2 * SECTOR_W])
    member = np.argmin(np.stack([circ_dist(d, peaks[0]), circ_dist(d, peaks[1])]), axis=0)
    regimes = []
    for k in range(2):
        sel = member == k
        md = circ_mean(d[sel])
        regimes.append({
            "name": compass_name(md), "mean_direction_deg": md,
            "share_of_reports": float(sel.mean()), "mean_speed_ms": float(spd[sel].mean()),
            "n_reports": int(sel.sum()), "peak_deg": float(peaks[k]),
        })
    regimes.sort(key=lambda g: -g["share_of_reports"])
    for key, g in zip(REGIME_KEYS, regimes):
        g["key"] = key
    raw = vonmises_mixture(d, tuple(peaks))
    order = sorted(range(2), key=lambda k: -raw["weights"][k])
    mix = {k: [raw[k][i] for i in order] for k in ("mean_direction_deg", "weights", "kappa")}
    mix["log_likelihood"] = raw["log_likelihood"]
    mix["difference_deg"] = [
        float(min(circ_dist(md, g["mean_direction_deg"]) for g in regimes))
        for md in mix["mean_direction_deg"]]
    mix["max_difference_deg"] = max(mix["difference_deg"])
    n_all = counts["n_calm"] + counts["n_used"] + counts["n_variable_or_missing_direction"]
    return {"regimes": regimes, "mixture": mix, "counts": counts,
            "calm_share": counts["n_calm"] / max(n_all, 1)}


def assign_keys(result: dict, reference: dict) -> dict:
    """Relabel result's regimes with the reference's keys by nearest mean
    direction, so climatology reg1 is the same regime as campaign reg1."""
    ref = reference["regimes"]
    a, b = result["regimes"]
    straight = circ_dist(a["mean_direction_deg"], ref[0]["mean_direction_deg"]) + \
        circ_dist(b["mean_direction_deg"], ref[1]["mean_direction_deg"])
    crossed = circ_dist(a["mean_direction_deg"], ref[1]["mean_direction_deg"]) + \
        circ_dist(b["mean_direction_deg"], ref[0]["mean_direction_deg"])
    if crossed < straight:
        a["key"], b["key"] = ref[1]["key"], ref[0]["key"]
    else:
        a["key"], b["key"] = ref[0]["key"], ref[1]["key"]
    result["regimes"] = sorted(result["regimes"], key=lambda g: g["key"])
    return result


def classify(drct, regimes: dict) -> np.ndarray:
    """Regime key per direction: nearer peak by circular distance."""
    reg = regimes["regimes"]
    d = np.asarray(drct, float)
    dist = np.stack([circ_dist(d, g["peak_deg"]) for g in reg])
    return np.array([g["key"] for g in reg])[np.argmin(dist, axis=0)]


def hourly_frequency(obs: pd.DataFrame, regimes: dict) -> pd.DataFrame:
    """Share of each regime and of calm by local hour (rows 0-23; columns
    reg1, reg2, calm sum to 1). Variable/missing-direction reports excluded."""
    o = obs[obs["speed_ms"].notna()]
    calm = o["speed_ms"] < CALM_MS
    o, calm = o[calm | o["drct"].notna()], calm[calm | o["drct"].notna()]
    label = pd.Series("calm", index=o.index, dtype=object)
    label[~calm] = classify(o.loc[~calm, "drct"].to_numpy(), regimes)
    hour = o["valid_utc"].dt.tz_convert(LOCAL_TZ).dt.hour
    tab = pd.crosstab(hour, label).reindex(index=range(24), columns=[*REGIME_KEYS, "calm"], fill_value=0)
    tab.index.name = "local_hour"
    return tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0)


def tag_walks(walks: pd.DataFrame, obs: pd.DataFrame, regimes: dict,
              max_gap_min: float = TAG_MAX_GAP_MIN) -> pd.DataFrame:
    """Nearest airport report to each walk's mid time. regime = regime name,
    or 'calm' (nearest report calm) / 'none' (nearest report more than
    max_gap_min away, or direction missing)."""
    name = {g["key"]: g["name"] for g in regimes["regimes"]}
    ref = obs.sort_values("valid_utc").reset_index(drop=True)
    rows = []
    for _, w in walks.iterrows():
        mid = pd.Timestamp(w["mid_utc"])
        mid = mid.tz_localize("UTC") if mid.tzinfo is None else mid.tz_convert("UTC")
        gaps = (ref["valid_utc"] - mid).dt.total_seconds() / 60.0
        i = gaps.abs().idxmin()
        r, gap = ref.loc[i], float(gaps[i])
        calm = bool(r["speed_ms"] < CALM_MS)
        if abs(gap) > max_gap_min:
            reg, dirn = "none", np.nan
        elif calm:
            reg, dirn = "calm", np.nan
        elif pd.isna(r["drct"]):
            reg, dirn = "none", np.nan
        else:
            reg, dirn = name[classify([r["drct"]], regimes)[0]], float(r["drct"])
        rows.append({"walk_id": w["walk_id"], "report_time_utc": r["valid_utc"], "regime": reg,
                     "direction_deg": dirn, "speed_ms": float(r["speed_ms"]), "minutes_from_mid": gap})
    return pd.DataFrame(rows)


def season_regimes(root: Path | str = DEFAULT_ROOT) -> dict:
    """Campaign season and 2015-2024 climatology; climatology keys follow the campaign's."""
    camp = find_regimes(load_campaign(root))
    clim = assign_keys(find_regimes(load_climatology(root)), camp)
    return {"campaign": camp, "climatology": clim}
