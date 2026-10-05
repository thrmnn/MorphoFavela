"""First look: street measures paired with the OM2 walk temperature readings.

Input to Jingxue's analysis (she leads the study): descriptive associations
only. Without temperature the sensor-matched columns are a moving average;
with it they let us estimate the sensor's effective time constant and see
which street measures track temperature.

Clocks. Walk timestamps are UTC. The fixed loggers write Rio local time
(their cross-correlation with Galeão airport temperature peaks at a 3 h
shift), so they are loaded with LOGGER_TZ.

Pipeline (run):
  1. readings: matched GPS fixes with a temperature, on route, snapped to the
     nearest 1 m point; points whose class is not street/projected (or, in an
     older package, flagged points) are dropped.
  2. warm-up: minutes since walk start explain the temperature minus logger
     background, after the along-route profile and the walk offset are taken
     out (two-way fixed effects, identified by differences in walking speed
     and start position). The window ends at the first minute after which
     every minute effect stays within WARMUP_TOL_C of zero.
  3. anomaly: temperature minus logger background minus walk offset; walks
     without logger cover fall back to a per-walk linear time detrend.
     The detrended version of every walk is kept for comparison: every walk
     runs 0 m to the end, so a time detrend also removes any along-route
     gradient.
  4. time constant: (a) explained within-walk variance of the anomaly by
     sensor-matched shade, 1 h sun dose and sky view factor over TAU_SCAN_S;
     (b) exponential fit to the mean response around sharp sun/shade
     transitions.
  5. associations at the chosen time constant: walk fixed effects, standard
     errors clustered by walk, leave-one-walk-out R2; 20 m segment means.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import fixed_loggers as fl
from .sensor_match import sensor_matched
from .walks import _walk_ids, load_walks

LOCAL_TZ = "America/Sao_Paulo"
LOGGER_TZ = "America/Sao_Paulo"
PERIODS = ("morning", "evening")
KEEP_CLASSES = ("street", "projected")
#: tau 0 is the 1 m value at the reading's point (no smoothing).
TAU_SCAN_S = (0, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300)
MEASURES = {"shade": "shaded_at_arrival", "dose": "dose_1h_before_wh_m2", "svf": "sky_view_factor"}
EXTRA = {"hw": "height_width_ratio"}
#: A walk counts as covered by the loggers at or above this share of its minutes.
FULL_COVER = 0.999
#: Warm-up: minute effects tested from 0 to WARMUP_MAX_MIN - 1; later minutes are the reference.
WARMUP_MAX_MIN = 15
WARMUP_TOL_C = 0.1
WARMUP_DIST_BIN_M = 50.0
#: Transition events: shade state constant at least this far on each side.
EVENT_SIDE_M = 30
#: Faster than this is a GPS jump, not walking: such runs are not events.
EVENT_MAX_SPEED_MS = 2.0
EVENT_WINDOW_S = (-30, 120)
EVENT_BIN_S = 5
#: Response bins are fitted only while at least this share of the events still contribute; later bins
#: hold only the longest stretches and are not comparable.
EVENT_MIN_SHARE = 0.5
#: The size of the response is summarised as the mean change in this window after the transition.
EVENT_CHANGE_WINDOW_S = (20, 35)
SEGMENT_M = 20
SEGMENT_MIN_READINGS = 30
#: Time constant for the association models. Fixed in advance rather than tuned on the response: neither
#: estimate pins tau down (report text); 30 s lies inside the event-based interval and is a shipped column.
ASSOC_TAU_S = 30
N_BOOT = 400
SEED = 20261005
#: Effect units: shade 0 to 1 (fully sunlit to fully shaded recent path), dose per 100 Wh/m2, svf per 0.1, ratio per 1.
UNITS = {"shade": 1.0, "dose": 100.0, "svf": 0.1, "hw": 1.0}


# 1 ---------------------------------------------------------------------------

def load_temperature(matched_dir: Path) -> pd.DataFrame:
    paths = sorted(Path(matched_dir).glob("OM_2_*.csv"))
    out = []
    for p, w in zip(paths, _walk_ids(paths)):
        d = pd.read_csv(p, usecols=["timestamp_utc", "Temperature"])
        d["t_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True, format="ISO8601", errors="coerce")
        d = d.dropna(subset=["t_utc", "Temperature"]).drop_duplicates("t_utc")
        out.append(pd.DataFrame({"walk_id": w, "t_utc": d["t_utc"], "temperature": d["Temperature"]}))
    return pd.concat(out, ignore_index=True)


def keep_mask(points: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "point_class" in points.columns:
        return points["point_class"].isin(KEEP_CLASSES).to_numpy(), "point_class"
    return ~points["route_geometry_flag"].astype(bool).to_numpy(), "route_geometry_flag"


def readings_table(points: pd.DataFrame, fixes: pd.DataFrame, temps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Matched on-route fixes with a temperature, snapped to the nearest point."""
    f = fixes[fixes["match_status"] == "matched"].merge(temps, on=["walk_id", "t_utc"], how="inner")
    p = points.sort_values("distance_along_m").reset_index(drop=True)
    d = p["distance_along_m"].to_numpy(float)
    q = f["distance_along_m"].to_numpy(float)
    i = np.clip(np.searchsorted(d, q), 1, len(d) - 1)
    i = np.where(np.abs(q - d[i - 1]) <= np.abs(d[i] - q), i - 1, i)
    keep, basis = keep_mask(p)
    r = pd.DataFrame({
        "walk_id": f["walk_id"].to_numpy(),
        "period": f["walk_id"].str.split("_").str[2].to_numpy(),
        "t_utc": f["t_utc"].to_numpy(),
        "distance_along_m": q,
        "point_id": p["point_id"].to_numpy()[i],
        "temperature": f["temperature"].to_numpy(float),
    })
    r["t_utc"] = pd.to_datetime(r["t_utc"], utc=True)
    r["t_local"] = r["t_utc"].dt.tz_convert(LOCAL_TZ)
    n_all = len(r)
    r = r[keep[i]].reset_index(drop=True)
    return r, {"n_fixes_on_route": int(len(fixes)), "n_matched_with_temperature": n_all,
               "n_dropped_off_street": n_all - len(r), "n_readings": len(r), "keep_basis": basis,
               "n_walks_readings": int(r["walk_id"].nunique())}


# 2-3 -------------------------------------------------------------------------

def add_background(r: pd.DataFrame, walks: pd.DataFrame, minutes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    r = r.copy()
    r["background"] = fl.background_temperature(r["t_utc"], minutes)
    cover = {w.walk_id: fl.coverage_share(w.start_utc, w.end_utc, minutes) for w in walks.itertuples()}
    full = {w for w, c in cover.items() if c >= FULL_COVER}
    r["logger_covered"] = r["walk_id"].isin(full) & r.groupby("walk_id")["background"].transform(lambda s: s.notna().all())
    start = walks.set_index("walk_id")["start_utc"]
    r["minutes_since_start"] = (r["t_utc"] - r["walk_id"].map(start)).dt.total_seconds() / 60.0
    n_w = len(walks)
    return r, {"n_walks": n_w, "n_walks_logger_full": len(full),
               "n_walks_logger_any": int(sum(c > 0 for c in cover.values())),
               "n_walks_covered_in_readings": int(r.loc[r["logger_covered"], "walk_id"].nunique())}


def _onehot(codes: np.ndarray, n: int, drop_ref: int | None = None) -> np.ndarray:
    m = np.zeros((len(codes), n))
    ok = codes >= 0
    m[np.flatnonzero(ok), codes[ok]] = 1.0
    if drop_ref is not None:
        m = np.delete(m, drop_ref, axis=1)
    return m


def warmup_effects(r: pd.DataFrame, walk_labels: np.ndarray | None = None) -> np.ndarray:
    """Minute effects (deg C, minute 0 to WARMUP_MAX_MIN - 1) of temperature minus background,
    net of walk offsets and a per-period profile in WARMUP_DIST_BIN_M bins."""
    y = (r["temperature"] - r["background"]).to_numpy(float)
    walk = pd.factorize(r["walk_id"] if walk_labels is None else walk_labels)[0]
    db = np.floor(r["distance_along_m"].to_numpy(float) / WARMUP_DIST_BIN_M).astype(int)
    pd_code = pd.factorize(r["period"].astype(str) + "_" + db.astype(str))[0]
    minute = np.floor(r["minutes_since_start"].to_numpy(float)).astype(int)
    mcode = np.where((minute >= 0) & (minute < WARMUP_MAX_MIN), minute, -1)
    X = np.hstack([_onehot(walk, walk.max() + 1), _onehot(pd_code, pd_code.max() + 1, drop_ref=0),
                   _onehot(mcode, WARMUP_MAX_MIN)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[-WARMUP_MAX_MIN:]


def _settle(t, amp, tau):
    return amp * np.exp(-t / tau)


def fit_settling(curve: pd.Series, n: pd.Series) -> tuple[float, float, int]:
    """Exponential settling amp * exp(-t / tau) fitted to the per-minute curve (minute centres, weighted by
    readings). Window: minutes until the fitted departure falls below WARMUP_TOL_C (0 if it never exceeds it)."""
    from scipy.optimize import curve_fit

    t = curve.index.to_numpy(float) + 0.5
    y = curve.to_numpy(float)
    w = np.sqrt(n.reindex(curve.index).to_numpy(float))
    try:
        (amp, tau), _ = curve_fit(_settle, t, y, p0=(y[0] if abs(y[0]) > 0.01 else 0.1, 3.0),
                                  sigma=1 / w, bounds=([-5.0, 0.2], [5.0, 60.0]), maxfev=10000)
    except RuntimeError:
        return np.nan, np.nan, 0
    return float(amp), float(tau), settle_window(amp, tau)


def settle_window(amp: float, tau: float, tol: float = WARMUP_TOL_C) -> int:
    """Capped at WARMUP_MAX_MIN: the walk's level after that is the reference, so nothing later is visible."""
    if not np.isfinite(amp) or abs(amp) <= tol:
        return 0
    return int(min(WARMUP_MAX_MIN, np.ceil(tau * np.log(abs(amp) / tol))))


def _start_curve(r: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    d = r.assign(dev=r["temperature"] - r["background"])
    late = d[d["minutes_since_start"] >= WARMUP_MAX_MIN].groupby("walk_id")["dev"].mean()
    d["rel"] = d["dev"] - d["walk_id"].map(late)
    d["minute"] = np.floor(d["minutes_since_start"]).astype(int)
    d = d[(d["minute"] >= 0) & (d["minute"] < WARMUP_MAX_MIN)].dropna(subset=["rel"])
    g = d.groupby("minute")["rel"]
    return g.mean(), g.size()


def estimate_warmup(r: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    """Per period: how far the first minutes of a walk sit from the walk's later level against the logger
    background, and how long until that departure fades (exponential fit). The adjusted two-way check
    (warmup_effects) says how much of it remains once position along the route is allowed for."""
    cov = r[r["logger_covered"]].reset_index(drop=True)
    facts, rows = {"warmup_tol_c": WARMUP_TOL_C, "warmup_max_min": WARMUP_MAX_MIN,
                   "warmup_dist_bin_m": WARMUP_DIST_BIN_M}, []
    for per in PERIODS:
        q = cov[cov["period"] == per]
        curve, n = _start_curve(q)
        amp, tau, win = fit_settling(curve, n)
        byw = {w: g for w, g in q.groupby("walk_id")}
        keys = list(byw)
        wins, amps, bc = [], [], []
        for _ in range(n_boot):
            b = pd.concat([byw[keys[k]].assign(walk_id=f"{keys[k]}#{j}")
                           for j, k in enumerate(rng.choice(len(keys), len(keys)))])
            c, nn = _start_curve(b)
            a, _, wnd = fit_settling(c, nn)
            wins.append(wnd); amps.append(a); bc.append(c.reindex(curve.index).to_numpy())
        bc = np.array(bc)
        rows.append(pd.DataFrame({"period": per, "minute": curve.index, "mean_c": curve.to_numpy(),
                                  "lo": np.nanpercentile(bc, 2.5, axis=0), "hi": np.nanpercentile(bc, 97.5, axis=0),
                                  "n_readings": n.to_numpy(), "fit_c": _settle(curve.index.to_numpy(float) + 0.5, amp, tau)}))
        facts[per] = {"start_c": float(curve.iloc[0]), "start_lo": float(np.nanpercentile(bc[:, 0], 2.5)),
                      "start_hi": float(np.nanpercentile(bc[:, 0], 97.5)), "fit_amp_c": amp, "fit_tau_min": tau,
                      "window_min": win, "window_lo": float(np.percentile(wins, 2.5)),
                      "window_hi": float(np.percentile(wins, 97.5)), "n_walks": len(keys)}
    eff = warmup_effects(cov)
    boots = []
    byw = {w: g for w, g in cov.groupby("walk_id")}
    keys = list(byw)
    for _ in range(min(n_boot, 200)):
        b = pd.concat([byw[keys[k]].assign(_bw=f"{keys[k]}#{j}") for j, k in enumerate(rng.choice(len(keys), len(keys)))],
                      ignore_index=True)
        boots.append(warmup_effects(b, b["_bw"].to_numpy()))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    facts["adjusted_max_abs_effect_c"] = float(np.abs(eff).max())
    facts["adjusted_n_minutes_interval_excludes_zero"] = int(((lo > 0) | (hi < 0)).sum())
    facts["adjusted_half_width_median_c"] = float(np.median((hi - lo) / 2))
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(pd.DataFrame({"minute": np.arange(WARMUP_MAX_MIN), "adjusted_effect_c": eff,
                                  "adjusted_lo": lo, "adjusted_hi": hi}), on="minute", how="left")
    return facts, out


def _detrend(g: pd.DataFrame) -> pd.Series:
    """Linear time trend fitted on the walk's rows outside the warm-up window, removed from all its rows."""
    x = (g["t_utc"] - g["t_utc"].min()).dt.total_seconds().to_numpy()
    y = g["temperature"].to_numpy(float)
    A = np.c_[x, np.ones_like(x)]
    fit = ~g["warmup"].to_numpy(bool)
    c = np.linalg.lstsq(A[fit], y[fit], rcond=None)[0] if fit.sum() >= 2 else np.array([0.0, np.nan])
    return pd.Series(y - A @ c, index=g.index)


def add_anomalies(r: pd.DataFrame) -> pd.DataFrame:
    """Offsets and trends come from rows outside the warm-up window; warm-up rows keep a value for display."""
    r = r.copy()
    r["anomaly_detrend"] = r.groupby("walk_id", group_keys=False)[["t_utc", "temperature", "warmup"]].apply(_detrend)
    dev = r["temperature"] - r["background"]
    r["anomaly_logger"] = dev - dev.where(~r["warmup"]).groupby(r["walk_id"]).transform("mean")
    r["anomaly"] = np.where(r["logger_covered"], r["anomaly_logger"], r["anomaly_detrend"])
    r["anomaly_source"] = np.where(r["logger_covered"], "logger_background", "walk_detrend")
    return r


def _slope_km(g: pd.DataFrame, col: str) -> float:
    x = g["distance_along_m"].to_numpy(float) / 1000.0
    return float(np.polyfit(x, g[col].to_numpy(float), 1)[0])


def profile(r: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Along-route mean anomaly per period in SEGMENT_M segments (logger and detrend versions)
    and gradient numbers with walk-bootstrap intervals (logger-covered walks only)."""
    allcov = r[r["logger_covered"]].copy()
    allcov["segment"] = np.floor(allcov["distance_along_m"] / SEGMENT_M).astype(int)
    cov = allcov[~allcov["warmup"]]
    rows, facts = [], {}
    for per in PERIODS:
        q = cov[cov["period"] == per]
        wu = allcov[(allcov["period"] == per) & allcov["warmup"]]
        walk_seg = q.groupby(["walk_id", "segment"])[["anomaly_logger", "anomaly_detrend"]].mean().reset_index()
        g = walk_seg.groupby("segment")
        seg = g[["anomaly_logger", "anomaly_detrend"]].mean()
        seg["n_walks"] = g.size()
        seg["n_readings"] = q.groupby("segment").size()
        wl = {w: s for w, s in walk_seg.groupby("walk_id")}
        keys = list(wl)
        bl = []
        for _ in range(n_boot):
            b = pd.concat([wl[keys[k]] for k in rng.choice(len(keys), len(keys))])
            bl.append(b.groupby("segment")["anomaly_logger"].mean().reindex(seg.index))
        bl = np.array(bl)
        seg["logger_lo"] = np.nanpercentile(bl, 2.5, axis=0)
        seg["logger_hi"] = np.nanpercentile(bl, 97.5, axis=0)
        wseg = wu.groupby(["walk_id", "segment"])["anomaly_logger"].mean().groupby("segment")
        seg = seg.join(wseg.mean().rename("anomaly_logger_warmup"), how="outer")
        seg["n_walks_warmup"] = wseg.size()
        seg = seg.reset_index().assign(period=per)
        seg["distance_m"] = (seg["segment"] + 0.5) * SEGMENT_M
        rows.append(seg)
        f = {}
        for col, key in (("anomaly_logger", "logger"), ("anomaly_detrend", "detrend")):
            pt = _slope_km(q, col)
            bs = []
            byw = {w: s for w, s in q.groupby("walk_id")}
            kk = list(byw)
            for _ in range(n_boot):
                b = pd.concat([byw[kk[k]] for k in rng.choice(len(kk), len(kk))])
                bs.append(_slope_km(b, col))
            f[key] = {"slope_c_per_km": pt, "slope_lo": float(np.percentile(bs, 2.5)),
                      "slope_hi": float(np.percentile(bs, 97.5))}
        s = seg[seg["n_readings"] >= SEGMENT_MIN_READINGS]
        f["profile_r_logger_vs_detrend"] = float(s["anomaly_logger"].corr(s["anomaly_detrend"]))
        f["profile_range_logger_c"] = float(s["anomaly_logger"].max() - s["anomaly_logger"].min())
        f["profile_range_detrend_c"] = float(s["anomaly_detrend"].max() - s["anomaly_detrend"].min())
        f["n_walks"] = int(q["walk_id"].nunique())
        # where at least half of the period's walks contribute (a few walks start mid-route)
        most = s[s["n_walks"] >= f["n_walks"] / 2]
        f["start_m"] = float(most["distance_m"].min() - SEGMENT_M / 2)
        facts[per] = f
    return pd.concat(rows, ignore_index=True), facts


# 4 ---------------------------------------------------------------------------

def sensor_features(p12: pd.DataFrame, points: pd.DataFrame, taus=TAU_SCAN_S) -> pd.DataFrame:
    """Sensor-matched shade, 1 h dose, sky view factor and height-to-width ratio per walk, point and tau.
    tau 0 is the 1 m value."""
    cols = {**MEASURES, **EXTRA}
    base = p12[["walk_id", "point_id", "t_arrival_utc", "shaded_at_arrival", "dose_1h_before_wh_m2"]].copy()
    base["t_arrival_utc"] = pd.to_datetime(base["t_arrival_utc"], utc=True, format="ISO8601")
    base["shaded_at_arrival"] = base["shaded_at_arrival"].astype(float)
    base = base.merge(points[["point_id", "sky_view_factor", "height_width_ratio"]], on="point_id", how="left")
    pos = [t for t in taus if t > 0]
    out = []
    for w, g in base.groupby("walk_id", sort=False):
        s = sensor_matched(g[["point_id", "walk_id", "t_arrival_utc"]], g[["point_id", *cols.values()]],
                           columns=list(cols.values()), taus=pos)
        for k, c in cols.items():
            s[f"{k}_tau0s"] = g[c].to_numpy()
            for t in pos:
                s[f"{k}_tau{t}s"] = s.pop(f"{c}_tau{t:g}s")
        out.append(s)
    return pd.concat(out, ignore_index=True)


def _demean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols] - df.groupby("walk_id")[cols].transform("mean")


def _walk_moments(df: pd.DataFrame, xcols: list[str], ycol: str = "anomaly"):
    """Per-walk within moments: X'X, X'y, y'y on within-walk demeaned data."""
    sub = df.dropna(subset=[ycol, *xcols])
    dm = _demean(sub, [ycol, *xcols])
    X, y = dm[xcols].to_numpy(float), dm[ycol].to_numpy(float)
    w = sub["walk_id"].to_numpy()
    keys, inv = np.unique(w, return_inverse=True)
    k = len(xcols)
    XX = np.zeros((len(keys), k, k)); Xy = np.zeros((len(keys), k)); yy = np.zeros(len(keys))
    for j in range(len(keys)):
        m = inv == j
        XX[j] = X[m].T @ X[m]; Xy[j] = X[m].T @ y[m]; yy[j] = y[m] @ y[m]
    return keys, XX, Xy, yy


def _r2_from(XX, Xy, yy, weights=None):
    wt = np.ones(len(yy)) if weights is None else weights
    A = np.tensordot(wt, XX, 1); b = wt @ Xy; s = wt @ yy
    beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(beta @ b / s)


def _cv_r2_from(XX, Xy, yy):
    """Leave one walk out: fit on the other walks' moments, score the held-out walk."""
    A, b = XX.sum(0), Xy.sum(0)
    sse = 0.0
    for j in range(len(yy)):
        beta = np.linalg.lstsq(A - XX[j], b - Xy[j], rcond=None)[0]
        sse += yy[j] - 2 * beta @ Xy[j] + beta @ XX[j] @ beta
    return float(1 - sse / yy.sum())


def tau_scan(df: pd.DataFrame, n_boot: int, rng: np.random.Generator, taus=TAU_SCAN_S) -> tuple[pd.DataFrame, dict]:
    rows, facts = [], {}
    for per in PERIODS:
        q = df[df["period"] == per]
        mom = {t: _walk_moments(q, [f"{k}_tau{t}s" for k in MEASURES]) for t in taus}
        keys = mom[taus[0]][0]
        assert all(np.array_equal(mom[t][0], keys) for t in taus)
        pt = np.array([_r2_from(*mom[t][1:]) for t in taus])
        boot = np.empty((n_boot, len(taus)))
        for b in range(n_boot):
            wt = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)
            boot[b] = [_r2_from(*mom[t][1:], weights=wt) for t in taus]
        best = np.array(taus)[boot.argmax(axis=1)]
        for i, t in enumerate(taus):
            rows.append({"period": per, "tau_s": t, "r2_within": pt[i], "cv_r2": _cv_r2_from(*mom[t][1:]),
                         "r2_lo": np.percentile(boot[:, i], 2.5),
                         "r2_hi": np.percentile(boot[:, i], 97.5), "share_boot_best": float((best == t).mean())})
        facts[per] = {"best_tau_s": int(np.array(taus)[pt.argmax()]), "best_r2": float(pt.max()),
                      "best_tau_lo": float(np.percentile(best, 2.5)), "best_tau_hi": float(np.percentile(best, 97.5)),
                      "r2_tau0": float(pt[0]), "n_readings": int(q.dropna(subset=["anomaly"]).shape[0]),
                      "n_walks": int(len(keys))}
    return pd.DataFrame(rows), facts


def find_events(p12: pd.DataFrame, side_m: int = EVENT_SIDE_M) -> pd.DataFrame:
    """Sun to shade and shade to sun transitions with the shade state constant >= side_m points on each side."""
    rows = []
    q = p12.dropna(subset=["t_arrival_utc"]).copy()
    q["t_arrival_utc"] = pd.to_datetime(q["t_arrival_utc"], utc=True, format="ISO8601")
    for w, g in q.groupby("walk_id"):
        g = g.sort_values("distance_along_m")
        s = g["shaded_at_arrival"].astype(bool).to_numpy()
        d = g["distance_along_m"].to_numpy(float)
        gps = (g["arrival_source"] == "gps").to_numpy()
        t = g["t_arrival_utc"]
        contiguous = np.r_[True, np.diff(d) <= 1.5] & gps
        run = np.cumsum((np.r_[True, s[1:] != s[:-1]]) | ~contiguous)
        starts = np.r_[0, np.flatnonzero(np.diff(run)) + 1]
        ends = np.r_[starts[1:], len(s)]
        for a in range(len(starts) - 1):
            i0, i1 = starts[a], ends[a]
            j0, j1 = starts[a + 1], ends[a + 1]
            if not contiguous[j0] or not (gps[i0:i1].all() and gps[j0:j1].all()):
                continue
            long_enough = (t.iloc[j0] - t.iloc[i0]).total_seconds() >= side_m / EVENT_MAX_SPEED_MS and \
                (t.iloc[j1 - 1] - t.iloc[j0]).total_seconds() >= side_m / EVENT_MAX_SPEED_MS
            if d[i1 - 1] - d[i0] + 1 >= side_m and d[j1 - 1] - d[j0] + 1 >= side_m and long_enough:
                rows.append({"walk_id": w, "period": w.split("_")[2], "distance_m": float(d[j0]),
                             "t_utc": g["t_arrival_utc"].iloc[j0],
                             "direction": "sun_to_shade" if s[j0] else "shade_to_sun",
                             "before_m": float(d[i1 - 1] - d[i0] + 1), "after_m": float(d[j1 - 1] - d[j0] + 1),
                             "before_s": float((t.iloc[j0] - t.iloc[i0]).total_seconds()),
                             "after_s": float((t.iloc[j1 - 1] - t.iloc[j0]).total_seconds())})
    return pd.DataFrame(rows)


def event_traces(events: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    """Anomaly around each event in EVENT_BIN_S bins, relative to the event's mean before the transition,
    signed so the expected response is negative (towards shade = cooler) for both directions."""
    lo, hi = EVENT_WINDOW_S
    out = []
    by = {w: g for w, g in r.groupby("walk_id")}
    for k, e in events.reset_index(drop=True).iterrows():
        g = by.get(e["walk_id"])
        if g is None:
            continue
        dt = (g["t_utc"] - e["t_utc"]).dt.total_seconds().to_numpy()
        # only the stretch where the shade state stays as it was on each side
        m = (dt >= max(lo, -e["before_s"])) & (dt < min(hi, e["after_s"]))
        if (dt[m] < 0).sum() < 2 or (dt[m] >= 0).sum() < 4:
            continue
        y = g["anomaly"].to_numpy(float)[m]
        y = y - y[dt[m] < 0].mean()
        sign = 1.0 if e["direction"] == "sun_to_shade" else -1.0
        out.append(pd.DataFrame({"event": k, "period": e["period"], "direction": e["direction"],
                                 "bin_s": np.floor(dt[m] / EVENT_BIN_S) * EVENT_BIN_S + EVENT_BIN_S / 2,
                                 "y": sign * y}))
    return pd.concat(out, ignore_index=True)


def _approach(t, amp, tau):
    return np.where(t < 0, 0.0, amp * (1.0 - np.exp(-np.clip(t, 0, None) / tau)))


def fit_approach(curve: pd.Series, n: pd.Series) -> tuple[float, float]:
    """Step response amp * (1 - exp(-t / tau)) after t = 0, zero before; bins weighted by event count."""
    from scipy.optimize import curve_fit

    nn = n.reindex(curve.index)
    ok = curve.notna() & (nn >= EVENT_MIN_SHARE * nn.max())
    t = curve.index.to_numpy(float)[ok]
    y = curve.to_numpy(float)[ok]
    w = np.sqrt(n.reindex(curve.index).to_numpy(float)[ok])
    try:
        (amp, tau), _ = curve_fit(_approach, t, y, p0=(y[t > 0].mean(), 30.0), sigma=1 / w,
                                  bounds=([-5.0, 1.0], [5.0, 600.0]), maxfev=10000)
    except RuntimeError:
        return np.nan, np.nan
    return float(amp), float(tau)


def event_fit(traces: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    per_event = traces.groupby(["event", "bin_s"])["y"].mean().unstack()
    mean = per_event.mean()
    amp, tau = fit_approach(mean, per_event.notna().sum())
    ev = per_event.index.to_numpy()
    taus, amps = [], []
    for _ in range(n_boot):
        b = per_event.loc[rng.choice(ev, len(ev))]
        a, t = fit_approach(b.mean(), b.notna().sum())
        taus.append(t); amps.append(a)
    taus, amps = np.array(taus), np.array(amps)
    lo_s, hi_s = EVENT_CHANGE_WINDOW_S
    cols = [c for c in per_event.columns if lo_s <= c < hi_s]
    late = per_event[cols].mean(axis=1).dropna()
    change_boot = [late.loc[rng.choice(late.index, len(late))].mean() for _ in range(n_boot)]
    curve = pd.DataFrame({"bin_s": mean.index, "mean_c": mean.to_numpy(),
                          "n_events": per_event.notna().sum().to_numpy()})
    return {"tau_s": tau, "tau_lo": float(np.nanpercentile(taus, 2.5)), "tau_hi": float(np.nanpercentile(taus, 97.5)),
            "amp_c": amp, "amp_lo": float(np.nanpercentile(amps, 2.5)), "amp_hi": float(np.nanpercentile(amps, 97.5)),
            "share_boot_at_bound": float(np.mean((taus <= 1.01) | (taus >= 599))),
            "change_c": float(late.mean()), "change_lo": float(np.percentile(change_boot, 2.5)),
            "change_hi": float(np.percentile(change_boot, 97.5)), "change_window_s": list(EVENT_CHANGE_WINDOW_S),
            "n_events": int(len(ev))}, curve


# 5 ---------------------------------------------------------------------------

def association_model(df: pd.DataFrame, tau: int, terms: list[str]) -> tuple[pd.DataFrame, dict]:
    """anomaly ~ terms + walk fixed effects, OLS with standard errors clustered by walk."""
    import statsmodels.api as sm

    cols = [f"{k}_tau{tau}s" for k in terms]
    sub = df.dropna(subset=["anomaly", *cols]).reset_index(drop=True)
    dm = _demean(sub, ["anomaly", *cols])
    X = dm[cols].to_numpy(float)
    y = dm["anomaly"].to_numpy(float)
    groups = pd.factorize(sub["walk_id"])[0]
    n_w = groups.max() + 1
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    # demeaning absorbs one parameter per walk; rescale the clustered covariance's dof accordingly
    n, k = X.shape
    adj = np.sqrt((n - k) / (n - k - n_w))
    se = res.bse * adj
    rows = []
    for j, t in enumerate(terms):
        u = UNITS[t]
        rows.append({"term": t, "column": cols[j], "unit": u, "effect_c": res.params[j] * u,
                     "lo": (res.params[j] - 1.96 * se[j]) * u, "hi": (res.params[j] + 1.96 * se[j]) * u})
    # leave one walk out
    sse = sst = 0.0
    for w in range(n_w):
        tr, te = groups != w, groups == w
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        sse += float(((y[te] - X[te] @ b) ** 2).sum()); sst += float((y[te] ** 2).sum())
    resid = y - X @ res.params
    lag1 = []
    for w in range(n_w):
        e = resid[groups == w]
        if len(e) > 2:
            lag1.append(np.corrcoef(e[:-1], e[1:])[0, 1])
    corr = sub[cols].corr()
    return pd.DataFrame(rows), {"r2_within": float(res.rsquared), "cv_r2": 1 - sse / sst, "n_readings": int(n),
                                "n_walks": int(n_w), "resid_lag1_r_median": float(np.median(lag1)),
                                "corr": {f"{a}|{b}": float(corr.iloc[i, j]) for i, a in enumerate(terms)
                                         for j, b in enumerate(terms) if j > i}}


def segment_associations(df: pd.DataFrame, tau: int, n_boot: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    cols = {k: f"{k}_tau{tau}s" for k in [*MEASURES, *EXTRA]}
    d = df.copy()
    d["segment"] = np.floor(d["distance_along_m"] / SEGMENT_M).astype(int)
    rows, facts = [], {}
    for per in PERIODS:
        q = d[d["period"] == per]
        ws = q.groupby(["walk_id", "segment"]).agg(anomaly=("anomaly", "mean"), n=("anomaly", "size"),
                                                   **{k: (c, "mean") for k, c in cols.items()}).reset_index()

        def seg_corr(t):
            g = t.groupby("segment").agg(anomaly=("anomaly", "mean"), n=("n", "sum"), **{k: (k, "mean") for k in cols})
            g = g[g["n"] >= SEGMENT_MIN_READINGS]
            return g, {k: float(g["anomaly"].corr(g[k])) for k in cols}

        g, pt = seg_corr(ws)
        byw = {w: s for w, s in ws.groupby("walk_id")}
        kk = list(byw)
        bs = [seg_corr(pd.concat([byw[kk[k]] for k in rng.choice(len(kk), len(kk))]))[1] for _ in range(n_boot)]
        facts[per] = {k: {"r": pt[k], "lo": float(np.nanpercentile([b[k] for b in bs], 2.5)),
                          "hi": float(np.nanpercentile([b[k] for b in bs], 97.5))} for k in cols}
        facts[per]["n_segments"] = int(len(g))
        rows.append(g.reset_index().assign(period=per))
    return pd.concat(rows, ignore_index=True), facts


# run -------------------------------------------------------------------------

def run(package_dir: Path, matched_dir: Path, route_json: Path, logger_dir: Path, *, n_boot: int = N_BOOT,
        seed: int = SEED) -> dict:
    package_dir = Path(package_dir)
    rng = np.random.default_rng(seed)
    points = pd.read_parquet(package_dir / "OM2" / "points.parquet").drop(columns="geometry", errors="ignore")
    p12 = pd.read_parquet(package_dir / "p12_walk_points.parquet")
    walks, fixes = load_walks(matched_dir, route_json)
    temps = load_temperature(matched_dir)
    minutes, _ = fl.load_loggers(logger_dir, tz=LOGGER_TZ)
    route_m = float(points["distance_along_m"].max())

    r, f_read = readings_table(points, fixes, temps)
    r, f_cov = add_background(r, walks, minutes)
    f_warm, warm_curve = estimate_warmup(r, n_boot, rng)
    r["warmup"] = r["minutes_since_start"] < r["period"].map({p: f_warm[p]["window_min"] for p in PERIODS})
    for p in PERIODS:
        f_warm[p]["n_dropped"] = int((r["warmup"] & (r["period"] == p)).sum())
        f_warm[p]["share_dropped"] = float(r.loc[r["period"] == p, "warmup"].mean())
    r = add_anomalies(r)
    seg_prof, f_prof = profile(r, n_boot, rng)
    r = r[~r["warmup"]].reset_index(drop=True)

    feats = sensor_features(p12, points)
    df = r.merge(feats, on=["walk_id", "point_id"], how="left")
    scan, f_scan = tau_scan(df, n_boot, rng)
    events = find_events(p12)
    traces = event_traces(events, r)
    f_event, ev_curve = event_fit(traces, n_boot, rng)
    f_event_per = {}
    for per in PERIODS:
        tp = traces[traces["period"] == per]
        f_event_per[per] = event_fit(tp, n_boot, rng)[0] if tp["event"].nunique() >= 10 else None
    tau_best = int(scan.groupby("tau_s")["r2_within"].sum().idxmax())
    tau = ASSOC_TAU_S

    coefs, f_model = [], {}
    specs = (("main", tau, list(MEASURES)), ("shade_svf", tau, ["shade", "svf"]),
             ("with_ratio", tau, [*MEASURES, *EXTRA]), ("main_scan_tau", tau_best, list(MEASURES)))
    for per in PERIODS:
        q = df[df["period"] == per]
        for name, t, terms in specs:
            c, f = association_model(q, t, terms)
            coefs.append(c.assign(period=per, model=name, tau_s=t))
            f_model[f"{per}_{name}"] = {**f, "tau_s": t,
                                        "effects": {row.term: {"effect_c": row.effect_c, "lo": row.lo, "hi": row.hi}
                                                    for row in c.itertuples()}}
    seg_assoc, f_seg = segment_associations(df, tau, n_boot, rng)

    out_cols = ["walk_id", "period", "t_utc", "t_local", "minutes_since_start", "distance_along_m", "point_id",
                "temperature", "background", "anomaly", "anomaly_source", "anomaly_detrend"]
    rd = df[out_cols].copy()
    rd["t_utc"] = rd["t_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    rd["t_local"] = rd["t_local"].map(lambda t: t.isoformat())
    rd.to_csv(package_dir / "p13_temperature_pairing_readings.csv", index=False, float_format="%.4f")
    scan.to_csv(package_dir / "p13_temperature_pairing_tau_scan.csv", index=False, float_format="%.5f")
    ev_out = events.copy()
    ev_out["t_utc"] = ev_out["t_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ev_out["used"] = ev_out.index.isin(traces["event"].unique())
    ev_out.to_csv(package_dir / "p13_temperature_pairing_events.csv", index=False, float_format="%.2f")
    ev_curve.to_csv(package_dir / "p13_temperature_pairing_event_response.csv", index=False, float_format="%.4f")
    pd.concat(coefs, ignore_index=True).to_csv(package_dir / "p13_temperature_pairing_coefficients.csv",
                                               index=False, float_format="%.5f")
    prof_out = seg_prof.merge(seg_assoc.drop(columns=["anomaly", "n"]), on=["period", "segment"], how="left")
    prof_out.to_csv(package_dir / "p13_temperature_pairing_segment_profile.csv", index=False, float_format="%.4f")
    warm_curve.to_csv(package_dir / "p13_temperature_pairing_warmup.csv", index=False, float_format="%.4f")

    n_cov_read = int(df["anomaly_source"].eq("logger_background").sum())
    facts = {
        **f_read, **f_cov, "warmup": f_warm,
        "n_dropped_warmup": int(sum(f_warm[p]["n_dropped"] for p in PERIODS)),
        "n_readings_analysed": int(len(df)),
        "n_readings_logger": n_cov_read,
        "n_walks_detrend_fallback": int(df.loc[df["anomaly_source"] == "walk_detrend", "walk_id"].nunique()),
        "anomaly_sd_c": float(df["anomaly"].std()),
        "reading_interval_s": float(r.groupby("walk_id")["t_utc"].diff().dt.total_seconds().median()),
        "logger_tz": LOGGER_TZ, "full_cover_share": FULL_COVER,
        "route_m": route_m, "segment_m": SEGMENT_M,
        "segment_min_readings": SEGMENT_MIN_READINGS,
        "profile": f_prof,
        "tau_scan_s": list(TAU_SCAN_S), "tau_scan": f_scan,
        "event_side_m": EVENT_SIDE_M, "event_window_s": list(EVENT_WINDOW_S), "events": f_event,
        "events_by_period": f_event_per,
        "n_events_found": int(len(events)),
        "n_events_sun_to_shade": int((events["direction"] == "sun_to_shade").sum()),
        "n_events_shade_to_sun": int((events["direction"] == "shade_to_sun").sum()),
        "chosen_tau_s": tau, "scan_best_tau_s": tau_best, "models": f_model, "segments": f_seg, "units": UNITS,
        "n_boot": n_boot, "seed": seed,
    }
    (package_dir / "OM2").mkdir(exist_ok=True)
    (package_dir / "OM2" / "temp_facts.json").write_text(json.dumps(facts, indent=2, default=float), encoding="utf-8")
    return facts


# text ------------------------------------------------------------------------

def _c(x: float, nd: int = 2) -> str:
    out = f"{x:.{nd}f}"
    return out[1:] if float(out) == 0 and out.startswith("-") else out


def _iv(lo: float, hi: float, unit: str = " °C", nd: int = 2) -> str:
    return f"95% interval {_c(lo, nd)} to {_c(hi, nd)}{unit}"


def _minutes(n: int) -> str:
    return "first minute" if n == 1 else f"first {n} minutes"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%" if x >= 0.005 else "below 1%"


def _s(x: float, nd: int = 0) -> str:
    return f"{x:,.{nd}f}"


def _tau_upper(ev: dict, xmax: float = 300.0) -> str:
    return f"beyond {xmax:g} s" if ev["tau_hi"] > xmax else f"{ev['tau_hi']:.0f} s"


def _eff(facts: dict, model: str, term: str, unit: str = " °C") -> str:
    e = facts["models"][model]["effects"][term]
    return f"{_c(e['effect_c'])}{unit} ({_iv(e['lo'], e['hi'])})"


def _cv_max(facts: dict, models: tuple[str, ...]) -> float:
    return max(facts["models"][m]["cv_r2"] for m in models)


def _adjusted(w: dict) -> str:
    n = w["adjusted_n_minutes_interval_excludes_zero"]
    if n == 0:
        return "no minute is clearly different from the rest"
    return f"{n} of the first {w['warmup_max_min']} minutes differ from the rest"


def _zero_note(f: dict, model: str, term: str) -> str:
    """Names the periods whose interval includes no difference."""
    inc = [p for p in PERIODS if f["models"][f"{p}_{model}"]["effects"][term]["lo"] <= 0
           <= f["models"][f"{p}_{model}"]["effects"][term]["hi"]]
    if not inc:
        return ""
    return f"; the {' and '.join(inc)} interval includes no difference"


def team_question(facts: dict) -> str:
    ev = facts["events"]
    return ("**One question for the team.** What is the time constant of the air temperature sensor as mounted, "
            "with its housing, and is the value you have the 63% or the 90% response time? The walk readings "
            f"bound it only loosely: the sun and shade changes put it between {ev['tau_lo']:.0f} s and "
            f"{_tau_upper(ev)}. The specification would fix τ for the sensor-matched columns and the segment length.\n")


def report_paragraphs(facts: dict) -> list[str]:
    """Markdown paragraphs of the report section, every number from facts (temp_facts.json).
    Figures are cited as {fig_temp_profile} and {fig_temp_tau}; the report fills in the numbers."""
    f = facts
    w = f["warmup"]
    pm, pe = f["profile"]["morning"], f["profile"]["evening"]
    sm, se = f["tau_scan"]["morning"], f["tau_scan"]["evening"]
    ev = f["events"]
    tau = f["chosen_tau_s"]
    mods = ("morning_main", "morning_shade_svf", "morning_with_ratio", "evening_main", "evening_shade_svf",
            "evening_with_ratio")
    cv = _cv_max(f, mods)
    mw, ew = f["models"]["morning_with_ratio"], f["models"]["evening_with_ratio"]
    segm, sege = f["segments"]["morning"], f["segments"]["evening"]
    shade_dose = min(f["models"]["morning_main"]["corr"]["shade|dose"], f["models"]["evening_main"]["corr"]["shade|dose"])
    svf_hw = min(mw["corr"]["svf|hw"], ew["corr"]["svf|hw"])
    lag1 = min(f["models"][m]["resid_lag1_r_median"] for m in mods)
    n_fall = f["n_walks"] - f["n_walks_covered_in_readings"]
    out = [
        "Street measures explain little of how the walk temperature readings vary along the route. Readings "
        "are a little cooler where the recent path was shaded and where the street is deeper, but no "
        f"combination of measures predicts more than {_pct(cv)} of the variation within a walk that was left "
        "out of the fit. This section is a first look, meant as input to Jingxue's analysis, which she leads. "
        "It describes associations; it does not test causes and it does not anticipate her conclusions.\n",

        f"**Readings and clocks.** The walks give one temperature reading every {f['reading_interval_s']:.0f} s. "
        f"We kept the {_s(f['n_readings'])} readings that fall on route points classed as street or projected; "
        f"{_s(f['n_dropped_off_street'])} readings on alleys missing from the street map, covered passages and "
        "unresolved points were left out. The walk timestamps are in universal time. The fixed loggers write Rio local time: their "
        "temperature follows Galeão airport temperature best with a 3 hour shift, so we read them on that clock. "
        "For each reading we subtract the background temperature of the two outdoor fixed loggers in Maré and "
        "then the walk's own mean difference from that background. What remains is the **anomaly**: how much "
        f"warmer or cooler than usual for that walk the reading is. {f['n_walks_covered_in_readings']} of the "
        f"{f['n_walks']} walks are fully covered by the loggers; for the other {n_fall} we remove a straight time "
        "trend per walk instead.\n",

        f"**Start of a walk.** Evening walks start {_c(-w['evening']['start_c'])} °C below their later level "
        f"against the loggers ({_iv(-w['evening']['start_hi'], -w['evening']['start_lo'])} below) and take about "
        f"{w['evening']['window_min']} minutes to settle. Morning walks show no such start (first minute "
        f"{_c(w['morning']['start_c'])} °C, {_iv(w['morning']['start_lo'], w['morning']['start_hi'])}). We left "
        f"out the first {w['evening']['window_min']} minutes of each evening walk and the "
        f"{_minutes(w['morning']['window_min'])} of each morning walk: {_s(f['n_dropped_warmup'])} readings, {_pct(w['evening']['share_dropped'])} of the evening "
        "readings. Every walk starts at 0 m, so the data cannot tell a sensor that is still settling from a "
        "first stretch of route that is cooler in the afternoon. Once position along the route is allowed for, "
        f"{_adjusted(w)} (intervals about ±{_c(w['adjusted_half_width_median_c'], 1)} °C). The cut is a precaution; it also means the evening "
        f"results describe only the route beyond about {_s(pe['start_m'])} m.\n",

        "{fig_temp_profile} shows the mean anomaly along the route for the morning and the evening walks: "
        "against the logger background (coloured, with the 95% interval from resampling walks) and with each "
        "walk's time trend removed instead (grey). Hatched stretches are the points left out; dotted lines are "
        "the first minutes of a walk, left out of everything else.\n",

        "**Along the route.** The two versions follow each other closely (correlation "
        f"{pm['profile_r_logger_vs_detrend']:.2f} in the morning, {pe['profile_r_logger_vs_detrend']:.2f} in the "
        "evening), and the logger version shows no clear gradient from start to end: "
        f"{_c(pm['logger']['slope_c_per_km'])} °C per km in the morning "
        f"({_iv(pm['logger']['slope_lo'], pm['logger']['slope_hi'], ' °C per km')}) and "
        f"{_c(pe['logger']['slope_c_per_km'])} °C per km in the evening "
        f"({_iv(pe['logger']['slope_lo'], pe['logger']['slope_hi'], ' °C per km')}). So a per-walk time trend, "
        "which would hide such a gradient, hides little here, within those intervals. The differences between "
        f"stretches are larger: the {f['segment_m']} m means span {_c(pm['profile_range_logger_c'], 1)} °C in "
        f"the morning and {_c(pe['profile_range_logger_c'], 1)} °C in the evening.\n",

        "{fig_temp_tau} shows the two ways we tried to estimate the sensor's effective time constant τ.\n",

        "**Time constant.** On the left, the share of the within-walk variation of the anomaly that "
        "sensor-matched shade, 1 hour sun dose and sky view factor explain together, for τ from 0 (the 1 m value) "
        f"to {f['tau_scan_s'][-1]} s. Within the fitted walks the share rises with τ, to "
        f"{_pct(sm['best_r2'])} in the morning (τ = {sm['best_tau_s']} s) and {_pct(se['best_r2'])} in the evening "
        f"(τ = {se['best_tau_s']} s). Scored on walks left out of the fit, the share is below zero at every τ: the "
        "gain does not carry over from one walk to the next. The rise with τ reflects broad patterns along the "
        "route, not the sensor's response. On the right, the readings are aligned on "
        f"{ev['n_events']} sharp changes between sun and shade, where the shade state stays the same for at least "
        f"{f['event_side_m']} m on each side. Between {ev['change_window_s'][0]} and {ev['change_window_s'][1]} s "
        f"after a change into shade, readings are {_c(-ev['change_c'])} °C lower than before it "
        f"({_iv(-ev['change_hi'], -ev['change_lo'])}), but the stretches end before the fall levels off, so the "
        f"fit bounds τ only loosely: {ev['tau_lo']:.0f} s to {_tau_upper(ev)}. The two estimates do not agree, and "
        "neither pins τ down.\n",

        f"**Associations.** We use τ = {tau} s, fixed in advance rather than tuned on the readings: it lies inside "
        "the interval from the sun and shade changes and is one of the shipped columns. Each model removes each "
        "walk's own level, and its intervals allow for readings within a walk being alike. A fully shaded recent "
        f"path, compared with a fully sunlit one, goes with {_eff(f, 'morning_shade_svf', 'shade')} in the morning "
        f"and {_eff(f, 'evening_shade_svf', 'shade')} in the evening{_zero_note(f, 'shade_svf', 'shade')}. Shade and the 1 hour sun dose carry nearly the "
        f"same information (correlation {shade_dose:.2f}), so a model with both cannot separate them. With "
        "height-to-width ratio added, each unit of the ratio goes with "
        f"{_eff(f, 'morning_with_ratio', 'hw')} in the morning and {_eff(f, 'evening_with_ratio', 'hw')} in the "
        "evening, and each 0.1 of sky view factor with "
        f"{_eff(f, 'morning_with_ratio', 'svf')} and {_eff(f, 'evening_with_ratio', 'svf')}. These models explain "
        f"{_pct(mw['r2_within'])} (morning) and {_pct(ew['r2_within'])} (evening) of the within-walk variation, and "
        f"{_pct(mw['cv_r2'])} and {_pct(ew['cv_r2'])} of it in walks left out. Averaged over walks in "
        f"{f['segment_m']} m segments, the anomaly correlates with shade at {segm['shade']['r']:.2f} "
        f"(morning, {_iv(segm['shade']['lo'], segm['shade']['hi'], '')}) and {sege['shade']['r']:.2f} "
        f"(evening, {_iv(sege['shade']['lo'], sege['shade']['hi'], '')}), and with height-to-width ratio at "
        f"{segm['hw']['r']:.2f} ({_iv(segm['hw']['lo'], segm['hw']['hi'], '')}) and {sege['hw']['r']:.2f} "
        f"({_iv(sege['hw']['lo'], sege['hw']['hi'], '')}).\n",

        f"**Limits.** Successive readings are nearly alike (correlation {lag1:.2f} between readings "
        f"{f['reading_interval_s']:.0f} s apart after the model), so the effective number of readings is far "
        "smaller than the count. Intervals treat walks as independent, but nearby stretches of route are not, "
        "so they are likely too narrow. The street measures move together: sky view factor and height-to-width "
        f"ratio correlate at {svf_hw:.2f}, so their separate effects are not well defined. There are "
        f"{sm['n_walks']} morning and {se['n_walks']} evening walks with usable readings, the sun measures assume a "
        "clear sky, and humidity, wind and traffic are not in these models.\n",

        "This is a first look to support Jingxue's analysis. The tables `p13_temperature_pairing_*.csv` hold "
        "every reading with its background and anomaly, the time constant scan, the sun and shade changes, the "
        "model coefficients and the segment profile, so each step can be redone with other choices.\n",
    ]
    return out


FIGURE_CAPTIONS = {
    "fig_temp_profile.png": (
        "Mean temperature anomaly along the route for morning (blue) and evening (vermilion) walks, in {seg} m "
        "segments, against the outdoor logger background (band: 95% interval from resampling walks) and with a "
        "per-walk time trend removed instead (grey). Dotted: first minutes of a walk, left out."),
    "fig_temp_tau.png": (
        "Left: share of the within-walk variation of the anomaly explained by sensor-matched shade, 1 hour sun "
        "dose and sky view factor against the time constant τ, within the fitted walks (filled, band: 95% "
        "interval) and scored on walks left out (open); the bar is the interval from the sun and shade changes. "
        "Right: mean temperature change around a change into shade (leaving shade with sign flipped) and the "
        "fitted exponential approach."),
}


def readme_subsection(facts: dict) -> str:
    f = facts
    w = f["warmup"]
    ev = f["events"]
    taus = ", ".join(str(t) for t in f["tau_scan_s"])
    return f"""### Temperature pairing (first look)

Files: `p13_temperature_pairing_readings.csv` (one row per reading: walk, UTC and Rio local time, minutes since
the walk started, distance along the route, nearest point, temperature, logger background, anomaly and its
source, and the per-walk detrended anomaly), `p13_temperature_pairing_tau_scan.csv`,
`p13_temperature_pairing_events.csv`, `p13_temperature_pairing_event_response.csv`,
`p13_temperature_pairing_coefficients.csv`, `p13_temperature_pairing_segment_profile.csv`,
`p13_temperature_pairing_warmup.csv`, and `OM2/temp_facts.json` (every number in the report text).
Code: `src/om_package/temp_pairing.py`. This is input to Jingxue's analysis, which she leads; it reports
descriptive associations only.

**Readings.** Matched GPS fixes on route edges that carry a temperature, snapped to the nearest 1 m point.
Readings on points whose class is not street or projected are dropped ({_s(f['n_dropped_off_street'])} of
{_s(f['n_matched_with_temperature'])}; basis: `{f['keep_basis']}`), leaving {_s(f['n_readings'])} readings
from {f['n_walks_readings']} walks.

**Clocks.** Walk timestamps are UTC. The fixed loggers write Rio local time (their cross-correlation with
Galeão airport temperature peaks at a 3 hour shift, and their daily peak matches the airport's), so they are
read with the America/Sao_Paulo clock. A walk counts as covered when the outdoor loggers have a minute within
every minute of the walk ({f['n_walks_logger_full']} of {f['n_walks']} walks).

**Background and anomaly.** Background = offset-corrected median of the outdoor loggers, interpolated to each
reading (`fixed_loggers.background_temperature`). Anomaly = temperature minus background minus the walk's mean
of that difference. Walks without cover ({f['n_walks_detrend_fallback']} with readings) use a per-walk linear
time detrend instead (`anomaly_source` = `walk_detrend`). The detrended anomaly of every walk is kept for
comparison; because every walk runs from 0 m to the end, a time detrend also removes any along-route gradient.

**Start of a walk.** Per period, the temperature minus background minus the walk's mean after
{w['warmup_max_min']} minutes, averaged by minute since start, is fitted with amplitude × exp(-t / T). The window is the
time until the fitted departure falls below {w['warmup_tol_c']:g} °C, capped at {w['warmup_max_min']} minutes:
{w['morning']['window_min']} minute(s) in the morning, {w['evening']['window_min']} minutes in the evening
(walk bootstrap {w['evening']['window_lo']:.0f} to {w['evening']['window_hi']:.0f}). Readings inside the window are
left out of all models ({_s(f['n_dropped_warmup'])} readings); walk offsets and trends are fitted without them.
Check: a two-way model with walk offsets, {w['warmup_dist_bin_m']:g} m distance bins per period and minute
effects finds no minute effect whose interval excludes zero (largest {w['adjusted_max_abs_effect_c']:.2f} °C),
so the data cannot separate settling from position along the route.

**Time constant, scan.** For τ in {taus} s, sensor-matched shade at arrival, 1 hour sun dose and sky view
factor are recomputed per walk with `sensor_match.sensor_matched` from the arrival times in
`p12_walk_points` (τ = 0: the 1 m value). The anomaly is regressed on the three after removing each walk's mean
(walk fixed effects), per period. The share explained comes with a walk bootstrap interval
({f['n_boot']} resamples) and a leave-one-walk-out score.

**Time constant, events.** A transition is a change of `shaded_at_arrival` with the state constant for at least
{f['event_side_m']} m on each side, GPS arrival times only, and no faster than walking pace. Readings from
{f['event_window_s'][0]} to {f['event_window_s'][1]} s around the transition, limited to the stable stretches
on either side, are taken relative to their mean before the change and signed so that a change into shade
should read negative. The mean response in 5 s bins (only bins with at least half of the events) is fitted with
amplitude × (1 - exp(-t / τ)); events are bootstrapped. Result: τ = {ev['tau_s']:.0f} s, interval
{ev['tau_lo']:.0f} to {ev['tau_hi']:.0f} s ({_pct(ev['share_boot_at_bound'])} of resamples at the fit limits of 1
or 600 s), from {ev['n_events']} transitions.

**Associations.** τ = {f['chosen_tau_s']} s, fixed in advance. Models per period: anomaly ~ shade + dose + sky
view factor; shade + sky view factor; and the three plus height-to-width ratio; all with walk fixed effects and
standard errors clustered by walk; leave-one-walk-out R² on demeaned data. The same three-term model at the
scan's best τ ({f['scan_best_tau_s']} s) is in the coefficient table for comparison. Effects are given per
fully shaded against fully sunlit recent path, per 100 Wh/m² of dose, per 0.1 of sky view factor and per unit of
height-to-width ratio. Segment level: per walk {f['segment_m']} m means, then means across walks; segments with
fewer than {f['segment_min_readings']} readings are dropped; correlation intervals from a walk bootstrap.
"""
