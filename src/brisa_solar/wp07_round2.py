"""WP-07 round-2 analyses for P1 (reviewer reports of 2026-10-07).

Producer for ``runs/wp07_round2_<UTC>/summary.json``; wp07_ledger copies every
number out of it by JSON pointer. Four analyses:

1. ``stratified`` (R2-M9 circularity): Table 1's deficit measures within the
   vertical-constraint strata, by the count (0–2) of the other two constraints.
2. ``bootstrap`` (R3/R2 uncertainty): spatial block-bootstrap percentile
   intervals for each site's ground share below 2 h (winter, equinox) and for
   the Table 1 / stratified cells. The block side is each site's own practical
   range of the winter below-floor indicator (empirical variogram), rounded up
   to 50 m — blocks shorter than the autocorrelation range would treat
   correlated neighbours as independent and understate the interval. A fixed
   100 m block is carried as a sensitivity.
3. 1 m medians: already in the ledger as ``site.<s>.ground.sun_h_winter.p50``
   (+ p25/p75) — same parquet rows as the share keys, nothing produced here.
4. ``december`` (heat axis, data only): direct-sun hours on the December
   solstice for every 1 m ground point. WP-04 stores binary patch visibility,
   NOT the marched horizon angles its sun hours are read against, so the
   ground observers are re-marched with the same engine and the June/equinox
   hours are recomputed and checked against the stored columns before the
   December column is trusted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import REPO_ROOT, load_params

WP04_RUN = "wp04_sites_20261007T201123Z"
CROSSTAB_IN_NAME = "per_patch_geometry_nodata0.csv"
FLOOR_H = 2.0

#: ledger slug -> on-disk dir name (WP-04 run folders and outputs/<site>/).
SITE_DIRS = {
    "vidigal": "vidigal",
    "rocinha": "rocinha",
    "complexo_do_alemao": "complexo_do_alemao",
    "riodaspedras": "riodaspedras",
    "mare": "maré",
}

N_BOOT = 2000
SEED = 20261007
VARIO_SAMPLE = 10_000
VARIO_LAG_M = 10.0
VARIO_MAX_M = 500.0
VARIO_SILL_FROM_M = 300.0
VARIO_SILL_FRACTION = 0.95
BLOCK_ROUND_M = 50.0
SENSITIVITY_BLOCK_M = 100.0

DEC_THRESHOLDS_H = (6, 8)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# 1. Stratified cross-tab
# ---------------------------------------------------------------------------

def _deficits(sub: pd.DataFrame) -> dict:
    n = len(sub)
    return {
        "cells": int(n),
        "point_deficit": float((1.0 - sub["share_ge_2h_winter"]).mean()) if n else float("nan"),
        "cell_deficit": float((sub["sun_h_winter_p50"] < FLOOR_H).mean()) if n else float("nan"),
    }


def stratified_crosstab(table: pd.DataFrame) -> dict:
    """{"v0": {"o0": .., "o1": .., "o2": ..}, "v1": {...}} — v = vertical
    constraint holds, o = lateral + directional count. Same estimators and the
    same drop rule (no ground sample) as wp07_crosstab.crosstab."""
    t = table.dropna(subset=["share_ge_2h_winter", "sun_h_winter_p50"])
    other = t["constraint_lateral"].astype(int) + t["constraint_directional"].astype(int)
    out = {}
    for v in (0, 1):
        out[f"v{v}"] = {
            f"o{o}": _deficits(t[(t["constraint_vertical"].astype(int) == v) & (other == o)])
            for o in (0, 1, 2)
        }
    return out


# ---------------------------------------------------------------------------
# 2. Block bootstrap
# ---------------------------------------------------------------------------

def practical_range_m(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int = SEED) -> dict:
    """Empirical semivariogram of `z` on a seeded subsample; the practical range
    is the first lag whose semivariance reaches VARIO_SILL_FRACTION of the sill
    (mean semivariance over lags ≥ VARIO_SILL_FROM_M)."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), min(VARIO_SAMPLE, len(x)), replace=False)
    xs, ys, zs = x[idx], y[idx], z[idx].astype(float)
    pairs = cKDTree(np.c_[xs, ys]).query_pairs(VARIO_MAX_M, output_type="ndarray")
    h = np.hypot(xs[pairs[:, 0]] - xs[pairs[:, 1]], ys[pairs[:, 0]] - ys[pairs[:, 1]])
    g = 0.5 * (zs[pairs[:, 0]] - zs[pairs[:, 1]]) ** 2
    edges = np.arange(0.0, VARIO_MAX_M + VARIO_LAG_M, VARIO_LAG_M)
    nb = len(edges) - 1
    b = np.clip(np.digitize(h, edges) - 1, 0, nb - 1)
    gamma = np.bincount(b, g, nb) / np.maximum(np.bincount(b, minlength=nb), 1)
    sill = float(gamma[edges[:-1] >= VARIO_SILL_FROM_M].mean())
    reached = np.nonzero(gamma >= VARIO_SILL_FRACTION * sill)[0]
    range_m = float(edges[1:][reached[0]]) if len(reached) else float(VARIO_MAX_M)
    return {
        "range_m": range_m,
        "sill": sill,
        "lag_upper_edges_m": edges[1:].tolist(),
        "semivariance": gamma.tolist(),
    }


def block_codes(x: np.ndarray, y: np.ndarray, block_m: float) -> np.ndarray:
    """Dense 0..B-1 code per observation for an absolute-coordinate square grid."""
    keys = np.floor(x / block_m).astype(np.int64) * 10_000_000 + np.floor(y / block_m).astype(np.int64)
    return np.unique(keys, return_inverse=True)[1]


def block_sums(codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """(n_blocks, k) per-block sums of `values` (n, k)."""
    n_blocks = int(codes.max()) + 1
    return np.stack([np.bincount(codes, values[:, j], n_blocks) for j in range(values.shape[1])], axis=1)


def resampled_sums(sums: np.ndarray, reps: int, rng: np.random.Generator) -> np.ndarray:
    """(reps, k): sum over a with-replacement draw of as many blocks as exist."""
    n_blocks = sums.shape[0]
    counts = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=reps)
    return counts @ sums


def interval(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """95% percentile interval of num/den over replicates; replicates with an
    empty class (den == 0) are dropped."""
    ok = den > 0
    r = num[ok] / den[ok]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def _class_matrix(labels: np.ndarray, classes: list, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num = np.stack([np.where(labels == c, weights, 0.0) for c in classes], axis=1)
    den = np.stack([(labels == c).astype(float) for c in classes], axis=1)
    return num, den


def _xtab_labels(t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = t["n_constraints"].astype(int).to_numpy()
    other = t["constraint_lateral"].astype(int) + t["constraint_directional"].astype(int)
    strat = (t["constraint_vertical"].astype(int).astype(str).radd("v") + "_o" + other.astype(str)).to_numpy()
    return n, strat


STRAT_CLASSES = [f"v{v}_o{o}" for v in (0, 1) for o in (0, 1, 2)]


def bootstrap_all(ground: dict, cells: dict, block_m: dict, reps: int, seed: int) -> dict:
    """ground[slug] = DataFrame(x, y, below_winter, below_equinox);
    cells[slug] = per-patch table (ground-sampled cells only);
    block_m[slug] = block side. One rng drives every draw in a fixed order, so
    the whole result is a function of (inputs, block_m, reps, seed)."""
    rng = np.random.default_rng(seed)
    out = {"per_site": {}, "pooled": {}}
    pooled_num = {"n": 0.0, "strat": 0.0}
    pooled_den = {"n": 0.0, "strat": 0.0}
    for slug in SITE_DIRS:
        g = ground[slug]
        b = block_codes(g["x"].to_numpy(), g["y"].to_numpy(), block_m[slug])
        vals = np.c_[g["below_winter"].to_numpy(float), g["below_equinox"].to_numpy(float), np.ones(len(g))]
        rs = resampled_sums(block_sums(b, vals), reps, rng)
        site = {"block_m": float(block_m[slug]), "n_blocks": int(b.max()) + 1, "n_points": int(len(g))}
        for j, day in enumerate(("winter", "equinox")):
            lo, hi = interval(rs[:, j], rs[:, 2])
            site[f"share_below_2h_{day}"] = {"estimate": float(vals[:, j].mean()), "lo": lo, "hi": hi}

        c = cells[slug]
        cb = block_codes(c["center_x"].to_numpy(), c["center_y"].to_numpy(), block_m[slug])
        deficit = (1.0 - c["share_ge_2h_winter"]).to_numpy(float)
        n_lab, s_lab = _xtab_labels(c)
        site["cell_n_blocks"] = int(cb.max()) + 1
        xt = {}
        for key, labels, classes in (("n", n_lab, [0, 1, 2, 3]), ("strat", s_lab, STRAT_CLASSES)):
            num, den = _class_matrix(labels, classes, deficit)
            # num and den must come from the SAME block draw — resample jointly.
            joint = resampled_sums(block_sums(cb, np.c_[num, den]), reps, rng)
            k = len(classes)
            rnum, rden = joint[:, :k], joint[:, k:]
            pooled_num[key] = pooled_num[key] + rnum
            pooled_den[key] = pooled_den[key] + rden
            xt[key] = {}
            for j, cls in enumerate(classes):
                est = float(num[:, j].sum() / den[:, j].sum()) if den[:, j].sum() else float("nan")
                lo, hi = interval(rnum[:, j], rden[:, j]) if den[:, j].sum() else (float("nan"), float("nan"))
                xt[key][_cls_key(cls)] = {"point_deficit": est, "lo": lo, "hi": hi}
        site["crosstab"] = xt
        out["per_site"][slug] = site

    all_cells = pd.concat([cells[s] for s in SITE_DIRS], ignore_index=True)
    deficit = (1.0 - all_cells["share_ge_2h_winter"]).to_numpy(float)
    n_lab, s_lab = _xtab_labels(all_cells)
    for key, labels, classes in (("n", n_lab, [0, 1, 2, 3]), ("strat", s_lab, STRAT_CLASSES)):
        num, den = _class_matrix(labels, classes, deficit)
        out["pooled"][key] = {}
        for j, cls in enumerate(classes):
            lo, hi = interval(pooled_num[key][:, j], pooled_den[key][:, j])
            out["pooled"][key][_cls_key(cls)] = {
                "point_deficit": float(num[:, j].sum() / den[:, j].sum()), "lo": lo, "hi": hi,
            }
    return out


def _cls_key(cls) -> str:
    return f"n{cls}" if isinstance(cls, (int, np.integer)) else str(cls)


# ---------------------------------------------------------------------------
# 4. December solstice by re-march
# ---------------------------------------------------------------------------

def december_metrics(hours_dec: np.ndarray, below_june: np.ndarray) -> dict:
    h = np.asarray(hours_dec, dtype="float64")
    below_june = np.asarray(below_june, dtype=bool)
    out = {
        "n": int(len(h)),
        "p25": float(np.percentile(h, 25)),
        "p50": float(np.percentile(h, 50)),
        "p75": float(np.percentile(h, 75)),
    }
    for k in DEC_THRESHOLDS_H:
        out[f"share_ge_{k}h"] = float((h >= k).mean())
    out["share_lt2h_june_and_ge6h_december"] = float((below_june & (h >= 6)).mean())
    out["share_lt2h_june_and_lt2h_december"] = float((below_june & (h < FLOOR_H)).mean())
    return out


def remarch_site(site_dir: str, wp04_dir: Path, out_dir: Path, tmp_dir: Path, *,
                 data_root: Path, directions, meta, days: dict, device: str,
                 chunk: int = 300_000) -> dict:
    """Re-march every stored ground observer with the WP-04 engine and read
    direct-sun hours for `days` off the marched horizon. Returns the check of
    the recomputed June/equinox hours against the stored columns."""
    from . import wp04_sites as w4
    from .wp02_horizon import patch_visibility

    stored = pd.read_parquet(
        wp04_dir / site_dir / "ground.parquet",
        columns=["row", "col", "x", "y", "hours_winter_solstice", "hours_equinox", "ge_2h_winter_solstice"],
    )
    surface, transform, _crs, is_building, _bid, _ground, _dtm, _fp = w4.build_site_surface(
        site_dir, data_root, w4.CELL_M, tmp_dir,
    )
    patch_az = w4.patch_azimuth_deg(directions)
    obs_xy = stored[["x", "y"]].to_numpy(dtype="float64")
    n = len(obs_xy)
    hours = {label: np.zeros(n, dtype=np.float32) for label in days}
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        _vis, _onb, horizon = patch_visibility(
            surface, transform, obs_xy[start:end], directions=directions, is_building=is_building,
            obs_height_m=w4.OBS_HEIGHT_M, max_dist_m=w4.MAX_DIST_M, march_sampling="nearest",
            device=device, return_horizon=True,
        )
        for label, date_str in days.items():
            hours[label][start:end] = w4.direct_sun_hours(horizon, patch_az, date_str, meta, [])["hours_fractional"]
        del _vis, _onb, horizon

    check = {}
    for label in ("winter_solstice", "equinox"):
        diff = np.abs(hours[label].astype("float64") - stored[f"hours_{label}"].to_numpy("float64"))
        check[label] = {"n_mismatch": int((diff > 0).sum()), "max_abs_diff_h": float(diff.max())}

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row": stored["row"].to_numpy(), "col": stored["col"].to_numpy(),
        "hours_summer_solstice": hours["summer_solstice"],
    }).to_parquet(out_dir / "december_ground.parquet", index=False)

    daylight = int(w4.sun_positions(days["summer_solstice"], meta, "1h")["apparent_elevation"].gt(0).sum())
    return {
        "remarch_check_vs_stored": check,
        "daylight_hours_pvlib_hourly": daylight,
        "metrics": december_metrics(hours["summer_solstice"], ~stored["ge_2h_winter_solstice"].to_numpy(bool)),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_cells(root: Path, site_dir: str) -> tuple[pd.DataFrame, str]:
    path = root / "outputs" / site_dir / "geometry_indicators" / CROSSTAB_IN_NAME
    cols = ["center_x", "center_y", "share_ge_2h_winter", "sun_h_winter_p50",
            "constraint_vertical", "constraint_lateral", "constraint_directional", "n_constraints"]
    t = pd.read_csv(path, usecols=cols)
    return t, str(path.relative_to(root))


def _load_ground(root: Path, site_dir: str) -> pd.DataFrame:
    g = pd.read_parquet(root / "runs" / WP04_RUN / site_dir / "ground.parquet",
                        columns=["x", "y", "ge_2h_winter_solstice", "ge_2h_equinox"])
    return pd.DataFrame({
        "x": g["x"], "y": g["y"],
        "below_winter": ~g["ge_2h_winter_solstice"], "below_equinox": ~g["ge_2h_equinox"],
    })


def build_summary(root: Path, run_dir: Path, *, reps: int = N_BOOT, seed: int = SEED,
                  december: bool = True) -> dict:
    root = Path(root)
    params = load_params()
    summary = {
        "_utc": _utc_now(),
        "git_sha": _git_sha(root),
        "wp04_run_id": WP04_RUN,
        "crosstab_in_name": CROSSTAB_IN_NAME,
        "floor_h": FLOOR_H,
        "inputs": {},
    }

    cells_raw, cells, ground = {}, {}, {}
    for slug, d in SITE_DIRS.items():
        t, rel = _load_cells(root, d)
        summary["inputs"][slug] = rel
        cells_raw[slug] = t
        cells[slug] = t.dropna(subset=["share_ge_2h_winter", "sun_h_winter_p50"]).reset_index(drop=True)
        ground[slug] = _load_ground(root, d)

    summary["stratified"] = {slug: stratified_crosstab(cells_raw[slug]) for slug in SITE_DIRS}
    summary["stratified"]["pooled"] = stratified_crosstab(pd.concat(cells_raw.values(), ignore_index=True))

    vario = {}
    for slug in SITE_DIRS:
        g = ground[slug]
        v = practical_range_m(g["x"].to_numpy(), g["y"].to_numpy(), g["below_winter"].to_numpy(), seed=seed)
        v["block_m"] = float(np.ceil(v["range_m"] / BLOCK_ROUND_M) * BLOCK_ROUND_M)
        vario[slug] = v
    summary["variogram"] = {
        "indicator": "ground point below 2 h on the winter solstice (1 m site run)",
        "sample_points": VARIO_SAMPLE, "lag_m": VARIO_LAG_M, "max_lag_m": VARIO_MAX_M,
        "sill_from_m": VARIO_SILL_FROM_M, "sill_fraction": VARIO_SILL_FRACTION,
        "block_rounding_m": BLOCK_ROUND_M, "seed": seed,
        "per_site": vario,
    }

    boot_cfg = {"replicates": reps, "seed": seed, "interval": "percentile 2.5/97.5",
                "scheme": "square blocks on absolute UTM coordinates, resampled with replacement "
                          "within each site (n_blocks draws); pooled = sum of per-site draws"}
    summary["bootstrap"] = {
        **boot_cfg,
        "block_rule": "per-site practical range of the winter indicator, rounded up to "
                      f"{BLOCK_ROUND_M:g} m",
        **bootstrap_all(ground, cells, {s: vario[s]["block_m"] for s in SITE_DIRS}, reps, seed),
    }
    summary["bootstrap_100m"] = {
        **boot_cfg,
        "block_rule": f"fixed {SENSITIVITY_BLOCK_M:g} m (sensitivity)",
        **bootstrap_all(ground, cells, {s: SENSITIVITY_BLOCK_M for s in SITE_DIRS}, reps, seed),
    }

    if december:
        import torch
        from . import wp04_sites as w4
        from src.svf_v2.compute import generate_tregenza_patches

        days = {k: params["reference_days"][k] for k in ("winter_solstice", "equinox", "summer_solstice")}
        directions, _w = generate_tregenza_patches()
        meta = w4.epw_meta(root / params["weather"]["primary_epw"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tmp_dir = run_dir / "_tmp"
        dec = {"reference_day": days["summer_solstice"], "device": device,
               "thresholds_h": list(DEC_THRESHOLDS_H), "per_site": {}}
        try:
            for slug, d in SITE_DIRS.items():
                t0 = time.time()
                r = remarch_site(d, root / "runs" / WP04_RUN, run_dir / slug, tmp_dir,
                                 data_root=root, directions=directions, meta=meta, days=days, device=device)
                r["wall_s"] = round(time.time() - t0, 1)
                dec["per_site"][slug] = r
                print(f"december {slug}: {r['wall_s']} s, check {r['remarch_check_vs_stored']}", flush=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        summary["december"] = dec

    return summary


# ---------------------------------------------------------------------------
# round2_additions.md — every number below is read from the ledgers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def render_additions(new: dict, old: dict, new_dir: str, old_dir: str) -> str:
    e = {k: v["value"] for k, v in new["entries"].items()}
    old_e = old["entries"]
    changed = [k for k, v in old_e.items()
               if k not in new["entries"] or new["entries"][k]["value"] != v["value"]
               or new["entries"][k]["source"] != v["source"]]
    added = sorted(set(new["entries"]) - set(old_e))
    sites = list(SITE_DIRS)
    run_r2 = new["_meta"]["runs_of_record"]["round2"]
    run_xt = new["_meta"]["runs_of_record"]["crosstab"]
    L = [
        "# Round-2 additions to the WP-07 ledger",
        "",
        f"New ledger `runs/{new_dir}` · previous `runs/{old_dir}`. Producer: "
        f"`src/brisa_solar/wp07_round2.py` → `runs/{run_r2}/summary.json`; Table 1 class sizes from "
        f"`runs/{run_xt}/summary.json`. Every value here is read from ledger.json; percentages rounded for reading.",
        "",
        "## Before / after",
        "",
        f"- Entries before: {len(old_e)}; after: {len(new['entries'])}; added: {len(added)}.",
        f"- Pre-existing entries whose value or source changed: {len(changed)}"
        + (f" ({', '.join(changed)})" if changed else " — every pre-existing key holds its value and pointer."),
        "",
        "## 1. Circularity check (R2-M9) and Table 1 class sizes (N10)",
        "",
        "Keys: `crosstab.<site|pooled>.n<k>.{cells,point_deficit}` (Table 1 as built, k = 0–3); "
        "`xtab_strat.<site|pooled>.v<0|1>.o<0..2>.{cells,point_deficit,cell_deficit}` — v = vertical "
        "constraint holds, o = count of the lateral and directional constraints. Same cells, estimator "
        "(cell-weighted mean of 1 − share ≥ 2 h, winter) and drop rule (cells without a ground sample) as Table 1.",
        "",
        "| site | vertical | o=0 | o=1 | o=2 |",
        "|---|---|---|---|---|",
    ]
    for slug in sites + ["pooled"]:
        for v in ("v0", "v1"):
            cells = [f"{_pct(e[f'xtab_strat.{slug}.{v}.o{o}.point_deficit'])} (n={e[f'xtab_strat.{slug}.{v}.o{o}.cells']})"
                     for o in range(3)]
            L.append(f"| {slug} | {'yes' if v == 'v1' else 'no'} | " + " | ".join(cells) + " |")
    L += ["", "Table 1 class sizes (cells):", "", "| site | n=0 | n=1 | n=2 | n=3 |", "|---|---|---|---|---|"]
    for slug in sites + ["pooled"]:
        L.append(f"| {slug} | " + " | ".join(str(e[f"crosstab.{slug}.n{k}.cells"]) for k in range(4)) + " |")

    vario = {s: e[f"boot.{s}.range_m"] for s in sites}
    L += [
        "",
        "## 2. Spatial block bootstrap (R3 / E4)",
        "",
        f"Keys: `boot.<site>.share_below_2h_{{winter,equinox}}.{{estimate,lo,hi}}`, "
        "`boot.crosstab.<site|pooled>.n<k>.point_deficit.{lo,hi}`, `boot.xtab_strat.pooled.v<v>.o<o>.point_deficit.{lo,hi}`, "
        "`boot.<site>.{range_m,block_m,n_blocks}`, `boot.replicates`, `boot.seed`; the same intervals with fixed 100 m "
        "blocks under `boot100.*` (sensitivity).",
        "",
        f"Method: square blocks on absolute UTM coordinates, resampled with replacement within each site "
        f"(as many draws as the site has blocks); pooled cells sum the per-site draws, so site composition is held. "
        f"{e['boot.replicates']} replicates, seed {e['boot.seed']}, 2.5/97.5 percentile interval. Block side = the "
        f"site's practical range of the winter below-2 h indicator (first {VARIO_LAG_M:g} m lag whose semivariance "
        f"reaches {VARIO_SILL_FRACTION:g} × the sill, sill = mean semivariance at lags ≥ {VARIO_SILL_FROM_M:g} m, "
        f"{VARIO_SAMPLE} seeded points), rounded up to {BLOCK_ROUND_M:g} m: blocks shorter than the correlation range "
        "treat correlated neighbours as independent and narrow the interval. The equinox indicator was not used to set "
        "blocks; its range is longer at some sites, so equinox intervals may be optimistic.",
        "",
        "| site | range m | block m | blocks | below 2 h winter [95% CI] | width | equinox [95% CI] | 100 m-block winter CI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sites:
        w = [e[f"boot.{s}.share_below_2h_winter.{b}"] for b in ("estimate", "lo", "hi")]
        q = [e[f"boot.{s}.share_below_2h_equinox.{b}"] for b in ("estimate", "lo", "hi")]
        h = [e[f"boot100.{s}.share_below_2h_winter.{b}"] for b in ("lo", "hi")]
        L.append(f"| {s} | {vario[s]:g} | {e[f'boot.{s}.block_m']:g} | {e[f'boot.{s}.n_blocks']} | "
                 f"{_pct(w[0])} [{_pct(w[1])}, {_pct(w[2])}] | {100 * (w[2] - w[1]):.1f} pts | "
                 f"{_pct(q[0])} [{_pct(q[1])}, {_pct(q[2])}] | [{_pct(h[0])}, {_pct(h[1])}] |")
    L += ["", "| pooled Table 1 | point deficit | 95% CI (range blocks) | 95% CI (100 m) |", "|---|---|---|---|"]
    for k in range(4):
        L.append(f"| n={k} | {_pct(e[f'crosstab.pooled.n{k}.point_deficit'])} | "
                 f"[{_pct(e[f'boot.crosstab.pooled.n{k}.point_deficit.lo'])}, {_pct(e[f'boot.crosstab.pooled.n{k}.point_deficit.hi'])}] | "
                 f"[{_pct(e[f'boot100.crosstab.pooled.n{k}.point_deficit.lo'])}, {_pct(e[f'boot100.crosstab.pooled.n{k}.point_deficit.hi'])}] |")
    L += ["", "| pooled stratum | point deficit | 95% CI (range blocks) |", "|---|---|---|"]
    for v in ("v0", "v1"):
        for o in range(3):
            L.append(f"| vertical {'yes' if v == 'v1' else 'no'}, o={o} | {_pct(e[f'xtab_strat.pooled.{v}.o{o}.point_deficit'])} | "
                     f"[{_pct(e[f'boot.xtab_strat.pooled.{v}.o{o}.point_deficit.lo'])}, {_pct(e[f'boot.xtab_strat.pooled.{v}.o{o}.point_deficit.hi'])}] |")

    L += [
        "",
        "## 3. Medians on one observer set (N1)",
        "",
        "No new keys. The 1 m site-run winter median and IQR already exist as `site.<site>.ground.sun_h_winter.p50` "
        "(IQR: `.p25`, `.p75`), read from the same `ground.parquet` rows (same summary.json) as "
        "`site.<site>.ground.share_ge_2h_winter_solstice`. The `favela.<site>.sun_h_winter.median` keys are the 5 m "
        "citywide lattice and are the ones N1 flags.",
        "",
        "| site | 1 m median h | IQR h | share below 2 h | 5 m lattice median h (not this set) |",
        "|---|---|---|---|---|",
    ]
    for s in sites:
        L.append(f"| {s} | {e[f'site.{s}.ground.sun_h_winter.p50']:.2f} | {e[f'site.{s}.ground.sun_h_winter.p25']:.2f}–"
                 f"{e[f'site.{s}.ground.sun_h_winter.p75']:.2f} | {_pct(1 - e[f'site.{s}.ground.share_ge_2h_winter_solstice'])} | "
                 f"{e[f'favela.{s}.sun_h_winter.median']:.2f} |")

    r2 = json.loads((REPO_ROOT / "runs" / run_r2 / "summary.json").read_text())
    dec = r2["december"]
    L += [
        "",
        "## 4. December solstice direct sun (heat axis — data only, PI decision)",
        "",
        f"Keys: `dec.reference_day` ({e['dec.reference_day']}), `dec.<site>.ground.sun_h_december.{{p25,p50,p75}}`, "
        "`dec.<site>.ground.{share_ge_6h,share_ge_8h,share_lt2h_june_and_ge6h_december,share_lt2h_june_and_lt2h_december}`. "
        "All `reviewer-defence-only` until the PI rules on heat.",
        "",
        "Method: WP-04 stores binary patch visibility per point, not the marched horizon angles its sun hours are read "
        "against, so the December hours do NOT come from the stored matrix. Every stored 1 m ground observer was re-marched "
        "with the same engine, surface build, 1.5 m eye height and 500 m reach, and direct-sun hours read with the same "
        "`direct_sun_hours` (10-min sun positions, nearest patch azimuth). The re-march recomputed the June and equinox "
        "hours; mismatches against the stored columns: "
        + ", ".join(f"{s} {dec['per_site'][s]['remarch_check_vs_stored']['winter_solstice']['n_mismatch']}/"
                    f"{dec['per_site'][s]['remarch_check_vs_stored']['equinox']['n_mismatch']}" for s in sites)
        + f" (June/equinox). Re-march wall time {sum(dec['per_site'][s]['wall_s'] for s in sites):.0f} s on {dec['device']}. "
        f"pvlib hourly daylight count on {e['dec.reference_day']}: {dec['per_site'][sites[0]]['daylight_hours_pvlib_hourly']} h. "
        "Per-point hours: `runs/" + run_r2 + "/<site>/december_ground.parquet` (row, col join to the WP-04 ground.parquet).",
        "",
        "| site | Dec median h | IQR h | ≥ 6 h | ≥ 8 h | < 2 h June AND ≥ 6 h Dec | < 2 h June AND < 2 h Dec |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sites:
        b = f"dec.{s}.ground"
        L.append(f"| {s} | {e[f'{b}.sun_h_december.p50']:.2f} | {e[f'{b}.sun_h_december.p25']:.2f}–{e[f'{b}.sun_h_december.p75']:.2f} | "
                 f"{_pct(e[f'{b}.share_ge_6h'])} | {_pct(e[f'{b}.share_ge_8h'])} | "
                 f"{_pct(e[f'{b}.share_lt2h_june_and_ge6h_december'])} | {_pct(e[f'{b}.share_lt2h_june_and_lt2h_december'])} |")
    L += ["", f"## Added keys ({len(added)})", "", ", ".join(f"`{k}`" for k in added), ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--reps", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-december", action="store_true")
    ap.add_argument("--additions", nargs=2, metavar=("NEW_LEDGER_DIR", "OLD_LEDGER_DIR"),
                    help="write <NEW>/round2_additions.md comparing two runs/wp07_ledger_* dirs; no producer run")
    args = ap.parse_args()

    root = Path(args.repo_root)
    if args.additions:
        new_dir, old_dir = args.additions
        new = json.loads((root / "runs" / new_dir / "ledger.json").read_text())
        old = json.loads((root / "runs" / old_dir / "ledger.json").read_text())
        out = root / "runs" / new_dir / "round2_additions.md"
        out.write_text(render_additions(new, old, new_dir, old_dir))
        print(f"Wrote {out}")
        return 0
    run_dir = root / "runs" / ("wp07_round2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True)
    summary = build_summary(root, run_dir, reps=args.reps, seed=args.seed, december=not args.no_december)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"Wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
