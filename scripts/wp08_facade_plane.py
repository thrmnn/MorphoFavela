#!/usr/bin/env python3
"""WP-08 option A: the external façade package as a second plane, clipped to the P1 site polygons.

    TMPDIR=/tmp python scripts/wp08_facade_plane.py [--out runs/wp08_facade_plane_<UTC>]

Aggregates only: no building IDs, no coordinates. Sites in fixed order, never ranked.
Façade values are daily irradiation (kWh/m2/day); the ground layer is winter sun-hours.
The two are compared by rank association only: no pooled index, no shared threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.brisa_solar.wp05_full import match_favela_group  # noqa: E402

PKG = ROOT / "data/external/mingze/package_20261008"
FAV = ROOT / "data/RJ/Favelas_Limit_2019.shp"
WP04 = ROOT / "runs/wp04_sites_20261007T201123Z"
SITES = ["Vidigal", "Rocinha", "Complexo do Alemão", "Maré", "Rio das Pedras"]
FACADE_SLUG = {"Vidigal": "vidigal", "Rocinha": "rocinha", "Complexo do Alemão": "complexo_do_alemao",
               "Maré": "mare", "Rio das Pedras": "rio_das_pedras"}
P1_DIR = {"Vidigal": "vidigal", "Rocinha": "rocinha", "Complexo do Alemão": "complexo_do_alemao",
          "Maré": "maré", "Rio das Pedras": "riodaspedras"}
DATES = ["mar20_kwh_m2_day", "jun20_kwh_m2_day", "sep22_kwh_m2_day", "dec21_kwh_m2_day"]
SCREEN = 0.5
EPS = 1e-12
CELL_M = 10.0
MIN_ROBUST = 500
MIN_GROUND_PTS = 3
MIN_BANDS = 3
VARIANTS = {"with_zeros": False, "without_persistent_zeros": True}


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, x))


def stats(v: np.ndarray, w: np.ndarray) -> dict:
    if len(v) == 0:
        return {"bands": 0}
    return {
        "bands": int(len(v)),
        "area_m2": float(w.sum()),
        "median_count": float(np.median(v)),
        "p10_count": float(np.quantile(v, 0.1)),
        "p90_count": float(np.quantile(v, 0.9)),
        "share_below_screen_count": float((v < SCREEN).mean()),
        "median_area_weighted": wquantile(v, w, 0.5),
        "p10_area_weighted": wquantile(v, w, 0.1),
        "p90_area_weighted": wquantile(v, w, 0.9),
        "share_below_screen_area_weighted": float(w[v < SCREEN].sum() / w.sum()),
    }


def robust_storeys(counts: pd.Series, minimum: int = MIN_ROBUST) -> list[int]:
    out = []
    for s in range(1, int(counts.index.max()) + 2 if len(counts) else 1):
        if counts.get(s, 0) >= minimum:
            out.append(s)
        else:
            break
    return out


def first_sustained(flags: dict[int, bool]) -> int | None:
    ks = sorted(flags)
    for i, k in enumerate(ks):
        if flags[k] and (i == len(ks) - 1 or flags[ks[i + 1]]):
            return k
    return None


def load_facade(site: str) -> pd.DataFrame:
    d = pd.read_csv(PKG / f"facade_bands_{FACADE_SLUG[site]}.csv.gz", low_memory=False)
    d["geometry_mapping_ok"] = d["geometry_mapping_ok"].astype(str).str.upper().eq("TRUE")
    return d


def population(site: str, poly) -> tuple[pd.DataFrame, dict]:
    d = load_facade(site)
    n0 = len(d)
    e = d[d.wall_class == "exposed_candidate"]
    n1 = len(e)
    m = e[e.geometry_mapping_ok]
    n2 = len(m)
    inside = shapely.contains_xy(poly, m.x_utm_m.to_numpy(), m.y_utm_m.to_numpy())
    p = m[inside].copy()
    p["storey"] = p.floor_id + 1
    p["persistent_zero"] = (p[DATES].abs() <= EPS).all(axis=1)
    p["four_date_mean"] = p[DATES].mean(axis=1)
    drops = {"all_rows": n0, "exposed_candidate": n1, "dropped_not_exposed": n0 - n1,
             "mapped": n2, "dropped_unmapped": n1 - n2,
             "in_p1_polygon": int(len(p)), "dropped_outside_polygon": n2 - int(len(p)),
             "persistent_exact_zero_in_population": int(p.persistent_zero.sum())}
    return p, drops


def storey_tables(p: pd.DataFrame) -> dict:
    out = {}
    for vname, drop_zero in VARIANTS.items():
        q = p[~p.persistent_zero] if drop_zero else p
        v = {"all_storeys": {}, "by_storey": {}}
        for lab, col in (("june", "jun20_kwh_m2_day"), ("four_date_mean", "four_date_mean")):
            v["all_storeys"][lab] = stats(q[col].to_numpy(), q.band_area_m2.to_numpy())
            v["by_storey"][lab] = {
                str(s): stats(g[col].to_numpy(), g.band_area_m2.to_numpy())
                for s, g in q.groupby("storey")}
        cnt = q.groupby("storey").size()
        rs = robust_storeys(cnt)
        crossing = {"robust_storeys": rs}
        for wl in ("count", "area_weighted"):
            flags, shares = {}, {}
            for s in rs:
                g = q[q.storey == s]
                ok = (g.jun20_kwh_m2_day >= SCREEN).to_numpy()
                sh = ok.mean() if wl == "count" else g.band_area_m2.to_numpy()[ok].sum() / g.band_area_m2.sum()
                shares[str(s)] = float(sh)
                flags[s] = bool(sh > 0.5)
            crossing[f"share_at_or_above_screen_{wl}"] = shares
            crossing[f"first_sustained_majority_{wl}"] = first_sustained(flags)
        v["june_crossing"] = crossing
        out[vname] = v
    return out


def ground_cells(site: str) -> pd.DataFrame:
    g = pd.read_csv(ROOT / "outputs" / P1_DIR[site] / "geometry_indicators/per_patch_geometry_nodata0.csv",
                    usecols=["patch_id", "center_x", "center_y", "share_ge_2h_winter", "sun_h_winter_p50"])
    x0 = g.center_x.min() - CELL_M / 2
    y0 = g.center_y.min() - CELL_M / 2
    g["ix"] = np.floor((g.center_x - x0) / CELL_M).astype(np.int64)
    g["iy"] = np.floor((g.center_y - y0) / CELL_M).astype(np.int64)
    gr = pq.read_table(WP04 / P1_DIR[site] / "ground.parquet", columns=["x", "y"]).to_pandas()
    gr["ix"] = np.floor((gr.x - x0) / CELL_M).astype(np.int64)
    gr["iy"] = np.floor((gr.y - y0) / CELL_M).astype(np.int64)
    n = gr.groupby(["ix", "iy"]).size().rename("ground_points").reset_index()
    g = g.merge(n, on=["ix", "iy"], how="left")
    g["ground_points"] = g.ground_points.fillna(0).astype(int)
    g.attrs["x0"], g.attrs["y0"] = x0, y0
    return g


def association(p: pd.DataFrame, g: pd.DataFrame) -> dict:
    x0, y0 = g.attrs["x0"], g.attrs["y0"]
    f = p[p.floor_id == 0].copy()
    f["ix"] = np.floor((f.x_utm_m - x0) / CELL_M).astype(np.int64)
    f["iy"] = np.floor((f.y_utm_m - y0) / CELL_M).astype(np.int64)
    agg = f.groupby(["ix", "iy"]).agg(bands=("jun20_kwh_m2_day", "size"),
                                      facade_median_june=("jun20_kwh_m2_day", "median")).reset_index()
    j = g.merge(agg, on=["ix", "iy"], how="inner")
    assert j.facade_median_june.ge(0).all()
    assert (j.bands >= 1).all()
    assert j.bands.sum() <= len(f)
    j = j.dropna(subset=["share_ge_2h_winter"])
    j["ground_share_below_2h"] = 1.0 - j.share_ge_2h_winter
    res = {"p1_cells_total": int(len(g)), "p1_cells_with_ground_value": int(g.share_ge_2h_winter.notna().sum()),
           "storey1_bands_in_population": int(len(f)),
           "storey1_bands_in_a_p1_cell": int(
               f.merge(g[["ix", "iy"]], on=["ix", "iy"]).shape[0]),
           "cells_with_both": int(len(j))}
    for lab, k in (("unfiltered", j),
                   (f"min{MIN_GROUND_PTS}_ground_points_and_min{MIN_BANDS}_bands",
                    j[(j.ground_points >= MIN_GROUND_PTS) & (j.bands >= MIN_BANDS)])):
        r = {"cells": int(len(k))}
        for mlab, col in (("ground_share_below_2h", "ground_share_below_2h"),
                          ("median_winter_sun_hours", "sun_h_winter_p50")):
            if len(k) >= 3:
                rho = spearmanr(k[col], k.facade_median_june).statistic
                r[f"rho_{mlab}"] = None if np.isnan(rho) else float(rho)
            else:
                r[f"rho_{mlab}"] = None
        res[lab] = r
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out) if a.out else ROOT / "runs" / f"wp08_facade_plane_{utc}"
    out.mkdir(parents=True, exist_ok=True)

    fav = gpd.read_file(FAV, engine="fiona")
    summary = {"_utc": utc, "status": "PROVISIONAL: option A pre-computation, PI has not ruled",
               "source": "the external façade package",
               "population_rule": "wall_class == exposed_candidate AND geometry_mapping_ok, inside the P1 site polygon "
                                  "(match_favela_group on Favelas_Limit_2019.shp, Maré = definition A)",
               "screen_kwh_m2_day": SCREEN, "storey": "floor_id + 1",
               "sustained_definition": "majority (share > 0.5) of bands at or above the screen at the storey and at "
                                       "the next robust storey, unless it is the last robust storey",
               "robust_storey_min_bands": MIN_ROBUST,
               "robust_storeys_rule": "consecutive storeys from storey 1 with at least the minimum band count",
               "cell_grid": "P1 WP-06 10 m built-cell grid (per_patch_geometry_nodata0.csv; origin as in bin_ground_to_cells)",
               "ground_metric": "1 - share_ge_2h_winter (share of ground points below the 2 h winter floor); "
                                "median winter sun-hours (sun_h_winter_p50) also given",
               "facade_cell_metric": "median June irradiation of first-storey (floor_id 0) population bands, band-count",
               "cell_filter": f"at least {MIN_GROUND_PTS} ground points and at least {MIN_BANDS} façade bands (mirrors the package's own filter, ground points in place of street observers)",
               "association_inputs": {"wp04_run": WP04.name, "wp06_run": "wp06_geometry_20261007T201840Z"},
               "sites": {}}
    for s in SITES:
        poly = match_favela_group(fav, s)[0].geometry.union_all()
        p, drops = population(s, poly)
        assert p.jun20_kwh_m2_day.ge(0).all() and p[DATES].ge(0).all().all(), f"negative value in {s}"
        assert drops["all_rows"] - drops["dropped_not_exposed"] - drops["dropped_unmapped"] - drops["dropped_outside_polygon"] == drops["in_p1_polygon"]
        tabs = storey_tables(p)
        for v in tabs.values():
            for lab in v["by_storey"]:
                assert sum(t["bands"] for t in v["by_storey"][lab].values()) == v["all_storeys"][lab]["bands"]
                assert all(t["bands"] >= 1 for t in v["by_storey"][lab].values())
        assert tabs["with_zeros"]["all_storeys"]["june"]["bands"] - tabs["without_persistent_zeros"]["all_storeys"]["june"]["bands"] == drops["persistent_exact_zero_in_population"]
        summary["sites"][s] = {"population": drops, "storeys": tabs, "association_with_p1_ground": association(p, ground_cells(s))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / "report.md").write_text(report(summary, out.name))
    print(out)


def fmt(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def report(S: dict, name: str) -> str:
    L = [f"# WP-08 façade plane, option A (pre-computation; PI has not ruled)", "",
         f"Run `{name}`. Script `scripts/wp08_facade_plane.py`. Source: the external façade package (daily irradiation, kWh/m2/day, four representative dates). Aggregates only; sites in fixed order, not ranked.", "",
         f"Population: {S['population_rule']}.", "",
         "## Counts dropped at each step", "",
         "| site | all rows | not exposed | unmapped | outside P1 polygon | population | persistent exact zeros in population |", "|---|---|---|---|---|---|---|"]
    for s in SITES:
        p = S["sites"][s]["population"]
        L.append(f"| {s} | {p['all_rows']} | {p['dropped_not_exposed']} | {p['dropped_unmapped']} | {p['dropped_outside_polygon']} | {p['in_p1_polygon']} | {p['persistent_exact_zero_in_population']} |")
    for vname in VARIANTS:
        L += ["", f"## June, storey profile ({vname.replace('_', ' ')})", "",
              "Median is band-count (area-weighted in brackets); share below the 0.5 screen is band-count.", "",
              "| site | storey | bands | median | p10 | p90 | share below 0.5 |", "|---|---|---|---|---|---|---|"]
        for s in SITES:
            t = S["sites"][s]["storeys"][vname]
            a = t["all_storeys"]["june"]
            L.append(f"| {s} | all | {a['bands']} | {fmt(a['median_count'])} ({fmt(a['median_area_weighted'])}) | {fmt(a['p10_count'])} | {fmt(a['p90_count'])} | {fmt(a['share_below_screen_count'], 3)} |")
            for st, b in t["by_storey"]["june"].items():
                L.append(f"| {s} | {st} | {b['bands']} | {fmt(b['median_count'])} ({fmt(b['median_area_weighted'])}) | {fmt(b['p10_count'])} | {fmt(b['p90_count'])} | {fmt(b['share_below_screen_count'], 3)} |")
    L += ["", "The four-date mean, with the same statistics, is in `summary.json` (`by_storey.four_date_mean`).", "",
          "## June storey crossing", "",
          f"Definition: {S['sustained_definition']}. Storeys counted only if robust: {S['robust_storeys_rule']}, minimum {S['robust_storey_min_bands']} bands. 'none' means no sustained majority within the robust storeys.", "",
          "| site | variant | robust storeys | first sustained (band-count) | first sustained (area-weighted) |", "|---|---|---|---|---|"]
    for s in SITES:
        for vname in VARIANTS:
            c = S["sites"][s]["storeys"][vname]["june_crossing"]
            fc, fa = c["first_sustained_majority_count"], c["first_sustained_majority_area_weighted"]
            L.append(f"| {s} | {vname.replace('_', ' ')} | {len(c['robust_storeys'])} | {'none' if fc is None else fc} | {'none' if fa is None else fa} |")
    L += ["", "## Rank association with the P1 ground layer", "",
          f"Per 10 m cell of the P1 WP-06 built-cell grid. Ground: {S['ground_metric']}. Façade: {S['facade_cell_metric']}. Filter: {S['cell_filter']}. Rank association only; the ground layer is winter sun-hours and the façade layer is June irradiation, so absolute levels and thresholds are not comparable and no pooled index is formed.", "",
          "| site | cells (unfiltered) | rho below-2h share (unfiltered) | rho sun-hours (unfiltered) | cells (filtered) | rho below-2h share (filtered) | rho sun-hours (filtered) |", "|---|---|---|---|---|---|---|"]
    for s in SITES:
        A = S["sites"][s]["association_with_p1_ground"]
        fk = [k for k in A if k.startswith("min")][0]
        u, f = A["unfiltered"], A[fk]
        L.append(f"| {s} | {u['cells']} | {fmt(u['rho_ground_share_below_2h'], 3)} | {fmt(u['rho_median_winter_sun_hours'], 3)} | {f['cells']} | {fmt(f['rho_ground_share_below_2h'], 3)} | {fmt(f['rho_median_winter_sun_hours'], 3)} |")
    L += ["", "Positive rho for sun-hours means more ground sun goes with more façade irradiation; the below-2h share has the opposite sign by construction.", "",
          "## Caveats (from the intake audit)", "",
          "- Four representative dates, not a seasonal mean; 0.5 kWh/m2/day is the package's operational screen, not a validated threshold.",
          "- Sky model and ground reflectance are not stated in the package.",
          "- Persistent exact zeros are treated as artefacts; both variants are reported.",
          "- No convergence claim; the high-accuracy rerun has not been done.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
