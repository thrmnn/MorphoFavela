#!/usr/bin/env python3
"""Audit of Mingze's July 2026 facade sun-hours CSV (aggregates only, no building IDs/coords).

    TMPDIR=/tmp python scripts/facade_audit_july.py [--out runs/facade_audit_july_<UTC>]

Writes CSVs + report.md into the run dir. Reads data/ only. Footprints are read with
fiona because the user-site pyogrio install is broken in this env.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.polygon import orient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.brisa_solar.wp05_full import match_favela_group  # noqa: E402

CSV = ROOT / "data/external/mingze/facade_20260710/rio_favelas_facade_sunhours_merged.csv"
FOOT = ROOT / "data/RJ/buildings_RJ_2019_utm.gpkg"
FAV = ROOT / "data/RJ/Favelas_Limit_2019.shp"
DATES = ["03-20", "06-21", "09-22", "12-21"]
SITE_MAP = {"mare": "Maré", "c. do alemao": "Complexo do Alemão", "rocinha": "Rocinha",
            "rio das pedras": "Rio das Pedras"}
SITES = ["Rio das Pedras", "Rocinha", "Complexo do Alemão", "Maré"]
# v3 Table 1 (mingze_facade_irradiation_v3_2026-09-24.md): bands, buildings, persistent-low (<0.5 kWh/m2/d on all 4 dates)
V3 = {"Rio das Pedras": (275810, 10720, 62.3), "Rocinha": (376208, 13766, 50.8),
      "Complexo do Alemão": (471449, 21720, 40.3), "Maré": (775224, 37165, 56.2)}
V3_ZERO = {"Rio das Pedras": 54.7}  # text: persistent exact zeros 17.2% (Vidigal) .. 54.7% (Rio das Pedras)


def md(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    d = df.copy()
    cols = [str(c) for c in d.columns]
    rows = []
    intcols = {i for i, c in enumerate(cols) if c in ("buildings", "bands")}
    for r in d.itertuples(index=False):
        rows.append([str(int(v)) if i in intcols and not pd.isna(v) else floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                     else ("PLACEHOLDER" if isinstance(v, float) and pd.isna(v) else str(v)) for i, v in enumerate(r)])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "runs" / ("facade_audit_july_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=True)
    T = {}

    raw = pd.read_csv(CSV)
    raw["site"] = raw["site"].str.strip().str.lower().map(SITE_MAP)
    assert raw["site"].notna().all()
    n_raw = len(raw)

    # ---------- Q1: collapse duplicates per date ----------
    key = ["site", "date", "building_id", "floor_id", "facade_key"]
    gb = raw.groupby(key, sort=False)
    col = gb.agg(sun=("sun_hours", "mean"), smin=("sun_hours", "min"), smax=("sun_hours", "max"),
                 n=("sun_hours", "size"), facade_id=("facade_id", "first"),
                 z_min=("z_min", "first"), z_max=("z_max", "first")).reset_index()
    dup = col.groupby(["site", "date"]).agg(
        rows_raw=("n", "sum"), bands=("n", "size"), dup_extra_rows=("n", lambda s: int((s - 1).sum())),
        dup_groups=("n", lambda s: int((s > 1).sum())),
        dup_groups_differing=("n", lambda s: 0)).reset_index()
    dg = col[col.n > 1].assign(diff=lambda d: d.smax != d.smin).groupby(["site", "date"])["diff"].sum()
    dup["dup_groups_differing"] = [int(dg.get((s, d), 0)) for s, d in zip(dup.site, dup.date)]
    T["q1_duplicates"] = dup
    T["q1_sun_hours_granularity"] = pd.DataFrame([dict(
        pct_values_multiple_of_1h=100 * (raw.sun_hours % 1 == 0).mean(), pct_values_multiple_of_0p5h=100 * (raw.sun_hours % 0.5 == 0).mean(),
        max_sun_hours=raw.sun_hours.max(), pct_deprived_flag_consistent_with_lt2=100 * ((raw.sun_hours < 2).astype(int) == raw.deprived_below_2h).mean())])
    del raw

    col["z"] = col.sun == 0
    col["lt2"] = col.sun < 2
    b = col.groupby(["site", "date"]).agg(
        n_bands=("sun", "size"), mean_h=("sun", "mean"), median_h=("sun", "median"),
        pct_lt2h=("lt2", lambda s: 100 * s.mean()), pct_zero=("z", lambda s: 100 * s.mean())).reset_index()
    b["site"] = pd.Categorical(b.site, SITES)
    T["q1_basics"] = b.sort_values(["site", "date"]).reset_index(drop=True)

    wide = col.pivot_table(index=["site", "building_id", "floor_id", "facade_key"], columns="date", values="sun")
    full = wide.dropna()
    per = []
    for s in SITES:
        w = wide.loc[s]
        f = full.loc[s]
        per.append(dict(site=s, bands_any_date=len(w), bands_all4_dates=len(f),
                        pct_persistent_zero=100 * (f == 0).all(axis=1).mean(),
                        pct_lt2h_all4=100 * (f < 2).all(axis=1).mean(),
                        pct_zero_on_0621=100 * (f["06-21"] == 0).mean(),
                        pct_persistent_zero_given_zero_0621=100 * (f == 0).all(axis=1).sum() / max((f["06-21"] == 0).sum(), 1)))
    T["q1_persistence"] = pd.DataFrame(per)

    # ---------- Q2: ID join ----------
    g = gpd.read_file(FOOT, engine="fiona")
    gi = g.set_index("OBJECTID")
    bld = col[col.date == "06-21"].groupby("site").building_id.unique()
    ids_all = np.unique(np.concatenate(bld.values))
    rows = []
    for fld in ["OBJECTID", "cod_projec", "cod_unico", "cod_edific", "cod_lote"]:
        v = pd.to_numeric(g[fld], errors="coerce").dropna().astype("int64")
        r = dict(field=fld, n_values=len(v), unique=bool(v.is_unique))
        for s in SITES:
            r[s] = 100 * np.isin(bld[s], v.values).mean()
        r["all_sites"] = 100 * np.isin(ids_all, v.values).mean()
        rows.append(r)
    T["q2_id_match"] = pd.DataFrame(rows)

    # validation: ID match alone is weak (dense integer range) -> check heights and location
    bz = col[col.date == "06-21"].groupby("building_id").agg(zmin=("z_min", "min"), zmax=("z_max", "max"),
                                                              site=("site", "first"), nsid=("facade_id", "max"))
    bz["nsid"] += 1
    j = bz.join(gi[["base", "altura"]], how="left")
    jshift = bz.join(gi[["base", "altura"]].rename(index=lambda i: i - 1), how="left")  # null: id off by one
    cen = gi.loc[bz.index, "geometry"].centroid
    polys = {s: match_favela_group(gpd.read_file(FAV, engine="fiona"), s)[0].geometry.union_all()
             for s in SITES} if False else None
    fav = gpd.read_file(FAV, engine="fiona")
    polys = {s: match_favela_group(fav, s)[0].geometry.union_all() for s in SITES}
    mare_e = gpd.read_file(ROOT / "data/maré/raw/ipp_territorios_sociais_territorio03.gpkg").to_crs(31983).union_all()
    val = []
    for s in SITES:
        m = j.site == s
        inside = shapely.contains_xy(polys[s], cen[m].x.values, cen[m].y.values)
        other = np.zeros(m.sum(), bool)
        for o in SITES:
            if o != s:
                other |= shapely.contains_xy(polys[o], cen[m].x.values, cen[m].y.values)
        ms = jshift.site == s
        val.append(dict(site=s, buildings=int(m.sum()),
                        pct_id_in_OBJECTID=100 * j.loc[m, "base"].notna().mean(),
                        pct_zmin_eq_base_0p05=100 * ((j.loc[m, "zmin"] - j.loc[m, "base"]).abs() < 0.05).mean(),
                        pct_height_eq_altura_0p01=100 * ((j.loc[m, "zmax"] - j.loc[m, "zmin"] - j.loc[m, "altura"]).abs() < 0.01).mean(),
                        null_idminus1_pct_zmin_eq_base=100 * ((jshift.loc[ms, "zmin"] - jshift.loc[ms, "base"]).abs() < 0.05).mean(),
                        pct_centroid_in_site_polygon=100 * inside.mean(),
                        pct_centroid_only_in_other_site_polygon=100 * ((~inside) & other).mean(),
                        pct_centroid_in_no_site_polygon=100 * ((~inside) & ~other).mean(),
                        pct_centroid_in_mare_E_outline=(100 * shapely.contains_xy(mare_e, cen[m].x.values, cen[m].y.values).mean()
                                                        if s == "Maré" else float("nan"))))
    T["q2_validation"] = pd.DataFrame(val)

    # ---------- Q3 ----------
    c6 = col[col.date == "06-21"]
    # side -> edge mapping test
    geom = gi.loc[bz.index, "geometry"]
    isp = (geom.geom_type == "Polygon").values
    nv = pd.Series(np.where(isp, [len(x.exterior.coords) - 1 if x.geom_type == "Polygon" else -1 for x in geom], -1), index=bz.index)
    exact = nv[(nv == bz.nsid) & (nv >= 4)].index
    cm = pd.DataFrame(dict(site=bz.site, d=bz.nsid - nv)).loc[nv[nv > 0].index]
    cm["sites_eq_vertices"] = cm.d == 0
    cmap_tab = cm.groupby("site").agg(buildings=("d", "size"), pct_nsides_eq_nvertices=("sites_eq_vertices", lambda s: 100 * s.mean()),
                                      mean_nsides_minus_nvertices=("d", "mean"), sd=("d", "std")).reset_index()
    rng = np.random.default_rng(1)
    pick = set(rng.choice(exact.values, min(6000, len(exact)), replace=False))
    up = col[(col.date == "06-21") & (col.floor_id >= 2) & col.building_id.isin(pick)]
    mm = up.groupby(["building_id", "facade_id"]).sun.mean().reset_index()
    corr = []
    for ccw in (True, False):
        for rev in (False, True):
            for k in range(4):
                xs, ys = [], []
                for bid, grp in mm.groupby("building_id"):
                    p = orient(gi.at[bid, "geometry"], 1.0 if ccw else -1.0)
                    arr = np.array(p.exterior.coords)
                    d = arr[1:] - arr[:-1]
                    n = np.c_[d[:, 1], -d[:, 0]]
                    n /= np.maximum(np.hypot(*n.T)[:, None], 1e-9)
                    if rev:
                        n = n[::-1]
                    n = np.roll(n, -k, axis=0)
                    xs.append(n[grp.facade_id.values, 1])
                    ys.append(grp.sun.values)
                corr.append(dict(orientation="CCW" if ccw else "CW", reversed=rev, start_offset=k,
                                 r_north_component_vs_sun_hours=float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1]),
                                 buildings=len(pick)))
    T["q3_side_edge_mapping_count"] = cmap_tab
    T["q3_side_edge_mapping_orientation_test"] = pd.DataFrame(corr)

    # building-level party-wall proxy (footprint adjacency; not side-resolved)
    ids = bz.index.values
    tg = shapely.make_valid(np.asarray(gi.loc[ids, "geometry"].values))
    ba = shapely.area(tg)
    other = np.asarray(gi["geometry"].values)
    tree = shapely.STRtree(other)
    oid = gi.index.values
    qi, qj = tree.query(tg, predicate="dwithin", distance=0.5)
    shared = np.zeros(len(ids))
    ovl = np.zeros(len(ids))
    bound = shapely.boundary(tg)
    perim = shapely.length(shapely.get_exterior_ring(np.where(shapely.get_type_id(tg) == 3, tg, shapely.get_geometry(tg, 0))))
    pairs = pd.DataFrame({"i": qi, "j": qj})
    pairs = pairs[oid[pairs.j] != ids[pairs.i]]
    oth_v = shapely.make_valid(other[pairs.j.values])
    inter = shapely.area(shapely.intersection(tg[pairs.i.values], oth_v))
    smaller = np.minimum(ba[pairs.i.values], shapely.area(oth_v))
    pairs["overlap"] = inter > 0.25 * smaller
    ov = pairs.groupby("i").overlap.any()
    ovl[ov.index.values] = ov.values
    keep = pairs[~pairs.overlap]
    for i, grp in keep.groupby("i"):
        u = shapely.union_all(shapely.buffer(shapely.make_valid(other[grp.j.values]), 0.5))
        shared[i] = shapely.length(shapely.intersection(bound[i], u))
    af = np.clip(shared / np.maximum(perim, 1e-9), 0, 1)
    bdf = pd.DataFrame(dict(building_id=ids, attached_frac=af, overlaps_other=ovl.astype(bool)))
    # per-building zero fractions
    wf = full.reset_index()
    wf["pz"] = (wf[DATES] == 0).all(axis=1)
    wf["z6"] = wf["06-21"] == 0
    wf["lt2all"] = (wf[DATES] < 2).all(axis=1)
    bb = wf.groupby(["site", "building_id"]).agg(n=("pz", "size"), pz=("pz", "mean"), z6=("z6", "mean"),
                                                 floors=("floor_id", "max")).reset_index().merge(bdf, on="building_id")
    bb["cls"] = pd.cut(bb.attached_frac, [-1, 0.02, 0.25, 0.5, 1.01], labels=["isolated(<2%)", "2-25%", "25-50%", ">=50%"])
    bb["w_pz"] = bb.pz * bb.n
    cl = bb.groupby(["site", "cls"], observed=True).agg(buildings=("n", "size"), bands=("n", "sum"), w=("w_pz", "sum"),
                                                        mean_attached_frac=("attached_frac", "mean")).reset_index()
    cl["pct_bands_persistent_zero"] = 100 * cl.w / cl.bands
    cl["site"] = pd.Categorical(cl.site, SITES)
    T["q3_zero_by_attachment_class"] = cl.drop(columns="w").sort_values(["site", "cls"])
    cr = bb.groupby("site").apply(lambda d: pd.Series(dict(
        buildings=len(d), pct_overlapping_other_footprint=100 * d.overlaps_other.mean(),
        pct_buildings_attached_ge25=100 * (d.attached_frac >= 0.25).mean(),
        pearson_attached_vs_persistent_zero_share=d.attached_frac.corr(d.pz),
        pct_bands_persistent_zero_isolated=100 * (d.loc[d.attached_frac < 0.02, "w_pz"].sum() / d.loc[d.attached_frac < 0.02, "n"].sum()),
        pct_bands_persistent_zero_all=100 * d.w_pz.sum() / d.n.sum()))).reset_index()
    T["q3_attachment_summary"] = cr

    fl = c6.assign(fg=np.minimum(c6.floor_id, 4)).groupby(["site", "fg"]).agg(
        bands=("sun", "size"), pct_zero_0621=("z", lambda s: 100 * s.mean()), pct_lt2h_0621=("lt2", lambda s: 100 * s.mean())).reset_index()
    fl["floor"] = fl.fg.map(lambda v: "4+" if v == 4 else str(v))
    wf["fg"] = np.minimum(wf.floor_id, 4)
    pf = wf.groupby(["site", "fg"]).pz.mean().mul(100).rename("pct_persistent_zero").reset_index()
    fl = fl.merge(pf, on=["site", "fg"]).drop(columns="fg")
    fl["site"] = pd.Categorical(fl.site, SITES)
    T["q3_zero_by_floor"] = fl.sort_values(["site", "floor"])

    wb = bb.groupby("site").apply(lambda d: pd.Series(dict(
        buildings=len(d), pct_bldg_all_bands_persistent_zero=100 * (d.pz == 1).mean(),
        pct_bldg_no_band_persistent_zero=100 * (d.pz == 0).mean(),
        pct_bldg_gt50pct_bands_persistent_zero=100 * (d.pz > 0.5).mean(),
        pct_bldg_all_bands_zero_0621=100 * (d.z6 == 1).mean(),
        pct_bands_persistent_zero_in_wholly_zero_bldgs=100 * d.loc[d.pz == 1, "n"].sum() / d.w_pz.sum()))).reset_index()
    T["q3_whole_building_zero"] = wb
    d = bb.copy()
    d["fg"] = pd.cut(d.floors, [-1, 0, 1, 3, 100], labels=["1 floor", "2 floors", "3-4 floors", "5+ floors"])
    T["q3_zero_by_building_height"] = d.groupby(["site", "fg"], observed=True).agg(
        buildings=("n", "size"), bands=("n", "sum"), w=("w_pz", "sum")).assign(
        pct_bands_persistent_zero=lambda x: 100 * x.w / x.bands).drop(columns="w").reset_index()

    # ---------- Q4 ----------
    q1p = T["q1_persistence"].set_index("site")
    cnt = c6.groupby("site").agg(bands_0621=("sun", "size"), buildings=("building_id", "nunique"))
    q4 = []
    for s in SITES:
        v3b, v3bl, v3pl = V3[s]
        q4.append(dict(site=s, july_bands_0621=int(cnt.loc[s, "bands_0621"]), v3_bands=v3b,
                       ratio=cnt.loc[s, "bands_0621"] / v3b, july_buildings=int(cnt.loc[s, "buildings"]), v3_buildings=v3bl,
                       july_pct_persistent_direct_zero=q1p.loc[s, "pct_persistent_zero"],
                       july_pct_lt2h_all4=q1p.loc[s, "pct_lt2h_all4"],
                       v3_pct_persistent_irradiation_zero=V3_ZERO.get(s, float("nan")),
                       v3_pct_persistent_irradiation_lt0p5=v3pl))
    T["q4_vs_v3"] = pd.DataFrame(q4)

    # ---------- Q5 ----------
    perim_s = pd.Series(perim, index=ids)
    c6 = c6.assign(nsid=c6.groupby("building_id").facade_id.transform("max") + 1)
    c6["width_proxy"] = c6.building_id.map(perim_s) / c6.nsid
    c6["area_proxy"] = c6.width_proxy * (c6.z_max - c6.z_min)
    c6["h"] = c6.z_max - c6.z_min
    q5 = []
    for s, d in c6.groupby("site"):
        q5.append(dict(site=s, bands=len(d), pct_lt2h_unweighted=100 * d.lt2.mean(),
                       pct_lt2h_height_weighted=100 * np.average(d.lt2, weights=d.h),
                       pct_lt2h_area_proxy_weighted=100 * np.average(d.lt2, weights=d.area_proxy),
                       pct_zero_unweighted=100 * d.z.mean(), pct_zero_area_proxy_weighted=100 * np.average(d.z, weights=d.area_proxy)))
    q5 = pd.DataFrame(q5)
    q5["site"] = pd.Categorical(q5.site, SITES)
    T["q5_area_weighting"] = q5.sort_values("site")

    for k, v in T.items():
        v.to_csv(out / f"{k}.csv", index=False)

    # ---------- report ----------
    A = lambda k, **kw: md(T[k], **kw)
    ex_best = T["q3_side_edge_mapping_orientation_test"].r_north_component_vs_sun_hours.abs().max()
    iso = T["q3_attachment_summary"]
    rep = f"""# Facade audit, Mingze July CSV (aggregates only)

Run: `{out.name}`. Script: `scripts/facade_audit_july.py`. Input rows: {n_raw:,}. Footprints: `data/RJ/buildings_RJ_2019_utm.gpkg` (EPSG:31983). Site polygons: `match_favela_group` on `Favelas_Limit_2019.shp` (Maré = definition A, 6 IPP polygons). All values below are computed by the script; no building IDs or coordinates are included.

## Q1 Basics (duplicates collapsed by mean per date on building_id, floor_id, facade_key)

Duplicates:

{A('q1_duplicates')}

Sun-hours granularity and flag consistency (all rows):

{A('q1_sun_hours_granularity')}

Per site x date:

{A('q1_basics')}

Persistence (bands present on all four dates):

{A('q1_persistence')}

## Q2 ID join (building_id -> footprint layer)

Share of Mingze building_id values found in each integer-like field:

{A('q2_id_match')}

OBJECTID matches 100%, but its values are a dense 1..N range, so a hit alone proves little. Validation: band z_min equals footprint `base`, and z_max - z_min equals `altura`; the null column shifts the ID by one. Centroid location against site polygons:

{A('q2_validation')}

## Q3 Zero diagnosis

Side index vs footprint edge: does the number of sides equal the number of footprint vertices?

{A('q3_side_edge_mapping_count')}

Orientation test on buildings where the counts match (upper floors, 06-21, {len(pick)} buildings): correlation of the outward-normal north component with sun_hours under every orientation/start-offset/reversal hypothesis. A real mapping would give a clearly positive r under one hypothesis (June sun is in the north). Largest |r| = {ex_best:.4f}.

{A('q3_side_edge_mapping_orientation_test', floatfmt='{:.4f}')}

Party-wall proxy at building level (share of the footprint perimeter within 0.5 m of another footprint, overlapping footprints excluded) against persistent-zero share of bands:

{A('q3_attachment_summary')}

{A('q3_zero_by_attachment_class')}

Zero share by floor (floor 0 = ground):

{A('q3_zero_by_floor')}

Whole-building zeros:

{A('q3_whole_building_zero')}

By building height (floors = max floor_id + 1 class):

{A('q3_zero_by_building_height')}

## Q4 July vs v3

{A('q4_vs_v3')}

v3 reports exact-zero persistence only as a range (17.2% Vidigal to 54.7% Rio das Pedras); other sites' exact-zero shares are PLACEHOLDER (not in the write-up). July bands are for 06-21 after collapse; v3 bands are collapsed across its own dates.

## Q5 Area weighting

No façade width in the CSV. Sides cannot be mapped to edges (Q3), so edge length x height is not available. Crude proxy: building exterior-ring perimeter / number of sides x (z_max - z_min); this assumes equal-width sides within a building.

{A('q5_area_weighting')}
"""
    q1 = T["q1_persistence"].set_index("site")
    q3w = T["q3_whole_building_zero"].set_index("site")
    q3a = T["q3_attachment_summary"].set_index("site")
    q2v = T["q2_validation"].set_index("site")
    q4t = T["q4_vs_v3"].set_index("site")
    q5t = T["q5_area_weighting"].set_index("site")
    rng_pz = f"{q1.pct_persistent_zero.min():.1f}% to {q1.pct_persistent_zero.max():.1f}%"
    rep += f"""
## What this means for using the facade layer in P1

1. The footprint join is sound: building_id is the footprint layer OBJECTID (z_min = base and height = altura for >= {q2v[['pct_zmin_eq_base_0p05']].min().iloc[0]:.1f}% of buildings in every site, vs a one-step ID shift that fails). The spatial check against site polygons is clean for Rocinha and Alemao; Maré falls inside definition A only {q2v.loc['Maré','pct_centroid_in_site_polygon']:.1f}% (inside the E outline: {q2v.loc['Maré','pct_centroid_in_mare_E_outline']:.1f}%), so Mingze's Maré extent is not definition A.
2. The 06-21 winter slice (share < 2 h: {q5t.pct_lt2h_unweighted.min():.1f}% to {q5t.pct_lt2h_unweighted.max():.1f}%) is internally consistent; the exact-zero share on 06-21 ({T['q1_basics'].query("date=='06-21'").pct_zero.min():.1f}% to {T['q1_basics'].query("date=='06-21'").pct_zero.max():.1f}%) is expected to include structurally unlit (poleward-facing) faces, so it is not itself an error rate.
3. The persistent-zero stratum (zero on all four dates, {rng_pz} by site; whole buildings with every band persistently zero: {q3w.pct_bldg_all_bands_persistent_zero.min():.1f}% to {q3w.pct_bldg_all_bands_persistent_zero.max():.1f}%, Rocinha highest) is the candidate geometry/join failure set. It is flat by floor and does not rise with building-level footprint attachment (Pearson r {q3a.pearson_attached_vs_persistent_zero_share.min():.2f} to {q3a.pearson_attached_vs_persistent_zero_share.max():.2f}), so the building-level party-wall proxy does not explain it; a side-resolved test is not possible with this CSV.
4. v3 and July look like the same band set (Rio das Pedras and Alemao band counts identical), yet v3 reports {q4t.loc['Rio das Pedras','v3_pct_persistent_irradiation_zero']:.1f}% persistent exact-zero irradiation at Rio das Pedras against {q4t.loc['Rio das Pedras','july_pct_persistent_direct_zero']:.1f}% persistent direct-sun zero here. Irradiation zero should be <= direct-sun zero, so one of the two runs has a failure mode; do not cite v3 zero-based medians (e.g. Rio das Pedras median 0.000) until resolved.
5. Area weighting cannot be done properly (no widths, no side-to-edge map). With the crude equal-side proxy the 06-21 share < 2 h moves by at most {(q5t.pct_lt2h_area_proxy_weighted - q5t.pct_lt2h_unweighted).abs().max():.2f} percentage points, which is not evidence either way; report band-count shares and label them so.

## Items to request from Mingze

- Per band: facade centroid x, y, z and outward normal (or the two end-point coordinates of the facade segment) in a stated CRS, so sides map to geometry and a party-wall/edge test is possible. Aggregates stay internal; per-building values are withheld from any release.
- Per band: facade width (or area) to allow area-weighted shares.
- The footprint source used (layer, CRS, any simplification or subdivision) and how facade_id is numbered; the CSV side counts equal the IPP footprint vertex count for only {T['q3_side_edge_mapping_count'].pct_nsides_eq_nvertices.min():.0f}% to {T['q3_side_edge_mapping_count'].pct_nsides_eq_nvertices.max():.0f}% of buildings depending on site.
- Whether the context geometry (neighbouring buildings, terrain) used in the simulation included all footprints around each site, and the Maré extent actually modelled (definition A, E, or other).
- The rule that produced duplicate (building_id, floor_id, facade_key) rows ({T['q1_duplicates'].eval('dup_extra_rows/rows_raw*100').min():.2f}% to {T['q1_duplicates'].eval('dup_extra_rows/rows_raw*100').max():.2f}% of rows by site-date; sun_hours differ within {T['q1_duplicates'].eval('dup_groups_differing/dup_groups*100').min():.0f}% to {T['q1_duplicates'].eval('dup_groups_differing/dup_groups*100').max():.0f}% of duplicate groups) and the Maré bands present on 06-21 but missing on 09-22 and 12-21 (band counts differ by date, see Q1).
- Per-band sun_hours and irradiation from the same run on the same band keys, so zero in irradiation can be checked against zero in direct sun band by band; plus a sample (aggregates reported only) of exact-zero bands inspected for hidden surfaces, flipped normals or result-join errors.
"""
    (out / "report.md").write_text(rep)
    print(out)


if __name__ == "__main__":
    main()
