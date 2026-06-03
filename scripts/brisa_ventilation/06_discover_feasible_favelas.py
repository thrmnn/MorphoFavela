"""Feasibility scan: which Rio favelas beyond the campaign-5 are ready
for the morphometric pipeline?

For each candidate favela in the citywide IPP cadaster, checks:
  - boundary polygon present in Favelas_Limit_2019.shp
  - building footprints present in buildings_RJ_2019.shp (with `altura`)
  - DTM_RJ.tif covers the boundary bbox (no nodata-dominated tiles)

The five campaign sites are excluded. Output:
  outputs/brisa_ventilation_fix/favela_feasibility.csv
  outputs/brisa_ventilation_fix/favela_feasibility.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

_ROOT = Path(__file__).resolve().parents[2]

RJ = _ROOT / "data" / "RJ"
BLD_SHP = RJ / "buildings_RJ_2019.shp"
DTM_TIF = RJ / "DTM_RJ.tif"
FAV_SHP = RJ / "Favelas_Limit_2019.shp"

CAMPAIGN_NAMES = {"Rocinha", "Rio das Pedras", "Vidigal", "Complexo do Alemão",
                  "Maré", "Complexo da Maré"}

OUT_DIR = _ROOT / "outputs" / "brisa_ventilation_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def assess(fav_row, dtm_ds) -> dict:
    """Return per-favela feasibility record."""
    name = fav_row["nome"]
    bairro = fav_row.get("bairro", "")
    ra = fav_row.get("ra", "")
    geom = fav_row.geometry
    area_km2 = float(geom.area / 1e6)

    minx, miny, maxx, maxy = geom.bounds

    # Building scan — bbox-filtered read of the citywide cadaster.
    try:
        bld = gpd.read_file(BLD_SHP, bbox=(minx, miny, maxx, maxy))
        bld = bld[bld.intersects(geom)].copy()
    except Exception as e:
        return {"name": name, "error_buildings": str(e), "feasible": False}
    n_bld = int(len(bld))
    if n_bld == 0:
        return {"name": name, "n_buildings": 0, "feasible": False,
                "reason_blocked": "no IPP buildings within boundary"}
    altura = bld.get("altura", pd.Series(dtype=float)).astype(float)
    n_altura = int(altura.gt(0).sum())
    h_med = float(altura[altura > 0].median()) if n_altura else float("nan")
    h_p95 = float(altura[altura > 0].quantile(0.95)) if n_altura else float("nan")

    # DTM scan — read window inside boundary bbox, count valid pixels.
    try:
        win = from_bounds(minx, miny, maxx, maxy, dtm_ds.transform)
        arr = dtm_ds.read(1, window=win, boundless=True, fill_value=dtm_ds.nodata or -9999.0)
        nd = dtm_ds.nodata if dtm_ds.nodata is not None else -9999.0
        valid = np.isfinite(arr) & (arr != nd) & (np.abs(arr) < 1e10)
        dtm_coverage = float(valid.mean()) if arr.size else 0.0
        z_med = float(np.median(arr[valid])) if valid.any() else float("nan")
        z_range = (float(arr[valid].max() - arr[valid].min())
                   if valid.any() else float("nan"))
    except Exception as e:
        return {"name": name, "error_dtm": str(e), "feasible": False}

    feasible = (n_bld >= 500 and n_altura / max(n_bld, 1) >= 0.5
                and dtm_coverage >= 0.95)
    reason = None
    if not feasible:
        reasons = []
        if n_bld < 500:
            reasons.append(f"only {n_bld} buildings")
        if n_altura / max(n_bld, 1) < 0.5:
            reasons.append(
                f"only {100 * n_altura / max(n_bld, 1):.0f}% have altura"
            )
        if dtm_coverage < 0.95:
            reasons.append(f"DTM coverage {100 * dtm_coverage:.0f}%")
        reason = "; ".join(reasons)

    return {
        "name": name,
        "bairro": bairro,
        "ra": ra,
        "area_km2": area_km2,
        "n_buildings": n_bld,
        "frac_with_height": float(n_altura / max(n_bld, 1)),
        "height_median_m": h_med,
        "height_p95_m": h_p95,
        "dtm_coverage": dtm_coverage,
        "dtm_z_median_m": z_med,
        "dtm_z_range_m": z_range,
        "is_hillside": z_range > 30.0 if np.isfinite(z_range) else False,
        "feasible": bool(feasible),
        "reason_blocked": reason,
    }


def main() -> None:
    favs = gpd.read_file(FAV_SHP)
    sel_csv = pd.read_csv(RJ / "selected_favelas" / "selected_favelas.csv")
    print(f"Scanning {len(sel_csv)} pre-ranked favelas, "
          f"excluding {len(CAMPAIGN_NAMES)} campaign sites...")

    dtm_ds = rasterio.open(DTM_TIF)

    rows = []
    for _, sel_row in sel_csv.iterrows():
        name = sel_row["nome"]
        if name in CAMPAIGN_NAMES:
            continue
        match = favs[favs["nome"] == name]
        if match.empty:
            rows.append({"name": name, "feasible": False,
                         "reason_blocked": "no boundary in Favelas_Limit_2019"})
            continue
        if len(match) > 1:
            match = match.iloc[[match.geometry.area.idxmax()]]
        rec = assess(match.iloc[0], dtm_ds)
        rec["ipp_rank_building_count"] = int(sel_row["building_count"])
        rows.append(rec)
        flag = "FEAS" if rec.get("feasible") else "BLOCK"
        extra = ""
        if rec.get("feasible"):
            extra = (
                f"n={rec['n_buildings']:>5d}  "
                f"h_med={rec['height_median_m']:.1f}m  "
                f"z_range={rec['dtm_z_range_m']:.0f}m  "
                f"hillside={'Y' if rec['is_hillside'] else 'N'}"
            )
        else:
            extra = rec.get("reason_blocked", "")
        print(f"  [{flag}] {name:35s}  {extra}")

    dtm_ds.close()
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "favela_feasibility.csv", index=False)
    (OUT_DIR / "favela_feasibility.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str)
    )

    feas = df[df["feasible"]].sort_values("ipp_rank_building_count", ascending=False)
    hill = feas[feas["is_hillside"]] if "is_hillside" in feas.columns else feas
    print()
    print(f"FEASIBLE: {len(feas)} / {len(df)} candidates")
    print(f"  of which hillside (DTM z-range > 30 m): {len(hill)}")
    if not feas.empty:
        print("\nTop 10 feasible by building count:")
        print(feas.head(10)[
            ["name", "bairro", "n_buildings", "height_median_m",
             "dtm_z_range_m", "is_hillside"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
