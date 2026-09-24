"""Where Maré sits in the citywide sky-view and irradiation distributions under
each candidate definition of "Maré", plus the per-community breakdown.

Reads the WP-05 run of record and data/maré/neighbourhoods.gpkg; writes
runs/mare_definitions_<UTC>/summary.json. Definition A is the one the WP-07
ledger uses, and is checked against it so a drift in the matching rule fails
loudly rather than silently comparing two different baselines.

    python scripts/mare_definition_sensitivity.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brisa_solar.wp05_full import favela_summary, match_favela_group  # noqa: E402
from src.brisa_solar.wp07_ledger import RUN_OF_RECORD  # noqa: E402

FAVELAS = ROOT / "data" / "RJ" / "Favelas_Limit_2019.shp"
NEIGHBOURHOODS = ROOT / "data" / "maré" / "neighbourhoods.gpkg"
IPP_OUTLINE = ROOT / "data" / "maré" / "raw" / "ipp_territorios_sociais_territorio03.gpkg"


def _summ(sub: pd.DataFrame, city_svf, city_kwh) -> dict:
    return {"svf": favela_summary(sub["svf"].to_numpy(), city_svf),
            "kwh_m2": favela_summary(sub["kwh_m2"].to_numpy(), city_kwh)}


def _within(df: pd.DataFrame, geom) -> pd.DataFrame:
    x0, y0, x1, y1 = geom.bounds
    box = df[(df.x >= x0) & (df.x <= x1) & (df.y >= y0) & (df.y <= y1)]
    return box[shapely.contains_xy(geom, box.x.to_numpy(), box.y.to_numpy())]


def main() -> int:
    run = ROOT / "runs" / RUN_OF_RECORD["wp05"]
    df = pd.read_parquet(run / "wp05_full.parquet", columns=["x", "y", "favela_id", "svf", "kwh_m2"])
    city_svf, city_kwh = df["svf"].to_numpy(), df["kwh_m2"].to_numpy()
    fav = gpd.read_file(FAVELAS)
    comm = gpd.read_file(NEIGHBOURHOODS, layer="communities")

    a_polys, _ = match_favela_group(fav, "Maré")
    b_polys = pd.concat([a_polys] + [match_favela_group(fav, "Parque Roquete Pinto")[0]])
    defs = {
        "A_ipp_complexo_mare": ("IPP 2019 polygons with complexo == 'Maré' (WP-07 ledger definition)",
                                df[df.favela_id.isin(a_polys.cod_favela.astype(int))]),
        "B_plus_roquete_pinto": ("A + the complexo 'Parque Roquete Pinto' (Roquete Pinto, Ramos)",
                                 df[df.favela_id.isin(b_polys.cod_favela.astype(int))]),
        "C_all_ipp_in_bairro": ("every IPP 2019 favela polygon in bairro Maré",
                                df[df.favela_id.isin(fav[fav.bairro == "Maré"].cod_favela.astype(int))]),
        "D_sixteen_communities": ("union of the 16 communities (data/maré/neighbourhoods.gpkg)",
                                  _within(df, comm.union_all())),
        "E_ipp_complex_outline": ("IPP Territórios Sociais outline of the complex (territory 03) — the site study area since 2026-09-24",
                                  _within(df, gpd.read_file(IPP_OUTLINE).to_crs(31983).union_all())),
    }
    out = {"_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "wp05_run": RUN_OF_RECORD["wp05"], "definitions": {}, "per_community": {}}
    for key, (label, sub) in defs.items():
        out["definitions"][key] = {"label": label, **_summ(sub, city_svf, city_kwh)}

    ledger_dist = json.loads(next(run.glob("*distribution*.json")).read_text())["study_favelas"]["Maré"]
    a = out["definitions"]["A_ipp_complexo_mare"]
    if a["kwh_m2"]["n"] != ledger_dist["kwh_m2"]["n"] or not np.isclose(a["kwh_m2"]["median"], ledger_dist["kwh_m2"]["median"]):
        raise SystemExit("definition A no longer reproduces the run of record's Maré summary")

    for _, r in comm.iterrows():
        out["per_community"][r["community"]] = {
            "match": r["match"], "source_parts": r["source_parts"],
            **_summ(_within(df, r.geometry), city_svf, city_kwh)}

    dest = ROOT / "runs" / f"mare_definitions_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    dest.mkdir(parents=True)
    (dest / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for k, v in out["definitions"].items():
        print(f"{k:24s} n={v['kwh_m2']['n']:6d} svf pct {v['svf']['citywide_percentile_position']:5.1f}  kWh pct {v['kwh_m2']['citywide_percentile_position']:5.1f}")
    print(dest / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
