"""OM2 compute stage "walks" (P-11 tables, P-12): wind regime tables, the
walks table with its airport-wind tag, and walk_points. Files written to
the stage work directory: walks.{parquet,csv}, walk_points.{parquet,csv},
wind_regimes.csv, wind_regime_by_hour.csv."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import p10_p11, walk_tables
from .io_utils import write_table
from .wind_regimes import load_campaign, tag_walks


def walks_stage(work: Path, *, root: Path, route_points: pd.DataFrame, walks: pd.DataFrame, fixes: pd.DataFrame,
                season: dict, regimes: list[dict], horizon_tab: pd.DataFrame, horizon_deg, azimuths_deg,
                lat: float, lon: float, log=print) -> dict:
    regimes_tbl = p10_p11.wind_regimes_table(season)
    regimes_tbl.to_csv(work / "wind_regimes.csv", index=False)
    by_hour_tbl = p10_p11.regime_by_hour_table(season, root)
    by_hour_tbl.to_csv(work / "wind_regime_by_hour.csv", index=False)

    tags = tag_walks(walks, load_campaign(root), season["campaign"])
    walks_tbl = walk_tables.walks_table(walks, tags)
    write_table(walks_tbl, work, "walks")
    regime_measures = [f"{stem}_{g['slug']}" for g in regimes for stem in p10_p11.REGIME_MEASURE_STEMS]
    t12 = time.time()
    p12 = walk_tables.walk_points_table(route_points, fixes, walks, horizon_tab, horizon_deg, azimuths_deg,
                                        regime_measures=regime_measures, lat=lat, lon=lon)
    write_table(p12, work, "walk_points")
    log(f"[build_om_package] P-12: {len(walks_tbl)} walks, {len(p12)} walk-point rows ({time.time() - t12:.0f} s); "
        f"walks tagged: {walks_tbl['wind_regime'].value_counts().to_dict()}")
    return {
        "wind_regimes_records": regimes_tbl.to_dict(orient="records"),
        "wind_regime_by_hour": by_hour_tbl,
        "walks_summary": {
            "n_walks": int(len(walks_tbl)), "n_partial": int(walks_tbl["partial"].sum()),
            "n_walk_point_rows": int(len(p12)),
            "tagged_by_regime": walks_tbl["wind_regime"].value_counts().to_dict(),
        },
    }
