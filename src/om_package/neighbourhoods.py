"""Which Maré community/ies each OM2 point falls in (data/maré/neighbourhoods.gpkg)."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .io_utils import Paths


def join_communities(points_gdf: gpd.GeoDataFrame, paths: Paths) -> pd.DataFrame:
    nbhd = gpd.read_file(paths.neighbourhoods_gpkg)[["community", "geometry"]]
    joined = gpd.sjoin(points_gdf[["point_id", "geometry"]], nbhd, how="left", predicate="within")
    joined = joined.drop_duplicates(subset="point_id", keep="first")
    return joined[["point_id", "community"]].rename(columns={"community": "neighbourhood"}).reset_index(drop=True)


def communities_crossed(points_gdf: gpd.GeoDataFrame, paths: Paths) -> list[str]:
    df = join_communities(points_gdf, paths)
    names = sorted(n for n in df["neighbourhood"].dropna().unique())
    return names
