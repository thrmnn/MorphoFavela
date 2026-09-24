"""Shared path/IO helpers for the OM2 morphology data package.

All data/outputs are read from an absolute ``--root`` (default the main
MorphoFavela checkout) since this package's worktree carries no data/
outputs/runs (gitignored, worktree-local).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

DEFAULT_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")
UTM23S = "EPSG:31983"


class Paths:
    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)

    # inputs
    @property
    def routes_dir(self) -> Path:
        return self.root / "data" / "maré" / "octopus" / "routes"

    def route_json(self, om: str) -> Path:
        return self.routes_dir / f"{om}_inferred_route.json"

    @property
    def buildings_mare(self) -> Path:
        return self.root / "data" / "maré" / "raw" / "buildings_mare.shp"

    @property
    def buildings_extended_300m(self) -> Path:
        return self.root / "data" / "maré" / "buildings_extended_300m.gpkg"

    @property
    def street_mare(self) -> Path:
        return self.root / "data" / "maré" / "raw" / "street_mare.shp"

    @property
    def wind_rose_json(self) -> Path:
        return self.root / "data" / "maré" / "wind_rose.json"

    @property
    def neighbourhoods_gpkg(self) -> Path:
        return self.root / "data" / "maré" / "neighbourhoods.gpkg"

    @property
    def features_grid(self) -> Path:
        return self.root / "outputs" / "maré" / "features" / "features_grid.parquet"

    @property
    def features_street(self) -> Path:
        return self.root / "outputs" / "maré" / "features" / "features_street.parquet"

    @property
    def svf_streets(self) -> Path:
        return self.root / "outputs" / "maré" / "svf_v2" / "svf_streets.gpkg"

    @property
    def hw_streets(self) -> Path:
        return self.root / "outputs" / "maré" / "morphometrics" / "canyon" / "hw_streets.gpkg"

    # output package root
    def package_dir(self, version: str = "v0.1") -> Path:
        return self.root / "outputs" / "_packages" / "mare_om2" / version


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, geo: bool = False) -> list[Path]:
    """Write a table as Parquet + CSV (+ GeoPackage if geo=True). Returns paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if geo:
        gpkg = out_dir / f"{stem}.gpkg"
        df.to_file(gpkg, driver="GPKG")
        written.append(gpkg)
        flat = pd.DataFrame(df.drop(columns="geometry"))
        flat["x"], flat["y"] = df.geometry.x, df.geometry.y
    else:
        flat = df
    pq = out_dir / f"{stem}.parquet"
    flat.to_parquet(pq, index=False)
    written.append(pq)
    csv = out_dir / f"{stem}.csv"
    flat.to_csv(csv, index=False)
    written.append(csv)
    return written


def read_gpkg(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)
