"""Shared path/IO helpers for the OM2 morphology data package.

All data/outputs are read from an absolute ``--root`` (default the main
MorphoFavela checkout) since this package's worktree carries no data/
outputs/runs (gitignored, worktree-local).
"""
from __future__ import annotations

import hashlib
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
    def dtm_extended_300m(self) -> Path:
        return self.root / "data" / "maré" / "dtm_extended_300m.tif"

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
    """Write a table as Parquet + CSV (+ GeoPackage if geo=True).

    When geo=True the Parquet is written from the GeoDataFrame itself (via
    geopandas' to_parquet), so it carries GeoParquet 'geo' metadata and a
    real geometry column — x/y are added alongside it (not instead of it)
    so a plain-pandas reader still gets flat coordinate columns without
    needing a GeoParquet-aware reader. The CSV export drops geometry (not
    CSV-representable) and keeps x/y. Returns paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if geo:
        gpkg = out_dir / f"{stem}.gpkg"
        df.to_file(gpkg, driver="GPKG")
        written.append(gpkg)

        geo_df = df.copy()
        geo_df["x"], geo_df["y"] = df.geometry.x, df.geometry.y
        pq = out_dir / f"{stem}.parquet"
        geo_df.to_parquet(pq, index=False)
        written.append(pq)

        csv_df = pd.DataFrame(geo_df.drop(columns="geometry"))
        csv = out_dir / f"{stem}.csv"
        csv_df.to_csv(csv, index=False)
        written.append(csv)
    else:
        pq = out_dir / f"{stem}.parquet"
        df.to_parquet(pq, index=False)
        written.append(pq)
        csv = out_dir / f"{stem}.csv"
        df.to_csv(csv, index=False)
        written.append(csv)
    return written


def read_gpkg(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def hash_tree(root: Path) -> dict[str, str]:
    """sha256 of every regular file under root, keyed by its path relative
    to root (POSIX separators) — used to build the manifest's per-file
    checksums. Excludes nothing; call it only after every other file in
    the package has been written."""
    out: dict[str, str] = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
