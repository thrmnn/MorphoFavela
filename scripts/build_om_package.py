#!/usr/bin/env python3
"""Build the "Maré morphology, OM2" data package (v0.1) — Octopus LRP #2
("Street by street: explaining air temperature differences across streets
and over time in Complexo da Maré", lead Jingxue, PI Simone). Théo (PI) is
a SUPPORT contributor here, supplying street-form variables only — this
script and everything under src/om_package/ never compute or state a
temperature conclusion; that is the Octopus team's analysis, not ours.

Builds, per requested route (default OM2 only — the spec's target route;
OM1/OM3/OM4 share the exact same code path so pass --route ALL to also
build them):
  P-02 route points (1 m spacing, pedestrian height, stable IDs)
  P-03 buffer variables (5/10/20/50 m) — segment aggregation is a
       separate script, scripts/aggregate_om_points.py, since the segment
       length is the team's choice, not fixed at build time
  P-04 airborne form variables
  P-06 ventilation proxies
  P-05 shade — empty-schema table (campaign dates unknown; see
       src/om_package/shade.py)
  P-07 quality report
  P-08 data dictionary
  contact sheet PNG (OM2 only)
  neighbourhoods crossed (printed + saved)

Run:
    python scripts/build_om_package.py --route OM2 --out outputs/_packages/mare_om2/v0.1 --root /home/theo/SCL/SCR/MorphoFavela
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd

from src.om_package.buffers import BUFFER_RADII_M, compute_buffer_variables
from src.om_package.contact_sheet import build_contact_sheet
from src.om_package.dictionary import dictionary_dataframe
from src.om_package.formvars import compute_form_variables
from src.om_package.io_utils import Paths, write_table
from src.om_package.neighbourhoods import communities_crossed, join_communities
from src.om_package.package_docs import render_changelog, render_readme
from src.om_package.quality import write_quality_report
from src.om_package.routes import densify_route, route_length_m
from src.om_package.shade import build_empty_shade_table
from src.om_package.ventilation import compute_ventilation_proxies

ALL_ROUTES = ["OM_1", "OM_2", "OM_3", "OM_4"]


def build_one_route(om: str, paths: Paths, out_dir: Path, radii=BUFFER_RADII_M) -> dict:
    route_json = paths.route_json(om)
    points = densify_route(route_json)
    length_m = route_length_m(route_json)

    form = compute_form_variables(points, paths)
    vent = compute_ventilation_proxies(points, form["street_orientation_deg"].to_numpy(), paths)
    nbhd = join_communities(points, paths)

    joined = points.merge(form, on="point_id").merge(vent, on="point_id").merge(nbhd, on="point_id")

    buf = compute_buffer_variables(points, paths, radii=radii)
    joined_with_buf = joined.merge(buf, on="point_id")

    route_dir = out_dir / om.replace("OM_", "OM")
    written = write_table(joined_with_buf, route_dir, "points", geo=True)

    variable_cols = [c for c in joined_with_buf.columns if c not in ("point_id", "route_id", "seq", "distance_along_m", "height_m", "geometry")]
    quality = write_quality_report(joined_with_buf, variable_cols, route_dir)

    communities = communities_crossed(points, paths)

    return {
        "route_id": om,
        "length_m": length_m,
        "n_points": len(points),
        "communities_crossed": communities,
        "output_files": [str(p) for p in written],
        "quality_summary": {
            "n_points": quality["n_points"],
            "n_columns_checked": len(variable_cols),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="OM2", help="OM1|OM2|OM3|OM4|ALL")
    ap.add_argument("--out", default=None, help="output package dir (default: <root>/outputs/_packages/mare_om2/v0.1)")
    ap.add_argument("--root", default=str(Paths().root), help="MorphoFavela repo root (absolute)")
    ap.add_argument("--version", default="v0.1")
    args = ap.parse_args()

    paths = Paths(args.root)
    out_dir = Path(args.out) if args.out else paths.package_dir(args.version)
    out_dir.mkdir(parents=True, exist_ok=True)

    route_sel = args.route.upper().replace("OM_", "OM")
    if route_sel == "ALL":
        routes = ALL_ROUTES
    else:
        routes = [f"OM_{route_sel[2:]}"]

    manifest = {"built_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(paths.root), "routes": []}

    for om in routes:
        print(f"[build_om_package] {om} ...")
        result = build_one_route(om, paths, out_dir)
        manifest["routes"].append(result)
        print(f"  length_m={result['length_m']:.1f} n_points={result['n_points']} communities={result['communities_crossed']}")

    # P-05: shade — empty schema, campaign dates unknown (see src/om_package/shade.py)
    shade_dir = out_dir / "OM2" if "OM_2" in routes else out_dir
    shade_table = build_empty_shade_table()
    write_table(shade_table, shade_dir, "p05_building_shade")

    # P-08: data dictionary (package-wide, not per-route)
    dict_df = dictionary_dataframe()
    write_table(dict_df, out_dir, "p08_data_dictionary")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build_om_package] wrote manifest to {out_dir / 'manifest.json'}")

    (out_dir / "README.md").write_text(render_readme())
    (out_dir / "CHANGELOG.md").write_text(render_changelog())

    if "OM_2" in routes:
        om2_points_path = out_dir / "OM2" / "points.parquet"
        om2_df = pd.read_parquet(om2_points_path)
        contact_sheet_path = out_dir / "OM2" / "contact_sheet.png"
        build_contact_sheet(om2_df, contact_sheet_path, route_id="OM2")
        print(f"[build_om_package] contact sheet: {contact_sheet_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
