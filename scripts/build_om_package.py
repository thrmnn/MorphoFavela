#!/usr/bin/env python3
"""Build the "Maré morphology, OM2" data package — Octopus LRP #2
("Street by street: explaining air temperature differences across streets
and over time in Complexo da Maré", lead Jingxue, PI Simone). Théo (PI) is
a SUPPORT contributor here, supplying street-form variables only — this
script and everything under src/om_package/ never compute or state a
temperature conclusion; that is the Octopus team's analysis, not ours.

Release scope (PI ruling 2026-09-24): the SHARED package path
(outputs/_packages/mare_om2/<version>/) contains OM2 only. OM1/OM3/OM4
share the exact same code path (pass --route ALL) but are written to an
INTERNAL build directory (outputs/_packages/_internal/mare_routes/<version>/)
that is never copied into the shared package.

Builds, per requested route:
  P-02 route points (1 m spacing, pedestrian height, stable IDs) +
       route_geometry_flag (within a building OR >10 m from the nearest
       street centreline — src/om_package/routes.py)
  P-03 buffer variables (5/10/20/50 m) — segment aggregation is a
       separate script, scripts/aggregate_om_points.py, since the segment
       length is the team's choice, not fixed at build time
  P-04 airborne form variables (incl. grid_cell_id)
  P-06 ventilation proxies (incl. the 8 lambda_f_<dir> columns)
  P-05 shade — empty-schema table (campaign dates unknown; see
       src/om_package/shade.py) — OM2/shared package only
  P-07 quality report (counts route_geometry_flag)
  P-08 data dictionary — OM2/shared package only
  contact sheet PNG (OM2 only)
  neighbourhoods crossed (printed + saved)
  manifest.json: package_version, crs, use_terms, relative paths, sha256
       per file (computed last, over every file this run wrote)
  outputs/_packages/mare_om2/index.html: the package page (stable URL
       across versions), rebuilt from this run's outputs by
       scripts/build_om_package_page.py — see that module for its content.

Run:
    python scripts/build_om_package.py --route OM2 --root /home/theo/SCL/SCR/MorphoFavela
    python scripts/build_om_package.py --route ALL --root /home/theo/SCL/SCR/MorphoFavela
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for build_om_package_page,
# a sibling module — needed when this file is loaded via importlib (tests)
# rather than run as `python scripts/build_om_package.py`.

import geopandas as gpd
import pandas as pd

from src.om_package.buffers import BUFFER_RADII_M, compute_buffer_variables
from src.om_package.dictionary import dictionary_dataframe
from src.om_package.figures import (
    build_map_form,
    build_map_shade,
    build_profiles,
    build_shade_calendar,
)
from src.om_package.formvars import compute_form_variables
from src.om_package.io_utils import Paths, hash_tree, write_table
from src.om_package.neighbourhoods import communities_crossed, join_communities
from src.om_package.package_docs import USE_TERMS, render_changelog, render_readme
from src.om_package.quality import write_quality_report
from src.om_package.routes import compute_route_geometry_flag, densify_route, route_length_m
from src.om_package.spec import render_conformance_markdown, write_conformance
from src.sites.territory import load_territory
from src.om_package.shade import (
    OM2_SHADE_MAX_DIST_M,
    build_empty_shade_table,
    compute_shade,
    infer_campaign_windows,
    point_horizon_profiles,
)
from src.om_package.ventilation import compute_ventilation_proxies

from build_om_package_page import build_page as build_om_package_page

ALL_ROUTES = ["OM_1", "OM_2", "OM_3", "OM_4"]
CRS = "EPSG:31983"


def route_output_dir(om: str, out_dir: Path, internal_dir: Path) -> Path:
    """OM2 goes to the shared package path; every other route goes to the
    internal build directory (PI ruling 2026-09-24: the shared package
    contains OM2 only)."""
    return out_dir if om == "OM_2" else internal_dir


def build_one_route(om: str, paths: Paths, out_dir: Path, radii=BUFFER_RADII_M) -> dict:
    """Build one route's P-02/P-03/P-04/P-06/P-07 outputs under out_dir.
    out_dir is either the shared package root (for OM2) or the internal
    build directory (for OM1/OM3/OM4) — output_files are reported relative
    to whichever out_dir was passed."""
    route_json = paths.route_json(om)
    points = densify_route(route_json)
    points["route_geometry_flag"] = compute_route_geometry_flag(points, paths).to_numpy()
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
        "output_files": [str(p.relative_to(out_dir)) for p in written],
        "quality_summary": {
            "n_points": quality["n_points"],
            "n_columns_checked": len(variable_cols),
            "route_geometry_flagged_points": quality.get("route_geometry_flagged_points"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="OM2", help="OM1|OM2|OM3|OM4|ALL — non-OM2 routes always go to the internal build dir")
    ap.add_argument("--out", default=None, help="shared package dir for OM2 (default: <root>/outputs/_packages/mare_om2/<version>)")
    ap.add_argument(
        "--internal-out",
        default=None,
        help="internal build dir for OM1/OM3/OM4 (default: <root>/outputs/_packages/_internal/mare_routes/<version>)",
    )
    ap.add_argument("--root", default=str(Paths().root), help="MorphoFavela repo root (absolute)")
    ap.add_argument("--version", default="v0.1.3")
    ap.add_argument(
        "--csv-dir",
        default=None,
        help="dir of raw Octopus campaign CSVs for real P-05 shade (default: <root>/data/maré/octopus/csv); "
        "empty-schema table ships if none found there",
    )
    args = ap.parse_args()

    paths = Paths(args.root)
    out_dir = Path(args.out) if args.out else paths.package_dir(args.version)
    internal_dir = (
        Path(args.internal_out)
        if args.internal_out
        else paths.root / "outputs" / "_packages" / "_internal" / "mare_routes" / args.version
    )

    route_sel = args.route.upper().replace("OM_", "OM")
    routes = ALL_ROUTES if route_sel == "ALL" else [f"OM_{route_sel[2:]}"]

    if "OM_2" in routes:
        out_dir.mkdir(parents=True, exist_ok=True)
    if any(r != "OM_2" for r in routes):
        internal_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "package_version": args.version,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "crs": CRS,
        "use_terms": USE_TERMS,
        "release_scope": "OM2 only. OM1/OM3/OM4 are built by the same code path into an internal directory outside this package.",
        "routes": [],
    }

    om2_df = None
    for om in routes:
        route_out_dir = route_output_dir(om, out_dir, internal_dir)
        print(f"[build_om_package] {om} -> {route_out_dir} ...")
        result = build_one_route(om, paths, route_out_dir)
        if om == "OM_2":
            manifest["routes"].append(result)
            om2_df = pd.read_parquet(route_out_dir / "OM2" / "points.parquet")
        else:
            print(f"  (internal-only, not part of the shared package)")
        print(f"  length_m={result['length_m']:.1f} n_points={result['n_points']} communities={result['communities_crossed']}")

    if om2_df is None:
        print("[build_om_package] OM2 not requested — nothing written to the shared package this run")
        return 0

    # P-05: shade. Real run when campaign CSVs exist under --csv-dir (v0.1.2:
    # the Zenodo_release/fixed_data pilot pull); empty-schema table otherwise
    # (no guessed demo run — see src/om_package/shade.py). OM2/shared only.
    csv_dir = Path(args.csv_dir) if args.csv_dir else paths.root / "data" / "maré" / "octopus" / "csv"
    csv_paths = sorted(csv_dir.glob("*.csv")) if csv_dir.exists() else []
    n_csv_pilot = len(csv_paths)
    n_campaign_dates = 0
    n_shade_rows = 0
    shade_fraction_pct = 0.0
    campaign_windows_df = None
    if csv_paths:
        print(f"[build_om_package] P-05: {n_csv_pilot} campaign CSVs found under {csv_dir} — computing real shade")
        # lat/lon for pvlib sun position: the OM2 route's own centroid, reprojected
        # WGS84 (EPSG:31983 -> 4326) — a single site-representative point, same
        # simplification WP-04/WP-05 make for one site's sun position; computed from
        # the actual route geometry, never a remembered/approximate coordinate.
        from pyproj import Transformer

        transformer = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
        mare_lon, mare_lat = transformer.transform(om2_df["x"].mean(), om2_df["y"].mean())
        om2_gdf = gpd.GeoDataFrame(
            om2_df[["point_id"]], geometry=gpd.points_from_xy(om2_df["x"], om2_df["y"]), crs=CRS
        )
        campaign_windows_df = infer_campaign_windows(csv_paths)
        write_table(campaign_windows_df, out_dir, "p05b_campaign_windows")
        horizon_deg, horizon_az = point_horizon_profiles(om2_gdf, paths, device="cuda", max_dist_m=OM2_SHADE_MAX_DIST_M)
        frames = []
        for _, row in campaign_windows_df.iterrows():
            d = str(row["date"])
            start_h = row["first_timestamp"].strftime("%H:00")
            end_h = min(row["last_timestamp"], row["last_timestamp"].normalize() + pd.Timedelta("23h59min")).strftime("%H:59")
            frames.append(
                compute_shade(
                    om2_gdf, [d], (start_h, end_h), step_min=5,
                    lat=mare_lat, lon=mare_lon, tz="UTC",
                    horizon_deg=horizon_deg, horizon_azimuths_deg=horizon_az,
                )
            )
        shade_table = pd.concat(frames, ignore_index=True)
        n_campaign_dates = len(campaign_windows_df)
        n_shade_rows = len(shade_table)
        shade_fraction_pct = round(100 * shade_table["shaded"].mean(), 1) if n_shade_rows else 0.0
        print(f"[build_om_package] P-05: {n_shade_rows} rows across {n_campaign_dates} campaign dates ({shade_fraction_pct}% shaded, tz=UTC)")
    else:
        shade_table = build_empty_shade_table()
    write_table(shade_table, out_dir, "p05_building_shade")

    # P-08: data dictionary (package-wide, not per-route). OM2/shared only.
    dict_df = dictionary_dataframe()
    write_table(dict_df, out_dir, "p08_data_dictionary")

    # Figures (PI, 2026-09-27): spatial result first (F1/F2, route overlaid
    # on the favela buildings), then the sampling along the route (F3/F4) —
    # replaces the old contact_sheet.py (route floating in blank space).
    try:
        buildings = gpd.read_file(paths.buildings_mare)
    except Exception as exc:  # pragma: no cover - missing source is a build-config error, not a figure bug
        print(f"[build_om_package] WARNING: could not load buildings_mare ({exc}); figures will ship without the building base layer")
        buildings = None
    try:
        territory = load_territory("maré", root=paths.root)
        subunits = territory.subunits
    except Exception as exc:  # pragma: no cover - same: a missing/broken territory registry entry, not a figure bug
        print(f"[build_om_package] WARNING: could not load Maré territory ({exc}); figures will ship without community outlines")
        subunits = None

    tz_label = "UTC" if n_shade_rows else "n/a (no campaign rows)"
    map_form_path = out_dir / "OM2" / "map_form.png"
    build_map_form(om2_df, buildings, subunits, map_form_path, route_id="OM2", version=args.version)
    print(f"[build_om_package] F1 map (form/SVF): {map_form_path}")

    map_shade_path = out_dir / "OM2" / "map_shade.png"
    build_map_shade(om2_df, shade_table, buildings, subunits, map_shade_path, route_id="OM2", version=args.version, tz=tz_label)
    print(f"[build_om_package] F2 map (shade): {map_shade_path}")

    profiles_path = out_dir / "OM2" / "profiles.png"
    build_profiles(om2_df, shade_table, profiles_path, dictionary_df=dict_df, route_id="OM2", version=args.version)
    print(f"[build_om_package] F3 profiles: {profiles_path}")

    shade_calendar_path = out_dir / "OM2" / "shade_calendar.png"
    build_shade_calendar(om2_df, shade_table, campaign_windows_df, shade_calendar_path, route_id="OM2", version=args.version)
    print(f"[build_om_package] F4 shade calendar: {shade_calendar_path}")

    # P-03/P-05, structural fix (PI, 2026-09-27): the aggregation script and
    # the shade join example now travel INSIDE the package, not just in the
    # repo's scripts/ — a recipient with only this directory can still
    # re-aggregate to any segment length and exercise the join example.
    shipped_dir = Path(__file__).resolve().parents[1] / "src" / "om_package" / "shipped"
    for shipped_name in ("aggregate_to_segments.py", "join_shade_example.py"):
        shutil.copyfile(shipped_dir / shipped_name, out_dir / "OM2" / shipped_name)
    print(f"[build_om_package] shipped P-03/P-05 scripts into {out_dir / 'OM2'}")

    n_om2_points = len(om2_df)
    n_route_geometry_flagged = int(om2_df["route_geometry_flag"].sum())
    lambda_p_ones = om2_df[om2_df["plan_density_lambda_p"] >= 1.0 - 1e-9]
    n_lambda_p_ones = len(lambda_p_ones)
    n_lambda_p_ones_flagged = int(lambda_p_ones["route_geometry_flag"].sum()) if n_lambda_p_ones else 0
    lambda_p_share_explained_pct = (
        round(100 * n_lambda_p_ones_flagged / n_lambda_p_ones, 1) if n_lambda_p_ones else 0.0
    )

    readme_kwargs = dict(
        n_om2_points=n_om2_points,
        n_route_geometry_flagged=n_route_geometry_flagged,
        n_lambda_p_ones=n_lambda_p_ones,
        n_lambda_p_ones_flagged=n_lambda_p_ones_flagged,
        lambda_p_share_explained_pct=lambda_p_share_explained_pct,
        n_csv_pilot=n_csv_pilot,
        n_campaign_dates=n_campaign_dates,
        n_shade_rows=n_shade_rows,
        shade_fraction_pct=shade_fraction_pct,
        shade_max_dist_m=OM2_SHADE_MAX_DIST_M,
    )
    # First pass: README/CHANGELOG without the conformance section, so
    # p00_spec_conformance can be computed over a package directory that
    # already has every other P-01..P-09 artefact (including a README with
    # its required headings) on disk.
    (out_dir / "README.md").write_text(render_readme(**readme_kwargs))
    (out_dir / "CHANGELOG.md").write_text(render_changelog())

    # P-00: mechanical conformance to the PI's package spec (P-01..P-09),
    # computed from the files just written — never typed by hand (see
    # src/om_package/spec.py).
    conf = write_conformance(out_dir)
    delivered = sum(1 for it in conf["items"] if it["status"] == "delivered")
    partial = sum(1 for it in conf["items"] if it["status"] == "partial")
    pending = sum(1 for it in conf["items"] if it["status"] == "pending")
    print(f"[build_om_package] P-00 spec conformance: {delivered} delivered, {partial} partial, {pending} pending (of {len(conf['items'])})")

    # Second pass: README with the conformance section filled in.
    (out_dir / "README.md").write_text(
        render_readme(conformance_section=render_conformance_markdown(conf) + "\n", **readme_kwargs)
    )

    # manifest: sha256 per file, computed last (over everything just written,
    # manifest.json itself does not exist yet so is never self-hashed)
    manifest["p05_shade"] = {
        "n_csv_pilot": n_csv_pilot,
        "n_campaign_dates": n_campaign_dates,
        "n_rows": n_shade_rows,
        "shade_fraction_pct": shade_fraction_pct,
        "tz": "UTC (labelling choice, campaign timezone UNRESOLVED)" if n_shade_rows else None,
        "max_dist_m": OM2_SHADE_MAX_DIST_M,
        "campaign_dates": [str(d) for d in campaign_windows_df["date"]] if campaign_windows_df is not None else [],
    }
    manifest["files"] = hash_tree(out_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build_om_package] wrote manifest to {out_dir / 'manifest.json'}")
    print(
        f"[build_om_package] route_geometry_flag: {n_route_geometry_flagged}/{n_om2_points} OM2 points flagged; "
        f"lambda_p=1.0 explained by flag: {n_lambda_p_ones_flagged}/{n_lambda_p_ones} ({lambda_p_share_explained_pct}%)"
    )

    page_path = build_om_package_page(paths.root)
    print(f"[build_om_package] rebuilt package page: {page_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
