#!/usr/bin/env python3
"""Run a complete morphometric analysis audit for any area.

Orchestrates: data loading, SVF audit, 10m grid morphometrics,
publication figures, and PDF report generation.

Usage:
    python scripts/run_morphometric_audit.py --area vidigal
    python scripts/run_morphometric_audit.py --area rocinha --cell-size 15
    python scripts/run_morphometric_audit.py --area vidigal --skip-figures
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_output_dir, is_informal_area  # noqa: E402
from src.svf_v2.paths import resolve_paths, resolve_boundary  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("morphometric_audit")


def load_data(area: str, buildings_override: str = None, dtm_override: str = None):
    """Load all input data for the area.

    Optional overrides let callers substitute extended building/DTM files
    (e.g., with city-wide context for CFD domain coverage).
    """
    dtm_path, fp_path, roads_path = resolve_paths(area)
    boundary_path = resolve_boundary(area)

    if buildings_override:
        fp_path = Path(buildings_override)
        logger.info("Using OVERRIDE buildings: %s", fp_path)
    if dtm_override:
        dtm_path = Path(dtm_override)
        logger.info("Using OVERRIDE DTM: %s", dtm_path)

    logger.info("Loading building footprints: %s", fp_path)
    buildings = gpd.read_file(fp_path)

    # Normalize height columns
    if "altura" in buildings.columns and "height" not in buildings.columns:
        buildings["height"] = buildings["altura"]
    if "base" in buildings.columns and "base_height" not in buildings.columns:
        buildings["base_height"] = buildings["base"]
    if "base_height" in buildings.columns and "altura" in buildings.columns:
        buildings["top_height"] = buildings["base_height"] + buildings["altura"]

    # Filter invalid buildings
    if "height" in buildings.columns:
        before = len(buildings)
        buildings = buildings[buildings["height"] > 0].copy()
        if is_informal_area(area):
            buildings = buildings[buildings["height"] <= 20].copy()
        buildings = buildings[buildings.geometry.area >= 9.0].copy()
        logger.info("Filtered buildings: %d -> %d", before, len(buildings))

    logger.info("Loading roads: %s", roads_path)
    streets = gpd.read_file(roads_path)

    boundary = None
    if boundary_path is not None:
        logger.info("Loading boundary: %s", boundary_path)
        boundary = gpd.read_file(boundary_path)

    # Load existing SVF results — check new layout first, then legacy
    svf_dir = get_area_output_dir(area) / "morphometrics" / "svf"
    if not svf_dir.exists():
        svf_dir = get_area_output_dir(area) / "svf_v2"

    svf_points = None
    for svf_file in ["svf_grid.gpkg", "svf_streets.gpkg"]:
        svf_path = svf_dir / svf_file
        if svf_path.exists():
            logger.info("Loading SVF points: %s", svf_path)
            svf_points = gpd.read_file(svf_path)
            break

    # Scene STL for terrain check
    scene_stl = None
    for stl_candidate in [
        svf_dir / "scene.stl",
        svf_dir / "scene.vtk",
        *sorted(Path(dtm_path).parent.glob("*.stl")),
    ]:
        if stl_candidate.exists():
            scene_stl = stl_candidate
            break

    # Load street-level SVF segments (the primary SVF dataset)
    streets_svf = None
    svf_segments_path = svf_dir / "svf_streets_segments.gpkg"
    if svf_segments_path.exists():
        logger.info("Loading street SVF segments: %s", svf_segments_path)
        streets_svf = gpd.read_file(svf_segments_path)

    return buildings, streets, boundary, dtm_path, svf_points, scene_stl, streets_svf


def run_audit(
    area: str,
    cell_size: float = 10.0,
    skip_figures: bool = False,
    buildings_override: str = None,
    dtm_override: str = None,
    output_suffix: str = "",
):
    """Run the complete morphometric audit pipeline."""
    t0 = time.time()

    # ── 1. Load data ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MORPHOMETRIC AUDIT: %s%s", area, f" ({output_suffix})" if output_suffix else "")
    logger.info("=" * 60)

    buildings, streets, boundary, dtm_path, svf_points, scene_stl, streets_svf = (
        load_data(area, buildings_override=buildings_override, dtm_override=dtm_override)
    )
    n_buildings = len(buildings)
    logger.info("Buildings: %d, Streets: %d segments", n_buildings, len(streets))
    if streets_svf is not None:
        logger.info("Street SVF segments: %d", len(streets_svf))

    # ── 2. SVF Audit ────────────────────────────────────────────────
    logger.info("Running SVF audit...")
    from src.morphometry.audit import audit_svf

    audit_result = audit_svf(
        svf_points=svf_points,
        buildings=buildings,
        boundary=boundary,
        scene_stl_path=scene_stl,
    )
    logger.info(
        "SVF audit: mean=%.3f, %d issues (%d critical)",
        audit_result.mean,
        len(audit_result.issues),
        sum(1 for i in audit_result.issues if i.severity == "critical"),
    )

    # ── 3. Grid morphometrics ──────────────────────────────────────
    logger.info("Computing grid morphometrics (cell_size=%dm)...", cell_size)
    from src.morphometry.grid import compute_grid_morphometrics

    grid = compute_grid_morphometrics(
        buildings=buildings,
        dtm_path=dtm_path,
        cell_size=cell_size,
        svf_points=svf_points,
        streets=streets,
        boundary=boundary,
    )

    # ── 4. Output directories ──────────────────────────────────────
    morph_dir_name = "morphometrics" + (f"_{output_suffix}" if output_suffix else "")
    out_dir = get_area_output_dir(area) / morph_dir_name
    svf_dir_out = out_dir / "svf"
    buildings_dir = out_dir / "buildings"
    grid_dir = out_dir / "grid"
    fig_dir = out_dir / "figures"
    report_dir = out_dir / "report"
    for d in [svf_dir_out, buildings_dir, grid_dir, fig_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 4b. Copy SVF data into morphometrics/svf/ ─────────────────
    svf_src_dir = get_area_output_dir(area) / "svf_v2"
    if svf_src_dir.exists():
        import shutil

        for svf_file in svf_src_dir.glob("*"):
            dest = svf_dir_out / svf_file.name
            if not dest.exists():
                if svf_file.is_file():
                    shutil.copy2(svf_file, dest)
        logger.info("SVF data linked to morphometrics/svf/")

    # ── 4c. Copy building metrics into morphometrics/buildings/ ────
    bld_src = get_area_output_dir(area) / "morphology_metrics"
    if bld_src.exists():
        import shutil

        for bld_file in bld_src.glob("*"):
            dest = buildings_dir / bld_file.name
            if not dest.exists() and bld_file.is_file():
                shutil.copy2(bld_file, dest)
        logger.info("Building metrics linked to morphometrics/buildings/")

    # ── 5. Save grid data ─────────────────────────────────────────
    logger.info("Saving grid data...")
    grid.to_file(grid_dir / "grid_metrics.gpkg", driver="GPKG")

    # CSV export
    csv_cols = [
        "zone_id",
        "centroid_x",
        "centroid_y",
        "svf",
        "lambda_p",
        "lambda_f_mean",
        "lambda_f_max",
        "porosity",
        "sigma_h",
        "H_mean",
        "street_orientation_entropy",
        "slope_deg",
        "aspect_deg",
        "far",
        "building_count",
        "svf_count",
    ]
    available_cols = [c for c in csv_cols if c in grid.columns]
    grid[available_cols].to_csv(grid_dir / "grid_metrics.csv", index=False)
    logger.info("Saved %d cells to grid_metrics.gpkg and .csv", len(grid))

    # ── 6. Generate figures ───────────────────────────────────────
    figure_paths = {}

    if not skip_figures:
        logger.info("Generating publication figures...")
        from src.morphometry.figures import (
            figure_site_overview,
            figure_building_heights,
            figure_street_svf_map,
            figure_svf_map,
            figure_morphometric_panel,
            figure_svf_slope_scatter,
            figure_correlation_matrix,
            figure_distributions,
        )

        area_display = area.replace("_", " ").title()

        fig_specs = [
            (
                "site_overview",
                "fig01_site_overview.png",
                lambda p: figure_site_overview(
                    buildings, dtm_path, p, boundary=boundary, area_name=area_display
                ),
            ),
            (
                "building_heights",
                "fig02_building_heights.png",
                lambda p: figure_building_heights(
                    buildings, dtm_path, p, boundary=boundary, area_name=area_display
                ),
            ),
            (
                "street_svf",
                "fig03_street_svf.png",
                lambda p: (
                    figure_street_svf_map(
                        streets_svf,
                        buildings,
                        dtm_path,
                        p,
                        boundary=boundary,
                        area_name=area_display,
                    )
                    if streets_svf is not None
                    else None
                ),
            ),
            (
                "svf_map",
                "fig04_svf_map.png",
                lambda p: figure_svf_map(grid, buildings, p, area_name=area_display),
            ),
            (
                "morphometric_panel",
                "fig05_morphometric_panel.png",
                lambda p: figure_morphometric_panel(
                    grid, buildings, p, area_name=area_display
                ),
            ),
            (
                "svf_slope_scatter",
                "fig06_svf_slope_scatter.png",
                lambda p: figure_svf_slope_scatter(grid, p, area_name=area_display),
            ),
            (
                "correlation_matrix",
                "fig07_correlation_matrix.png",
                lambda p: figure_correlation_matrix(grid, p, area_name=area_display),
            ),
            (
                "distributions",
                "fig08_distributions.png",
                lambda p: figure_distributions(grid, p, area_name=area_display),
            ),
        ]

        for key, fname, gen_fn in fig_specs:
            try:
                p = fig_dir / fname
                gen_fn(p)
                if p.exists():
                    figure_paths[key] = p
            except Exception as e:
                logger.error("Figure %s failed: %s", key, e)

        logger.info("Generated %d / %d figures", len(figure_paths), len(fig_specs))

    # ── 7. Generate PDF report ────────────────────────────────────
    logger.info("Generating PDF report...")
    from src.morphometry.report import generate_pdf_report

    pdf_path = report_dir / "morphometric_report.pdf"
    generate_pdf_report(
        output_path=pdf_path,
        area=area,
        grid=grid,
        audit_result=audit_result,
        figure_paths=figure_paths,
        n_buildings=n_buildings,
        streets_svf=streets_svf,
        buildings=buildings,
        dtm_path=dtm_path,
    )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("AUDIT COMPLETE in %.1f seconds", elapsed)
    logger.info("  Grid data: %s", grid_dir / "grid_metrics.gpkg")
    logger.info("  Figures:   %s/", fig_dir)
    logger.info("  Report:    %s", pdf_path)
    logger.info("=" * 60)

    return grid, audit_result


def main():
    parser = argparse.ArgumentParser(
        description="Morphometric analysis audit for informal settlements",
    )
    parser.add_argument(
        "--area", required=True, help="Area name (e.g., vidigal, rocinha, riodaspedras)"
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=10.0,
        help="Grid cell size in metres (default: 10)",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation (data + report only)",
    )
    parser.add_argument(
        "--buildings",
        default=None,
        help="Override building footprints path (e.g., extended context file)",
    )
    parser.add_argument(
        "--dtm",
        default=None,
        help="Override DTM raster path (e.g., extended DTM)",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for output directory (e.g., 'extended' → morphometrics_extended/)",
    )
    args = parser.parse_args()

    run_audit(
        args.area,
        cell_size=args.cell_size,
        skip_figures=args.skip_figures,
        buildings_override=args.buildings,
        dtm_override=args.dtm,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
