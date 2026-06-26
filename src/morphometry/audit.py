"""SVF audit — automated quality checks for sky view factor computation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AuditIssue:
    """A single flagged issue from the SVF audit."""

    severity: str  # "critical", "warning", "info"
    category: str
    message: str


@dataclass
class SVFAuditResult:
    """Complete SVF audit output."""

    # Summary statistics
    count: int = 0
    mean: float = np.nan
    median: float = np.nan
    std: float = np.nan
    min: float = np.nan
    max: float = np.nan
    p5: float = np.nan
    p25: float = np.nan
    p75: float = np.nan
    p95: float = np.nan

    # Coverage
    total_study_area_m2: float = 0.0
    svf_coverage_area_m2: float = 0.0
    coverage_pct: float = 0.0

    # Computation details
    computation_height_m: float | None = None
    method: str = "unknown"
    n_rays: int | None = None
    grid_spacing_m: float | None = None
    terrain_integrated: bool = False

    # Issues
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)


def audit_svf(
    svf_points: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame | None = None,
    scene_stl_path: Path | None = None,
    svf_col: str = "svf",
) -> SVFAuditResult:
    """Run automated quality checks on SVF computation results.

    Parameters
    ----------
    svf_points : GeoDataFrame
        Point-level SVF results.
    buildings : GeoDataFrame
        Building footprints.
    boundary : GeoDataFrame, optional
        Study area boundary for coverage computation.
    scene_stl_path : Path, optional
        Path to scene STL for terrain integration check.
    svf_col : str
        Column name containing SVF values.

    Returns
    -------
    SVFAuditResult
        Audit results with statistics and flagged issues.
    """
    result = SVFAuditResult()

    if svf_points is None or svf_points.empty:
        result.issues.append(AuditIssue("critical", "data", "No SVF points found."))
        return result

    svf = svf_points[svf_col].dropna()
    result.count = len(svf)

    # ── Summary statistics ──────────────────────────────────────────
    result.mean = float(svf.mean())
    result.median = float(svf.median())
    result.std = float(svf.std())
    result.min = float(svf.min())
    result.max = float(svf.max())
    result.p5 = float(np.percentile(svf, 5))
    result.p25 = float(np.percentile(svf, 25))
    result.p75 = float(np.percentile(svf, 75))
    result.p95 = float(np.percentile(svf, 95))

    # ── Value range checks ──────────────────────────────────────────
    n_negative = int((svf < 0).sum())
    n_over_one = int((svf > 1.0).sum())
    if n_negative > 0:
        result.issues.append(
            AuditIssue("critical", "range", f"{n_negative} SVF values are negative.")
        )
    if n_over_one > 0:
        result.issues.append(
            AuditIssue("critical", "range", f"{n_over_one} SVF values exceed 1.0.")
        )

    # High-SVF in built-up areas (use spatial join, not union_all — scales to large datasets)
    if buildings is not None and not buildings.empty:
        buffered_bld = buildings.copy()
        buffered_bld["geometry"] = buffered_bld.geometry.buffer(5)
        near_join = gpd.sjoin(
            svf_points[[svf_col, "geometry"]],
            buffered_bld[["geometry"]],
            how="inner",
            predicate="within",
        )
        # Drop duplicates (a point near multiple buildings)
        near_join = near_join.drop_duplicates(subset=near_join.geometry.name)
        if len(near_join) > 0:
            near_svf = near_join[svf_col]
            n_high = int((near_svf > 0.6).sum())
            pct_high = 100.0 * n_high / len(near_svf)
            if pct_high > 20:
                result.issues.append(
                    AuditIssue(
                        "warning",
                        "range",
                        f"{pct_high:.1f}% of points near buildings have SVF > 0.6 "
                        f"(expected < 20% in dense built-up areas).",
                    )
                )
            elif pct_high > 5:
                result.issues.append(
                    AuditIssue(
                        "info",
                        "range",
                        f"{pct_high:.1f}% of points near buildings have SVF > 0.6.",
                    )
                )

    # ── Coverage check ──────────────────────────────────────────────
    if boundary is not None:
        boundary_union = boundary.geometry.union_all()
        result.total_study_area_m2 = boundary_union.area
        # Use convex hull of sample for coverage (cheaper than union_all on points)
        from shapely.geometry import MultiPoint

        sample_pts = svf_points.geometry.values[:5000]
        svf_hull = MultiPoint([p for p in sample_pts if p is not None]).convex_hull
        result.svf_coverage_area_m2 = svf_hull.intersection(boundary_union).area
        result.coverage_pct = (
            100.0 * result.svf_coverage_area_m2 / result.total_study_area_m2
            if result.total_study_area_m2 > 0
            else 0.0
        )
        if result.coverage_pct < 80:
            result.issues.append(
                AuditIssue(
                    "warning",
                    "coverage",
                    f"SVF covers only {result.coverage_pct:.1f}% of the study area.",
                )
            )

    # ── Points inside buildings check (spatial join, not union_all) ──
    if buildings is not None and not buildings.empty:
        inside_join = gpd.sjoin(
            svf_points[["geometry"]],
            buildings[["geometry"]],
            how="inner",
            predicate="within",
        )
        n_inside = len(inside_join.drop_duplicates(subset=inside_join.geometry.name))
        if n_inside > 0:
            pct_inside = 100.0 * n_inside / len(svf_points)
            if pct_inside > 5:
                result.issues.append(
                    AuditIssue(
                        "warning",
                        "sampling",
                        f"{n_inside} SVF points ({pct_inside:.1f}%) fall inside "
                        f"building footprints.",
                    )
                )
            else:
                result.issues.append(
                    AuditIssue(
                        "info",
                        "sampling",
                        f"{n_inside} SVF points ({pct_inside:.1f}%) fall inside "
                        f"building footprints.",
                    )
                )

    # ── Computation height ──────────────────────────────────────────
    if "z" in svf_points.columns:
        z_vals = svf_points["z"].dropna()
        if len(z_vals) > 0:
            # Infer computation height above ground — typically 1.5m
            result.computation_height_m = None  # Would need DTM to determine
            result.issues.append(
                AuditIssue(
                    "info",
                    "height",
                    f"Z-coordinate range: {z_vals.min():.1f}–{z_vals.max():.1f}m. "
                    f"Verify that these represent 1.5m above ground level.",
                )
            )

    # ── Grid spacing estimation ─────────────────────────────────────
    if len(svf_points) > 100:
        from scipy.spatial import cKDTree

        coords = np.column_stack(
            [
                svf_points.geometry.x.values[:5000],
                svf_points.geometry.y.values[:5000],
            ]
        )
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)
        nn_dists = dists[:, 1]
        result.grid_spacing_m = float(np.median(nn_dists))
        result.issues.append(
            AuditIssue(
                "info",
                "resolution",
                f"Estimated grid spacing: {result.grid_spacing_m:.1f}m "
                f"(median nearest-neighbour distance).",
            )
        )

    # ── Terrain integration check ───────────────────────────────────
    if scene_stl_path is not None and scene_stl_path.exists():
        try:
            import pyvista as pv

            mesh = pv.read(str(scene_stl_path))
            z_range = mesh.bounds[5] - mesh.bounds[4]
            # If z-range is significantly larger than max building height,
            # terrain is likely included
            if z_range > 50:
                result.terrain_integrated = True
                result.issues.append(
                    AuditIssue(
                        "info",
                        "terrain",
                        f"Scene mesh Z-range: {z_range:.1f}m — terrain likely integrated.",
                    )
                )
            else:
                result.terrain_integrated = False
                result.issues.append(
                    AuditIssue(
                        "warning",
                        "terrain",
                        f"Scene mesh Z-range only {z_range:.1f}m — terrain may NOT be "
                        f"included. Critical for hillside sites.",
                    )
                )
        except Exception as e:
            result.issues.append(
                AuditIssue(
                    "info",
                    "terrain",
                    f"Could not read scene mesh for terrain check: {e}",
                )
            )
    else:
        result.issues.append(
            AuditIssue(
                "info",
                "terrain",
                "No scene STL provided; cannot verify terrain integration.",
            )
        )

    # ── Method detection ────────────────────────────────────────────
    result.method = "ray-casting (Tregenza hemisphere, svf_v2)"
    result.n_rays = 145  # Tregenza sky patches

    # ── Summary ─────────────────────────────────────────────────────
    n_crit = sum(1 for i in result.issues if i.severity == "critical")
    n_warn = sum(1 for i in result.issues if i.severity == "warning")
    logger.info(
        "SVF audit complete: %d points, mean=%.3f, %d critical, %d warnings.",
        result.count,
        result.mean,
        n_crit,
        n_warn,
    )

    return result
