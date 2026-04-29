#!/usr/bin/env python3
"""Synthetic street canyon validation for facade solar methodology.

Builds a simple 2-building canyon, runs the facade solar pipeline with
and without the self-intersection fix, and generates a publication-quality
validation figure.

Usage:
    python scripts/validate_facade_solar.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cartography import apply_publication_style
from src.solar.facade import (
    aggregate_by_building_floor,
    compute_facade_solar_access,
    summarize_housing_units,
)

# ---------------------------------------------------------------------------
# Canyon geometry
# ---------------------------------------------------------------------------

CANYON_WIDTH = 6.0
BUILDING_HEIGHT = 10.0
FLOOR_HEIGHT = 2.5


def build_canyon():
    """Build E-W canyon: 2 buildings (10m tall), 6m street, flat terrain."""
    terrain = pv.Plane(
        center=(8, 0, 0),
        direction=(0, 0, 1),
        i_size=50,
        j_size=50,
        i_resolution=10,
        j_resolution=10,
    )
    bldg_a = pv.Box(bounds=(-5, 5, -10, 10, 0, BUILDING_HEIGHT))
    bldg_b = pv.Box(bounds=(11, 21, -10, 10, 0, BUILDING_HEIGHT))
    return terrain + bldg_a + bldg_b


def build_facade_points(inset=0.1):
    """Facade points on inner canyon walls at multiple heights."""
    rows = []
    y_positions = np.linspace(-8, 8, 9)
    z_positions = np.arange(1.25, BUILDING_HEIGHT, 1.25)

    # Building A — east-facing inner wall
    for y in y_positions:
        for z in z_positions:
            rows.append(
                {
                    "x": 5.0 + inset,
                    "y": y,
                    "z": z,
                    "normal_x": 1.0,
                    "normal_y": 0.0,
                    "normal_z": 0.0,
                    "building_id": 0,
                    "facade_azimuth": 90.0,
                    "height_above_ground": z,
                    "geometry": Point(5.0 + inset, y),
                }
            )

    # Building B — west-facing inner wall
    for y in y_positions:
        for z in z_positions:
            rows.append(
                {
                    "x": 21.0 - inset,
                    "y": y,
                    "z": z,
                    "normal_x": -1.0,
                    "normal_y": 0.0,
                    "normal_z": 0.0,
                    "building_id": 1,
                    "facade_azimuth": 270.0,
                    "height_above_ground": z,
                    "geometry": Point(21.0 - inset, y),
                }
            )

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:31983")
    gdf["svf"] = 1.0
    return gdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("  FACADE SOLAR VALIDATION — Synthetic Street Canyon")
    print("=" * 70)

    scene = build_canyon()
    facade_gdf = build_facade_points(inset=0.1)

    print(f"\nCanyon: {CANYON_WIDTH}m wide, buildings {BUILDING_HEIGHT}m tall")
    print(f"Facade points: {len(facade_gdf)} ({len(facade_gdf) // 2} per wall)")
    print(f"Floor height: {FLOOR_HEIGHT}m")

    # --- Run with fix ---
    print("\n--- Running with min_hit_distance=0.5 (fix applied) ---")
    result = compute_facade_solar_access(
        facade_gdf,
        scene,
        date="2026-06-21",
        latitude=-22.97,
        longitude=-43.17,
        min_hit_distance=0.5,
        floor_height=FLOOR_HEIGHT,
        n_jobs=1,
    )

    # --- Run without fix (for comparison) ---
    print("\n--- Running with min_hit_distance=0.0 (no fix, for comparison) ---")
    # Place points slightly inside building to trigger bug
    facade_inside = facade_gdf.copy()
    facade_inside.loc[facade_inside["building_id"] == 0, "x"] = 4.99
    facade_inside.loc[facade_inside["building_id"] == 1, "x"] = 11.01
    result_buggy = compute_facade_solar_access(
        facade_inside,
        scene,
        date="2026-06-21",
        latitude=-22.97,
        longitude=-43.17,
        min_hit_distance=0.0,
        floor_height=FLOOR_HEIGHT,
        n_jobs=1,
    )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    for label, r in [("With fix", result), ("Without fix (bug)", result_buggy)]:
        n_sunlit = (r["facade_sunlit_hours"] > 0).sum()
        n_who = r["who_compliant"].sum()
        floor_agg = aggregate_by_building_floor(r)
        hu = summarize_housing_units(floor_agg)
        print(f"\n{label}:")
        print(f"  Sunlit points: {n_sunlit}/{len(r)} ({100 * n_sunlit / len(r):.1f}%)")
        print(
            f"  Sunlit hours: mean={r['facade_sunlit_hours'].mean():.2f}, "
            f"max={r['facade_sunlit_hours'].max():.2f}"
        )
        print(f"  WHO compliant: {n_who}/{len(r)} ({100 * n_who / len(r):.1f}%)")
        print(
            f"  Housing units: {hu['total_units']} total, "
            f"{hu['units_below_who']} below WHO ({hu['pct_below']:.1f}%)"
        )

    # --- Generate figure ---
    print("\n--- Generating validation figure ---")
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel (a): Facade points colored by sunlit hours (plan view)
    ax = axes[0]
    sc = ax.scatter(
        result["x"],
        result["z"],
        c=result["facade_sunlit_hours"],
        cmap="YlOrRd",
        s=30,
        edgecolors="gray",
        linewidth=0.3,
        vmin=0,
        zorder=5,
    )
    # Draw building outlines
    for bx in [(-5, 5), (11, 21)]:
        rect = plt.Rectangle(
            (bx[0], 0),
            bx[1] - bx[0],
            BUILDING_HEIGHT,
            fill=True,
            facecolor="#e0e0e0",
            edgecolor="black",
            linewidth=1.5,
            zorder=1,
        )
        ax.add_patch(rect)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("(a) Cross-Section: Sunlit Hours", fontsize=12, fontweight="bold")
    ax.set_xlim(-8, 24)
    ax.set_ylim(-1, 12)
    ax.set_aspect("equal")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Sunlit hours")

    # Draw sun angle
    sun_alt = 43.6  # max altitude at Rio winter solstice
    sun_x = 8 + 15 * np.cos(np.radians(sun_alt))
    sun_z = 15 * np.sin(np.radians(sun_alt))
    ax.annotate(
        f"Sun max alt={sun_alt:.0f}deg",
        xy=(8, 0),
        xytext=(sun_x, sun_z),
        arrowprops=dict(arrowstyle="->", color="orange", lw=2),
        fontsize=9,
        color="orange",
    )

    # Panel (b): Sunlit hours by floor level
    ax = axes[1]
    by_floor = result.groupby("floor_level")["facade_sunlit_hours"].mean()
    by_floor_buggy = result_buggy.groupby("floor_level")["facade_sunlit_hours"].mean()

    floors = sorted(by_floor.index)
    x_pos = np.arange(len(floors))
    width = 0.35

    ax.bar(
        x_pos - width / 2,
        [by_floor.get(f, 0) for f in floors],
        width,
        label="With fix",
        color="#3b82f6",
        edgecolor="white",
    )
    ax.bar(
        x_pos + width / 2,
        [by_floor_buggy.get(f, 0) for f in floors],
        width,
        label="Without fix (bug)",
        color="#ef4444",
        edgecolor="white",
        alpha=0.7,
    )
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1, label="WHO 2h threshold")
    ax.set_xlabel("Floor Level")
    ax.set_ylabel("Mean Sunlit Hours")
    ax.set_title("(b) Sunlit Hours by Floor", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"F{f}" for f in floors])
    ax.legend(fontsize=9)

    # Panel (c): WHO compliance comparison
    ax = axes[2]
    labels = ["With fix\n(correct)", "Without fix\n(bug)"]
    who_pcts = [
        100 * result["who_compliant"].mean(),
        100 * result_buggy["who_compliant"].mean(),
    ]
    colors = ["#3b82f6", "#ef4444"]
    bars = ax.bar(labels, who_pcts, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("WHO 2h Compliance (%)")
    ax.set_title("(c) Impact of Self-Intersection Fix", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    for bar, pct in zip(bars, who_pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{pct:.1f}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        f"Facade Solar Validation — {CANYON_WIDTH}m Canyon, "
        f"{BUILDING_HEIGHT}m Buildings, Winter Solstice (Rio)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    output_dir = PROJECT_ROOT / "outputs" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "facade_solar_validation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")

    # --- Methodology explanation ---
    print("\n" + "=" * 70)
    print("  METHODOLOGY")
    print("=" * 70)
    print(
        """
The facade solar pipeline computes direct sunlight hours for vertical
building surfaces using ray-casting against a 3D scene mesh (terrain +
extruded buildings).

For each facade point, rays are cast toward 21 sun positions (30-min
intervals, June 21 winter solstice) computed via pvlib. A facade point
is "sunlit" at timestep t if:
  1. The ray from the point toward the sun is unobstructed (no mesh hit)
  2. The sun is in front of the facade (cos(theta_i) > 0)

SELF-INTERSECTION FIX: Facade points are offset 0.1m from the wall
surface, but this is insufficient to prevent rays from intersecting the
building's own roof geometry at short distances. The fix introduces a
min_hit_distance parameter (default 0.5m) that ignores ray intersections
closer than this threshold, filtering out self-shadowing artifacts.

HOUSING UNITS: Each (building_id, floor_level) combination represents
one housing unit, using 2.5m floor-to-floor height. The WHO recommends
a minimum of 2 hours of direct sunlight at winter solstice for healthy
housing (WHO Housing and Health Guidelines, 2018).
"""
    )


if __name__ == "__main__":
    main()
