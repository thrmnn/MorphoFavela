#!/usr/bin/env python3
"""Generate interactive HTML report for facade solar analysis.

Creates:
1. Interactive 3D view of facade points colored by sunlit hours (PyVista)
2. Interactive 2D map with building footprints and WHO compliance (Folium)
3. Summary statistics and methodology in a combined HTML report

Usage:
    python scripts/generate_facade_solar_report.py --area vidigal_tls
    python scripts/generate_facade_solar_report.py --area vidigal
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyvista as pv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_output_dir
from src.svf_v2.paths import resolve_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("facade_solar_report")


def load_data(area: str):
    """Load facade solar data, building footprints, and scene mesh."""
    output_dir = get_area_output_dir(area) / "facade_solar"
    svf_dir = get_area_output_dir(area) / "svf_v2"

    facade_path = output_dir / "facade_solar_20260621.gpkg"
    buildings_path = output_dir / "buildings_who_20260621.csv"
    floors_path = output_dir / "floors_who_20260621.csv"
    scene_path = svf_dir / "scene.vtk"

    if not facade_path.exists():
        raise FileNotFoundError(f"No facade solar data: {facade_path}")

    logger.info("Loading facade data from %s", facade_path)
    facade_gdf = gpd.read_file(facade_path)
    buildings_who = pd.read_csv(buildings_path) if buildings_path.exists() else None
    floors_who = pd.read_csv(floors_path) if floors_path.exists() else None
    scene_mesh = pv.read(str(scene_path)) if scene_path.exists() else None

    _, fp_path, _ = resolve_paths(area)
    footprints = gpd.read_file(fp_path)

    return facade_gdf, buildings_who, floors_who, scene_mesh, footprints


def generate_3d_html(facade_gdf, scene_mesh, output_path, area):
    """Generate interactive 3D HTML view of facade points using Plotly."""
    import plotly.graph_objects as go

    logger.info("Generating 3D interactive HTML...")

    # Subsample for performance if too many points
    max_pts = 50000
    if len(facade_gdf) > max_pts:
        df = facade_gdf.sample(max_pts, random_state=42).copy()
        logger.info("  Subsampled %d -> %d points for 3D view", len(facade_gdf), max_pts)
    else:
        df = facade_gdf.copy()

    # Normalize coordinates to local origin for better rendering
    x0, y0 = df["x"].min(), df["y"].min()
    x = df["x"].values - x0
    y = df["y"].values - y0
    z = df["z"].values
    hours = df["facade_sunlit_hours"].values
    who = df["who_compliant"].values

    hover_text = [
        f"Sunlit: {h:.2f}h<br>Floor: F{int(f)}<br>"
        f"Azimuth: {int(az)}°<br>WHO: {'Yes' if w else 'No'}"
        for h, f, az, w in zip(hours, df["floor_level"], df["facade_azimuth"], who)
    ]

    # Facade points colored by sunlit hours
    facade_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=2,
            color=hours,
            colorscale="YlOrRd",
            cmin=0,
            cmax=max(hours.max(), 2.0),
            colorbar=dict(title="Sunlit<br>Hours", x=1.02),
            opacity=0.8,
        ),
        text=hover_text,
        hoverinfo="text",
        name="Facade Points",
    )

    traces = [facade_trace]

    # Add building mesh as wireframe if available
    if scene_mesh is not None:
        # Extract mesh surface as triangles for Plotly
        surf = scene_mesh.extract_surface().triangulate()
        verts = surf.points
        faces = surf.faces.reshape(-1, 4)[:, 1:]  # strip leading 3

        mesh_trace = go.Mesh3d(
            x=verts[:, 0] - x0,
            y=verts[:, 1] - y0,
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="#cccccc",
            opacity=0.15,
            name="Buildings",
            hoverinfo="skip",
        )
        traces.insert(0, mesh_trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"Facade Solar Access — {area.replace('_', ' ').title()}",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Elevation (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.02, y=0.98),
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info("3D HTML saved to %s", output_path)


def generate_folium_map(facade_gdf, footprints, output_path, area):
    """Generate interactive 2D map with WHO compliance overlay."""
    import folium
    from branca.colormap import LinearColormap

    logger.info("Generating Folium 2D map...")

    # Convert to WGS84 for Folium
    facade_wgs = facade_gdf.to_crs(epsg=4326)
    fp_wgs = footprints.to_crs(epsg=4326)

    # Center map
    center_lat = facade_wgs.geometry.y.mean()
    center_lon = facade_wgs.geometry.x.mean()
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles="CartoDB positron",
    )

    # Building footprints layer
    building_group = folium.FeatureGroup(name="Building Footprints")
    for _, row in fp_wgs.iterrows():
        if row.geometry is None:
            continue
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x: {
                "fillColor": "#cccccc",
                "color": "#888888",
                "weight": 0.5,
                "fillOpacity": 0.4,
            },
        ).add_to(building_group)
    building_group.add_to(m)

    # WHO compliance heatmap — sample points for performance
    max_pts = 5000
    if len(facade_wgs) > max_pts:
        sample = facade_wgs.sample(max_pts, random_state=42)
    else:
        sample = facade_wgs

    # Sunlit hours layer
    cmap = LinearColormap(
        ["#3b82f6", "#fbbf24", "#ef4444"],
        vmin=0,
        vmax=max(sample["facade_sunlit_hours"].max(), 2.0),
        caption="Sunlit Hours",
    )

    sunlit_group = folium.FeatureGroup(name="Facade Sunlit Hours")
    for _, row in sample.iterrows():
        hours = row["facade_sunlit_hours"]
        who = row["who_compliant"]
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2 if not who else 4,
            color=cmap(hours),
            fill=True,
            fill_opacity=0.7 if who else 0.4,
            popup=folium.Popup(
                f"<b>Sunlit: {hours:.2f}h</b><br>"
                f"Floor: {row['floor_level']}<br>"
                f"Irradiance: {row['facade_irradiance_wh']:.0f} Wh/m²<br>"
                f"WHO: {'Yes' if who else 'No'}<br>"
                f"Azimuth: {row['facade_azimuth']:.0f}°",
                max_width=200,
            ),
        ).add_to(sunlit_group)
    sunlit_group.add_to(m)
    cmap.add_to(m)

    # WHO compliant layer (separate)
    who_group = folium.FeatureGroup(name="WHO Compliant Only", show=False)
    compliant = sample[sample["who_compliant"]]
    for _, row in compliant.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="#22c55e",
            fill=True,
            fill_opacity=0.9,
            popup=f"Sunlit: {row['facade_sunlit_hours']:.2f}h | Floor {row['floor_level']}",
        ).add_to(who_group)
    who_group.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(str(output_path))
    logger.info("Folium map saved to %s", output_path)


def generate_summary_html(
    facade_gdf,
    buildings_who,
    floors_who,
    area,
    output_path,
    html_3d_path,
    html_map_path,
):
    """Generate combined HTML report with stats, embedded 3D view, and map."""
    logger.info("Generating summary HTML report...")

    n_points = len(facade_gdf)
    n_who = int(facade_gdf["who_compliant"].sum())
    pct_who = 100 * n_who / n_points if n_points > 0 else 0

    n_buildings = facade_gdf["building_id"].nunique()
    n_bldg_who = 0
    if buildings_who is not None:
        n_bldg_who = int((buildings_who["who_compliance_rate"] > 0).sum())

    n_units = 0
    n_units_below = 0
    if floors_who is not None:
        n_units = len(floors_who)
        n_units_below = int((floors_who["who_compliance_rate"] < 0.5).sum())

    hours = facade_gdf["facade_sunlit_hours"]
    irrad = facade_gdf["facade_irradiance_wh"]

    # Floor-level stats
    floor_stats = (
        facade_gdf.groupby("floor_level")["facade_sunlit_hours"]
        .agg(["mean", "max", "count"])
        .reset_index()
    )

    # Azimuth stats (bin by 45 degrees)
    facade_gdf = facade_gdf.copy()
    facade_gdf["az_bin"] = ((facade_gdf["facade_azimuth"] + 22.5) // 45 * 45).astype(int) % 360
    az_labels = {
        0: "N",
        45: "NE",
        90: "E",
        135: "SE",
        180: "S",
        225: "SW",
        270: "W",
        315: "NW",
    }
    az_stats = (
        facade_gdf.groupby("az_bin")["facade_sunlit_hours"].agg(["mean", "count"]).reset_index()
    )
    az_stats["direction"] = az_stats["az_bin"].map(az_labels)

    # Relative paths for embedding
    rel_3d = Path(html_3d_path).name
    rel_map = Path(html_map_path).name

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Facade Solar Report — {area}</title>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 0; color: #1a1a1a; background: #f8f8f8; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 2em; border-bottom: 3px solid #2A5FA5; padding-bottom: 10px; }}
  h2 {{ font-size: 1.4em; color: #2A5FA5; margin-top: 2em; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 20px 0; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
  .stat-card .value {{ font-size: 2.2em; font-weight: 700; color: #2A5FA5; }}
  .stat-card .label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
  .stat-card.alert .value {{ color: #dc2626; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
  th {{ background: #2A5FA5; color: white; padding: 10px 14px; text-align: left; font-weight: 600; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) {{ background: #f9fafb; }}
  .iframe-container {{ width: 100%; height: 600px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; margin: 16px 0; background: white; }}
  iframe {{ width: 100%; height: 100%; border: none; }}
  .method {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); line-height: 1.6; }}
  .method code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  .warning {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 16px 0; }}
</style>
</head>
<body>
<div class="container">

<h1>Facade Solar Access Report — {area.replace("_", " ").title()}</h1>
<p>Winter solstice (June 21) analysis at latitude -22.97&deg;, longitude -43.17&deg;.
WHO Housing and Health Guidelines recommend minimum 2 hours of direct sunlight.</p>

<h2>Key Findings</h2>
<div class="stat-grid">
  <div class="stat-card alert">
    <div class="value">{pct_who:.1f}%</div>
    <div class="label">Facade Points WHO Compliant</div>
  </div>
  <div class="stat-card alert">
    <div class="value">{n_units_below:,} / {n_units:,}</div>
    <div class="label">Housing Units Below WHO</div>
  </div>
  <div class="stat-card">
    <div class="value">{hours.mean():.2f}h</div>
    <div class="label">Mean Sunlit Hours</div>
  </div>
  <div class="stat-card">
    <div class="value">{hours.max():.1f}h</div>
    <div class="label">Max Sunlit Hours</div>
  </div>
  <div class="stat-card">
    <div class="value">{n_points:,}</div>
    <div class="label">Facade Sample Points</div>
  </div>
  <div class="stat-card">
    <div class="value">{n_buildings:,}</div>
    <div class="label">Buildings Analysed</div>
  </div>
</div>

{"<div class='warning'><strong>Note:</strong> Results may change after the winding-order fix is applied and HPC jobs complete.</div>" if pct_who < 2 else ""}

<h2>Interactive 3D View</h2>
<p>Facade points colored by direct sunlight hours. Rotate, zoom, and pan to explore.</p>
<div class="iframe-container">
  <iframe src="{rel_3d}" title="3D Facade Solar View"></iframe>
</div>

<h2>Interactive Map</h2>
<p>Toggle layers: building footprints, sunlit hours, WHO-compliant points only.</p>
<div class="iframe-container">
  <iframe src="{rel_map}" title="2D Facade Solar Map"></iframe>
</div>

<h2>Sunlit Hours by Floor Level</h2>
<table>
<tr><th>Floor</th><th>Mean Hours</th><th>Max Hours</th><th>Points</th></tr>
"""

    for _, row in floor_stats.iterrows():
        html += f"<tr><td>F{int(row['floor_level'])}</td><td>{row['mean']:.3f}</td><td>{row['max']:.2f}</td><td>{int(row['count']):,}</td></tr>\n"

    html += """</table>

<h2>Sunlit Hours by Facade Orientation</h2>
<table>
<tr><th>Direction</th><th>Mean Hours</th><th>Points</th></tr>
"""

    for _, row in az_stats.iterrows():
        html += f"<tr><td>{row['direction']} ({int(row['az_bin'])}°)</td><td>{row['mean']:.3f}</td><td>{int(row['count']):,}</td></tr>\n"

    html += f"""</table>

<h2>Methodology</h2>
<div class="method">
<p><strong>Facade sampling:</strong> Points generated at {facade_gdf["height_above_ground"].min():.1f}–{facade_gdf["height_above_ground"].max():.1f}m height
on exterior walls, offset 0.1m outward from the wall surface. Each <code>(building_id, floor_level)</code>
combination at 2.5m intervals represents one housing unit.</p>

<p><strong>Solar ray-casting:</strong> For each facade point and each of 21 sun positions (30-min intervals,
6:00–18:00 solar time), a ray is cast toward the sun against the 3D scene mesh (terrain + extruded
buildings). A point is "sunlit" if the ray is unobstructed AND the sun is in the forward hemisphere
of the facade (<code>cos(theta_i) &gt; 0</code>).</p>

<p><strong>Self-intersection filtering:</strong> Ray intersections closer than 0.5m are ignored to prevent
facade points from being falsely shadowed by their own building's roof geometry.</p>

<p><strong>Irradiance model:</strong> Tilted-surface irradiance (Duffie &amp; Beckman, 2013) with Hottel
clear-sky beam transmittance and Liu-Jordan isotropic diffuse model for vertical surfaces.</p>

<p><strong>WHO threshold:</strong> A facade point is WHO-compliant if it receives &ge;2 hours of direct
sunlight. A housing unit is classified as "below WHO" if fewer than 50% of its facade points
are compliant (WHO Housing and Health Guidelines, 2018).</p>
</div>

</div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    logger.info("Summary report saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Facade solar interactive HTML report")
    parser.add_argument("--area", required=True, help="Area name (e.g. vidigal_tls)")
    args = parser.parse_args()

    facade_gdf, buildings_who, floors_who, scene_mesh, footprints = load_data(args.area)

    output_dir = get_area_output_dir(args.area) / "facade_solar"
    output_dir.mkdir(parents=True, exist_ok=True)

    html_3d = output_dir / "facade_solar_3d.html"
    html_map = output_dir / "facade_solar_map.html"
    html_report = output_dir / "facade_solar_report.html"

    generate_3d_html(facade_gdf, scene_mesh, html_3d, args.area)
    generate_folium_map(facade_gdf, footprints, html_map, args.area)
    generate_summary_html(
        facade_gdf,
        buildings_who,
        floors_who,
        args.area,
        html_report,
        html_3d,
        html_map,
    )

    print(f"\nReport generated at: {html_report}")
    print(f"  3D view: {html_3d}")
    print(f"  Map: {html_map}")


if __name__ == "__main__":
    main()
