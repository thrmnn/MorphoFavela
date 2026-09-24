"""The three definitions of Maré on one map, for the PI's territory review:
the data extent (official bairro, where footprints exist), the study area
(the 16 communities inside that extent) and the citywide definition (the IPP
polygons the WP-05/07 ledger matches). Community fills say which SABREN layer
each polygon came from — provenance, not a comparison.

    python scripts/render_mare_territory_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brisa_solar import mare_study_area as msa  # noqa: E402
from src.brisa_solar.wp05_full import match_favela_group  # noqa: E402

OUT = ROOT / "outputs" / "maré" / "territory" / "mare_territory_map.png"
DPI = 300
INK, EXTENT, CITYWIDE = "#1b1b1b", "#6b6b6b", "#b0306e"
SOURCE_FILL = {"favelas_2022": "#cfe3d6", "conjuntos": "#d9dcef"}
SOURCE_LABEL = {"favelas_2022": "community polygon from SABREN favela limits 2022",
                "conjuntos": "community polygon from SABREN conjuntos habitacionais"}


def main() -> int:
    extent = msa.load_data_extent(ROOT)
    comm = msa.load_communities(ROOT)
    inside = comm[comm.geometry.intersection(extent).area / comm.area >= 0.5]
    outside = comm.drop(inside.index)
    citywide, _ = match_favela_group(gpd.read_file(ROOT / "data" / "RJ" / "Favelas_Limit_2019.shp"), "Maré")
    buildings = gpd.read_file(ROOT / "data" / "maré" / "raw" / "buildings_mare.shp").to_crs(31983)

    fig, ax = plt.subplots(figsize=(7.5, 10), dpi=DPI)
    px = 72.0 / DPI
    buildings.plot(ax=ax, color="#9a9a9a", linewidth=0, zorder=1)
    for _, r in inside.iterrows():
        src = r["source_parts"].split(":")[0]
        gpd.GeoSeries([r.geometry], crs=31983).plot(ax=ax, color=SOURCE_FILL[src], alpha=0.75, linewidth=0, zorder=0)
    inside.boundary.plot(ax=ax, color=INK, linewidth=1.2 * px * 3, zorder=3)
    citywide.boundary.plot(ax=ax, color=CITYWIDE, linewidth=2.0 * px * 3, linestyle=(0, (4, 2)), zorder=4)
    gpd.GeoSeries([extent], crs=31983).boundary.plot(ax=ax, color=EXTENT, linewidth=1.0 * px * 3, zorder=2)
    for _, r in inside.iterrows():
        p = r.geometry.representative_point()
        ax.annotate(r["community"] + (" *" if r["match"] == "inferred" else ""), (p.x, p.y),
                    fontsize=6.5, ha="center", va="center", color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    handles = [Patch(fc=SOURCE_FILL[k], label=v) for k, v in SOURCE_LABEL.items()] + [
        Line2D([], [], color=INK, lw=1.2, label="study area: the communities (site sheet, brief)"),
        Line2D([], [], color=CITYWIDE, lw=2.0, ls=(0, (4, 2)), label=f"citywide definition: {len(citywide)} IPP polygons, complexo 'Maré'"),
        Line2D([], [], color=EXTENT, lw=1.0, label="data extent: official bairro (building footprints end here)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False)
    notes = ["* name match inferred — see data/maré/neighbourhoods_provenance.json"]
    for _, r in outside.iterrows():
        km = r.geometry.distance(extent) / 1000.0
        notes.append(f"{r['community']} lies {km:.1f} km outside the data extent and is not drawn")
    ax.text(0.99, 0.01, "\n".join(notes), transform=ax.transAxes, fontsize=6, ha="right", va="bottom", color=EXTENT)
    x0, y0, x1, y1 = gpd.GeoSeries([extent], crs=31983).total_bounds
    ax.plot([x1 - 600, x1 - 100], [y0 + 150, y0 + 150], color=INK, lw=1.5)
    ax.text(x1 - 350, y0 + 180, "500 m", fontsize=6.5, ha="center", color=INK)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Maré — three definitions of the territory", fontsize=10, loc="left", color=INK)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
