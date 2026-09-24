"""Every site's territory, one map: the data extent (where footprints exist),
the study area (what counts in that site's statistics) and the citywide
definition (the IPP polygons the WP-05/07 ledger matches) — plus, for a site
that declares them, its subunits and any not-yet-promoted study-area
candidates (today: Maré's IPP Territórios Sociais complex outline).

Generalised (2026-09-24, SITETERR) from a Maré-only script to any registry
site (config/sites.yaml, src/sites/territory.py); the visual language and
output path for Maré are unchanged so scripts/build_pi_review_folder.py's
existing card keeps working without edits.

    python scripts/render_mare_territory_map.py                # maré (default)
    python scripts/render_mare_territory_map.py --site rocinha
    python scripts/render_mare_territory_map.py --all
"""

from __future__ import annotations

import argparse
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

from src.sites.territory import ROOT_DEFAULT, load_sites_config, load_territory  # noqa: E402
from src.svf_v2.paths import AREA_FILES  # noqa: E402

DPI = 300
INK, EXTENT, CITYWIDE, CANDIDATE = "#1b1b1b", "#6b6b6b", "#b0306e", "#1a6fa8"
SOURCE_FILL_DEFAULT = "#cfe3d6"
SOURCE_FILL = {"favelas_2022": "#cfe3d6", "conjuntos": "#d9dcef"}
SOURCE_LABEL = {"favelas_2022": "community polygon from SABREN favela limits 2022",
                "conjuntos": "community polygon from SABREN conjuntos habitacionais"}

#: Maré keeps its original ascii output name (outputs/maré/territory/
#: mare_territory_map.png) because scripts/build_pi_review_folder.py already
#: links it by that exact path.
OUT_NAME = {"maré": "mare_territory_map"}


def _out_path(site: str, root: Path) -> Path:
    name = OUT_NAME.get(site, f"{site.replace(' ', '_')}_territory_map")
    return root / "outputs" / site / "territory" / f"{name}.png"


def _buildings(site: str, root: Path) -> gpd.GeoDataFrame | None:
    reg = AREA_FILES.get(site)
    if not reg or "footprints" not in reg:
        return None
    p = root / "data" / site / "raw" / reg["footprints"]
    if not p.exists():
        return None
    gdf = gpd.read_file(p)
    return gdf.set_crs(31983) if gdf.crs is None else gdf.to_crs(31983)


def render_one(site: str, root: Path = ROOT_DEFAULT) -> Path:
    # A test elsewhere in the suite leaks matplotlib style into the process;
    # rc_context(rcParamsDefault) keeps this render's fonts/sizes from
    # depending on import order (docs/briefs/mare/render_figures.py's
    # render_all does the same, for the same reason).
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        return _render_one(site, root)


def _render_one(site: str, root: Path) -> Path:
    t = load_territory(site, root=root)
    buildings = _buildings(t.site, root)

    fig, ax = plt.subplots(figsize=(7.5, 10), dpi=DPI)
    px = 72.0 / DPI
    if buildings is not None and len(buildings):
        buildings.plot(ax=ax, color="#9a9a9a", linewidth=0, zorder=1)

    handles = []
    has_source_col = t.subunits is not None and "source_parts" in t.subunits.columns
    if t.subunits is not None and len(t.subunits):
        included = t.subunits_included if t.subunits_included is not None else t.subunits
        excluded = t.subunits_excluded if t.subunits_excluded is not None else t.subunits.iloc[0:0]
        for _, r in included.iterrows():
            fill = SOURCE_FILL.get(r["source_parts"].split(":")[0], SOURCE_FILL_DEFAULT) if has_source_col else SOURCE_FILL_DEFAULT
            gpd.GeoSeries([r.geometry], crs=31983).plot(ax=ax, color=fill, alpha=0.75, linewidth=0, zorder=0)
        included.boundary.plot(ax=ax, color=INK, linewidth=1.2 * px * 3, zorder=3)
        name_col = "name" if "name" in included.columns else included.columns[0]
        for _, r in included.iterrows():
            p = r.geometry.representative_point()
            inferred = has_source_col and r.get("match") == "inferred"
            ax.annotate(str(r[name_col]) + (" *" if inferred else ""), (p.x, p.y),
                        fontsize=6.5, ha="center", va="center", color=INK, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        if has_source_col:
            handles += [Patch(fc=v, label=SOURCE_LABEL[k]) for k, v in SOURCE_FILL.items()]
        handles.append(Line2D([], [], color=INK, lw=1.2, label="study area: the subunits (site sheet, brief)"))
    else:
        gpd.GeoSeries([t.study_area], crs=31983).boundary.plot(
            ax=ax, color=INK, linewidth=1.2 * px * 3, zorder=3)
        handles.append(Line2D([], [], color=INK, lw=1.2, label="study area"))

    if len(t.citywide):
        t.citywide.boundary.plot(ax=ax, color=CITYWIDE, linewidth=2.0 * px * 3,
                                  linestyle=(0, (4, 2)), zorder=4)
        handles.append(Line2D([], [], color=CITYWIDE, lw=2.0, ls=(0, (4, 2)),
                        label=f"citywide definition: {len(t.citywide)} IPP polygon(s), "
                              f"{t.citywide_method}"))

    gpd.GeoSeries([t.data_extent], crs=31983).boundary.plot(
        ax=ax, color=EXTENT, linewidth=1.0 * px * 3, zorder=2)
    handles.append(Line2D([], [], color=EXTENT, lw=1.0,
                    label="data extent: boundary (building footprints end here)"))

    for cand_id, cand in t.study_area_candidates.items():
        gpd.GeoSeries([cand["geometry"]], crs=31983).boundary.plot(
            ax=ax, color=CANDIDATE, linewidth=1.6 * px * 3, linestyle=(0, (1, 1.5)), zorder=5)
        handles.append(Line2D([], [], color=CANDIDATE, lw=1.6, ls=(0, (1, 1.5)),
                        label=f"candidate — {cand['label']} (not active)"))

    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False,
              bbox_to_anchor=(0.0, 0.02 + 0.028 * len(handles)))

    excluded = t.subunits_excluded if t.subunits is not None else None
    if excluded is not None and len(excluded):
        name_col = "name" if "name" in excluded.columns else excluded.columns[0]
        notes = ["* name match inferred — see territory_provenance.json" if has_source_col else ""]
        for _, r in excluded.iterrows():
            km = r.geometry.distance(t.data_extent) / 1000.0
            notes.append(f"{r[name_col]} lies {km:.1f} km outside the data extent and is not drawn")
        ax.text(0.99, 0.01, "\n".join(n for n in notes if n), transform=ax.transAxes,
                fontsize=6, ha="right", va="bottom", color=EXTENT)

    x0, y0, x1, y1 = gpd.GeoSeries([t.data_extent], crs=31983).total_bounds
    ax.plot([x1 - 600, x1 - 100], [y0 + 150, y0 + 150], color=INK, lw=1.5)
    ax.text(x1 - 350, y0 + 180, "500 m", fontsize=6.5, ha="center", color=INK)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"{t.display_name} — territory", fontsize=10, loc="left", color=INK)

    out = _out_path(t.site, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="maré")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = ap.parse_args()
    sites = list(load_sites_config()) if args.all else [args.site]
    for site in sites:
        render_one(site, root=args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
