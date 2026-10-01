"""Ventilation-proxy figures for the OM2 package (PI request: ventilation
indices need figures). Every quantity drawn is a geometry-derived PROXY
(see vent_indices.py), never measured air temperature or airflow; the wind
rose is SBGL airport METAR, not wind at the route.

  V1 map_vent_shelter.png  route coloured by upwind shelter angle at the
                           prevailing wind, same base map as figures.map_form,
                           prevailing direction drawn as an arrow.
  V2 profiles_vent.png     windward lambda_f, canyon alignment, shelter angle,
                           z0 along route distance (1 m raw, 10 m means).
  V3 wind_rose_compare.png campaign-window (observed) vs 2015-2024
                           climatology, same radial and colour scale.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patheffects as mpe
import numpy as np
import pandas as pd

from .figures import (SEGMENT_LENGTH_M, _distance_tick_rows, _draw_base_map, _draw_route_line, _rc)
from .io_utils import DEFAULT_ROOT, Paths
from .segments import aggregate_to_segments
from .vent_indices import compute_indices, prevailing_direction_deg
from .wind_obs import campaign_window_rose, climatology_rose, load_obs

SHELTER_CMAP = "magma_r"
_CAPTION = ("Brisa+ (MorphoFavela). Geometry-derived PROXIES from 2019 building geometry; "
            "not measured air temperature or airflow.")

PROFILE_PANELS = [
    ("windward_lambda_f_proxy", "windward λf\n(proxy, -)"),
    ("canyon_alignment_deg_proxy", "canyon alignment\n(proxy, deg; 0 = along)"),
    ("upwind_shelter_deg_proxy", "upwind shelter angle\n(proxy, deg)"),
    ("z0_m_proxy", "roughness length z0\n(proxy, m)"),
]


def horizon_cache_path(root=DEFAULT_ROOT, version: str = "v0.1.3") -> Path:
    return Path(root) / "data" / "maré" / "octopus" / "wind" / f"om2_point_horizon_{version}.npz"


def load_or_compute_horizon(points_gdf, root=DEFAULT_ROOT, version: str = "v0.1.3", force: bool = False):
    """(horizon_deg, azimuths_deg) per point, cached next to the wind data;
    the marched profile is the one shade.point_horizon_profiles returns."""
    cache = horizon_cache_path(root, version)
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        if list(z["point_id"]) == list(points_gdf["point_id"]):
            return z["horizon_deg"].astype(float), z["azimuths_deg"]
    from .shade import point_horizon_profiles

    h, az = point_horizon_profiles(points_gdf, Paths(root))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, point_id=points_gdf["point_id"].to_numpy(), horizon_deg=h, azimuths_deg=az)
    return h.astype(float), az


def _with_indices(points_df, indices):
    cols = [c for c in indices.columns if c not in points_df.columns or c == "point_id"]
    return points_df.merge(indices[cols], on="point_id", how="left")


def _draw_wind_arrow(ax, wind_from_deg: float, text: str):
    """Arrow in axes coordinates pointing where the wind blows TO (from+180),
    anchored top-left."""
    import math

    to = math.radians((wind_from_deg + 180.0) % 360.0)
    dx, dy = math.sin(to), math.cos(to)
    cx, cy, half = 0.10, 0.90, 0.07
    ax.annotate("", xy=(cx + half * dx, cy + half * dy), xytext=(cx - half * dx, cy - half * dy),
                xycoords="axes fraction", textcoords="axes fraction", zorder=7,
                arrowprops=dict(arrowstyle="-|>", lw=3, color="#0b5394", mutation_scale=22))
    ax.annotate(text, (cx, cy - 0.11), xycoords="axes fraction", ha="center", va="top", fontsize=8,
                color="#0b5394", zorder=7, path_effects=[mpe.withStroke(linewidth=2.5, foreground="white")])


def build_map_shelter(points_df, indices, buildings, subunits, out_path: Path, wind_dir_deg: float,
                      route_id: str = "OM2", version: str = "") -> Path:
    with _rc():
        import matplotlib.pyplot as plt

        from src.cartography import apply_publication_style
        apply_publication_style()

        merged = _with_indices(points_df, indices)
        fig, ax = plt.subplots(figsize=(9, 9))
        _draw_base_map(ax, merged, buildings, subunits)
        _draw_route_line(fig, ax, merged, "upwind_shelter_deg_proxy", SHELTER_CMAP,
                         "upwind shelter angle, proxy (deg)")
        _draw_wind_arrow(ax, wind_dir_deg, f"prevailing wind from {wind_dir_deg:.0f}°\n(SBGL 2015–2024, circular mean)")
        suffix = f" {version}" if version else ""
        ax.set_title(f"{route_id} route{suffix}\nupwind shelter angle (proxy) at the prevailing wind", fontsize=10)
        fig.text(0.02, 0.01, _CAPTION, fontsize=7, color="#555555")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def build_profiles_vent(points_df, indices, out_path: Path, wind_dir_deg: float, route_id: str = "OM2",
                        version: str = "", segment_length_m: float = SEGMENT_LENGTH_M) -> Path:
    with _rc():
        import matplotlib.pyplot as plt

        from src.cartography import apply_publication_style
        apply_publication_style()

        frame = _with_indices(points_df, indices).sort_values("distance_along_m")
        segments = aggregate_to_segments(frame, segment_length_m)
        ticks = _distance_tick_rows(frame)
        fig, axes = plt.subplots(len(PROFILE_PANELS), 1, figsize=(10, 1.7 * len(PROFILE_PANELS) + 1.2), sharex=True)
        for ax, (col, label) in zip(axes, PROFILE_PANELS):
            ax.plot(frame["distance_along_m"], frame[col], color="#b0b0b0", lw=0.6, zorder=1, label="1 m raw")
            seg_x = (segments["segment_start_m"] + segments["segment_end_m"]) / 2.0
            ax.plot(seg_x, segments[col], color="#1a5fa5", lw=1.8, marker="o", markersize=2.5, zorder=2, label="10 m mean")
            for _, row in ticks.iterrows():
                ax.axvline(row["tick_distance_m"], color="#dddddd", lw=0.7, zorder=0)
            ax.set_ylabel(label, fontsize=7.5)
        axes[1].set_ylim(0, 90)
        axes[1].set_yticks([0, 45, 90])
        axes[0].legend(loc="upper right", fontsize=6, frameon=False)
        axes[-1].set_xlabel("distance along route (m)")
        for _, row in ticks.iterrows():
            axes[0].annotate(str(row.get("neighbourhood") or ""), (row["tick_distance_m"], 1.02),
                             xycoords=("data", "axes fraction"), fontsize=6.5, rotation=45, ha="left",
                             va="bottom", color="#444444")
        suffix = f" {version}" if version else ""
        fig.suptitle(f"{route_id} ventilation proxies along the route at wind from {wind_dir_deg:.0f}°{suffix}", fontsize=10)
        fig.text(0.01, 0.005, _CAPTION, fontsize=7, color="#555555")
        fig.tight_layout(rect=(0, 0.015, 1, 0.96))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def build_wind_rose_compare(campaign: dict, climatology: dict, out_path: Path) -> Path:
    """Two polar bar roses (16 sectors) on one radial and one colour scale:
    bar length = frequency of directional reports, colour = mean speed."""
    with _rc():
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        rmax = max(max(campaign["frequencies"]), max(climatology["frequencies"])) * 100 * 1.1
        vmax = np.nanmax([np.nanmax(campaign["mean_speed_ms"]), np.nanmax(climatology["mean_speed_ms"])])
        norm = Normalize(0, vmax)
        cmap = matplotlib.colormaps["viridis"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), subplot_kw={"projection": "polar"})
        for ax, r, title in zip(axes, (campaign, climatology),
                                (f"Campaign window, observed\n{r_window(campaign)}", climatology["label"])):
            centres = np.radians(r["centres_deg"])
            width = 2 * np.pi / len(centres) * 0.9
            speeds = np.nan_to_num(np.array(r["mean_speed_ms"], float))
            ax.bar(centres, np.array(r["frequencies"]) * 100, width=width, color=cmap(norm(speeds)),
                   edgecolor="white", linewidth=0.5)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(0, rmax)
            ax.set_xticks(np.radians(np.arange(0, 360, 45)))
            ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
            ax.set_rlabel_position(225)
            ax.tick_params(labelsize=8)
            ax.set_title(f"{title}\nn={r['n_obs']:,} reports, calm or no direction {100 * r['calm_fraction']:.1f}%",
                         fontsize=9, pad=14)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.04, label="mean speed (m/s)")
        fig.suptitle("Wind at Galeão (SBGL), where the wind blows FROM; bar length = % of directional reports",
                     fontsize=10)
        fig.text(0.01, 0.01, "SBGL airport METAR at 10 m, about 3 km from Maré; not measured at the route. "
                 "Brisa+ (MorphoFavela).", fontsize=7, color="#555555")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def r_window(r: dict) -> str:
    w = r.get("window_utc")
    return f"{w[0][:10]} to {w[1][:10]} UTC" if w else ""


def build_all(out_dir: Path, root=DEFAULT_ROOT, version: str = "v0.1.3") -> list[Path]:
    """Render V1-V3 from the shipped points table of a package version."""
    import geopandas as gpd

    from src.sites.territory import load_territory

    paths = Paths(root)
    points = gpd.read_parquet(paths.package_dir(version) / "OM2" / "points.parquet")
    horizon, az = load_or_compute_horizon(points, root, version)
    wind = prevailing_direction_deg(root)
    idx = compute_indices(points, wind, horizon, az)
    df = pd.DataFrame(points.drop(columns="geometry"))
    buildings = gpd.read_file(paths.buildings_mare)
    subunits = load_territory("maré", root=paths.root).subunits
    out_dir = Path(out_dir)
    obs = load_obs(root)
    return [
        build_map_shelter(df, idx, buildings, subunits, out_dir / "map_vent_shelter.png", wind, version=version),
        build_profiles_vent(df, idx, out_dir / "profiles_vent.png", wind, version=version),
        build_wind_rose_compare(campaign_window_rose(obs), climatology_rose(root), out_dir / "wind_rose_compare.png"),
    ]
