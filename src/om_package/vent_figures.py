"""OM2 report figures, part 2: wind regimes and ventilation.

Filenames are the contract with the report text (written under OM2/):
fig_wind, fig_vent_profiles, fig_shelter_maps. Regime colours come from
wind_regimes.REGIME_COLOURS; every ventilation measure is a geometry-derived
proxy, the wind is airport (Galeao) reports.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.patheffects
import numpy as np
import pandas as pd

from . import fig_style as fs
from .figures import _route_line
from .wind_regimes import (COMPASS, N_SECTORS, REGIME_COLOURS, SECTOR_W, classify, clean)

#: (point-column stem, y label) of the three ventilation profiles.
VENT_PANELS = [
    ("frontal_area_density_windward", "windward frontal\narea density"),
    ("canyon_alignment_deg", "canyon alignment\n(degrees)"),
    ("upwind_shelter_angle_deg", "upwind shelter\nangle (degrees)"),
]


def regime_title(name: str) -> str:
    return f"{name} wind"


def sector_shares(obs: pd.DataFrame) -> np.ndarray:
    """Share (%) of reports with a direction in each of the 16 sectors."""
    act, _ = clean(obs)
    idx = (((act["drct"].to_numpy(float) % 360.0) + SECTOR_W / 2) // SECTOR_W).astype(int) % N_SECTORS
    f = np.bincount(idx, minlength=N_SECTORS).astype(float)
    return 100.0 * f / f.sum()


def sector_regime_keys(regimes: dict) -> np.ndarray:
    centres = np.arange(N_SECTORS) * SECTOR_W
    return classify(centres, regimes)


def build_fig_wind(season: dict, campaign_obs: pd.DataFrame, climatology_obs: pd.DataFrame,
                   by_hour: pd.DataFrame, out_path: Path) -> tuple[Path, dict]:
    """Two 16-sector roses (campaign season, 2015 to 2024) with bars coloured by
    regime and the regime mean direction as a line, plus the share of reports by
    local hour of day per regime."""
    with fs.figure_style():
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        periods = [("campaign season", "campaign", season["campaign"], campaign_obs),
                   ("2015 to 2024", "climatology", season["climatology"], climatology_obs)]
        shares = [sector_shares(o) for *_, o in periods]
        rmax = float(np.ceil(max(s.max() for s in shares) / 5.0) * 5.0)

        fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, 5.6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.32, wspace=0.16, left=0.085, right=0.955,
                              top=0.9, bottom=0.115)
        centres = np.radians(np.arange(N_SECTORS) * SECTOR_W)
        for col, ((title, _, res, _), share) in enumerate(zip(periods, shares)):
            ax = fig.add_subplot(gs[0, col], projection="polar")
            keys = sector_regime_keys(res)
            ax.bar(centres, share, width=np.radians(SECTOR_W) * 0.92, color=[REGIME_COLOURS[k] for k in keys],
                   edgecolor="white", linewidth=0.5, zorder=2)
            for g in res["regimes"]:
                th = np.radians(g["mean_direction_deg"])
                ax.plot([th, th], [0, rmax], color=REGIME_COLOURS[g["key"]], lw=1.8, zorder=3,
                        path_effects=[matplotlib.patheffects.withStroke(linewidth=3.2, foreground="white")])
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(0, rmax)
            ax.set_xticks(np.radians([0, 90, 180, 270]))
            ax.set_xticklabels(["N", "E", "S", "W"])
            ticks = np.arange(10, rmax + 1, 10)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:.0f}%" for t in ticks], fontsize=fs.FONT_PT)
            ax.set_rlabel_position(255)
            ax.grid(color="#cccccc", lw=0.5)
            ax.tick_params(axis="x", pad=2)
            ax.set_title(title, fontsize=fs.FONT_PT, pad=14)

        axh = fig.add_subplot(gs[1, :])
        hours = np.arange(24)
        for (title, key, res, _), ls in zip(periods, ("-", (0, (4, 2)))):
            sub = by_hour[by_hour["period"] == key]
            for rk, colour in (("reg1", REGIME_COLOURS["reg1"]), ("reg2", REGIME_COLOURS["reg2"])):
                s = sub[sub["regime_key"] == rk].set_index("local_hour")["share"].reindex(hours) * 100
                axh.plot(hours, s, color=colour, lw=1.6, ls=ls)
            c = sub[sub["regime_key"] == "calm"].set_index("local_hour")["share"].reindex(hours) * 100
            axh.plot(hours, c, color="#8c8c8c", lw=0.7, ls=ls)
        axh.set_xlim(0, 23)
        axh.set_xticks(np.arange(0, 24, 3))
        axh.set_xlabel("hour of day, Rio local time")
        axh.set_ylabel("share of reports (%)")
        axh.set_ylim(0, 100)
        names = {g["key"]: g["name"] for g in season["campaign"]["regimes"]}
        handles = [Line2D([], [], color=REGIME_COLOURS[k], lw=1.8, label=regime_title(names[k])) for k in ("reg1", "reg2")]
        handles.append(Line2D([], [], color="#8c8c8c", lw=0.8, label="calm"))
        handles += [Line2D([], [], color="black", lw=1.2, ls="-", label="campaign season"),
                    Line2D([], [], color="black", lw=1.2, ls=(0, (4, 2)), label="2015 to 2024")]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0),
                   handlelength=1.8, columnspacing=1.0, handletextpad=0.5, fontsize=fs.FONT_PT)
        out = fs.save(fig, out_path)
    return out, {"rose_radial_max_percent": rmax,
                 "campaign_regimes": [{"key": g["key"], "name": g["name"], "mean_direction_deg": g["mean_direction_deg"],
                                       "share_of_reports": g["share_of_reports"]} for g in season["campaign"]["regimes"]],
                 "climatology_regimes": [{"key": g["key"], "name": g["name"], "mean_direction_deg": g["mean_direction_deg"],
                                          "share_of_reports": g["share_of_reports"]} for g in season["climatology"]["regimes"]],
                 "n_sectors": N_SECTORS}


def build_fig_vent_profiles(points: pd.DataFrame, regimes: list[dict], out_path: Path) -> Path:
    """Windward frontal area density, canyon alignment and upwind shelter angle
    along the route (10 m means), both regimes overlaid; neighbourhood band on top."""
    with fs.figure_style():
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        p = points.sort_values("distance_along_m")
        total = float(np.ceil(p["distance_along_m"].max() / 50) * 50)
        cols = [f"{stem}_{g['slug']}" for stem, _ in VENT_PANELS for g in regimes]
        means = fs.ten_m_means(p, cols)
        fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, 5.6))
        gs = fig.add_gridspec(4, 1, height_ratios=[0.55, 1, 1, 1], hspace=0.28, left=0.13, right=0.985, top=0.99,
                              bottom=0.13)
        axb = fig.add_subplot(gs[0])
        fs.draw_neighbourhood_band(axb, fs.neighbourhood_stretches(p), total, axes_in=fs.TEXT_WIDTH_IN * 0.855)
        axes = [fig.add_subplot(gs[i + 1], sharex=axb) for i in range(3)]
        for ax, (stem, label) in zip(axes, VENT_PANELS):
            for g in regimes:
                ax.plot(means["x"], means[f"{stem}_{g['slug']}"], color=REGIME_COLOURS[g["key"]], lw=1.2, zorder=3)
            ax.set_ylabel(label)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0, labelbottom=False)
        axes[1].set_ylim(0, 90)
        axes[1].set_yticks([0, 45, 90])
        axes[2].set_ylim(bottom=0)
        axes[0].set_ylim(bottom=0)
        axes[-1].spines["bottom"].set_visible(True)
        axes[-1].tick_params(axis="x", length=3, labelbottom=True)
        fs.distance_axis(axes[-1], total)
        axes[-1].set_xlabel("distance along the route (m)")
        handles = [Line2D([], [], color=REGIME_COLOURS[g["key"]], lw=1.8, label=regime_title(g["name"])) for g in regimes]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 0.0),
                   handlelength=1.8, fontsize=fs.FONT_PT)
        return fs.save(fig, out_path)


def build_fig_shelter_maps(points: pd.DataFrame, regimes: list[dict], buildings: gpd.GeoDataFrame | None,
                           out_path: Path) -> tuple[Path, dict]:
    """Upwind shelter angle per point for each regime, side by side, one colour
    scale and one set of limits, a wind arrow in the regime colour on each map."""
    from matplotlib.collections import LineCollection

    with fs.figure_style():
        import matplotlib.pyplot as plt

        cols = [f"upwind_shelter_angle_deg_{g['slug']}" for g in regimes]
        vmin = 0.0
        vmax = float(np.ceil(np.nanmax([points[c].max() for c in cols]) / 5.0) * 5.0)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = fs.VAR_CMAP["shelter_angle"]
        o = points.sort_values("distance_along_m")
        extent = fs.route_extent(o, margin_m=30.0)
        map_w = 0.485
        h = fs.map_height_in(extent, fs.TEXT_WIDTH_IN * map_w)
        bar_h_in = 0.75
        fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, h + bar_h_in))
        top = h / (h + bar_h_in)
        xy = o[["x", "y"]].to_numpy()
        lc = None
        for i, (g, c) in enumerate(zip(regimes, cols)):
            ax = fig.add_axes([0.01 + i * 0.5, 1 - top, map_w, top])
            fs.draw_buildings(ax, buildings, extent)
            v = o[c].to_numpy(float)
            lc = LineCollection(np.stack([xy[:-1], xy[1:]], axis=1), cmap=cmap, norm=norm, linewidths=2.2, zorder=4,
                                capstyle="round")
            lc.set_array((v[:-1] + v[1:]) / 2)
            ax.add_collection(lc)
            fs.wind_arrow(ax, g["mean_direction_deg"], REGIME_COLOURS[g["key"]], regime_title(g["name"]),
                         centre=(0.78, 0.45))
            fs.north_arrow(ax, loc=(0.92, 0.88), size=0.08)
            if i == 0:
                fs.scale_bar(ax, 100.0)
        cax = fig.add_axes([0.3, 0.5 / (h + bar_h_in), 0.4, 0.13 / (h + bar_h_in)])
        cb = fig.colorbar(lc, cax=cax, orientation="horizontal")
        cb.set_label("upwind shelter angle (degrees)", labelpad=2)
        out = fs.save(fig, out_path)
    return out, {"shelter_colour_limits_deg": [vmin, vmax]}
