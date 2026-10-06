"""OM2 report figures, part 1: route, street form, shade and sun dose.

Filenames are the contract with the report text (written under OM2/):
fig_route, fig_form, fig_shade_map, fig_shade_calendar, fig_sun_dose,
fig_svf_sensor. The ventilation and wind figures are in vent_figures.py.
Every figure is drawn at print width and styled by fig_style.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

from . import fig_style as fs
from .shade import daylight_rows

#: Segment length of the 10 m means in every profile figure.
SEGMENT_LENGTH_M = 10.0
ROUTE_TICK_M = 250.0

#: Local slots used by the old dose figure; kept for the package page and report helpers.
_DOSE_SLOT_QUANTILES = (0.25, 0.5, 0.75)

#: Dose below this (Wh/m2, the rounding step of the table) counts as zero and is drawn in fs.NO_SUN.
DOSE_ZERO_BELOW = 0.1
#: Rows closer than this to the walks' median duration are not preferred for the sensor figure; see pick_representative_walk.
COVERAGE_FULL = 0.95


#: Designed heights (inches). Figures print at 100 % of their size, so each one
#: has to fit its page with the section heading and the lead paragraph.
ROUTE_MAX_HEIGHT_IN = 5.85
SHADE_MAP_MAX_HEIGHT_IN = 4.85
SHADE_BAR_COLUMN_IN = 1.0
FORM_HEIGHT_IN = 6.2
SUN_DOSE_HEIGHT_IN = 7.4


def load_shade_frame(parquet_path: Path) -> pd.DataFrame:
    """The full shipped shade table (every 5 min step), columns the figures need, ids dictionary-encoded."""
    import pyarrow.parquet as pq

    t = pq.read_table(parquet_path, columns=["point_id", "timestamp_local", "date", "sun_altitude_deg", "shaded"],
                      read_dictionary=["point_id", "date"])
    return t.to_pandas()


def mean_shaded_fraction_by_point(shade_df: pd.DataFrame) -> pd.Series:
    """Mean of `shaded` over the daylight steps of every walk date, per point_id.
    Night steps are excluded: `shaded` is True there, which is not building shade."""
    if shade_df is None or len(shade_df) == 0 or "shaded" not in shade_df.columns:
        return pd.Series(dtype=float, name="mean_shaded_fraction")
    out = daylight_rows(shade_df).groupby("point_id")["shaded"].mean().astype(float)
    out.name = "mean_shaded_fraction"
    return out


def dose_slots_for_figure(envelope_df: pd.DataFrame, dose_df: pd.DataFrame,
                          quantiles: tuple = _DOSE_SLOT_QUANTILES) -> list[str]:
    sun_up = sorted(envelope_df.loc[envelope_df["class"] != "night", "local_slot"].unique())
    on_grid = sorted(set(dose_df["local_slot"].unique()))
    pool = [x for x in sun_up if x in on_grid]
    if not pool:
        return []
    return [pool[min(int(q * len(pool)), len(pool) - 1)] for q in quantiles]


def _route_line(ax, points: pd.DataFrame, colour="black", lw=1.8, zorder=4):
    from matplotlib.collections import LineCollection

    o = points.sort_values("distance_along_m")
    xy = o[["x", "y"]].to_numpy()
    lc = LineCollection(np.stack([xy[:-1], xy[1:]], axis=1), colors=colour, linewidths=lw, zorder=zorder,
                        capstyle="round")
    ax.add_collection(lc)


def _distance_ticks(points: pd.DataFrame, step: float = ROUTE_TICK_M) -> pd.DataFrame:
    o = points.sort_values("distance_along_m")
    d = o["distance_along_m"].to_numpy()
    ticks = np.arange(0.0, d.max() + 1e-6, step)
    rows = o.iloc[[int(np.argmin(np.abs(d - t))) for t in ticks]].copy()
    rows["tick_m"] = ticks
    return rows


def _unit_normal(points: pd.DataFrame, distance_m: float, half_window_m: float = 15.0):
    o = points.sort_values("distance_along_m")
    d = o["distance_along_m"].to_numpy()
    lo = o.iloc[int(np.argmin(np.abs(d - max(distance_m - half_window_m, 0))))]
    hi = o.iloc[int(np.argmin(np.abs(d - (distance_m + half_window_m))))]
    tx, ty = hi["x"] - lo["x"], hi["y"] - lo["y"]
    n = float(np.hypot(tx, ty)) or 1.0
    return -ty / n, tx / n


def build_fig_route(points: pd.DataFrame, buildings: gpd.GeoDataFrame | None, out_path: Path) -> Path:
    """Route over building footprints: distance ticks every 250 m, neighbourhood
    names once per stretch, scale bar, north arrow. No colour-coded variable."""
    with fs.figure_style():
        import matplotlib.pyplot as plt

        extent = fs.route_extent(points, margin_m=45.0)
        w_in = fs.fit_map_width_in(extent, ROUTE_MAX_HEIGHT_IN)
        fig, ax = plt.subplots(figsize=(w_in, fs.map_height_in(extent, w_in)))
        fs.draw_buildings(ax, buildings, extent)
        _route_line(ax, points, colour="#111111", lw=1.8)

        ticks = _distance_ticks(points)
        ax.scatter(ticks["x"], ticks["y"], s=16, color="white", edgecolor="black", linewidth=0.9, zorder=6)
        o = points.sort_values("distance_along_m")
        items = [{"xy": (r["x"], r["y"]), "text": f"{r['tick_m']:,.0f} m"} for _, r in ticks.iterrows()]
        for s_ in fs.neighbourhood_stretches(points):
            mid = (s_["start"] + s_["end"]) / 2
            r = o.iloc[int(np.argmin(np.abs(o["distance_along_m"].to_numpy() - mid)))]
            items.append({"xy": (r["x"], r["y"]), "text": s_["name"], "kw": {"fontstyle": "italic"}})
        fs.place_labels(fig, ax, items, o[["x", "y"]].to_numpy())

        fs.scale_bar(ax, 100.0)
        fs.north_arrow(ax)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
        return fs.save(fig, out_path)


def build_fig_form(points: pd.DataFrame, out_path: Path) -> Path:
    """Four stacked profiles sharing distance: building height, height-to-width
    ratio, sky view factor, plan area density. Neighbourhood band once on top."""
    panels = [
        ("building_height_m", "building\nheight (m)"),
        ("height_width_ratio", "height-to-width\nratio"),
        ("sky_view_factor", "sky view factor"),
        ("plan_density_lambda_p", "plan area density"),
    ]
    with fs.figure_style():
        import matplotlib.pyplot as plt

        p = points.sort_values("distance_along_m")
        total = float(np.ceil(p["distance_along_m"].max() / 50) * 50)
        means = fs.ten_m_means(p, [c for c, _ in panels])
        fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, FORM_HEIGHT_IN))
        gs = fig.add_gridspec(5, 1, height_ratios=[0.55, 1, 1, 1, 1], hspace=0.28, left=0.13, right=0.985,
                              top=0.99, bottom=0.115)
        axb = fig.add_subplot(gs[0])
        fs.draw_neighbourhood_band(axb, fs.neighbourhood_stretches(p), total, axes_in=fs.TEXT_WIDTH_IN * 0.855)
        axes = [fig.add_subplot(gs[i + 1], sharex=axb) for i in range(4)]
        for ax, (col, label) in zip(axes, panels):
            ax.plot(p["distance_along_m"], p[col], color=fs.POINT_GREY, lw=0.5, zorder=1)
            ax.plot(means["x"], means[col], color=fs.LINE_DARK, lw=1.2, zorder=3)
            ax.set_ylabel(label)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0, labelbottom=False)
            hi = float(max(np.nanpercentile(p[col], 99.5), np.nanmax(means[col])))
            lo = float(np.nanmin(p[col]))
            ax.set_ylim(min(lo, 0.0) if col != "sky_view_factor" else 0.0, hi * 1.05 if col != "sky_view_factor" else 1.0)
        for ax in axes:
            ax.sharex(axb)
        axes[-1].spines["bottom"].set_visible(True)
        axes[-1].tick_params(axis="x", length=3, labelbottom=True)
        fs.distance_axis(axes[-1], total)
        axes[-1].set_xlabel("distance along the route (m)")
        axes[0].set_xlim(0, total)
        fs.shade_flagged(axes, fs.flagged_spans(p))
        fig.legend(handles=[fs.flag_handle()], loc="lower center", frameon=False, bbox_to_anchor=(0.55, 0.0))
        return fs.save(fig, out_path)


def _sun_share_norm():
    return matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)


def build_fig_shade_map(points: pd.DataFrame, shade_df: pd.DataFrame, buildings: gpd.GeoDataFrame | None,
                        out_path: Path) -> Path:
    """Share of daylight time in direct sun (1 minus the building shade share) per point, over the walk dates."""
    from matplotlib.collections import LineCollection

    with fs.figure_style():
        import matplotlib.pyplot as plt

        frac = mean_shaded_fraction_by_point(shade_df)
        o = points.merge(frac, left_on="point_id", right_index=True, how="left").sort_values("distance_along_m")
        extent = fs.route_extent(o, margin_m=45.0)
        map_in = fs.fit_map_width_in(extent, SHADE_MAP_MAX_HEIGHT_IN, fs.TEXT_WIDTH_IN - SHADE_BAR_COLUMN_IN)
        h_in = fs.map_height_in(extent, map_in)
        w_in = map_in + SHADE_BAR_COLUMN_IN
        fig = plt.figure(figsize=(w_in, h_in))
        ax = fig.add_axes([0.005 * 6.3 / w_in, 0.005, map_in / w_in, 0.99])
        fs.draw_buildings(ax, buildings, extent)
        xy = o[["x", "y"]].to_numpy()
        v = 1.0 - o["mean_shaded_fraction"].to_numpy(float)
        lc = LineCollection(np.stack([xy[:-1], xy[1:]], axis=1), cmap=fs.VAR_CMAP["sun_share"], norm=_sun_share_norm(),
                            linewidths=2.6, zorder=4, capstyle="round")
        lc.set_array((v[:-1] + v[1:]) / 2)
        ax.add_collection(lc)
        fs.scale_bar(ax, 100.0)
        fs.north_arrow(ax)
        cax = fig.add_axes([(map_in + 0.3) / w_in, 0.25, 0.14 / w_in, 0.5])
        cb = fig.colorbar(lc, cax=cax)
        cb.set_label("share of daylight in direct sun")
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        return fs.save(fig, out_path)


def shade_calendar_matrix(shade_df: pd.DataFrame, bin_min: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """Rows = walk dates, columns = local minutes since midnight (daylight bins),
    values = share of route points in building shade."""
    day = daylight_rows(shade_df)
    ts = day["timestamp_local"] if "timestamp_local" in day.columns else day["timestamp"]
    ts = pd.DatetimeIndex(ts)
    minutes = (ts.hour * 60 + ts.minute) // bin_min * bin_min
    share = day.groupby([day["date"].astype(str).to_numpy(), np.asarray(minutes)])["shaded"].mean()
    mat = share.unstack(level=1).sort_index()
    return mat, [str(d) for d in mat.index]


def build_fig_shade_calendar(shade_df: pd.DataFrame, out_path: Path, bin_min: int = 5) -> tuple[Path, dict]:
    with fs.figure_style():
        import matplotlib.pyplot as plt

        mat, dates = shade_calendar_matrix(shade_df, bin_min)
        mins = mat.columns.to_numpy(float)
        edges_x = np.append(mins, mins[-1] + bin_min) / 60.0
        n = len(dates)
        fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, 5.4))
        ax = fig.add_axes([0.085, 0.085, 0.765, 0.905])
        cax = fig.add_axes([0.87, 0.085, 0.022, 0.905])
        cmap = matplotlib.colormaps[fs.VAR_CMAP["sun_share"]].copy()
        cmap.set_bad("white")
        mesh = ax.pcolormesh(edges_x, np.arange(n + 1), np.ma.masked_invalid(1.0 - mat.to_numpy(float)), cmap=cmap,
                             norm=_sun_share_norm(), shading="flat", rasterized=True)
        ax.set_ylim(n, 0)
        ax.set_xlim(edges_x[0], edges_x[-1])
        lab = [f"{pd.Timestamp(d).day} {pd.Timestamp(d).strftime('%b')}" for d in dates]
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_yticklabels(lab)
        ax.tick_params(axis="y", length=0)
        hours = np.arange(np.ceil(edges_x[0]), edges_x[-1] + 0.01, 2)
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{int(h):02d}:00" for h in hours])
        ax.set_xlabel("time of day, Rio local time")
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label("share of route points in direct sun")
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        out = fs.save(fig, out_path)
    return out, {"n_dates": n, "bin_min": bin_min, "first_bin_local": f"{int(mins[0]) // 60:02d}:{int(mins[0]) % 60:02d}",
                 "last_bin_local": f"{int(mins[-1]) // 60:02d}:{int(mins[-1]) % 60:02d}",
                 "colour_limits": [0.0, 1.0]}


def walk_order(walks: pd.DataFrame) -> pd.DataFrame:
    """Walks sorted by date within each period, morning first, with the row label."""
    w = walks.copy()
    start = pd.to_datetime(w["start_local"].str[:19])
    w["_start"] = start
    w["label"] = [f"{t.day} {t.strftime('%b')} {t:%H:%M}" for t in start]
    w["_grp"] = (w["period"] != "morning").astype(int)
    return w.sort_values(["_grp", "_start"]).reset_index(drop=True)


def dose_matrix(p12: pd.DataFrame, walk_ids: list[str], column: str, total_m: float, step_m: float = 10.0) -> np.ndarray:
    nb = int(np.ceil(total_m / step_m))
    b = np.minimum(np.floor(p12["distance_along_m"].to_numpy(float) / step_m).astype(int), nb - 1)
    g = p12.assign(_b=b).groupby(["walk_id", "_b"])[column].mean()
    wide = g.unstack("_b").reindex(index=walk_ids, columns=range(nb))
    return wide.to_numpy(float)


def build_fig_sun_dose(walks: pd.DataFrame, p12: pd.DataFrame, total_m: float, out_path: Path) -> tuple[Path, dict]:
    """R10: walks as rows (mornings above evenings), distance as columns, colour
    = clear-sky direct sun dose in the 1 h (left) and 3 h (right) before arrival;
    one shared scale, no sun darker than the scale, points outside a walk blank."""
    with fs.figure_style():
        import matplotlib.pyplot as plt

        order = walk_order(walks)
        ids = order["walk_id"].tolist()
        mats = {h: dose_matrix(p12, ids, f"dose_{h}h_before_wh_m2", total_m) for h in (1, 3)}
        vmax = float(np.nanpercentile(mats[3], 99.5))
        vmax = float(np.ceil(vmax / 250.0) * 250.0)
        norm = matplotlib.colors.Normalize(vmin=DOSE_ZERO_BELOW, vmax=vmax)
        cmap = matplotlib.colormaps[fs.VAR_CMAP["sun_dose"]].copy()
        cmap.set_under(fs.NO_SUN)
        cmap.set_bad("white")

        n_m = int((order["_grp"] == 0).sum())
        n_e = len(order) - n_m
        gap = 1.2
        y_m = np.arange(n_m + 1)
        y_e = np.arange(n_e + 1) + n_m + gap
        edges_x = np.arange(0, mats[1].shape[1] + 1) * 10.0

        fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, SUN_DOSE_HEIGHT_IN))
        left, width, gapx = 0.215, 0.375, 0.02
        H = SUN_DOSE_HEIGHT_IN
        bottom, height = 0.95 / H, (H - 0.95 - 0.25) / H
        axes = [fig.add_axes([left + i * (width + gapx), bottom, width, height]) for i in range(2)]
        for ax, h, name in zip(axes, (1, 3), ("1 hour before", "3 hours before")):
            m = np.ma.masked_invalid(mats[h])
            ax.pcolormesh(edges_x, y_m, m[:n_m], cmap=cmap, norm=norm, shading="flat", rasterized=True)
            ax.pcolormesh(edges_x, y_e, m[n_m:], cmap=cmap, norm=norm, shading="flat", rasterized=True)
            ax.set_ylim(y_e[-1], 0)
            ax.set_xlim(0, total_m)
            fs.distance_axis(ax, total_m, step=500.0)
            ax.set_xlabel("distance along the route (m)")
            ax.set_title(name, fontsize=fs.FONT_PT, pad=3)
            for s in ("left", "bottom"):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis="y", length=0)
        centres = np.concatenate([y_m[:-1] + 0.5, y_e[:-1] + 0.5])
        axes[0].set_yticks(centres)
        axes[0].set_yticklabels(order["label"].tolist())
        axes[1].set_yticks([])
        for lab, y0, y1 in (("morning walks", y_m[0], y_m[-1]), ("evening walks", y_e[0], y_e[-1])):
            axes[0].annotate(lab, xy=(-0.40, 0), xycoords=("axes fraction", "data"), xytext=(-0.40, (y0 + y1) / 2),
                             textcoords=("axes fraction", "data"), rotation=90, ha="center", va="center",
                             fontsize=fs.FONT_PT)
        cax = fig.add_axes([left + 0.09, 0.39 / H, 2 * width + gapx - 0.11, 0.09 / H])
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        ticks = [DOSE_ZERO_BELOW, *range(500, int(vmax) + 1, 500)]
        cb.set_ticks(ticks)
        cb.set_ticklabels(["0", *[f"{t:,}" for t in ticks[1:]]])
        cb.set_label("clear-sky direct sun dose (Wh/m²)", labelpad=2)
        sw = fig.add_axes([left + 0.02, 0.39 / H, 0.03, 0.09 / H])
        sw.set_facecolor(fs.NO_SUN)
        sw.set_xticks([])
        sw.set_yticks([])
        for sp in sw.spines.values():
            sp.set_visible(True)
        fig.text(left + 0.035, 0.30 / H, "no sun", ha="center", va="top", fontsize=fs.FONT_PT)
        out = fs.save(fig, out_path)
    return out, {"colour_limits_wh_m2": [DOSE_ZERO_BELOW, vmax], "zero_drawn_below_wh_m2": DOSE_ZERO_BELOW,
                 "colour_scale_note": "one scale for both panels, upper limit set from the 99.5th percentile of the 3 hour doses; larger values take the top colour",
                 "n_walks": len(order), "n_morning": n_m, "n_evening": n_e, "bin_m": 10,
                 "max_10m_mean_1h_wh_m2": float(np.nanmax(mats[1])), "max_10m_mean_3h_wh_m2": float(np.nanmax(mats[3]))}


def pick_representative_walk(walks: pd.DataFrame) -> pd.Series:
    """Among walks with coverage >= 0.95, the one closest to the median duration of all walks."""
    med = float(walks["duration_min"].median())
    full = walks[walks["coverage_share"] >= COVERAGE_FULL]
    return full.loc[(full["duration_min"] - med).abs().idxmin()]


def build_fig_svf_sensor(points: pd.DataFrame, walks: pd.DataFrame, p12: pd.DataFrame, out_path: Path) -> tuple[Path, dict]:
    """Sky view factor at 1 m and sensor-matched (tau = 10 s and 30 s) along one full-coverage walk."""
    walk = pick_representative_walk(walks)
    with fs.figure_style():
        import matplotlib.pyplot as plt

        w = p12[p12["walk_id"] == walk["walk_id"]].sort_values("distance_along_m")
        pts = points.sort_values("distance_along_m")
        fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH_IN, 2.6))
        ax.plot(pts["distance_along_m"], pts["sky_view_factor"], color="#a0a0a0", lw=0.6, label="1 m values", zorder=1)
        for tau in (10, 30):
            ax.plot(w["distance_along_m"], w[f"sky_view_factor_tau{tau}s"], color=fs.TAU_COLOURS[tau], lw=1.4,
                    label=f"sensor-matched, {tau} s", zorder=3)
        fs.distance_axis(ax, float(pts["distance_along_m"].max()))
        ax.set_xlabel("distance along the route (m)")
        ax.set_ylabel("sky view factor")
        ax.set_ylim(0, 1)
        fs.shade_flagged([ax], fs.flagged_spans(pts))
        h, l = ax.get_legend_handles_labels()
        fig.legend(h + [fs.flag_handle()], l + [fs.FLAG_LABEL], loc="lower left", ncol=2, frameon=False,
                   handlelength=1.8, fontsize=fs.FONT_PT, bbox_to_anchor=(0.0, 0.0))
        fig.subplots_adjust(left=0.1, right=0.985, top=0.97, bottom=0.36)
        out = fs.save(fig, out_path)
    start = pd.Timestamp(str(walk["start_local"])[:19])
    return out, {"walk_id": str(walk["walk_id"]), "date": str(walk["date"]), "start_local": start.strftime("%H:%M"),
                 "period": str(walk["period"]), "duration_min": float(walk["duration_min"]),
                 "coverage_share": float(walk["coverage_share"]),
                 "median_duration_min_all_walks": float(walks["duration_min"].median())}
