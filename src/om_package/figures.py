"""OM2 package figures (PI, 2026-09-27): "I would like to see the spatial
result and then the sampling along the route; overlay the route on top of
the favela buildings to be easier to understand." Replaces the old
contact_sheet.py (route floating in blank space, three noisy 1 m profiles)
with:

  F1 map_form.png   — route coloured by sky_view_factor, over building
                       footprints + community outlines (the spatial result).
  F2 map_shade.png  — same base map, route coloured by mean shaded fraction
                       (P-05, building-only).
  F3 profiles.png   — sampling along the route: 1 m raw + 10 m segment
                       means for building_height_m, height_width_ratio,
                       sky_view_factor, plan_density_lambda_p,
                       ventilation_frontal_area_proxy (PROXY) and mean
                       shaded fraction.
  F4 shade_calendar.png — one strip per campaign date, distance x time of
                       day (UTC), shaded/sunlit, walk-window bracket.

  F5 sun_envelope.png — (v0.2.0, P-10) share of points always sunlit /
                       date-dependent / always shaded by local time of day,
                       plus a map of each point's date-dependent share of
                       daylight.
  F6 sun_dose.png    — (v0.2.0, P-10) 1 h clear-sky direct-sun dose along the
                       route at three local times: season envelope band and
                       the campaign dates.

Every renderer wraps its whole body in
``matplotlib.rc_context(matplotlib.rcParamsDefault)`` — a leaked rcParam
from one figure function once broke the next one drawn in the same
process (see git history) — so each is self-contained regardless of call
order.
"""
from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.patheffects as mpe
import numpy as np
import pandas as pd

from .segments import aggregate_to_segments
from .shade import daylight_rows

#: Tick spacing along the route for both maps (F1/F2) and the profile
#: panels' vertical guides (F3) — one constant so the three figures can
#: never disagree on where a "100 m" mark falls.
DISTANCE_TICK_INTERVAL_M = 100.0

#: Segment length P-03 aggregates to for the bold line in F3 — the same
#: length the package's shipped OM2/aggregate_to_segments.py defaults a
#: recipient towards, so the figure and a recipient's own re-aggregation
#: agree without them having to guess a length.
SEGMENT_LENGTH_M = 10.0

#: Padding around the route's own bounding box, as a fraction of its
#: diagonal — never a typed metre count, so a longer or shorter route (or
#: a different site's route reusing this module) gets a proportionate
#: margin rather than a fixed-size one that swamps a short route or crops
#: a long one.
MAP_MARGIN_FRACTION = 0.15
#: Floor under the fraction-derived margin, for the degenerate case of a
#: near-zero-extent route (e.g. a single-point synthetic test) where
#: `diagonal * MAP_MARGIN_FRACTION` would otherwise collapse the axes to a
#: singular (zero-width) view.
_MIN_MARGIN_M = 10.0

SVF_CMAP = "cividis"  # src.config.SVF_CMAP — the project-standard SVF colormap
SHADE_CMAP = "viridis"  # deliberately distinct from SVF_CMAP so the two maps read as different variables at a glance

_BUILDING_FACE = "#e6e6e6"
_BUILDING_EDGE = "#999999"
_COMMUNITY_LINE = "#4d4d4d"

_PROFILE_LABELS = {
    "building_height_m": "building height",
    "height_width_ratio": "H/W ratio",
    "sky_view_factor": "sky view factor",
    "plan_density_lambda_p": "plan density (λp)",
    "ventilation_frontal_area_proxy": "ventilation frontal-area PROXY",
    "mean_shaded_fraction": "daylight shaded fraction",
}
#: Order matches the PI's spec list; mean_shaded_fraction is derived (not a
#: p08 dictionary row) so its unit is stated here rather than looked up.
PROFILE_COLUMNS = [
    "building_height_m",
    "height_width_ratio",
    "sky_view_factor",
    "plan_density_lambda_p",
    "ventilation_frontal_area_proxy",
    "mean_shaded_fraction",
]
_MEAN_SHADED_FRACTION_UNIT = "fraction [0,1]"


def _rc():
    return matplotlib.rc_context(matplotlib.rcParamsDefault)


def _route_bounds_padded(points_df: pd.DataFrame, margin_fraction: float = MAP_MARGIN_FRACTION):
    """Route bbox padded by a margin derived from the route's own extent
    (never a typed metre count) — see MAP_MARGIN_FRACTION."""
    xmin, xmax = float(points_df["x"].min()), float(points_df["x"].max())
    ymin, ymax = float(points_df["y"].min()), float(points_df["y"].max())
    diag = math.hypot(xmax - xmin, ymax - ymin)
    margin = max(diag * margin_fraction, _MIN_MARGIN_M)
    return xmin - margin, xmax + margin, ymin - margin, ymax + margin


def _distance_tick_rows(points_df: pd.DataFrame, interval: float = DISTANCE_TICK_INTERVAL_M) -> pd.DataFrame:
    """One row per tick (0, interval, 2*interval, ... up to the route's max
    distance), each the point in `points_df` nearest that tick's distance.
    """
    max_d = float(points_df["distance_along_m"].max())
    ticks = np.arange(0.0, max_d + interval, interval)
    ticks = ticks[ticks <= max_d + 1e-6]
    dist = points_df["distance_along_m"].to_numpy()
    rows = []
    for t in ticks:
        idx = int(np.argmin(np.abs(dist - t)))
        row = points_df.iloc[idx].to_dict()
        row["tick_distance_m"] = float(t)
        rows.append(row)
    return pd.DataFrame(rows)


def _communities_crossed(points_df: pd.DataFrame) -> set:
    if "neighbourhood" not in points_df.columns:
        return set()
    return set(points_df["neighbourhood"].dropna().unique())


def _draw_base_map(ax, points_df: pd.DataFrame, buildings: gpd.GeoDataFrame | None,
                    subunits: gpd.GeoDataFrame | None):
    """Buildings + community outlines/names + route bbox — the shared base
    map for F1 and F2 (PI, 2026-09-27: overlay the route on the buildings,
    not floating in blank space)."""
    from shapely.geometry import box as shapely_box

    xmin, xmax, ymin, ymax = _route_bounds_padded(points_df)
    bbox_poly = shapely_box(xmin, ymin, xmax, ymax)
    crossed = _communities_crossed(points_df)

    if buildings is not None and len(buildings):
        clipped = buildings.cx[xmin:xmax, ymin:ymax]
        if len(clipped):
            clipped.plot(ax=ax, facecolor=_BUILDING_FACE, edgecolor=_BUILDING_EDGE, linewidth=0.3, zorder=1)

    if subunits is not None and len(subunits):
        name_col = "name" if "name" in subunits.columns else subunits.columns[0]
        visible = subunits[subunits.geometry.intersects(bbox_poly)]
        if len(visible):
            visible.boundary.plot(ax=ax, color=_COMMUNITY_LINE, linewidth=0.6, zorder=2)
        for _, row in visible.iterrows():
            name = row[name_col]
            if name not in crossed:
                continue
            clip = row.geometry.intersection(bbox_poly)
            if clip.is_empty:
                continue
            cx, cy = clip.centroid.x, clip.centroid.y
            ax.annotate(
                str(name), (cx, cy), fontsize=7, color="#333333", ha="center", va="center", zorder=3,
                path_effects=[mpe.withStroke(linewidth=2.2, foreground="white")],
            )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return xmin, xmax, ymin, ymax


def _draw_route_line(fig, ax, points_df: pd.DataFrame, color_col: str, cmap: str, label: str,
                      vmin: float | None = None, vmax: float | None = None):
    """Route as a LineCollection coloured by `color_col`, with distance
    ticks every DISTANCE_TICK_INTERVAL_M beside the line, a colourbar, a
    scale bar and a north arrow."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    from src.cartography import add_north_arrow, add_scale_bar

    ordered = points_df.sort_values("distance_along_m")
    xy = ordered[["x", "y"]].to_numpy()
    values = ordered[color_col].to_numpy(dtype=float)
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    seg_values = (values[:-1] + values[1:]) / 2.0

    norm = Normalize(
        vmin=vmin if vmin is not None else np.nanmin(values) if np.isfinite(values).any() else 0.0,
        vmax=vmax if vmax is not None else np.nanmax(values) if np.isfinite(values).any() else 1.0,
    )
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.6, zorder=4)
    lc.set_array(seg_values)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label=label, shrink=0.75, pad=0.02)

    ticks = _distance_tick_rows(ordered)
    ax.scatter(ticks["x"], ticks["y"], s=10, color="black", zorder=5)
    for _, row in ticks.iterrows():
        ax.annotate(
            f"{row['tick_distance_m']:.0f}", (row["x"], row["y"]), fontsize=7,
            xytext=(4, 4), textcoords="offset points", zorder=5,
            path_effects=[mpe.withStroke(linewidth=2.2, foreground="white")],
        )

    add_scale_bar(ax)
    add_north_arrow(ax)


def build_map_form(points_df: pd.DataFrame, buildings: gpd.GeoDataFrame | None,
                    subunits: gpd.GeoDataFrame | None, out_path: Path,
                    route_id: str = "OM2", version: str = "") -> Path:
    """F1 — the spatial result: route coloured by sky_view_factor over
    building footprints and community outlines."""
    with _rc():
        import matplotlib.pyplot as plt

        from src.cartography import apply_publication_style
        apply_publication_style()

        fig, ax = plt.subplots(figsize=(9, 9))
        _draw_base_map(ax, points_df, buildings, subunits)
        _draw_route_line(fig, ax, points_df, "sky_view_factor", SVF_CMAP, "sky_view_factor")

        version_suffix = f" {version}" if version else ""
        ax.set_title(f"{route_id} route — n={len(points_df)} points{version_suffix}\ncoloured by sky_view_factor (airborne, 2019 source)", fontsize=10)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def mean_shaded_fraction_by_point(shade_df: pd.DataFrame) -> pd.Series:
    """Mean of `shaded` over the DAYLIGHT 5-min steps (sun above the
    horizon) of every campaign date, per point_id — the P-05 result F2/F3
    colour/plot by. Night steps are excluded: `shaded` is True there (no
    direct sun), which is not building shade. Empty input yields an empty
    (float) Series, never a guessed value."""
    if shade_df is None or len(shade_df) == 0 or "shaded" not in shade_df.columns:
        return pd.Series(dtype=float, name="mean_shaded_fraction")
    out = daylight_rows(shade_df).groupby("point_id")["shaded"].mean().astype(float)
    out.name = "mean_shaded_fraction"
    return out


def build_map_shade(points_df: pd.DataFrame, shade_df: pd.DataFrame,
                     buildings: gpd.GeoDataFrame | None, subunits: gpd.GeoDataFrame | None,
                     out_path: Path, route_id: str = "OM2", version: str = "", tz: str = "UTC") -> Path:
    """F2 — same base map, route coloured by mean shaded fraction (P-05,
    building-only; tree_shade PENDING)."""
    with _rc():
        import matplotlib.pyplot as plt

        from src.cartography import apply_publication_style
        apply_publication_style()

        frac = mean_shaded_fraction_by_point(shade_df)
        merged = points_df.merge(frac.rename("mean_shaded_fraction"), left_on="point_id", right_index=True, how="left")

        fig, ax = plt.subplots(figsize=(9, 9))
        _draw_base_map(ax, merged, buildings, subunits)
        _draw_route_line(fig, ax, merged, "mean_shaded_fraction", SHADE_CMAP, "share of daylight in building shade",
                          vmin=0.0, vmax=1.0)

        version_suffix = f" {version}" if version else ""
        n_dates = shade_df["date"].nunique() if shade_df is not None and len(shade_df) and "date" in shade_df.columns else 0
        ax.set_title(
            f"{route_id} route — n={len(points_df)} points{version_suffix}\n"
            f"share of daylight in building shade, {n_dates} campaign date(s)",
            fontsize=10,
        )
        caption = (
            f"Building shade only, daylight steps only (sun above the horizon); times labelled {tz}."
            if n_dates else "No campaign-date shade rows in this build — empty-schema P-05 table."
        )
        fig.text(0.02, 0.01, caption, fontsize=7, color="#555555")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def _profile_frame(points_df: pd.DataFrame, shade_df: pd.DataFrame) -> pd.DataFrame:
    """points_df + a mean_shaded_fraction column — the frame F3's raw
    traces and its `aggregate_to_segments` call both operate on, so the
    figure and a directly-called `aggregate_to_segments` never disagree."""
    frac = mean_shaded_fraction_by_point(shade_df)
    return points_df.merge(frac.rename("mean_shaded_fraction"), left_on="point_id", right_index=True, how="left")


def segment_means_for_profiles(points_df: pd.DataFrame, shade_df: pd.DataFrame,
                                segment_length_m: float = SEGMENT_LENGTH_M) -> pd.DataFrame:
    """The exact 10 m segment-mean table F3's bold trace is drawn from —
    `src/om_package/segments.py aggregate_to_segments` on the profile
    frame, nothing recomputed here. Exposed so a test can assert equality
    with calling `aggregate_to_segments` directly (same function, same
    input)."""
    frame = _profile_frame(points_df, shade_df)
    return aggregate_to_segments(frame, segment_length_m)


def _dictionary_units(dictionary_df: pd.DataFrame | None) -> dict:
    if dictionary_df is None or "id" not in dictionary_df.columns:
        return {}
    return dict(zip(dictionary_df["id"], dictionary_df.get("unit", [])))


def _panel_label(col: str, units: dict) -> str:
    base = _PROFILE_LABELS.get(col, col.replace("_", " "))
    unit = units.get(col) if col != "mean_shaded_fraction" else _MEAN_SHADED_FRACTION_UNIT
    if unit and unit not in ("-",):
        return f"{base}\n({unit})"
    return base


def build_profiles(points_df: pd.DataFrame, shade_df: pd.DataFrame, out_path: Path,
                    dictionary_df: pd.DataFrame | None = None, route_id: str = "OM2",
                    version: str = "", segment_length_m: float = SEGMENT_LENGTH_M) -> Path:
    """F3 — sampling along the route: faint 1 m raw values + bold 10 m
    segment means, stacked panels sharing x = distance along route, with
    vertical guides + community names at DISTANCE_TICK_INTERVAL_M."""
    with _rc():
        import matplotlib.pyplot as plt

        from src.cartography import apply_publication_style
        apply_publication_style()

        frame = _profile_frame(points_df, shade_df).sort_values("distance_along_m")
        segments = segment_means_for_profiles(points_df, shade_df, segment_length_m)
        units = _dictionary_units(dictionary_df)
        ticks = _distance_tick_rows(frame)

        cols = [c for c in PROFILE_COLUMNS if c in frame.columns]
        fig, axes = plt.subplots(len(cols), 1, figsize=(10, 1.7 * len(cols) + 1.2), sharex=True)
        if len(cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, cols):
            ax.plot(frame["distance_along_m"], frame[col], color="#b0b0b0", lw=0.6, zorder=1, label="1 m raw")
            if col in segments.columns:
                seg_x = (segments["segment_start_m"] + segments["segment_end_m"]) / 2.0
                ax.plot(seg_x, segments[col], color="#1a5fa5", lw=1.8, marker="o", markersize=2.5, zorder=2, label="10 m mean")
            for _, row in ticks.iterrows():
                ax.axvline(row["tick_distance_m"], color="#dddddd", lw=0.7, zorder=0)
            ax.set_ylabel(_panel_label(col, units), fontsize=7.5)

        axes[0].legend(loc="upper right", fontsize=6, frameon=False)
        axes[-1].set_xlabel("distance along route (m)")

        for _, row in ticks.iterrows():
            name = row.get("neighbourhood") or ""
            axes[0].annotate(
                str(name), (row["tick_distance_m"], 1.02), xycoords=("data", "axes fraction"),
                fontsize=6.5, rotation=45, ha="left", va="bottom", color="#444444",
            )

        version_suffix = f" {version}" if version else ""
        fig.suptitle(f"{route_id} sampling along the route — n={len(frame)} points, {segment_length_m:g} m segments{version_suffix}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def _dates_in_shade_table(shade_df: pd.DataFrame) -> list:
    if shade_df is None or len(shade_df) == 0 or "date" not in shade_df.columns:
        return []
    return sorted(shade_df["date"].dropna().unique().tolist())


def build_shade_calendar(points_df: pd.DataFrame, shade_df: pd.DataFrame,
                          campaign_windows_df: pd.DataFrame | None, out_path: Path,
                          route_id: str = "OM2", version: str = "") -> Path:
    """F4 — one strip per campaign date: x = distance along route, y = time
    of day (UTC, 5-min resolution), cell = building shade (dark) / sunlit
    (light) / night, sun below the horizon (grey, never drawn as shade),
    walk window drawn as a bracket."""
    with _rc():
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        from src.cartography import apply_publication_style
        apply_publication_style()

        dates = _dates_in_shade_table(shade_df)
        dist_by_point = points_df.set_index("point_id")["distance_along_m"]

        n_rows = max(len(dates), 1)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 1.6 * n_rows + 1.0), sharex=True, squeeze=False)
        axes = axes[:, 0]
        cmap = ListedColormap(["#f4f1e8", "#2b2b2b", "#9aa3ad"])  # 0 sunlit, 1 building shade, 2 night

        if not dates:
            axes[0].text(0.5, 0.5, "No campaign-date shade rows in this build (empty-schema P-05 table).",
                         ha="center", va="center", fontsize=9, transform=axes[0].transAxes)
            axes[0].set_axis_off()
        for ax, d in zip(axes, dates):
            day = shade_df[shade_df["date"] == d].copy()
            day["distance_along_m"] = day["point_id"].map(dist_by_point)
            day = day.dropna(subset=["distance_along_m"])
            day["state"] = day["shaded"].astype(float).where(day["sun_altitude_deg"] > 0, 2.0)
            pivot = day.pivot_table(index="timestamp", columns="distance_along_m", values="state", aggfunc="first")
            pivot = pivot.sort_index()
            if pivot.shape[0] and pivot.shape[1]:
                times = pd.to_datetime(pivot.index)
                y_frac = [t.hour + t.minute / 60.0 for t in times]
                ax.imshow(
                    pivot.to_numpy(dtype=float),
                    aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
                    extent=[pivot.columns.min(), pivot.columns.max(), max(y_frac), min(y_frac)],
                )
            ax.set_ylabel(f"{d}\ntime (UTC)", fontsize=7.5)

            if campaign_windows_df is not None and len(campaign_windows_df) and "date" in campaign_windows_df.columns:
                win = campaign_windows_df[campaign_windows_df["date"].astype(str) == str(d)]
                if len(win):
                    first_t = pd.to_datetime(win["first_timestamp"].iloc[0])
                    last_t = pd.to_datetime(win["last_timestamp"].iloc[0])
                    y0 = first_t.hour + first_t.minute / 60.0
                    y1 = last_t.hour + last_t.minute / 60.0
                    xmin = points_df["distance_along_m"].min()
                    bracket_x = xmin - (points_df["distance_along_m"].max() - xmin) * 0.03
                    ax.annotate(
                        "", xy=(bracket_x, y1), xytext=(bracket_x, y0),
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color="#c0392b", lw=1.4,
                                         connectionstyle="bar,fraction=0.15"),
                    )

        axes[-1].set_xlabel("distance along route (m)")
        version_suffix = f" {version}" if version else ""
        fig.suptitle(f"{route_id} building shade calendar — {len(dates)} campaign date(s){version_suffix}\n"
                     "dark = building shade · light = sun · grey = night (sun below the horizon)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


#: Colours for the three P-10 classes; date_dependent carries the message.
_CLASS_COLOURS = {"always_shaded": "#2b2b2b", "date_dependent": "#d98c1f", "always_sunlit": "#f4efd6"}
_CLASS_ORDER = ["always_shaded", "date_dependent", "always_sunlit"]
DATE_DEPENDENT_CMAP = "YlOrBr"
_DATE_COLOURS = ["#1b6ca8", "#c0392b", "#2e8b57", "#8e44ad", "#7f6000"]
#: How many local times the dose figure shows: the early / middle / late
#: sun-up slot of the window (quartiles of the slot list, read from the data).
_DOSE_SLOT_QUANTILES = (0.25, 0.5, 0.75)


def _slot_hours(slots) -> np.ndarray:
    return np.array([int(x[:2]) + int(x[3:5]) / 60.0 for x in slots])


def daylight_date_dependent_share_by_point(envelope_df: pd.DataFrame) -> pd.Series:
    """Per point: share of its daylight slots (class != night) that are
    date_dependent."""
    day = envelope_df[envelope_df["class"] != "night"]
    out = (day["class"] == "date_dependent").groupby(day["point_id"]).mean().astype(float)
    out.name = "date_dependent_share"
    return out


def class_shares_by_slot(envelope_df: pd.DataFrame) -> pd.DataFrame:
    """Rows = local slot (daylight only), columns = class, values = share of
    points in that class at that slot."""
    day = envelope_df[envelope_df["class"] != "night"]
    counts = day.groupby(["local_slot", "class"]).size().unstack(fill_value=0)
    for c in _CLASS_ORDER:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[_CLASS_ORDER]
    return counts.div(counts.sum(axis=1), axis=0)


def build_sun_envelope(points_df: pd.DataFrame, envelope_df: pd.DataFrame,
                       buildings: gpd.GeoDataFrame | None, subunits: gpd.GeoDataFrame | None,
                       out_path: Path, route_id: str = "OM2", version: str = "", window: tuple | None = None,
                       tz_label: str = "Rio local time", geometry_label: str = "") -> Path:
    """F5 — how much the unknown campaign date costs: (left) share of the
    route's points always sunlit / date-dependent / always shaded at each
    local time of day; (right) map of each point's date-dependent share of
    daylight. Geometry-derived proxy (building horizon vs sun position)."""
    with _rc():
        import matplotlib.pyplot as plt

        from src.cartography import apply_publication_style
        apply_publication_style()

        shares = class_shares_by_slot(envelope_df)
        hours = _slot_hours(shares.index)
        per_point = daylight_date_dependent_share_by_point(envelope_df)
        merged = points_df.merge(per_point, left_on="point_id", right_index=True, how="left")

        fig = plt.figure(figsize=(14, 7.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        ax.stackplot(hours, [shares[c].to_numpy() * 100 for c in _CLASS_ORDER],
                     colors=[_CLASS_COLOURS[c] for c in _CLASS_ORDER], edgecolor="#777777", linewidth=0.4,
                     labels=["always shaded", "date-dependent", "always sunlit"])
        ax.set_xlim(hours.min(), hours.max())
        ax.set_ylim(0, 100)
        ax.set_xlabel(f"local time of day ({tz_label})")
        ax.set_ylabel("share of route points (%)")
        ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=True, framealpha=0.95)
        wtxt = f"{window[0]} to {window[1]}" if window else "the analysis window"
        ax.set_title(f"Sun class by time of day over {wtxt}\n(days with the sun up only)", fontsize=10)

        axm = fig.add_subplot(gs[0, 1])
        _draw_base_map(axm, merged, buildings, subunits)
        _draw_route_line(fig, axm, merged, "date_dependent_share", DATE_DEPENDENT_CMAP,
                         "share of daylight that is date-dependent", vmin=0.0, vmax=1.0)
        axm.set_title("Where the campaign date matters\n(per point, over its daylight slots)", fontsize=10)

        suffix = f" {version}" if version else ""
        fig.suptitle(f"{route_id} sun exposure envelope{suffix}", fontsize=11)
        geo = f" Geometry: {geometry_label}." if geometry_label else ""
        fig.text(0.01, -0.02, "Brisa+ (MorphoFavela). Geometry-derived proxy (building and terrain horizon vs sun position); "
                 "no cloud, no tree shade; not measured sunlight.\n" + geo.strip(), fontsize=7, color="#555555")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def dose_slots_for_figure(envelope_df: pd.DataFrame, dose_df: pd.DataFrame,
                          quantiles: tuple = _DOSE_SLOT_QUANTILES) -> list[str]:
    """Local slots shown in F6: the slots at the given quantiles of the
    sun-up slot list that also exist on the dose table's slot grid."""
    sun_up = sorted(envelope_df.loc[envelope_df["class"] != "night", "local_slot"].unique())
    on_grid = sorted(set(dose_df["local_slot"].unique()))
    pool = [x for x in sun_up if x in on_grid]
    if not pool:
        return []
    return [pool[min(int(q * len(pool)), len(pool) - 1)] for q in quantiles]


def build_sun_dose(points_df: pd.DataFrame, envelope_df: pd.DataFrame, dose_df: pd.DataFrame, out_path: Path,
                   route_id: str = "OM2", version: str = "", tz_label: str = "Rio local time",
                   geometry_label: str = "") -> Path:
    """F6 — 1 h clear-sky direct-sun dose along the route at three local
    times of day: band = min to max over every day of the season window,
    grey line = median, coloured lines = the campaign dates. UPPER BOUND
    (clear sky), geometry-derived proxy."""
    with _rc():
        import matplotlib.pyplot as plt

        from src.cartography import apply_publication_style
        apply_publication_style()

        col = "dose_1h_wh_m2"
        slots = dose_slots_for_figure(envelope_df, dose_df)
        dist = points_df.set_index("point_id")["distance_along_m"]
        dates = sorted(s for s in dose_df["scope"].astype(str).unique() if not s.startswith("envelope_"))
        n = max(len(slots), 1)
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n + 1.2), sharex=True, squeeze=False)
        axes = axes[:, 0]
        for ax, slot in zip(axes, slots):
            sl = dose_df[dose_df["local_slot"] == slot].copy()
            sl["d"] = sl["point_id"].map(dist)
            sl = sl.dropna(subset=["d"]).sort_values("d")
            wide = sl.pivot_table(index="d", columns="scope", values=col, aggfunc="first")
            lo, hi, med = (wide[f"envelope_{k}"] for k in ("min", "max", "median"))
            ax.fill_between(wide.index, lo, hi, color="#cfd8e3", label="season envelope (min to max)", zorder=1)
            ax.plot(wide.index, med, color="#555555", lw=0.8, label="season median", zorder=2)
            for d, c in zip(dates, _DATE_COLOURS * (len(dates) // len(_DATE_COLOURS) + 1)):
                if d in wide.columns:
                    ax.plot(wide.index, wide[d], color=c, lw=0.9, label=d, zorder=3)
            ax.set_ylabel(f"1 h dose up to {slot}\n(Wh/m2)", fontsize=8)
            ax.set_ylim(bottom=0)
        if not slots:
            axes[0].text(0.5, 0.5, "No dose rows in this build.", ha="center", va="center", transform=axes[0].transAxes)
            axes[0].set_axis_off()
        else:
            axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
        axes[-1].set_xlabel("distance along route (m)")
        suffix = f" {version}" if version else ""
        fig.suptitle(f"{route_id} direct-sun dose, 1 h window, along the route{suffix}\n"
                     f"campaign dates vs the season envelope ({tz_label})", fontsize=10)
        geo = f" Geometry: {geometry_label}." if geometry_label else ""
        fig.text(0.01, 0.005, "Brisa+ (MorphoFavela). Clear-sky direct beam on a horizontal plane, zero where building/terrain horizon\n"
                 "blocks the sun: an UPPER BOUND and a geometry-derived proxy, not measured sunlight." + geo,
                 fontsize=7, color="#555555")
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path
