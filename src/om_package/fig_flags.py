"""Route points by class (overview map) and close-ups of the biggest problem stretches."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from . import fig_style as fs
from .route_repair import stretch_agreement

#: Okabe and Ito colour-blind-safe palette, one colour per class.
CLASS_COLOURS = {
    "street": "#56B4E9",
    "projected": "#E69F00",
    "beco": "#009E73",
    "covered_passage": "#CC79A7",
    "unresolved": "#222222",
}
CLASS_LABELS = {
    "street": "on a mapped street",
    "projected": "moved to the nearest open ground",
    "beco": "alley missing from the street map",
    "covered_passage": "covered passage",
    "unresolved": "unresolved",
}


def pick_insets(result: pd.DataFrame, n: int = 3) -> list[tuple[float, float]]:
    """The longest stretch with a covered or unresolved point, then the longest
    alley stretch, then the longest projected stretch (distinct stretches)."""
    st = stretch_agreement(result).sort_values("length_m", ascending=False)
    r = result.sort_values("distance_along_m")
    picks: list[dict] = []

    def has(row, classes):
        s = r[(r["distance_along_m"] >= row.start_m) & (r["distance_along_m"] <= row.end_m)]
        return s["point_class"].isin(classes).any()

    for classes in (["unresolved"], ["covered_passage"], ["beco"], ["projected"]):
        for row in st.itertuples():
            if has(row, classes) and row.start_m not in [p["start_m"] for p in picks]:
                picks.append(row._asdict())
                break
        if len(picks) == n:
            break
    return [(p["start_m"], p["end_m"]) for p in sorted(picks, key=lambda p: p["start_m"])]


def _inset(ax, res, buildings, fixes, span, pad=10.0):
    s = res[(res["distance_along_m"] >= span[0] - 3) & (res["distance_along_m"] <= span[1] + 3)]
    xs = np.r_[s["x_original"], s["x_repaired"]]
    ys = np.r_[s["y_original"], s["y_repaired"]]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + pad
    ext = (cx - half, cx + half, cy - half, cy + half)
    fs.draw_buildings(ax, buildings, ext)
    f = fixes[(fixes["x"] > ext[0]) & (fixes["x"] < ext[1]) & (fixes["y"] > ext[2]) & (fixes["y"] < ext[3])]
    ax.scatter(f["x"], f["y"], s=2.5, color="#6a6a6a", alpha=0.22, lw=0, zorder=2)
    ax.plot(s["x_original"], s["y_original"], color="#333333", lw=0.8, zorder=3)
    mv = s[s["shift_m"] > 0.01]
    for r in mv.itertuples():
        ax.plot([r.x_original, r.x_repaired], [r.y_original, r.y_repaired], color="#777777", lw=0.4, zorder=3)
    for c, col in CLASS_COLOURS.items():
        q = s[s["point_class"] == c]
        ax.scatter(q["x_repaired"], q["y_repaired"], s=9, color=col, edgecolor="white", linewidth=0.3, zorder=5)
    fs.scale_bar(ax, length_m=20.0 if half < 80 else 50.0, loc=(0.05, 0.05))
    return ext


def build_fig_flags(result: pd.DataFrame, buildings, fixes: pd.DataFrame, out_path, insets=None):
    """result: the route_repair table; buildings: footprints (GeoDataFrame);
    fixes: raw GPS fixes (x, y). Writes the flagged-points figure at text width."""
    insets = insets or pick_insets(result)
    letters = "abc"[: len(insets)]
    ext = fs.route_extent(result.rename(columns={"x_original": "x", "y_original": "y"}), margin_m=40.0)
    with fs.figure_style():
        w = fs.TEXT_WIDTH_IN
        map_w = fs.fit_map_width_in(ext, 3.4, w * 0.55)
        over_h = fs.map_height_in(ext, map_w)
        gap = 0.08
        in_w = (w - gap * (len(insets) - 1)) / len(insets)
        cap_h, key_h = 0.22, 0.28
        fig_h = over_h + 0.1 + in_w + cap_h + key_h
        fig = plt.figure(figsize=(w, fig_h))

        def frac(x_in, y_in, w_in, h_in):
            return [x_in / w, y_in / fig_h, w_in / w, h_in / fig_h]

        ax0 = fig.add_axes(frac(0, fig_h - over_h, map_w, over_h))
        fs.draw_buildings(ax0, buildings, ext)
        ax0.plot(result["x_original"], result["y_original"], color="#444444", lw=0.5, zorder=2)
        for c in CLASS_COLOURS:
            q = result[result["point_class"] == c]
            ax0.scatter(q["x_repaired"], q["y_repaired"], s=3 if c == "street" else 12, color=CLASS_COLOURS[c],
                        edgecolor="white" if c != "street" else "none", linewidth=0.2, zorder=4 if c == "street" else 6)
        fs.north_arrow(ax0, loc=(0.93, 0.88))
        fs.scale_bar(ax0, 200.0, loc=(0.04, 0.04))
        handles = [Line2D([], [], marker="o", ls="", color=CLASS_COLOURS[c], markeredgecolor="white",
                          markersize=6, label=f"{CLASS_LABELS[c]} ({int((result['point_class'] == c).sum()):,})")
                   for c in CLASS_COLOURS]
        lax = fig.add_axes(frac(map_w + 0.05, fig_h - over_h, w - map_w - 0.05, over_h))
        lax.set_axis_off()
        lax.legend(handles=handles, loc="center left", frameon=False, handletextpad=0.2, labelspacing=1.0,
                   borderaxespad=0, title="Route points (number)", alignment="left")
        lax.get_legend().get_title().set_fontweight("bold")
        for k, span in enumerate(insets):
            x0 = k * (in_w + gap)
            ax = fig.add_axes(frac(x0, key_h + cap_h, in_w, in_w))
            e = _inset(ax, result, buildings, fixes, span)
            ax.add_patch(Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2], fill=False, ec="#222222", lw=0.8, zorder=9))
            ax.text(0.03, 0.97, letters[k], transform=ax.transAxes, ha="left", va="top", fontsize=fs.FONT_PT + 1,
                    fontweight="bold", path_effects=fs.HALO, zorder=10)
            fig.text((x0 + in_w / 2) / w, (key_h + cap_h - 0.04) / fig_h,
                     f"{span[0]:,.0f} to {span[1]:,.0f} m along the route", ha="center", va="top", fontsize=fs.FONT_PT)
            ax0.add_patch(Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2], fill=False, ec="#222222", lw=0.8, zorder=7))
            ax0.text((e[0] + e[1]) / 2, e[3] + 6, letters[k], fontsize=fs.FONT_PT + 1, fontweight="bold",
                     ha="center", va="bottom", path_effects=fs.HALO, zorder=9)
        key = [Line2D([], [], color="#333333", lw=0.8, label="route trace"),
               Line2D([], [], color="#777777", lw=0.6, label="move to repaired point"),
               Line2D([], [], marker="o", ls="", color="#6a6a6a", alpha=0.5, markersize=3, label="GPS fixes"),
               Patch(facecolor="#e6e6e6", edgecolor="#a6a6a6", label="building outlines")]
        fig.legend(handles=key, loc="lower center", ncol=4, frameon=False, fontsize=fs.FONT_PT, handlelength=1.6,
                   columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
        fs.save(fig, out_path)
    return out_path
