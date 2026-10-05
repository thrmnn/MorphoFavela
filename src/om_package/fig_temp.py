"""Figures of the temperature pairing section (fig_temp_tau, fig_temp_profile).

Both read only the files temp_pairing.run writes into the package, so they can
be redrawn without rerunning the analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import fig_style as fs
from .temp_pairing import keep_mask

#: One colour per period (Okabe and Ito): not the wind regime or time constant colours.
PERIOD_COLOURS = {"morning": "#56B4E9", "evening": "#D55E00"}
DETREND_GREY = "#7a7a7a"
#: Where tau = 0 (the 1 m value) sits on the logarithmic axis.
TAU0_X = 2.5
EXCLUDED_LABEL = "points left out (alley missing from the street map, covered passage or unresolved)"


def _tau_x(t):
    t = np.asarray(t, float)
    return np.where(t == 0, TAU0_X, t)


def build_fig_temp_tau(package_dir: Path, out_path: Path | None = None) -> Path:
    """Left: share of within-walk variance explained against tau (in sample with walk bootstrap band, and
    leaving one walk out), with the event-based interval. Right: mean response around sun and shade changes."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .temp_pairing import EVENT_MIN_SHARE

    package_dir = Path(package_dir)
    facts = json.loads((package_dir / "OM2" / "temp_facts.json").read_text(encoding="utf-8"))
    scan = pd.read_csv(package_dir / "p13_temperature_pairing_tau_scan.csv")
    resp = pd.read_csv(package_dir / "p13_temperature_pairing_event_response.csv")
    ev = facts["events"]
    out_path = out_path or package_dir / "OM2" / "fig_temp_tau.png"
    xmax = 330.0
    with fs.figure_style():
        fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, 3.0))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.34, left=0.095, right=0.985, top=0.97,
                              bottom=0.33)
        ax = fig.add_subplot(gs[0])
        ax.axhline(0, color=fs.POINT_GREY, lw=0.6)
        for per, c in PERIOD_COLOURS.items():
            s = scan[scan["period"] == per]
            x = _tau_x(s["tau_s"])
            ax.fill_between(x, 100 * s["r2_lo"], 100 * s["r2_hi"], color=c, alpha=0.18, lw=0)
            ax.plot(x, 100 * s["r2_within"], color=c, lw=1.5, marker="o", ms=2.8)
            ax.plot(x, 100 * s["cv_r2"], color=c, lw=0.8, marker="o", ms=2.8, mfc="white")
        ax.set_xscale("log")
        ticks = [0, 5, 10, 30, 60, 120, 300]
        ax.set_xticks(_tau_x(ticks))
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(TAU0_X * 0.8, xmax)
        ax.set_xlabel("time constant τ (s)")
        ax.set_ylabel("share of variance explained (%)")
        y0, y1 = ax.get_ylim()
        y = y1 + (y1 - y0) * 0.06
        lo, hi = max(ev["tau_lo"], TAU0_X), min(ev["tau_hi"], xmax)
        ax.plot([lo, hi], [y, y], color=fs.LINE_DARK, lw=1.0)
        ax.plot([lo], [y], marker="|", color=fs.LINE_DARK, ms=6)
        ax.plot([hi], [y], marker=">" if ev["tau_hi"] > xmax else "|", color=fs.LINE_DARK, ms=4 if ev["tau_hi"] > xmax else 6)
        if ev["tau_s"] <= xmax:
            ax.plot([ev["tau_s"]], [y], marker="D", color=fs.LINE_DARK, ms=3.5)
        ax.set_ylim(y0, y1 + (y1 - y0) * 0.12)

        ax2 = fig.add_subplot(gs[1])
        n_max = resp["n_events"].max()
        used = resp["n_events"] >= EVENT_MIN_SHARE * n_max
        late = ~used & (resp["n_events"] >= 5)
        ax2.axhline(0, color=fs.POINT_GREY, lw=0.6)
        ax2.axvline(0, color=fs.POINT_GREY, lw=0.6)
        ax2.plot(resp.loc[used, "bin_s"], resp.loc[used, "mean_c"], color=fs.LINE_DARK, lw=0, marker="o", ms=3)
        ax2.plot(resp.loc[late, "bin_s"], resp.loc[late, "mean_c"], color=fs.POINT_GREY, lw=0, marker="o", ms=3,
                 mfc="white")
        t = np.linspace(resp.loc[used, "bin_s"].min(), resp.loc[used, "bin_s"].max(), 200)
        fit = np.where(t < 0, 0.0, ev["amp_c"] * (1 - np.exp(-np.clip(t, 0, None) / ev["tau_s"])))
        ax2.plot(t, fit, color=fs.LINE_DARK, lw=1.0)
        ax2.set_xlabel("time from the change (s)")
        ax2.set_ylabel("temperature change (°C)")

        dark = fs.LINE_DARK
        handles = [Line2D([], [], color=PERIOD_COLOURS["morning"], lw=1.5, marker="o", ms=3, label="morning"),
                   Line2D([], [], color=PERIOD_COLOURS["evening"], lw=1.5, marker="o", ms=3, label="evening"),
                   Line2D([], [], color=dark, lw=0.8, marker="o", ms=3, mfc="white",
                          label="scored on walks left out of the fit"),
                   Line2D([], [], color=dark, lw=1.0, marker="|", ms=6,
                          label="time constant from sun and shade changes"),
                   Line2D([], [], color=dark, lw=0, marker="o", ms=3, label="mean change after a change of shade"),
                   Line2D([], [], color=fs.POINT_GREY, lw=0, marker="o", ms=3, mfc="white",
                          label="fewer than half of the changes, not fitted"),
                   Line2D([], [], color=dark, lw=1.0, label="fitted exponential approach")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0),
                   fontsize=fs.FONT_PT, handlelength=1.8, columnspacing=1.5, labelspacing=0.35)
        return fs.save(fig, out_path)


def build_fig_temp_profile(package_dir: Path, out_path: Path | None = None) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    package_dir = Path(package_dir)
    seg = pd.read_csv(package_dir / "p13_temperature_pairing_segment_profile.csv")
    points = pd.read_parquet(package_dir / "OM2" / "points.parquet").drop(columns="geometry", errors="ignore")
    keep, _ = keep_mask(points.sort_values("distance_along_m").reset_index(drop=True))
    excl = points.sort_values("distance_along_m").reset_index(drop=True).assign(route_geometry_flag=~keep)
    out_path = out_path or package_dir / "OM2" / "fig_temp_profile.png"
    total = float(np.ceil(points["distance_along_m"].max() / 50) * 50)
    with fs.figure_style():
        fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, 4.6))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 1, 1], hspace=0.25, left=0.11, right=0.985, top=0.99,
                              bottom=0.22)
        axb = fig.add_subplot(gs[0])
        fs.draw_neighbourhood_band(axb, fs.neighbourhood_stretches(points), total, axes_in=fs.TEXT_WIDTH_IN * 0.875)
        axes = [fig.add_subplot(gs[i + 1], sharex=axb) for i in range(2)]
        lims = []
        for ax, per in zip(axes, ("morning", "evening")):
            c = PERIOD_COLOURS[per]
            s = seg[seg["period"] == per].sort_values("distance_m")
            ok = s["n_readings"] >= 30
            m = s.where(ok)
            ax.axhline(0, color=fs.POINT_GREY, lw=0.6, zorder=1)
            ax.fill_between(m["distance_m"], m["logger_lo"], m["logger_hi"], color=c, alpha=0.2, lw=0, zorder=2)
            ax.plot(m["distance_m"], m["anomaly_detrend"], color=DETREND_GREY, lw=0.9, zorder=3)
            ax.plot(m["distance_m"], m["anomaly_logger"], color=c, lw=1.5, zorder=4)
            w = s.where(s["n_walks_warmup"] >= 5)
            ax.plot(w["distance_m"], w["anomaly_logger_warmup"], color=c, lw=0.9, ls=(0, (1, 1.2)), zorder=4)
            ax.text(0.005, 0.96, per, transform=ax.transAxes, ha="left", va="top", fontsize=fs.FONT_PT,
                    color="black", path_effects=fs.HALO)
            ax.set_ylabel("anomaly (°C)")
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0, labelbottom=False)
            lims += [np.nanmin(np.r_[m["logger_lo"], w["anomaly_logger_warmup"]]),
                     np.nanmax(np.r_[m["logger_hi"], w["anomaly_logger_warmup"]])]
        lo, hi = np.nanmin(lims), np.nanmax(lims)
        for ax in axes:
            ax.set_ylim(np.floor(lo * 4) / 4, np.ceil(hi * 4) / 4)
        axes[-1].spines["bottom"].set_visible(True)
        axes[-1].tick_params(axis="x", length=3, labelbottom=True)
        fs.distance_axis(axes[-1], total)
        axes[-1].set_xlabel("distance along the route (m)")
        fs.shade_flagged(axes, fs.flagged_spans(excl))
        handles = [Line2D([], [], color=PERIOD_COLOURS["morning"], lw=1.5, label="morning, logger background"),
                   Line2D([], [], color=PERIOD_COLOURS["evening"], lw=1.5, label="evening, logger background"),
                   Line2D([], [], color=DETREND_GREY, lw=0.9, label="per walk time trend removed instead"),
                   Line2D([], [], color=fs.LINE_DARK, lw=0.9, ls=(0, (1, 1.2)), label="first minutes of a walk (left out)"),
                   Patch(facecolor="#ececec", edgecolor="#b5b5b5", hatch="////", lw=0, label=EXCLUDED_LABEL)]
        fig.legend(handles=handles[:4], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.54, 0.04),
                   fontsize=fs.FONT_PT, handlelength=1.8)
        fig.legend(handles=handles[4:], loc="lower center", frameon=False, bbox_to_anchor=(0.54, 0.0),
                   fontsize=fs.FONT_PT)
        return fs.save(fig, out_path)


def build_all(package_dir: Path) -> list[Path]:
    return [build_fig_temp_tau(package_dir), build_fig_temp_profile(package_dir)]
