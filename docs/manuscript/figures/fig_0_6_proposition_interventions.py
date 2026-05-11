#!/usr/bin/env python3
"""Figure 0.6 (proposition A) — From diagnosis to morphological lever.

Three counterfactual morphological interventions are applied at the
cell level and the resulting flip rate is computed from a logistic
classifier trained on the same predictors as Fig 0.5. The figure
asks: *if a planner could move one feature, how much of the failing
fabric would the diagnosis re-classify as adequate?*

Levers (all chosen from Fig 0.5 predictors — no new feature engineering):
  L1  SVF +0.10        roof-level setback / top-floor removal
  L2  λp −0.10         opening building gaps / courtyard insertion
  L3  σH ×0.5          height homogenization (extreme outliers clipped)

Panels
------
A   Recovery bars. 3 levers × 3 typologies = 9 bars. Height =
    % of cells in baseline-failure that flip to adequate under the lever.
B   Before / after 4-state maps for each site under the **best lever
    per typology** (most recoveries). 5 rows × 2 columns.
C   Cost-effectiveness frontier: x = "intervention magnitude" proxy
    (the feature delta), y = % recovery. One marker per (lever ×
    typology). The upper-left Pareto frontier is highlighted.

Run:
    python docs/manuscript/figures/fig_0_6_proposition_interventions.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
sys.path.insert(0, str(PROJECT_ROOT / "docs" / "manuscript" / "figures"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fig_style import (  # noqa: E402
    SITE_LABELS,
    SITE_ORDER,
    WIDTH_DOUBLE,
    add_north_arrow,
    add_scalebar,
    apply_style,
    clean_map_axes,
    load_boundary,
    load_buildings,
)
from fig_0_4_diagnostic import (  # noqa: E402
    BUILDING_COLOR,
    NOT_ASSESSED_COLOR,
    STATE_COLORS,
    STATE_KEYS,
    STATE_LABELS,
    THRESHOLD_SUN_HOURS,
    THRESHOLD_U_VENT,
    TYPOLOGY_AGGREGATE_ORDER,
    TYPOLOGY_OF,
)
from run_diagnostic_models import (  # noqa: E402
    PREDICTORS,
    TARGETS,
    build_cell_table,
)

EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

LEVERS = [
    {
        "key": "svf_plus_0p10",
        "label": "SVF +0.10",
        "narrative": "roof setback",
        "feature": "svf",
        "delta": +0.10,
        "magnitude": 0.10,
    },
    {
        "key": "lambda_p_minus_0p10",
        "label": "λp −0.10",
        "narrative": "open courtyards",
        "feature": "lambda_p",
        "delta": -0.10,
        "magnitude": 0.10,
    },
    {
        "key": "sigma_h_half",
        "label": "σH ×0.5",
        "narrative": "height homogenize",
        "feature": "sigma_h",
        "delta_mode": "scale",
        "scale": 0.5,
        "magnitude": 0.5,
    },
]

LEVER_COLORS = {
    "SVF +0.10": "#0072B2",   # blue
    "λp −0.10": "#009E73",    # green
    "σH ×0.5": "#CC79A7",     # purple
}


def fit_logits(df: pd.DataFrame) -> tuple[StandardScaler, dict[str, LogisticRegression]]:
    sub = df.dropna(subset=PREDICTORS + TARGETS).copy()
    scaler = StandardScaler().fit(sub[PREDICTORS])
    Xz = scaler.transform(sub[PREDICTORS])
    models = {}
    for tgt in TARGETS:
        y = sub[tgt].to_numpy()
        clf = LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000,
            class_weight="balanced", random_state=0,
        ).fit(Xz, y)
        models[tgt] = clf
        score = clf.score(Xz, y)
        print(f"  logit {tgt}: accuracy={score:.3f}  beta={clf.coef_[0].round(3).tolist()}")
    return scaler, models


def classify_from_probs(p_vent: np.ndarray, p_sun: np.ndarray) -> np.ndarray:
    """Threshold the predicted failure probabilities at 0.5 to assign state."""
    vent_fail = p_vent >= 0.5
    sun_fail = p_sun >= 0.5
    out = np.empty(len(p_vent), dtype=object)
    out[~vent_fail & ~sun_fail] = "adequate"
    out[~vent_fail & sun_fail] = "sun"
    out[vent_fail & ~sun_fail] = "vent"
    out[vent_fail & sun_fail] = "compound"
    return out


def apply_lever(df: pd.DataFrame, lever: dict) -> pd.DataFrame:
    out = df.copy()
    f = lever["feature"]
    if "delta" in lever:
        out[f] = out[f] + lever["delta"]
    elif lever.get("delta_mode") == "scale":
        # Move toward the median by scaling deviations from median by the scale factor.
        med = float(out[f].median())
        out[f] = med + (out[f] - med) * float(lever["scale"])
    return out


def compute_recovery_per_typology(df: pd.DataFrame, baseline_state: np.ndarray,
                                   scaler: StandardScaler,
                                   models: dict[str, LogisticRegression]
                                   ) -> dict[str, dict[str, dict]]:
    """For each (typology, lever), compute:
       - n_failing_before
       - n_recovered (was failing, now adequate)
       - recovery_pct = 100 * n_recovered / n_failing_before
       - n_new_failure (was adequate, now failing) — sanity check
    Returns {typology: {lever_label: {...}}}
    """
    result: dict[str, dict[str, dict]] = {}
    for typ in TYPOLOGY_AGGREGATE_ORDER:
        members = [s for s in SITE_ORDER if TYPOLOGY_OF[s] == typ]
        mask = df["site"].isin(members)
        if not mask.any():
            continue
        result[typ] = {}
        sub_idx = np.where(mask)[0]
        was_failing = baseline_state[sub_idx] != "adequate"
        was_adequate = baseline_state[sub_idx] == "adequate"
        n_fail = int(was_failing.sum())
        for lever in LEVERS:
            df_pert = apply_lever(df, lever)
            Xz_pert = scaler.transform(df_pert[PREDICTORS])
            pv = models["vent_fail"].predict_proba(Xz_pert)[:, 1]
            ps = models["sun_fail"].predict_proba(Xz_pert)[:, 1]
            new_state = classify_from_probs(pv, ps)
            new_state_sub = new_state[sub_idx]
            now_adequate = new_state_sub == "adequate"
            n_recovered = int((was_failing & now_adequate).sum())
            n_new_failure = int((was_adequate & ~now_adequate).sum())
            result[typ][lever["label"]] = {
                "n_failing_before": n_fail,
                "n_recovered": n_recovered,
                "recovery_pct": 100.0 * n_recovered / max(n_fail, 1),
                "n_new_failure": n_new_failure,
                "magnitude": lever["magnitude"],
            }
    return result


# ---------------------------------------------------------------------------
# Panel A: bar plot
# ---------------------------------------------------------------------------
def draw_panel_a(ax, recovery: dict) -> None:
    typs = [t for t in TYPOLOGY_AGGREGATE_ORDER if t in recovery]
    lever_labels = [lvr["label"] for lvr in LEVERS]
    n_typs = len(typs)
    n_levers = len(lever_labels)
    width = 0.26
    x = np.arange(n_typs)

    for j, lab in enumerate(lever_labels):
        vals = [recovery[t][lab]["recovery_pct"] for t in typs]
        new_fails = [recovery[t][lab]["n_new_failure"]
                     / max(sum(recovery[t][lab]["n_failing_before"]
                               for t in typs), 1) * 100 for t in typs]
        offset = (j - (n_levers - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width,
                      color=LEVER_COLORS[lab], edgecolor="white",
                      linewidth=0.5, zorder=3, label=lab)
        for k, v in enumerate(vals):
            ax.text(x[k] + offset, v + 0.6, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=5.5,
                    color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(typs, fontsize=7, fontweight="bold")
    ax.set_ylabel("Failure cells flipped to adequate (%)", fontsize=7,
                  labelpad=2)
    ax.tick_params(axis="y", labelsize=6, length=2, width=0.4, pad=2)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#eaeaea", linewidth=0.4, zorder=0)
    leg = ax.legend(loc="upper right", fontsize=6.5, frameon=False,
                    handlelength=1.2, labelspacing=0.4, ncol=1,
                    title="Lever (Δ feature)", title_fontsize=6.5,
                    borderpad=0.0)
    leg.get_title().set_fontweight("bold")


# ---------------------------------------------------------------------------
# Panel B: before / after maps per site
# ---------------------------------------------------------------------------
def best_lever_per_typology(recovery: dict) -> dict[str, str]:
    out = {}
    for typ, by_lever in recovery.items():
        out[typ] = max(by_lever.keys(),
                       key=lambda k: by_lever[k]["recovery_pct"])
    return out


def draw_site_map(ax, site: str, grid: gpd.GeoDataFrame,
                  buildings: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame,
                  state_col: str, scalebar: bool, title_suffix: str) -> None:
    minx, miny, maxx, maxy = boundary.total_bounds
    pad = 0.02 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    buildings.plot(ax=ax, facecolor=BUILDING_COLOR, edgecolor="none",
                   linewidth=0.0, zorder=1)
    unc = grid[grid[state_col].isna()]
    if len(unc):
        unc.plot(ax=ax, facecolor=NOT_ASSESSED_COLOR, edgecolor="none",
                 alpha=0.5, zorder=2)
    for k in STATE_KEYS:
        sub = grid[grid[state_col] == k]
        if len(sub):
            sub.plot(ax=ax, facecolor=STATE_COLORS[k], edgecolor="none",
                     alpha=0.92, zorder=3)
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.4,
                           linestyle=(0, (3, 2)), zorder=4, alpha=0.65)
    clean_map_axes(ax)
    if scalebar:
        ew = maxx - minx
        bar = 100 if ew < 1500 else (200 if ew < 2500 else 500)
        add_scalebar(ax, length_m=bar, loc="lower left")
        add_north_arrow(ax, loc="upper right", size=0.05)
    ax.set_title(f"{SITE_LABELS[site]}{title_suffix}",
                 fontsize=7, pad=3, fontweight="bold", color="#222")


# ---------------------------------------------------------------------------
# Panel C: cost-effectiveness frontier
# ---------------------------------------------------------------------------
def draw_panel_c(ax, recovery: dict) -> None:
    pts = []
    for typ in recovery:
        for lab, info in recovery[typ].items():
            pts.append({
                "typology": typ, "lever": lab,
                "magnitude": info["magnitude"],
                "recovery_pct": info["recovery_pct"],
            })
    pts_df = pd.DataFrame(pts)
    # Marker shape by typology, color by lever.
    typ_marker = {"Hillside": "o", "Mixed": "s", "Flatland": "^"}
    for _, r in pts_df.iterrows():
        ax.scatter(r["magnitude"], r["recovery_pct"],
                   marker=typ_marker[r["typology"]],
                   color=LEVER_COLORS[r["lever"]],
                   s=46, edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(r["magnitude"] + 0.005, r["recovery_pct"] + 0.4,
                f"{r['typology'][:1]}", fontsize=5.5, color="#444",
                fontweight="bold")
    # Pareto front (upper-left).
    sorted_pts = pts_df.sort_values("magnitude").reset_index(drop=True)
    frontier = []
    best = -np.inf
    for _, r in sorted_pts.iterrows():
        if r["recovery_pct"] > best:
            frontier.append((r["magnitude"], r["recovery_pct"]))
            best = r["recovery_pct"]
    if frontier:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, color="#444", linestyle=(0, (3, 2)),
                linewidth=0.6, zorder=2, alpha=0.6, label="Pareto frontier")
    ax.set_xlabel("Intervention magnitude (feature units)", fontsize=7, labelpad=2)
    ax.set_ylabel("Recovery (%)", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6, length=2, width=0.4, pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)
    ax.set_axisbelow(True)
    ax.grid(color="#eaeaea", linewidth=0.4, zorder=0)
    # Marker legend (typology).
    handles = [
        plt.scatter([], [], marker=m, color="#888", s=46,
                    edgecolor="white", linewidth=0.6, label=t)
        for t, m in typ_marker.items()
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6,
                    frameon=False, handlelength=1.2, labelspacing=0.4,
                    title="Typology", title_fontsize=6, borderpad=0.0)
    leg.get_title().set_fontweight("bold")


def main() -> None:
    apply_style()
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("Building cell-level table ...")
    df_full = build_cell_table()
    df = df_full.dropna(subset=PREDICTORS + TARGETS).copy().reset_index(drop=True)
    print(f"  pooled classifiable: {len(df)}")

    print("Fitting in-script logistic models ...")
    scaler, models = fit_logits(df)
    Xz_base = scaler.transform(df[PREDICTORS])
    p_vent_base = models["vent_fail"].predict_proba(Xz_base)[:, 1]
    p_sun_base = models["sun_fail"].predict_proba(Xz_base)[:, 1]
    state_base = classify_from_probs(p_vent_base, p_sun_base)

    print("Computing recovery per typology × lever ...")
    recovery = compute_recovery_per_typology(df, state_base, scaler, models)
    for typ, by_lever in recovery.items():
        print(f"  {typ}:")
        for lab, info in by_lever.items():
            print(f"    {lab:>14}  recovered {info['n_recovered']:4} / "
                  f"{info['n_failing_before']:4} failing "
                  f"({info['recovery_pct']:5.1f}%)   "
                  f"new failures: {info['n_new_failure']}")

    # Pick best lever per typology and apply for the maps.
    best = best_lever_per_typology(recovery)
    print(f"Best lever per typology: {best}")

    # Pre-compute per-site after-best-lever state for visualization.
    site_grids: dict[str, gpd.GeoDataFrame] = {}
    for site in SITE_ORDER:
        gmpath = (PROJECT_ROOT / "outputs" / site / "morphometrics"
                  / "grid" / "grid_metrics.gpkg")
        g = gpd.read_file(gmpath)[["zone_id", "geometry"]]
        g_sub = df[df["site"] == site].set_index("zone_id")
        g_sub["state_baseline"] = state_base[df["site"].to_numpy() == site]
        # Apply the typology's best lever to this site's rows.
        typ = TYPOLOGY_OF[site]
        chosen_lever = next(l for l in LEVERS if l["label"] == best[typ])
        df_pert = apply_lever(df, chosen_lever)
        Xz_pert = scaler.transform(df_pert[PREDICTORS])
        pv = models["vent_fail"].predict_proba(Xz_pert)[:, 1]
        ps = models["sun_fail"].predict_proba(Xz_pert)[:, 1]
        state_after = classify_from_probs(pv, ps)
        g_sub["state_after"] = state_after[df["site"].to_numpy() == site]
        g_sub = g_sub.reset_index()
        merged = g.merge(g_sub[["zone_id", "state_baseline", "state_after"]],
                         on="zone_id", how="left")
        site_grids[site] = merged

    boundaries = {s: load_boundary(s) for s in SITE_ORDER}
    buildings = {s: load_buildings(s, extended=False) for s in SITE_ORDER}

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.20))
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.4, 3.0, 1.2],
        hspace=0.30,
        left=0.05, right=0.96, top=0.93, bottom=0.06,
    )

    # Row 0: panel A (bar plot, left, wider), legend / caption (right).
    a_gs = outer[0].subgridspec(1, 2, width_ratios=[0.62, 0.38], wspace=0.10)
    ax_a = fig.add_subplot(a_gs[0, 0])
    draw_panel_a(ax_a, recovery)

    ax_legend = fig.add_subplot(a_gs[0, 1])
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    for sp in ax_legend.spines.values():
        sp.set_visible(False)
    handles_state = [Patch(facecolor=STATE_COLORS[k], edgecolor="none",
                           label=STATE_LABELS[k]) for k in STATE_KEYS]
    leg = ax_legend.legend(handles=handles_state, loc="upper left",
                           bbox_to_anchor=(0.02, 0.98),
                           frameon=False, fontsize=6.5,
                           handlelength=1.4, labelspacing=0.45,
                           title="State (panels B)",
                           title_fontsize=6.5, borderpad=0.0)
    leg.get_title().set_fontweight("bold")
    caption = (
        "Recovery is computed from a logistic classifier with the same\n"
        "8 predictors as Fig 0.5. Each lever perturbs a single feature;\n"
        "P(vent_fail) and P(sun_fail) are recomputed and the cell is\n"
        "thresholded at 0.5. Recovery = % of baseline-failure cells that\n"
        "flip to 'adequate'. New failures (adequate → fail) are tracked\n"
        "as a sanity check but are typically near zero."
    )
    ax_legend.text(0.02, 0.50, caption, transform=ax_legend.transAxes,
                   ha="left", va="top", fontsize=5.8, color="#333",
                   linespacing=1.4)

    # Row 1: before / after maps.
    n_sites = len(SITE_ORDER)
    maps_gs = outer[1].subgridspec(n_sites, 2, wspace=0.04, hspace=0.18)
    for i, site in enumerate(SITE_ORDER):
        g = site_grids[site]
        typ = TYPOLOGY_OF[site]
        chosen = best[typ]
        ax_before = fig.add_subplot(maps_gs[i, 0])
        ax_after = fig.add_subplot(maps_gs[i, 1])
        draw_site_map(ax_before, site, g, buildings[site], boundaries[site],
                      state_col="state_baseline", scalebar=(i == 0),
                      title_suffix="  (baseline)")
        draw_site_map(ax_after, site, g, buildings[site], boundaries[site],
                      state_col="state_after", scalebar=False,
                      title_suffix=f"  (after {chosen})")

    # Row 2: panel C frontier (left), caption (right).
    c_gs = outer[2].subgridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.10)
    ax_c = fig.add_subplot(c_gs[0, 0])
    draw_panel_c(ax_c, recovery)

    ax_cap = fig.add_subplot(c_gs[0, 1])
    ax_cap.set_xticks([])
    ax_cap.set_yticks([])
    for sp in ax_cap.spines.values():
        sp.set_visible(False)
    cap2 = (
        "Cost-effectiveness frontier. Magnitude is the feature delta\n"
        "(absolute units of svf, λp, σH). Markers above the dashed\n"
        "Pareto line dominate alternatives at a smaller intervention.\n"
        "For a planner with a fixed mass-removal budget this is the\n"
        "intuition for which lever to spend first.\n\n"
        "Note: rectangular σH ×0.5 dollar amounts are not strictly\n"
        "comparable to ±0.10 SVF / λp — magnitudes are intra-feature."
    )
    ax_cap.text(0.0, 0.95, cap2, transform=ax_cap.transAxes,
                ha="left", va="top", fontsize=5.8, color="#333",
                linespacing=1.4)

    # Panel labels.
    pos_a = ax_a.get_position()
    fig.text(pos_a.x0 - 0.025, pos_a.y1 + 0.005, "a",
             fontsize=9, fontweight="bold", va="bottom", ha="left")
    first_map_ax = fig.axes[2]  # 0=A, 1=legend, 2=first map
    pos_m = first_map_ax.get_position()
    fig.text(pos_m.x0 - 0.025, pos_m.y1 + 0.015, "b",
             fontsize=9, fontweight="bold", va="bottom", ha="left")
    pos_c = ax_c.get_position()
    fig.text(pos_c.x0 - 0.025, pos_c.y1 + 0.005, "c",
             fontsize=9, fontweight="bold", va="bottom", ha="left")

    fig.text(0.5, 0.97,
             "From diagnosis to lever: counterfactual interventions recover failing cells unevenly across typologies",
             ha="center", va="top", fontsize=8.5, fontweight="bold",
             color="#1a1a1a")
    fig.text(0.5, 0.95,
             "PROPOSITION A — framework demonstration on synthetic CFD; recovery rates re-quantified on real-campaign CFD.",
             ha="center", va="top", fontsize=5.5, style="italic", color="#a0522d")

    out_png = EXPORTS_DIR / "fig_0_6_proposition_interventions.png"
    out_svg = EXPORTS_DIR / "fig_0_6_proposition_interventions.svg"
    print(f"Saving {out_png.name} + {out_svg.name} ...")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
