#!/usr/bin/env python3
"""Figure 3: SVF-lambda_p structural coupling.

Central panel: 3 typology-grouped KDE contours (hillside, flatland, mixed)
with 50% (filled, solid) and 90% (outline, dashed) density levels.
Marginals: per-site KDE curves (5 sites individually).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from fig_style import *
from scipy import stats
from scipy.stats import gaussian_kde

# ── Typology colors (colorblind-safe, print-friendly) ──
TYPO_COLORS = {
    "Hillside": "#C75B3B",  # terracotta
    "Flatland": "#4878A8",  # steel blue
    "Mixed": "#8B8C3E",  # olive-gold
}
TYPO_COLORS_DARK = {
    "Hillside": "#9A3F25",
    "Flatland": "#30587A",
    "Mixed": "#6A6C2A",
}
TYPO_SITES = {
    "Hillside": ["vidigal", "rocinha"],
    "Flatland": ["riodaspedras", "maré"],
    "Mixed": ["complexo_do_alemao"],
}
TYPO_ORDER = ["Hillside", "Mixed", "Flatland"]


def main():
    apply_style()

    # ── Load data ──
    site_data = {}
    for site in SITE_ORDER:
        try:
            g = load_grid(site)
        except FileNotFoundError:
            continue
        valid = g[["svf", "lambda_p"]].dropna()
        site_data[site] = (valid["svf"].values, valid["lambda_p"].values)

    # ── Figure layout ──
    # Marginal ratios: ~13% of main panel
    fig = plt.figure(figsize=(WIDTH_1_5, WIDTH_1_5 * 0.95))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[5, 0.8],
        height_ratios=[0.8, 5],
        hspace=0.02,
        wspace=0.02,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    fig.add_subplot(gs[0, 1]).set_visible(False)  # corner

    # ── Structurally empty region ──
    ax_main.add_patch(
        mpatches.FancyBboxPatch(
            (0.0, 0.0),
            0.15,
            0.50,
            boxstyle="round,pad=0.008",
            facecolor="#F4D4C4",
            alpha=0.30,
            edgecolor="#B85A4A",
            linewidth=1.0,
            zorder=2,
        )
    )
    ax_main.text(
        0.075,
        0.33,
        "Structurally\nempty",
        ha="center",
        va="center",
        fontsize=8,
        color="#8B3A2A",
        fontweight="bold",
        zorder=10,
    )
    ax_main.text(
        0.075,
        0.19,
        "Low SVF requires\nhigh $\\lambda_p$",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#A05040",
        fontstyle="italic",
        zorder=10,
    )

    # ── KDE contours (3 typologies) ──
    xi = np.linspace(-0.05, 1.05, 300)
    yi = np.linspace(-0.05, 1.10, 300)
    Xi, Yi = np.meshgrid(xi, yi)
    positions = np.vstack([Xi.ravel(), Yi.ravel()])

    legend_handles = []
    label_positions = {}  # store centroid of 50% contour for label placement

    for typo in TYPO_ORDER:
        color = TYPO_COLORS[typo]
        sites_in = [s for s in TYPO_SITES[typo] if s in site_data]
        if not sites_in:
            continue

        svf_p = np.concatenate([site_data[s][0] for s in sites_in])
        lp_p = np.concatenate([site_data[s][1] for s in sites_in])

        # Subsample for KDE (keep large for smooth contours)
        n = len(svf_p)
        rng = np.random.default_rng(42)
        if n > 8000:
            idx = rng.choice(n, 8000, replace=False)
            svf_k, lp_k = svf_p[idx], lp_p[idx]
        else:
            svf_k, lp_k = svf_p, lp_p

        try:
            kde = gaussian_kde(np.vstack([svf_k, lp_k]), bw_method=0.10)
            Z = kde(positions).reshape(Xi.shape)

            # Density thresholds for probability mass levels
            d_pts = kde(np.vstack([svf_k, lp_k]))
            sorted_d = np.sort(d_pts)[::-1]
            cumsum = np.cumsum(sorted_d) / sorted_d.sum()
            level_50 = sorted_d[min(np.searchsorted(cumsum, 0.50), len(sorted_d) - 1)]
            level_90 = sorted_d[min(np.searchsorted(cumsum, 0.90), len(sorted_d) - 1)]

            # 90% contour — dashed outline only
            ax_main.contour(
                Xi,
                Yi,
                Z,
                levels=[level_90],
                colors=[color],
                linewidths=[1.0],
                linestyles=["--"],
                alpha=0.55,
                zorder=3,
            )

            # 50% contour — filled + solid outline
            ax_main.contourf(
                Xi,
                Yi,
                Z,
                levels=[level_50, Z.max()],
                colors=[color],
                alpha=0.20,
                zorder=2,
            )
            ax_main.contour(
                Xi,
                Yi,
                Z,
                levels=[level_50],
                colors=[color],
                linewidths=[2.0],
                linestyles=["-"],
                alpha=0.90,
                zorder=4,
            )

            # Find centroid of 50% region for label placement
            mask_50 = Z >= level_50
            if mask_50.any():
                label_positions[typo] = (
                    float(np.average(Xi[mask_50], weights=Z[mask_50])),
                    float(np.average(Yi[mask_50], weights=Z[mask_50])),
                )
        except Exception as e:
            print(f"  WARNING: KDE failed for {typo}: {e}")

        site_names = ", ".join(SITE_LABELS[s] for s in TYPO_SITES[typo])
        legend_handles.append(
            mpatches.Patch(
                facecolor=color,
                alpha=0.30,
                edgecolor=color,
                linewidth=1.5,
                label=f"{typo} ({site_names})",
            )
        )

    # ── Typology labels (outside/adjacent to 50% contour) ──
    # Manual positioning for clarity
    offsets = {
        "Hillside": (-0.08, -0.08),
        "Flatland": (0.06, -0.06),
        "Mixed": (0.10, 0.06),
    }
    for typo in TYPO_ORDER:
        if typo not in label_positions:
            continue
        cx, cy = label_positions[typo]
        dx, dy = offsets.get(typo, (0, 0))
        ax_main.text(
            cx + dx,
            cy + dy,
            typo.lower(),
            ha="center",
            va="center",
            fontsize=9,
            color=TYPO_COLORS_DARK[typo],
            fontstyle="italic",
            fontweight="demibold",
            zorder=8,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                alpha=0.80,
                edgecolor="none",
            ),
        )

    # ── Stratum boundary lines (very subtle) ──
    for t in SVF_THRESHOLDS:
        ax_main.axvline(t, color="#CCCCCC", linewidth=0.5, zorder=1)
    ax_main.axhline(LAMBDA_P_THRESHOLD, color="#CCCCCC", linewidth=0.5, zorder=1)

    # ── Correlation annotation ──
    svf_all = np.concatenate([d[0] for d in site_data.values()])
    lp_all = np.concatenate([d[1] for d in site_data.values()])
    valid = ~(np.isnan(svf_all) | np.isnan(lp_all))
    r_val, _ = stats.pearsonr(svf_all[valid], lp_all[valid])
    n_total = valid.sum()
    ax_main.text(
        0.97,
        0.97,
        f"$r$ = {r_val:.2f}, $n$ = {n_total:,d}",
        transform=ax_main.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    # ── Legend (compact, upper area near annotation) ──
    ax_main.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.90),
        frameon=True,
        framealpha=0.90,
        edgecolor="#dddddd",
        fontsize=6,
        handletextpad=0.4,
        handlelength=1.2,
        borderpad=0.4,
        labelspacing=0.3,
    )

    # ── Marginal distributions (per-site, 5 individual curves) ──
    x_pts = np.linspace(-0.02, 1.02, 400)
    y_pts = np.linspace(-0.02, 1.05, 400)

    for site in SITE_ORDER:
        if site not in site_data:
            continue
        svf, lp = site_data[site]
        color = SITE_COLORS[site]

        # Top marginal (SVF) — thin line + very light fill
        try:
            kx = gaussian_kde(svf[~np.isnan(svf)], bw_method=0.06)
            vals = kx(x_pts)
            ax_top.fill_between(x_pts, vals, color=color, alpha=0.08, linewidth=0)
            ax_top.plot(x_pts, vals, color=color, linewidth=0.8, alpha=0.70)
        except Exception:
            pass

        # Right marginal (λp) — outline only to reduce clutter
        try:
            ky = gaussian_kde(lp[~np.isnan(lp)], bw_method=0.06)
            vals = ky(y_pts)
            ax_right.plot(vals, y_pts, color=color, linewidth=0.8, alpha=0.70)
        except Exception:
            pass

    # ── Axis formatting ──
    ax_main.set_xlabel(r"$SVF$", fontsize=7)
    ax_main.set_ylabel(r"$\lambda_p$", fontsize=7)
    ax_main.set_xlim(-0.02, 1.02)
    ax_main.set_ylim(-0.02, 1.05)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["left"].set_linewidth(0.5)
    ax_main.spines["bottom"].set_linewidth(0.5)

    # Clean marginals — no labels, no ticks, minimal spines
    ax_top.set_yticks([])
    for sp in ["left", "top", "right"]:
        ax_top.spines[sp].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)

    ax_right.set_xticks([])
    for sp in ["bottom", "top", "right"]:
        ax_right.spines[sp].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(labelleft=False, left=False)

    # Thin separator lines between marginals and main panel
    ax_top.axhline(0, color="#dddddd", linewidth=0.3)
    ax_right.axvline(0, color="#dddddd", linewidth=0.3)

    save_fig(fig, "fig03_svf_lambda_coupling")


if __name__ == "__main__":
    main()
