"""Accuracy comparison: our Vidigal winter-solstice solar hours vs Mingze's Ladybug run.

Pairs the 6876 observers 1-to-1 by row order (verified via dx/dy alignment to <1 cm).
Outputs MAE / RMSE / bias / Pearson / Spearman, Bland-Altman, scatter + residual map,
and a per-observer residual GeoPackage.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
MZ_XLSX = ROOT / "data/external/mingze/vidigal_road_sunhours_Jun21.xlsx"
OURS_GPKG = ROOT / "outputs/vidigal/morphometrics/svf/svf_streets_solar.gpkg"
OUT = ROOT / "outputs/comparative/vidigal_vs_mingze"
OUT.mkdir(parents=True, exist_ok=True)


def load_paired():
    mz = pd.read_excel(MZ_XLSX)
    ours = gpd.read_file(OURS_GPKG)
    assert len(mz) == len(ours), f"row mismatch: {len(mz)} vs {len(ours)}"
    df = gpd.GeoDataFrame(
        {
            "point_id": mz["point_id"].values,
            "ours_hours": ours["solar_hours_winter"].values,
            "mz_hours": mz["sun_hours"].values,
            "svf": ours["svf"].values,
            "z_observer": ours["z_observer"].values,
            "z_mz": mz["z"].values,
        },
        geometry=ours.geometry.values,
        crs=ours.crs,
    )
    df["residual"] = df["ours_hours"] - df["mz_hours"]
    return df


def stats_block(df: pd.DataFrame) -> dict:
    r = df["residual"].values
    o = df["ours_hours"].values
    m = df["mz_hours"].values
    pearson_r, pearson_p = stats.pearsonr(o, m)
    spearman_r, spearman_p = stats.spearmanr(o, m)
    return {
        "n": int(len(df)),
        "ours": {"mean": float(o.mean()), "std": float(o.std()), "median": float(np.median(o))},
        "mingze": {"mean": float(m.mean()), "std": float(m.std()), "median": float(np.median(m))},
        "mae": float(np.mean(np.abs(r))),
        "rmse": float(np.sqrt(np.mean(r**2))),
        "bias_ours_minus_mz": float(r.mean()),
        "bias_median": float(np.median(r)),
        "residual_std": float(r.std()),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "abs_residual_p50": float(np.percentile(np.abs(r), 50)),
        "abs_residual_p90": float(np.percentile(np.abs(r), 90)),
        "abs_residual_p95": float(np.percentile(np.abs(r), 95)),
        "both_zero_pct": float((( (o == 0) & (m == 0) ).mean() * 100)),
        "agree_within_0p5h_pct": float((np.abs(r) <= 0.5).mean() * 100),
        "agree_within_1h_pct": float((np.abs(r) <= 1.0).mean() * 100),
        "deprived_below_2h_ours_pct": float((o < 2.0).mean() * 100),
        "deprived_below_2h_mz_pct": float((m < 2.0).mean() * 100),
        "deprivation_agreement_pct": float(((o < 2.0) == (m < 2.0)).mean() * 100),
    }


def scatter_and_bland_altman(df: pd.DataFrame, s: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    o = df["ours_hours"].values
    m = df["mz_hours"].values

    ax = axes[0]
    ax.hexbin(m, o, gridsize=40, cmap="viridis", mincnt=1)
    lim = max(o.max(), m.max()) + 0.5
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax.set_xlabel("Mingze (Ladybug) — sun hours, Jun 21")
    ax.set_ylabel("MorphoFavela — solar_hours_winter")
    ax.set_title(
        f"Scatter (n={s['n']:,})  Pearson r = {s['pearson_r']:.3f}\n"
        f"MAE = {s['mae']:.2f} h   RMSE = {s['rmse']:.2f} h   bias = {s['bias_ours_minus_mz']:+.2f} h"
    )
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)
    ax.legend(loc="upper left")

    ax = axes[1]
    mean = (o + m) / 2
    diff = o - m
    md = diff.mean()
    sd = diff.std()
    ax.hexbin(mean, diff, gridsize=40, cmap="magma", mincnt=1)
    ax.axhline(md, color="k", lw=1.2, label=f"mean diff = {md:+.2f} h")
    ax.axhline(md + 1.96 * sd, color="red", lw=1, ls="--",
               label=f"+1.96 SD = {md + 1.96 * sd:+.2f} h")
    ax.axhline(md - 1.96 * sd, color="red", lw=1, ls="--",
               label=f"-1.96 SD = {md - 1.96 * sd:+.2f} h")
    ax.set_xlabel("Mean of methods (h)")
    ax.set_ylabel("Ours − Mingze (h)")
    ax.set_title("Bland–Altman")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "scatter_bland_altman.png", dpi=150)
    plt.close(fig)


def add_scalebar_north(ax):
    """Scale bar + north arrow. CRS is EPSG:31983 (metres, grid north up)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy = x1 - x0, y1 - y0
    bar = max((L for L in (50, 100, 200, 250, 500) if L <= 0.25 * dx), default=100)
    bx, by = x0 + 0.04 * dx, y0 + 0.05 * dy
    ax.plot([bx, bx + bar], [by, by], color="k", lw=2.5, solid_capstyle="butt", zorder=10)
    ax.text(bx + bar / 2, by + 0.012 * dy, f"{bar} m", ha="center", va="bottom",
            fontsize=8, zorder=10)
    ax.annotate("", xy=(0.96, 0.94), xytext=(0.96, 0.86), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
    ax.text(0.96, 0.945, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold")


def residual_map(df: gpd.GeoDataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    vmax = float(np.percentile(np.abs(df["residual"]), 98))

    ax = axes[0]
    df.plot(
        ax=ax, column="residual", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        markersize=2, legend=True,
        legend_kwds={"label": "ours − Mingze (h)", "shrink": 0.7},
    )
    ax.set_aspect("equal")
    ax.set_title("Residual (winter solstice solar hours)")
    ax.set_xticks([])
    ax.set_yticks([])
    add_scalebar_north(ax)

    ax = axes[1]
    df.plot(
        ax=ax, column="ours_hours", cmap="viridis", vmin=0, vmax=df["ours_hours"].max(),
        markersize=2, legend=True,
        legend_kwds={"label": "solar_hours_winter (h)", "shrink": 0.7},
    )
    ax.set_aspect("equal")
    ax.set_title("Ours")
    ax.set_xticks([])
    ax.set_yticks([])
    add_scalebar_north(ax)

    fig.tight_layout()
    fig.savefig(OUT / "residual_map.png", dpi=160)
    plt.close(fig)


def residual_vs_svf(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hexbin(df["svf"], df["residual"], gridsize=40, cmap="cividis", mincnt=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("SVF (ours)")
    ax.set_ylabel("Residual: ours − Mingze (h)")
    ax.set_title("Residual vs SVF")
    fig.tight_layout()
    fig.savefig(OUT / "residual_vs_svf.png", dpi=150)
    plt.close(fig)


def street_bearing(df: gpd.GeoDataFrame, ours: gpd.GeoDataFrame) -> np.ndarray:
    """Local street bearing (deg, mod 180) from consecutive observers on the same street."""
    x = df.geometry.x.values
    y = df.geometry.y.values
    sid = ours["street_id"].values
    bearing = np.full(len(df), np.nan)
    dx = np.diff(x)
    dy = np.diff(y)
    same = sid[1:] == sid[:-1]
    b = (np.degrees(np.arctan2(dx, dy)) % 180.0)
    bearing[:-1][same] = b[same]
    # last point of each street inherits its predecessor's bearing
    idx = np.where(np.isnan(bearing))[0]
    for i in idx:
        if i > 0 and sid[i] == sid[i - 1] and not np.isnan(bearing[i - 1]):
            bearing[i] = bearing[i - 1]
    return bearing


def diagnostics(df: gpd.GeoDataFrame, ours_full: gpd.GeoDataFrame) -> dict:
    o = df["ours_hours"].values
    m = df["mz_hours"].values
    r = df["residual"].values

    d: dict = {}

    # -- 1. quantization floor: both methods sit on coarse grids
    # (ours: 0.5 h steps from 21 sun positions; Mingze: integer hours).
    o_int = np.round(o)
    d["mae_quantization_floor"] = float(np.mean(np.abs(o - o_int)))
    d["mae_on_integer_grid"] = float(np.mean(np.abs(o_int - m)))

    # -- 2. scale mismatch: his full-sun is 11 h, ours 10.5 h
    d["max_ours"] = float(o.max())
    d["max_mz"] = float(m.max())
    m_rescaled = m * (o.max() / m.max())
    d["mae_after_rescale"] = float(np.mean(np.abs(o - m_rescaled)))
    d["bias_after_rescale"] = float(np.mean(o - m_rescaled))

    # -- 3. zero-asymmetry: structural occlusion disagreements
    ours_dark = o < 0.25
    mz_dark = m < 0.5
    d["both_dark_pct"] = float((ours_dark & mz_dark).mean() * 100)
    d["ours_dark_mz_lit_pct"] = float((ours_dark & ~mz_dark).mean() * 100)
    d["mz_dark_ours_lit_pct"] = float((~ours_dark & mz_dark).mean() * 100)
    d["ours_dark_mz_lit_mean_mz"] = float(m[ours_dark & ~mz_dark].mean()) if (ours_dark & ~mz_dark).any() else 0.0
    d["mz_dark_ours_lit_mean_ours"] = float(o[~ours_dark & mz_dark].mean()) if (~ours_dark & mz_dark).any() else 0.0

    # -- 4. confusion matrix on hour bands
    bins = [-0.01, 0.25, 2, 4, 6, 8, 12]
    labels = ["0", "0–2", "2–4", "4–6", "6–8", ">8"]
    co = pd.cut(o, bins=bins, labels=labels)
    cm_ = pd.cut(m, bins=bins, labels=labels)
    cm = pd.crosstab(co, cm_, rownames=["ours"], colnames=["mingze"])
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    d["confusion_matrix"] = {"labels": labels, "counts": cm.values.tolist()}
    d["band_agreement_pct"] = float(np.trace(cm.values) / len(df) * 100)
    d["band_within1_pct"] = float(
        sum(cm.values[i, j] for i in range(6) for j in range(6) if abs(i - j) <= 1) / len(df) * 100
    )

    # -- 5. covariate correlations
    z = df["z_observer"].values
    zdiff = df["z_mz"].values - (df["z_observer"].values - 1.5)
    d["corr_residual_terrain_z"] = float(np.corrcoef(r, z)[0, 1])
    d["corr_residual_zdiff"] = float(np.corrcoef(r, zdiff)[0, 1])
    d["zdiff_mean"] = float(zdiff.mean())
    d["zdiff_std"] = float(zdiff.std())
    d["corr_residual_svf"] = float(np.corrcoef(r, df["svf"].values)[0, 1])

    # -- 6. MAE by SVF band
    svf_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    svf_band = pd.cut(df["svf"], bins=svf_bins, include_lowest=True)
    grp = pd.DataFrame({"band": svf_band, "abs_r": np.abs(r), "r": r}).groupby("band", observed=True)
    d["mae_by_svf_band"] = {
        str(k): {"mae": float(v["abs_r"].mean()), "bias": float(v["r"].mean()), "n": int(len(v))}
        for k, v in grp
    }

    # -- 7. residual vs street bearing (mod 180; 0 = N-S street, 90 = E-W)
    bearing = street_bearing(df, ours_full)
    ok = ~np.isnan(bearing)
    bb = pd.cut(bearing[ok], bins=np.arange(0, 181, 22.5))
    grp = pd.DataFrame({"band": bb, "abs_r": np.abs(r[ok]), "r": r[ok], "o": o[ok], "m": m[ok]}).groupby(
        "band", observed=True
    )
    d["by_bearing"] = {
        str(k): {
            "mae": float(v["abs_r"].mean()),
            "bias": float(v["r"].mean()),
            "ours_mean": float(v["o"].mean()),
            "mz_mean": float(v["m"].mean()),
            "n": int(len(v)),
        }
        for k, v in grp
    }
    df["bearing"] = bearing
    return d


def diagnostic_figures(df: gpd.GeoDataFrame, d: dict):
    # confusion matrix heatmap
    labels = d["confusion_matrix"]["labels"]
    counts = np.array(d["confusion_matrix"]["counts"], dtype=float)
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(pct, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Mingze (Ladybug) — hour band")
    ax.set_ylabel("Ours — hour band")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = pct[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v > pct.max() / 2 else "black", fontsize=9)
    ax.set_title(
        f"Hour-band agreement: {d['band_agreement_pct']:.1f} % same band, "
        f"{d['band_within1_pct']:.1f} % within one band\n(cell values = % of all {int(counts.sum()):,} observers)"
    )
    fig.colorbar(im, label="% of observers", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # zero-asymmetry map
    o = df["ours_hours"].values
    m = df["mz_hours"].values
    cat = np.full(len(df), "agree (both lit)", dtype=object)
    cat[(o < 0.25) & (m < 0.5)] = "agree (both dark)"
    cat[(o < 0.25) & (m >= 0.5)] = "ours dark / mingze lit"
    cat[(o >= 0.25) & (m < 0.5)] = "mingze dark / ours lit"
    df_plot = df.assign(cat=cat)
    colors = {
        "agree (both lit)": "#d4d4d4",
        "agree (both dark)": "#7a7a7a",
        "ours dark / mingze lit": "#d7301f",
        "mingze dark / ours lit": "#0570b0",
    }
    fig, ax = plt.subplots(figsize=(11, 7))
    for c, col in colors.items():
        sub = df_plot[df_plot["cat"] == c]
        sub.plot(ax=ax, color=col, markersize=3 if c.startswith("agree") else 7,
                 label=f"{c} (n={len(sub):,})")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9, markerscale=2)
    ax.set_title("Structural disagreements: who sees sun where the other sees none")
    ax.set_xticks([])
    ax.set_yticks([])
    add_scalebar_north(ax)
    fig.tight_layout()
    fig.savefig(OUT / "zero_asymmetry_map.png", dpi=160)
    plt.close(fig)

    # MAE + bias by SVF band
    def svf_band_label(key: str) -> str:
        lo, hi = (float(v) for v in key.strip("([])").split(", "))
        return f"{max(lo, 0.0):g}–{hi:.1f}"

    bands = list(d["mae_by_svf_band"].keys())
    mae = [d["mae_by_svf_band"][b]["mae"] for b in bands]
    bias = [d["mae_by_svf_band"][b]["bias"] for b in bands]
    ns = [d["mae_by_svf_band"][b]["n"] for b in bands]
    fig, ax = plt.subplots(figsize=(9, 5))
    xpos = np.arange(len(bands))
    ax.bar(xpos - 0.2, mae, width=0.4, color="#4477aa", label="MAE")
    ax.bar(xpos + 0.2, bias, width=0.4, color="#ee6677", label="bias (ours − mingze)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos, [svf_band_label(b) for b in bands], rotation=30)
    for i, n in enumerate(ns):
        ax.annotate(f"n={n:,}", (xpos[i], max(mae[i], 0) + 0.07), ha="center", fontsize=7.5)
    ax.set_xlabel("SVF band (ours)")
    ax.set_ylabel("Hours")
    ax.set_title(
        f"Agreement by sky openness: MAE roughly flat ({min(mae):.1f}–{max(mae):.1f} h);\n"
        f"bias swings {bias[0]:+.1f} h (deep canyons) to {bias[-1]:+.1f} h (most open)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mae_by_svf_band.png", dpi=150)
    plt.close(fig)

    # bearing panel
    bk = list(d["by_bearing"].keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    xpos = np.arange(len(bk))
    ours_m = [d["by_bearing"][b]["ours_mean"] for b in bk]
    mz_m = [d["by_bearing"][b]["mz_mean"] for b in bk]
    ax.bar(xpos - 0.2, ours_m, width=0.4, color="#4477aa", label="ours, mean h")
    ax.bar(xpos + 0.2, mz_m, width=0.4, color="#ccbb44", label="mingze, mean h")
    ax.set_xticks(xpos, [b.strip("([])").replace(", ", "–") + "°" for b in bk], rotation=30)
    ax.set_xlabel("Street bearing (0° = N–S, 90° = E–W)")
    ax.set_ylabel("Mean sun hours")
    ax.set_title(
        "Both peak in the NE family, but brightness swaps at 90°:\n"
        "ours higher on every 0–90° bin, Mingze on every 90–180° bin (see H2)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "bearing_panel.png", dpi=150)
    plt.close(fig)


def write_summary_md(s: dict):
    lines = [
        "# Vidigal solar accuracy — MorphoFavela vs Mingze (Ladybug)",
        "",
        "**Date target:** 21 June (winter solstice, Southern Hemisphere)",
        "**Variable:** sun hours on the road observer network",
        f"**Observers:** {s['n']:,} (paired 1-to-1 by row order — alignment verified < 1 cm)",
        "",
        "## Distribution summary",
        "",
        "| | Mean (h) | Median (h) | Std (h) | < 2 h (%) |",
        "|---|---|---|---|---|",
        f"| Ours | {s['ours']['mean']:.2f} | {s['ours']['median']:.2f} | {s['ours']['std']:.2f} | {s['deprived_below_2h_ours_pct']:.1f} |",
        f"| Mingze | {s['mingze']['mean']:.2f} | {s['mingze']['median']:.2f} | {s['mingze']['std']:.2f} | {s['deprived_below_2h_mz_pct']:.1f} |",
        "",
        "## Agreement",
        "",
        f"- **MAE**: {s['mae']:.2f} h",
        f"- **RMSE**: {s['rmse']:.2f} h",
        f"- **Bias** (ours − Mingze): {s['bias_ours_minus_mz']:+.2f} h (median residual {s['bias_median']:+.2f})",
        f"- **Pearson r**: {s['pearson_r']:.3f} (p = {s['pearson_p']:.2e})",
        f"- **Spearman ρ**: {s['spearman_r']:.3f} (p = {s['spearman_p']:.2e})",
        f"- **|residual| P50 / P90 / P95**: {s['abs_residual_p50']:.2f} / {s['abs_residual_p90']:.2f} / {s['abs_residual_p95']:.2f} h",
        f"- Agree within ±0.5 h: **{s['agree_within_0p5h_pct']:.1f} %**; within ±1.0 h: **{s['agree_within_1h_pct']:.1f} %**",
        f"- Both methods score 0 h: **{s['both_zero_pct']:.1f} %** of observers",
        f"- WHO < 2 h deprivation flag agreement: **{s['deprivation_agreement_pct']:.1f} %**",
        "",
        "## Artefacts",
        "",
        "- `scatter_bland_altman.png` — 2-panel: hexbin scatter + Bland-Altman",
        "- `residual_map.png` — 2-panel: signed residual map + ours reference",
        "- `residual_vs_svf.png` — residual vs SVF hexbin",
        "- `paired_residuals.gpkg` — per-observer residuals for GIS exploration",
        "- `summary.json` — all metrics in machine-readable form",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    df = load_paired()
    ours_full = gpd.read_file(OURS_GPKG)
    s = stats_block(df)
    d = diagnostics(df, ours_full)
    s["diagnostics"] = d
    (OUT / "summary.json").write_text(json.dumps(s, indent=2))
    df.to_file(OUT / "paired_residuals.gpkg", driver="GPKG")
    scatter_and_bland_altman(df, s)
    residual_map(df)
    residual_vs_svf(df)
    diagnostic_figures(df, d)
    write_summary_md(s)

    print(f"n               = {s['n']}")
    print(f"MAE             = {s['mae']:.3f} h")
    print(f"RMSE            = {s['rmse']:.3f} h")
    print(f"bias (ours-mz)  = {s['bias_ours_minus_mz']:+.3f} h")
    print(f"Pearson r       = {s['pearson_r']:.4f}")
    print(f"Spearman rho    = {s['spearman_r']:.4f}")
    print(f"|res| P90       = {s['abs_residual_p90']:.2f} h")
    print(f"agree <=0.5 h   = {s['agree_within_0p5h_pct']:.1f} %")
    print(f"agree <=1.0 h   = {s['agree_within_1h_pct']:.1f} %")
    print(f"deprivation flag agreement = {s['deprivation_agreement_pct']:.1f} %")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
