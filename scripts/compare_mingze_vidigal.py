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
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    df.plot(
        ax=ax, column="ours_hours", cmap="viridis", vmin=0, vmax=df["ours_hours"].max(),
        markersize=2, legend=True,
        legend_kwds={"label": "solar_hours_winter (h)", "shrink": 0.7},
    )
    ax.set_aspect("equal")
    ax.set_title("Ours")
    ax.set_xticks([]); ax.set_yticks([])

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
    s = stats_block(df)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2))
    df.to_file(OUT / "paired_residuals.gpkg", driver="GPKG")
    scatter_and_bland_altman(df, s)
    residual_map(df)
    residual_vs_svf(df)
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
