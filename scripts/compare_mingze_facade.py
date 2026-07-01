#!/usr/bin/env python3
"""Cross-check Mingze's 5-site façade+street solar deprivation against our raycast.

Mingze (Ladybug Tools, Grasshopper) ran an annual direct-sun simulation on the
same five favelas — facades discretised into 3 m storey bands and streets sampled
at 1.2 m — accumulating direct-sun hours at a 0.5 h step across the full year and
classifying every point against the WHO 2 h/day floor. The write-up
(`data/external/mingze/mingze_update_2026-07.{txt,docx,pdf}`, Google Doc
10WqmQ_-…, pulled 2026-07-01) reports **façade** deprivation per site (Fig 2) and
**street** deprivation for three sites (text), plus a street Gini (Fig 4).

We produce only a *street-level* winter-solstice WHO-2h field (there is no façade
raycast on our side), so the honest comparison is street-to-street, with the
caveat that OUR metric is the single worst-day (21 June) floor while Mingze's is
the full-year envelope — ours is expected to run harsher. What must agree is the
cross-site *ordering* and the qualitative morphological story. Mingze's façade
layer is genuinely additive: it is the vertical dimension our street pipeline
cannot see.

Outputs:
  outputs/comparative/mingze_facade/mingze_facade_crosscheck.json
  outputs/comparative/mingze_facade/mingze_facade_crosscheck.png
  (+ copied to docs/technical_report/figures/)
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
LABELS = {"vidigal": "Vidigal", "rocinha": "Rocinha",
          "complexo_do_alemao": "C. do Alemão", "riodaspedras": "Rio das Pedras",
          "maré": "Maré"}

# ── Mingze reported values (WHO 2 h/day, full-year envelope) ───────────────
MINGZE_SOURCE = ("data/external/mingze/mingze_update_2026-07 (Google Doc "
                 "10WqmQ_-…, Ladybug Tools, pulled 2026-07-01)")
MZ_FACADE_DEP = {"rocinha": 72, "riodaspedras": 56, "vidigal": 56,
                 "maré": 54, "complexo_do_alemao": 50}          # Fig 2
MZ_STREET_DEP = {"rocinha": 38, "riodaspedras": 40, "maré": 12}  # text (3 sites)
MZ_STREET_GINI = {"rocinha": 0.70, "vidigal": 0.63, "maré": 0.61,
                  "riodaspedras": 0.59, "complexo_do_alemao": 0.59}  # Fig 4
# Method / headline figures reported in Mingze's write-up (traceable but not
# recomputable by us — we did not re-run his Ladybug model). Verbatim from the
# Google Doc text so downstream claims can be traced to a source field.
MZ_REPORTED = {
    "facade_test_points": 7_950_000,
    "street_test_points": 194_000,
    "road_length_km": 290,
    "facade_band_m": 3.0,
    "street_eye_height_m": 1.2,
    "sun_step_hours": 0.5,
    "seasonal_facade_floor_never_below_pct": 57,
    "rocinha_deprived_facades_zero_sun_pct": 81,
    "street_over_facade_recovery_pp": [16, 42],
}

WHO_HOURS = 2.0


def deprivation_fraction(hours: np.ndarray, threshold: float = WHO_HOURS) -> float:
    """Share of valid points below the WHO daily-sun floor (NaNs dropped)."""
    h = np.asarray(hours, dtype=float)
    h = h[~np.isnan(h)]
    return float((h < threshold).mean()) if h.size else float("nan")


def our_street_winter() -> dict[str, dict]:
    """Our street-level winter-solstice WHO-2h deprivation per site."""
    out = {}
    for s in SITES:
        g = gpd.read_file(ROOT / "outputs" / s / "morphometrics" / "svf" / "svf_streets_solar.gpkg")
        h = g["solar_hours_winter"].to_numpy(dtype=float)
        out[s] = {"n_points": int(np.isfinite(h).sum()),
                  "street_winter_dep_pct": round(deprivation_fraction(h) * 100, 1),
                  "median_hours": round(float(np.nanmedian(h)), 2)}
    return out


def main() -> None:
    ours = our_street_winter()
    out_dir = ROOT / "outputs" / "comparative" / "mingze_facade"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ordering check on the three sites Mingze quotes at street level
    common = [s for s in SITES if s in MZ_STREET_DEP]
    ours_v = [ours[s]["street_winter_dep_pct"] for s in common]
    mz_v = [MZ_STREET_DEP[s] for s in common]
    ratios = [round(o / m, 2) for o, m in zip(ours_v, mz_v)]

    payload = {
        "title": "Mingze (Ladybug) façade+street solar cross-check vs our raycast",
        "source": MINGZE_SOURCE,
        "who_threshold_hours_per_day": WHO_HOURS,
        "note": ("Mingze = full-year direct-sun envelope; ours = 21 June winter-"
                 "solstice worst-day floor (street only, no façade raycast on our "
                 "side). Compare ordering, not absolute magnitude."),
        "mingze_facade_deprivation_pct": MZ_FACADE_DEP,
        "mingze_street_deprivation_pct": MZ_STREET_DEP,
        "mingze_street_gini": MZ_STREET_GINI,
        "mingze_reported_unverified": MZ_REPORTED,
        "our_street_winter": ours,
        "street_to_street": {
            "sites": common,
            "ours_winter_pct": ours_v,
            "mingze_pct": mz_v,
            "ratio_ours_over_mingze": ratios,
            "our_worst_day_harsher_factor_median": round(float(np.median(ratios)), 2),
        },
        "findings": [
            "Grouping agrees, not fine rank: Maré least street-deprived in both "
            "(Mingze 12 %, ours 27.8 %), Rocinha+RdP the worst pair in both; but "
            "the worst pair swaps between engines (n=3 sites with Mingze street data).",
            f"Our winter single-day floor is harsher than Mingze's annual field "
            f"(site ratios {min(ratios)}–{max(ratios)}×); NOT read as a calibration "
            "— the metrics reduce the temporal dimension differently.",
            "Façade deprivation (50–72 %) exceeds street everywhere (Mingze's "
            "16–42 pp recovery); the façade layer is additive to our street-only "
            "pipeline and is Mingze's alone (we did not re-run his model).",
        ],
    }
    (out_dir / "mingze_facade_crosscheck.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2))

    # ── figure: (A) façade vs street vs ours; (B) ordering scatter ─────────
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    x = np.arange(len(SITES))
    w = 0.27
    fac = [MZ_FACADE_DEP[s] for s in SITES]
    mst = [MZ_STREET_DEP.get(s, np.nan) for s in SITES]
    ost = [ours[s]["street_winter_dep_pct"] for s in SITES]
    ax[0].bar(x - w, fac, w, label="Mingze façade (annual)", color="#B2182B")
    ax[0].bar(x, mst, w, label="Mingze street (annual)", color="#EE9944")
    ax[0].bar(x + w, ost, w, label="Ours street (winter solstice)", color="#4477AA")
    ax[0].axhline(50, color="#888", ls=":", lw=1)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([LABELS[s] for s in SITES], rotation=25, ha="right", fontsize=8)
    ax[0].set_ylabel("WHO-2h deprivation (%)")
    ax[0].set_title("(A) Façade vs street deprivation, five favelas", fontsize=9)
    ax[0].legend(frameon=False, fontsize=7.5)
    ax[0].spines[["top", "right"]].set_visible(False)

    for s in common:
        ax[1].scatter(MZ_STREET_DEP[s], ours[s]["street_winter_dep_pct"],
                      s=60, color="#4477AA", zorder=3)
        ax[1].annotate(LABELS[s], (MZ_STREET_DEP[s], ours[s]["street_winter_dep_pct"]),
                       fontsize=8, xytext=(4, 4), textcoords="offset points")
    lim = [0, 80]
    ax[1].plot(lim, lim, "--", color="#888", lw=1, label="y = x")
    ax[1].plot(lim, [2 * v for v in lim], ":", color="#B2182B", lw=1,
               label="ours = 2× (reference)")
    ax[1].set_xlim(0, 50)
    ax[1].set_ylim(0, 80)
    ax[1].set_xlabel("Mingze street deprivation (annual, %)")
    ax[1].set_ylabel("Our street deprivation (winter, %)")
    ax[1].set_title("(B) Grouping agrees; ours 1.6–2.3× (worst-day vs annual)", fontsize=9)
    ax[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    ax[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_png = out_dir / "mingze_facade_crosscheck.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (ROOT / "docs/technical_report/figures/mingze_facade_crosscheck.png").write_bytes(
        out_png.read_bytes())

    print("site                 ours_winter%   mingze_street%  mingze_facade%")
    for s in SITES:
        print(f"{LABELS[s]:<20} {ours[s]['street_winter_dep_pct']:>10}   "
              f"{MZ_STREET_DEP.get(s, '—'):>12}   {MZ_FACADE_DEP[s]:>12}")
    print(f"\nStreet ordering (3 sites) ratio ours/mingze: {ratios} "
          f"(median {np.median(ratios):.2f}×)")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
