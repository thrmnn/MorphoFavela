"""Urban-physics schematics for the roughness + wall-treatment explainer.

Four hand-built diagrams (no data — pure concept art) that make the theory
reviewable at a glance:
  1. log_profile      — the log wind profile + what z0 / zd / H actually mean
  2. two_roles        — the two decoupled roles of z0 (inlet fetch vs patch ground)
  3. regimes          — isolated → wake → skimming, and why favela density breaks it
  4. wall_treatment   — rough-wall function: ks = 9.793·z0/Cs and the ks < yP rule

Output: docs/roughness_explainer/*.png (committed; referenced by
docs/roughness_wall_treatment_explainer.md and surfaced in the hub).

    python scripts/build_roughness_schematics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "roughness_explainer"

INK = "#1a1a1a"
BLD = "#8c95a3"      # buildings
WIND = "#0f766e"     # teal accent (matches the hub)
HOT = "#b2182b"
MUT = "#6a6a6a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888",
                     "svg.fonttype": "none"})


def _buildings(ax, xs, hs, w=0.7, color=BLD):
    for x, h in zip(xs, hs):
        ax.add_patch(Rectangle((x - w / 2, 0), w, h, facecolor=color,
                               edgecolor="#5b626d", lw=0.8, zorder=3))


def fig_log_profile():
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    H = 3.0           # mean building height
    zd = 3.4          # displacement height (can exceed H_mean in dense fabric)
    z0 = 0.35         # roughness length (drawn large for legibility)
    xs = [0.6, 1.5, 2.4, 3.3, 4.2, 5.1, 6.0]
    hs = [2.6, 3.4, 2.9, 3.6, 2.7, 3.2, 3.0]
    _buildings(ax, xs, hs)

    # the log-law wind profile U(z) = (u*/k) ln((z-zd)/z0), drawn sideways
    z = np.linspace(zd + z0, 9.5, 200)
    u = np.log((z - zd) / z0)
    u = 6.7 * u / u.max() + 0.4    # scale into the panel width
    ax.plot(u, z, color=WIND, lw=2.6, zorder=5)
    mid = 150
    ax.annotate("mean wind\nprofile  U(z)", (u[mid], z[mid]), (u[mid] + 0.3, z[mid] - 0.6),
                color=WIND, fontsize=9, ha="left", va="top", weight="bold",
                arrowprops=dict(arrowstyle="->", color=WIND, lw=1))

    for zz, lab, col in [(H, "H  — mean building height", INK),
                         (zd, "z_d  — displacement height\n(the aerodynamic 'ground')", HOT),
                         (zd + z0, "z_d + z_0", MUT)]:
        ax.axhline(zz, color=col, ls=(0, (5, 3)), lw=1.1, alpha=0.8, zorder=2)
    ax.annotate("", (7.4, zd), (7.4, zd + z0),
                arrowprops=dict(arrowstyle="<->", color=MUT, lw=1.2))
    ax.text(7.55, zd + z0 / 2, "z_0\nroughness\nlength", fontsize=8.5, va="center", color=MUT)
    ax.text(0.05, H + 0.05, "H", color=INK, fontsize=11, weight="bold")
    ax.text(0.05, zd + 0.08, "z_d", color=HOT, fontsize=11, weight="bold")

    ax.text(0.4, 8.7, r"$U(z)=\dfrac{u_*}{\kappa}\,\ln\!\dfrac{z-z_d}{z_0}$",
            fontsize=13, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f3f6f5", ec=WIND, lw=1))
    ax.text(3.2, 8.55, "z_0 = how rough (bumpiness the wind feels)\n"
            "z_d = how high the wind's 'floor' is lifted\n"
            "In dense fabric z_d can sit ABOVE H (tall roofs dominate drag)",
            fontsize=8.3, color=MUT, va="top")

    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 9.8)
    ax.set_xlabel("wind speed  →")
    ax.set_ylabel("height above terrain  z")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("1 · The log wind profile — what z₀ and z_d mean", weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "log_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_two_roles():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.axvline(9.2, color="#bbb", ls="--", lw=1)

    # left: upstream fetch → an inlet profile
    ax.text(3.5, 6.6, "UPSTREAM FETCH", ha="center", weight="bold", color=INK)
    ax.text(3.5, 6.15, "the fabric the wind crosses BEFORE the patch", ha="center",
            fontsize=8.5, color=MUT)
    rng = np.random.default_rng(1)
    xs = np.arange(0.6, 7.2, 0.62)
    _buildings(ax, xs, 0.6 + rng.random(len(xs)) * 0.9, w=0.42, color="#a7adb8")
    z = np.linspace(0.05, 4.6, 100)
    prof = np.log((z + 0.2) / 0.2)
    prof = 1.7 * prof / prof.max()
    ax.plot(8.9 - prof, z + 0.0, color=WIND, lw=2.4)
    ax.add_patch(FancyArrowPatch((0.2, 5.4), (7.0, 5.4), arrowstyle="-|>",
                 mutation_scale=16, color=WIND, lw=2))
    ax.text(3.5, 5.55, "wind", color=WIND, fontsize=9, ha="center")
    ax.text(8.75, 4.8, "inlet ABL set from\nz₀_inlet(θ)  (rough wall)", color=WIND,
            fontsize=8.6, ha="right", va="bottom", weight="bold")

    # right: resolved patch — buildings drawn explicitly, ground stays smooth
    ax.text(13.6, 6.6, "RESOLVED PATCH", ha="center", weight="bold", color=INK)
    ax.text(13.6, 6.15, "the patch's own buildings are MESHED explicitly", ha="center",
            fontsize=8.5, color=MUT)
    xs2 = [10.4, 11.5, 12.5, 13.6, 14.7, 15.8, 16.8]
    hs2 = [2.4, 3.1, 2.0, 3.4, 2.2, 2.9, 2.5]
    _buildings(ax, xs2, hs2, w=0.82, color=BLD)
    ax.add_patch(Rectangle((9.9, 0), 7.4, 0.22, facecolor="#d8b26a", edgecolor="none"))
    ax.text(13.6, 0.5, "ground z₀ ≈ 0.01–0.03 m  (small, mesh-valid)", ha="center",
            fontsize=8.6, color="#8a6d2f")

    ax.text(9.6, 7.5, "Two roles of z₀ — never double-count", ha="center", weight="bold",
            fontsize=11.5, color=INK)
    ax.text(9.6, -0.9, "The upwind fabric sets how rough the ARRIVING wind is; the patch's own "
            "buildings already create drag by being drawn.\nInflating the ground roughness under "
            "them too would count the same buildings twice (the Blocken trap).", ha="center",
            fontsize=8.3, color=MUT)

    ax.set_xlim(0, 17.6)
    ax.set_ylim(-1.4, 7.9)
    ax.axis("off")
    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(OUT / "two_roles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_regimes():
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    lp = np.linspace(0.02, 0.98, 300)
    # schematic z0/H: rise to a peak near lp~0.15 then decay toward 0 (skimming)
    z0H = (lp * np.exp(-lp / 0.13))
    z0H = 0.16 * z0H / z0H.max()
    ax.plot(lp, z0H, color=WIND, lw=2.8, zorder=5)

    for x0, x1, lab in [(0.02, 0.12, "isolated\nroughness"),
                        (0.12, 0.40, "wake\ninterference"),
                        (0.40, 0.98, "skimming\nflow")]:
        ax.axvspan(x0, x1, color="#f0f2f2" if lab[0] != "s" else "#fbecec",
                   zorder=0)
        ax.text((x0 + x1) / 2, 0.172, lab, ha="center", va="top", fontsize=9,
                color=MUT)

    ax.axvspan(0.5, 0.98, facecolor="none", hatch="///", edgecolor="#e6a6a6", lw=0, zorder=1)
    ax.text(0.74, 0.055, "FAVELA FABRIC\nλ_p > 0.5 — outside every\nmethod's calibration;\n"
            "z₀ collapses toward 0", ha="center", fontsize=8.4, color=HOT, weight="bold")
    ax.axhline(0.012, color=INK, ls=":", lw=1.3)
    ax.text(0.02, 0.016, "floor: z₀ clamped to 0.03 m so the log profile stays usable",
            fontsize=8, color=INK)

    # little building rows under each regime
    for cx, n, w in [(0.07, 2, 0.010), (0.26, 4, 0.010), (0.7, 8, 0.008)]:
        for i in range(n):
            ax.add_patch(Rectangle((cx - n * w + i * 2 * w, -0.022), w, 0.014,
                         facecolor=BLD, edgecolor="none", clip_on=False, zorder=6))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 0.19)
    ax.set_xlabel("plan area density  λ_p  (built fraction of ground)")
    ax.set_ylabel("roughness  z₀ / H")
    ax.set_title("3 · Roughness regimes — and why favela density breaks the estimate",
                 weight="bold")
    ax.text(0.145, 0.15, "peak roughness:\nsparse-enough to snag the wind",
            fontsize=8.3, color=WIND)
    fig.tight_layout()
    fig.savefig(OUT / "regimes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_wall_treatment():
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    # ground line
    ax.plot([0, 12], [0, 0], color=INK, lw=1.5)
    # first-cell band (the near-wall mesh cell)
    yP = 1.0
    ax.add_patch(Rectangle((0, 0), 12, yP, facecolor="#eef3f2", edgecolor="#bcd",
                 lw=0.8, zorder=0))
    ax.axhline(yP, color="#88a", ls="--", lw=1)
    ax.text(6.8, yP + 0.08, "first mesh cell (top = y_P)", ha="center", fontsize=8.5,
            color="#4a4a70")
    ax.plot([0.5], [yP / 2], marker="x", color="#4a4a70")
    ax.annotate("y_P = first-cell\ncentroid height", (0.5, yP / 2), (1.1, yP / 2 + 0.35),
                fontsize=8, color="#4a4a70",
                arrowprops=dict(arrowstyle="->", color="#4a4a70", lw=0.9))

    # the sand-grain roughness ks drawn as bumps inside the cell
    ks = 0.55
    bx = np.linspace(0.3, 5.4, 26)
    ax.fill_between(bx, 0, ks * (0.6 + 0.4 * np.sin(bx * 6)), color="#d8b26a",
                    edgecolor="none", zorder=2)
    ax.annotate("", (5.7, 0), (5.7, ks), arrowprops=dict(arrowstyle="<->", color="#8a6d2f"))
    ax.text(5.62, ks + 0.12, "k_s", fontsize=10, color="#8a6d2f", va="bottom", ha="center",
            weight="bold")
    ax.text(3.2, 1.55, "k_s = equivalent sand-grain roughness height", fontsize=8.3,
            color="#8a6d2f")
    ax.text(2.7, -0.5, "APPROACH FLOOR — rough-wall function\nz₀ = z₀_inlet", ha="center",
            fontsize=8.7, color=INK)

    # right: under resolved buildings, small ground z0
    _buildings(ax, [8.4, 9.6, 10.8], [1.9, 2.5, 2.0], w=0.9, color=BLD)
    ax.add_patch(Rectangle((7.4, 0), 4.4, 0.06, facecolor="#d8b26a", edgecolor="none", zorder=2))
    ax.text(9.6, -0.5, "UNDER RESOLVED BUILDINGS —\nsmall mesh-valid z₀", ha="center",
            fontsize=8.7, color=INK)

    ax.text(6.0, 3.5, r"$k_s=\dfrac{9.793\,z_0}{C_s}\quad(C_s\!\approx\!0.5)$", fontsize=13,
            ha="center", bbox=dict(boxstyle="round,pad=0.4", fc="#f3f6f5", ec=WIND, lw=1))
    ax.text(6.0, 2.55, "the wall model turns a roughness LENGTH z₀ into an equivalent grain height k_s",
            ha="center", fontsize=8.4, color=MUT)
    ax.text(6.0, 2.05, "hard rule:  k_s < y_P  — the roughness must fit INSIDE the first cell",
            ha="center", fontsize=9, color=HOT, weight="bold")

    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-1.0, 4.2)
    ax.axis("off")
    ax.set_title("4 · Wall treatment — turning z₀ into a rough-wall boundary condition",
                 weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "wall_treatment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_log_profile()
    fig_two_roles()
    fig_regimes()
    fig_wall_treatment()
    print(f"4 schematics → {OUT.relative_to(ROOT)}/")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
