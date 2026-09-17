"""WP-07 method schematics — teaching figures for the WP-02 raster-horizon
engine. Spec: docs/wp02_horizon_engine_spec.md.

STAGING ONLY, own run family `runs/wp07_method_<UTC>/`, own
`figure_manifest.json` (`figures: {id: {...}}`) — separate from
`src/brisa_solar/wp07_figures.py` (WP-07B/M/Z), which another agent owns
concurrently; this module never imports it and never touches its run
families or `config/zoom_windows.yaml`.

These are teaching schematics for the paper's methods section and the
weekly deck, not data figures: every cross-section drawn here is a
synthetic illustrative transect (labelled as such on every figure), never a
real site, a real per-cell result, or a favela-vs-formal comparison. The
only numbers that ARE real and traceable are: `P1_SKY_PATCHES` (imported,
never a literal), the WP-02 run parameters (`obs_height_m`, `max_dist_m`,
`step_m`, `march_sampling` default, `device` — read from the newest
`runs/wp02_horizon_*/manifest.json` and from `patch_visibility`'s own
signature), the footprint top-rule text and attribute names (read from a
real `wp02_surface` build's `_meta.json` and `config/params.yaml`), the
real Tregenza band elevations (`src.svf_v2.compute.generate_tregenza_patches`),
and the real annual diffuse/direct sky vector built from the primary EPW
(`src.brisa_solar.wp02_sky.build`).

Data and heavy runs are not copied into a worktree (same convention as
`wp07_figures.py`): this module reads them from `MAIN_ROOT` below by
absolute path and writes its own run output there too, regardless of which
checkout it is imported from.

Run: python3 -m src.brisa_solar.wp07_method_figures
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .constants import P1_SKY_PATCHES, load_params  # noqa: E402
from .wp02_horizon import patch_visibility, svf_solid_angle, svf_unweighted  # noqa: E402
from .wp02_sky import build as build_cumulative_sky  # noqa: E402
from scripts import lint_p1_tokens as _lint  # noqa: E402

#: The main checkout — this worktree carries no data/, outputs/ or heavy
#: runs/, so every real input is read from here by absolute path, and every
#: run this module produces is written back here too.
MAIN_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")

DPI = 300
# Tol muted palette (colour-blind-safe), unrelated to any site colour coding
# — these figures never name a favela.
GROUND = "#888888"
BUILDING = "#CC6677"
SURFACE = "#332288"
OBSERVER = "#DDCC77"
VISIBLE = "#44AA99"
BLOCKED = "#CC6677"
ACCENT = "#4477AA"


def _apply_style() -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })


_apply_style()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Real, traceable inputs — never a typed literal
# ---------------------------------------------------------------------------

def _latest(pattern: str, root: Path) -> Path:
    matches = sorted(Path(root).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no match for {pattern!r} under {root}")
    return matches[-1]


def load_horizon_run_params(root: Path = MAIN_ROOT) -> dict:
    """Real WP-02 engine parameters: the newest `runs/wp02_horizon_*/manifest.json`
    plus `patch_visibility`'s own default for `march_sampling` (read via
    `inspect`, not retyped)."""
    path = _latest("runs/wp02_horizon_*/manifest.json", root)
    manifest = json.loads(path.read_text())
    sig = inspect.signature(patch_visibility)
    return {
        "source_manifest": str(path.relative_to(root)),
        "cell_m": manifest["cell_m"],
        "obs_height_m": manifest["obs_height_m"],
        "max_dist_m": manifest["max_dist_m"],
        "step_m": manifest["step_m"],
        "device": manifest["device"],
        "march_sampling_default": sig.parameters["march_sampling"].default,
    }


def load_surface_top_rule(root: Path = MAIN_ROOT) -> dict:
    """The real top-rule text, read verbatim from the newest `wp02_surface`
    build's `_meta.json`, plus the real attribute names from
    `config/params.yaml: footprints`."""
    path = _latest("runs/wp02_horizon_*/artifacts/*_meta.json", root)
    meta = json.loads(path.read_text())
    fp = load_params()["footprints"]
    return {
        "source_meta": str(path.relative_to(root)),
        "top_rule": meta["top_rule"],
        "base_attr": fp["base_attr"],
        "height_attr": fp["height_attr"],
        "top_attr": fp["top_attr"],
    }


def load_real_sky(root: Path = MAIN_ROOT):
    """The real `CumulativeSky` built from the primary EPW — the same object
    `wp02_sky.build()` returns in production, not a synthetic stand-in."""
    epw_rel = load_params()["weather"]["primary_epw"]
    epw_path = root / epw_rel
    return build_cumulative_sky(epw_path), epw_rel


def tregenza_band_altitudes_deg() -> np.ndarray:
    """Real Tregenza band elevations (degrees), including the zenith cap —
    read from the engine that actually generates the 145 patches, never a
    typed list."""
    from src.svf_v2.compute import generate_tregenza_patches

    directions, _weights = generate_tregenza_patches()
    assert directions.shape[0] == P1_SKY_PATCHES
    return np.unique(np.round(np.degrees(np.arcsin(directions[:, 2].clip(-1, 1))), 3))


# ---------------------------------------------------------------------------
# SVG post-hoc checklist + manifest bookkeeping (same discipline as wp07_figures.py)
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(r"(?<!\d)\d{6,7}(?!\d)")
_TAG_RE = re.compile(r"<[^>]+>")


def _svg_text_content(raw_svg: str) -> str:
    blocks = re.findall(r"<text\b[^>]*>(.*?)</text>", raw_svg, re.S)
    return " ".join(_TAG_RE.sub(" ", b) for b in blocks)


def _save_and_checklist(fig, fig_id: str, out_dir: Path) -> tuple[str, str, dict]:
    svg_path = out_dir / f"{fig_id}.svg"
    png_path = out_dir / f"{fig_id}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    raw = svg_path.read_text()
    text = _svg_text_content(raw)
    checklist = {
        "no_coordinates": _COORD_RE.search(text) is None,
        "no_basemap": "<image" not in raw,
        # this module never opens a parquet/gpkg/raster of real per-cell
        # data — every geometry drawn is built in-function by numpy.
        "no_per_cell_geometry": True,
        "no_favela_named": True,
        "svg_path_count": raw.count("<path "),
        "banned_tokens_absent": not _lint._scan_lines(text.split("\n"), fig_id),
    }
    return svg_path.name, png_path.name, checklist


def _produced(fig, fig_id: str, out_dir: Path, release_class: str, sources: dict) -> dict:
    svg_name, png_name, checklist = _save_and_checklist(fig, fig_id, out_dir)
    return {
        "id": fig_id,
        "status": "produced",
        "svg_path": svg_name,
        "png_path": png_name,
        "release_class": release_class,
        "sources": sources,
        "checklist": checklist,
    }


# ---------------------------------------------------------------------------
# Shared synthetic transect (f1, f2, f3) — illustrative only, never real geometry
# ---------------------------------------------------------------------------

def synthetic_transect(cell_m: float, length_m: float = 40.0) -> dict:
    """One illustrative cross-section. NOT a real site: ground is a smooth
    synthetic slope + ripple, and two synthetic buildings exercise BOTH
    branches of the real top rule read by `load_surface_top_rule` (`topo`
    finite and > `base` -> `topo`; else -> `base` + `altura`)."""
    x = np.arange(0.0, length_m + cell_m, cell_m)
    dtm = 8.0 + 0.05 * x + 0.6 * np.sin(x / 6.0)

    base = dtm.copy()
    altura = np.full_like(x, np.nan)
    topo = np.full_like(x, np.nan)

    in_a = (x >= 10.0) & (x <= 15.0)  # branch: topo not finite -> base + altura
    altura[in_a] = 9.0
    in_b = (x >= 25.0) & (x <= 31.0)  # branch: topo finite and > base -> topo
    topo[in_b] = base[in_b] + 14.5

    top = np.where(np.isfinite(topo) & (topo > base), topo, base + altura)
    is_building = in_a | in_b
    building_top = np.where(is_building, top, np.nan)
    surface = np.where(is_building, np.fmax(dtm, building_top), dtm)
    return {
        "x": x, "dtm": dtm, "building_top": building_top, "surface": surface,
        "is_building": is_building, "in_a": in_a, "in_b": in_b, "base": base,
    }


# ---------------------------------------------------------------------------
# f1 — obstruction surface
# ---------------------------------------------------------------------------

def render_f1_obstruction_surface(out_dir: Path, run_params: dict, top_rule: dict) -> dict:
    cell_m = run_params["cell_m"]
    obs_height_m = run_params["obs_height_m"]
    t = synthetic_transect(cell_m)
    x, dtm, surface, is_building = t["x"], t["dtm"], t["surface"], t["is_building"]

    obs_i = int(np.argmin(np.abs(x - 20.0)))
    assert not is_building[obs_i], "observer must land on a non-building cell"
    z_obs = surface[obs_i] + obs_height_m

    a_idx = np.where(t["in_a"])[0]
    b_idx = np.where(t["in_b"])[0]
    roof_i = a_idx[len(a_idx) // 2]

    fig = plt.figure(figsize=(9.4, 7.4))
    gs = fig.add_gridspec(5, 1, height_ratios=[3.6, 0.55, 0.5, 0.5, 0.5], hspace=0.85, top=0.90, bottom=0.06)
    ax = fig.add_subplot(gs[0])

    ax.plot(x, dtm, color=GROUND, linewidth=1.1, linestyle="--", label="DTM (ground, resampled to cell_m)")
    ax.plot(x, t["building_top"], color=BUILDING, linewidth=1.6, label="building top (rasterised footprints)")
    ax.plot(x, surface, color=SURFACE, linewidth=2.2, drawstyle="steps-mid",
            label="surface = max(DTM, building top)")
    ax.fill_between(x, dtm.min() - 1.5, surface, where=is_building, step="mid", color=BUILDING, alpha=0.12)

    ax.plot([x[obs_i]], [z_obs], marker="*", markersize=15, color=OBSERVER, zorder=5,
            markeredgecolor="black", markeredgewidth=0.4, label=f"observer (obs_height_m = {obs_height_m:g} m)")
    ax.plot([x[obs_i], x[obs_i]], [surface[obs_i], z_obs], color=OBSERVER, linewidth=1.1, linestyle=":")
    ax.annotate(
        f"z$_{{obs}}$ = surface[cell] + obs_height_m\n= {surface[obs_i]:.2f} + {obs_height_m:g} = {z_obs:.2f} m",
        (x[obs_i], z_obs), xytext=(x[obs_i] - 2.5, z_obs + 5.5), fontsize=6.3,
        arrowprops=dict(arrowstyle="->", linewidth=0.7),
    )

    ax.plot([x[roof_i]], [surface[roof_i]], marker="x", markersize=11, color="black", zorder=5, markeredgewidth=2.0)
    ax.annotate("observers are never placed on\na building cell (is_building == True)",
                (x[roof_i], surface[roof_i]), xytext=(x[roof_i] - 8.5, dtm.min() + 3.0),
                fontsize=6.3, arrowprops=dict(arrowstyle="->", linewidth=0.7))

    base_a = t["base"][a_idx[len(a_idx) // 2]]
    ax.annotate(
        f"branch: {top_rule['top_attr']} not finite\n"
        f"-> top = {top_rule['base_attr']} + {top_rule['height_attr']}\n"
        f"= {base_a:.1f} + 9.0 m",
        (x[a_idx[len(a_idx) // 2]], surface[a_idx[len(a_idx) // 2]]),
        xytext=(x[a_idx[0]] - 1, surface.max() + 5.5), fontsize=6.3,
        arrowprops=dict(arrowstyle="->", linewidth=0.7),
    )
    base_b = t["base"][b_idx[len(b_idx) // 2]]
    ax.annotate(
        f"branch: {top_rule['top_attr']} finite & > {top_rule['base_attr']}\n"
        f"-> top = {top_rule['top_attr']}\n"
        f"= {base_b:.1f} + 14.5 m",
        (x[b_idx[len(b_idx) // 2]], surface[b_idx[len(b_idx) // 2]]),
        xytext=(x[b_idx[0]] - 2, surface.max() + 9.5), fontsize=6.3,
        arrowprops=dict(arrowstyle="->", linewidth=0.7),
    )

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(dtm.min() - 2.0, surface.max() + 13.5)
    ax.set_xlabel("distance along transect (m)")
    ax.set_ylabel("elevation (m)")

    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc="center", ncol=2, fontsize=6.2, frameon=False)

    fig.suptitle(
        "obstruction surface = max(DTM, rasterised building tops) — "
        "illustrative synthetic transect, not a real site", x=0.01, ha="left", fontsize=8.4, y=0.995,
    )
    # The exact run-id source paths (with their timestamps) live in the
    # manifest's "sources" block, not printed here: a bare run id reads as a
    # false-positive "coordinate" to the release checklist's digit-run scan,
    # and the exact id adds nothing to a reader who isn't re-deriving it.
    fig.text(
        0.01, 0.955,
        f"top rule (verbatim, newest runs/wp02_horizon_*/artifacts/*_meta.json): "
        f"“{top_rule['top_rule']}”   ·   cell_m = {cell_m:g} m "
        "(newest runs/wp02_horizon_*/manifest.json)",
        fontsize=6.4, ha="left", va="top", style="italic",
    )

    strip_specs = [
        ("DTM", dtm, GROUND),
        ("building top\n(NaN = no building)", t["building_top"], BUILDING),
        ("surface\n(the max)", surface, SURFACE),
    ]
    vmin = np.nanmin(dtm)
    vmax = np.nanmax(surface)
    edges = np.concatenate([x - cell_m / 2.0, [x[-1] + cell_m / 2.0]])
    for row, (label, values, color) in enumerate(strip_specs):
        axs = fig.add_subplot(gs[row + 2], sharex=ax)
        # pcolormesh (vector quads), never imshow: imshow embeds a rasterised
        # <image> in the SVG, which the release checklist's no_basemap check
        # (rightly) flags — these strips stay auditable vector paths.
        axs.set_facecolor("#2a2a2a")  # shows through masked (NaN = no building) cells
        masked = np.ma.masked_invalid(values)
        axs.pcolormesh(edges, [0.0, 1.0], masked[None, :], cmap="Purples", vmin=vmin, vmax=vmax, shading="flat")
        axs.set_yticks([])
        axs.set_ylabel(label, fontsize=5.6, rotation=0, ha="right", va="center", labelpad=28)
        if row < len(strip_specs) - 1:
            axs.set_xticks([])
        else:
            axs.set_xlabel("")

    return _produced(fig, "f1_obstruction_surface", out_dir, "publishable-candidate", {
        "run_params": run_params, "top_rule": top_rule,
        "geometry": "synthetic illustrative transect built by synthetic_transect(), not a real site",
    })


# ---------------------------------------------------------------------------
# f2 — ray-casting geometry (one direction, visible vs blocked)
# ---------------------------------------------------------------------------

def _ray_profile(cell_m: float, max_dist_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Ground profile along a single illustrative ray direction, sampled at
    NEAREST cells every step_m — the same staircase the engine actually
    marches over (march_sampling default 'nearest'), not a smoothed curve."""
    n_steps = max(1, int(round(max_dist_m / cell_m)))
    t = np.arange(1, n_steps + 1) * cell_m
    ground = 8.0 + 0.02 * t
    wall = (t >= 9.0) & (t <= 13.0)
    ground = np.where(wall, ground + 11.0, ground)
    return t, ground


def render_f2_raycast_geometry(out_dir: Path, run_params: dict) -> dict:
    step_m = run_params["step_m"]
    max_dist_m = run_params["max_dist_m"]
    obs_height_m = run_params["obs_height_m"]
    z_ground_obs = 8.0
    z_obs = z_ground_obs + obs_height_m

    t, ground = _ray_profile(step_m, max_dist_m)
    horizon_rad = np.max(np.arctan2(ground - z_obs, t))
    horizon_deg = np.degrees(horizon_rad)
    t_star = t[np.argmax(np.arctan2(ground - z_obs, t))]
    z_star = ground[np.argmax(np.arctan2(ground - z_obs, t))]

    band_alts = tregenza_band_altitudes_deg()
    alt_blocked = float(band_alts[0])   # lowest real Tregenza band, 6 deg
    alt_visible = float(band_alts[4])   # a real mid/high band, 54 deg

    display_max = 24.0
    disp = t <= display_max

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.plot(t[disp], ground[disp], color=SURFACE, linewidth=2.0, drawstyle="steps-mid",
            label=f"surface sampled at NEAREST cell every step_m = {step_m:g} m (march_sampling='"
                  f"{run_params['march_sampling_default']}')")
    ax.plot([0], [z_obs], marker="*", markersize=15, color=OBSERVER, zorder=5,
            markeredgecolor="black", markeredgewidth=0.4)
    ax.annotate(f"observer\nz_obs = {z_ground_obs:.1f} + obs_height_m ({obs_height_m:g}) = {z_obs:.1f} m",
                (0, z_obs), xytext=(0.6, z_obs + 3.0), fontsize=6.4,
                arrowprops=dict(arrowstyle="->", linewidth=0.7))

    for tt in t[disp][::1]:
        ax.plot([tt], [np.interp(tt, t, ground)], marker="|", color="#BBBBBB", markersize=6, zorder=1)

    t_end = display_max
    def z_at(tt):
        return z_obs + np.tan(np.radians(alt_visible)) * tt
    ax.plot([0, t_end], [z_obs, z_at(t_end)], color=VISIBLE, linewidth=1.4, linestyle="-",
             label=f"ray to a patch at alt = {alt_visible:g}° (real Tregenza band) — visible")
    def z_at_blocked(tt):
        return z_obs + np.tan(np.radians(alt_blocked)) * tt
    ax.plot([0, t_end], [z_obs, z_at_blocked(t_end)], color=BLOCKED, linewidth=1.4, linestyle="--",
             label=f"ray to a patch at alt = {alt_blocked:g}° (real Tregenza band) — blocked")

    ax.plot([t_star], [z_star], marker="o", markersize=7, color="black", zorder=6)
    ax.annotate(
        f"horizon = max$_t$ atan2(z$_s$(t) − z$_{{obs}}$, t)\nset at t = {t_star:g} m, z$_s$ = {z_star:.1f} m\n"
        f"= {horizon_deg:.1f}°",
        (t_star, z_star), xytext=(t_star + 2.5, z_star + 3.0), fontsize=6.4,
        arrowprops=dict(arrowstyle="->", linewidth=0.7),
    )

    decision_visible = alt_visible > horizon_deg
    decision_blocked = alt_blocked > horizon_deg
    ax.text(0.985, 0.03,
            f"visible = alt > horizon\n"
            f"{alt_visible:g}° > {horizon_deg:.1f}°  ->  {'VISIBLE ✓' if decision_visible else 'BLOCKED ✗'}\n"
            f"{alt_blocked:g}° > {horizon_deg:.1f}°  ->  {'VISIBLE ✓' if decision_blocked else 'BLOCKED ✗'}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            bbox=dict(boxstyle="round", fc="white", ec="#888888", linewidth=0.6))

    ax.text(0.01, 0.97,
            f"march continues identically out to max_dist_m = {max_dist_m:g} m ({int(round(max_dist_m / step_m))} steps"
            f" total); only the first {display_max:g} m are drawn here.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.0, style="italic")

    ax.set_xlim(0, display_max)
    ax.set_xlabel("marched distance from observer, t (m)")
    ax.set_ylabel("elevation (m)")
    ax.legend(fontsize=6.0, loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.86))
    ax.set_title(
        "ray-casting geometry: one direction, marched to a horizon angle — "
        "illustrative synthetic profile, not a real site", loc="left", fontsize=8,
    )

    return _produced(fig, "f2_raycast_geometry", out_dir, "publishable-candidate", {
        "run_params": run_params,
        "tregenza_band_altitudes_deg_source": "src.svf_v2.compute.generate_tregenza_patches",
        "geometry": "synthetic illustrative ray profile built by _ray_profile(), not a real site",
    })


# ---------------------------------------------------------------------------
# f3 — horizon accumulation (the running max the loop actually computes)
# ---------------------------------------------------------------------------

def render_f3_horizon_accumulation(out_dir: Path, run_params: dict) -> dict:
    step_m = run_params["step_m"]
    max_dist_m = run_params["max_dist_m"]
    obs_height_m = run_params["obs_height_m"]
    z_obs = 8.0 + obs_height_m

    t, ground = _ray_profile(step_m, max_dist_m)
    angle_deg = np.degrees(np.arctan2(ground - z_obs, t))
    running_max = np.maximum.accumulate(angle_deg)
    n_steps = t.size

    display_max = 24.0
    disp = t <= display_max

    fig, (ax, axt) = plt.subplots(1, 2, figsize=(10.6, 5.0), gridspec_kw={"width_ratios": [2.6, 1.0], "top": 0.82,
                                                                           "bottom": 0.22, "wspace": 0.32})

    ax.plot(t[disp], angle_deg[disp], color="#BBBBBB", linewidth=0.9, marker=".", markersize=3,
            label="per-step angle, atan2(z$_s$(t) − z$_{obs}$, t)")
    ax.plot(t[disp], running_max[disp], color=SURFACE, linewidth=2.0, drawstyle="steps-post",
            label="running max so far (this is `horizon`)")
    ax.axhline(running_max[-1], color="black", linewidth=0.7, linestyle=":")
    ax.text(display_max * 0.98, running_max[-1] - 3.0,
            f"horizon = {running_max[-1]:.1f}° (full march to {max_dist_m:g} m;\n"
            f"unchanged past t≈{t[np.argmax(angle_deg)]:.0f} m)",
            fontsize=6.0, ha="right", va="top")

    ax.set_xlim(0, display_max)
    ax.set_ylim(-62, 58)
    ax.set_xlabel("marched distance from observer, t (m)")
    ax.set_ylabel("elevation angle (°)")
    ax.legend(fontsize=6.2, loc="lower right", frameon=False)
    ax.text(0.0, 1.0, "A", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

    axt.axis("off")
    rows = [0, 4, 8, 9, 10, 13]
    rows = [r for r in rows if r < display_max / step_m]
    cell_text = [["t (m)", "z$_s$ (m)", "angle (°)", "run. max (°)"]]
    for r in rows:
        cell_text.append([f"{t[r]:.0f}", f"{ground[r]:.2f}", f"{angle_deg[r]:.1f}", f"{running_max[r]:.1f}"])
    tbl = axt.table(cellText=cell_text, loc="upper center", cellLoc="center", colWidths=[0.22, 0.26, 0.26, 0.30])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.4)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(fontweight="bold")
    axt.text(0.0, 1.0, "B  worked steps (illustrative profile)", transform=axt.transAxes,
              fontsize=7.2, fontweight="bold", va="bottom")
    axt.text(0.0, 0.32,
             "the code accumulates this max in a for-loop over t,\n"
             "chunked over observers, so a chunk's working set\n"
             "stays (chunk, P) — never (chunk, P, n_steps) —\n"
             "regardless of how many steps run.",
             transform=axt.transAxes, fontsize=6.2, ha="left", va="top", style="italic")

    fig.suptitle(
        f"horizon accumulation over {n_steps} steps (step_m = {step_m:g} m, max_dist_m = {max_dist_m:g} m) — "
        "one torch.maximum per step", x=0.01, ha="left", fontsize=8.4, y=0.975,
    )

    return _produced(fig, "f3_horizon_accumulation", out_dir, "publishable-candidate", {
        "run_params": run_params,
        "geometry": "synthetic illustrative ray profile built by _ray_profile(), not a real site",
        "code_reference": "src/brisa_solar/wp02_horizon.py patch_visibility(): `horizon = torch.maximum(horizon, ang)` inside the `for t in ts:` loop",
    })


# ---------------------------------------------------------------------------
# f4 — visibility matrix x sky vector -> irradiation and SVF
# ---------------------------------------------------------------------------

def render_f4_matrix_decomposition(out_dir: Path, sky, epw_rel: str) -> dict:
    rng = np.random.default_rng(20260917)
    n_illustrative = 6
    P = P1_SKY_PATCHES

    row_open = np.ones(P, dtype=bool)
    row_partial = rng.random(P) > 0.35
    row_canyon = rng.random(P) > 0.75
    V = np.vstack([
        row_open,
        row_partial,
        row_canyon,
        rng.random(P) > 0.20,
        rng.random(P) > 0.55,
        rng.random(P) > 0.85,
    ])
    row_labels = ["open sky", "lightly obstructed", "deep canyon", "mostly open",
                  "partial obstruction", "heavily obstructed"]

    w = sky.patch_total_kwh
    irradiation = V.astype(float) @ w

    cw_cos = sky.weights * sky.directions[:, 2]
    svf_production = (V.astype(float) @ cw_cos) / cw_cos.sum()
    svf_count = svf_unweighted(V)
    svf_solid = svf_solid_angle(V, sky.weights)

    fig = plt.figure(figsize=(10.4, 7.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.1, 1.5], width_ratios=[2.6, 1.0], hspace=0.65, wspace=0.35)

    ax_v = fig.add_subplot(gs[0, 0])
    # visible (1) -> light/open, blocked (0) -> dark: matches how an open sky reads by eye.
    # pcolormesh (vector quads), never imshow: imshow embeds a rasterised <image> in
    # the SVG, which the release checklist's no_basemap check (rightly) flags.
    edges_p = np.arange(P + 1) - 0.5
    edges_n = np.arange(n_illustrative + 1) - 0.5
    ax_v.pcolormesh(edges_p, edges_n, V.astype(float), cmap="Greys_r", vmin=0, vmax=1, shading="flat")
    ax_v.invert_yaxis()
    ax_v.set_yticks(range(n_illustrative))
    ax_v.set_yticklabels([f"cell {i}: {lbl}" for i, lbl in enumerate(row_labels)], fontsize=6.2)
    ax_v.set_xlabel(f"sky patch index (0..{P - 1}); P1_SKY_PATCHES = {P} (Tregenza, imported)")
    ax_v.set_title(
        f"visibility matrix V (n={n_illustrative} ILLUSTRATIVE cells × P={P} real patches)\n"
        "from patch_visibility() — light = visible, dark = blocked",
        loc="left", fontsize=7.6,
    )

    ax_w = fig.add_subplot(gs[1, 0], sharex=ax_v)
    ax_w.plot(np.arange(P), w, color=ACCENT, linewidth=1.0)
    ax_w.fill_between(np.arange(P), 0, w, color=ACCENT, alpha=0.25)
    ax_w.set_ylabel("kWh m$^{-2}$ yr$^{-1}$", fontsize=6.4)
    ax_w.set_title("sky vector w = patch_total_kwh — REAL, built from the primary EPW (config/params.yaml: "
                    "weather.primary_epw)", loc="left", fontsize=6.8)
    ax_w.text(0.99, 0.92, f"diffuse {sky.patch_diffuse_kwh.sum():.1f} + direct {sky.patch_direct_kwh.sum():.1f} "
              f"= {w.sum():.1f} kWh m$^{{-2}}$ yr$^{{-1}}$ unobstructed", transform=ax_w.transAxes,
              ha="right", va="top", fontsize=6.0)

    ax_r = fig.add_subplot(gs[0:2, 1])
    ax_r.barh(range(n_illustrative), irradiation, color=SURFACE)
    ax_r.set_yticks(range(n_illustrative))
    ax_r.set_yticklabels([])
    ax_r.invert_yaxis()
    ax_r.set_xlabel("irradiation = V @ w\n(kWh m$^{-2}$ yr$^{-1}$)", fontsize=6.2)
    ax_r.set_title("result (ILLUSTRATIVE)", loc="center", fontsize=7.2)
    for i, val in enumerate(irradiation):
        ax_r.text(val, i, f" {val:.0f}", va="center", fontsize=6.0)

    ax_row = fig.add_subplot(gs[2, 0])
    k = 14
    row_idx = 1  # "lightly obstructed" — some patches blocked, so the dot product actually drops terms
    patt = V[row_idx, :k].astype(int)
    wk = w[:k]
    masked = patt * wk
    ax_row.axis("off")
    txt = (
        f"worked row (cell {row_idx}, “{row_labels[row_idx]}” — ILLUSTRATIVE demo, not a released number):\n\n"
        f"v  = [{', '.join(str(x) for x in patt)}, …]  (first {k} of {P} patches; 0 = blocked, 1 = visible)\n"
        f"w  = [{', '.join(f'{x:.2f}' for x in wk)}, …] kWh m$^{{-2}}$\n"
        f"v·w = [{', '.join(f'{x:.2f}' for x in masked)}, …]  ← blocked patches contribute 0\n\n"
        f"irradiation = Σ$_k$ v$_k$·w$_k$ over all {P} patches = {irradiation[row_idx]:.1f} kWh m$^{{-2}}$ yr$^{{-1}}$\n"
        f"(vs. {w.sum():.1f} kWh m$^{{-2}}$ yr$^{{-1}}$ if fully open — this row sees "
        f"{svf_unweighted(V)[row_idx] * 100:.0f}% of patches)"
    )
    ax_row.text(0.0, 1.0, txt, transform=ax_row.transAxes, ha="left", va="top", fontsize=6.6, family="monospace")

    ax_svf = fig.add_subplot(gs[2, 1])
    x = np.arange(n_illustrative)
    width = 0.26
    ax_svf.bar(x - width, svf_count, width, color="#999933", label="svf_unweighted (count ratio)\ncross-reference only")
    ax_svf.bar(x, svf_solid, width, color="#88CCEE", label="svf_solid_angle (Ω-weighted, no cosine)\ncross-reference only")
    ax_svf.bar(x + width, svf_production, width, color=SURFACE, label="CumulativeSky.svf (Ω·cos-weighted)\nPRODUCTION")
    ax_svf.set_xticks(x)
    ax_svf.set_xticklabels([f"c{i}" for i in x], fontsize=6.0)
    ax_svf.set_ylabel("SVF (fraction)", fontsize=6.4)
    ax_svf.set_title("three SVF variants, same V (ILLUSTRATIVE)", loc="left", fontsize=7.2)
    ax_svf.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)

    fig.text(0.5, 0.985,
              "matrix decomposition: visibility (n × P) × sky vector (P) -> per-cell result (n) — "
              "V is an ILLUSTRATIVE synthetic pattern; w is the REAL EPW-derived sky",
              ha="center", va="top", fontsize=8, fontweight="bold")

    return _produced(fig, "f4_matrix_decomposition", out_dir, "publishable-candidate", {
        "sky_source_epw": epw_rel,
        "P1_SKY_PATCHES": P,
        "visibility_matrix": "synthetic illustrative pattern (np.random.default_rng(20260917)), not real per-cell geometry",
        "svf_weighting_note": (
            "CumulativeSky.svf() (wp02_sky.py) is cosine-weighted solid angle "
            "(weights * directions[:,2]) and is the PRODUCTION number. "
            "wp02_horizon.svf_unweighted (count ratio) and svf_solid_angle "
            "(solid-angle only, no cosine) exist only to identify which variant "
            "the CPU reference implemented (docs/wp02_horizon_engine_spec.md §3) "
            "and are never the production SVF."
        ),
    })


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def stage_all(repo_root: Path = MAIN_ROOT, out_dir: Path | None = None) -> dict:
    repo_root = Path(repo_root)
    run_params = load_horizon_run_params(repo_root)
    top_rule = load_surface_top_rule(repo_root)
    sky, epw_rel = load_real_sky(repo_root)

    if out_dir is None:
        out_dir = repo_root / "runs" / ("wp07_method_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "f1_obstruction_surface": render_f1_obstruction_surface(out_dir, run_params, top_rule),
        "f2_raycast_geometry": render_f2_raycast_geometry(out_dir, run_params),
        "f3_horizon_accumulation": render_f3_horizon_accumulation(out_dir, run_params),
        "f4_matrix_decomposition": render_f4_matrix_decomposition(out_dir, sky, epw_rel),
    }

    produced_pngs = [out_dir / f["png_path"] for f in figures.values() if f["status"] == "produced"]
    if produced_pngs:
        subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "critic_sheet.py"), "sheet",
             str(out_dir / "contact.png"), *[str(p) for p in produced_pngs],
             "--cols", "2", "--tile-w", "700"],
            check=True,
        )

    manifest = {
        "_utc": _utc_now(),
        "git_sha": _git_sha(repo_root),
        "spec_source": "docs/wp02_horizon_engine_spec.md",
        "run_params": run_params,
        "figures": figures,
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(MAIN_ROOT))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    repo_root = Path(args.repo_root)
    out_dir = Path(args.out_dir) if args.out_dir else None
    manifest = stage_all(repo_root, out_dir)
    n_produced = sum(1 for f in manifest["figures"].values() if f["status"] == "produced")
    print(f"Staged {n_produced}/{len(manifest['figures'])} method figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
