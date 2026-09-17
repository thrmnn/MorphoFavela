"""WP-07B — staged C′ figure candidates. Spec: docs/wp07_figures_spec.md.

STAGING ONLY: renders f1-f4 into `runs/wp07_figures_<UTC>/` (PNG 300 dpi +
SVG each) plus `figure_manifest.json` carrying the guardian-readiness
checklist per figure. The orchestrator runs the ethics gate afterwards; the
PI alone promotes anything into `shared/figures/` or `papers/`. No map, no
per-cell scatter, no favela-vs-formal panel, ever — and every printed number
is a read from the newest `runs/wp07_ledger_*/ledger.json`, never typed.

Distributions come only from the two parquet sources named in the spec
(`runs/wp05_full_20260914T215419Z/wp05_full.parquet`,
`runs/wp04_sites_20260914T230606Z/<site>/ground.parquet`) and are always
read through `_read_columns`, which refuses `x`/`y`/`row`/`col` — the
citywide file is 8.4 M rows and this module never holds per-cell geometry.
Both real paths are gitignored (`runs/**/*.parquet`) and are typically
absent from a git worktree — same convention as tests/test_wp05_full.py.
When a required parquet is absent, the corresponding figure is skipped
cleanly (recorded in the manifest, no partial/fake output) rather than
raising; f3 and f4 read only the ledger and never skip for that reason.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from .constants import REPO_ROOT  # noqa: E402
from .wp07_ledger import FAVELAS, LOCKED_VARIANT, RUN_OF_RECORD, SITE_DIRS  # noqa: E402
from scripts import lint_p1_tokens as _lint  # noqa: E402

# ---------------------------------------------------------------------------
# Local style (≤30 lines) — outputs/paper_figures/fig_style.py imports clean
# (no simulation module pulled into sys.modules), but it mkdir's
# outputs/paper_figures/exports/ as an import side effect, and
# outputs/paper_figures/ is explicitly OUT of scope for WP-07B. A local
# style avoids writing there at all.
# ---------------------------------------------------------------------------
DPI = 300
COLORS = {  # Tol muted palette, colour-blind-safe
    "vidigal": "#CC6677",
    "rocinha": "#DDCC77",
    "complexo_do_alemao": "#999933",
    "mare": "#332288",
    "riodaspedras": "#44AA99",
}


def _apply_style() -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        # real <text> glyphs, not path-per-letter — the guardian regex checks
        # (a)/(d) need actual text content in the SVG.
        "svg.fonttype": "none",
    })


_apply_style()

# Fixed display order (never ranked by value): Table H's own order file was
# searched for (`rg -l figures_table_h` across ~/SCL/SCR) and not found on
# disk, so this uses the spec's stated fallback — hillside then flatland.
FIGURE_SITE_ORDER = ("vidigal", "rocinha", "complexo_do_alemao", "mare", "riodaspedras")
TABLE_H_SEARCH_NOTE = (
    "rg -l figures_table_h across ~/SCL/SCR found only docs/wp07_figures_spec.md and "
    "brisaverse/shared/ethics/red_lines.md (both merely name the requirement, neither is "
    "an order file) — using the spec's stated fallback order: Vidigal, Rocinha, "
    "Complexo do Alemão, Maré, Rio das Pedras."
)

# Athens Charter (1943), Point 26 — a fixed normative reference constant (an
# external citation, not a pipeline measurement), so it is not a ledger read.
ATHENS_CHARTER_FLOOR_HOURS = 2.0

_FORBIDDEN_COLUMNS = {"x", "y", "row", "col"}
_COORD_RE = re.compile(r"(?<!\d)\d{6,7}(?!\d)")
_TAG_RE = re.compile(r"<[^>]+>")
_GRID_RE = re.compile(r"^g3\.grid_(\d{3})_(\d{2})\.")


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


def fmt3(value) -> str:
    """3 significant figures — the same rounding ledger.md itself uses."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.3g}"


# ---------------------------------------------------------------------------
# Ledger access
# ---------------------------------------------------------------------------

def find_latest_ledger(repo_root: Path) -> Path:
    candidates = sorted((Path(repo_root) / "runs").glob("wp07_ledger_*/ledger.json"))
    if not candidates:
        raise FileNotFoundError("no runs/wp07_ledger_*/ledger.json found")
    return candidates[-1]


def get_value(ledger: dict, ledger_id: str) -> tuple[float, str]:
    """Resolve a ledger id: `entries` (a dict keyed by id) first, then the
    `g3.spread.<favela>.svf` ids that only live under `derived.spread`."""
    entries = ledger["entries"]
    if ledger_id in entries:
        e = entries[ledger_id]
        return e["value"], e["unit"]
    if ledger_id.startswith("g3.spread.") and ledger_id.endswith(".svf"):
        slug = ledger_id[len("g3.spread."):-len(".svf")]
        s = ledger["derived"]["spread"][slug]
        return s["svf_percentile_spread_max_minus_min"], "percentile"
    raise KeyError(ledger_id)


def discover_grid_variants(ledger: dict) -> list[tuple[str, float, float]]:
    """(slug, threshold, distance_m) for every g3.grid_<slug>.*.svf_percentile
    id actually present in the ledger, sorted by (threshold, distance) so
    variants sharing a coverage threshold are adjacent — never hardcode the
    9 variants' numbers."""
    seen: dict[str, tuple[float, float]] = {}
    for eid in ledger["entries"]:
        m = _GRID_RE.match(eid)
        if m:
            slug = f"{m.group(1)}_{m.group(2)}"
            seen[slug] = (int(m.group(1)) / 100.0, float(int(m.group(2))))
    return sorted(((slug, t, d) for slug, (t, d) in seen.items()), key=lambda x: (x[1], x[2]))


# ---------------------------------------------------------------------------
# Parquet access — column-selected, never per-cell geometry
# ---------------------------------------------------------------------------

def citywide_parquet_path(repo_root: Path) -> Path:
    return Path(repo_root) / "runs" / RUN_OF_RECORD["wp05"] / "wp05_full.parquet"


def site_ground_parquet_path(repo_root: Path, slug: str) -> Path:
    return Path(repo_root) / "runs" / RUN_OF_RECORD["wp04"] / SITE_DIRS[slug] / "ground.parquet"


def _read_columns(path: Path, columns: list[str]):
    assert not (_FORBIDDEN_COLUMNS & set(columns)), f"refusing to read per-cell geometry columns: {columns}"
    return pq.read_table(path, columns=columns)


def _parquet_column_range(path: Path, column: str) -> tuple[float, float]:
    assert column not in _FORBIDDEN_COLUMNS
    pf = pq.ParquetFile(path)
    idx = pf.schema_arrow.get_field_index(column)
    lo = hi = None
    for rg in range(pf.num_row_groups):
        stats = pf.metadata.row_group(rg).column(idx).statistics
        if stats is not None and stats.has_min_max:
            lo = stats.min if lo is None else min(lo, stats.min)
            hi = stats.max if hi is None else max(hi, stats.max)
    if lo is None or hi is None:
        arr = _read_columns(path, [column]).column(0).to_numpy(zero_copy_only=False)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    return float(lo), float(hi)


def _streaming_histogram(path: Path, column: str, bins: int, value_range: tuple[float, float]):
    assert column not in _FORBIDDEN_COLUMNS
    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    counts = np.zeros(bins, dtype=np.int64)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=[column], batch_size=500_000):
        vals = batch.column(0).to_numpy(zero_copy_only=False)
        vals = vals[np.isfinite(vals)]
        c, _ = np.histogram(vals, bins=edges)
        counts += c
    return counts, edges


# ---------------------------------------------------------------------------
# SVG post-hoc checklist helpers
# ---------------------------------------------------------------------------

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
        # scanned on the <text> content only — the raw SVG's own canvas
        # geometry (width/height/viewBox/path floats) can coincidentally
        # contain a 6-7 digit run (e.g. "272.531531pt") that is not a
        # coordinate anyone can read off the figure.
        "no_coordinates": _COORD_RE.search(text) is None,
        "no_basemap": "<image" not in raw,
        # guaranteed by _read_columns' forbidden-column assertion above —
        # this module never selects x/y/row/col off a parquet.
        "no_per_cell_geometry": True,
        "sites_fixed_order": True,
        "svg_path_count": raw.count("<path "),
        "banned_tokens_absent": not _lint._scan_lines(text.split("\n"), fig_id),
    }
    return svg_path.name, png_path.name, checklist


def _skip(fig_id: str, reason: str) -> dict:
    return {"id": fig_id, "status": "skipped", "reason": reason}


def _produced(fig, fig_id: str, out_dir: Path, ledger_ids: list[str],
              source_parquets: list[str], release_class: str,
              plotted_ids: list[str] | None = None) -> dict:
    svg_name, png_name, checklist = _save_and_checklist(fig, fig_id, out_dir)
    return {
        "id": fig_id,
        "status": "produced",
        "svg_path": svg_name,
        "png_path": png_name,
        "ledger_ids_used": sorted(set(ledger_ids)),
        "ledger_ids_plotted": sorted(set(plotted_ids or [])),
        "source_parquets": source_parquets,
        "release_class_proposed": release_class,
        "checklist": checklist,
    }


# ---------------------------------------------------------------------------
# f1 — citywide position
# ---------------------------------------------------------------------------

def render_f1(ledger: dict, repo_root: Path, out_dir: Path) -> dict:
    path = citywide_parquet_path(repo_root)
    if not path.exists():
        return _skip("f1_citywide_position", f"citywide parquet absent: {path}")

    ledger_ids: list[str] = []
    plotted_ids: list[str] = []
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    panels = (
        ("A", "svf", axA, "sky-view factor (fraction)", (0.0, 1.0)),
        ("B", "kwh_m2", axB, "annual ground irradiation (kWh m$^{-2}$)", None),
    )
    for tag, metric, ax, xlabel, fixed_range in panels:
        value_range = fixed_range or _parquet_column_range(path, metric)
        counts, edges = _streaming_histogram(path, metric, 100, value_range)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts, width=(edges[1] - edges[0]), color="#88AACC", edgecolor="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("citywide ground cells (thousands)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v / 1e3:.0f}"))
        ax.set_title(tag, loc="left", fontsize=8)
        ymax = ax.get_ylim()[1]
        order = sorted(FIGURE_SITE_ORDER, key=lambda s: get_value(ledger, f"favela.{s}.{metric}.median")[0])
        for k, slug in enumerate(order):
            display = FAVELAS[slug]
            median_id, pct_id = f"favela.{slug}.{metric}.median", f"favela.{slug}.{metric}.percentile"
            median_val, _ = get_value(ledger, median_id)
            pct_val, _ = get_value(ledger, pct_id)
            ledger_ids.append(pct_id)
            plotted_ids.append(median_id)
            ax.axvline(median_val, color=COLORS[slug], linewidth=1.0, linestyle="--")
            # neighbours in x alternate label height so adjacent medians do not overprint
            ax.text(median_val, ymax * (0.97 - 0.28 * (k % 2)), f"{display} · p{fmt3(pct_val)}",
                    color=COLORS[slug], rotation=90, ha="right", va="top", fontsize=5.5)

    source_parquets = [str(path.relative_to(repo_root))]
    return _produced(fig, "f1_citywide_position", out_dir, ledger_ids, source_parquets,
                      "publishable-candidate", plotted_ids=plotted_ids)


# ---------------------------------------------------------------------------
# f2 — direct-sun reference days
# ---------------------------------------------------------------------------

def render_f2(ledger: dict, repo_root: Path, out_dir: Path) -> dict:
    paths = {slug: site_ground_parquet_path(repo_root, slug) for slug in FIGURE_SITE_ORDER}
    missing = [slug for slug, p in paths.items() if not p.exists()]
    if missing:
        return _skip("f2_direct_sun_reference_days",
                      f"ground.parquet absent for site(s): {', '.join(missing)}")

    ledger_ids: list[str] = []
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    panels = (("A", "winter_solstice", "winter solstice", axA), ("B", "equinox", "equinox", axB))
    for tag, day_key, day_label, ax in panels:
        for slug in FIGURE_SITE_ORDER:
            display = FAVELAS[slug]
            vals = _read_columns(paths[slug], [f"hours_{day_key}"]).column(0).to_numpy(zero_copy_only=False)
            vals = np.sort(vals[np.isfinite(vals)])
            frac = np.arange(1, len(vals) + 1) / len(vals)
            share_id = f"site.{slug}.ground.share_ge_2h_{day_key}"
            share_val, _ = get_value(ledger, share_id)
            ledger_ids.append(share_id)
            ax.plot(frac, vals, color=COLORS[slug], linewidth=1.0,
                    label=f"{display} (≥{fmt3(ATHENS_CHARTER_FLOOR_HOURS)} h: {fmt3(share_val)})")
        ax.axhline(ATHENS_CHARTER_FLOOR_HOURS, color="black", linewidth=0.8, linestyle=":")
        ax.text(0.01, ATHENS_CHARTER_FLOOR_HOURS, "Athens Charter (1943), Point 26",
                fontsize=5.5, va="bottom")
        ax.set_xlabel("cumulative fraction of ground cells")
        ax.set_title(tag, loc="left", fontsize=8)
        ax.set_ylabel(f"direct-sun hours, {day_label} (h)" if tag == "A" else "")
    axA.legend(fontsize=5, loc="upper left", frameon=False)
    axB.legend(fontsize=5, loc="upper left", frameon=False)

    source_parquets = [str(paths[slug].relative_to(repo_root)) for slug in FIGURE_SITE_ORDER]
    return _produced(fig, "f2_direct_sun_reference_days", out_dir, ledger_ids, source_parquets,
                      "publishable-candidate")


# ---------------------------------------------------------------------------
# f3 — domain sensitivity (ledger only)
# ---------------------------------------------------------------------------

def render_f3(ledger: dict, repo_root: Path, out_dir: Path) -> dict:
    variants = discover_grid_variants(ledger)
    if not variants:
        return _skip("f3_domain_sensitivity", "no g3.grid_*.*.svf_percentile ids in ledger")

    ledger_ids: list[str] = []
    plotted_ids: list[str] = []
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(variants))
    locked_idx = next((i for i, (_, t, d) in enumerate(variants) if (t, d) == LOCKED_VARIANT), None)

    for row, slug in enumerate(FIGURE_SITE_ORDER):
        display = FAVELAS[slug]
        ys = []
        for i, (gslug, _t, _d) in enumerate(variants):
            gid = f"g3.grid_{gslug}.{slug}.svf_percentile"
            val, _ = get_value(ledger, gid)
            (ledger_ids if i == locked_idx else plotted_ids).append(gid)
            ys.append(val)
        spread_id = f"g3.spread.{slug}.svf"
        spread_val, _ = get_value(ledger, spread_id)
        ledger_ids.append(spread_id)
        ax.plot(x, ys, marker="o", markersize=3, linewidth=1.0, color=COLORS[slug],
                label=f"{display} (spread {fmt3(spread_val)} pts)")
        if locked_idx is not None:
            ax.annotate(fmt3(ys[locked_idx]), (x[locked_idx], ys[locked_idx]), xytext=(6, -2),
                        textcoords="offset points", ha="left", fontsize=5, color=COLORS[slug])

    if locked_idx is not None:
        ax.axvline(locked_idx, color="black", linewidth=0.8, linestyle=":")
        ax.text(locked_idx, ax.get_ylim()[1], "locked domain", fontsize=5.5, ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g}/{d:g} m" for _, t, d in variants], rotation=45, ha="right")
    ax.set_xlabel("grid variant (fabric coverage threshold / footprint distance)")
    ax.set_ylabel("SVF percentile of citywide median")
    ax.legend(fontsize=5, loc="best", frameon=False)

    return _produced(fig, "f3_domain_sensitivity", out_dir, ledger_ids, [], "publishable-candidate",
                      plotted_ids=plotted_ids)


# ---------------------------------------------------------------------------
# f4 — geometry constraints (ledger only)
# ---------------------------------------------------------------------------

def render_f4(ledger: dict, repo_root: Path, out_dir: Path) -> dict:
    ledger_ids: list[str] = []
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(FIGURE_SITE_ORDER))
    bottoms = np.zeros(len(FIGURE_SITE_ORDER))
    shades = ["#E8E8E8", "#B8B8D0", "#7878A8", "#383868"]

    for k in range(4):
        vals = []
        for slug in FIGURE_SITE_ORDER:
            gid = f"wp06.{slug}.share_n{k}"
            val, _ = get_value(ledger, gid)
            ledger_ids.append(gid)
            vals.append(val)
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottoms, color=shades[k], edgecolor="white", linewidth=0.4,
               label=f"{k} constraint{'s' if k != 1 else ''}")
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(xi, b + v / 2, fmt3(v), ha="center", va="center", fontsize=5.5)
        bottoms += vals

    for xi, slug in enumerate(FIGURE_SITE_ORDER):
        nid = f"wp06.{slug}.n"
        nval, _ = get_value(ledger, nid)
        ledger_ids.append(nid)
        ax.text(xi, 1.02, f"n={fmt3(nval)}", ha="center", va="bottom", fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels([FAVELAS[s] for s in FIGURE_SITE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("share of built 10 m grid cells (fraction)")
    ax.set_ylim(0, 1.14)
    ax.legend(fontsize=5.5, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.22))

    return _produced(fig, "f4_geometry_constraints", out_dir, ledger_ids, [], "publishable-candidate")


# ---------------------------------------------------------------------------
# f5 / f5b — WP-07M: the citywide map (compute-but-withhold, red line L1).
# Spec: docs/wp07_map_spec.md. Separate manifest/run dir from f1-f4 on
# purpose: these two panels render our own aggregated data as a raster
# gradient (a matplotlib <image> element is the point, not a leak the way it
# would be for a basemap tile under a chart), so they carry their own
# checklist rather than reusing _save_and_checklist's "no_basemap" gate,
# which stays meaningful only for the chart figures above.
# ---------------------------------------------------------------------------

MAP_TARGET_MAX_PX = 1024  # a rendering choice (legible frame at screen/print
# size), not a measured number — never claims sub-meter precision.
MAP_CMAP_SVF = "cividis"      # perceptually uniform, colour-blind-safe
MAP_CMAP_KWH = "inferno"


def map_favela_boundary_path(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "RJ" / "Favelas_Limit_2019.shp"


def _load_favela_boundaries(repo_root: Path) -> dict:
    """Study-favela boundary polygons in EXPECTED_CRS, matched by display
    name. Deferred imports: wp05_full.match_favela_group is reused unmodified
    (same rule g3_domain follows) rather than re-implemented, but pulling it
    in also pulls wp05_full's torch/rasterio chain — kept out of this
    module's top-level imports so f1-f4 (and the ledger-only chart path)
    never pay that cost. Returns {} if the shapefile is absent (map figures
    skip cleanly, same discipline as f1/f2's missing-parquet skip)."""
    path = map_favela_boundary_path(repo_root)
    if not path.exists():
        return {}
    from src.config import EXPECTED_CRS
    from .wp05_full import match_favela_group

    favelas_gdf = gpd.read_file(path)
    if favelas_gdf.crs is not None:
        favelas_gdf = favelas_gdf.to_crs(EXPECTED_CRS)
    out: dict = {}
    for slug, display in FAVELAS.items():
        matched, _method = match_favela_group(favelas_gdf, display)
        if len(matched) > 0:
            out[slug] = matched
    return out


def _map_bounds(path: Path) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) of the citywide frame, from parquet row-group
    stats where available. Map-only: x/y are the point of this figure, unlike
    the chart functions above which refuse per-cell geometry columns
    entirely via _read_columns's forbidden-column assertion."""
    pf = pq.ParquetFile(path)
    out = []
    for col in ("x", "y"):
        idx = pf.schema_arrow.get_field_index(col)
        lo = hi = None
        for rg in range(pf.num_row_groups):
            stats = pf.metadata.row_group(rg).column(idx).statistics
            if stats is not None and stats.has_min_max:
                lo = stats.min if lo is None else min(lo, stats.min)
                hi = stats.max if hi is None else max(hi, stats.max)
        if lo is None or hi is None:
            arr = pf.read(columns=[col]).column(0).to_numpy(zero_copy_only=False)
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        out.extend([float(lo), float(hi)])
    return out[0], out[1], out[2], out[3]


def _streaming_pixel_mean(path: Path, value_columns: list[str], pixel_m: float,
                           bounds: tuple[float, float, float, float]):
    """Mean-per-pixel aggregation over the citywide parquet, streamed in
    batches of 500k rows — the frame's 8.4 M cells are never held at once and
    never scattered; only the (ny, nx) grid of means is returned."""
    xmin, xmax, ymin, ymax = bounds
    nx = max(1, int(np.ceil((xmax - xmin) / pixel_m)))
    ny = max(1, int(np.ceil((ymax - ymin) / pixel_m)))
    n_bins = nx * ny
    sums = {c: np.zeros(n_bins, dtype=np.float64) for c in value_columns}
    counts = {c: np.zeros(n_bins, dtype=np.int64) for c in value_columns}

    pf = pq.ParquetFile(path)
    cols = ["x", "y"] + value_columns
    for batch in pf.iter_batches(columns=cols, batch_size=500_000):
        x = batch.column(0).to_numpy(zero_copy_only=False)
        y = batch.column(1).to_numpy(zero_copy_only=False)
        col_idx = np.clip(((x - xmin) / pixel_m).astype(np.int64), 0, nx - 1)
        row_idx = np.clip(((ymax - y) / pixel_m).astype(np.int64), 0, ny - 1)
        flat = row_idx * nx + col_idx
        for i, c in enumerate(value_columns):
            vals = batch.column(2 + i).to_numpy(zero_copy_only=False)
            finite = np.isfinite(vals)
            f = flat[finite]
            counts[c] += np.bincount(f, minlength=n_bins)
            sums[c] += np.bincount(f, weights=vals[finite], minlength=n_bins)

    means = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for c in value_columns:
            m = sums[c] / counts[c]
            m[counts[c] == 0] = np.nan
            means[c] = m.reshape(ny, nx)
    return means, (nx, ny)


def _nice_scalebar_length(span_m: float) -> float:
    """Round a scalebar to a clean 1/2/5 x 10^n value near a fifth of the span."""
    target = span_m * 0.2
    if target <= 0:
        return 1.0
    exp = np.floor(np.log10(target))
    base = target / (10 ** exp)
    nice = min((1, 2, 5, 10), key=lambda n: abs(n - base))
    return float(nice * (10 ** exp))


def _plot_map_panel(ax, grid: np.ndarray, bounds: tuple[float, float, float, float],
                     boundaries: dict, cmap: str, label: str):
    xmin, xmax, ymin, ymax = bounds
    im = ax.imshow(grid, extent=(xmin, xmax, ymin, ymax), origin="upper",
                    cmap=cmap, aspect="equal", interpolation="nearest")
    for slug, gdf in boundaries.items():
        gdf.boundary.plot(ax=ax, color=COLORS[slug], linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(label, fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    return im


def _add_scalebar_north(ax, bounds: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = bounds
    span_x, span_y = xmax - xmin, ymax - ymin
    bar_m = _nice_scalebar_length(span_x)
    x0 = xmin + 0.05 * span_x
    y0 = ymin + 0.05 * span_y
    ax.plot([x0, x0 + bar_m], [y0, y0], color="black", linewidth=1.5, solid_capstyle="butt")
    if bar_m >= 1000:
        label = f"{bar_m / 1000:g} km"
    else:
        label = f"{bar_m:g} m"
    ax.text(x0 + bar_m / 2, y0, label, ha="center", va="bottom", fontsize=5.5)

    nx0 = xmax - 0.06 * span_x
    ny0 = ymin + 0.08 * span_y
    ax.annotate("N", xy=(nx0, ny0 + 0.06 * span_y), xytext=(nx0, ny0), ha="center",
                fontsize=6, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))


def _save_and_checklist_map(fig, fig_id: str, out_dir: Path) -> tuple[str, str, dict]:
    svg_path = out_dir / f"{fig_id}.svg"
    png_path = out_dir / f"{fig_id}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    raw = svg_path.read_text()
    text = _svg_text_content(raw)
    checklist = {
        "no_coordinates": _COORD_RE.search(text) is None,
        # this figure IS a raster of our own citywide-aggregated data by
        # design (that is the deliverable); a basemap-style photographic
        # tile is the thing being ruled out, and this pipeline never reads
        # one — see the module-level note above.
        "data_raster_present": "<image" in raw,
        "svg_path_count": raw.count("<path "),
        "banned_tokens_absent": not _lint._scan_lines(text.split("\n"), fig_id),
    }
    return svg_path.name, png_path.name, checklist


def _skip_map(fig_id: str, reason: str) -> dict:
    return {"id": fig_id, "status": "skipped", "reason": reason}


def _produced_map(fig, fig_id: str, out_dir: Path, source_parquets: list[str],
                   aggregation: dict, colormap: dict, boundary_note: dict) -> dict:
    svg_name, png_name, checklist = _save_and_checklist_map(fig, fig_id, out_dir)
    return {
        "id": fig_id,
        "status": "produced",
        "svg_path": svg_name,
        "png_path": png_name,
        "ledger_ids_used": [],
        "ledger_ids_plotted": [],
        "source_parquets": source_parquets,
        "source_run_of_record": RUN_OF_RECORD["wp05"],
        "release_class": "withheld",
        "red_line": "L1",
        "aggregation": aggregation,
        "colormap": colormap,
        "favela_boundaries": boundary_note,
        "checklist": checklist,
    }


def render_f5(ledger: dict, repo_root: Path, out_dir: Path) -> dict:
    path = citywide_parquet_path(repo_root)
    if not path.exists():
        return _skip_map("f5_citywide_svf_map", f"citywide parquet absent: {path}")
    boundaries = _load_favela_boundaries(repo_root)
    if not boundaries:
        return _skip_map(
            "f5_citywide_svf_map",
            f"favela boundary shapefile absent or matched none of the five study favelas: "
            f"{map_favela_boundary_path(repo_root)}",
        )

    bounds = _map_bounds(path)
    xmin, xmax, ymin, ymax = bounds
    pixel_m = max(xmax - xmin, ymax - ymin) / MAP_TARGET_MAX_PX
    means, (nx, ny) = _streaming_pixel_mean(path, ["svf", "kwh_m2"], pixel_m, bounds)
    n_cells = pq.ParquetFile(path).metadata.num_rows

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 4.4))
    panels = (
        (axA, "svf", "sky-view factor (fraction)", MAP_CMAP_SVF),
        (axB, "kwh_m2", "annual ground irradiation (kWh m$^{-2}$)", MAP_CMAP_KWH),
    )
    for tag, (ax, metric, label, cmap) in zip("AB", panels):
        _plot_map_panel(ax, means[metric], bounds, boundaries, cmap, label)
        _add_scalebar_north(ax, bounds)
        ax.set_title(tag, loc="left", fontsize=8)
    fig.suptitle(
        f"{n_cells:,} citywide ground cells · {pixel_m:.1f} m output pixel · "
        f"mean per pixel",
        fontsize=6.5,
    )

    aggregation = {
        "method": "streamed mean per output pixel (never a per-cell scatter)",
        "pixel_m": pixel_m,
        "grid_shape_rows_cols": [ny, nx],
        "target_max_px": MAP_TARGET_MAX_PX,
        "n_cells_aggregated": int(n_cells),
    }
    boundary_note = {
        "matched_favelas": sorted(boundaries),
        "missing_favelas": sorted(set(FAVELAS) - set(boundaries)),
        "source": "data/RJ/Favelas_Limit_2019.shp",
    }
    return _produced_map(fig, "f5_citywide_svf_map", out_dir, [str(path.relative_to(repo_root))],
                          aggregation, {"panel_A_svf": MAP_CMAP_SVF, "panel_B_kwh_m2": MAP_CMAP_KWH},
                          boundary_note)


def render_f5b(ledger: dict, repo_root: Path, out_dir: Path) -> dict:
    path = citywide_parquet_path(repo_root)
    if not path.exists():
        return _skip_map("f5b_citywide_svf_map_coarse", f"citywide parquet absent: {path}")
    boundaries = _load_favela_boundaries(repo_root)
    if not boundaries:
        return _skip_map(
            "f5b_citywide_svf_map_coarse",
            f"favela boundary shapefile absent or matched none of the five study favelas: "
            f"{map_favela_boundary_path(repo_root)}",
        )

    # Deferred import: g3_domain pulls in torch/rasterio for its own
    # citywide-sensitivity pass; we only need the one tuple of sensitivity
    # distances it already defends (config/params.yaml domain section, "#
    # sensitivity 5 / 20" beside the locked 10 m), never a typed literal.
    from .g3_domain import FABRIC_FOOTPRINT_DISTANCE_GRID_M

    coarse_cell_m = max(FABRIC_FOOTPRINT_DISTANCE_GRID_M)
    bounds = _map_bounds(path)
    means, (nx, ny) = _streaming_pixel_mean(path, ["svf"], coarse_cell_m, bounds)
    n_cells = pq.ParquetFile(path).metadata.num_rows

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    _plot_map_panel(ax, means["svf"], bounds, boundaries, MAP_CMAP_SVF, "sky-view factor (fraction)")
    _add_scalebar_north(ax, bounds)
    fig.suptitle(
        f"{n_cells:,} citywide ground cells · {coarse_cell_m:.0f} m cell · mean per cell",
        fontsize=6.5,
    )

    aggregation = {
        "method": "streamed mean per coarse cell (never a per-cell scatter)",
        "cell_m": coarse_cell_m,
        "cell_m_source": "src.brisa_solar.g3_domain.FABRIC_FOOTPRINT_DISTANCE_GRID_M, the coarsest "
                          "of the {5, 10, 20} m sensitivity sweep already defended for the locked "
                          "10 m domain parameter (config/params.yaml domain.fabric_footprint_distance_m)",
        "grid_shape_rows_cols": [ny, nx],
        "n_cells_aggregated": int(n_cells),
    }
    boundary_note = {
        "matched_favelas": sorted(boundaries),
        "missing_favelas": sorted(set(FAVELAS) - set(boundaries)),
    }
    result = _produced_map(fig, "f5b_citywide_svf_map_coarse", out_dir,
                            [str(path.relative_to(repo_root))], aggregation,
                            {"svf": MAP_CMAP_SVF}, boundary_note)
    result["release_class_note"] = (
        "release_class is withheld here, as for f5 — whether this coarser aggregation instead "
        "reads as reviewer-defence-only is the ethics gate's call, then the PI's; not decided by "
        "this manifest."
    )
    return result


def stage_map(repo_root: Path, out_dir: Path | None = None) -> dict:
    """WP-07M orchestration — separate run dir and manifest from stage_all's
    f1-f4 (docs/wp07_map_spec.md: 'produced into runs/wp07_map_<UTC>/'), same
    manifest shape (figures: {id: {...}}) so a generator glob extended to
    also match wp07_map_* would pick this file up unchanged."""
    repo_root = Path(repo_root)
    ledger_path = find_latest_ledger(repo_root)
    ledger = json.loads(ledger_path.read_text())

    if out_dir is None:
        out_dir = repo_root / "runs" / ("wp07_map_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "f5_citywide_svf_map": render_f5(ledger, repo_root, out_dir),
        "f5b_citywide_svf_map_coarse": render_f5b(ledger, repo_root, out_dir),
    }

    produced_pngs = [out_dir / f["png_path"] for f in figures.values() if f["status"] == "produced"]
    if produced_pngs:
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "critic_sheet.py"), "sheet",
             str(out_dir / "contact.png"), *[str(p) for p in produced_pngs],
             "--cols", "2", "--tile-w", "573"],
            check=True,
        )

    manifest = {
        "_utc": _utc_now(),
        "git_sha": _git_sha(repo_root),
        "ledger_source": str(ledger_path.relative_to(repo_root)),
        "figures": figures,
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def stage_all(repo_root: Path, out_dir: Path | None = None) -> dict:
    repo_root = Path(repo_root)
    ledger_path = find_latest_ledger(repo_root)
    ledger = json.loads(ledger_path.read_text())

    if out_dir is None:
        out_dir = repo_root / "runs" / ("wp07_figures_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "f1_citywide_position": render_f1(ledger, repo_root, out_dir),
        "f2_direct_sun_reference_days": render_f2(ledger, repo_root, out_dir),
        "f3_domain_sensitivity": render_f3(ledger, repo_root, out_dir),
        "f4_geometry_constraints": render_f4(ledger, repo_root, out_dir),
    }

    manifest = {
        "_utc": _utc_now(),
        "git_sha": _git_sha(repo_root),
        "ledger_source": str(ledger_path.relative_to(repo_root)),
        "site_order": {
            "order": list(FIGURE_SITE_ORDER),
            "source": "spec fallback order (Table H order file not found on disk)",
            "table_h_search": TABLE_H_SEARCH_NOTE,
        },
        "figures": figures,
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--target", choices=("figures", "map"), default="figures",
                     help="'figures' (default, unchanged): f1-f4 into runs/wp07_figures_<UTC>/. "
                          "'map': WP-07M f5/f5b into runs/wp07_map_<UTC>/ (docs/wp07_map_spec.md).")
    args = ap.parse_args()
    repo_root = Path(args.repo_root)
    out_dir = Path(args.out_dir) if args.out_dir else None
    stager = stage_map if args.target == "map" else stage_all
    label = "map figures" if args.target == "map" else "figures"
    manifest = stager(repo_root, out_dir)
    n_produced = sum(1 for f in manifest["figures"].values() if f["status"] == "produced")
    print(f"Staged {n_produced}/{len(manifest['figures'])} {label}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
