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
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
              source_parquets: list[str], release_class: str) -> dict:
    svg_name, png_name, checklist = _save_and_checklist(fig, fig_id, out_dir)
    return {
        "id": fig_id,
        "status": "produced",
        "svg_path": svg_name,
        "png_path": png_name,
        "ledger_ids_used": sorted(set(ledger_ids)),
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
        ax.set_ylabel("citywide ground cells (count)")
        ax.set_title(tag, loc="left", fontsize=8)
        ymax = ax.get_ylim()[1]
        for slug in FIGURE_SITE_ORDER:
            display = FAVELAS[slug]
            median_id, pct_id = f"favela.{slug}.{metric}.median", f"favela.{slug}.{metric}.percentile"
            median_val, _ = get_value(ledger, median_id)
            pct_val, _ = get_value(ledger, pct_id)
            ledger_ids += [median_id, pct_id]
            ax.axvline(median_val, color=COLORS[slug], linewidth=1.0, linestyle="--")
            ax.text(median_val, ymax * 0.97, f"{display} · p{fmt3(pct_val)}", color=COLORS[slug],
                    rotation=90, ha="right", va="top", fontsize=5.5)

    source_parquets = [str(path.relative_to(repo_root))]
    return _produced(fig, "f1_citywide_position", out_dir, ledger_ids, source_parquets,
                      "publishable-candidate")


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
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(variants))
    locked_idx = next((i for i, (_, t, d) in enumerate(variants) if (t, d) == LOCKED_VARIANT), None)

    for row, slug in enumerate(FIGURE_SITE_ORDER):
        display = FAVELAS[slug]
        ys = []
        for gslug, _t, _d in variants:
            gid = f"g3.grid_{gslug}.{slug}.svf_percentile"
            val, _ = get_value(ledger, gid)
            ledger_ids.append(gid)
            ys.append(val)
        spread_id = f"g3.spread.{slug}.svf"
        spread_val, _ = get_value(ledger, spread_id)
        ledger_ids.append(spread_id)
        ax.plot(x, ys, marker="o", markersize=3, linewidth=1.0, color=COLORS[slug],
                label=f"{display} (spread {fmt3(spread_val)} pts)")
        for xi, yi in zip(x, ys):
            ax.annotate(fmt3(yi), (xi, yi), xytext=(0, 3 + 6 * (row % 2)),
                        textcoords="offset points", ha="center", fontsize=4,
                        color=COLORS[slug])

    if locked_idx is not None:
        ax.axvline(locked_idx, color="black", linewidth=0.8, linestyle=":")
        ax.text(locked_idx, ax.get_ylim()[1], "locked domain", fontsize=5.5, ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g}/{d:g} m" for _, t, d in variants], rotation=45, ha="right")
    ax.set_xlabel("grid variant (fabric coverage threshold / footprint distance)")
    ax.set_ylabel("SVF percentile of citywide median")
    ax.legend(fontsize=5, loc="best", frameon=False)

    return _produced(fig, "f3_domain_sensitivity", out_dir, ledger_ids, [], "publishable-candidate")


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
    ax.set_ylabel("share of buildings (fraction)")
    ax.set_ylim(0, 1.14)
    ax.legend(fontsize=5.5, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.22))

    return _produced(fig, "f4_geometry_constraints", out_dir, ledger_ids, [], "publishable-candidate")


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
    args = ap.parse_args()
    manifest = stage_all(Path(args.repo_root), Path(args.out_dir) if args.out_dir else None)
    n_produced = sum(1 for f in manifest["figures"].values() if f["status"] == "produced")
    print(f"Staged {n_produced}/{len(manifest['figures'])} figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
