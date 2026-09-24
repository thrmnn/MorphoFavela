"""Why Maré's citywide irradiation percentile moves between definition A (the
6 IPP favela polygons, complexo == "Maré") and definition E (the IPP
Territórios Sociais complex outline, the site study area since 2026-09-24).

PI ruling 2026-09-24 (resolved_decisions, brisaverse/shared/facts/tasks.json):
keep BOTH definitions — "an interesting result to investigate", not a choice
to make. This script decomposes the gap into three measured, disjoint
components of E's ground:

    (i)   conjuntos habitacionais       — the complex's public-housing blocks
    (ii)  other favela-layer communities — favela polygons in the complex but
                                            outside A (Parque Roquete Pinto,
                                            the Ramos remainder)
    (iii) between-communities ground     — streets, canals, open ground inside
                                            the outline but inside no named
                                            community polygon

A's own cells are matched exactly as the WP-05/07 ledger does (favela_id in
the 6 complexo=="Maré" polygons — src.brisa_solar.wp05_full.match_favela_group),
never re-derived from the community layer. The three components are then
found for every cell that is inside E but NOT inside A, using
src.sites.territory's load_territory/label_subunits (the same, already-
tested membership logic the territory map and site pages use) — so a cell
lands in exactly one of {A, (i), (ii), (iii)}, or, for the small share of A
that sits outside E's outline (a genuine boundary difference between the
2019 favela-polygon vintage and the 2026-digitized complex outline), a
separate A_outside_E residual that is reported, never folded into a bucket
it does not belong to.

Reads the newest runs/mare_definitions_*/summary.json (for the headline A/E
numbers, cross-checked against this script's own recomputation), the WP-05
run of record, data/maré/neighbourhoods.gpkg and the IPP outline. Writes a
new figure family runs/mare_definitions_explain_<UTC>/:
    decomposition.json     — every number the figures and the note read
    fig_a_waterfall.png    — cumulative percentile shift, A -> E
    fig_b_decile_shares.png — each component's own decile-share bars
    fig_c_component_map.png — the components on Maré's outline (horizontal)
    note.md                 - plain-language summary (<=250 words)
    figure_manifest.json    - lineage: derived_from the definitions run

Descriptive only: no favela-vs-formal deficit framing (red line L1), no
ranking of communities. Staged for PI review, not P1.

    python scripts/explain_mare_definitions.py [--root PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

THIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_ROOT))

from src.brisa_solar.wp05_full import favela_summary, match_favela_group  # noqa: E402
from src.brisa_solar.wp07_ledger import RUN_OF_RECORD  # noqa: E402
from src.sites.territory import (  # noqa: E402
    BETWEEN_SUBUNITS_LABEL,
    label_subunits,
    load_territory,
    rotate_for_display,
    within_mask,
)

INK, MUTED = "#1b1b1b", "#6b6b66"
COLOR = {
    "A": "#2a78d6",
    "favelas_2022": "#8a5cb0",
    "conjuntos": "#d9a441",
    "between": "#3f9e6d",
    "A_outside_E": "#b0306e",
}
BUCKET_LABEL = {
    "A": "A · 6 IPP favela polygons",
    "favelas_2022": "(ii) other favela-layer communities",
    "conjuntos": "(i) conjuntos habitacionais",
    "between": "(iii) between communities",
}
WATERFALL_ORDER = ["A", "favelas_2022", "conjuntos", "between"]  # A -> ... -> E


def _git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _newest_definitions_run(root: Path) -> Path:
    runs = sorted((root / "runs").glob("mare_definitions_2*"))
    runs = [r for r in runs if (r / "summary.json").exists()]
    if not runs:
        raise SystemExit("no runs/mare_definitions_*/summary.json found — run "
                          "scripts/mare_definition_sensitivity.py first")
    return runs[-1]


def _load_components(root: Path):
    """Returns (box, city_kwh, territory, a_polys) where `box` is the WP-05
    lattice restricted to E's bounding box with an added `bucket` column in
    {"A", "favelas_2022", "conjuntos", "between", "A_outside_E", None}."""
    t = load_territory("maré", root=root)
    df = pd.read_parquet(root / "runs" / RUN_OF_RECORD["wp05"] / "wp05_full.parquet",
                          columns=["x", "y", "favela_id", "kwh_m2"])
    city_kwh = df["kwh_m2"].to_numpy()

    fav = gpd.read_file(root / "data" / "RJ" / "Favelas_Limit_2019.shp")
    a_polys, method = match_favela_group(fav, "Maré")
    if method != "complexo_exact" or len(a_polys) != 6:
        raise SystemExit(f"definition A drifted: match method={method!r}, n={len(a_polys)} (expected complexo_exact, 6)")
    a_mask_full = df["favela_id"].isin(a_polys["cod_favela"].astype(int)).to_numpy()

    x0, y0, x1, y1 = t.study_area.bounds
    pad = 50.0
    box_sel = (df.x >= x0 - pad) & (df.x <= x1 + pad) & (df.y >= y0 - pad) & (df.y <= y1 + pad)
    box = df[box_sel].copy()
    box["in_E"] = within_mask(box.x.to_numpy(), box.y.to_numpy(), t.study_area)
    box["in_A"] = a_mask_full[box.index.to_numpy()]
    box["label"] = label_subunits(box.x.to_numpy(), box.y.to_numpy(), t)

    src_of = dict(zip(t.subunits["community"], t.subunits["source_parts"].str.split(":").str[0]))

    def _bucket(r):
        if r.in_A:
            return "A" if r.in_E else "A_outside_E"
        if not r.in_E:
            return None
        if r.label in (None, BETWEEN_SUBUNITS_LABEL):
            return "between"
        return src_of.get(r.label, "between")

    box["bucket"] = box.apply(_bucket, axis=1)
    return box, city_kwh, t, a_polys


def _decile_shares(values: np.ndarray, city_sorted: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(10)
    p = 100.0 * np.searchsorted(city_sorted, values, side="right") / len(city_sorted)
    return np.histogram(p, np.linspace(0, 100, 11))[0] / len(values) * 100


def compute_decomposition(root: Path) -> dict:
    defs_run = _newest_definitions_run(root)
    defs_summary = json.loads((defs_run / "summary.json").read_text())

    box, city_kwh, t, a_polys = _load_components(root)
    city_sorted = np.sort(city_kwh[np.isfinite(city_kwh)])

    buckets = {}
    for b in WATERFALL_ORDER + ["A_outside_E"]:
        sub = box.loc[box["bucket"] == b, "kwh_m2"].to_numpy()
        buckets[b] = {
            "label": BUCKET_LABEL.get(b, b),
            **favela_summary(sub, city_kwh),
            "decile_share": _decile_shares(sub, city_sorted).tolist(),
        }

    e_recomputed = box.loc[box["bucket"].isin(WATERFALL_ORDER), "kwh_m2"].to_numpy()
    e_stats = favela_summary(e_recomputed, city_kwh)
    e_ledger = defs_summary["definitions"]["E_ipp_complex_outline"]["kwh_m2"]
    if e_stats["n"] != e_ledger["n"]:
        raise SystemExit(f"E recomputed n={e_stats['n']} != ledger n={e_ledger['n']} ({defs_run.name}) — drift, stop")

    waterfall = []
    cum_mask = np.zeros(len(box), dtype=bool)
    prev_pct = None
    for b in WATERFALL_ORDER:
        cum_mask |= (box["bucket"] == b).to_numpy()
        cs = favela_summary(box.loc[cum_mask, "kwh_m2"].to_numpy(), city_kwh)
        waterfall.append({
            "step": b, "label": BUCKET_LABEL[b], "n_added": int((box["bucket"] == b).sum()),
            "cumulative_n": cs["n"], "cumulative_median_kwh_m2": cs["median"],
            "cumulative_percentile": cs["citywide_percentile_position"],
            "delta_percentile": None if prev_pct is None else cs["citywide_percentile_position"] - prev_pct,
        })
        prev_pct = cs["citywide_percentile_position"]

    a_full = defs_summary["definitions"]["A_ipp_complexo_mare"]["kwh_m2"]

    return {
        "_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "derived_from": defs_run.name,
        "wp05_run": RUN_OF_RECORD["wp05"],
        "display_rotation_deg": t.display_rotation_deg,
        "definition_A_full": {"n": a_full["n"], "median_kwh_m2": a_full["median"],
                               "citywide_percentile": a_full["citywide_percentile_position"]},
        "definition_E": {"n": e_stats["n"], "median_kwh_m2": e_stats["median"],
                          "citywide_percentile": e_stats["citywide_percentile_position"]},
        "buckets": buckets,
        "waterfall": waterfall,
        "deciles_note": "decile_share: % of the bucket's own cells in each citywide decile D1..D10",
        "a_outside_e_share": buckets["A_outside_E"]["n"] / a_full["n"],
    }


def render_waterfall(dec: dict, out: Path) -> Path:
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                              "xtick.color": MUTED, "ytick.color": MUTED,
                              "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=200)
        steps = dec["waterfall"]
        short = {"favelas_2022": "(ii) other\nfavela-layer", "conjuntos": "(i) conjuntos\nhabitacionais",
                 "between": "(iii) between\ncommunities"}
        labels = ["A\n(6 favela\npolygons)"] + [short[s["step"]] for s in steps[1:]] + ["E\n(complex\noutline)"]
        xs = np.arange(len(steps) + 1)
        first = steps[0]
        ax.bar(0, first["cumulative_percentile"], color=COLOR["A"], width=0.6)
        ax.text(0, first["cumulative_percentile"] + 0.6, f"{first['cumulative_percentile']:.1f}",
                ha="center", fontsize=8, color=INK)
        prev = first["cumulative_percentile"]
        for i, s in enumerate(steps[1:], start=1):
            delta = s["delta_percentile"]
            bottom = min(prev, prev + delta)
            height = abs(delta)
            col = COLOR.get(s["step"], MUTED)
            ax.bar(i, height, bottom=bottom, color=col, width=0.6)
            sign = "+" if delta >= 0 else "−"
            ax.text(i, max(prev, prev + delta) + 0.6, f"{sign}{abs(delta):.1f}",
                     ha="center", fontsize=8, color=INK)
            ax.plot([i - 1 + 0.3, i - 0.3], [prev, prev], color=MUTED, lw=0.8, ls=(0, (2, 2)))
            prev += delta
        ax.bar(len(steps), prev, color="#c9541c", width=0.6, alpha=0.9)
        ax.text(len(steps), prev + 0.6, f"{prev:.1f}", ha="center", fontsize=8, color=INK, fontweight="bold")
        ax.set_xticks(list(xs), labels, fontsize=7.5)
        ax.set_ylabel("citywide percentile of the group's median irradiation")
        ax.set_ylim(0, max(prev, first["cumulative_percentile"]) * 1.25)
        ax.set_title("Where Maré's median citywide percentile moves as the definition widens from A to E",
                     loc="left", fontsize=10, color=INK)
        ax.text(0.0, -0.22, "Each bar after A is the marginal move in the cumulative group's percentile when "
                             "that component is added; bars are not the components' own percentiles (see fig b).",
                transform=ax.transAxes, fontsize=6.5, color=MUTED, ha="left")
        fig.text(0.01, 0.01, f"WP-05 run of record {dec['wp05_run']} · descriptive only, not ranked · "
                              f"staged for PI review, not P1", fontsize=6, color=MUTED)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return out


def render_decile_shares(dec: dict, out: Path) -> Path:
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                              "xtick.color": MUTED, "ytick.color": MUTED,
                              "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=200)
        idx = np.arange(10)
        n_b = len(WATERFALL_ORDER)
        w = 0.8 / n_b
        for j, b in enumerate(WATERFALL_ORDER):
            share = np.array(dec["buckets"][b]["decile_share"])
            off = (j - (n_b - 1) / 2) * w
            n = dec["buckets"][b]["n"]
            ax.bar(idx + off, share, w * 0.92, color=COLOR[b], label=f"{BUCKET_LABEL[b]} (n={n:,})")
        ax.axhline(10, color=MUTED, lw=1, ls=(0, (3, 2)))
        ax.text(9.6, 10.6, "city = 10% in each", ha="right", fontsize=6.5, color=MUTED)
        ax.set_xticks(idx, [f"D{i + 1}" for i in idx])
        ax.set_xlabel("citywide irradiation decile  (D1 = darkest tenth of the city's ground)")
        ax.set_ylabel("share of the component's own ground cells (%)")
        ax.set_title("Each component's own spread across the city's deciles", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        fig.text(0.01, 0.01, "descriptive only, not ranked · staged for PI review, not P1", fontsize=6, color=MUTED)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return out


def render_component_map(root: Path, dec: dict, out: Path) -> Path:
    t = load_territory("maré", root=root)
    fav = gpd.read_file(root / "data" / "RJ" / "Favelas_Limit_2019.shp")
    a_polys, _ = match_favela_group(fav, "Maré")
    a_geom = a_polys.geometry.union_all()

    comm = t.subunits
    src_of = dict(zip(comm["community"], comm["source_parts"].str.split(":").str[0]))
    conjuntos_geom = comm[comm["community"].map(src_of.get) == "conjuntos"].geometry.union_all()
    favelas_geom = comm[comm["community"].map(src_of.get) == "favelas_2022"].geometry.union_all()
    other_favela_geom = favelas_geom.difference(a_geom)
    conjuntos_geom = conjuntos_geom.difference(a_geom)
    named_union = a_geom.union(other_favela_geom).union(conjuntos_geom)
    between_geom = t.study_area.difference(named_union)

    rot = t.display_rotation_deg
    layers = [
        ("A", a_geom.intersection(t.study_area)),
        ("favelas_2022", other_favela_geom.intersection(t.study_area)),
        ("conjuntos", conjuntos_geom.intersection(t.study_area)),
        ("between", between_geom),
    ]

    # A single shared rotation origin — every layer must turn about the SAME
    # point, or independent per-geometry "center" origins (rotate_for_display's
    # default) scatter them relative to each other. The study area's own
    # centroid is neutral (not any one bucket's).
    origin = (t.study_area.centroid.x, t.study_area.centroid.y)

    def _rot(geom):
        return rotate_for_display(gpd.GeoDataFrame(geometry=[geom], crs=31983), rot, origin=origin)

    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=200)
        px = 72.0 / 200
        _rot(t.study_area).boundary.plot(ax=ax, color=INK, linewidth=1.4 * px * 3, zorder=5)
        handles = []
        for key, geom in layers:
            if geom.is_empty:
                continue
            _rot(geom).plot(ax=ax, color=COLOR[key], alpha=0.8, linewidth=0, zorder=2)
            handles.append(Patch(fc=COLOR[key], label=BUCKET_LABEL[key]))
        handles.append(Patch(fc="none", ec=INK, label="study area boundary (definition E)"))

        a_out_e_geom = a_geom.difference(t.study_area)
        if not a_out_e_geom.is_empty:
            _rot(a_out_e_geom).boundary.plot(ax=ax, color=COLOR["A_outside_E"], linewidth=1.2 * px * 3,
                                              linestyle=(0, (1, 1.5)), zorder=6)
            handles.append(Patch(fc="none", ec=COLOR["A_outside_E"],
                                  label=f"A outside E — {dec['a_outside_e_share']:.0%} of A's own cells (not in the outline)"))

        ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False,
                  bbox_to_anchor=(0.0, 0.02 + 0.03 * len(handles)))
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{t.display_name} — the components of E, relative to A ({rot}° display rotation, north not up)",
                     fontsize=9.5, loc="left", color=INK)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return out


def _ordinal(n: float) -> str:
    n = int(round(n))
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


NOTE_TEMPLATE = """# Why Maré's citywide percentile differs between A and E

Definition A (the 6 IPP favela polygons under complexo "Maré", n={a_n:,} \
ground cells) sits at the {a_pct} percentile of the city's annual \
ground irradiation (median {a_med:,.0f} kWh/m²·yr). Definition E \
(the IPP Territórios Sociais complex outline, n={e_n:,}) sits at the \
{e_pct} percentile (median {e_med:,.0f} kWh/m²·yr). E adds \
{added_n:,} cells beyond A, split three ways.

**(ii) Other favela-layer communities** in the outline but outside A \
(Parque Roquete Pinto, the Ramos remainder; n={ii_n:,}) sit \
*lower* than A ({ii_pct} percentile) — adding them alone would \
pull the cumulative percentile down, not up.

**(i) Conjuntos habitacionais** — the complex's public-housing blocks \
(n={i_n:,}) sit much higher ({i_pct} percentile) and are the larger \
driver of the upward shift.

**(iii) Between-communities ground** — streets, canals and open land \
inside the outline but inside no named community (n={iii_n:,}) sits \
highest of all ({iii_pct} percentile) and is the largest single \
contributor to the gap.

So the A→E shift is not "more favela ground" — the extra favela-\
layer communities pull the other way. It comes from built forms with more \
open ground around them (conjuntos) and unbuilt ground between \
communities, both receiving more irradiation than the denser polygons in \
A. Separately, {a_out_share:.0%} of A's own cells fall outside E's outline \
— a boundary difference between the 2019 favela-polygon vintage and \
the 2026-digitized complex outline, not a component of E.

Descriptive only — no ranking, no formal/informal comparison. Staged \
for PI review, not P1.
"""


def build_note(dec: dict) -> str:
    b = dec["buckets"]
    note = NOTE_TEMPLATE.format(
        a_n=dec["definition_A_full"]["n"], a_pct=_ordinal(dec["definition_A_full"]["citywide_percentile"]),
        a_med=dec["definition_A_full"]["median_kwh_m2"],
        e_n=dec["definition_E"]["n"], e_pct=_ordinal(dec["definition_E"]["citywide_percentile"]),
        e_med=dec["definition_E"]["median_kwh_m2"],
        added_n=dec["definition_E"]["n"] - b["A"]["n"],
        ii_n=b["favelas_2022"]["n"], ii_pct=_ordinal(b["favelas_2022"]["citywide_percentile_position"]),
        i_n=b["conjuntos"]["n"], i_pct=_ordinal(b["conjuntos"]["citywide_percentile_position"]),
        iii_n=b["between"]["n"], iii_pct=_ordinal(b["between"]["citywide_percentile_position"]),
        a_out_share=dec["a_outside_e_share"],
    )
    n_words = len(note.split())
    if n_words > 250:
        raise SystemExit(f"note is {n_words} words, over the 250-word limit — trim NOTE_TEMPLATE")
    return note


def _work_packages_status(root: Path) -> str:
    wp_yaml = root / "config" / "work_packages.yaml"
    if wp_yaml.exists():
        return "present — family registration attempted"
    return ("config/work_packages.yaml does not exist in this checkout (O2 'Registry input' "
            "from docs/charter/figure_organization_spec.md §7 has not merged) — family NOT "
            "registered; register it once O2 lands.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=THIS_ROOT,
                     help="checkout to read data/ and runs/ from (main checkout when run from a worktree)")
    args = ap.parse_args()
    root = args.root.resolve()

    dec = compute_decomposition(root)

    dest = THIS_ROOT / "runs" / f"mare_definitions_explain_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    dest.mkdir(parents=True)

    (dest / "decomposition.json").write_text(json.dumps(dec, indent=2, ensure_ascii=False) + "\n")

    fig_a = render_waterfall(dec, dest / "fig_a_waterfall.png")
    fig_b = render_decile_shares(dec, dest / "fig_b_decile_shares.png")
    fig_c = render_component_map(root, dec, dest / "fig_c_component_map.png")

    note = build_note(dec)
    (dest / "note.md").write_text(note)

    manifest = {
        "_utc": dec["_utc"],
        "git_sha": _git_sha(THIS_ROOT),
        "derived_from": [dec["derived_from"]],
        "release_class": "staged",
        "p1_status": "not for P1 — descriptive investigation only",
        "figures": {
            "fig_a_waterfall": {
                "id": "fig_a_waterfall", "status": "produced",
                "png_path": fig_a.name, "derived_from": [dec["derived_from"]],
                "caption": "Cumulative citywide percentile of Maré's median irradiation, A → E.",
            },
            "fig_b_decile_shares": {
                "id": "fig_b_decile_shares", "status": "produced",
                "png_path": fig_b.name, "derived_from": [dec["derived_from"]],
                "caption": "Each component's own share of the city's irradiation deciles.",
            },
            "fig_c_component_map": {
                "id": "fig_c_component_map", "status": "produced",
                "png_path": fig_c.name, "derived_from": [dec["derived_from"]],
                "caption": "The three components mapped on Maré's outline (horizontal display rotation).",
            },
        },
        "work_packages_yaml": _work_packages_status(root),
    }
    (dest / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"A (in E)      n={dec['buckets']['A']['n']:6d}  pct={dec['buckets']['A']['citywide_percentile_position']:5.1f}")
    for b in ["favelas_2022", "conjuntos", "between"]:
        v = dec["buckets"][b]
        print(f"{BUCKET_LABEL[b]:38s} n={v['n']:6d}  pct={v['citywide_percentile_position']:5.1f}")
    print(f"A outside E   n={dec['buckets']['A_outside_E']['n']:6d}  ({dec['a_outside_e_share']:.1%} of A)")
    print(dest)
    print(manifest["work_packages_yaml"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
