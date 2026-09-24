#!/usr/bin/env python3
"""Collect every number the Maré morphology brief prints into mare_numbers.json.

Every entry is read by code from an output-of-record file (never hand-typed):
value, unit, kind (how build_brief.py should format it), source path, and the
column/expression used to compute it. Run standalone to (re)generate the
JSON, or `import collect_numbers` and call `collect(outputs_root)` from
build_brief.py / tests.

Usage: python3 docs/briefs/mare/collect_numbers.py --outputs-root <path>
"""
from __future__ import annotations

import argparse
import json
import re
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("MORPHOFAVELA_ROOT", HERE.parent.parent.parent))
TR_MD = REPO_ROOT / "docs" / "technical_report" / "technical_report.md"
OUT_JSON = HERE / "mare_numbers.json"

sys.path.insert(0, str(REPO_ROOT))
from src.brisa_solar import mare_study_area as msa  # noqa: E402
from src.sites.territory import load_territory, within_mask  # noqa: E402,F401

SITE = "maré"


class MissingSource(Exception):
    pass


def _entry(id_, value, unit, kind, source, expr):
    return {
        "id": id_,
        "value": value,
        "unit": unit,
        "kind": kind,  # "int" | "float" | "percent" | "text"
        "source": source,
        "expression": expr,
    }


def _study_area(numbers: list[dict]) -> dict:
    """Load the Maré territory (config/sites.yaml maré.study_area; loader
    src/sites/territory.py — SITETERR, 2026-09-24) and append its provenance
    numbers. Every downstream _* function filters its own source table to
    this same geometry before computing anything — never a second,
    independently-typed definition of "Maré".

    Returns a dict shaped like src.brisa_solar.mare_study_area.load_study_area
    (communities/included/excluded/data_extent/study_area) so the rest of
    this file — written against that shape — is unchanged."""
    t = load_territory("maré", root=msa.ROOT)  # msa.ROOT is the hardcoded main checkout, not REPO_ROOT
    sa = {
        "communities": t.subunits, "included": t.subunits_included,
        "excluded": t.subunits_excluded, "data_extent": t.data_extent,
        "study_area": t.study_area,
    }
    sub_prov = t.provenance["subunits"]
    rel = "data/" + sub_prov["file"]
    prov_rel = "data/" + sub_prov["provenance_file"]
    # The study area's own source/expression (not the subunits file above):
    # since 2026-09-24 (PI ruling) it is the IPP Territórios Sociais outline
    # (config/sites.yaml maré.study_area.path), not a formula over the
    # subunits — build the description from the registry rather than
    # assuming a kind.
    sa_prov = t.provenance["study_area"]
    if "path" in sa_prov:
        sa_source, sa_expr = "data/" + sa_prov["path"], "union(features in config/sites.yaml maré.study_area.path)"
    else:
        sa_source, sa_expr = rel, "union(included communities).intersection(data_extent).area"

    numbers.append(_entry("mare_study_area_km2", sa["study_area"].area / 1e6, "km²", "float",
                           sa_source, sa_expr))
    numbers.append(_entry("mare_study_area_n_communities_total", len(sa["communities"]), "communities", "int",
                           rel, "len(communities)"))
    numbers.append(_entry("mare_study_area_n_communities_included", len(sa["included"]), "communities", "int",
                           rel, "len(communities[in_study_area])"))
    numbers.append(_entry("mare_study_area_n_communities_excluded", len(sa["excluded"]), "communities", "int",
                           rel, "len(communities[~in_study_area])"))
    excluded_name = ", ".join(sorted(sa["excluded"]["community"])) if len(sa["excluded"]) else "none"
    numbers.append(_entry("mare_study_area_excluded_name", excluded_name, "", "text",
                           rel, "communities[~in_study_area].community"))
    numbers.append(_entry("mare_study_area_n_inferred_matches", int(sub_prov["provenance"]["qa"]["n_inferred"]),
                           "communities", "int", prov_rel, "qa.n_inferred"))
    return sa


def _parse_tr_site_row(numbers: list[dict]) -> None:
    """Parse the Maré row of technical_report.md §1's site table (not retyped)."""
    if not TR_MD.exists():
        raise MissingSource(f"missing {TR_MD}")
    text = TR_MD.read_text()
    m = re.search(
        r"^\|\s*Maré\s*\|\s*([\d.]+)\s*km²\s*\|\s*(\w+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d.]+)\s*m\s*\|\s*([\d.]+)\s*h\s*\|",
        text,
        re.MULTILINE,
    )
    if not m:
        raise MissingSource("Maré site-table row not found in technical_report.md §1")
    area_km2, typology, buildings_ext, cells_10m, mean_h, annual_sun = m.groups()
    src = f"{TR_MD.relative_to(REPO_ROOT)} §1 site table, Maré row"
    numbers.append(_entry("mare_area_km2", float(area_km2), "km²", "float", src, "table column 'Area'"))
    numbers.append(_entry("mare_typology", typology, "", "text", src, "table column 'Type'"))
    numbers.append(_entry("mare_buildings_extended", int(buildings_ext.replace(",", "")), "buildings", "int", src, "table column 'Buildings (extended)'"))
    numbers.append(_entry("mare_cells_10m", int(cells_10m.replace(",", "")), "cells", "int", src, "table column '10 m cells'"))
    numbers.append(_entry("mare_mean_H_m", float(mean_h), "m", "float", src, "table column 'Mean H'"))
    numbers.append(_entry("mare_annual_mean_sun_h", float(annual_sun), "h", "float", src, "table column 'Annual mean sun'"))
    # Pipeline constant (10 m grid resolution), stated in TR §4.1 prose, project-wide.
    m2 = re.search(r"A regular (\d+) m grid", text)
    if not m2:
        raise MissingSource("grid resolution sentence not found in technical_report.md §4.1")
    numbers.append(_entry(
        "mare_grid_resolution_m", int(m2.group(1)), "m", "int",
        f"{TR_MD.relative_to(REPO_ROOT)} §4.1", "pipeline constant, all sites",
    ))
    m3 = re.search(r"Digital terrain model\*\* \(\.tif\) at (\d+) m resolution", text)
    if not m3:
        raise MissingSource("DTM resolution sentence not found in technical_report.md §2.1")
    numbers.append(_entry(
        "mare_dtm_resolution_m", int(m3.group(1)), "m", "int",
        f"{TR_MD.relative_to(REPO_ROOT)} §2.1", "pipeline constant, all sites",
    ))


def _grid_metrics(numbers: list[dict], outputs_root: Path, sa: dict) -> None:
    path = outputs_root / SITE / "morphometrics" / "grid" / "grid_metrics.csv"
    if not path.exists():
        raise MissingSource(f"missing {path}")
    df = pd.read_csv(path)
    rel = str(path)
    # Study-area filter: every cell/statistic below counts only within the
    # active study area (the IPP Territórios Sociais outline — communities
    # plus the ground between them), never the whole bairro (MAREBOUND).
    # Marcílio Dias contributes nothing here — it lies entirely outside
    # both the outline and the site data extent (grid_metrics.csv has no
    # cells there to begin with).
    in_area = msa.within_mask(df["centroid_x"].to_numpy(), df["centroid_y"].to_numpy(), sa["study_area"])
    df = df.loc[in_area].reset_index(drop=True)
    built = df[df["building_count"] > 0]

    # Overrides technical_report.md's whole-bairro cell count (§1 site
    # table) with the study-area count — collect() runs _grid_metrics()
    # after _parse_tr_site_row(), and by_id collapses on id, so this is the
    # value that actually reaches the template.
    numbers.append(_entry("mare_cells_10m", int(len(df)), "cells", "int", rel, "count(cells within the study area)"))
    numbers.append(_entry("mare_built_cells_n", int(len(built)), "cells", "int", rel, "count(building_count > 0)"))
    numbers.append(_entry("mare_built_cells_share_pct", 100 * len(built) / len(df) if len(df) else 0.0, "%", "percent", rel, "count(building_count > 0) / n_cells_total"))

    def add_median_iqr(prefix, col, unit, subset):
        numbers.append(_entry(f"mare_{prefix}_median", float(subset[col].median()), unit, "float", rel, f"median({col})"))
        numbers.append(_entry(f"mare_{prefix}_iqr_lo", float(subset[col].quantile(0.25)), unit, "float", rel, f"{col}.quantile(0.25)"))
        numbers.append(_entry(f"mare_{prefix}_iqr_hi", float(subset[col].quantile(0.75)), unit, "float", rel, f"{col}.quantile(0.75)"))

    # λp and SVF: median over ALL grid cells (unbuilt cells are λp=0, SVF measured
    # from open passageways) — this is the basis technical_report.md §4.3 uses
    # (verified: matches the published 0.43 / 0.53 medians for Maré).
    add_median_iqr("lambda_p", "lambda_p", "", df)
    add_median_iqr("svf_grid", "svf", "", df)
    # H_mean, porosity, sigma_h are defined on built cells only (NaN elsewhere).
    add_median_iqr("H_mean", "H_mean", "m", built)
    add_median_iqr("porosity", "porosity", "", built)
    add_median_iqr("sigma_h", "sigma_h", "m", built)


def _svf_streets(numbers: list[dict], outputs_root: Path, sa: dict) -> None:
    seg_path = outputs_root / SITE / "svf_v2" / "svf_streets_segments.gpkg"
    if not seg_path.exists():
        raise MissingSource(f"missing {seg_path}")
    seg = gpd.read_file(seg_path)
    seg_c = seg.geometry.centroid
    seg = seg.loc[msa.within_mask(seg_c.x.to_numpy(), seg_c.y.to_numpy(), sa["study_area"])].reset_index(drop=True)
    numbers.append(_entry("mare_svf_street_n_segments", int(len(seg)), "segments", "int", str(seg_path), "len(gdf), study area (segment centroid)"))
    numbers.append(_entry("mare_svf_street_median", float(seg["svf_median"].median()), "", "float", str(seg_path), "median(svf_median)"))

    solar_path = outputs_root / SITE / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    if not solar_path.exists():
        raise MissingSource(f"missing {solar_path}")
    pts = gpd.read_file(solar_path)
    pts = pts.loc[msa.within_mask(pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy(), sa["study_area"])].reset_index(drop=True)
    numbers.append(_entry("mare_street_points_n", int(len(pts)), "points", "int", str(solar_path), "len(gdf), study area"))

    winter_mean = float(pts["solar_hours_winter"].mean())
    annual_mean = float(pts["solar_hours_annual"].mean())
    summer_mean = float(pts["solar_hours_summer"].mean())
    numbers.append(_entry("mare_sun_winter_mean_h", winter_mean, "h", "float", str(solar_path), "mean(solar_hours_winter)"))
    numbers.append(_entry("mare_sun_annual_mean_h", annual_mean, "h", "float", str(solar_path), "mean(solar_hours_annual)"))
    numbers.append(_entry("mare_sun_summer_mean_h", summer_mean, "h", "float", str(solar_path), "mean(solar_hours_summer)"))
    numbers.append(_entry("mare_sun_seasonal_range_h", summer_mean - winter_mean, "h", "float", str(solar_path), "mean(solar_hours_summer) - mean(solar_hours_winter)"))

    for season, col in (("winter", "solar_hours_winter"), ("annual", "solar_hours_annual"), ("summer", "solar_hours_summer")):
        share_below = 100 * float((pts[col] < 2.0).mean())
        numbers.append(_entry(f"mare_share_below_2h_{season}_pct", share_below, "%", "percent", str(solar_path), f"100 * mean({col} < 2.0)"))


def _diagnostic_stats(numbers: list[dict], outputs_root: Path) -> None:
    # study-area variant, written by `python scripts/build_diagnostic_map.py
    # --site maré --study-area` (never the default diagnostic_stats.json,
    # which technical_report.md/the manuscript figures/the project hub read
    # unfiltered — MAREBOUND must not change what they see).
    path = outputs_root / SITE / "paper_figures" / "diagnostic_stats_study_area.json"
    if not path.exists():
        raise MissingSource(f"missing {path} — run: python scripts/build_diagnostic_map.py --site maré --study-area")
    d = json.loads(path.read_text())
    rel = str(path)
    numbers.append(_entry("mare_diag_n_total", int(d["n_cells_total"]), "cells", "int", rel, "n_cells_total"))
    numbers.append(_entry("mare_diag_n_classified", int(d["n_cells_classified"]), "cells", "int", rel, "n_cells_classified"))
    numbers.append(_entry("mare_diag_sun_floor_h", float(d["thresholds"]["sun_hours_winter_min"]), "h", "float", rel, "thresholds.sun_hours_winter_min"))
    numbers.append(_entry("mare_diag_enclosure_threshold", float(d["thresholds"]["lambda_f_max"]), "", "float", rel, "thresholds.lambda_f_max"))
    label_map = {
        "adequate": "mare_diag_share_unconstrained_pct",
        "sunlight_constraint": "mare_diag_share_sun_only_pct",
        "ventilation_constraint": "mare_diag_share_enclosure_only_pct",
        "compound_constraint": "mare_diag_share_both_pct",
    }
    for key, out_id in label_map.items():
        numbers.append(_entry(out_id, 100 * float(d["shares"][key]), "%", "percent", rel, f"100 * shares.{key}"))
    nodata_share = 100 * d["counts"]["nodata"] / d["n_cells_total"]
    numbers.append(_entry("mare_diag_share_nodata_pct", nodata_share, "%", "percent", rel, "100 * counts.nodata / n_cells_total"))


def _geometry_indicators(numbers: list[dict], outputs_root: Path, sa: dict) -> None:
    path = outputs_root / SITE / "geometry_indicators" / "per_patch_geometry.csv"
    if not path.exists():
        raise MissingSource(f"missing {path}")
    df = pd.read_csv(path)
    rel = str(path)
    in_area = msa.within_mask(df["center_x"].to_numpy(), df["center_y"].to_numpy(), sa["study_area"])
    df = df.loc[in_area].reset_index(drop=True)
    numbers.append(_entry("mare_geom_n_cells", int(len(df)), "cells", "int", rel, "len(df), study area"))
    for name, col in (("vertical", "constraint_vertical"), ("lateral", "constraint_lateral"), ("directional", "constraint_directional")):
        numbers.append(_entry(f"mare_constraint_{name}_share_pct", 100 * float(df[col].mean()), "%", "percent", rel, f"100 * mean({col})"))
    vc = df["n_constraints"].value_counts(normalize=True).sort_index()
    for k in range(4):
        pct = 100 * float(vc.get(k, 0.0))
        numbers.append(_entry(f"mare_n_constraints_{k}_share_pct", pct, "%", "percent", rel, f"100 * mean(n_constraints == {k})"))
    numbers.append(_entry("mare_exposure_ratio_median", float(df["exposure_ratio"].median()), "", "float", rel, "median(exposure_ratio)"))


def _roughness_envelope(numbers: list[dict], outputs_root: Path, sa: dict) -> None:
    path = outputs_root / "cross_site" / "roughness" / "patch_roughness.csv"
    if not path.exists():
        raise MissingSource(f"missing {path}")
    df = pd.read_csv(path)
    d = df[df["site"] == SITE]
    rel = str(path)

    # patch_roughness.csv carries patch_id but no coordinates; its own
    # geometry lives in the CFD campaign-sampling patch layer, keyed by the
    # same patch_id.
    patches_path = outputs_root / SITE / "sampling_cfd" / "campaign_sampling" / "campaign_patches.gpkg"
    if not patches_path.exists():
        raise MissingSource(f"missing {patches_path}")
    patches = gpd.read_file(patches_path)
    in_area = msa.within_mask(patches["center_x"].to_numpy(), patches["center_y"].to_numpy(), sa["study_area"])
    ids_in_area = set(patches.loc[in_area, "patch_id"])
    d = d[d["patch_id"].isin(ids_in_area)]
    numbers.append(_entry("mare_roughness_n_patches", int(len(d)), "patches", "int", rel, "count(site == 'maré'), study area (via campaign_patches.gpkg patch_id)"))
    numbers.append(_entry("mare_roughness_out_of_envelope_share_pct", 100 * float(d["flag_pai_over_envelope"].mean()), "%", "percent", rel, "100 * mean(flag_pai_over_envelope)"))
    numbers.append(_entry("mare_roughness_floored_share_pct", 100 * float(d["flag_z0_floored"].mean()), "%", "percent", rel, "100 * mean(flag_z0_floored)"))


def _wind_rose(numbers: list[dict], outputs_root: Path) -> None:
    data_root = outputs_root.parent / "data"
    path = data_root / SITE / "wind_rose.json"
    if not path.exists():
        raise MissingSource(f"missing {path}")
    d = json.loads(path.read_text())
    rel = str(path)
    freqs = d["frequencies"]
    dom_sector = max(freqs, key=freqs.get)
    numbers.append(_entry("mare_wind_dominant_sector", dom_sector, "", "text", rel, "argmax(frequencies)"))
    numbers.append(_entry("mare_wind_dominant_freq_pct", 100 * float(freqs[dom_sector]), "%", "percent", rel, "100 * frequencies[argmax]"))
    numbers.append(_entry("mare_wind_calm_share_pct", 100 * float(d["calm_fraction"]), "%", "percent", rel, "100 * calm_fraction"))
    numbers.append(_entry("mare_wind_n_obs", int(d["n_observations"]), "observations", "int", rel, "n_observations"))
    numbers.append(_entry("mare_wind_station", d["station_name"], "", "text", rel, "station_name"))
    numbers.append(_entry("mare_wind_year_start", d["time_window_start"][:4], "", "text", rel, "time_window_start[:4]"))
    numbers.append(_entry("mare_wind_year_end", d["time_window_end"][:4], "", "text", rel, "time_window_end[:4]"))


def _composition(numbers: list[dict], outputs_root: Path) -> None:
    # NOT filtered to the study area (MAREBOUND scope note): the fabric-
    # cluster labels behind this table are only retained as a per-site
    # aggregate (outputs/cross_site/signature/composition_by_site.csv) — the
    # per-cell cluster assignment the clustering pipeline used to produce it
    # is not written to outputs/ anywhere, so there is no cell-level source
    # of record to re-filter here. This still describes the whole bairro's
    # fabric, not the study area (the IPP outline). Recomputing a study-area
    # version would mean re-running the cross-site clustering fit restricted
    # to Maré's study area, which changes methodology (the fit is shared
    # across all 5 campaign sites) — out of MAREBOUND's scope; flagged in
    # the PI report rather than done silently.
    path = outputs_root / "cross_site" / "signature" / "composition_by_site.csv"
    if not path.exists():
        raise MissingSource(f"missing {path}")
    df = pd.read_csv(path, index_col=0)
    if SITE not in df.index:
        raise MissingSource(f"'{SITE}' row not found in {path}")
    row = df.loc[SITE]
    rel = str(path)
    for t in range(6):
        col = str(t)
        if col not in row.index:
            continue
        numbers.append(_entry(f"mare_morphotype_T{t}_pct", 100 * float(row[col]), "%", "percent", rel, f"100 * row['{t}']"))


def collect(outputs_root: Path) -> list[dict]:
    numbers: list[dict] = []
    sa = _study_area(numbers)
    _parse_tr_site_row(numbers)
    _grid_metrics(numbers, outputs_root, sa)
    _svf_streets(numbers, outputs_root, sa)
    _diagnostic_stats(numbers, outputs_root)
    _geometry_indicators(numbers, outputs_root, sa)
    _roughness_envelope(numbers, outputs_root, sa)
    _wind_rose(numbers, outputs_root)
    _composition(numbers, outputs_root)
    return numbers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, required=True)
    args = ap.parse_args()

    try:
        numbers = collect(args.outputs_root)
    except MissingSource as e:
        print(f"collect_numbers: SKIP — {e}")
        return 0

    by_id = {n["id"]: n for n in numbers}
    OUT_JSON.write_text(json.dumps(by_id, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"collect_numbers: wrote {len(by_id)} entries -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
