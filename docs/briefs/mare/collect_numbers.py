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
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("MORPHOFAVELA_ROOT", HERE.parent.parent.parent))
TR_MD = REPO_ROOT / "docs" / "technical_report" / "technical_report.md"
OUT_JSON = HERE / "mare_numbers.json"

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


def _grid_metrics(numbers: list[dict], outputs_root: Path) -> None:
    path = outputs_root / SITE / "morphometrics" / "grid" / "grid_metrics.csv"
    if not path.exists():
        raise MissingSource(f"missing {path}")
    df = pd.read_csv(path)
    rel = str(path)
    built = df[df["building_count"] > 0]

    numbers.append(_entry("mare_built_cells_n", int(len(built)), "cells", "int", rel, "count(building_count > 0)"))
    numbers.append(_entry("mare_built_cells_share_pct", 100 * len(built) / len(df), "%", "percent", rel, "count(building_count > 0) / n_cells_total"))

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


def _svf_streets(numbers: list[dict], outputs_root: Path) -> None:
    seg_path = outputs_root / SITE / "svf_v2" / "svf_streets_segments.gpkg"
    if not seg_path.exists():
        raise MissingSource(f"missing {seg_path}")
    seg = gpd.read_file(seg_path)
    numbers.append(_entry("mare_svf_street_n_segments", int(len(seg)), "segments", "int", str(seg_path), "len(gdf)"))
    numbers.append(_entry("mare_svf_street_median", float(seg["svf_median"].median()), "", "float", str(seg_path), "median(svf_median)"))

    solar_path = outputs_root / SITE / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    if not solar_path.exists():
        raise MissingSource(f"missing {solar_path}")
    pts = gpd.read_file(solar_path)
    numbers.append(_entry("mare_street_points_n", int(len(pts)), "points", "int", str(solar_path), "len(gdf)"))

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
    path = outputs_root / SITE / "paper_figures" / "diagnostic_stats.json"
    if not path.exists():
        raise MissingSource(f"missing {path}")
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


def _geometry_indicators(numbers: list[dict], outputs_root: Path) -> None:
    path = outputs_root / SITE / "geometry_indicators" / "per_patch_geometry.csv"
    if not path.exists():
        raise MissingSource(f"missing {path}")
    df = pd.read_csv(path)
    rel = str(path)
    numbers.append(_entry("mare_geom_n_cells", int(len(df)), "cells", "int", rel, "len(df)"))
    for name, col in (("vertical", "constraint_vertical"), ("lateral", "constraint_lateral"), ("directional", "constraint_directional")):
        numbers.append(_entry(f"mare_constraint_{name}_share_pct", 100 * float(df[col].mean()), "%", "percent", rel, f"100 * mean({col})"))
    vc = df["n_constraints"].value_counts(normalize=True).sort_index()
    for k in range(4):
        pct = 100 * float(vc.get(k, 0.0))
        numbers.append(_entry(f"mare_n_constraints_{k}_share_pct", pct, "%", "percent", rel, f"100 * mean(n_constraints == {k})"))
    numbers.append(_entry("mare_exposure_ratio_median", float(df["exposure_ratio"].median()), "", "float", rel, "median(exposure_ratio)"))


def _roughness_envelope(numbers: list[dict], outputs_root: Path) -> None:
    path = outputs_root / "cross_site" / "roughness" / "patch_roughness.csv"
    if not path.exists():
        raise MissingSource(f"missing {path}")
    df = pd.read_csv(path)
    d = df[df["site"] == SITE]
    rel = str(path)
    numbers.append(_entry("mare_roughness_n_patches", int(len(d)), "patches", "int", rel, "count(site == 'maré')"))
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
    _parse_tr_site_row(numbers)
    _grid_metrics(numbers, outputs_root)
    _svf_streets(numbers, outputs_root)
    _diagnostic_stats(numbers, outputs_root)
    _geometry_indicators(numbers, outputs_root)
    _roughness_envelope(numbers, outputs_root)
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
