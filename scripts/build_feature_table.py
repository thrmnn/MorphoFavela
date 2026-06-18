"""Build the WS-0 feature substrate for the morpho-signature track.

For each site, emit two *linked* tables under ``outputs/{site}/features/``:

- ``features_grid.parquet``   — fabric (areal, cell-native) + support-aware
                                street summaries merged on ``zone_id``.
- ``features_street.parquet`` — experience (per observer) enriched with the
                                fabric covariates of its cell.

Plus ``run_meta.json`` (producer git sha) per site and one
``outputs/features_dictionary.md``. Sites are discovered by globbing rather
than a name registry, so the accented ``maré`` dir and the calibration sites
(grid + SVF only, no canyon/openness) are handled automatically.

    python scripts/build_feature_table.py            # all sites
    python scripts/build_feature_table.py --site vidigal
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # win over the stale brisa-0.1.0 editable install

from src.morphometry.features import (  # noqa: E402
    build_experience_table,
    build_fabric_table,
    summarize_streets_to_cells,
)
from src.svf_v2.io import _git_sha  # noqa: E402


def _read(path: Path) -> gpd.GeoDataFrame | None:
    return gpd.read_file(path) if path.exists() else None


def discover_sites(outputs_root: Path) -> list[Path]:
    return sorted(
        p.parents[2]
        for p in outputs_root.glob("*/morphometrics/grid/grid_metrics.gpkg")
    )


def build_site(site_dir: Path) -> dict:
    site = site_dir.name
    m = site_dir / "morphometrics"
    grid = gpd.read_file(m / "grid" / "grid_metrics.gpkg")
    svf_solar = gpd.read_file(m / "svf" / "svf_streets_solar.gpkg")
    openness = _read(m / "svf" / "ventilation_openness_streets.gpkg")
    hw = _read(m / "canyon" / "hw_streets.gpkg")

    fabric = build_fabric_table(grid)
    experience = build_experience_table(svf_solar, grid, openness=openness, hw=hw)
    summary = summarize_streets_to_cells(experience, grid)
    features_grid = fabric.merge(summary, on="zone_id", how="left")

    out = site_dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    features_grid.to_parquet(out / "features_grid.parquet")
    experience.to_parquet(out / "features_street.parquet")

    meta = {
        "site": site,
        "git_sha": _git_sha(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_cells": int(len(features_grid)),
        "n_observers": int(len(experience)),
        "cells_with_street_support": int(features_grid["has_street_support"].sum()),
        "has_openness": openness is not None,
        "has_hw": hw is not None,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def write_dictionary(outputs_root: Path) -> None:
    text = """# Feature substrate — data dictionary

Two linked tables per site under `outputs/{site}/features/`. See
`docs/morpho_signature_plan.md` for the change-of-support rationale.

## `features_grid.parquet` — FABRIC (support level: areal, 10 m cell)

Cell-native morphometrics only. The street-aggregated `svf`/`svf_count` from
`grid_metrics.gpkg` are **excluded** here on purpose (outcomes, not fabric).

| column | meaning |
|--------|---------|
| zone_id | cell id (join key) |
| lambda_p, far, sigma_h, H_mean, building_count | density / massing |
| lambda_f_{N..NW}, lambda_f_mean, lambda_f_max | frontal-area density, 8 azimuths |
| porosity, slope_deg, aspect_deg | void fraction, terrain |
| street_orientation_entropy | street-bearing Shannon entropy |
| northness, eastness | cos/sin(aspect) — terrain-aware encoding |
| built_mask | lambda_p>0.01 or building_count>0 |

### support-aware street summaries (merged from the experience table)

| column | meaning |
|--------|---------|
| n_street_obs | observer count in cell (the support; 0 if none) |
| has_street_support | bool — False ⇒ all summaries NULL, NOT interpolated |
| svf_p50, svf_p10 | median + worst-decile sky view |
| solar_winter_p50, solar_winter_p10 | median + worst-decile winter sun hours |
| solar_winter_frac_below2h | fraction of observers below the WHO 2 h floor |
| frac_deep_canyon | fraction in DEEP_CANYON openness class (campaign sites) |
| hw_p50 | median canyon H/W (campaign sites) |

## `features_street.parquet` — EXPERIENCE (support level: observer point)

Per pedestrian sample point: svf, solar_hours_* / irradiance_*, openness_class,
HW (+ has_hw), each row carrying `zone_id` and the fabric covariates
(lambda_p, lambda_f_mean, porosity, slope_deg, aspect_deg, H_mean, sigma_h) of
its containing cell via point-in-polygon lookup.
"""
    (outputs_root / "features_dictionary.md").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", help="single site dir name (default: all discovered)")
    ap.add_argument("--outputs-root", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    sites = discover_sites(outputs_root)
    if args.site:
        sites = [s for s in sites if s.name == args.site]
        if not sites:
            raise SystemExit(f"site {args.site!r} not found under {outputs_root}")

    for site_dir in sites:
        meta = build_site(site_dir)
        print(
            f"{meta['site']:20s} cells={meta['n_cells']:6d} "
            f"obs={meta['n_observers']:6d} "
            f"support={meta['cells_with_street_support']:6d} "
            f"(openness={meta['has_openness']}, hw={meta['has_hw']})"
        )
    write_dictionary(outputs_root)
    print(f"\nwrote {len(sites)} site(s) + features_dictionary.md")


if __name__ == "__main__":
    main()
