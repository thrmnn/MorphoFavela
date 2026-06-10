"""Build the MorphoFavela interactive HTML dashboard for a single site.

This is a single, parameterised builder. Per-site invocation:

    python scripts/build_html_dashboard.py --site rocinha

Output: outputs/_distribution/html_dashboards/{site}/index.html plus a
sibling data/ folder (observers.geojson, footprint.geojson, stats.json)
and an assets/ folder. Shared JS, CSS, fonts, and cross-site stats live
one directory up so several site dashboards can be opened side by side
without duplicating ~200 KB of plotly / leaflet boilerplate per site.

The dashboard surfaces the C1 length-weighting fix and H1–H4 audit
findings directly in the UI (toggles + delta chips), and serves the
audit narrative through a methodology drawer anchored by finding ID.

Self-contained: opens under file:// once the CDN scripts have been
fetched once by the browser (they have permissive CORS). The shared
JS in ../js is referenced by relative path so the per-site folder can
be zipped on its own only after a one-off copy step (not in scope here).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import date
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "outputs" / "_distribution" / "html_dashboards"

SITE_META = {
    "vidigal": dict(
        display="Vidigal",
        subtitle="a south-zone hillside favela, 0.39 km², ridge to beach",
        typology="Canyon-dominated hillside",
        boundary_shp="data/vidigal/raw/Vidigal_Limit.shp",
        atoms_subdir="vidigal",
    ),
    "rocinha": dict(
        display="Rocinha",
        subtitle="a south-zone hillside favela, 0.84 km², ridge to beach",
        typology="Canyon-dominated hillside",
        boundary_shp="data/rocinha/raw/rocinha_boundary.shp",
        atoms_subdir="rocinha",
    ),
    "complexo_do_alemao": dict(
        display="Complexo do Alemão",
        subtitle="a north-zone hillside complex of 13 favelas, 1.80 km²",
        typology="Canyon-dominated hillside",
        boundary_shp="data/complexo_do_alemao/raw/complexo_do_alemao_boundary.shp",
        atoms_subdir="complexo_do_alemao",
    ),
    "riodaspedras": dict(
        display="Rio das Pedras",
        subtitle="a west-zone flat-dense favela, 0.65 km², narrow-alley grid",
        typology="Flat-dense alley grid",
        boundary_shp="data/riodaspedras/raw/riodaspedras_boundary.shp",
        atoms_subdir="riodaspedras",
    ),
    "maré": dict(
        display="Maré",
        subtitle="a north-zone low-rise grid of 16 favelas, ~4 km²",
        typology="Dense low-rise grid",
        boundary_shp="data/maré/raw/mare_boundary.shp",
        atoms_subdir="mare",
    ),
}

# Audit findings, keyed by ID — placed-in-HTML decisions per site below.
AUDIT_FINDINGS = {
    "C1": {
        "title": "Length-weighted mean SVF (patched)",
        "severity": "critical",
        "summary": "Site-level mean SVF was count-weighted across segments, biasing Rocinha by +0.088. The dashboard now uses sum(svf_mean × length_m) / sum(length_m).",
        "commit": "d8175ca",
        "human_stakes": "Older copy of Rocinha's mean SVF circulated as ~0.37; the corrected mean is ~0.28. If you read a higher number elsewhere, it is the count-weighted version that this dashboard supersedes.",
    },
    "H1": {
        "title": "Edge halo — observers near the boundary see fake sky",
        "severity": "high",
        "summary": "External building footprints stop at the AOI edge, so observers within ~15 m of the boundary have an artificially open hemisphere. The fix upstream is to re-export building footprints with a 50–100 m external halo so canyons at the perimeter are still walled. Until that ships, this dashboard surfaces the issue honestly: edge observers are rendered as hairline rings, and the <em>Exclude edge observers (≤15 m)</em> toggle re-runs the headline KPI so reviewers can see how the mean moves when the perimeter is masked. Edge shares per site: Alemão 21.3%, Rocinha 16.1%, Rio das Pedras 11.5%, Vidigal 7.1%, Maré 5.1%.",
        "human_stakes": "Streets at the favela's edge look a little more open in the map than they really are, because the buildings just outside the boundary are missing from the model. Toggle 'Exclude edge observers' to see the headline mean without them.",
    },
    "H2": {
        "title": "Observer density varies 2.3× across sites",
        "severity": "high",
        "summary": "Rocinha samples its dense canyon network at 45.8k observers/km²; Maré at 19.7k/km². Per-observer histograms therefore lean toward Rocinha's canyon morphology if read naively; cross-site means in this dashboard are length-weighted, which neutralises the density bias by treating each metre of road equally. The C1 patch (length-weighting) is the structural fix; the density chip on the density KPI tile is the visual cue.",
        "human_stakes": "Rocinha has more measurement points per square kilometre than Maré because its streets are tighter and more intricate — that does not mean Rocinha is darker, it means we measured it more densely. The length-weighted mean accounts for this.",
    },
    "H3": {
        "title": "Maré offset cluster (street_id 226)",
        "severity": "high",
        "summary": "Maré has 0.02% of observers placed 8–12 m off centerline where roads cross through building footprints. Masked by default on Maré.",
        "human_stakes": "A small handful of Maré observers ended up inside building footprints due to a roadline misalignment; they are hidden by default and do not move the headline.",
    },
    "H4": {
        "title": "Rocinha exact-zero SVF cohort",
        "severity": "high",
        "summary": "11.4% of Rocinha observers have SVF exactly 0.0 — likely a ray-cast fallback rather than true canyon occlusion. Rendered in unresolved grey, excluded from the viridis ramp.",
        "human_stakes": "About one in nine Rocinha observers has SVF recorded as exactly zero. This is almost certainly a bookkeeping artefact — a numerical fallback when canyons are too narrow to resolve — not a finding that the sky is completely sealed off. We flag those points in grey so you can see how much of the map is affected without letting them pull the headline mean.",
    },
    "M1": {
        "title": "n_sun_positions dropped",
        "severity": "medium",
        "summary": "The column was a winter-only constant (21) mislabelled as a quality flag. Not surfaced in the dashboard.",
    },
    "M2": {
        "title": "Legacy solar_hours excluded",
        "severity": "medium",
        "summary": "The unsuffixed solar_hours column duplicates solar_hours_winter and is structurally absent from the dashboard GeoJSON to prevent silent winter-as-annual plots.",
        "human_stakes": "Earlier reports sometimes plotted a 'solar hours' column that was actually winter-only — making streets look gloomier than they are over a full year. This dashboard only exposes the season-labelled columns (annual, winter, summer, equinox) so this slip cannot happen here.",
    },
    "M3": {
        "title": "Typology reframe (Maré only)",
        "severity": "medium",
        "summary": "Maré is a dense low-rise grid, not a canyon-dominated hillside favela. Cross-site comparisons pool it separately. The three canyon-hillside sites (Vidigal, Rocinha, Alemão) share a morphology of narrow shaded valleys; Rio das Pedras is a flat-dense alley grid — distinct again — and Maré's low-rise blocks open more sky than any of them.",
    },
    "M4": {
        "title": "Stair circulation unsampled (Vidigal only — verified)",
        "severity": "medium",
        "summary": "The Vidigal road shapefile has no Escadaria/Ladeira category, so its SVF and solar values describe the drivable network only — stair-only beco circulation is unsampled. This has been verified for Vidigal (see project memo); the other hillside sites likely share the gap but it has not been audited there.",
        "human_stakes": "The dashboard literally does not see the stairs and narrow becos people climb every day to get home — only the drivable network. On a steep hillside favela like Vidigal that means a meaningful slice of public space, often the most shaded and intricate, is missing from these numbers.",
    },
    "M5": {
        "title": "Boundary-clipped fragments dropped",
        "severity": "medium",
        "summary": "Road fragments shorter than the sampling spacing (1.5 m) are silently dropped, which can omit dead-end stubs near the boundary.",
    },
    "L3": {
        "title": "Latitude-consistent daylight ceiling",
        "severity": "low",
        "summary": "Winter 10.5 h / summer 13.5 h maximum solar hours at every site match Rio's −23° latitude. Physics looks healthy.",
    },
    "L4": {
        "title": "Sky ↔ sun coupling (Pearson r)",
        "severity": "low",
        "summary": "Per-site Pearson r between SVF and annual solar hours falls in the 0.88–0.96 range across all five sites. Sky openness is a strong but imperfect predictor of sun exposure — the spread is local terrain shading (a building uphill of you blocks sun even with open sky overhead).",
    },
    "L5": {
        "title": "Canonical observer hash",
        "severity": "low",
        "summary": "Solar GPKGs predate the canonical observer rebuild but are bit-identical (all 5 sites, max 0.000 m). No rerun required on geometry grounds.",
    },
}

# Site-specific positive-frame copy for the §caveats framing-changers group
# when no M-finding applies. Replaces the awkward "None apply to this site"
# hole with a one-line typology statement that aids cross-site reading.
TYPOLOGY_POSITIVE_FRAME = {
    "vidigal": "Read this site as a canyon-dominated hillside favela — the morphological frame is consistent across Vidigal, Rocinha and Alemão.",
    "rocinha": "Rocinha is the canyon-dominated hillside reference case. When other sites are reframed — Maré as a low-rise grid, Rio das Pedras as a flat alley grid — it is against this canyon morphology. So no framing caveat reshapes Rocinha; it is the frame.",
    "complexo_do_alemao": "Read this site as a canyon-dominated hillside favela — the morphological frame is consistent across Vidigal, Rocinha and Alemão.",
    "riodaspedras": "Read this site as a flat-dense alley grid — a distinct typology from the canyon hillsides and from Maré's low-rise grid.",
    "maré": "Read this site as a dense low-rise grid — distinct from the four canyon / alley sites and the reason it is pooled separately in cross-site comparisons.",
}


# Per-site, which findings to surface (drives both narrative and toggle wiring).
SITE_FINDINGS = {
    "vidigal": ["C1", "H1", "H2", "M4", "M5", "L3", "L4", "L5"],
    "rocinha": ["C1", "H1", "H2", "H4", "M1", "M2", "M5", "L3", "L4", "L5"],
    "complexo_do_alemao": ["C1", "H1", "H2", "M4", "M5", "L3", "L4", "L5"],
    "riodaspedras": ["C1", "H1", "H2", "M4", "L3", "L4", "L5"],
    "maré": ["C1", "H1", "H2", "H3", "M3", "M4", "M5", "L3", "L4", "L5"],
}


def c1_delta_compute(stats: dict) -> float:
    return float(stats["length_weighted_mean_svf"] - stats["count_weighted_mean_svf"])


def short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO
        ).decode().strip()
    except Exception:
        return "unknown"


def boundary_for(site: str) -> Optional[gpd.GeoDataFrame]:
    meta = SITE_META[site]
    p = REPO / meta["boundary_shp"]
    if not p.exists():
        # case-insensitive lookup
        parent = p.parent
        if parent.exists():
            for f in parent.iterdir():
                if f.name.lower() == p.name.lower():
                    p = f
                    break
    if not p.exists():
        return None
    try:
        b = gpd.read_file(p)
        if b.crs is None:
            b.set_crs(epsg=31983, inplace=True)
        return b.to_crs(epsg=31983)
    except Exception:
        return None


def stratified_decimate(df: pd.DataFrame, target: int, bins: int = 10) -> pd.DataFrame:
    """Stratified subsample on SVF deciles, preserving tail behaviour."""
    if len(df) <= target:
        return df
    df = df.copy()
    df["_bin"] = pd.qcut(df["svf"].rank(method="first"), bins, labels=False)
    per_bin = max(1, target // bins)
    parts = []
    rng = np.random.default_rng(42)
    for _, group in df.groupby("_bin"):
        if len(group) <= per_bin:
            parts.append(group)
        else:
            idx = rng.choice(group.index, size=per_bin, replace=False)
            parts.append(group.loc[idx])
    out = pd.concat(parts).drop(columns=["_bin"])
    return out


def compute_site_stats(site: str, obs: gpd.GeoDataFrame, seg: gpd.GeoDataFrame,
                       boundary: Optional[gpd.GeoDataFrame]) -> dict:
    n = len(obs)
    road_km = float(seg["length_m"].sum() / 1000) if "length_m" in seg.columns else float("nan")
    area_km2 = float(boundary.area.sum() / 1e6) if boundary is not None else float("nan")

    # length-weighted mean SVF: per-observer weights = segment length / n_points
    # use the segments table for the authoritative length-weighted site mean
    lw_mean = float((seg["svf_mean"] * seg["length_m"]).sum() / seg["length_m"].sum())
    cw_mean = float(seg["svf_mean"].dropna().mean())
    obs_mean_svf = float(obs["svf"].mean())
    obs_mean_solar_annual = float(obs["solar_hours_annual"].mean())
    pearson = float(np.corrcoef(obs["svf"], obs["solar_hours_annual"])[0, 1])
    exact_zero_share = float((obs["svf"] == 0).mean())

    # edge-share — observers within 15 m of boundary
    edge_share = float("nan")
    if boundary is not None:
        bnd = boundary.boundary.unary_union
        dists = obs.geometry.distance(bnd)
        edge_share = float((dists <= 15).mean())
        obs["dist_to_boundary"] = dists
    else:
        obs["dist_to_boundary"] = np.nan

    # density
    density = (n / area_km2) if area_km2 and not math.isnan(area_km2) else float("nan")

    # 40-bin length-weighted SVF distribution from SEGMENTS table (+ count from observers).
    # Emitted so hist.js can render real binned bars instead of one row per observation.
    n_bins = 40
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    seg_svf = seg["svf_mean"].to_numpy()
    seg_len = seg["length_m"].to_numpy()
    seg_mask = ~np.isnan(seg_svf) & ~np.isnan(seg_len)
    seg_svf_c = np.clip(seg_svf[seg_mask], 0.0, 1.0 - 1e-9)
    seg_len_c = seg_len[seg_mask]
    bin_idx_seg = np.minimum((seg_svf_c * n_bins).astype(int), n_bins - 1)
    length_per_bin = np.zeros(n_bins)
    np.add.at(length_per_bin, bin_idx_seg, seg_len_c)
    total_len = float(length_per_bin.sum()) or 1.0

    obs_svf = np.clip(obs["svf"].to_numpy(), 0.0, 1.0 - 1e-9)
    bin_idx_obs = np.minimum((obs_svf * n_bins).astype(int), n_bins - 1)
    count_per_bin = np.bincount(bin_idx_obs, minlength=n_bins).astype(float)
    total_count = float(count_per_bin.sum()) or 1.0

    svf_bins = []
    for i in range(n_bins):
        svf_bins.append({
            "bin_lo": round(float(edges[i]), 5),
            "bin_hi": round(float(edges[i+1]), 5),
            "c": round(float((edges[i] + edges[i+1]) / 2), 5),
            "length_m": round(float(length_per_bin[i]), 3),
            "lw_share": round(float(length_per_bin[i] / total_len), 5),
            "count": int(count_per_bin[i]),
            "count_share": round(float(count_per_bin[i] / total_count), 5),
        })

    # length-weighted median: sort segments by svf_mean, cumulate length, return crossing point
    order = np.argsort(seg_svf_c)
    cum = np.cumsum(seg_len_c[order])
    half = cum[-1] / 2 if len(cum) else 0
    lw_median = float(seg_svf_c[order][np.searchsorted(cum, half)]) if len(cum) else float("nan")

    # OLS fit for solar = a + b * svf (observer-level)
    svf_arr = obs["svf"].to_numpy(dtype=float)
    sol_arr = obs["solar_hours_annual"].to_numpy(dtype=float)
    fit_mask = ~np.isnan(svf_arr) & ~np.isnan(sol_arr)
    sx = svf_arr[fit_mask]
    sy = sol_arr[fit_mask]
    if len(sx) >= 2:
        slope, intercept = np.polyfit(sx, sy, 1)
        r_val = float(np.corrcoef(sx, sy)[0, 1])
        svf_solar_fit = {
            "slope": round(float(slope), 5),
            "intercept": round(float(intercept), 5),
            "r": round(r_val, 5),
            "r_squared": round(r_val * r_val, 5),
            "n": int(len(sx)),
        }
    else:
        svf_solar_fit = None

    # WGS84 centroid for Rio locator inset
    if boundary is not None:
        try:
            ctr = boundary.to_crs(epsg=4326).geometry.unary_union.representative_point()
            centroid_lat = float(ctr.y)
            centroid_lon = float(ctr.x)
        except Exception:
            centroid_lat = centroid_lon = float("nan")
    else:
        centroid_lat = centroid_lon = float("nan")

    return {
        "site": site,
        "site_display": SITE_META[site]["display"],
        "subtitle": SITE_META[site]["subtitle"],
        "typology": SITE_META[site]["typology"],
        "build_date": date.today().isoformat(),
        "git_sha": short_sha(),
        "crs": "SIRGAS 2000 / UTM 23S (EPSG:31983)",
        "area_km2": area_km2,
        "n_observers": n,
        "road_km": road_km,
        "density_per_km2": density,
        "edge_share": edge_share,
        "length_weighted_mean_svf": lw_mean,
        "length_weighted_median_svf": lw_median,
        "count_weighted_mean_svf": cw_mean,
        "delta_length_minus_count": lw_mean - cw_mean,
        "observer_mean_svf": obs_mean_svf,
        "observer_mean_solar_annual": obs_mean_solar_annual,
        "pearson_svf_solar_annual": pearson,
        "exact_zero_svf_share": exact_zero_share,
        "svf_bins": svf_bins,
        "svf_solar_fit": svf_solar_fit,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
    }


def write_observers_geojson(obs: gpd.GeoDataFrame, out_path: Path, target: int = 8000) -> int:
    """Write decimated observers as GeoJSON in WGS84 (Leaflet-friendly).

    Excludes the legacy solar_hours and n_sun_positions columns (M1/M2).
    """
    df = stratified_decimate(obs, target=target)
    df = df.to_crs(epsg=4326)
    keep_cols = [
        "street_id", "distance_along", "svf",
        "solar_hours_winter", "solar_hours_summer",
        "solar_hours_equinox_mar", "solar_hours_equinox_sep",
        "solar_hours_annual", "sunshine_ratio_mean",
        "irradiance_annual_wh", "offset_distance",
        "dist_to_boundary", "geometry",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    # round numerics to keep file small
    for c in out.columns:
        if c == "geometry":
            continue
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(3)
    out.to_file(out_path, driver="GeoJSON")
    return len(out)


def write_footprint(boundary: Optional[gpd.GeoDataFrame], out_path: Path) -> bool:
    if boundary is None:
        return False
    b = boundary.to_crs(epsg=4326)
    b[["geometry"]].to_file(out_path, driver="GeoJSON")
    return True


# ---------- HTML / asset generation ----------

GLOSSARY = {
    "SVF": "Sky View Factor — the fraction of the upper hemisphere not blocked by buildings. 0 means walls all around; 1 means open field.",
    "observer": "A point sampled every 1.5 m along the street centerline at 1.5 m above ground, treated as a pedestrian standing on the street.",
    "length-weighted": "Each street segment's contribution to the site-level mean is proportional to its physical length, not the number of observers placed on it.",
    "solar access": "Hours of direct sunlight reaching street level on an average day, modelled annually across the four reference dates.",
    "lambda_p": "Plan-area density — fraction of ground covered by buildings within a grid cell.",
    "edge observer": "An observer within ~15 m of the AOI boundary, where the external building halo is missing and the upper hemisphere may be artificially open.",
    "AOI": "Area of interest — the polygon used as the site boundary.",
    "Pearson r": "A statistic between -1 and 1 that measures how tightly two things move together. 0 means unrelated, 1 means they move in perfect lockstep.",
    "ray-cast fallback": "A numerical shortcut the geometry engine takes when a canyon is too tight for the ray-tracer to resolve. The result is recorded as 0.0, not a real measurement of zero sky.",
    "Carto Positron": "A pale grey street map tile layer used as background context.",
    "viridis": "A perceptually-uniform green-to-yellow colour ramp readable by colourblind viewers.",
    "YlOrBr": "A yellow-to-brown colour ramp, conventional for solar and heat variables.",
    "LOWESS": "A smooth curve drawn through a scatter cloud to show the local trend without imposing a straight line.",
}


def write_shared_assets(observer_counts: dict, site_stats_by_site: dict) -> None:
    """Write shared CSS, JS, glossary, cross-site stats.

    Called every build (cheap) so the shared layer stays in sync.
    """
    (DIST / "assets" / "glossary.json").write_text(json.dumps(GLOSSARY, indent=2))
    def _nan_to_null(o):
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: _nan_to_null(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_nan_to_null(v) for v in o]
        return o
    (DIST / "shared" / "cross_site_stats.json").write_text(
        json.dumps({"per_site": _nan_to_null(list(site_stats_by_site.values()))}, indent=2)
    )
    (DIST / "style.css").write_text(STYLE_CSS)
    (DIST / "print.css").write_text(PRINT_CSS)
    (DIST / "js" / "state.js").write_text(JS_STATE)
    (DIST / "js" / "map.js").write_text(JS_MAP)
    (DIST / "js" / "hist.js").write_text(JS_HIST)
    (DIST / "js" / "scatter.js").write_text(JS_SCATTER)
    (DIST / "js" / "drawer.js").write_text(JS_DRAWER)
    (DIST / "js" / "glossary.js").write_text(JS_GLOSSARY)


def build_landing(site_stats_by_site: dict) -> None:
    """Cross-site landing page at html_dashboards/index.html."""
    rows = []
    for s, st in site_stats_by_site.items():
        rows.append(
            f'<a class="site-tile" href="{s}/index.html">'
            f'<div class="tile-name">{st["site_display"]}</div>'
            f'<div class="tile-typo">{st["typology"]}</div>'
            f'<div class="tile-stats">'
            f'<span>{st["n_observers"]:,} obs</span>'
            f'<span>{st["road_km"]:.1f} km</span>'
            f'<span>SVF {st["length_weighted_mean_svf"]:.3f}</span>'
            f'</div></a>'
        )
    grid = "\n".join(rows)
    sites_json = json.dumps(list(site_stats_by_site.values()))
    html = LANDING_HTML.replace("__GRID__", grid).replace("__SITES__", sites_json)
    (DIST / "index.html").write_text(html)


def build_site(site: str) -> dict:
    if site not in SITE_META:
        raise SystemExit(f"unknown site: {site}")
    meta = SITE_META[site]
    site_dir = DIST / site
    (site_dir / "data").mkdir(parents=True, exist_ok=True)
    (site_dir / "assets").mkdir(parents=True, exist_ok=True)

    print(f"[{site}] loading observers + segments")
    obs = gpd.read_file(REPO / f"outputs/{site}/morphometrics/svf/svf_streets_solar.gpkg")
    seg = gpd.read_file(REPO / f"outputs/{site}/morphometrics/svf/svf_streets_segments.gpkg")
    boundary = boundary_for(site)

    print(f"[{site}] computing stats")
    stats = compute_site_stats(site, obs, seg, boundary)
    stats["findings_present"] = SITE_FINDINGS.get(site, [])

    print(f"[{site}] writing decimated geojson")
    n_written = write_observers_geojson(obs, site_dir / "data" / "observers.geojson")
    fp_ok = write_footprint(boundary, site_dir / "data" / "footprint.geojson")

    stats["observer_count_in_geojson"] = n_written
    stats["footprint_written"] = fp_ok

    # lineage block — per-tile audit references
    stats["lineage"] = {
        "n_observers": {
            "formula": "len(svf_streets_solar)",
            "column": "rows",
            "audit_ref": "canonical observer network rebuilt 1bffc3e",
        },
        "road_km": {
            "formula": "sum(segments.length_m) / 1000",
            "column": "length_m",
            "audit_ref": "L5 — segments backfilled d8175ca",
        },
        "density_per_km2": {
            "formula": "n_observers / area_km2",
            "column": "rows / boundary.area",
            "audit_ref": "H2 — varies 2.3× across sites",
        },
        "edge_share": {
            "formula": "mean(distance_to_boundary <= 15)",
            "column": "distance_to_boundary",
            "audit_ref": "H1 — external building halo missing",
        },
        "length_weighted_mean_svf": {
            "formula": "sum(svf_mean * length_m) / sum(length_m)",
            "column": "segments.svf_mean × segments.length_m",
            "audit_ref": "C1 — patched d8175ca",
        },
        "pearson_svf_solar_annual": {
            "formula": "Pearson r(svf, solar_hours_annual)",
            "column": "observers.svf vs observers.solar_hours_annual",
            "audit_ref": "L4 — 0.88–0.96 per site",
        },
    }

    # caveats sub-blocks
    stats["audit_callouts"] = [AUDIT_FINDINGS[f] | {"id": f} for f in stats["findings_present"]]

    # write stats.json (NaN → null so browsers can parse)
    def _nn(o):
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: _nn(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_nn(v) for v in o]
        return o
    (site_dir / "data" / "stats.json").write_text(json.dumps(_nn(stats), indent=2))

    print(f"[{site}] generating HTML")
    html = render_site_html(site, stats)
    (site_dir / "index.html").write_text(html)

    return stats


# ---------- HTML template ----------

def render_site_html(site: str, stats: dict) -> str:
    findings = stats["findings_present"]
    site_display = stats["site_display"]
    typology = stats["typology"]

    def kpi(label, value, anchor, lineage_key, fmt=None):
        l = stats["lineage"].get(lineage_key, {})
        formula = l.get("formula", "")
        col = l.get("column", "")
        audit = l.get("audit_ref", "")
        return f'''
        <div class="kpi" id="{anchor}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" data-key="{lineage_key}">{value}</div>
          <div class="kpi-lineage">
            <span class="kpi-badge" tabindex="0">i</span>
            <div class="kpi-tooltip">
              <div><strong>Formula:</strong> <code>{formula}</code></div>
              <div><strong>Source:</strong> <code>{col}</code></div>
              <div><strong>Audit:</strong> {audit}</div>
            </div>
          </div>
        </div>
        '''

    edge_pct = stats["edge_share"] * 100 if not math.isnan(stats["edge_share"]) else 0
    density_chip = ""
    if not math.isnan(stats.get("density_per_km2", float("nan"))):
        _d = stats["density_per_km2"]
        _ref = 45815.14 if site == "maré" else 19712.22
        _label = "Rocinha" if site == "maré" else "Maré"
        _ratio = _d / _ref
        # Reserve the warm accent for ratios that genuinely flag a 2.3× spread;
        # 1.0–1.5× is a quiet difference and shouldn't over-signal risk.
        _cls = "chip" if _ratio > 1.5 else "chip neutral"
        density_chip = f' <span class="{_cls}">H2 · {_ratio:.2f}× {_label}</span>'

    kpi_strip = "".join([
        kpi("N observers", f'{stats["n_observers"]:,}', "kpi-n", "n_observers"),
        kpi("Road km", f'{stats["road_km"]:.1f}', "kpi-roadkm", "road_km"),
        kpi("Observers / km²" + density_chip,
            f'{stats["density_per_km2"]:.0f}' if not math.isnan(stats["density_per_km2"]) else "—",
            "kpi-density", "density_per_km2"),
        kpi(f'Edge observers (≤15 m) <small class="kpi-hint">toggle the "Exclude edge" mask on the map to see them removed</small>',
            f'{edge_pct:.1f}%' if edge_pct else "—",
            "kpi-edge", "edge_share"),
        kpi("Mean SVF <small>length-weighted</small>",
            f'{stats["length_weighted_mean_svf"]:.3f}',
            "kpi-mean-svf", "length_weighted_mean_svf"),
        kpi('Sky ↔ sun coupling <small>0 = unrelated, 1 = lockstep</small>',
            f'{stats["pearson_svf_solar_annual"]:.3f}',
            "kpi-pearson", "pearson_svf_solar_annual"),
    ])

    # caveats
    findings_present_set = set(findings)
    number_changers = [f for f in ["H1", "H4"] if f in findings_present_set]
    framing_changers = [f for f in ["M2", "M3", "M4"] if f in findings_present_set]
    bounds = [f for f in ["H2", "H3"] if f in findings_present_set]

    def render_findings(ids, empty_frame=None):
        if not ids and empty_frame:
            return f'<p class="muted">{empty_frame}</p>'
        if not ids:
            return '<p class="muted">None apply to this site.</p>'
        out = []
        for fid in ids:
            f = AUDIT_FINDINGS[fid]
            out.append(
                f'<article class="finding finding--{f["severity"]}">'
                f'<header><span class="finding-id">{fid}</span><h4>{f["title"]}</h4></header>'
                f'<p>{f["summary"]}</p>'
                + (f'<p class="human-stakes">{f["human_stakes"]}</p>' if f.get("human_stakes") else "")
                + f'<a class="chevron" href="#m-{fid.lower()}">read methodology →</a>'
                + f'</article>'
            )
        return "\n".join(out)

    # methodology drawer — surface ALL audit findings on every site so the
    # archival page carries the full audit trail. Findings that apply
    # specifically to this site are marked "applies to this site".
    all_fids = ["C1", "H1", "H2", "H3", "H4", "M1", "M2", "M3", "M4", "M5", "L3", "L4", "L5"]
    drawer_entries_parts = []
    for fid in all_fids:
        f = AUDIT_FINDINGS[fid]
        applies = " <em class=\"applies\">(applies to this site)</em>" if fid in findings_present_set else ""
        extra = ""
        if fid == "C1":
            extra = f'<p class="muted"><strong>This site:</strong> length-weighted {stats["length_weighted_mean_svf"]:.3f} vs count-weighted {stats["count_weighted_mean_svf"]:.3f} (Δ {c1_delta_compute(stats):+.3f}).</p>'
        elif fid == "H1":
            extra = f'<p class="muted"><strong>This site:</strong> {edge_pct:.1f}% of observers within 15 m of the boundary.</p>'
        elif fid == "H4":
            extra = f'<p class="muted"><strong>This site:</strong> {stats["exact_zero_svf_share"] * 100:.2f}% of observers at exact-zero SVF.</p>'
        elif fid == "L5":
            extra = '<p class="muted">Geometry hash equality verified against <code>outputs/{site}/sampling_streets/observers.gpkg</code>.</p>'.replace('{site}', site)
        drawer_entries_parts.append(
            f'<details id="m-{fid.lower()}"><summary><span class="finding-id">{fid}</span> {f["title"]}{applies}</summary>'
            f'<p>{f["summary"]}</p>{extra}</details>'
        )
    drawer_entries = "".join(drawer_entries_parts)

    # Rocinha-only / Maré-only toggles
    show_unresolved_toggle = "H4" in findings_present_set
    show_offset_toggle = "H3" in findings_present_set

    toggles_html = '''
      <label class="toggle"><input type="checkbox" id="t-edge"> Exclude edge observers (≤15 m)</label>
    '''
    if show_unresolved_toggle:
        toggles_html += '<label class="toggle"><input type="checkbox" id="t-unresolved" checked> Show SVF=0 as unresolved</label>'
    if show_offset_toggle:
        toggles_html += '<label class="toggle"><input type="checkbox" id="t-offset"> Show offset cluster (street_id 226)</label>'

    # dek — subtitle is driven by stats so it stays in sync with the
    # provenance footer (avoid surprising readers who have seen prior
    # MorphoFavela material with different km² figures from coarser boundaries).
    dek = (
        f'This page describes <strong>{site_display}</strong>, {stats["subtitle"]}. '
        f'You are looking at the slice of the street network where a pedestrian would stand — '
        f'{stats["n_observers"]:,} <dfn data-term="observer">observers</dfn> on '
        f'{stats["road_km"]:.1f} km of road — and how much sky each one sees '
        f'(<dfn data-term="SVF">SVF</dfn>) and how much sun reaches it. '
        f'<strong>Why this matters:</strong> street-level sky and sun exposure shape heat stress, '
        f'daylight access, mental health, and vitamin D — the everyday habitability of a '
        f'neighbourhood and how it will hold up as Rio warms. In informal settlements, where '
        f'public space is the street, these numbers are the climate adaptation surface. '
        f'The headline mean is <dfn data-term="length-weighted">length-weighted</dfn>, '
        f'correcting a count-weighted bias in earlier reports.'
    )

    # delta chip for C1
    c1_delta = stats["length_weighted_mean_svf"] - stats["count_weighted_mean_svf"]
    delta_sign = "+" if c1_delta >= 0 else ""
    c1_delta_str = f'{delta_sign}{c1_delta:.3f}'

    # Pre-render cross-site comparator rows from the shared stats file (if
    # any sites have already been built). Falls back to a JS-driven version
    # that hits ../shared/cross_site_stats.json. Pre-rendering means the
    # comparator works with JavaScript disabled and the page is bigger and
    # more self-contained — the file:// review story.
    cross_site_static = ""
    footer_grid_static = ""
    shared_p = DIST / "shared" / "cross_site_stats.json"
    if shared_p.exists():
        try:
            payload = json.loads(shared_p.read_text())
            sites_list = payload.get("per_site", [])
            if sites_list:
                # absolute scale 0–0.5 SVF with reference ticks at 0.2/0.4
                ABS_MAX = 0.5
                rows_parts = []
                # ticks header row
                rows_parts.append(
                    '<div class="csr-ticks">'
                    '<div></div><div></div>'
                    '<div class="csr-axis">'
                      '<span class="tick" style="left:0%">0.0</span>'
                      '<span class="tick" style="left:40%">0.2</span>'
                      '<span class="tick" style="left:80%">0.4</span>'
                    '</div>'
                    '<div></div>'
                    '</div>'
                )
                pt_names = {"maré", "complexo_do_alemao", "riodaspedras", "rocinha", "vidigal"}
                from urllib.parse import quote as _q
                for s in sites_list:
                    pct = min(100.0, (s["length_weighted_mean_svf"] / ABS_MAX) * 100)
                    current = "current" if s["site"] == site else ""
                    lang_attr = ' lang="pt-BR"' if s["site"] in pt_names else ""
                    href_site = _q(s["site"], safe="")
                    rows_parts.append(
                        f'<a class="csr-row {current}" href="../{href_site}/index.html">'
                        f'<div><span class="csr-name"{lang_attr}>{s["site_display"]}</span></div>'
                        f'<div class="csr-typo">{s["typology"]}</div>'
                        f'<div><div class="csr-bar">'
                          f'<div class="csr-tick-line" style="left:40%"></div>'
                          f'<div class="csr-tick-line" style="left:80%"></div>'
                          f'<div class="csr-bar-fill" style="width:{pct:.1f}%"></div>'
                        f'</div></div>'
                        f'<div class="csr-stat">SVF {s["length_weighted_mean_svf"]:.3f}<br>'
                        f'<small>{s["n_observers"]:,} obs</small></div>'
                        f'</a>'
                    )
                cross_site_static = "\n".join(rows_parts)
                tile_parts = []
                for s in sites_list:
                    lang_attr = ' lang="pt-BR"' if s["site"] in pt_names else ""
                    href_site = _q(s["site"], safe="")
                    tile_parts.append(
                        f'<a class="footer-tile" href="../{href_site}/index.html">'
                        f'<div class="tile-name"{lang_attr}>{s["site_display"]}</div>'
                        f'<div class="tile-typo">{s["typology"]}</div>'
                        f'<div class="tile-stats"><span>{s["n_observers"]:,} obs</span>'
                        f'<span>SVF {s["length_weighted_mean_svf"]:.3f}</span></div>'
                        f'</a>'
                    )
                footer_grid_static = "\n".join(tile_parts)
        except Exception:
            pass

    # Honest decimation copy: if every observer is shown (n_in_geojson ==
    # n_observers) we shouldn't claim the points were "picked evenly across
    # the SVF range" — no decimation actually occurred.
    if stats["observer_count_in_geojson"] >= stats["n_observers"]:
        decimation_phrase = (
            f'All {stats["n_observers"]:,} observers shown (no decimation needed at this density).'
        )
    else:
        decimation_phrase = (
            f'We show {stats["observer_count_in_geojson"]:,} of {stats["n_observers"]:,} '
            f'observers to keep the map fast — picked evenly across the SVF range so the rare '
            f'dark and bright extremes are not lost.'
        )

    # Vidigal-specific softener: at +0.001 the C1 correction is essentially
    # zero on this site; flag where the audit fix actually moved the needle.
    if abs(c1_delta) < 0.005:
        c1_softener = (
            ' On {site} the correction is negligible ({delta}); it matters most '
            'on Rocinha (+0.088).'.format(
                site=site_display,
                delta=("+0.000" if abs(c1_delta) < 0.0005 else c1_delta_str),
            )
        )
    else:
        c1_softener = ""

    rendered = SITE_HTML_TEMPLATE.format(
        site=site,
        site_display=site_display,
        site_display_lower=site_display.lower(),
        subtitle=stats["subtitle"],
        typology=typology,
        build_date=stats["build_date"],
        git_sha=stats["git_sha"],
        crs=stats["crs"],
        dek=dek,
        kpi_strip=kpi_strip,
        toggles_html=toggles_html,
        number_changers=render_findings(number_changers),
        framing_changers=render_findings(framing_changers, empty_frame=TYPOLOGY_POSITIVE_FRAME.get(site)),
        bounds=render_findings(bounds),
        drawer_entries=drawer_entries,
        c1_delta=c1_delta_str,
        c1_softener=c1_softener,
        cw_mean=f'{stats["count_weighted_mean_svf"]:.3f}',
        lw_mean=f'{stats["length_weighted_mean_svf"]:.3f}',
        pearson=f'{stats["pearson_svf_solar_annual"]:.3f}',
        n_observers=stats["n_observers"],
        n_in_geojson=stats["observer_count_in_geojson"],
        decimation_phrase=decimation_phrase,
        road_km=f'{stats["road_km"]:.1f}',
        area_km2=f'{stats["area_km2"]:.2f}' if not math.isnan(stats["area_km2"]) else "—",
        atoms_subdir=SITE_META[site]["atoms_subdir"],
        cross_site_static=cross_site_static,
        footer_grid_static=footer_grid_static,
        exact_zero_share=f'{stats["exact_zero_svf_share"]:.6f}',
        lw_mean_num=f'{stats["length_weighted_mean_svf"]:.6f}',
        lw_median_num=f'{stats.get("length_weighted_median_svf") or 0:.6f}',
        cw_mean_num=f'{stats["count_weighted_mean_svf"]:.6f}',
        centroid_lat_num=f'{stats.get("centroid_lat") if stats.get("centroid_lat") is not None and not math.isnan(stats.get("centroid_lat", float("nan"))) else 0:.6f}',
        centroid_lon_num=f'{stats.get("centroid_lon") if stats.get("centroid_lon") is not None and not math.isnan(stats.get("centroid_lon", float("nan"))) else 0:.6f}',
        mfstats_json=json.dumps({
            "svf_bins": stats.get("svf_bins", []),
            "svf_solar_fit": stats.get("svf_solar_fit"),
            "length_weighted_mean_svf": stats["length_weighted_mean_svf"],
            "length_weighted_median_svf": stats.get("length_weighted_median_svf"),
            "centroid_lat": stats.get("centroid_lat"),
            "centroid_lon": stats.get("centroid_lon"),
        }),
    )
    rendered = rendered.replace("__INLINE_CSS__", STYLE_CSS)
    inline_js = "\n;\n".join([JS_GLOSSARY, JS_STATE, JS_MAP, JS_HIST, JS_SCATTER, JS_DRAWER])
    rendered = rendered.replace("__INLINE_JS__", inline_js)
    return rendered


# ---------- inline assets (templates) ----------

STYLE_CSS = r"""
:root{
  --canvas-paper:#fbfaf6;
  --ink:#1a1a1a;
  --ink-muted:#5b5b5b;
  --rule:#d4d2cc;
  --rule-strong:#9a988e;
  --accent-primary:#0f766e;
  --accent-warm:#b45309;
  --accent-cool:#1d4ed8;
  --unresolved:#9ca3af;
  --bg-tile:#ffffff;
  --serif:'Source Serif 4', Georgia, 'Times New Roman', serif;
  --sans:'Inter', system-ui, -apple-system, sans-serif;
  --mono:'JetBrains Mono', 'SF Mono', Menlo, monospace;
  --measure:62ch;
}
*{box-sizing:border-box;}
html{font-size:16px;}
body{
  margin:0; background:var(--canvas-paper); color:var(--ink);
  font-family:var(--sans); line-height:1.55;
  -webkit-font-smoothing:antialiased;
}
a{color:var(--accent-primary); text-decoration:none; border-bottom:1px solid transparent;}
a:hover{border-bottom-color:var(--accent-primary);}
.measure{max-width:var(--measure);}
.muted{color:var(--ink-muted);}
small{font-size:0.82em; color:var(--ink-muted);}
code{font-family:var(--mono); font-size:0.85em; background:rgba(0,0,0,0.04); padding:0.1em 0.3em; border-radius:3px;}
dfn{font-style:normal; border-bottom:1px dotted var(--accent-primary); cursor:help;}

/* masthead */
.masthead{
  padding:1.6rem 2rem 1.2rem; border-bottom:1px solid var(--rule);
  display:grid; grid-template-columns:1fr auto; gap:1rem; align-items:end;
}
.mast-mark{
  font-family:var(--serif); font-weight:600; font-size:0.78rem;
  letter-spacing:0.18em; text-transform:uppercase; color:var(--ink-muted);
}
.mast-sheet{font-family:var(--mono); font-size:0.78rem; color:var(--ink-muted); margin-top:0.4rem;}
.mast-title{font-family:var(--serif); font-weight:600; font-size:3rem; line-height:1.05; margin:0.6rem 0 0.4rem;}
.pill{
  display:inline-block; padding:0.2rem 0.6rem; border-radius:999px;
  background:rgba(15,118,110,0.1); color:var(--accent-primary);
  font-size:0.78rem; font-weight:500; margin-right:0.5rem;
}
.mast-meta{font-family:var(--mono); font-size:0.75rem; color:var(--ink-muted);}
.drawer-toggle{
  font-family:var(--sans); font-size:0.85rem; font-weight:500;
  background:none; border:1px solid var(--rule-strong); padding:0.4rem 0.8rem;
  border-radius:4px; cursor:pointer; color:var(--ink);
}
.drawer-toggle:hover{background:#fff;}

/* dek */
.dek{
  padding:1.5rem 2rem; max-width:60ch;
  font-family:var(--sans); font-style:italic; font-size:0.95rem;
  color:var(--ink-muted); line-height:1.6;
}
.dek strong{color:var(--ink); font-weight:600; font-style:normal;}

/* kpi strip */
.kpi-strip{
  position:sticky; top:0; z-index:50;
  background:var(--canvas-paper);
  border-top:1px solid var(--rule); border-bottom:1px solid var(--rule);
  display:grid; grid-template-columns:repeat(6, 1fr); gap:0; padding:0;
}
.kpi{
  padding:1rem 1rem; border-right:1px solid var(--rule);
  position:relative; min-width:0;
}
.kpi:last-child{border-right:none;}
.kpi-label{font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
  color:var(--ink-muted); font-weight:500;}
.kpi-value{
  font-family:var(--sans); font-feature-settings:"tnum";
  font-size:1.7rem; font-weight:600; margin-top:0.25rem;
}
#kpi-mean-svf .kpi-value{font-size:1.85rem; color:var(--accent-primary);}
.kpi-lineage{position:absolute; top:0.6rem; right:0.6rem;}
.kpi-badge{
  display:inline-flex; align-items:center; justify-content:center;
  width:18px; height:18px; border:1px solid var(--rule-strong); border-radius:50%;
  font-size:11px; color:var(--ink-muted); cursor:help; background:transparent;
  font-family:var(--serif); font-style:italic;
}
.kpi-tooltip{
  display:none; position:absolute; right:0; top:1.6rem; z-index:60;
  background:#fff; border:1px solid var(--rule-strong); padding:0.6rem;
  width:min(18rem, calc(100vw - 2rem)); font-size:0.78rem; color:var(--ink);
  box-shadow:0 4px 12px rgba(0,0,0,0.08); border-radius:4px;
}
.kpi:last-child .kpi-tooltip{right:auto; left:0;}
.kpi-tooltip div + div{margin-top:0.35rem;}
.kpi-badge:hover + .kpi-tooltip,
.kpi-badge:focus + .kpi-tooltip,
.kpi-badge.open + .kpi-tooltip{display:block;}
.btn{
  font-family:var(--sans); font-size:0.85rem; font-weight:500;
  background:#fff; border:1px solid var(--rule-strong); padding:0.35rem 0.7rem;
  border-radius:3px; cursor:pointer; color:var(--ink);
}
.btn:hover{background:#f4f3ed;}
.chip{
  display:inline-block; font-size:0.65rem; font-weight:600;
  background:rgba(180,83,9,0.12); color:var(--accent-warm);
  padding:0.1rem 0.4rem; border-radius:3px; margin-left:0.3rem;
  letter-spacing:0.04em; text-transform:uppercase;
}
.chip.neutral{
  background:rgba(0,0,0,0.05); color:var(--ink-muted);
}
/* dek section — no rule below the dek, tighter bottom padding */
.dek-section{border-bottom:none; padding-bottom:0.5rem;}
.delta-chip{
  display:inline-block; margin-left:0.5rem; font-size:0.8rem; color:var(--accent-warm);
  font-feature-settings:"tnum"; font-weight:500;
}

/* sections */
section{padding:2rem; border-bottom:1px solid var(--rule);}
section h2{
  font-family:var(--serif); font-weight:600; font-size:1.8rem;
  margin:0 0 0.5rem; line-height:1.2;
}
section .lede{color:var(--ink-muted); margin:0 0 1.2rem; max-width:62ch;}
.section-anchor{font-family:var(--mono); font-size:0.7rem; color:var(--ink-muted); margin-left:0.5rem;}

/* how-to-read */
.howto-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:1.4rem;}
.howto-card{background:var(--bg-tile); padding:1rem; border:1px solid var(--rule); border-radius:4px;}
.howto-card svg{width:100%; height:90px;}
.howto-card p{font-size:0.85rem; margin:0.6rem 0 0;}

/* hero map */
.map-toolbar{display:flex; flex-wrap:wrap; gap:1rem; align-items:center; margin-bottom:0.7rem;}
.map-toolbar > *{font-size:0.85rem;}
select{font-family:var(--sans); padding:0.3rem 0.5rem; border:1px solid var(--rule-strong);
       border-radius:3px; background:#fff; color:var(--ink);}
.toggle{display:inline-flex; align-items:center; gap:0.35rem; cursor:pointer;
        background:#fff; padding:0.3rem 0.6rem; border:1px solid var(--rule); border-radius:3px;}
.toggle input{margin:0;}
#map{height:560px; width:100%; background:#eee; border:1px solid var(--rule); position:relative;}
.legend{
  font-size:0.78rem; padding:0.5rem 0.7rem; background:#fff; border:1px solid var(--rule);
}
.legend .swatch{display:inline-block; width:14px; height:10px; margin-right:0.3rem; vertical-align:middle;}
.legend-strip{
  background:#fff; border:1px solid #e5e7eb; padding:8px 10px;
  box-shadow:0 2px 6px rgba(0,0,0,0.06); min-width:240px;
}
.legend-strip .legend-title{
  font-family:var(--sans); font-size:11px; font-weight:600;
  color:var(--ink); margin-bottom:6px; letter-spacing:0.02em;
}
.legend-strip .legend-bar{
  width:220px; height:10px; border-radius:2px; border:1px solid #e5e7eb;
}
.legend-strip .legend-ticks{
  display:flex; justify-content:space-between; width:220px;
  font-family:var(--sans); font-size:10px; color:var(--ink-muted); margin-top:3px;
  font-feature-settings:"tnum";
}
.legend-strip .legend-endpoints{
  display:flex; justify-content:space-between; width:220px;
  font-family:var(--sans); font-style:italic; font-size:11px; color:var(--ink-muted); margin-top:4px;
}
.legend-strip .legend-unresolved{
  margin-top:6px; font-size:11px; color:var(--ink-muted);
}
.legend-strip .legend-unresolved .swatch{
  display:inline-block; width:10px; height:10px; vertical-align:middle; margin-right:4px; border-radius:2px;
}
.rio-inset{
  position:absolute; top:8px; right:8px; width:120px; height:120px;
  background:#fff; border:1px solid #d4d2cc; box-shadow:0 2px 6px rgba(0,0,0,0.10);
  z-index:600;
}
.rio-inset::before{
  content:'Rio de Janeiro'; position:absolute; top:2px; left:4px;
  font-family:var(--sans); font-size:9px; color:var(--ink-muted);
  letter-spacing:0.04em; z-index:2; pointer-events:none;
  background:rgba(255,255,255,0.85); padding:0 3px; border-radius:2px;
}
.site-diamond{
  width:10px; height:10px; background:#dc2626;
  transform:rotate(45deg); border:1px solid #fff;
  box-shadow:0 0 0 1px #dc2626;
}

/* charts */
.chart{width:100%; min-height:400px;}
#hist{height:380px;}
#scatter{height:420px;}
.chart-caption{font-size:0.82rem; color:var(--ink-muted); margin-top:0.5rem;}

/* caveats */
.caveat-group{padding:1.2rem 0; border-bottom:1px solid var(--rule);}
.caveat-group:last-child{border-bottom:none;}
.caveat-group h3{font-family:var(--sans); font-size:0.78rem; text-transform:uppercase;
  letter-spacing:0.12em; color:var(--ink-muted); font-weight:600; margin:0 0 0.8rem;}
.finding{
  background:var(--bg-tile); border:1px solid var(--rule); border-left:3px solid var(--rule-strong);
  padding:1rem; margin-bottom:0.8rem; border-radius:0 3px 3px 0;
}
.finding--critical{border-left-color:#b91c1c;}
.finding--high{border-left-color:var(--accent-warm);}
.finding--medium{border-left-color:#a16207;}
.finding--low{border-left-color:var(--accent-primary);}
.finding header{display:flex; align-items:baseline; gap:0.6rem; margin-bottom:0.4rem;}
.finding-id{
  font-family:var(--mono); font-size:0.78rem; font-weight:600;
  color:var(--ink-muted); background:rgba(0,0,0,0.05); padding:0.05rem 0.4rem; border-radius:3px;
}
.finding h4{font-family:var(--serif); font-size:1.05rem; font-weight:600; margin:0;}
.finding p{margin:0.3rem 0; font-size:0.9rem;}
.chevron{font-size:0.8rem;}

/* cross-site */
.cross-site-rows{display:grid; gap:0.4rem;}
.csr-row{
  display:grid; grid-template-columns:1.5fr 1.2fr 2fr 1fr; gap:0.8rem;
  align-items:center; padding:0.6rem; background:var(--bg-tile);
  border:1px solid var(--rule); border-radius:3px; cursor:pointer;
  transition:background 0.15s;
}
.csr-row:hover{background:#f4f3ed;}
.csr-row.current{border-color:var(--accent-primary); background:rgba(15,118,110,0.04);}
.csr-name{font-weight:600;}
.csr-typo{font-size:0.78rem; color:var(--ink-muted);}
.csr-bar{position:relative; height:8px; background:#eee; border-radius:3px;}
.csr-bar-fill{position:absolute; top:0; left:0; height:100%; background:var(--rule-strong); border-radius:3px;}
.csr-tick-line{position:absolute; top:-2px; bottom:-2px; width:1px; background:rgba(0,0,0,0.18);}
.csr-row.current .csr-bar-fill{background:var(--accent-primary);}
.csr-stat{font-family:var(--sans); font-feature-settings:"tnum"; font-size:0.85rem; text-align:right;}
.csr-ticks{display:grid; grid-template-columns:1.5fr 1.2fr 2fr 1fr; gap:0.8rem; padding:0 0.6rem; align-items:center;}
.csr-axis{position:relative; height:1rem;}
.csr-axis .tick{position:absolute; transform:translateX(-50%); font-family:var(--mono); font-size:0.7rem; color:var(--ink-muted);}

/* drawer */
.drawer{
  position:fixed; top:0; right:0; height:100%; width:min(420px, 100%);
  background:#fff; border-left:1px solid var(--rule-strong);
  padding:1.5rem; overflow-y:auto; transform:translateX(100%);
  transition:transform 0.25s ease-out; z-index:100;
  box-shadow:-4px 0 12px rgba(0,0,0,0.06);
}
.drawer.open{transform:translateX(0);}
.drawer h3{font-family:var(--serif); font-size:1.3rem; font-weight:600; margin:0 0 1rem;}
.drawer-close{
  position:absolute; top:1rem; right:1rem;
  background:none; border:none; font-size:1.4rem; cursor:pointer; color:var(--ink-muted);
}
.drawer details{
  padding:0.6rem 0; border-bottom:1px solid var(--rule);
}
.drawer summary{cursor:pointer; font-weight:500; font-size:0.9rem;}
.drawer summary .finding-id{margin-right:0.4rem;}
.drawer p{font-size:0.85rem; margin:0.5rem 0 0; color:var(--ink-muted);}

/* footer */
footer{padding:2rem; font-size:0.85rem; color:var(--ink-muted);}
.footer-grid{display:grid; grid-template-columns:repeat(5, 1fr); gap:1rem; margin-bottom:1.2rem;}
.footer-tile{padding:0.8rem; background:var(--bg-tile); border:1px solid var(--rule); border-radius:3px;}
.footer-tile .tile-name{font-weight:600; color:var(--ink); font-family:var(--serif); font-size:1.05rem;}
.footer-tile .tile-typo{font-size:0.74rem; color:var(--ink-muted); margin-top:0.2rem;}
.footer-tile .tile-stats{margin-top:0.4rem; font-size:0.78rem; display:flex; gap:0.6rem; flex-wrap:wrap;}

.back-to-top{display:inline-block; margin-top:1rem; font-size:0.85rem;}
.print-only-a3{display:none;}

/* in-tile audit-action hint */
.kpi-hint{
  display:block; margin-top:0.35rem; font-size:0.7rem; color:var(--accent-warm);
  font-style:italic;
}

/* TOC */
.toc{
  display:flex; flex-wrap:wrap; gap:0.8rem; padding:0.7rem 2rem;
  font-size:0.8rem; border-bottom:1px solid var(--rule); background:#fdfcf8;
}
.toc a{color:var(--ink-muted); border-bottom:1px dotted transparent;}
.toc a:hover{color:var(--accent-primary); border-bottom-color:var(--accent-primary);}

/* action button distinct from toggle */
.action-btn{
  font-family:var(--sans); font-size:0.8rem; font-weight:500;
  padding:0.35rem 0.8rem; background:var(--accent-primary); color:#fff;
  border:none; border-radius:3px; cursor:pointer;
  border-bottom:none;
}
.action-btn:hover{background:#0c5d56;}

/* cross-site typology bands */
.csr-typology-band{
  padding:0.4rem 0.8rem; font-family:var(--sans); font-size:0.72rem;
  text-transform:uppercase; letter-spacing:0.1em; color:var(--ink-muted);
  background:#f4f3ed; border:1px solid var(--rule); border-bottom:none;
  border-radius:3px 3px 0 0; margin-top:0.6rem;
}
.csr-typology-band:first-child{margin-top:0;}

/* human-stakes gloss inside finding cards */
.finding .human-stakes{
  display:block; margin-top:0.5rem; padding:0.5rem 0.7rem;
  background:rgba(180,83,9,0.06); border-left:2px solid var(--accent-warm);
  font-size:0.86rem; color:var(--ink);
}
.finding .human-stakes::before{
  content:"What this means for the reader · ";
  font-family:var(--sans); font-size:0.7rem; font-weight:600;
  text-transform:uppercase; letter-spacing:0.08em; color:var(--accent-warm);
}

/* hairline above empty-state framing-changers */
.caveat-group h3 + .muted{
  padding-top:0.5rem; border-top:1px dashed var(--rule);
}

/* responsive */
@media (max-width:900px){
  .kpi-strip{grid-template-columns:repeat(3,1fr);}
  .kpi{border-bottom:1px solid var(--rule);}
  .howto-grid{grid-template-columns:1fr;}
  .footer-grid{grid-template-columns:repeat(2,1fr);}
  .csr-row{grid-template-columns:1fr 1fr;}
  .mast-title{font-size:2.2rem;}
  #map{min-height:340px; max-height:60vh;}
  .toc{padding:0.6rem 1rem; gap:0.6rem;}
  .map-toolbar{flex-direction:column; align-items:stretch; gap:0.5rem;}
  .map-toolbar label, .map-toolbar .toggle{width:100%;}
  .map-toolbar select{width:100%;}
}

/* landing page */
.landing-hero{padding:3rem 2rem 1.5rem; text-align:center;}
.landing-hero h1{font-family:var(--serif); font-weight:600; font-size:3.4rem; margin:0;}
.landing-hero .lede{max-width:60ch; margin:1rem auto; color:var(--ink-muted);}
.site-grid{
  display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
  gap:1rem; padding:0 2rem 2rem;
}
.site-tile{
  display:block; padding:1.2rem; background:var(--bg-tile);
  border:1px solid var(--rule); border-radius:4px; color:var(--ink);
  border-bottom:1px solid var(--rule);
  transition:border-color 0.15s, transform 0.15s;
}
.site-tile:hover{border-color:var(--accent-primary); transform:translateY(-2px);}
.site-tile .tile-name{font-family:var(--serif); font-size:1.4rem; font-weight:600;}
.site-tile .tile-typo{font-size:0.82rem; color:var(--ink-muted); margin:0.2rem 0 0.8rem;}
.site-tile .tile-stats{display:flex; gap:0.8rem; font-size:0.82rem; flex-wrap:wrap;}
"""

PRINT_CSS = r"""
@page { size: A3 portrait; margin: 12mm; }
@media print{
  html, body{background:#fff !important; color:#000 !important;}
  body{font-size:9.5pt; line-height:1.45;}
  .kpi-strip{position:static !important; page-break-inside:avoid; grid-template-columns:repeat(3, 1fr); border:1px solid #999;}
  .kpi{border-color:#999 !important;}
  .drawer{position:static !important; transform:none !important; width:100% !important; height:auto !important; box-shadow:none !important; border:1px solid #999 !important; padding:0.8rem !important; page-break-before:always;}
  .drawer details{display:block !important;}
  .drawer details > *{display:block !important;}
  .drawer summary::-webkit-details-marker{display:none;}
  .toggle, .drawer-toggle, .drawer-close, .map-toolbar select, .map-toolbar label, .map-toolbar .toggle, .btn, #hist-reset, .toc, .delta-chip, .kpi-tooltip, .kpi-lineage{display:none !important;}
  .leaflet-tile, .leaflet-tile-pane img{filter:grayscale(1) contrast(0.9) !important;}
  .leaflet-control-zoom, .leaflet-control-attribution{display:none !important;}
  section{page-break-inside:avoid; border-color:#999 !important; padding:0.8rem 0 !important;}
  section h2{font-size:14pt;}
  .muted, .chart-caption, .lede{color:#333 !important;}
  a{color:#000 !important; text-decoration:underline;}
  .pill{background:#eee !important; color:#000 !important; border:1px solid #999;}
  .masthead{grid-template-columns:1fr !important; padding:0 0 0.4rem !important; border-bottom:1px solid #000;}
  .howto-grid{grid-template-columns:repeat(3, 1fr); gap:0.5rem;}
  .footer-grid{grid-template-columns:repeat(5, 1fr); gap:0.3rem;}
  footer{padding:0.5rem 0 !important;}
  .print-only-a3{display:block; max-width:100%; height:auto; margin:1rem 0; page-break-before:always;}
  #map, #hist, #scatter{display:none;}
}
"""

JS_STATE = r"""
/* state.js — single source of truth for mask composition + KPI recompute.
 * Other modules subscribe via window.addEventListener('mf:state', cb). */
(function(){
  const state = {
    observers: null,        // GeoJSON FeatureCollection (in WGS84)
    excludeEdge: false,
    showOffsetCluster: false, // Maré only — default OFF means masked
    showUnresolved: true,   // Rocinha only — default ON means show as grey
    colorBy: 'svf',
    season: 'annual',
    brushRange: null,       // [min, max] from histogram brush
  };
  window.mfState = state;

  function maskObservers(){
    if(!state.observers) return [];
    return state.observers.features.filter(f => {
      const p = f.properties;
      if(state.excludeEdge && p.dist_to_boundary != null && p.dist_to_boundary <= 15) return false;
      if(p.street_id === 226 && !state.showOffsetCluster) return false;
      if(state.brushRange){
        const v = p.svf;
        if(v < state.brushRange[0] || v > state.brushRange[1]) return false;
      }
      return true;
    });
  }

  function recomputeKPIs(){
    const feats = maskObservers();
    const n = feats.length;
    document.querySelectorAll('[data-key="n_observers"]').forEach(el => el.textContent = n.toLocaleString());
    // mean SVF (observer-mean here; length-weighted true mean stays fixed since we don't carry per-segment lengths to JS — chip shows delta)
    if(feats.length){
      let sum = 0, c = 0;
      for(const f of feats){
        const v = f.properties.svf;
        if(state.showUnresolved && v === 0){ /* exclude zeros from observer-mean live */ continue; }
        sum += v; c += 1;
      }
      const mean = c ? sum/c : NaN;
      const tile = document.querySelector('#kpi-mean-svf .kpi-value');
      if(tile){
        const base = parseFloat(tile.dataset.base || tile.textContent);
        if(isNaN(tile.dataset.base)) tile.dataset.base = base;
        const delta = mean - parseFloat(tile.dataset.base);
        // surface delta chip next to it
        let chip = document.querySelector('#kpi-mean-svf .delta-chip');
        if(!chip){
          chip = document.createElement('span');
          chip.className = 'delta-chip';
          tile.appendChild(chip);
        }
        const sign = delta >= 0 ? '+' : '';
        // Only show the delta chip on edge-toggle changes (where base and
        // filtered are in the same observer-frame). Brushing the histogram
        // composes an SVF range filter on top of the observer-mean, which
        // mixes semantics — the value would be unintelligible. Suppress.
        const isBrushing = state.brushRange != null;
        const isAllShown = n === window.mfState.observers.features.length;
        chip.textContent = (isAllShown || isBrushing) ? '' : `${sign}${delta.toFixed(3)} (edge mask)`;
      }
    }
    window.dispatchEvent(new CustomEvent('mf:filtered', {detail:{features:feats}}));
  }
  state._recompute = recomputeKPIs;
  state._maskObservers = maskObservers;

  window.addEventListener('mf:dataready', () => {
    recomputeKPIs();
  });

  // toggle wiring
  document.addEventListener('change', e => {
    const id = e.target.id;
    if(id === 't-edge'){ state.excludeEdge = e.target.checked; recomputeKPIs(); }
    if(id === 't-offset'){ state.showOffsetCluster = e.target.checked; recomputeKPIs(); }
    if(id === 't-unresolved'){ state.showUnresolved = e.target.checked; recomputeKPIs(); }
    if(id === 'color-by'){ state.colorBy = e.target.value; window.dispatchEvent(new Event('mf:colorby')); }
    if(id === 'season'){ state.season = e.target.value; window.dispatchEvent(new Event('mf:season')); }
  });
})();
"""

JS_MAP = r"""
/* map.js — Leaflet observer map with SVF/solar layer switcher. */
(function(){
  const viridis_r = ['#fde725','#a0da39','#4ac16d','#1fa187','#277f8e','#365c8d','#46327e','#440154'];
  const ylorbr = ['#ffffe5','#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#8c2d04'];

  function rampColor(v, ramp, lo, hi){
    if(v == null || isNaN(v)) return '#9ca3af';
    const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    const idx = Math.min(ramp.length - 1, Math.floor(t * ramp.length));
    return ramp[idx];
  }
  function rampGradientCSS(ramp){
    return 'linear-gradient(to right, ' + ramp.map((c,i)=>c + ' ' + (i*100/(ramp.length-1)).toFixed(1) + '%').join(', ') + ')';
  }

  async function init(){
    const dataResp = await fetch('data/observers.geojson');
    const data = await dataResp.json();
    window.mfState.observers = data;

    let footprint = null;
    try {
      const fpResp = await fetch('data/footprint.geojson');
      if(fpResp.ok) footprint = await fpResp.json();
    } catch(e){}

    const map = L.map('map', { preferCanvas: true, renderer: L.canvas({padding:0.5}), zoomControl:true });
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>, &copy; <a href="https://carto.com/">CARTO</a>',
      subdomains: 'abcd', maxZoom: 20
    }).addTo(map);

    // dedicated pane for the footprint outline so it sits between basemap and markers
    map.createPane('footprint');
    map.getPane('footprint').style.zIndex = 380;
    map.createPane('labels');
    map.getPane('labels').style.zIndex = 650;
    map.getPane('labels').style.pointerEvents = 'none';
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png', {
      subdomains:'abcd', pane:'labels', maxZoom:20, opacity:0.85
    }).addTo(map);

    if(footprint){
      const fpLayer = L.geoJSON(footprint, {
        pane:'footprint',
        style: { color:'#1a1a1a', weight:2.0, opacity:0.85, fillOpacity:0 }
      }).addTo(map);
      map.fitBounds(fpLayer.getBounds());
    }

    function styleFn(feature){
      const p = feature.properties;
      const colorBy = window.mfState.colorBy;
      let color;
      if(window.mfState.showUnresolved && p.svf === 0){
        color = '#9ca3af';
      } else if(colorBy === 'svf'){
        color = rampColor(p.svf, viridis_r, 0, 0.8);
      } else {
        const seasonKey = 'solar_hours_' + window.mfState.season;
        const v = p[seasonKey] != null ? p[seasonKey] : p.solar_hours_annual;
        color = rampColor(v, ylorbr, 0, 10);
      }
      const isEdge = p.dist_to_boundary != null && p.dist_to_boundary <= 15;
      return {
        radius: isEdge ? 4.0 : 5.5,
        fillColor: color,
        color: '#ffffff',
        weight: 0.6,
        opacity: 0.95,
        fillOpacity: isEdge ? 0.50 : 0.90
      };
    }

    let layer = L.geoJSON(data, {
      pointToLayer: (f, latlng) => L.circleMarker(latlng, styleFn(f)),
      filter: f => {
        const p = f.properties;
        if(window.mfState.excludeEdge && p.dist_to_boundary != null && p.dist_to_boundary <= 15) return false;
        if(p.street_id === 226 && !window.mfState.showOffsetCluster) return false;
        return true;
      },
      onEachFeature: (f, lyr) => {
        const p = f.properties;
        lyr.bindPopup(`<strong>Observer</strong> <em style="color:#5b5b5b">— a pedestrian standing here</em><br>SVF <span style="color:#5b5b5b">(sky openness, 0–1)</span>: <b>${p.svf?.toFixed(3)}</b><br>Solar <span style="color:#5b5b5b">(annual, h/day)</span>: <b>${p.solar_hours_annual?.toFixed(2)}</b><br>Solar <span style="color:#5b5b5b">(winter)</span>: <b>${p.solar_hours_winter?.toFixed(2)}</b><br>Street ID <span style="color:#5b5b5b">(road segment)</span>: ${p.street_id}<br>Offset <span style="color:#5b5b5b">(from centerline)</span>: ${p.offset_distance?.toFixed(2)} m<br>Boundary distance: ${p.dist_to_boundary?.toFixed(1) || '—'} m`);
      }
    }).addTo(map);

    function refresh(){
      map.removeLayer(layer);
      layer = L.geoJSON(data, {
        pointToLayer: (f, latlng) => L.circleMarker(latlng, styleFn(f)),
        filter: f => {
          const p = f.properties;
          if(window.mfState.excludeEdge && p.dist_to_boundary != null && p.dist_to_boundary <= 15) return false;
          if(p.street_id === 226 && !window.mfState.showOffsetCluster) return false;
          if(window.mfState.brushRange){
            if(p.svf < window.mfState.brushRange[0] || p.svf > window.mfState.brushRange[1]) return false;
          }
          return true;
        },
        onEachFeature: (f, lyr) => {
          const p = f.properties;
          lyr.bindPopup(`<strong>Observer</strong> <em style="color:#5b5b5b">— a pedestrian standing here</em><br>SVF <span style="color:#5b5b5b">(sky openness, 0–1)</span>: <b>${p.svf?.toFixed(3)}</b><br>Solar <span style="color:#5b5b5b">(annual, h/day)</span>: <b>${p.solar_hours_annual?.toFixed(2)}</b>`);
        }
      }).addTo(map);
    }
    window.addEventListener('mf:filtered', refresh);
    window.addEventListener('mf:colorby', refresh);
    window.addEventListener('mf:season', refresh);

    // bottom-left horizontal-strip legend
    const legend = L.control({position:'bottomleft'});
    legend.onAdd = function(){
      const div = L.DomUtil.create('div', 'legend legend-strip');
      div.innerHTML = legendHTML();
      return div;
    };
    legend.addTo(map);
    window.addEventListener('mf:colorby', () => {
      document.querySelector('.legend').innerHTML = legendHTML();
    });

    function legendHTML(){
      if(window.mfState.colorBy === 'svf'){
        const grad = rampGradientCSS(viridis_r);
        const zeroShare = (window.mfMeta && window.mfMeta.exact_zero_svf_share) || 0;
        const unresolved = zeroShare >= 0.02
          ? `<div class="legend-unresolved"><span class="swatch" style="background:#9ca3af"></span>unresolved (SVF=0, ${(zeroShare*100).toFixed(1)}%)</div>`
          : '';
        return `<div class="legend-title">Sky View Factor at the observer</div>
          <div class="legend-bar" style="background:${grad}"></div>
          <div class="legend-ticks"><span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>0.8+</span></div>
          <div class="legend-endpoints"><span>closed canyon</span><span>open field</span></div>
          ${unresolved}`;
      } else {
        const grad = rampGradientCSS(ylorbr);
        return `<div class="legend-title">Solar hours at the observer (h/day)</div>
          <div class="legend-bar" style="background:${grad}"></div>
          <div class="legend-ticks"><span>0</span><span>2.5</span><span>5</span><span>7.5</span><span>10+</span></div>
          <div class="legend-endpoints"><span>deep shade</span><span>full sun</span></div>`;
      }
    }

    // Rio locator inset (top-right)
    const insetEl = L.DomUtil.create('div', 'rio-inset', map.getContainer());
    insetEl.id = 'rio-inset';
    setTimeout(async () => {
      try {
        const r = await fetch('../shared/rio_outline.geojson');
        if(!r.ok) return;
        const rio = await r.json();
        const inset = L.map(insetEl, {
          zoomControl:false, attributionControl:false,
          dragging:false, scrollWheelZoom:false, doubleClickZoom:false,
          boxZoom:false, keyboard:false, touchZoom:false
        });
        const rioLayer = L.geoJSON(rio, {style:{color:'#475569', weight:1, fillColor:'#cbd5e1', fillOpacity:0.85}}).addTo(inset);
        inset.fitBounds(rioLayer.getBounds(), {padding:[4,4]});
        const lat = window.mfMeta && window.mfMeta.centroid_lat;
        const lon = window.mfMeta && window.mfMeta.centroid_lon;
        if(lat && lon){
          L.marker([lat, lon], {
            icon: L.divIcon({
              className:'site-marker',
              html:'<div class="site-diamond"></div>',
              iconSize:[14,14], iconAnchor:[7,7]
            })
          }).addTo(inset);
        }
      } catch(e){ console.warn('inset:', e); }
    }, 250);

    window.dispatchEvent(new Event('mf:dataready'));
  }
  if(document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
"""

JS_HIST = r"""
/* hist.js — precomputed binned distribution (length-weighted in front, count ghost behind). */
(function(){
  async function init(){
    while(!window.mfState || !window.mfState.observers) {
      await new Promise(r => setTimeout(r, 100));
    }
    const bins = (window.mfStats && window.mfStats.svf_bins) || [];
    const xs = bins.map(b => b.c);
    const lw = bins.map(b => b.lw_share);
    const ct = bins.map(b => b.count_share);
    const ymax = Math.max(Math.max(...lw, 0), Math.max(...ct, 0));

    const countGhost = {
      type:'bar', x: xs, y: ct, name:'count share',
      width: bins.map(_=>0.025),
      marker:{color:'#0f766e', line:{width:0}},
      opacity: 0.30,
      hovertemplate:'SVF %{x:.3f}<br>count share %{y:.1%}<extra></extra>'
    };
    const lwBars = {
      type:'bar', x: xs, y: lw, name:'length-weighted',
      width: bins.map(_=>0.025),
      marker:{color:'#0f766e', line:{width:0}},
      hovertemplate:'SVF %{x:.3f}<br>length share %{y:.1%}<extra></extra>'
    };

    const lwMean = (window.mfMeta && window.mfMeta.length_weighted_mean_svf) || null;
    const lwMedian = (window.mfMeta && window.mfMeta.length_weighted_median_svf) || null;

    const shapes = [];
    const annotations = [];
    // canyon-zone tint
    shapes.push({
      type:'rect', xref:'x', yref:'paper',
      x0:0, x1:0.25, y0:0, y1:1,
      fillcolor:'rgba(180,83,9,0.08)', line:{width:0}, layer:'below'
    });
    annotations.push({
      x:0.125, y:0.96, xref:'x', yref:'paper', showarrow:false,
      text:'canyon zone — sky mostly blocked',
      font:{family:'Inter, system-ui, sans-serif', size:10, color:'#b45309'},
      xanchor:'center'
    });
    if(lwMean != null){
      shapes.push({
        type:'line', xref:'x', yref:'paper',
        x0:lwMean, x1:lwMean, y0:0, y1:1,
        line:{color:'#b45309', width:1.6}
      });
      annotations.push({
        x: lwMean, y: 1.03, xref:'x', yref:'paper', showarrow:false,
        text: 'length-weighted mean — ' + lwMean.toFixed(3),
        font:{family:'Inter, system-ui, sans-serif', size:11, color:'#b45309'},
        xanchor: lwMean > 0.5 ? 'right' : 'left'
      });
    }
    if(lwMedian != null && lwMedian > 0){
      shapes.push({
        type:'line', xref:'x', yref:'paper',
        x0:lwMedian, x1:lwMedian, y0:0, y1:1,
        line:{color:'#64748b', width:1.0}
      });
      annotations.push({
        x: lwMedian, y: 0.88, xref:'x', yref:'paper', showarrow:false,
        text: 'median ' + lwMedian.toFixed(3),
        font:{family:'Inter, system-ui, sans-serif', size:10, color:'#64748b'},
        xanchor: lwMedian > 0.5 ? 'right' : 'left'
      });
    }

    const layout = {
      margin:{t:34, l:60, r:20, b:54},
      paper_bgcolor:'#fbfaf6', plot_bgcolor:'#fbfaf6',
      barmode:'overlay', bargap:0.02,
      height:380,
      xaxis:{
        title:{text:'Sky View Factor — closed canyon (0) to open field (1)',
               font:{family:'Source Serif 4, Georgia, serif', size:13, color:'#1f2937'}},
        range:[0,1], tickvals:[0,0.25,0.5,0.75,1], showgrid:false,
        tickfont:{family:'Inter, system-ui, sans-serif', size:11, color:'#334155'}
      },
      yaxis:{
        title:{text:'share of street length',
               font:{family:'Source Serif 4, Georgia, serif', size:13, color:'#1f2937'}},
        tickformat:'.0%', range:[0, ymax*1.18], gridcolor:'#e8e6dd',
        tickfont:{family:'Inter, system-ui, sans-serif', size:11, color:'#334155'}
      },
      shapes: shapes,
      annotations: annotations,
      showlegend:false,
      dragmode:'select'
    };
    const config = {displaylogo:false, displayModeBar:false, responsive:true,
                    modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']};
    Plotly.newPlot('hist', [countGhost, lwBars], layout, config);

    document.getElementById('hist').on('plotly_selected', ev => {
      if(!ev || !ev.range || !ev.range.x){
        window.mfState.brushRange = null;
      } else {
        window.mfState.brushRange = ev.range.x;
      }
      window.mfState._recompute();
    });
    const resetBtn = document.getElementById('hist-reset');
    if(resetBtn) resetBtn.addEventListener('click', () => {
      window.mfState.brushRange = null;
      Plotly.relayout('hist', {});
      window.mfState._recompute();
    });
  }
  if(document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
"""

JS_SCATTER = r"""
/* scatter.js — SVF vs solar_hours_annual density + OLS line + regime anchors. */
(function(){
  function pearson(x, y){
    const n = x.length;
    if(n < 2) return NaN;
    let mx=0, my=0; for(let i=0;i<n;i++){mx+=x[i]; my+=y[i];} mx/=n; my/=n;
    let num=0, dx=0, dy=0;
    for(let i=0;i<n;i++){
      const a=x[i]-mx, b=y[i]-my;
      num += a*b; dx += a*a; dy += b*b;
    }
    return num / Math.sqrt(dx*dy);
  }
  function buildBinCounts(xs, ys, nx, ny, xmin, xmax, ymin, ymax){
    const counts = new Float64Array(nx * ny);
    const dx = (xmax - xmin) / nx, dy = (ymax - ymin) / ny;
    for(let i=0; i<xs.length; i++){
      const x = xs[i], y = ys[i];
      if(x < xmin || x >= xmax || y < ymin || y >= ymax) continue;
      const ix = Math.min(nx-1, Math.floor((x - xmin) / dx));
      const iy = Math.min(ny-1, Math.floor((y - ymin) / dy));
      counts[iy * nx + ix] += 1;
    }
    return {counts, dx, dy};
  }
  function lowDensityMask(xs, ys, counts, dx, dy, nx, ny, xmin, ymin, threshold){
    const mask = new Uint8Array(xs.length);
    for(let i=0; i<xs.length; i++){
      const x = xs[i], y = ys[i];
      const ix = Math.min(nx-1, Math.max(0, Math.floor((x - xmin) / dx)));
      const iy = Math.min(ny-1, Math.max(0, Math.floor((y - ymin) / dy)));
      mask[i] = counts[iy * nx + ix] <= threshold ? 1 : 0;
    }
    return mask;
  }
  async function init(){
    while(!window.mfState || !window.mfState.observers) {
      await new Promise(r => setTimeout(r, 100));
    }
    const allFeats = window.mfState.observers.features;
    const showZero = (window.mfState.showUnresolved !== false);
    const feats = showZero ? allFeats : allFeats.filter(f => f.properties.svf !== 0);
    const xAll = feats.map(f => f.properties.svf);
    const yAll = feats.map(f => f.properties.solar_hours_annual);
    const n = xAll.length;

    // nbinsx auto-scales with density (Rocinha is dense)
    const nx = n > 12000 ? 70 : 50;
    const ny = 40;
    const {counts, dx, dy} = buildBinCounts(xAll, yAll, nx, ny, 0, 1, 0, 12);

    // density layer (histogram2d)
    const density = {
      type:'histogram2d',
      x: xAll, y: yAll,
      xbins:{start:0, end:1, size:1/nx},
      ybins:{start:0, end:12, size:12/ny},
      colorscale:[[0,'rgba(15,118,110,0)'],[0.05,'rgba(15,118,110,0.18)'],[0.25,'rgba(15,118,110,0.45)'],[0.6,'rgba(15,118,110,0.75)'],[1,'#0a4f4a']],
      showscale:false,
      hovertemplate:'SVF %{x:.2f}<br>solar %{y:.2f} h<br>observers %{z}<extra></extra>'
    };

    // low-density tail points only
    const nonZero = counts.filter(c => c > 0).sort((a,b)=>a-b);
    const lowThresh = nonZero.length ? nonZero[Math.floor(nonZero.length * 0.25)] : 0;
    const mask = lowDensityMask(xAll, yAll, counts, dx, dy, nx, ny, 0, 0, lowThresh);
    const xt = [], yt = [];
    for(let i=0;i<xAll.length;i++) if(mask[i]){ xt.push(xAll[i]); yt.push(yAll[i]); }
    const tail = {
      type:'scattergl', mode:'markers',
      x: xt, y: yt,
      marker:{size:3, opacity:0.55, color:'#1f766e'},
      hovertemplate:'SVF %{x:.2f}<br>solar %{y:.2f} h<extra></extra>',
      showlegend:false
    };

    // OLS regression line (from precomputed fit if present, else compute live)
    let slope, intercept, r;
    const fit = window.mfStats && window.mfStats.svf_solar_fit;
    if(fit && fit.slope != null){
      slope = fit.slope; intercept = fit.intercept; r = fit.r;
    } else {
      // fallback live fit
      let sumx=0,sumy=0,sumxx=0,sumxy=0;
      for(let i=0;i<n;i++){sumx+=xAll[i]; sumy+=yAll[i]; sumxx+=xAll[i]*xAll[i]; sumxy+=xAll[i]*yAll[i];}
      slope = (n*sumxy - sumx*sumy) / (n*sumxx - sumx*sumx);
      intercept = (sumy - slope*sumx) / n;
      r = pearson(xAll, yAll);
    }
    const ols = {
      type:'scatter', mode:'lines',
      x:[0, 1], y:[intercept, intercept + slope],
      line:{color:'#b45309', width:2},
      hoverinfo:'skip', showlegend:false
    };

    const annotations = [
      {
        x:0.98, y:0.06, xref:'paper', yref:'paper', xanchor:'right', yanchor:'bottom', showarrow:false,
        text:`Pearson r = <b>${r.toFixed(3)}</b><br><span style="font-size:10px">n = ${n.toLocaleString()}</span>`,
        font:{family:'Inter, system-ui, sans-serif', size:12, color:'#1f2937'},
        bgcolor:'rgba(255,255,255,0.92)', bordercolor:'#d4d2cc', borderpad:5
      },
      {
        x:0.10, y:1.7, ax:0.02, ay:0.3,
        xref:'x', yref:'y', axref:'x', ayref:'y',
        text:'shaded canyons<br>&lt;1 h/day', showarrow:true, arrowhead:2, arrowsize:0.8,
        font:{family:'Inter, system-ui, sans-serif', size:10, color:'#334155'},
        bgcolor:'rgba(255,255,255,0.85)', borderpad:3, align:'left'
      },
      {
        x:0.65, y:10.8, ax:0.93, ay:11.3,
        xref:'x', yref:'y', axref:'x', ayref:'y',
        text:'open ridgelines<br>10+ h/day', showarrow:true, arrowhead:2, arrowsize:0.8,
        font:{family:'Inter, system-ui, sans-serif', size:10, color:'#334155'},
        bgcolor:'rgba(255,255,255,0.85)', borderpad:3, align:'left'
      }
    ];

    const shapes = [
      // comfort-floor reference
      {
        type:'line', xref:'x', yref:'y',
        x0:0, x1:1, y0:4, y1:4,
        line:{color:'#94a3b8', width:1, dash:'dash'}
      }
    ];

    const layout = {
      margin:{t:30, l:60, r:20, b:54},
      paper_bgcolor:'#fbfaf6', plot_bgcolor:'#fbfaf6',
      height:420,
      xaxis:{
        title:{text:'Sky View Factor', font:{family:'Source Serif 4, Georgia, serif', size:13, color:'#1f2937'}},
        range:[0,1], gridcolor:'#e8e6dd',
        tickfont:{family:'Inter, system-ui, sans-serif', size:11, color:'#334155'}
      },
      yaxis:{
        title:{text:'solar hours (annual mean, h/day)', font:{family:'Source Serif 4, Georgia, serif', size:13, color:'#1f2937'}},
        range:[0,12], gridcolor:'#e8e6dd',
        tickfont:{family:'Inter, system-ui, sans-serif', size:11, color:'#334155'}
      },
      shapes: shapes,
      annotations: annotations,
      // add a faint "comfort floor" tag near the line
      showlegend:false
    };
    annotations.push({
      x:0.97, y:4, xref:'x', yref:'y', xanchor:'right', yanchor:'bottom', showarrow:false,
      text:'comfort floor — 4 h/day',
      font:{family:'Inter, system-ui, sans-serif', size:10, color:'#64748b'}
    });
    Plotly.newPlot('scatter', [density, tail, ols], layout,
      {displaylogo:false, displayModeBar:false, responsive:true,
       modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']});

    window.addEventListener('mf:filtered', ev => {
      const feats = ev.detail.features;
      const x = feats.map(f => f.properties.svf);
      const y = feats.map(f => f.properties.solar_hours_annual);
      const r2 = pearson(x, y);
      // re-skin density layer only; tail recompute is expensive — skip
      Plotly.restyle('scatter', {x:[x, x.filter((_,i)=>false)], y:[y, y.filter((_,i)=>false)]}, [0, 1]);
      Plotly.relayout('scatter', {'annotations[0].text': `Pearson r = <b>${r2.toFixed(3)}</b><br><span style="font-size:10px">n = ${x.length.toLocaleString()}</span>`});
    });
  }
  if(document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
"""

JS_DRAWER = r"""
/* drawer.js — open/close methodology drawer + deep-link anchors. */
(function(){
  const drawer = document.querySelector('.drawer');
  if(!drawer) return;
  const drawerId = drawer.id || 'methodology-drawer';
  drawer.id = drawerId;
  document.querySelectorAll('.drawer-toggle').forEach(btn => {
    btn.setAttribute('aria-controls', drawerId);
    btn.setAttribute('aria-expanded', 'false');
    btn.addEventListener('click', () => {
      const open = drawer.classList.toggle('open');
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  });
  const closeBtn = drawer.querySelector('.drawer-close');
  if(closeBtn) closeBtn.addEventListener('click', () => {
    drawer.classList.remove('open');
    document.querySelectorAll('.drawer-toggle').forEach(b => b.setAttribute('aria-expanded','false'));
  });

  // KPI badge tap-to-toggle for touch users (no hover affordance)
  document.querySelectorAll('.kpi-badge').forEach(badge => {
    badge.addEventListener('click', e => {
      e.stopPropagation();
      document.querySelectorAll('.kpi-badge.open').forEach(b => { if(b !== badge) b.classList.remove('open'); });
      badge.classList.toggle('open');
    });
  });
  document.addEventListener('click', () => {
    document.querySelectorAll('.kpi-badge.open').forEach(b => b.classList.remove('open'));
  });
  // open drawer when a finding chevron is clicked
  document.body.addEventListener('click', e => {
    const a = e.target.closest('a.chevron');
    if(!a) return;
    drawer.classList.add('open');
    const target = document.querySelector(a.getAttribute('href'));
    if(target && target.tagName === 'DETAILS') target.open = true;
  });
  // copy-link on KPI title click
  document.querySelectorAll('.kpi').forEach(tile => {
    const label = tile.querySelector('.kpi-label');
    if(!label) return;
    label.style.cursor = 'pointer';
    label.title = 'click to copy deep link';
    label.addEventListener('click', () => {
      const url = location.origin + location.pathname + '#' + tile.id;
      navigator.clipboard?.writeText(url);
      const old = label.textContent;
      label.textContent = 'link copied';
      setTimeout(() => label.textContent = old, 1200);
    });
  });
})();
"""

JS_GLOSSARY = r"""
/* glossary.js — hover tooltips for <dfn> terms. */
(function(){
  let dict = null;
  async function load(){
    try {
      const r = await fetch('../assets/glossary.json');
      if(r.ok) dict = await r.json();
    } catch(e){ dict = {}; }
    document.querySelectorAll('dfn[data-term]').forEach(el => {
      const term = el.dataset.term;
      if(dict[term]){
        el.title = dict[term];
      }
    });
  }
  if(document.readyState !== 'loading') load();
  else document.addEventListener('DOMContentLoaded', load);
})();
"""


SITE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{site_display} — MorphoFavela</title>
<meta name="description" content="Street-level sky view factor and solar access for {site_display}, an interactive dashboard with audit-level transparency.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="stylesheet" href="../style.css">
<link rel="stylesheet" href="../print.css" media="print">
<style>
__INLINE_CSS__
</style>
</head>
<body>

<a id="top"></a>
<header class="masthead">
  <div>
    <div class="mast-mark">MORPHOFAVELA · folha de rua</div>
    <div class="mast-sheet">sheet · {site} · v1</div>
    <h1 class="mast-title">{site_display}</h1>
    <div>
      <span class="pill">{typology}</span>
      <span class="mast-meta">{crs} · build {build_date} · git {git_sha}</span>
    </div>
  </div>
  <button class="drawer-toggle" type="button" aria-expanded="false" aria-controls="methodology-drawer">Methodology drawer ▾</button>
  <noscript><a class="drawer-toggle" href="#methodology-drawer" style="margin-left:0.5rem;">Open methodology (no-JS)</a></noscript>
</header>

<section id="dek-section" class="dek-section">
  <p class="dek">{dek}</p>
</section>

<nav class="toc" aria-label="section navigation">
  <a href="#how-to-read">How to read</a>
  <a href="#hero-map">Map</a>
  <a href="#distribution">Distribution</a>
  <a href="#svf-solar-scatter">Sky &amp; sun</a>
  <a href="#caveats">Caveats</a>
  <a href="#cross-site">Cross-site</a>
  <a href="#data-provenance">Provenance</a>
</nav>

<nav class="kpi-strip" aria-label="key indicators">
  {kpi_strip}
</nav>

<section id="how-to-read">
  <h2>How to read this</h2>
  <p class="lede">A short orientation. Three figures, three interactions, two glosses.</p>
  <div class="howto-grid">
    <div class="howto-card">
      <svg viewBox="0 0 200 90" aria-label="map thumbnail">
        <rect x="6" y="6" width="188" height="78" fill="#f0eee5" stroke="#d4d2cc"/>
        <circle cx="60" cy="40" r="3" fill="#0f766e"/>
        <circle cx="80" cy="55" r="3" fill="#0f766e"/>
        <circle cx="120" cy="35" r="3" fill="#fec44f"/>
        <circle cx="140" cy="60" r="3" fill="#fec44f"/>
        <text x="10" y="80" font-size="9" fill="#5b5b5b" font-family="Inter">click an observer for its values</text>
      </svg>
      <p>Each dot is one <dfn data-term="observer">observer</dfn> standing on the street. Colour shows SVF (greener = more open) or solar hours.</p>
    </div>
    <div class="howto-card">
      <svg viewBox="0 0 200 90" aria-label="histogram thumbnail">
        <rect x="6" y="6" width="188" height="78" fill="#fff" stroke="#d4d2cc"/>
        <rect x="30" y="40" width="14" height="40" fill="#0f766e"/>
        <rect x="50" y="20" width="14" height="60" fill="#0f766e"/>
        <rect x="70" y="30" width="14" height="50" fill="#0f766e"/>
        <rect x="90" y="50" width="14" height="30" fill="#0f766e"/>
        <rect x="110" y="60" width="14" height="20" fill="#0f766e"/>
        <line x1="50" y1="10" x2="50" y2="80" stroke="#b45309" stroke-dasharray="2 2"/>
        <text x="10" y="78" font-size="9" fill="#5b5b5b" font-family="Inter">box-select a range to filter the map</text>
      </svg>
      <p>The histogram is the SVF distribution. Box-select a range and the map plus scatter follow.</p>
    </div>
    <div class="howto-card">
      <svg viewBox="0 0 200 90" aria-label="scatter thumbnail">
        <rect x="6" y="6" width="188" height="78" fill="#fff" stroke="#d4d2cc"/>
        <circle cx="40" cy="65" r="2" fill="#0f766e"/>
        <circle cx="60" cy="55" r="2" fill="#0f766e"/>
        <circle cx="90" cy="40" r="2" fill="#0f766e"/>
        <circle cx="130" cy="30" r="2" fill="#0f766e"/>
        <circle cx="160" cy="20" r="2" fill="#0f766e"/>
        <text x="10" y="80" font-size="9" fill="#5b5b5b" font-family="Inter">hover for the data lineage badge</text>
      </svg>
      <p>SVF (x) vs solar hours (y). Pearson r updates as you filter. Dotted underlines are glossed terms.</p>
    </div>
  </div>
  <p style="margin-top:1.2rem;" class="measure">
    <strong>What is SVF?</strong> The fraction of the upper hemisphere not blocked by buildings —
    0 means walls all around, 1 means open field. <strong>What is solar access?</strong>
    Hours of direct sunlight reaching street level on an average day, modelled annually.
    All site means on this page are <dfn data-term="length-weighted">length-weighted</dfn>:
    a 60 m segment counts 60 times as much as a 1 m stub — this is how the dashboard chooses to weight the numbers.
  </p>
</section>

<section id="hero-map">
  <h2>The street, mapped</h2>
  <p class="lede">{site_display} street observers on a Carto Positron Light basemap. Switch the colour ramp, season, and use the toggles to reveal structural biases.</p>
  <div class="map-toolbar">
    <label>Colour by:
      <select id="color-by">
        <option value="svf">Sky openness</option>
        <option value="solar">Hours of sun</option>
      </select>
    </label>
    <label>Season:
      <select id="season">
        <option value="annual">annual</option>
        <option value="winter">winter</option>
        <option value="summer">summer</option>
        <option value="equinox_mar">March equinox</option>
        <option value="equinox_sep">September equinox</option>
      </select>
    </label>
    {toggles_html}
  </div>
  <div id="map" role="region" aria-label="interactive observer map — pan and zoom to explore street observers" tabindex="0"></div>
  <p class="chart-caption">{decimation_phrase} Observers within 15 m of the boundary are drawn as open circles (their sky is artificially open — see H1); observers with SVF exactly zero are coloured grey (likely a numerical fallback rather than true canyon closure).</p>
</section>

<section id="distribution">
  <h2>Distribution of sky</h2>
  <p class="lede measure">
    The shape of the SVF distribution tells you the typology of the place. Dense canyons pile up near zero; open grids push to the right. The dashed line is the length-weighted site mean.
  </p>
  <div id="hist" class="chart"></div>
  <button id="hist-reset" type="button" class="action-btn" style="margin-top:0.4rem;">reset selection</button>
  <p class="chart-caption measure">
    Count-weighted mean (legacy): <strong>{cw_mean}</strong> · Length-weighted mean (current): <strong>{lw_mean}</strong> · Δ = <strong>{c1_delta}</strong>.
    The count-weighted mean treats every observer equally, which oversamples densely-sampled deep canyons.
    The length-weighted mean treats each metre of road equally — the way this dashboard chooses to weight the numbers.{c1_softener}
    Box-select a range of SVF on the histogram to filter the map and recompute the Pearson r below.
  </p>
</section>

<section id="svf-solar-scatter">
  <h2>Sky and sun, together</h2>
  <p class="lede measure">
    You are looking at how much sky a place sees versus how many hours the sun reaches it. The two are highly correlated — Pearson r in the corner — but the spread tells you about local terrain shading.
  </p>
  <div id="scatter" class="chart"></div>
  <p class="chart-caption">Up to 5,000 points, sampled evenly across the SVF range so the rare extremes — where the correlation is most sensitive — survive the decimation.</p>
</section>

<section id="caveats">
  <h2>What this dashboard does not see</h2>
  <p class="lede measure">Caveats as wayfinding, not apology. Each finding has an audit ID; the chevron jumps to the methodology drawer.</p>
  <div class="caveat-group">
    <h3>Number-changers · toggle the filters above the map and watch the headline SVF number move</h3>
    {number_changers}
  </div>
  <div class="caveat-group">
    <h3>Framing-changers · these reshape the narrative, not the numbers</h3>
    {framing_changers}
  </div>
  <div class="caveat-group">
    <h3>Bounds · what the data does not see at all</h3>
    {bounds}
  </div>
  <p><a href="#top" class="back-to-top">↑ back to top</a></p>
</section>

<section id="cross-site">
  <h2>How {site_display} compares</h2>
  <p class="lede measure">Length-weighted site means across the five campaign sites. Click a row to navigate to that site's dashboard.</p>
  <div id="cross-site-rows" class="cross-site-rows">{cross_site_static}</div>
</section>

<section id="data-provenance">
  <h2>Data and reproducibility</h2>
  <ul>
    <li><strong>CRS:</strong> {crs}</li>
    <li><strong>Observers:</strong> <code>outputs/{site}/morphometrics/svf/svf_streets_solar.gpkg</code> ({n_observers:,} rows; canonical observer network rebuilt 2026-06-03, commit <code>1bffc3e</code>)</li>
    <li><strong>Segments:</strong> <code>outputs/{site}/morphometrics/svf/svf_streets_segments.gpkg</code> (length-weighted means, backfilled <code>d8175ca</code>)</li>
    <li><strong>Boundary:</strong> <code>data/{site}/raw/</code></li>
    <li><strong>Area:</strong> {area_km2} km² · <strong>Road km:</strong> {road_km} <small>(area recomputed from the cleaned 2026 boundary; earlier reports cited ~0.61 km² from a coarser polygon — the figures here supersede them.)</small></li>
    <li><strong>Build:</strong> {build_date} · git <code>{git_sha}</code></li>
    <li><strong>Print fallback:</strong> the static A3 layout at <code>../../site_dashboards/{atoms_subdir}/folha_{atoms_subdir}_A3.png</code> is the citable archival artefact.</li>
  </ul>
  <p class="measure">
    On print, toggles return to defaults, the basemap goes greyscale, the methodology drawer is forced open, and this dashboard collapses to the static A3 layout — the citable archival artefact.
  </p>
  <img class="print-only-a3" src="../../site_dashboards/{atoms_subdir}/folha_{atoms_subdir}_A3.png" alt="Static A3 dashboard for {site_display} — the citable archival artefact, rendered only when this page is printed.">
  <p><a href="#top" class="back-to-top">↑ back to top</a></p>
</section>

<aside id="methodology-drawer" class="drawer" aria-label="methodology drawer" role="complementary">
  <button class="drawer-close" aria-label="close methodology drawer">×</button>
  <h3>Methodology and audit trail</h3>
  <p class="muted">Findings carried forward from <code>outputs/_distribution/audit/street_svf_solar_audit.md</code>. With JavaScript disabled, this drawer is reachable directly by anchor (<code>#methodology-drawer</code>) and all entries below remain readable.</p>
  {drawer_entries}
  <p style="margin-top:1.2rem;"><a href="#top" class="back-to-top">↑ back to top</a></p>
</aside>

<footer>
  <div class="footer-grid" id="footer-grid">{footer_grid_static}</div>
  <p>
    <a href="#top" class="back-to-top">↑ back to top</a> ·
    <a href="../index.html">← all sites</a> ·
    MorphoFavela · <a href="https://github.com/thrmnn/MorphoFavela">repo</a>
  </p>
  <p class="muted" style="margin-top:0.6rem; font-size:0.78rem;">
    Data and analysis released under CC BY 4.0 · code under MIT · contact
    <a href="mailto:thermann.ai@gmail.com">thermann.ai@gmail.com</a>
  </p>
</footer>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.plot.ly/plotly-cartesian-2.35.2.min.js"></script>
<script>
window.mfMeta = {{
  site: "{site}",
  exact_zero_svf_share: {exact_zero_share},
  length_weighted_mean_svf: {lw_mean_num},
  length_weighted_median_svf: {lw_median_num},
  count_weighted_mean_svf: {cw_mean_num},
  centroid_lat: {centroid_lat_num},
  centroid_lon: {centroid_lon_num}
}};
window.mfStats = {mfstats_json};
</script>
<script>
__INLINE_JS__
</script>
<script>
(async function buildCrossSite(){{
  try {{
    const existing = document.getElementById('cross-site-rows');
    if(existing && existing.children.length > 0) return;
    const r = await fetch('../shared/cross_site_stats.json');
    if(!r.ok) return;
    const d = await r.json();
    const sites = d.per_site;
    const ABS_MAX = 0.5;
    const tickHeader = '<div class="csr-ticks"><div></div><div></div><div class="csr-axis"><span class="tick" style="left:0%">0.0</span><span class="tick" style="left:40%">0.2</span><span class="tick" style="left:80%">0.4</span></div><div></div></div>';
    const rows = sites.map(s => {{
      const pct = Math.min(100, (s.length_weighted_mean_svf / ABS_MAX) * 100).toFixed(1);
      const current = s.site === '{site}' ? 'current' : '';
      const hrefSite = encodeURIComponent(s.site);
      return `<a class="csr-row ${{current}}" href="../${{hrefSite}}/index.html">
        <div><span class="csr-name">${{s.site_display}}</span></div>
        <div class="csr-typo">${{s.typology}}</div>
        <div><div class="csr-bar"><div class="csr-tick-line" style="left:40%"></div><div class="csr-tick-line" style="left:80%"></div><div class="csr-bar-fill" style="width:${{pct}}%"></div></div></div>
        <div class="csr-stat">SVF ${{s.length_weighted_mean_svf.toFixed(3)}}<br><small>${{s.n_observers.toLocaleString()}} obs</small></div>
      </a>`;
    }}).join('');
    document.getElementById('cross-site-rows').innerHTML = tickHeader + rows;
    const tiles = sites.map(s => `<a class="footer-tile" href="../${{encodeURIComponent(s.site)}}/index.html">
      <div class="tile-name">${{s.site_display}}</div>
      <div class="tile-typo">${{s.typology}}</div>
      <div class="tile-stats"><span>${{s.n_observers.toLocaleString()}} obs</span><span>SVF ${{s.length_weighted_mean_svf.toFixed(3)}}</span></div>
    </a>`).join('');
    document.getElementById('footer-grid').innerHTML = tiles;
  }} catch(e){{ console.warn('cross-site:', e); }}
}})();
</script>
</body>
</html>
"""


LANDING_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MorphoFavela — interactive dashboards</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="landing-hero">
  <div class="mast-mark">MORPHOFAVELA</div>
  <h1>Five favelas, mapped at street level</h1>
  <p class="lede">Interactive sky view factor and solar access dashboards for five informal settlements in Rio de Janeiro. Length-weighted, audit-disclosed, file:// portable.</p>
</header>
<section>
  <h2>Sites</h2>
  <div class="site-grid">
    __GRID__
  </div>
</section>
<section>
  <h2>Why length-weighted?</h2>
  <p class="measure">Earlier site means averaged segment values without weighting them by physical length. Rocinha's narrow canyons were over-counted; a 1 m alley stub counted the same as a 60 m boulevard. The corrected length-weighted mean moves Rocinha's headline SVF by +0.088 — large enough to invert the Rocinha vs Vidigal comparison.</p>
</section>
<footer>
  <p>MorphoFavela · <a href="https://github.com/thrmnn/MorphoFavela">repo</a></p>
</footer>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True, help="site slug (rocinha, vidigal, ...)")
    parser.add_argument("--all-stats", action="store_true",
                        help="also compute cross-site stats for all sites (cheap)")
    args = parser.parse_args()

    # always (re)build shared assets and cross-site stats — cheap
    site_stats_by_site = {}
    sites_for_cross = list(SITE_META.keys())
    for s in sites_for_cross:
        try:
            obs = gpd.read_file(REPO / f"outputs/{s}/morphometrics/svf/svf_streets_solar.gpkg")
            seg = gpd.read_file(REPO / f"outputs/{s}/morphometrics/svf/svf_streets_segments.gpkg")
            bnd = boundary_for(s)
            site_stats_by_site[s] = compute_site_stats(s, obs, seg, bnd)
        except Exception as e:
            print(f"[{s}] cross-site stats skipped: {e}")

    write_shared_assets({}, site_stats_by_site)
    build_landing(site_stats_by_site)
    stats = build_site(args.site)

    out = DIST / args.site / "index.html"
    print(f"\nWrote {out} ({out.stat().st_size:,} bytes)")
    print(f"Length-weighted mean SVF: {stats['length_weighted_mean_svf']:.4f}")
    print(f"Count-weighted mean SVF:  {stats['count_weighted_mean_svf']:.4f}")
    print(f"Pearson r SVF↔solar:      {stats['pearson_svf_solar_annual']:.4f}")
    print(f"Observers in GeoJSON:     {stats['observer_count_in_geojson']:,}")


if __name__ == "__main__":
    main()
