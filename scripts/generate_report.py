#!/usr/bin/env python3
"""
Technical Report Generator for IVF Urban Morphology Analysis.

Collects analysis outputs for one or more areas and generates a technical
report in Markdown and/or PDF format.

Usage:
    python scripts/generate_report.py --area rocinha --format both
    python scripts/generate_report.py --area rocinha --format markdown
    python scripts/generate_report.py --areas rocinha,vidigal --mode comparative --format pdf
"""

import argparse
import json
import logging
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

# ── Project imports ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SUPPORTED_AREAS,
    EXPECTED_CRS,
    MIN_BUILDING_AREA,
    MAX_FILTER_HEIGHT,
    MAX_FILTER_AREA,
    MAX_FILTER_VOLUME,
    BUILDING_CLUSTER_BUFFER,
    DPI,
    get_area_output_dir,
    get_area_analysis_dir,
    get_comparative_analysis_dir,
    is_formal_area,
)
from src.cartography import apply_publication_style

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Swiss design palette ─────────────────────────────────────────────────
SWISS = {
    "primary": "#1A1A1A",        # Near-black
    "secondary": "#5C5C5C",      # Medium gray
    "accent": "#2A5FA5",         # Deeper blue
    "accent_light": "#E8EFF7",   # Very light blue for backgrounds
    "background": "#FFFFFF",
    "light_gray": "#F2F2F2",     # Slightly warmer gray
    "text": "#1A1A1A",
    "grid": "#D9D9D9",
    "caption": "#666666",        # For figure captions
}

PAGE_SIZE = (8.5, 11)  # US Letter, portrait

# ── Consistent page layout margins ──────────────────────────────────────
MARGIN = {"left": 0.08, "right": 0.92, "top": 0.93, "bottom": 0.06}

# ── Figure numbering counter ────────────────────────────────────────────
_figure_counter = 0


def _reset_figure_counter():
    """Reset figure counter at start of each report."""
    global _figure_counter
    _figure_counter = 0


def _next_figure_number() -> int:
    """Increment and return the next figure number."""
    global _figure_counter
    _figure_counter += 1
    return _figure_counter


# ── Valid section names for --sections filter ────────────────────────────
ALL_SECTIONS = [
    "title", "summary", "methodology", "theory", "parameters",
    "results", "comparative", "figures",
]

# ── Human-readable metric labels ────────────────────────────────────────
METRIC_LABELS = {
    "height": "Building Height (m)",
    "shape_index": "Shape Index",
    "shared_walls": "Shared Wall Length (m)",
    "building_adjacency": "Building Adjacency Count",
    "tessellation_area": "Tessellation Area (m\u00b2)",
    "squareness": "Squareness",
    "elongation": "Elongation",
    "convexity": "Convexity",
    "covered_area_ratio": "Covered Area Ratio",
    "inter_building_distance": "Inter-building Distance (m)",
    "area": "Footprint Area (m\u00b2)",
    "perimeter": "Perimeter (m)",
    "fractal_dimension": "Fractal Dimension",
    "hw_ratio": "Height-to-Width Ratio",
    "volume": "Volume (m\u00b3)",
    "bcr": "BCR",
    "far": "FAR",
    "sigma_h": "\u03c3_h",
    "lambda_f": "\u03bb_f",
}


def _label(col: str) -> str:
    """Return a human-readable label for a metric column name."""
    return METRIC_LABELS.get(col, col.replace("_", " ").title())


# ── Human-readable area display names ─────────────────────────────────
AREA_DISPLAY_NAMES = {
    "vidigal": "Vidigal",
    "vidigal_tls": "Vidigal (TLS)",
    "rocinha": "Rocinha",
    "riodaspedras": "Rio das Pedras",
    "complexo_do_alemao": "Complexo do Alemão",
    "maré": "Maré",
    "cidade_de_deus": "Cidade de Deus",
    "copacabana": "Copacabana",
}


def _area_name(slug: str) -> str:
    """Return a human-readable display name for an area slug."""
    return AREA_DISPLAY_NAMES.get(slug, slug.replace("_", " ").title())


# ═══════════════════════════════════════════════════════════════════════════
#  DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

def collect_area_data(area: str) -> Dict[str, Any]:
    """
    Collect all available analysis outputs for a single area.

    Returns a dict with keys for each analysis type.  Missing outputs are
    set to None so downstream code can degrade gracefully.
    """
    logger.info("Collecting data for %s ...", area)
    out: Dict[str, Any] = {
        "area": area,
        "area_type": "formal" if is_formal_area(area) else "informal",
    }

    # ── 1. Morphology metrics (building-level) ───────────────────────────
    morph_dir = get_area_analysis_dir(area, "morphology_metrics")
    buildings_path = morph_dir / "buildings_with_morphology_metrics.gpkg"
    stats_path = morph_dir / "morphology_summary_stats.csv"

    if buildings_path.exists():
        gdf = gpd.read_file(buildings_path)
        out["buildings"] = gdf
        out["building_count"] = len(gdf)
        bounds = gdf.total_bounds  # minx, miny, maxx, maxy
        out["study_area_m2"] = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        area_km2 = out["study_area_m2"] / 1e6
        out["building_density"] = out["building_count"] / area_km2 if area_km2 > 0 else 0
        logger.info("  buildings: %d", out["building_count"])

        # ── Building height statistics ──────────────────────────────────
        height_col = None
        for candidate in ["height", "altura"]:
            if candidate in gdf.columns:
                height_col = candidate
                break
        if height_col is not None:
            h = gdf[height_col].dropna()
            h = h[h > 0]
            if len(h) > 0:
                out["height_mean"] = float(h.mean())
                out["height_std"] = float(h.std())
                out["height_median"] = float(h.median())
                out["height_max"] = float(h.max())
                out["height_min"] = float(h.min())
                logger.info("  height stats: mean=%.2f, max=%.2f", out["height_mean"], out["height_max"])
            else:
                logger.warning("  no valid height values found")
        else:
            logger.warning("  no height column found in buildings GeoDataFrame")

        # ── Building-level morphometric means for comparison ────────────
        for col in ["shared_walls", "building_adjacency", "shape_index",
                     "squareness", "tessellation_area", "elongation",
                     "convexity", "inter_building_distance", "covered_area_ratio"]:
            if col in gdf.columns:
                vals = gdf[col].dropna()
                if len(vals) > 0:
                    out[f"mean_{col}"] = float(vals.mean())
    else:
        out["buildings"] = None
        out["building_count"] = 0
        out["study_area_m2"] = 0
        out["building_density"] = 0
        logger.warning("  buildings file not found: %s", buildings_path)

    if stats_path.exists():
        out["morphology_stats"] = pd.read_csv(stats_path, index_col=0)
        logger.info("  morphology_summary_stats loaded")
    else:
        out["morphology_stats"] = None

    # ── 2. SVF v2 (street-level) ─────────────────────────────────────────
    svf_dir = get_area_analysis_dir(area, "svf_v2")
    for fname in ["svf_streets_segments.gpkg", "svf_streets.gpkg"]:
        seg_path = svf_dir / fname
        if seg_path.exists():
            seg = gpd.read_file(seg_path)
            out["svf_segments"] = seg
            if "svf_mean" in seg.columns:
                vals = seg["svf_mean"].dropna()
                out["svf_mean"] = float(vals.mean())
                out["svf_std"] = float(vals.std())
                out["svf_median"] = float(vals.median())
                out["svf_min"] = float(vals.min())
                out["svf_max"] = float(vals.max())

            # Segment length diagnostics (favela-relevant)
            seg_lengths = seg.geometry.length
            out["svf_seg_count"] = len(seg)
            out["svf_seg_length_mean"] = float(seg_lengths.mean())
            out["svf_seg_length_median"] = float(seg_lengths.median())
            n_short = int((seg_lengths < 10).sum())
            n_long = int((seg_lengths > 100).sum())
            out["svf_seg_pct_short"] = 100.0 * n_short / len(seg) if len(seg) else 0.0
            out["svf_seg_pct_long"] = 100.0 * n_long / len(seg) if len(seg) else 0.0
            # Within-segment SVF range as proxy for averaging loss
            if "svf_max" in seg.columns and "svf_min" in seg.columns:
                svf_range_vals = seg["svf_max"] - seg["svf_min"]
                out["svf_seg_range_mean"] = float(svf_range_vals.mean())
            if "n_points" in seg.columns:
                n_single = int((seg["n_points"] == 1).sum())
                out["svf_seg_pct_single_pt"] = 100.0 * n_single / len(seg) if len(seg) else 0.0

            logger.info("  SVF segments: %d rows from %s", len(seg), fname)
            break
    else:
        out["svf_segments"] = None

    # Also look for street-level SVF stats CSV (alternate directories)
    for svf_stats_dir_name in ["svf_v2", "svf_street_s3_s300", "svf_street"]:
        svf_stats_dir = get_area_analysis_dir(area, svf_stats_dir_name)
        svf_stats_path = svf_stats_dir / "street_svf_statistics.csv"
        if svf_stats_path.exists():
            out["svf_street_stats"] = pd.read_csv(svf_stats_path)
            logger.info("  SVF street stats loaded from %s", svf_stats_dir_name)
            break
    else:
        out["svf_street_stats"] = None

    # ── 3. Urban morphology (zone-level) ─────────────────────────────────
    urban_dir = get_area_analysis_dir(area, "urban_morphology")
    zone_path = urban_dir / "zone_metrics.gpkg"
    moran_path = urban_dir / "moran_summary.csv"

    if zone_path.exists():
        zm = gpd.read_file(zone_path)
        out["zone_metrics"] = zm
        for col in ["bcr", "far", "sigma_h", "lambda_f"]:
            if col in zm.columns:
                vals = zm[col].dropna()
                out[f"zone_{col}_mean"] = float(vals.mean())
                out[f"zone_{col}_std"] = float(vals.std())
        logger.info("  zone_metrics: %d zones", len(zm))
    else:
        out["zone_metrics"] = None

    if moran_path.exists():
        out["moran_summary"] = pd.read_csv(moran_path)
        logger.info("  moran_summary loaded")
    else:
        out["moran_summary"] = None

    # ── 4. Discover existing figures ─────────────────────────────────────
    out["figures"] = _discover_figures(area)
    logger.info("  discovered %d figures", len(out["figures"]))

    return out


def _discover_figures(area: str) -> Dict[str, Path]:
    """Scan output directories for embeddable PNG figures."""
    figs: Dict[str, Path] = {}

    search_dirs = {
        "svf_v2": get_area_analysis_dir(area, "svf_v2"),
        "morphology_metrics": get_area_analysis_dir(area, "morphology_metrics"),
        "urban_morphology": get_area_analysis_dir(area, "urban_morphology"),
    }

    for label, d in search_dirs.items():
        if not d.exists():
            continue
        for png in sorted(d.rglob("*.png")):
            key = f"{label}/{png.stem}"
            figs[key] = png

    return figs


# ═══════════════════════════════════════════════════════════════════════════
#  FORMATTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(val: float, decimals: int = 3) -> str:
    """Format a float for display, handling None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_int(val) -> str:
    """Format an integer with comma separators."""
    if val is None:
        return "N/A"
    return f"{int(val):,}"


def _fmt_auto(val, decimals: int = 2) -> str:
    """Format a number: integers with commas, floats to given decimals."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if abs(val) >= 1000:
        return f"{val:,.{decimals}f}"
    return f"{val:.{decimals}f}"


def _section_header(level: int, title: str) -> str:
    return f"\n{'#' * level} {title}\n"


# ═══════════════════════════════════════════════════════════════════════════
#  NARRATIVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _narrative_summary(d: Dict[str, Any]) -> str:
    """Generate a brief narrative synthesis for one area."""
    area = _area_name(d["area"])
    atype = d["area_type"]
    parts = []

    parts.append(
        f"{area} is {'a formally planned' if atype == 'formal' else 'an informal'} "
        f"settlement comprising {d['building_count']:,} buildings "
        f"across {d['study_area_m2']/1e6:.2f} km\u00b2 "
        f"({d['building_density']:,.0f} buildings/km\u00b2)."
    )

    if d.get("height_mean") is not None:
        parts.append(
            f"Buildings have a mean height of {d['height_mean']:.1f} m "
            f"(max {d['height_max']:.1f} m, \u03c3 = {d['height_std']:.1f} m)."
        )

    if d.get("svf_mean") is not None:
        svf = d["svf_mean"]
        qualifier = "very low" if svf < 0.15 else "low" if svf < 0.3 else "moderate"
        parts.append(
            f"Street-level sky view factor is {qualifier} "
            f"(mean SVF = {svf:.3f}, range {d['svf_min']:.3f}\u2013{d['svf_max']:.3f}), "
            f"reflecting {'severe sky obstruction typical of dense informal fabric' if svf < 0.15 else 'compact urban morphology with restricted sky access' if svf < 0.3 else 'reasonable sky exposure'}."
        )

    bcr = d.get("zone_bcr_mean")
    far = d.get("zone_far_mean")
    if bcr is not None and far is not None:
        parts.append(
            f"Zone-level analysis yields a mean BCR of {bcr:.3f} and FAR of {far:.3f}."
        )

    sigma_h = d.get("zone_sigma_h_mean")
    lambda_f = d.get("zone_lambda_f_mean")
    if sigma_h is not None:
        parts.append(
            f"Height variability (\u03c3_h = {sigma_h:.2f} m) "
            + (f"and frontal area index (\u03bb_f = {lambda_f:.3f}) " if lambda_f is not None else "")
            + "characterize the vertical roughness of the urban canopy."
        )

    return " ".join(parts)


def _interpret_bcr(bcr: float) -> str:
    """Contextual interpretation of BCR values."""
    if bcr > 0.6:
        return (
            f"Very high BCR ({bcr:.3f}) indicates extremely dense ground coverage, "
            "leaving minimal open space for airflow and ventilation. "
            "This level is characteristic of consolidated informal settlements."
        )
    elif bcr > 0.5:
        return (
            f"High BCR ({bcr:.3f}) indicates dense ground coverage with limited "
            "open space. At this density, natural ventilation is significantly "
            "impaired, particularly at the pedestrian level."
        )
    elif bcr > 0.3:
        return (
            f"Moderate BCR ({bcr:.3f}) is within the expected range for informal "
            "settlements, allowing some airflow corridors between structures."
        )
    else:
        return f"Relatively low BCR ({bcr:.3f}) suggests more open urban fabric."


def _interpret_sigma_h(sigma_h: float) -> str:
    """Contextual interpretation of sigma_h values."""
    if sigma_h > 3.0:
        return (
            f"High height variability (\u03c3_h = {sigma_h:.2f} m) creates "
            "significant vertical roughness, promoting turbulent mixing but also "
            "channeling effects in narrow passages."
        )
    elif sigma_h > 1.5:
        return (
            f"Moderate height variability (\u03c3_h = {sigma_h:.2f} m) indicates "
            "a heterogeneous building stock typical of incrementally built settlements."
        )
    else:
        return (
            f"Low height variability (\u03c3_h = {sigma_h:.2f} m) suggests "
            "relatively uniform building heights."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  MARKDOWN GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_markdown(areas_data: List[Dict[str, Any]], mode: str = "single") -> str:
    """Build a complete Markdown report string."""
    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d")
    area_names = [d["area"] for d in areas_data]

    # ── Title page ──────────────────────────────────────────────────────
    lines.append("# Urban Morphology and Environmental Performance Analysis of Informal Settlements in Rio de Janeiro\n")
    area_label = ", ".join(_area_name(a) for a in area_names)
    if mode == "comparative":
        lines.append(f"## Comparative Study: {area_label}\n")
    else:
        lines.append(f"## Area Study: {area_label}\n")
    lines.append("**Informal Ventilation Flows (IVF) Project**")
    lines.append("Massachusetts Institute of Technology")
    lines.append(f"**Date:** {now}")
    lines.append("")
    lines.append("---\n")

    # ── Table of Contents ────────────────────────────────────────────────
    lines.append("## Table of Contents\n")
    toc = [
        "1. [Executive Summary](#1-executive-summary)",
        "2. [Methodology & Workflow](#2-methodology--workflow)",
        "3. [Parameters & Configuration](#3-parameters--configuration)",
        "4. [Morphometric Analysis](#4-morphometric-analysis)",
    ]
    if mode == "comparative":
        toc.append("5. [Comparative Analysis](#5-comparative-analysis)")
    toc.extend([
        "6. [Discussion](#6-discussion)",
        "7. [Recommendations & Next Steps](#7-recommendations--next-steps)",
    ])
    lines.extend(toc)
    lines.append("")

    # ── 1. Executive Summary ─────────────────────────────────────────────
    lines.append(_section_header(2, "1. Executive Summary"))

    for d in areas_data:
        lines.append(f"### {_area_name(d['area'])} ({d['area_type'].title()})\n")

        # Narrative paragraph
        lines.append(_narrative_summary(d))
        lines.append("")

        # Concise key metrics table (6-8 most important)
        lines.append("| Metric | Value |")
        lines.append("|:-------|------:|")
        lines.append(f"| Buildings | {d['building_count']:,} |")
        lines.append(f"| Study Area | {d['study_area_m2']/1e6:.2f} km\u00b2 |")
        lines.append(f"| Building Density | {d['building_density']:,.0f} /km\u00b2 |")
        if d.get("height_mean") is not None:
            lines.append(f"| Mean Building Height | {d['height_mean']:.1f} m |")
            lines.append(f"| Max Building Height | {d['height_max']:.1f} m |")
        if d.get("svf_mean") is not None:
            lines.append(f"| Mean SVF | {_fmt(d['svf_mean'])} |")
        if d.get("zone_bcr_mean") is not None:
            lines.append(f"| Mean BCR | {_fmt(d['zone_bcr_mean'])} |")
        if d.get("zone_lambda_f_mean") is not None:
            lines.append(f"| Mean \u03bb_f | {_fmt(d['zone_lambda_f_mean'])} |")
        lines.append("")

    # ── 2. Methodology & Workflow ────────────────────────────────────────
    lines.append(_section_header(2, "2. Methodology & Workflow"))
    lines.append("The IVF analysis pipeline comprises the following phases:\n")
    phases = [
        ("Data Acquisition & Preprocessing",
         "Building footprints, DTM rasters, and road networks are loaded, "
         "reprojected to UTM 23S (EPSG:32723), and filtered for informal settlements "
         "(height, area, volume, h/w ratio thresholds)."),
        ("Morphology Metrics",
         "25 building-level morphometric indicators computed via momepy/libpysal: "
         "shape index, compactness, elongation, tessellation, adjacency, shared walls, etc."),
        ("Sky View Factor (SVF)",
         "Street-level SVF computed along road centerlines using 3D ray-casting "
         "against building volumes and DTM terrain. Results are segment-averaged."),
        ("Urban Morphology (Zone-Level)",
         "Grid-based zoning (tessellation) with BCR, FAR, \u03c3_h, \u03bb_f, "
         "and street orientation entropy per zone. Spatial autocorrelation via "
         "Moran's I and LISA clusters."),
        ("Patch Selection for CFD",
         "Grid tiling of study area; 31 morphometric features per tile; PCA + "
         "k-means clustering; representative patch selection for OpenFOAM simulation."),
    ]
    for i, (name, desc) in enumerate(phases, 1):
        lines.append(f"**Phase {i}: {name}**")
        lines.append(f"{desc}\n")

    # ── 3. Parameters & Configuration ────────────────────────────────────
    lines.append(_section_header(2, "3. Parameters & Configuration"))
    lines.append("| Parameter | Value |")
    lines.append("|:----------|------:|")
    lines.append(f"| CRS | {EXPECTED_CRS} |")
    lines.append(f"| Min Building Area | {MIN_BUILDING_AREA} m\u00b2 |")
    lines.append(f"| Max Filter Height (informal) | {MAX_FILTER_HEIGHT} m |")
    lines.append(f"| Max Filter Area (informal) | {MAX_FILTER_AREA:,} m\u00b2 |")
    lines.append(f"| Max Filter Volume (informal) | {MAX_FILTER_VOLUME:,} m\u00b3 |")
    lines.append(f"| Building Cluster Buffer | {BUILDING_CLUSTER_BUFFER} m |")
    lines.append(f"| Output DPI | {DPI} |")
    lines.append("")

    # ── 4. Morphometric Analysis ─────────────────────────────────────────
    lines.append(_section_header(2, "4. Morphometric Analysis"))

    # Key morphometric columns for informal settlement analysis
    MORPH_KEY_COLS = [
        "height", "shape_index", "shared_walls", "building_adjacency",
        "tessellation_area", "squareness", "elongation", "convexity",
        "covered_area_ratio", "inter_building_distance",
    ]

    for d in areas_data:
        area = d["area"]
        lines.append(f"### 4.{areas_data.index(d)+1}. {_area_name(area)}\n")

        # 4.x.1 Building-Level Morphology
        if d.get("morphology_stats") is not None:
            lines.append("#### Building-Level Morphology\n")
            mstats = d["morphology_stats"]
            available = [c for c in MORPH_KEY_COLS if c in mstats.columns]
            if available:
                lines.append("| Metric | Count | Mean | Std | Min | Max |")
                lines.append("|:-------|------:|-----:|----:|----:|----:|")
                for col in available:
                    row = mstats[col]
                    lines.append(
                        f"| {_label(col)} "
                        f"| {_fmt_int(row.get('count', 0))} "
                        f"| {_fmt_auto(row.get('mean', None))} "
                        f"| {_fmt_auto(row.get('std', None))} "
                        f"| {_fmt_auto(row.get('min', None))} "
                        f"| {_fmt_auto(row.get('max', None))} |"
                    )
                lines.append("")

        # 4.x.2 SVF
        if d.get("svf_mean") is not None:
            lines.append("#### Sky View Factor (SVF)\n")
            lines.append(f"- **Mean SVF:** {_fmt(d['svf_mean'])}")
            lines.append(f"- **Median SVF:** {_fmt(d['svf_median'])}")
            lines.append(f"- **Std SVF:** {_fmt(d['svf_std'])}")
            lines.append(f"- **Range:** {_fmt(d['svf_min'])} \u2013 {_fmt(d['svf_max'])}")
            if d.get("svf_segments") is not None:
                lines.append(f"- **Street segments analysed:** {len(d['svf_segments']):,}")

            # Segment length diagnostics
            if d.get("svf_seg_count") is not None:
                lines.append("")
                lines.append("**Segment length diagnostics:**\n")
                lines.append(
                    f"- **Mean / median segment length:** "
                    f"{d['svf_seg_length_mean']:.1f} m / "
                    f"{d['svf_seg_length_median']:.1f} m"
                )
                lines.append(
                    f"- **Very short segments (<10 m):** "
                    f"{d['svf_seg_pct_short']:.1f}%"
                )
                lines.append(
                    f"- **Long segments (>100 m):** "
                    f"{d['svf_seg_pct_long']:.1f}%"
                )
                if d.get("svf_seg_range_mean") is not None:
                    lines.append(
                        f"- **Mean within-segment SVF range:** "
                        f"{d['svf_seg_range_mean']:.3f}"
                    )
                if d.get("svf_seg_pct_single_pt") is not None:
                    lines.append(
                        f"- **Single-point segments:** "
                        f"{d['svf_seg_pct_single_pt']:.1f}%"
                    )
            lines.append("")

        # 4.x.3 Zone-level metrics
        if d.get("zone_metrics") is not None:
            lines.append("#### Zone-Level Urban Morphology\n")
            zm = d["zone_metrics"]
            lines.append(f"- **Zones analysed:** {len(zm):,}")
            for col in ["bcr", "far", "sigma_h", "lambda_f"]:
                if col in zm.columns:
                    vals = zm[col].dropna()
                    lines.append(
                        f"- **{_label(col)}:** mean = {_fmt(vals.mean())}, "
                        f"\u03c3 = {_fmt(vals.std())}, "
                        f"median = {_fmt(vals.median())}"
                    )
            lines.append("")

            # Contextual interpretation
            bcr = d.get("zone_bcr_mean")
            if bcr is not None:
                lines.append(f"> {_interpret_bcr(bcr)}")
                lines.append("")
            sigma_h = d.get("zone_sigma_h_mean")
            if sigma_h is not None:
                lines.append(f"> {_interpret_sigma_h(sigma_h)}")
                lines.append("")

            # Moran's I
            if d.get("moran_summary") is not None:
                lines.append("**Spatial Autocorrelation (Moran's I):**\n")
                ms = d["moran_summary"]
                lines.append("| Variable | I | Expected I | z-score | p-value |")
                lines.append("|:---------|--:|----------:|--------:|--------:|")
                for _, row in ms.iterrows():
                    col_name = row.get("column", "")
                    lines.append(
                        f"| {_label(col_name)} "
                        f"| {_fmt(row.get('I', None), 4)} "
                        f"| {_fmt(row.get('expected_I', None), 6)} "
                        f"| {_fmt(row.get('z_score', None), 2)} "
                        f"| {row.get('p_value', 'N/A')} |"
                    )
                lines.append("")

    # ── 5. Comparative Analysis ──────────────────────────────────────────
    if mode == "comparative" and len(areas_data) > 1:
        lines.append(_section_header(2, "5. Comparative Analysis"))
        lines.append("### Cross-Area Summary\n")

        # Build comparison table
        header_cols = ["Metric"] + [_area_name(d["area"]) for d in areas_data]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|:" + "|".join(["------" + (":" if i > 0 else "") for i in range(len(header_cols))]) + "|")

        metrics_to_compare = [
            ("Buildings", "building_count", ",.0f"),
            ("Study Area (km\u00b2)", "study_area_m2", ".2f", 1e-6),
            ("Density (/km\u00b2)", "building_density", ",.0f"),
            ("Mean Height (m)", "height_mean", ".1f"),
            ("Max Height (m)", "height_max", ".1f"),
            ("Mean SVF", "svf_mean", ".3f"),
            ("Mean BCR", "zone_bcr_mean", ".3f"),
            ("Mean FAR", "zone_far_mean", ".3f"),
            ("Mean \u03c3_h (m)", "zone_sigma_h_mean", ".2f"),
            ("Mean \u03bb_f", "zone_lambda_f_mean", ".3f"),
            ("Mean Shared Walls (m)", "mean_shared_walls", ".1f"),
            ("Mean Adjacency Count", "mean_building_adjacency", ".1f"),
        ]

        for entry in metrics_to_compare:
            name = entry[0]
            key = entry[1]
            fmt = entry[2]
            scale = entry[3] if len(entry) > 3 else 1.0
            row_vals = [name]
            for d in areas_data:
                val = d.get(key)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    row_vals.append(f"{val * scale:{fmt}}")
                else:
                    row_vals.append("N/A")
            lines.append("| " + " | ".join(row_vals) + " |")
        lines.append("")

    # ── 6. Discussion ────────────────────────────────────────────────────
    lines.append(_section_header(2, "6. Discussion"))
    for d in areas_data:
        area = d["area"]
        lines.append(f"### {_area_name(area)}\n")
        issues: List[str] = []

        # SVF interpretation
        if d.get("svf_mean") is not None:
            svf = d["svf_mean"]
            if svf < 0.15:
                issues.append(
                    f"Very low mean SVF ({svf:.3f}) indicates severe sky obstruction -- "
                    "typical of dense informal fabric with narrow alleys."
                )
            elif svf < 0.3:
                issues.append(
                    f"Low mean SVF ({svf:.3f}) suggests restricted sky access "
                    "consistent with compact urban morphology."
                )
            else:
                issues.append(
                    f"Moderate SVF ({svf:.3f}) indicates reasonable sky exposure."
                )

        # BCR interpretation
        bcr = d.get("zone_bcr_mean")
        if bcr is not None:
            issues.append(_interpret_bcr(bcr))

        # sigma_h interpretation
        sigma_h = d.get("zone_sigma_h_mean")
        if sigma_h is not None:
            issues.append(_interpret_sigma_h(sigma_h))

        # lambda_f interpretation
        lambda_f = d.get("zone_lambda_f_mean")
        if lambda_f is not None:
            if lambda_f > 0.3:
                issues.append(
                    f"High frontal area index (\u03bb_f = {lambda_f:.3f}) indicates "
                    "substantial wind obstruction from building facades."
                )
            elif lambda_f > 0.15:
                issues.append(
                    f"Moderate frontal area index (\u03bb_f = {lambda_f:.3f}) suggests "
                    "partial wind channeling through the built environment."
                )

        # Moran's I
        if d.get("moran_summary") is not None:
            ms = d["moran_summary"]
            for _, row in ms.iterrows():
                col_name = row.get("column", "")
                I_val = row.get("I", 0)
                p_val = row.get("p_value", 1)
                if p_val < 0.05 and I_val > 0.5:
                    issues.append(
                        f"Strong positive spatial autocorrelation for {_label(col_name)} "
                        f"(I = {I_val:.3f}, p = {p_val}) -- clear spatial clustering."
                    )

        if not issues:
            issues.append("Insufficient data for detailed interpretation.")

        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")

    # ── 7. Recommendations & Next Steps ──────────────────────────────────
    lines.append(_section_header(2, "7. Recommendations & Next Steps"))
    recommendations = [
        "Run urban morphology (zone-level BCR/FAR/\u03c3_h) for all areas to enable full cross-area comparison.",
        "Execute CFD simulations on selected patches using OpenFOAM to validate ventilation predictions.",
        "Correlate SVF with solar access and thermal comfort field measurements where available.",
        "Extend patch selection to cross-area clustering for settlement typology classification.",
        "Generate deprivation index maps combining morphometric and environmental indicators.",
    ]
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec}")
    lines.append("")

    lines.append("---\n")
    lines.append(
        f"*Report generated on {now} by `scripts/generate_report.py` "
        f"| IVF Project, MIT*"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  PDF GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _apply_swiss_style():
    """Configure matplotlib for Swiss design aesthetics.

    Builds on the publication base style from src.cartography and overlays
    the IVF report-specific typography and colour choices.
    """
    # Start from the shared publication base
    apply_publication_style()

    # Report-specific overrides
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,                 # Body text
        "axes.titlesize": 15,            # Section titles
        "axes.labelsize": 12,            # Subsection / axis labels
        "axes.labelcolor": SWISS["text"],
        "text.color": SWISS["text"],
        "axes.edgecolor": SWISS["secondary"],
        "axes.linewidth": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _pdf_base_page(pdf: PdfPages, area_label: str = "") -> plt.Figure:
    """Create a base page with consistent header/footer.

    Returns the figure so callers can add content. Caller is responsible
    for calling ``pdf.savefig(fig, ...)`` and ``plt.close(fig)`` when done.

    Parameters
    ----------
    pdf : PdfPages
        The target PDF.
    area_label : str
        Area name shown in the footer (left-aligned).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")

    # ── Top accent line (thin) ──────────────────────────────────────
    ax_top = fig.add_axes([MARGIN["left"], 0.955, MARGIN["right"] - MARGIN["left"], 0.002])
    ax_top.add_patch(Rectangle((0, 0), 1, 1, facecolor=SWISS["accent"]))
    ax_top.axis("off")

    # ── Footer: area name (left) | page number (center) | date (right) ──
    date_str = datetime.now().strftime("%B %Y")
    fig.text(MARGIN["left"], 0.025, area_label,
             ha="left", va="center", fontsize=8,
             color=SWISS["secondary"], family="sans-serif")
    fig.text(0.5, 0.025, f"{pdf.get_pagecount() + 1}",
             ha="center", va="center", fontsize=8,
             color=SWISS["secondary"], family="sans-serif")
    fig.text(MARGIN["right"], 0.025, date_str,
             ha="right", va="center", fontsize=8,
             color=SWISS["secondary"], family="sans-serif")

    return fig


def _save_page(pdf: PdfPages, fig: plt.Figure):
    """Save a page figure to the PDF and close it."""
    pdf.savefig(fig, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)


def _pdf_title_page(pdf: PdfPages, areas_data: List[Dict], mode: str):
    """Render a clean, Swiss-academic title page (no header/footer)."""
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")

    # ── Title: 20pt bold ────────────────────────────────────────────
    fig.text(0.5, 0.65,
             "Urban Morphology and\nEnvironmental Performance Analysis\n"
             "of Informal Settlements in Rio de Janeiro",
             ha="center", va="center", fontsize=20, fontweight="bold",
             color=SWISS["text"], family="sans-serif",
             linespacing=1.5)

    # ── Subtitle: area names — 13pt secondary ───────────────────────
    area_label = " / ".join(_area_name(d["area"]) for d in areas_data)
    mode_label = "Comparative Analysis" if mode == "comparative" else "Area Report"
    fig.text(0.5, 0.55, f"{mode_label}: {area_label}",
             ha="center", va="center", fontsize=13,
             color=SWISS["secondary"], family="sans-serif")

    # ── Single accent bar — centered, 40% width ────────────────────
    ax_bar = fig.add_axes([0.30, 0.50, 0.40, 0.003])
    ax_bar.add_patch(Rectangle((0, 0), 1, 1, facecolor=SWISS["accent"]))
    ax_bar.axis("off")

    # ── Institution: 11pt ───────────────────────────────────────────
    fig.text(0.5, 0.42, "Massachusetts Institute of Technology",
             ha="center", va="center", fontsize=11,
             color=SWISS["text"], family="sans-serif")

    # ── Project: 11pt italic ────────────────────────────────────────
    fig.text(0.5, 0.38, "Informal Ventilation Flows (IVF)",
             ha="center", va="center", fontsize=11,
             color=SWISS["text"], family="sans-serif",
             style="italic")

    # ── Date: 11pt ──────────────────────────────────────────────────
    fig.text(0.5, 0.34, datetime.now().strftime("%B %Y"),
             ha="center", va="center", fontsize=11,
             color=SWISS["secondary"], family="sans-serif")

    pdf.savefig(fig, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)


def _pdf_text_page(pdf: PdfPages, title: str, text_lines: List[str],
                   area_label: str = ""):
    """Render a page of text content with proper word wrapping."""
    fig = _pdf_base_page(pdf, area_label=area_label)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Section title — 15pt bold
    ax.text(0.5, MARGIN["top"], title,
            ha="center", va="top", fontsize=15, fontweight="bold",
            color=SWISS["text"], family="sans-serif",
            transform=ax.transAxes)

    # Decorative line under title
    line = plt.Line2D([MARGIN["left"], MARGIN["right"]],
                      [MARGIN["top"] - 0.025, MARGIN["top"] - 0.025],
                      transform=ax.transAxes,
                      color=SWISS["accent"], linewidth=1.0)
    ax.add_line(line)

    # Body text with proper word wrapping
    y = MARGIN["top"] - 0.045
    line_h = 0.022
    para_spacing = 0.012
    wrap_width = 90

    for text in text_lines:
        if not text.strip():
            y -= para_spacing
            continue

        # Bold lines start with **
        is_bold = text.startswith("**")
        display = text.replace("**", "")
        fontsize = 10.5 if is_bold else 10
        weight = "bold" if is_bold else "normal"

        # Word-aware wrapping
        wrapped = textwrap.wrap(display, width=wrap_width) or [""]
        for wline in wrapped:
            if y < MARGIN["bottom"]:
                break
            ax.text(MARGIN["left"], y, wline,
                    transform=ax.transAxes, fontsize=fontsize,
                    fontweight=weight, color=SWISS["text"],
                    family="sans-serif", verticalalignment="top")
            y -= line_h
        y -= para_spacing * 0.3

        if y < MARGIN["bottom"]:
            break

    _save_page(pdf, fig)


def _pdf_table_page(pdf: PdfPages, title: str, df: pd.DataFrame,
                    col_widths: Optional[List[float]] = None,
                    area_label: str = ""):
    """Render a table as a full PDF page."""
    fig = _pdf_base_page(pdf, area_label=area_label)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.5, MARGIN["top"], title,
            ha="center", va="top", fontsize=15, fontweight="bold",
            color=SWISS["text"], family="sans-serif",
            transform=ax.transAxes)

    line = plt.Line2D([MARGIN["left"], MARGIN["right"]],
                      [MARGIN["top"] - 0.025, MARGIN["top"] - 0.025],
                      transform=ax.transAxes,
                      color=SWISS["accent"], linewidth=1.0)
    ax.add_line(line)

    # Truncate to fit on page
    max_rows = 25
    display_df = df.head(max_rows).copy()

    # Format numeric values: integers with commas, floats to 2-3 decimals
    for col in display_df.columns:
        if display_df[col].dtype in [np.float64, np.float32, float]:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:,.2f}" if pd.notna(x) and abs(x) >= 10
                else f"{x:.3f}" if pd.notna(x) else "N/A"
            )
        elif display_df[col].dtype in [np.int64, np.int32, int]:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:,}" if pd.notna(x) else "N/A"
            )

    cell_text = display_df.values.tolist()
    col_labels = list(display_df.columns)

    # Apply human-readable labels to the Metric column if present
    if "Metric" in col_labels:
        metric_idx = col_labels.index("Metric")
        for row in cell_text:
            row[metric_idx] = _label(str(row[metric_idx]))

    if not col_widths:
        col_widths = [1.0 / len(col_labels)] * len(col_labels)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="upper center",
        bbox=[0.05, MARGIN["bottom"], 0.9, 0.84],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    # Style header row — 9pt bold white on accent
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(SWISS["accent"])
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    # Alternate row colors
    for i in range(1, len(cell_text) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor(SWISS["light_gray"])
            else:
                cell.set_facecolor("white")

    _save_page(pdf, fig)


def _pdf_figure_page(pdf: PdfPages, title: str, img_path: Path,
                     caption: Optional[str] = None,
                     area_label: str = ""):
    """Embed an existing PNG figure as a full PDF page.

    Parameters
    ----------
    pdf : PdfPages
        Target PDF.
    title : str
        Figure title (will be prefixed with ``Fig. N --``).
    img_path : Path
        Path to the PNG image.
    caption : str, optional
        Descriptive caption rendered in 9pt italic below the image.
    area_label : str
        Area name for the footer.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig = _pdf_base_page(pdf, area_label=area_label)

    # Auto-number figures
    fig_num = _next_figure_number()
    numbered_title = f"Fig. {fig_num} \u2014 {title}"

    # Figure title at 0.96
    fig.text(0.5, 0.96, numbered_title,
             ha="center", va="top", fontsize=10, fontweight="bold",
             color=SWISS["text"], family="sans-serif")

    # Image area: 0.07 to 0.91 vertically
    img_bottom = 0.07
    img_top = 0.91
    img_left = MARGIN["left"]
    img_right = MARGIN["right"]
    img_width = img_right - img_left
    img_height = img_top - img_bottom

    try:
        img = imread(str(img_path))
        img_h, img_w = img.shape[:2]
        aspect = img_w / img_h

        # Determine layout based on aspect ratio
        if aspect >= 1.2:
            # Landscape: use full width
            ax = fig.add_axes([img_left, img_bottom, img_width, img_height])
        elif aspect >= 0.8:
            # Near-square: slight inset
            inset = 0.02
            ax = fig.add_axes([img_left + inset, img_bottom,
                               img_width - 2 * inset, img_height])
        else:
            # Portrait: center with margin
            w = img_width * 0.75
            ax = fig.add_axes([(1 - w) / 2, img_bottom, w, img_height])

        ax.imshow(img)
        ax.axis("off")

        # Thin 0.5pt gray border around the figure image area
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color(SWISS["grid"])

    except Exception as e:
        logger.warning("Could not embed figure %s: %s", img_path, e)
        fig.text(0.5, 0.5, f"[Figure not available: {e}]",
                 ha="center", va="center", fontsize=10,
                 color=SWISS["secondary"])

    # Caption at 0.04 — 9pt italic
    if caption:
        fig.text(0.5, 0.04, caption,
                 ha="center", va="top", fontsize=9, style="italic",
                 color=SWISS["caption"], family="sans-serif",
                 wrap=True)

    _save_page(pdf, fig)


def _pdf_theory_page(pdf: PdfPages, area_label: str = ""):
    """Render a metric definitions / theory page."""
    theory_lines = [
        "**Zone-Level Metrics:**",
        "",
        "BCR (Building Coverage Ratio): \u03bb_p = \u03a3(A_footprint) / A_zone. "
        "Measures ground coverage density.",
        "",
        "FAR (Floor Area Ratio): \u03a3(A_footprint \u00d7 n_floors) / A_zone. "
        "Volumetric density indicator.",
        "",
        "\u03c3_h (Height Variability): Standard deviation of building heights. "
        "Characterizes vertical roughness of the urban canopy layer.",
        "",
        "\u03bb_f (Frontal Area Index): \u03a3(w_\u22a5 \u00d7 h) / A_zone for wind "
        "direction \u03b8. Proxy for aerodynamic drag.",
        "",
        "Moran\u2019s I: Spatial autocorrelation statistic. "
        "I \u2248 1 indicates clustering, I \u2248 0 randomness, I \u2248 \u22121 dispersion.",
        "",
        "",
        "**Street-Level SVF:**",
        "",
        "SVF (Sky View Factor): Fraction of the hemisphere visible from a point. "
        "Range [0, 1]. Computed via Tregenza 145-patch ray-casting against "
        "building volumes and DTM terrain.",
    ]
    _pdf_text_page(pdf, "Metric Definitions", theory_lines, area_label=area_label)


def _pdf_methodology_diagram(pdf: PdfPages, area_label: str = ""):
    """Render a simple pipeline flowchart using matplotlib."""
    from matplotlib.patches import FancyArrowPatch

    fig = _pdf_base_page(pdf, area_label=area_label)

    fig.text(0.5, MARGIN["top"], "Methodology Pipeline",
             ha="center", va="top", fontsize=15, fontweight="bold",
             color=SWISS["text"], family="sans-serif")

    # Create axes for the diagram
    ax = fig.add_axes([MARGIN["left"], 0.15,
                       MARGIN["right"] - MARGIN["left"], 0.72])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # 5 pipeline boxes
    boxes = [
        (1.0, 3.0, "Data\nAcquisition"),
        (3.0, 3.0, "Morphometry"),
        (5.0, 3.0, "SVF\nComputation"),
        (7.0, 3.0, "Zone\nAnalysis"),
        (9.0, 3.0, "Patch\nSelection"),
    ]

    box_w = 1.5
    box_h = 1.2

    for (cx, cy, label) in boxes:
        rect = plt.Rectangle(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            facecolor="white", edgecolor=SWISS["accent"],
            linewidth=1.5, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color=SWISS["text"], family="sans-serif", zorder=4)

    # Arrows between boxes
    arrow_kw = dict(
        arrowstyle="->,head_width=0.2,head_length=0.15",
        color=SWISS["secondary"], linewidth=1.2, zorder=2,
    )
    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + box_w / 2
        x_end = boxes[i + 1][0] - box_w / 2
        arrow = FancyArrowPatch(
            (x_start, 3.0), (x_end, 3.0), **arrow_kw,
        )
        ax.add_patch(arrow)

    # Input labels on top
    inputs = [
        (1.0, "Footprints"),
        (3.0, "DTM"),
        (5.0, "Roads"),
    ]
    for (cx, label) in inputs:
        ax.text(cx, 3.0 + box_h / 2 + 0.5, label,
                ha="center", va="center", fontsize=8.5, style="italic",
                color=SWISS["secondary"], family="sans-serif")
        ax.annotate("", xy=(cx, 3.0 + box_h / 2),
                    xytext=(cx, 3.0 + box_h / 2 + 0.3),
                    arrowprops=dict(arrowstyle="->",
                                    color=SWISS["grid"], lw=0.8))

    # Output labels on bottom
    outputs = [
        (7.0, "Reports"),
        (9.0, "CFD Patches"),
    ]
    for (cx, label) in outputs:
        ax.text(cx, 3.0 - box_h / 2 - 0.5, label,
                ha="center", va="center", fontsize=8.5, style="italic",
                color=SWISS["secondary"], family="sans-serif")
        ax.annotate("", xy=(cx, 3.0 - box_h / 2 - 0.3),
                    xytext=(cx, 3.0 - box_h / 2),
                    arrowprops=dict(arrowstyle="->",
                                    color=SWISS["grid"], lw=0.8))

    _save_page(pdf, fig)


def generate_pdf(areas_data: List[Dict[str, Any]], output_path: Path,
                 mode: str = "single", sections: Optional[List[str]] = None):
    """Generate a multi-page PDF technical report.

    Parameters
    ----------
    sections : list[str], optional
        Subset of section names to generate. If *None*, all sections are
        rendered. Valid names: title, summary, methodology, theory,
        parameters, results, comparative, figures.
    """
    _apply_swiss_style()
    _reset_figure_counter()
    logger.info("Generating PDF report at %s", output_path)

    # Section filter helper
    if sections is None:
        sections = list(ALL_SECTIONS)

    def _want(name: str) -> bool:
        return name in sections

    # Area label for footers
    area_footer = " / ".join(_area_name(d["area"]) for d in areas_data)

    # Key morphometric columns for tables
    MORPH_KEY_COLS = [
        "height", "shape_index", "shared_walls", "building_adjacency",
        "tessellation_area", "squareness", "elongation", "convexity",
        "covered_area_ratio", "inter_building_distance",
    ]

    # ── Figure captions keyed by figure key ─────────────────────────
    FIGURE_CAPTIONS = {
        "svf_v2/svf_dashboard":
            "Sky View Factor dashboard showing spatial distribution and summary statistics.",
        "svf_v2/svf_distribution":
            "Distribution of street-level SVF values across sampled segments.",
        "svf_v2/svf_streets_map":
            "Spatial map of SVF values along street centerlines.",
        "svf_v2/svf_streets_segments_map":
            "Segment-level SVF map with colour-coded classification.",
        "morphology_metrics/morphology_distributions":
            "Kernel density estimates for key building-level morphometric indicators.",
        "urban_morphology/maps/zone_metrics_panel":
            "Zone-level panel showing BCR, FAR, height variability, and frontal area index.",
        "urban_morphology/maps/lisa_clusters_bcr":
            "LISA cluster map for BCR identifying statistically significant spatial clusters.",
    }

    with PdfPages(output_path) as pdf:

        # ── Title page ───────────────────────────────────────────────────
        if _want("title"):
            _pdf_title_page(pdf, areas_data, mode)

        # ── Executive summary ────────────────────────────────────────────
        if _want("summary"):
            for d in areas_data:
                summary_lines = [
                    f"**{_area_name(d['area'])} ({d['area_type'].title()})**",
                    "",
                    _narrative_summary(d),
                    "",
                ]

                summary_lines.append("**Key Metrics:**")
                summary_lines.append(f"  Buildings: {d['building_count']:,}")
                summary_lines.append(f"  Study area: {d['study_area_m2']/1e6:.2f} km\u00b2")
                summary_lines.append(f"  Density: {d['building_density']:,.0f} buildings/km\u00b2")
                if d.get("height_mean") is not None:
                    summary_lines.append(
                        f"  Mean height: {d['height_mean']:.1f} m "
                        f"(max {d['height_max']:.1f} m)"
                    )
                if d.get("svf_mean") is not None:
                    summary_lines.append(
                        f"  Mean SVF: {d['svf_mean']:.3f} "
                        f"(range {d['svf_min']:.3f}\u2013{d['svf_max']:.3f})"
                    )
                if d.get("zone_bcr_mean") is not None:
                    summary_lines.append(f"  Mean BCR: {d['zone_bcr_mean']:.3f}")
                if d.get("zone_lambda_f_mean") is not None:
                    summary_lines.append(f"  Mean \u03bb_f: {d['zone_lambda_f_mean']:.3f}")
                summary_lines.append("")

                _pdf_text_page(pdf, "Executive Summary", summary_lines,
                               area_label=area_footer)

        # ── Methodology text + diagram ───────────────────────────────────
        if _want("methodology"):
            method_lines = [
                "**Phase 1: Data Acquisition & Preprocessing**",
                "Building footprints, DTM rasters, and road networks are loaded, "
                "reprojected to UTM 23S (EPSG:32723), and filtered for informal "
                "settlements (height, area, volume, h/w ratio thresholds).",
                "",
                "**Phase 2: Morphology Metrics**",
                "25 building-level morphometric indicators computed via momepy/"
                "libpysal: shape index, compactness, elongation, tessellation, "
                "adjacency, shared walls, etc.",
                "",
                "**Phase 3: Sky View Factor (SVF)**",
                "Street-level SVF computed along road centerlines using 3D "
                "ray-casting against building volumes and DTM terrain. Results "
                "are segment-averaged.",
                "",
                "**Phase 4: Urban Morphology (Zone-Level)**",
                "Grid-based zoning (tessellation) with BCR, FAR, \u03c3_h, \u03bb_f, "
                "and street orientation entropy per zone. Spatial autocorrelation "
                "via Moran\u2019s I and LISA clusters.",
                "",
                "**Phase 5: Patch Selection for CFD**",
                "Grid tiling of study area; 31 morphometric features per tile; "
                "PCA + k-means clustering; representative patch selection for "
                "OpenFOAM simulation.",
            ]
            _pdf_text_page(pdf, "Methodology & Workflow", method_lines,
                           area_label=area_footer)
            _pdf_methodology_diagram(pdf, area_label=area_footer)

        # ── Theory / metric definitions ──────────────────────────────────
        if _want("theory"):
            _pdf_theory_page(pdf, area_label=area_footer)

        # ── Parameters table ─────────────────────────────────────────────
        if _want("parameters"):
            params_df = pd.DataFrame([
                {"Parameter": "CRS", "Value": EXPECTED_CRS},
                {"Parameter": "Min Building Area", "Value": f"{MIN_BUILDING_AREA} m\u00b2"},
                {"Parameter": "Max Filter Height (informal)", "Value": f"{MAX_FILTER_HEIGHT} m"},
                {"Parameter": "Max Filter Area (informal)", "Value": f"{MAX_FILTER_AREA:,} m\u00b2"},
                {"Parameter": "Max Filter Volume (informal)", "Value": f"{MAX_FILTER_VOLUME:,} m\u00b3"},
                {"Parameter": "Cluster Buffer", "Value": f"{BUILDING_CLUSTER_BUFFER} m"},
                {"Parameter": "Output DPI", "Value": str(DPI)},
            ])
            _pdf_table_page(pdf, "Parameters & Configuration", params_df,
                            col_widths=[0.55, 0.45], area_label=area_footer)

        # ── Per-area results ─────────────────────────────────────────────
        if _want("results"):
            for d in areas_data:
                area = d["area"]
                a_label = _area_name(area)

                # Morphology summary stats table
                if d.get("morphology_stats") is not None:
                    mstats = d["morphology_stats"]
                    available = [c for c in MORPH_KEY_COLS if c in mstats.columns]
                    if available:
                        display = mstats[available].T
                        display.index.name = "Metric"
                        display = display.reset_index()
                        display["Metric"] = display["Metric"].map(
                            lambda x: _label(x)
                        )
                        stat_rows = ["count", "mean", "std", "min", "50%", "max"]
                        keep_cols = ["Metric"] + [r for r in stat_rows if r in display.columns]
                        display = display[keep_cols]
                        col_rename = {"count": "Count", "mean": "Mean", "std": "Std",
                                      "min": "Min", "50%": "Median", "max": "Max"}
                        display = display.rename(columns=col_rename)
                        _pdf_table_page(
                            pdf,
                            f"Morphology Metrics \u2014 {a_label}",
                            display,
                            area_label=a_label,
                        )

                # Zone metrics interpretive text page
                if d.get("zone_metrics") is not None:
                    zone_lines = [
                        f"**Zone-Level Metrics for {a_label}**",
                        "",
                        f"Zones analysed: {len(d['zone_metrics']):,}",
                        "",
                    ]
                    zm = d["zone_metrics"]
                    for col in ["bcr", "far", "sigma_h", "lambda_f"]:
                        if col in zm.columns:
                            vals = zm[col].dropna()
                            zone_lines.append(
                                f"{_label(col)}: mean = {vals.mean():.3f}, "
                                f"\u03c3 = {vals.std():.3f}, "
                                f"median = {vals.median():.3f}"
                            )
                    zone_lines.append("")

                    bcr = d.get("zone_bcr_mean")
                    if bcr is not None:
                        zone_lines.append(_interpret_bcr(bcr))
                        zone_lines.append("")
                    sigma_h = d.get("zone_sigma_h_mean")
                    if sigma_h is not None:
                        zone_lines.append(_interpret_sigma_h(sigma_h))
                        zone_lines.append("")

                    _pdf_text_page(pdf, f"Zone Morphology \u2014 {a_label}",
                                   zone_lines, area_label=a_label)

                # Moran's I table
                if d.get("moran_summary") is not None:
                    moran_df = d["moran_summary"].copy()
                    if "column" in moran_df.columns:
                        moran_df["column"] = moran_df["column"].map(_label)
                        moran_df = moran_df.rename(columns={"column": "Variable"})
                    _pdf_table_page(
                        pdf,
                        f"Spatial Autocorrelation \u2014 {a_label}",
                        moran_df,
                        area_label=a_label,
                    )

        # ── Embed key figures ────────────────────────────────────────
        if _want("figures"):
            for d in areas_data:
                area = d["area"]
                a_label = _area_name(area)
                figs = d.get("figures", {})

                priority = [
                    "svf_v2/svf_dashboard",
                    "svf_v2/svf_distribution",
                    "svf_v2/svf_streets_map",
                    "svf_v2/svf_streets_segments_map",
                    "morphology_metrics/morphology_distributions",
                    "urban_morphology/maps/zone_metrics_panel",
                    "urban_morphology/maps/lisa_clusters_bcr",
                ]
                for key in sorted(figs.keys()):
                    if "street_svf" in key and key not in priority:
                        priority.append(key)

                for fig_key in priority:
                    if fig_key in figs:
                        label = fig_key.split("/")[-1].replace("_", " ").title()
                        caption = FIGURE_CAPTIONS.get(fig_key)
                        _pdf_figure_page(
                            pdf,
                            f"{a_label} \u2014 {label}",
                            figs[fig_key],
                            caption=caption,
                            area_label=a_label,
                        )

        # ── Comparative table (if multiple areas) ────────────────────────
        if _want("comparative") and mode == "comparative" and len(areas_data) > 1:
            rows = []
            metrics_to_compare = [
                ("Buildings", "building_count"),
                ("Study Area (km\u00b2)", "study_area_m2"),
                ("Density (/km\u00b2)", "building_density"),
                ("Mean Height (m)", "height_mean"),
                ("Max Height (m)", "height_max"),
                ("Mean SVF", "svf_mean"),
                ("Mean BCR", "zone_bcr_mean"),
                ("Mean FAR", "zone_far_mean"),
                ("Mean \u03c3_h (m)", "zone_sigma_h_mean"),
                ("Mean \u03bb_f", "zone_lambda_f_mean"),
                ("Mean Shared Walls (m)", "mean_shared_walls"),
                ("Mean Adjacency Count", "mean_building_adjacency"),
            ]
            for name, key in metrics_to_compare:
                row = {"Metric": name}
                for d in areas_data:
                    val = d.get(key)
                    if key == "study_area_m2" and val is not None:
                        val = val / 1e6
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        row[_area_name(d["area"])] = val
                    else:
                        row[_area_name(d["area"])] = "N/A"
                rows.append(row)
            comp_df = pd.DataFrame(rows)
            _pdf_table_page(pdf, "Comparative Analysis", comp_df,
                            area_label=area_footer)

    logger.info("PDF saved: %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate IVF technical report (Markdown / PDF).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python scripts/generate_report.py --area rocinha --format both
          python scripts/generate_report.py --areas rocinha,vidigal --mode comparative --format pdf
          python scripts/generate_report.py --area rocinha --format pdf --sections title,methodology,results
        """),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--area", type=str, help="Single area name")
    group.add_argument("--areas", type=str,
                       help="Comma-separated area names for comparative mode")
    parser.add_argument("--mode", choices=["single", "comparative"],
                        default="single",
                        help="Report mode (default: single)")
    parser.add_argument("--format", choices=["markdown", "pdf", "both"],
                        default="both", dest="output_format",
                        help="Output format (default: both)")
    parser.add_argument(
        "--sections", type=str, default=None,
        help=(
            "Comma-separated list of PDF sections to generate "
            "(default: all). Valid names: "
            + ", ".join(ALL_SECTIONS)
        ),
    )

    args = parser.parse_args()

    # Resolve area list
    if args.area:
        area_names = [args.area]
    else:
        area_names = [a.strip() for a in args.areas.split(",")]
        if len(area_names) > 1 and args.mode == "single":
            args.mode = "comparative"
            logger.info("Multiple areas provided -- switching to comparative mode.")

    # Validate area names
    for a in area_names:
        if a not in SUPPORTED_AREAS:
            logger.error("Unknown area '%s'. Supported: %s", a, SUPPORTED_AREAS)
            sys.exit(1)

    # Parse --sections filter
    section_list: Optional[List[str]] = None
    if args.sections is not None:
        section_list = [s.strip() for s in args.sections.split(",")]
        invalid = [s for s in section_list if s not in ALL_SECTIONS]
        if invalid:
            logger.error(
                "Unknown section(s): %s. Valid: %s", invalid, ALL_SECTIONS
            )
            sys.exit(1)
        logger.info("Generating sections: %s", section_list)

    # Collect data
    areas_data = [collect_area_data(a) for a in area_names]

    # Determine output directory
    if args.mode == "comparative":
        report_dir = get_comparative_analysis_dir() / "reports"
    else:
        report_dir = get_area_output_dir(area_names[0]) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    if args.output_format in ("markdown", "both"):
        md_text = generate_markdown(areas_data, mode=args.mode)
        md_path = report_dir / "technical_report.md"
        md_path.write_text(md_text, encoding="utf-8")
        logger.info("Markdown report: %s", md_path)

    if args.output_format in ("pdf", "both"):
        pdf_path = report_dir / "technical_report.pdf"
        generate_pdf(areas_data, pdf_path, mode=args.mode, sections=section_list)

    logger.info("Done.")


if __name__ == "__main__":
    main()
