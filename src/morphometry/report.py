"""PDF report generator for morphometric audit.

Uses matplotlib PdfPages to produce a multi-page academic report in the
Swiss design language established by the existing IVF report generator.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from matplotlib.patches import Rectangle

import geopandas as gpd

from src.morphometry.audit import SVFAuditResult

logger = logging.getLogger(__name__)

# ── Swiss design palette ─────────────────────────────────────────────────
SWISS = {
    "text": "#1A1A1A",
    "secondary": "#5C5C5C",
    "accent": "#2A5FA5",
    "accent_light": "#E8EFF7",
    "background": "#FFFFFF",
    "light_gray": "#F2F2F2",
    "grid": "#D9D9D9",
    "caption": "#666666",
    "warning": "#dc2626",
    "ok": "#16a34a",
}

PAGE_SIZE = (8.5, 11)
MARGIN = {"left": 0.08, "right": 0.92, "top": 0.93, "bottom": 0.06}

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

# LCZ 3 (Compact Low-Rise) benchmarks — Stewart & Oke (2012)
LCZ3_BENCHMARKS = {
    "svf": {"range": (0.2, 0.6), "label": "SVF"},
    "lambda_p": {"range": (0.4, 0.7), "label": r"$\lambda_p$"},
    "H_mean": {"range": (3, 10), "label": "H (m)"},
}

_figure_counter = 0


def _reset():
    global _figure_counter
    _figure_counter = 0


def _next_fig() -> int:
    global _figure_counter
    _figure_counter += 1
    return _figure_counter


def _area_name(slug: str) -> str:
    return AREA_DISPLAY_NAMES.get(slug, slug.replace("_", " ").title())


def _fmt(v, dec=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{dec}f}"


def _base_page(pdf: PdfPages, area_label: str = "") -> plt.Figure:
    """Create page with header accent line and footer."""
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")

    ax_top = fig.add_axes(
        [MARGIN["left"], 0.955, MARGIN["right"] - MARGIN["left"], 0.002]
    )
    ax_top.add_patch(Rectangle((0, 0), 1, 1, facecolor=SWISS["accent"]))
    ax_top.axis("off")

    date_str = datetime.now().strftime("%B %Y")
    fig.text(
        MARGIN["left"],
        0.025,
        area_label,
        ha="left",
        va="center",
        fontsize=8,
        color=SWISS["secondary"],
        family="sans-serif",
    )
    fig.text(
        0.5,
        0.025,
        f"{pdf.get_pagecount() + 1}",
        ha="center",
        va="center",
        fontsize=8,
        color=SWISS["secondary"],
        family="sans-serif",
    )
    fig.text(
        MARGIN["right"],
        0.025,
        date_str,
        ha="right",
        va="center",
        fontsize=8,
        color=SWISS["secondary"],
        family="sans-serif",
    )
    return fig


def _save(pdf: PdfPages, fig: plt.Figure):
    pdf.savefig(fig, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)


def _title_page(pdf: PdfPages, area: str):
    """Swiss-academic title page."""
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")

    fig.text(
        0.5,
        0.68,
        "Morphometric Analysis\nAudit & Report",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=SWISS["text"],
        family="sans-serif",
        linespacing=1.5,
    )

    fig.text(
        0.5,
        0.58,
        _area_name(area),
        ha="center",
        va="center",
        fontsize=16,
        color=SWISS["accent"],
        family="sans-serif",
    )

    ax_bar = fig.add_axes([0.30, 0.53, 0.40, 0.003])
    ax_bar.add_patch(Rectangle((0, 0), 1, 1, facecolor=SWISS["accent"]))
    ax_bar.axis("off")

    fig.text(
        0.5,
        0.45,
        "Informal Ventilation Flows (IVF)",
        ha="center",
        va="center",
        fontsize=12,
        color=SWISS["text"],
        family="sans-serif",
        style="italic",
    )

    fig.text(
        0.5,
        0.41,
        "Massachusetts Institute of Technology",
        ha="center",
        va="center",
        fontsize=11,
        color=SWISS["text"],
        family="sans-serif",
    )

    fig.text(
        0.5,
        0.36,
        "Brisa+ Research Program — Nature Cities",
        ha="center",
        va="center",
        fontsize=10,
        color=SWISS["secondary"],
        family="sans-serif",
    )

    fig.text(
        0.5,
        0.30,
        datetime.now().strftime("%B %Y"),
        ha="center",
        va="center",
        fontsize=11,
        color=SWISS["secondary"],
        family="sans-serif",
    )

    _save(pdf, fig)


def _strip_md_bold(text: str) -> str:
    """Strip markdown bold markers (**) from text."""
    import re

    return re.sub(r"\*\*(.+?)\*\*", r"\1", text)


def _has_bold_lead(text: str) -> bool:
    """Check if line starts with a bold-marked segment like **Label:** ..."""
    return text.startswith("**") and "**" in text[2:]


def _text_page(pdf: PdfPages, title: str, lines: List[str], area_label: str = ""):
    """Page of wrapped text content with inline bold support."""
    import textwrap

    fig = _base_page(pdf, area_label=area_label)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.5,
        MARGIN["top"],
        title,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=SWISS["text"],
        family="sans-serif",
        transform=ax.transAxes,
    )

    # Separator line
    sep = plt.Line2D(
        [MARGIN["left"], MARGIN["right"]],
        [MARGIN["top"] - 0.025, MARGIN["top"] - 0.025],
        transform=ax.transAxes,
        color=SWISS["accent"],
        linewidth=1.0,
    )
    ax.add_line(sep)

    y = MARGIN["top"] - 0.05
    for text in lines:
        if y < MARGIN["bottom"] + 0.04:
            break

        if text == "":
            y -= 0.015
            continue

        is_heading = text.startswith("### ")
        is_icon_line = text and text[0] in ("\u2716", "\u26a0", "\u2139", "\u2714")

        if is_heading:
            y -= 0.01
            ax.text(
                MARGIN["left"],
                y,
                text[4:],
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                color=SWISS["accent"],
                family="sans-serif",
                transform=ax.transAxes,
            )
            y -= 0.03
        elif is_icon_line:
            # Audit issue lines — render with icon emphasis
            clean = _strip_md_bold(text)
            wrapped = textwrap.wrap(clean, width=90)
            for wline in wrapped:
                if y < MARGIN["bottom"] + 0.04:
                    break
                ax.text(
                    MARGIN["left"],
                    y,
                    wline,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    color=SWISS["text"],
                    family="sans-serif",
                    transform=ax.transAxes,
                )
                y -= 0.02
            y -= 0.005
        elif _has_bold_lead(text):
            # Line with bold lead: "**Label:** rest of text"
            # Render bold portion + regular portion on same line
            end_bold = text.index("**", 2) + 2
            bold_part = text[2 : end_bold - 2]
            rest = text[end_bold:].lstrip()
            combined = bold_part + " " + rest if rest else bold_part
            wrapped = textwrap.wrap(combined, width=90)
            for i, wline in enumerate(wrapped):
                if y < MARGIN["bottom"] + 0.04:
                    break
                weight = "bold" if i == 0 else "normal"
                ax.text(
                    MARGIN["left"],
                    y,
                    wline,
                    ha="left",
                    va="top",
                    fontsize=9,
                    fontweight=weight,
                    color=SWISS["text"],
                    family="sans-serif",
                    transform=ax.transAxes,
                )
                y -= 0.02
            y -= 0.005
        else:
            # Regular text — strip any remaining ** markers
            clean = _strip_md_bold(text)
            wrapped = textwrap.wrap(clean, width=90)
            for wline in wrapped:
                if y < MARGIN["bottom"] + 0.04:
                    break
                ax.text(
                    MARGIN["left"],
                    y,
                    wline,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color=SWISS["text"],
                    family="sans-serif",
                    transform=ax.transAxes,
                )
                y -= 0.02

    _save(pdf, fig)


def _table_page(pdf: PdfPages, title: str, df: pd.DataFrame, area_label: str = ""):
    """Render a table as a PDF page."""
    fig = _base_page(pdf, area_label=area_label)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.5,
        MARGIN["top"],
        title,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=SWISS["text"],
        family="sans-serif",
        transform=ax.transAxes,
    )

    line = plt.Line2D(
        [MARGIN["left"], MARGIN["right"]],
        [MARGIN["top"] - 0.025, MARGIN["top"] - 0.025],
        transform=ax.transAxes,
        color=SWISS["accent"],
        linewidth=1.0,
    )
    ax.add_line(line)

    display = df.head(28).copy()

    # Format numbers
    for col in display.columns:
        if display[col].dtype in [np.float64, np.float32, float]:
            display[col] = display[col].apply(
                lambda x: (
                    f"{x:,.2f}"
                    if pd.notna(x) and abs(x) >= 10
                    else f"{x:.3f}"
                    if pd.notna(x)
                    else "—"
                )
            )

    cell_text = display.values.tolist()
    col_labels = list(display.columns)
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

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(SWISS["accent"])
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    for i in range(1, len(cell_text) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor(SWISS["light_gray"] if i % 2 == 0 else "white")

    _save(pdf, fig)


def _figure_page(
    pdf: PdfPages,
    title: str,
    img_path: Path,
    caption: Optional[str] = None,
    area_label: str = "",
):
    """Embed a figure image as a full PDF page."""
    fig = _base_page(pdf, area_label=area_label)

    fig_num = _next_fig()
    numbered_title = f"Fig. {fig_num} \u2014 {title}"

    fig.text(
        0.5,
        0.96,
        numbered_title,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=SWISS["text"],
        family="sans-serif",
    )

    try:
        img = imread(str(img_path))
        img_h, img_w = img.shape[:2]
        aspect = img_w / img_h

        img_bottom = 0.07
        img_top = 0.91
        img_left = MARGIN["left"]
        img_right = MARGIN["right"]
        img_width = img_right - img_left
        img_height = img_top - img_bottom

        if aspect >= 1.2:
            ax = fig.add_axes([img_left, img_bottom, img_width, img_height])
        elif aspect >= 0.8:
            inset = 0.02
            ax = fig.add_axes(
                [img_left + inset, img_bottom, img_width - 2 * inset, img_height]
            )
        else:
            w = img_width * 0.75
            ax = fig.add_axes([(1 - w) / 2, img_bottom, w, img_height])

        ax.imshow(img)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color(SWISS["grid"])

    except Exception as e:
        logger.warning("Could not embed figure %s: %s", img_path, e)
        fig.text(
            0.5,
            0.5,
            f"[Figure not available: {e}]",
            ha="center",
            va="center",
            fontsize=10,
            color=SWISS["secondary"],
        )

    if caption:
        fig.text(
            0.5,
            0.04,
            caption,
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color=SWISS["caption"],
            family="sans-serif",
            wrap=True,
        )

    _save(pdf, fig)


# ═══════════════════════════════════════════════════════════════════════════
#  NARRATIVE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════


def _svf_qualifier(svf: float) -> str:
    if svf < 0.10:
        return "extremely low"
    if svf < 0.15:
        return "very low"
    if svf < 0.30:
        return "low"
    if svf < 0.50:
        return "moderate"
    return "relatively open"


def _density_qualifier(lp: float) -> str:
    if lp > 0.6:
        return "very high"
    if lp > 0.4:
        return "high"
    if lp > 0.2:
        return "moderate"
    return "low"


def _build_morphometric_profile(grid_data: Dict[str, Any]) -> List[str]:
    """Build narrative lines for section 4.2."""
    lines = []
    g = grid_data

    svf_mean = g.get("svf_mean")
    lp_mean = g.get("lambda_p_mean")
    sigma_h = g.get("sigma_h_mean")
    h_mean = g.get("H_mean_mean")
    slope_mean = g.get("slope_deg_mean")
    porosity_mean = g.get("porosity_mean")
    n_cells = g.get("n_cells", 0)
    n_buildings = g.get("n_buildings", 0)

    lines.append(
        f"The study area comprises {n_buildings:,} buildings analysed "
        f"across {n_cells:,} grid cells (10 m resolution)."
    )
    lines.append("")

    if svf_mean is not None:
        q = _svf_qualifier(svf_mean)
        lines.append(
            f"**Sky View Factor.** The mean SVF is {q} at {svf_mean:.3f}, "
            f"indicating {'severe sky obstruction characteristic of dense informal fabric' if svf_mean < 0.15 else 'restricted sky access typical of compact settlements' if svf_mean < 0.3 else 'moderate openness'}. "
            f"LCZ 3 (Compact Low-Rise) benchmarks range 0.20\u20130.60; "
            f"{'Vidigal falls well below this range, suggesting conditions more extreme than standard compact classifications.' if svf_mean < 0.2 else 'values are within the lower end of the expected range.' if svf_mean < 0.4 else 'values are within the expected range.'}"
        )
        lines.append("")

    if lp_mean is not None:
        q = _density_qualifier(lp_mean)
        lines.append(
            f"**Plan Area Density (\u03bbp).** Mean \u03bbp = {lp_mean:.3f} ({q} density). "
            f"LCZ 3 benchmark: 0.40\u20130.70. "
            f"{'Density exceeds compact low-rise expectations.' if lp_mean > 0.7 else 'Within expected range.' if lp_mean > 0.4 else 'Lower than typical compact low-rise.'}"
        )
        lines.append("")

    if h_mean is not None:
        lines.append(
            f"**Building Heights.** Mean height = {h_mean:.1f} m"
            + (f", \u03c3H = {sigma_h:.2f} m" if sigma_h is not None else "")
            + ". LCZ 3 benchmark: 3\u201310 m. "
            + (
                "Height variability is high, creating significant vertical roughness "
                "that promotes turbulent mixing but may channel wind in narrow passages."
                if sigma_h is not None and sigma_h > 3
                else "Moderate height variability indicates a heterogeneous building stock "
                "typical of incrementally built settlements."
                if sigma_h is not None and sigma_h > 1.5
                else "Relatively uniform building heights."
            )
        )
        lines.append("")

    if slope_mean is not None:
        lines.append(
            f"**Terrain.** Mean slope = {slope_mean:.1f}\u00b0. "
            + (
                "Steep terrain (> 15\u00b0) significantly affects wind patterns "
                "and makes standard flat-terrain ventilation models unreliable."
                if slope_mean > 15
                else "Moderate slopes influence wind channeling along contour lines."
                if slope_mean > 5
                else "Relatively flat terrain."
            )
        )
        lines.append("")

    if porosity_mean is not None:
        lines.append(
            f"**Porosity.** Mean volumetric porosity = {porosity_mean:.3f}. "
            f"{'Low porosity indicates limited void space within the urban canopy for air circulation.' if porosity_mean < 0.4 else 'Moderate porosity allows some ventilation within the canopy layer.' if porosity_mean < 0.7 else 'High porosity with significant void space.'}"
        )
        lines.append("")

    return lines


def _build_spatial_patterns(grid_data: Dict[str, Any]) -> List[str]:
    """Build narrative lines for section 4.3."""
    lines = []

    svf_slope_r = grid_data.get("svf_slope_r")
    svf_lp_r = grid_data.get("svf_lp_r")

    lines.append("### Density\u2013Enclosure Patterns")
    if svf_lp_r is not None:
        lines.append(
            f"SVF and \u03bbp show a Pearson correlation of r = {svf_lp_r:.3f}. "
            + (
                "Strong negative correlation confirms that higher plan density "
                "directly reduces sky access."
                if svf_lp_r < -0.5
                else "Moderate negative correlation between density and sky access."
                if svf_lp_r < -0.3
                else "Weak correlation suggests that sky obstruction is driven more by "
                "building height and proximity than plan density alone."
            )
        )
    lines.append("")

    lines.append("### Slope\u2013Density Relationship")
    if svf_slope_r is not None:
        lines.append(
            f"SVF and terrain slope correlate at r = {svf_slope_r:.3f}. "
            + (
                "Positive correlation indicates that steeper areas tend to have "
                "higher SVF, likely because buildings are sparser on steep terrain."
                if svf_slope_r > 0.2
                else "Negative correlation suggests that steeper slopes have denser "
                "building patterns or that hillside terrain itself obstructs sky."
                if svf_slope_r < -0.2
                else "No strong slope\u2013SVF relationship, suggesting building density "
                "is independent of terrain steepness."
            )
        )
    lines.append("")

    return lines


def _build_cfd_implications(grid_data: Dict[str, Any]) -> List[str]:
    """Build narrative lines for section 4.4."""
    lines = []
    strata = grid_data.get("strata_summary")

    lines.append(
        "For the CFD simulation campaign, the morphometric feature space "
        "defines which strata of SVF \u00d7 slope \u00d7 \u03bbp are represented "
        "in the study area."
    )
    lines.append("")

    if strata is not None:
        lines.append(
            "**Occupied strata (SVF bins \u00d7 slope bins \u00d7 density bins):**"
        )
        for s in strata:
            lines.append(f"  \u2022 {s}")
        lines.append("")

    lines.append(
        "Strata absent from the study area would need to be sampled from "
        "other sites (Rocinha, Rio das Pedras, etc.) to ensure full coverage "
        "of the Brazilian favela parameter space."
    )
    lines.append("")

    return lines


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════


def generate_pdf_report(
    output_path: Path,
    area: str,
    grid: gpd.GeoDataFrame,
    audit_result: SVFAuditResult,
    figure_paths: Dict[str, Path],
    n_buildings: int = 0,
):
    """Generate the complete morphometric audit PDF report.

    Parameters
    ----------
    output_path : Path
        Where to save the PDF.
    area : str
        Area slug (e.g. "vidigal").
    grid : GeoDataFrame
        Grid cells with all morphometric columns.
    audit_result : SVFAuditResult
        Results from SVF audit.
    figure_paths : dict
        Mapping of figure keys to file paths.
    n_buildings : int
        Total number of buildings in the study area.
    """
    _reset()
    area_label = _area_name(area)

    # Pre-compute stats for narrative
    grid_data: Dict[str, Any] = {
        "n_cells": len(grid),
        "n_buildings": n_buildings,
    }

    for col in [
        "svf",
        "lambda_p",
        "lambda_f_mean",
        "porosity",
        "sigma_h",
        "H_mean",
        "slope_deg",
        "aspect_deg",
        "street_orientation_entropy",
    ]:
        if col in grid.columns:
            vals = grid[col].dropna()
            if len(vals) > 0:
                grid_data[f"{col}_mean"] = float(vals.mean())
                grid_data[f"{col}_std"] = float(vals.std())
                grid_data[f"{col}_min"] = float(vals.min())
                grid_data[f"{col}_max"] = float(vals.max())
                grid_data[f"{col}_median"] = float(vals.median())

    # Correlations
    if "svf" in grid.columns and "slope_deg" in grid.columns:
        valid = grid[["svf", "slope_deg"]].dropna()
        if len(valid) > 10:
            grid_data["svf_slope_r"] = float(valid.corr().iloc[0, 1])

    if "svf" in grid.columns and "lambda_p" in grid.columns:
        valid = grid[["svf", "lambda_p"]].dropna()
        if len(valid) > 10:
            grid_data["svf_lp_r"] = float(valid.corr().iloc[0, 1])

    # Strata summary for CFD
    if all(c in grid.columns for c in ["svf", "slope_deg", "lambda_p"]):
        valid = grid[["svf", "slope_deg", "lambda_p"]].dropna()
        if len(valid) > 10:
            svf_bins = pd.cut(
                valid["svf"],
                bins=[0, 0.1, 0.2, 0.4, 1.0],
                labels=["<0.1", "0.1\u20130.2", "0.2\u20130.4", ">0.4"],
            )
            slope_bins = pd.cut(
                valid["slope_deg"],
                bins=[0, 5, 15, 30, 90],
                labels=["<5\u00b0", "5\u201315\u00b0", "15\u201330\u00b0", ">30\u00b0"],
            )
            lp_bins = pd.cut(
                valid["lambda_p"],
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=["<0.3", "0.3\u20130.5", "0.5\u20130.7", ">0.7"],
            )
            strata = valid.groupby(
                [svf_bins, slope_bins, lp_bins], observed=True
            ).size()
            strata = strata[strata > 0]
            grid_data["strata_summary"] = [
                f"SVF {s[0]}, Slope {s[1]}, \u03bbp {s[2]}: {n} cells"
                for s, n in strata.items()
            ]

    logger.info("Generating PDF report: %s", output_path)

    with PdfPages(output_path) as pdf:
        # ── Page 1: Title ──────────────────────────────────────────────
        _title_page(pdf, area)

        # ── Page 2: Data Quality Summary (4.1) ─────────────────────────
        dq_lines = [
            f"**Study area:** {area_label}",
            "**Grid resolution:** 10 m \u00d7 10 m",
            f"**Grid cells:** {grid_data['n_cells']:,}",
            f"**Buildings:** {n_buildings:,}",
            "**CRS:** EPSG:31983 (SIRGAS 2000 / UTM 23S)",
            "",
            "### SVF Audit Results",
            f"**SVF computation method:** {audit_result.method}",
            f"**Number of rays:** {audit_result.n_rays or 'N/A'}",
            f"**Sample points:** {audit_result.count:,}",
            f"**Estimated grid spacing:** {_fmt(audit_result.grid_spacing_m, 1)} m",
            f"**Terrain integrated:** {'Yes' if audit_result.terrain_integrated else 'Not confirmed'}",
            f"**Coverage:** {_fmt(audit_result.coverage_pct, 1)}%",
            "",
        ]

        # Audit issues
        for issue in audit_result.issues:
            icon = (
                "\u2716"
                if issue.severity == "critical"
                else "\u26a0"
                if issue.severity == "warning"
                else "\u2139"
            )
            dq_lines.append(f"{icon} [{issue.severity.upper()}] {issue.message}")
        if not audit_result.issues:
            dq_lines.append("\u2714 No issues flagged.")

        _text_page(pdf, "1. Data Quality Summary", dq_lines, area_label)

        # ── Page 3: SVF Statistics Table ───────────────────────────────
        svf_stats = pd.DataFrame(
            {
                "Statistic": [
                    "Count",
                    "Mean",
                    "Median",
                    "Std Dev",
                    "Min",
                    "Max",
                    "5th %ile",
                    "25th %ile",
                    "75th %ile",
                    "95th %ile",
                ],
                "Value": [
                    f"{audit_result.count:,}",
                    _fmt(audit_result.mean),
                    _fmt(audit_result.median),
                    _fmt(audit_result.std),
                    _fmt(audit_result.min),
                    _fmt(audit_result.max),
                    _fmt(audit_result.p5),
                    _fmt(audit_result.p25),
                    _fmt(audit_result.p75),
                    _fmt(audit_result.p95),
                ],
            }
        )
        _table_page(pdf, "SVF Summary Statistics", svf_stats, area_label)

        # ── Page 4: Morphometric Profile (4.2) ────────────────────────
        profile_lines = _build_morphometric_profile(grid_data)
        _text_page(pdf, "2. Morphometric Profile", profile_lines, area_label)

        # ── Page 5: Summary Statistics Table ───────────────────────────
        indicator_cols = [
            ("svf", "SVF"),
            ("lambda_p", "\u03bbp"),
            ("lambda_f_mean", "\u03bbf (mean)"),
            ("porosity", "Porosity"),
            ("sigma_h", "\u03c3H (m)"),
            ("H_mean", "H mean (m)"),
            ("street_orientation_entropy", "Entropy"),
            ("slope_deg", "Slope (\u00b0)"),
        ]

        stats_rows = []
        for col, label in indicator_cols:
            if col in grid.columns:
                vals = grid[col].dropna()
                if len(vals) > 0:
                    stats_rows.append(
                        {
                            "Indicator": label,
                            "Count": f"{len(vals):,}",
                            "Mean": f"{vals.mean():.3f}",
                            "Std": f"{vals.std():.3f}",
                            "Min": f"{vals.min():.3f}",
                            "Median": f"{vals.median():.3f}",
                            "Max": f"{vals.max():.3f}",
                        }
                    )
        if stats_rows:
            stats_df = pd.DataFrame(stats_rows)
            _table_page(pdf, "Morphometric Summary Statistics", stats_df, area_label)

        # ── Page 6: Spatial Patterns (4.3) ────────────────────────────
        patterns_lines = _build_spatial_patterns(grid_data)
        _text_page(pdf, "3. Spatial Patterns", patterns_lines, area_label)

        # ── Page 7: CFD Implications (4.4) ────────────────────────────
        cfd_lines = _build_cfd_implications(grid_data)
        _text_page(pdf, "4. CFD Campaign Implications", cfd_lines, area_label)

        # ── Figures ───────────────────────────────────────────────────
        figure_specs = [
            (
                "site_overview",
                "Site Overview",
                "Building footprints on hillshade DTM with elevation contours.",
            ),
            (
                "svf_map",
                "Sky View Factor Map",
                "SVF on 10 m grid with diverging colormap (breakpoint at SVF = 0.15). "
                "Red indicates severe sky obstruction; blue indicates open sky.",
            ),
            (
                "morphometric_panel",
                "Morphometric Indicator Panel",
                "Six indicators at 10 m resolution: plan area density, frontal area density, "
                "porosity, height variability, street orientation entropy, and terrain slope.",
            ),
            (
                "svf_slope_scatter",
                "SVF vs. Terrain Slope",
                "Each point represents a 10 m grid cell. Color encodes plan area density. "
                "Marginal histograms show univariate distributions.",
            ),
            (
                "correlation_matrix",
                "Indicator Correlation Matrix",
                "Pairwise Pearson correlations. Bold values indicate |r| > 0.7 "
                "(high collinearity flagged for predictor selection).",
            ),
            (
                "distributions",
                "Morphometric Distributions",
                "Violin plots showing the distribution of each indicator. "
                "Green bands indicate LCZ 3 (Compact Low-Rise) reference ranges "
                "from Stewart & Oke (2012).",
            ),
        ]

        for key, title, caption in figure_specs:
            path = figure_paths.get(key)
            if path is not None and path.exists():
                _figure_page(pdf, title, path, caption=caption, area_label=area_label)

    logger.info("PDF report saved: %s (%d pages)", output_path, _figure_counter + 7)
