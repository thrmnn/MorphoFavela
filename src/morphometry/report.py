"""PDF report generator for morphometric audit.

Uses matplotlib PdfPages to produce a multi-page academic report in the
Swiss design language. Figures fill available page space; tables are compact
with LaTeX-style thin rules; narrative is analytical, not decorative.
"""

from __future__ import annotations

import logging
import re
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
from matplotlib.patches import Rectangle

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
    "complexo_do_alemao": "Complexo do Alemao",
    "mare": "Mare",
    "cidade_de_deus": "Cidade de Deus",
    "copacabana": "Copacabana",
}

# LCZ 3 (Compact Low-Rise) benchmarks -- Stewart & Oke (2012)
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


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════


def _base_page(pdf: PdfPages, area_label: str = "") -> plt.Figure:
    """Create page with header accent line and footer."""
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")

    # Top accent line
    ax_top = fig.add_axes(
        [MARGIN["left"], 0.955, MARGIN["right"] - MARGIN["left"], 0.002]
    )
    ax_top.add_patch(Rectangle((0, 0), 1, 1, facecolor=SWISS["accent"]))
    ax_top.axis("off")

    # Footer
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


# ── Title Page ───────────────────────────────────────────────────────────


def _title_page(pdf: PdfPages, area: str):
    """Swiss-academic title page with refined spacing."""
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")

    fig.text(
        0.5,
        0.68,
        "Morphometric Analysis",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=SWISS["text"],
        family="sans-serif",
    )
    fig.text(
        0.5,
        0.62,
        "Technical Report",
        ha="center",
        va="center",
        fontsize=14,
        color=SWISS["secondary"],
        family="sans-serif",
    )

    fig.text(
        0.5,
        0.55,
        _area_name(area),
        ha="center",
        va="center",
        fontsize=18,
        color=SWISS["accent"],
        family="sans-serif",
    )

    ax_bar = fig.add_axes([0.30, 0.51, 0.40, 0.003])
    ax_bar.add_patch(Rectangle((0, 0), 1, 1, facecolor=SWISS["accent"]))
    ax_bar.axis("off")

    fig.text(
        0.5,
        0.44,
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
        0.40,
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
        "Brisa+ Research Program \u2014 Nature Cities",
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


# ── Figure Page (aspect-aware) ───────────────────────────────────────────


def _figure_page(
    pdf: PdfPages,
    title: str,
    img_path: Path,
    caption: Optional[str] = None,
    area_label: str = "",
):
    """Embed a figure image filling maximum available page area."""
    fig = _base_page(pdf, area_label=area_label)

    fig_num = _next_fig()
    numbered_title = f"Fig. {fig_num}  |  {title}"

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
        img_aspect = img_w / img_h

        # Available area for the image
        avail_left = MARGIN["left"]
        avail_right = MARGIN["right"]
        avail_bottom = 0.07 if caption else 0.06
        avail_top = 0.92
        avail_w = avail_right - avail_left
        avail_h = avail_top - avail_bottom
        page_aspect = avail_w / avail_h

        # Fit image to available area (CSS object-fit: contain, biased large)
        if img_aspect > page_aspect:
            # Image is wider -- constrain by width
            ax_w = avail_w
            ax_h = avail_w / img_aspect
        else:
            # Image is taller -- constrain by height
            ax_h = avail_h
            ax_w = avail_h * img_aspect

        # Push figure to TOP of available area (not centered — avoids dead space above)
        ax_x = avail_left + (avail_w - ax_w) / 2
        ax_y = avail_top - ax_h

        ax = fig.add_axes([ax_x, ax_y, ax_w, ax_h])
        ax.imshow(img)
        ax.axis("off")

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
            0.045,
            caption,
            ha="center",
            va="top",
            fontsize=8.5,
            style="italic",
            color=SWISS["caption"],
            family="sans-serif",
            wrap=True,
        )

    _save(pdf, fig)


# ── Text Page (rich inline formatting) ───────────────────────────────────


def _render_text_block(
    ax,
    lines: List[str],
    y_start: float,
    y_stop: float = 0.06,
    fontsize: float = 9.5,
    line_height: float = 0.020,
    wrap_width: int = 95,
) -> float:
    """Render lines of text onto an axes, returning final y position.

    Supports:
    - ``### Heading`` for sub-section headings
    - ``**bold lead:** rest`` for bold-lead lines
    - Empty string for paragraph spacing
    - Unicode icons at line start for audit issue lines
    """
    y = y_start

    for text in lines:
        if y < y_stop:
            break

        # Blank line = paragraph spacer
        if text == "":
            y -= line_height * 0.6
            continue

        is_heading = text.startswith("### ")
        is_icon_line = bool(text) and text[0] in (
            "\u2716",
            "\u26a0",
            "\u2139",
            "\u2714",
        )

        if is_heading:
            y -= line_height * 0.4
            ax.text(
                MARGIN["left"],
                y,
                text[4:],
                ha="left",
                va="top",
                fontsize=fontsize + 1.5,
                fontweight="bold",
                color=SWISS["accent"],
                family="sans-serif",
                transform=ax.transAxes,
            )
            y -= line_height * 1.6
            continue

        # Strip markdown bold for display, detect bold-lead pattern
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        has_bold_lead = text.startswith("**") and "**" in text[2:]

        if has_bold_lead:
            end_bold = text.index("**", 2) + 2
            bold_part = text[2 : end_bold - 2]
            rest = text[end_bold:].lstrip()
            combined = bold_part + " " + rest if rest else bold_part
            wrapped = textwrap.wrap(combined, width=wrap_width)
            for i, wline in enumerate(wrapped):
                if y < y_stop:
                    break
                ax.text(
                    MARGIN["left"],
                    y,
                    wline,
                    ha="left",
                    va="top",
                    fontsize=fontsize,
                    fontweight="bold" if i == 0 else "normal",
                    color=SWISS["text"],
                    family="sans-serif",
                    transform=ax.transAxes,
                )
                y -= line_height
            y -= line_height * 0.2
        elif is_icon_line:
            wrapped = textwrap.wrap(clean, width=wrap_width)
            for wline in wrapped:
                if y < y_stop:
                    break
                ax.text(
                    MARGIN["left"],
                    y,
                    wline,
                    ha="left",
                    va="top",
                    fontsize=fontsize - 0.5,
                    color=SWISS["text"],
                    family="sans-serif",
                    transform=ax.transAxes,
                )
                y -= line_height
            y -= line_height * 0.15
        else:
            wrapped = textwrap.wrap(clean, width=wrap_width)
            for wline in wrapped:
                if y < y_stop:
                    break
                ax.text(
                    MARGIN["left"],
                    y,
                    wline,
                    ha="left",
                    va="top",
                    fontsize=fontsize,
                    color=SWISS["text"],
                    family="sans-serif",
                    transform=ax.transAxes,
                )
                y -= line_height
            y -= line_height * 0.2

    return y


def _section_header(ax, title: str, y: float) -> float:
    """Draw a section title with an accent underline. Returns y below the line."""
    ax.text(
        0.5,
        y,
        title,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=SWISS["text"],
        family="sans-serif",
        transform=ax.transAxes,
    )
    line_y = y - 0.025
    sep = plt.Line2D(
        [MARGIN["left"], MARGIN["right"]],
        [line_y, line_y],
        transform=ax.transAxes,
        color=SWISS["accent"],
        linewidth=1.0,
    )
    ax.add_line(sep)
    return line_y - 0.020


def _text_page(
    pdf: PdfPages,
    title: str,
    lines: List[str],
    area_label: str = "",
):
    """Single page of wrapped text content."""
    fig = _base_page(pdf, area_label=area_label)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = _section_header(ax, title, MARGIN["top"])
    _render_text_block(ax, lines, y)
    _save(pdf, fig)


# ── Compact Table Renderer ───────────────────────────────────────────────


def _compact_table(
    ax,
    df: pd.DataFrame,
    y_top: float,
    col_widths: Optional[List[float]] = None,
    fontsize: float = 8.5,
    row_height: float = 0.024,
) -> float:
    """Render a compact table positioned at y_top. Returns y below the table.

    Uses thin rules and tight row spacing instead of stretching to fill
    the page. Returns the y-coordinate below the last row.
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    if n_cols == 0 or n_rows == 0:
        return y_top

    # Format numeric columns
    display = df.copy()
    for col in display.columns:
        if display[col].dtype in [np.float64, np.float32, float]:
            display[col] = display[col].apply(
                lambda x: (
                    f"{x:,.2f}"
                    if pd.notna(x) and abs(x) >= 10
                    else f"{x:.3f}"
                    if pd.notna(x)
                    else "\u2014"
                )
            )

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    # Table height = header + data rows
    table_h = row_height * (n_rows + 1)
    table_bottom = y_top - table_h
    table_w = MARGIN["right"] - MARGIN["left"]
    table_left = MARGIN["left"]

    cell_text = display.values.tolist()
    col_labels = list(display.columns)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="upper center",
        bbox=[table_left, table_bottom, table_w, table_h],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(SWISS["accent"])
        cell.set_text_props(color="white", fontweight="bold", fontsize=fontsize + 0.5)
        cell.set_edgecolor(SWISS["accent"])
        cell.set_linewidth(0.5)

    # Style data rows -- alternate bands, thin edges
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_facecolor(SWISS["light_gray"] if i % 2 == 0 else "white")
            cell.set_edgecolor(SWISS["grid"])
            cell.set_linewidth(0.3)

    return table_bottom - 0.015


def _table_page(
    pdf: PdfPages,
    title: str,
    df: pd.DataFrame,
    area_label: str = "",
    col_widths: Optional[List[float]] = None,
):
    """Render a table as a compact PDF page (not stretched to fill)."""
    fig = _base_page(pdf, area_label=area_label)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = _section_header(ax, title, MARGIN["top"])
    _compact_table(ax, df.head(35), y, col_widths=col_widths)
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


def _build_study_site_lines(
    area: str,
    grid_data: Dict[str, Any],
    audit_result: SVFAuditResult,
    streets_svf: Optional[gpd.GeoDataFrame],
    buildings: Optional[gpd.GeoDataFrame],
) -> List[str]:
    """Page 2: Study Site & Data Sources."""
    area_label = _area_name(area)
    n_cells = grid_data.get("n_cells", 0)
    n_buildings = grid_data.get("n_buildings", 0)
    lines: List[str] = []

    lines.append("### Study Site")
    lines.append(
        f"{area_label} is an informal settlement in the municipality of "
        f"Rio de Janeiro. The analysis encompasses {n_buildings:,} building "
        f"footprints analyzed across {n_cells:,} grid cells at 10 m resolution."
    )
    lines.append("")

    lines.append("### Data Inventory")
    lines.append(
        f"**Municipal cadastral footprints:** {n_buildings:,} buildings "
        f"(height > 0 m, footprint >= 9 m2"
        + ("; buildings > 20 m excluded for informal areas" if n_buildings > 0 else "")
        + ")."
    )

    dtm_label = "ALS-derived DTM (5 m resolution)"
    lines.append(f"**Digital terrain model:** {dtm_label}.")

    # Road network
    n_streets = grid_data.get("n_streets", 0)
    if n_streets > 0:
        lines.append(f"**Road network:** {n_streets:,} segments.")

    # SVF data
    svf_count = audit_result.count
    lines.append(
        f"**Pre-computed SVF:** {svf_count:,} grid sample points "
        f"(Tregenza hemisphere ray-casting, {audit_result.n_rays or 145} rays, "
        f"1.5 m pedestrian height)."
    )

    if streets_svf is not None and not streets_svf.empty:
        n_seg = len(streets_svf)
        n_pts = (
            int(streets_svf["svf_count"].sum())
            if "svf_count" in streets_svf.columns
            else 0
        )
        pts_label = f"{n_pts:,} sample points along " if n_pts > 0 else ""
        lines.append(f"**Street-level SVF:** {pts_label}{n_seg:,} street segments.")
    lines.append("")

    lines.append("### Coordinate Reference System")
    lines.append("EPSG:31983 (SIRGAS 2000 / UTM zone 23S).")
    lines.append("")

    # Building height summary if available
    if buildings is not None and "height" in buildings.columns:
        h = buildings["height"].dropna()
        if len(h) > 0:
            lines.append("### Building Height Summary")
            lines.append(
                f"Mean height: {h.mean():.1f} m, median: {h.median():.1f} m, "
                f"std: {h.std():.1f} m. Range: {h.min():.1f}\u2013{h.max():.1f} m "
                f"(n = {len(h):,})."
            )
            lines.append("")

    return lines


def _build_methodology_lines(
    grid_data: Dict[str, Any],
    audit_result: SVFAuditResult,
    streets_svf: Optional[gpd.GeoDataFrame],
) -> List[str]:
    """Page 3: Methodology."""
    n_cells = grid_data.get("n_cells", 0)
    lines: List[str] = []

    lines.append("### Grid Morphometrics")
    lines.append(
        f"A 10 m regular grid ({n_cells:,} cells clipped to the study boundary) "
        f"provides the spatial framework. For each cell the following indicators "
        f"are computed:"
    )
    lines.append(
        "  \u2022 Plan area density (lambda_p): ratio of building footprint area to cell area."
    )
    lines.append(
        "  \u2022 Frontal area density (lambda_f): projected facade area in 8 cardinal "
        "directions, normalized by cell area times mean building height."
    )
    lines.append("  \u2022 Floor area ratio (FAR): total floor area to cell area.")
    lines.append(
        "  \u2022 Height variability (sigma_H): standard deviation of building heights within the cell."
    )
    lines.append(
        "  \u2022 Volumetric porosity: fraction of the canopy volume not occupied by buildings."
    )
    lines.append(
        "  \u2022 Slope and aspect: derived from the DTM using a 3x3 gradient kernel."
    )
    lines.append("")

    lines.append("### Sky View Factor")
    lines.append(
        f"SVF is pre-computed via Tregenza hemisphere ray-casting "
        f"({audit_result.n_rays or 145} rays per point) at 1.5 m above ground. "
        f"Point-level values are aggregated to 10 m grid cells by spatial mean."
    )
    lines.append("")

    if streets_svf is not None and not streets_svf.empty:
        n_seg = len(streets_svf)
        lines.append("### Street-Level SVF")
        lines.append(
            f"SVF sample points along road centerlines are aggregated to "
            f"{n_seg:,} street segments, providing a pedestrian-scale measure "
            f"of sky exposure along the movement network. Per-segment statistics "
            f"(mean, min, count) capture intra-street variability."
        )
        lines.append("")

    lines.append("### Street Orientation Entropy")
    lines.append(
        "Per-cell Shannon entropy of street bearings quantifies the regularity "
        "of the street network. Low entropy indicates a dominant orientation "
        "(grid-like layout); high entropy indicates an irregular, organic network."
    )
    lines.append("")

    lines.append("### References")
    lines.append(
        "Stewart, I. D. & Oke, T. R. (2012). Local Climate Zones for urban "
        "temperature studies. Bulletin of the AMS, 93(12), 1879\u20131900. "
        "LCZ 3 (Compact Low-Rise) benchmarks: SVF 0.2\u20130.6, lambda_p "
        "0.4\u20130.7, H 3\u201310 m."
    )
    lines.append("")

    return lines


def _build_audit_lines(
    audit_result: SVFAuditResult,
    grid_data: Dict[str, Any],
) -> List[str]:
    """Page 4: SVF Audit Results -- compact summary + inline findings."""
    lines: List[str] = []

    lines.append("### Computation Details")
    lines.append(f"**Method:** {audit_result.method}.")
    lines.append(f"**Rays:** {audit_result.n_rays or 'N/A'}.")
    lines.append(f"**Grid spacing:** {_fmt(audit_result.grid_spacing_m, 1)} m.")
    lines.append(
        f"**Terrain:** {'integrated' if audit_result.terrain_integrated else 'not confirmed'}."
    )
    lines.append("")

    lines.append("### Summary Statistics")
    lines.append(
        f"n = {audit_result.count:,}. "
        f"Mean SVF = {_fmt(audit_result.mean)} "
        f"(median {_fmt(audit_result.median)}, "
        f"std {_fmt(audit_result.std)}). "
        f"Range: [{_fmt(audit_result.min)}, {_fmt(audit_result.max)}]. "
        f"IQR: [{_fmt(audit_result.p25)}, {_fmt(audit_result.p75)}]."
    )
    lines.append(f"Coverage: {_fmt(audit_result.coverage_pct, 1)}% of study area.")
    lines.append("")

    # Audit issues
    n_crit = sum(1 for i in audit_result.issues if i.severity == "critical")
    n_warn = sum(1 for i in audit_result.issues if i.severity == "warning")
    n_info = sum(1 for i in audit_result.issues if i.severity == "info")

    lines.append("### Quality Flags")
    if not audit_result.issues:
        lines.append("\u2714 No issues flagged.")
    else:
        lines.append(f"{n_crit} critical, {n_warn} warnings, {n_info} informational:")
        lines.append("")
        for issue in audit_result.issues:
            icon = (
                "\u2716"
                if issue.severity == "critical"
                else "\u26a0"
                if issue.severity == "warning"
                else "\u2139"
            )
            lines.append(f"{icon} [{issue.severity.upper()}] {issue.message}")
    lines.append("")

    return lines


def _build_morphometric_narrative(
    grid_data: Dict[str, Any],
    buildings: Optional[gpd.GeoDataFrame],
) -> List[str]:
    """Page 9: Morphometric Results -- analytical narrative."""
    lines: List[str] = []
    g = grid_data

    svf_mean = g.get("svf_mean")
    lp_mean = g.get("lambda_p_mean")
    sigma_h = g.get("sigma_h_mean")
    h_mean = g.get("H_mean_mean")
    slope_mean = g.get("slope_deg_mean")
    porosity_mean = g.get("porosity_mean")
    lf_mean = g.get("lambda_f_mean_mean")

    lines.append("### Sky View Factor")
    if svf_mean is not None:
        q = _svf_qualifier(svf_mean)
        lcz_cmp = (
            "well below the LCZ 3 range (0.2\u20130.6), indicating conditions "
            "more extreme than standard compact low-rise classifications"
            if svf_mean < 0.2
            else "at the lower end of the LCZ 3 range (0.2\u20130.6)"
            if svf_mean < 0.4
            else "within the LCZ 3 range (0.2\u20130.6)"
        )
        lines.append(
            f"Mean SVF is {q} at {svf_mean:.3f}, {lcz_cmp}. "
            f"This level of sky obstruction limits radiative cooling at night "
            f"and reduces daytime ventilation potential through the urban canopy."
        )
    lines.append("")

    lines.append("### Building Density")
    if lp_mean is not None:
        q = _density_qualifier(lp_mean)
        lcz_lp = (
            "exceeds the LCZ 3 upper bound (0.70)"
            if lp_mean > 0.7
            else "falls within the LCZ 3 range (0.40\u20130.70)"
            if lp_mean > 0.4
            else "is below typical LCZ 3 values (0.40\u20130.70)"
        )
        lines.append(
            f"Plan area density lambda_p = {lp_mean:.3f} ({q}), which {lcz_lp}. "
        )
        if lf_mean is not None:
            lines.append(
                f"Mean frontal area density lambda_f = {lf_mean:.3f}, "
                f"reflecting the three-dimensional obstruction presented by "
                f"building facades to horizontal airflow."
            )
    lines.append("")

    lines.append("### Building Heights")
    if h_mean is not None:
        lcz_h = (
            "within the LCZ 3 benchmark (3\u201310 m)"
            if 3 <= h_mean <= 10
            else "below the LCZ 3 lower bound of 3 m"
            if h_mean < 3
            else "above the LCZ 3 upper bound of 10 m"
        )
        h_text = f"Mean building height is {h_mean:.1f} m, {lcz_h}."
        if sigma_h is not None:
            h_text += f" Height variability (sigma_H = {sigma_h:.2f} m) " + (
                "is high, creating significant vertical roughness that "
                "promotes turbulent mixing but may channel wind in narrow passages."
                if sigma_h > 3
                else "indicates a heterogeneous building stock typical of "
                "incrementally constructed settlements."
                if sigma_h > 1.5
                else "is relatively low, suggesting uniform building heights."
            )
        lines.append(h_text)

        if buildings is not None and "height" in buildings.columns:
            h = buildings["height"].dropna()
            if len(h) > 0:
                pct_1story = 100.0 * (h <= 4).sum() / len(h)
                pct_3plus = 100.0 * (h > 9).sum() / len(h)
                lines.append(
                    f"Of {len(h):,} buildings, {pct_1story:.0f}% are single-story "
                    f"(<= 4 m) and {pct_3plus:.0f}% exceed three stories (> 9 m)."
                )
    lines.append("")

    lines.append("### Terrain & Porosity")
    if slope_mean is not None:
        terrain_text = f"Mean terrain slope is {slope_mean:.1f}\u00b0. "
        if slope_mean > 15:
            terrain_text += (
                "Steep slopes (> 15\u00b0) significantly affect wind patterns "
                "and render flat-terrain ventilation assumptions unreliable."
            )
        elif slope_mean > 5:
            terrain_text += (
                "Moderate slopes influence wind channeling along contour lines "
                "and create thermally-driven upslope/downslope flows."
            )
        else:
            terrain_text += "Relatively flat terrain."
        lines.append(terrain_text)
    if porosity_mean is not None:
        lines.append(
            f"Mean volumetric porosity = {porosity_mean:.3f}. "
            + (
                "Low porosity indicates limited void space within the urban "
                "canopy for air circulation."
                if porosity_mean < 0.4
                else "Moderate porosity allows some ventilation pathways "
                "within the canopy layer."
                if porosity_mean < 0.7
                else "High porosity with significant void space."
            )
        )
    lines.append("")

    return lines


def _build_spatial_patterns(grid_data: Dict[str, Any]) -> List[str]:
    """Page 11: Spatial Patterns & Correlations narrative."""
    lines: List[str] = []

    svf_slope_r = grid_data.get("svf_slope_r")
    svf_lp_r = grid_data.get("svf_lp_r")

    lines.append("### Density\u2013Enclosure Coupling")
    if svf_lp_r is not None:
        strength = (
            "Strong"
            if abs(svf_lp_r) > 0.5
            else "Moderate"
            if abs(svf_lp_r) > 0.3
            else "Weak"
        )
        lines.append(
            f"SVF and lambda_p exhibit a Pearson correlation of r = {svf_lp_r:.3f}. "
            f"{strength} "
            + (
                "negative correlation confirms that higher plan-area density "
                "directly reduces sky access, as expected from geometric "
                "principles. Cells with lambda_p > 0.5 consistently show "
                "SVF < 0.15, indicating severe mutual obstruction."
                if svf_lp_r < -0.5
                else "negative correlation between density and sky access. "
                "The relationship is not purely geometric: building height "
                "variability and irregular setbacks introduce scatter."
                if svf_lp_r < -0.3
                else "correlation suggests that sky obstruction is driven "
                "more by building height and proximity than by plan density "
                "alone. This decoupling is characteristic of steeply sloped "
                "sites where vertical offsets between buildings open sky views."
            )
        )
    lines.append("")

    lines.append("### Slope\u2013Density Relationship")
    if svf_slope_r is not None:
        lines.append(
            f"SVF and terrain slope correlate at r = {svf_slope_r:.3f}. "
            + (
                "Positive correlation indicates steeper areas tend toward "
                "higher SVF, likely because buildings thin out on steep "
                "terrain and vertical offsets between rows open sky views."
                if svf_slope_r > 0.2
                else "Negative correlation suggests denser building patterns "
                "or that hillside terrain itself obstructs sky through "
                "topographic shadowing."
                if svf_slope_r < -0.2
                else "No strong slope\u2013SVF relationship, suggesting "
                "building density is largely independent of terrain steepness "
                "in this settlement."
            )
        )
    lines.append("")

    lines.append("### Implications for Ventilation")
    lines.append(
        "The combination of low SVF and high plan-area density creates conditions "
        "where nocturnal radiative cooling is impaired (limited sky view) and "
        "daytime convective cooling is restricted (limited porosity). Wind "
        "channeling through the street network becomes the dominant ventilation "
        "mechanism, making street orientation and width the critical design "
        "parameters for thermal comfort."
    )
    lines.append("")

    return lines


def _build_strata_table(grid: gpd.GeoDataFrame) -> Optional[pd.DataFrame]:
    """Build a compact CFD strata DataFrame (SVF x Slope x lambda_p).

    Returns None if insufficient data.
    """
    if not all(c in grid.columns for c in ["svf", "slope_deg", "lambda_p"]):
        return None

    valid = grid[["svf", "slope_deg", "lambda_p"]].dropna()
    if len(valid) < 10:
        return None

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

    strata = (
        valid.groupby([svf_bins, slope_bins, lp_bins], observed=True)
        .size()
        .reset_index(name="Cells")
    )
    strata = strata[strata["Cells"] > 0].copy()
    strata.columns = ["SVF", "Slope", "\u03bbp", "Cells"]
    strata["% of Total"] = (100.0 * strata["Cells"] / len(valid)).round(1)
    strata = strata.sort_values("Cells", ascending=False).reset_index(drop=True)
    return strata


def _build_cfd_lines(
    grid_data: Dict[str, Any],
    strata_df: Optional[pd.DataFrame],
) -> List[str]:
    """Page 15: CFD Campaign Implications -- narrative (table rendered separately)."""
    lines: List[str] = []

    lines.append(
        "The CFD simulation campaign requires representative patches drawn "
        "from the morphometric feature space defined by SVF, terrain slope, "
        "and plan-area density. The table below shows occupied strata and "
        "their cell counts."
    )
    lines.append("")

    if strata_df is not None:
        n_strata = len(strata_df)
        top = strata_df.iloc[0]
        lp_col = "\u03bbp"
        top_lp = top[lp_col]
        top_pct = top["% of Total"]
        lines.append(
            f"The study area populates {n_strata} distinct strata. "
            f"The dominant stratum (SVF {top['SVF']}, slope {top['Slope']}, "
            f"\u03bbp {top_lp}) contains {int(top['Cells'])} cells "
            f"({top_pct:.1f}% of total)."
        )
        lines.append("")

        # Identify under-represented strata
        small = strata_df[strata_df["Cells"] <= 5]
        if len(small) > 0:
            lines.append(
                f"{len(small)} strata contain 5 or fewer cells and may be "
                f"under-represented for robust CFD patch selection. "
                f"Cross-site sampling from other favela datasets (Rocinha, "
                f"Rio das Pedras, Mare) should be considered to fill gaps "
                f"in the parameter space."
            )
            lines.append("")

    lines.append(
        "Strata absent from this site would need to be sourced from other "
        "settlements to ensure full coverage of the Brazilian favela "
        "morphometric parameter space."
    )
    lines.append("")

    return lines


def _build_street_svf_lines(
    streets_svf: gpd.GeoDataFrame,
) -> List[str]:
    """Narrative for the street-level SVF discussion (inserted before/after figure)."""
    lines: List[str] = []
    svf_col = "svf_mean" if "svf_mean" in streets_svf.columns else "svf"

    if svf_col not in streets_svf.columns:
        lines.append("Street-level SVF data available but SVF column not found.")
        return lines

    vals = streets_svf[svf_col].dropna()
    if len(vals) == 0:
        return lines

    lines.append("### Street-Level SVF")
    lines.append(
        f"Across {len(vals):,} street segments, mean SVF = {vals.mean():.3f} "
        f"(std = {vals.std():.3f}). The most enclosed segments "
        f"(min SVF = {vals.min():.3f}) represent narrow alleys where sky is "
        f"almost entirely occluded; the most open (max = {vals.max():.3f}) are "
        f"wider collector roads or segments at the settlement periphery."
    )
    lines.append("")

    q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
    lines.append(
        f"The interquartile range [{q25:.3f}, {q75:.3f}] captures the typical "
        f"pedestrian experience. "
        + (
            "Over 75% of streets fall below SVF 0.2, indicating that severe "
            "sky obstruction is the norm, not the exception."
            if q75 < 0.2
            else "Most streets experience restricted sky access, though a "
            "meaningful fraction exceeds SVF 0.3."
            if q75 < 0.4
            else "Street-level SVF shows moderate variability."
        )
    )
    lines.append("")

    return lines


# ═══════════════════════════════════════════════════════════════════════════
#  COMPOSITE PAGES (text + table on same page)
# ═══════════════════════════════════════════════════════════════════════════


def _morphometric_results_page(
    pdf: PdfPages,
    grid: gpd.GeoDataFrame,
    grid_data: Dict[str, Any],
    buildings: Optional[gpd.GeoDataFrame],
    area_label: str,
):
    """Page 9: Compact summary-statistics table + analytical narrative."""
    fig = _base_page(pdf, area_label=area_label)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = _section_header(ax, "Morphometric Results", MARGIN["top"])

    # ---- Compact summary table ----
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

    rows = []
    for col, label in indicator_cols:
        if col in grid.columns:
            vals = grid[col].dropna()
            if len(vals) > 0:
                rows.append(
                    {
                        "Indicator": label,
                        "n": f"{len(vals):,}",
                        "Mean": f"{vals.mean():.3f}",
                        "Std": f"{vals.std():.3f}",
                        "Min": f"{vals.min():.3f}",
                        "Median": f"{vals.median():.3f}",
                        "Max": f"{vals.max():.3f}",
                    }
                )

    if rows:
        stats_df = pd.DataFrame(rows)
        y = _compact_table(
            ax,
            stats_df,
            y,
            col_widths=[0.18, 0.10, 0.13, 0.13, 0.13, 0.13, 0.13],
            fontsize=8,
            row_height=0.023,
        )
        y -= 0.005

    # ---- Narrative below table ----
    narrative = _build_morphometric_narrative(grid_data, buildings)
    _render_text_block(ax, narrative, y, fontsize=9.0, line_height=0.019, wrap_width=98)

    _save(pdf, fig)


def _cfd_implications_page(
    pdf: PdfPages,
    grid: gpd.GeoDataFrame,
    grid_data: Dict[str, Any],
    area_label: str,
):
    """Page 15: CFD strata as compact table + narrative."""
    fig = _base_page(pdf, area_label=area_label)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = _section_header(ax, "CFD Campaign Implications", MARGIN["top"])

    strata_df = _build_strata_table(grid)

    # Narrative first
    cfd_lines = _build_cfd_lines(grid_data, strata_df)
    y = _render_text_block(ax, cfd_lines, y, fontsize=9.5, line_height=0.020)

    # Then the compact strata table
    if strata_df is not None and len(strata_df) > 0:
        y -= 0.005
        # Show top 25 strata
        display_strata = strata_df.head(25)
        _compact_table(
            ax,
            display_strata,
            y,
            col_widths=[0.20, 0.20, 0.20, 0.15, 0.15],
            fontsize=7.5,
            row_height=0.020,
        )

    _save(pdf, fig)


def _audit_results_page(
    pdf: PdfPages,
    audit_result: SVFAuditResult,
    grid_data: Dict[str, Any],
    area_label: str,
):
    """Page 4: SVF Audit -- text + small data-quality table."""
    fig = _base_page(pdf, area_label=area_label)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = _section_header(ax, "SVF Audit Results", MARGIN["top"])

    audit_lines = _build_audit_lines(audit_result, grid_data)
    y = _render_text_block(ax, audit_lines, y, fontsize=9.5, line_height=0.020)

    # Small quality table
    quality_rows = [
        {"Metric": "Sample points", "Value": f"{audit_result.count:,}"},
        {"Metric": "Mean SVF", "Value": _fmt(audit_result.mean)},
        {"Metric": "Median SVF", "Value": _fmt(audit_result.median)},
        {"Metric": "Std deviation", "Value": _fmt(audit_result.std)},
        {"Metric": "5th percentile", "Value": _fmt(audit_result.p5)},
        {"Metric": "95th percentile", "Value": _fmt(audit_result.p95)},
        {"Metric": "Coverage", "Value": f"{_fmt(audit_result.coverage_pct, 1)}%"},
        {
            "Metric": "Grid spacing",
            "Value": f"{_fmt(audit_result.grid_spacing_m, 1)} m",
        },
    ]
    q_df = pd.DataFrame(quality_rows)
    _compact_table(ax, q_df, y, col_widths=[0.45, 0.45], fontsize=8.5, row_height=0.024)

    _save(pdf, fig)


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
    streets_svf: Optional[gpd.GeoDataFrame] = None,
    buildings: Optional[gpd.GeoDataFrame] = None,
    dtm_path: Optional[Path] = None,
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
        Mapping of figure keys to file paths. Expected keys:
        ``site_overview``, ``building_heights``, ``street_svf``,
        ``svf_map``, ``morphometric_panel``, ``svf_slope_scatter``,
        ``correlation_matrix``, ``distributions``.
    n_buildings : int
        Total number of buildings in the study area.
    streets_svf : GeoDataFrame, optional
        Segment-level street SVF (one row per street segment).
    buildings : GeoDataFrame, optional
        Building footprints with ``height`` column for height stats.
    dtm_path : Path, optional
        Path to the DTM raster (for terrain metadata).
    """
    _reset()
    area_label = _area_name(area)

    # ── Pre-compute grid statistics for narrative ────────────────────
    grid_data: Dict[str, Any] = {
        "n_cells": len(grid),
        "n_buildings": n_buildings,
    }

    # Count streets from grid if available
    if "street_orientation_entropy" in grid.columns:
        n_streets_est = int(grid["street_orientation_entropy"].notna().sum())
        grid_data["n_streets"] = n_streets_est

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

    logger.info("Generating PDF report: %s", output_path)

    with PdfPages(output_path) as pdf:
        # ── Page 1: Title ────────────────────────────────────────────
        _title_page(pdf, area)

        # ── Page 2: Study Site & Data Sources ────────────────────────
        site_lines = _build_study_site_lines(
            area,
            grid_data,
            audit_result,
            streets_svf,
            buildings,
        )
        _text_page(pdf, "Study Site & Data Sources", site_lines, area_label)

        # ── Page 3: Methodology ──────────────────────────────────────
        method_lines = _build_methodology_lines(
            grid_data,
            audit_result,
            streets_svf,
        )
        _text_page(pdf, "Methodology", method_lines, area_label)

        # ── Page 4: SVF Audit Results ────────────────────────────────
        _audit_results_page(pdf, audit_result, grid_data, area_label)

        # ── Pages 5-6: Site Overview + Building Heights (figures) ────
        _fig = figure_paths.get("site_overview")
        if _fig and _fig.exists():
            _figure_page(
                pdf,
                "Site Overview",
                _fig,
                caption=(
                    "Building footprints on hillshade DTM with elevation contours."
                ),
                area_label=area_label,
            )

        _fig = figure_paths.get("building_heights")
        if _fig and _fig.exists():
            _figure_page(
                pdf,
                "Building Heights",
                _fig,
                caption=(
                    "Building heights draped on terrain. Color encodes height (m). "
                    "Buildings > 20 m excluded from informal-area analysis."
                ),
                area_label=area_label,
            )

        # ── Page 7: Street-Level SVF figure ──────────────────────────
        _fig = figure_paths.get("street_svf")
        if _fig and _fig.exists():
            caption_text = "Mean SVF per street segment."
            if streets_svf is not None:
                svf_col = "svf_mean" if "svf_mean" in streets_svf.columns else "svf"
                if svf_col in streets_svf.columns:
                    vals = streets_svf[svf_col].dropna()
                    if len(vals) > 0:
                        caption_text = (
                            f"Mean SVF per street segment (n = {len(vals):,}, "
                            f"overall mean = {vals.mean():.3f}). "
                            f"Red segments indicate severe sky obstruction; "
                            f"blue indicates relatively open corridors."
                        )
            _figure_page(
                pdf,
                "Street-Level Sky View Factor",
                _fig,
                caption=caption_text,
                area_label=area_label,
            )

        # ── Page 8: SVF Grid Map ────────────────────────────────────
        _fig = figure_paths.get("svf_map")
        if _fig and _fig.exists():
            _figure_page(
                pdf,
                "Sky View Factor Map (10 m Grid)",
                _fig,
                caption=(
                    "SVF on 10 m grid with diverging colormap (breakpoint at "
                    "SVF = 0.15). Red = severe sky obstruction; blue = open sky."
                ),
                area_label=area_label,
            )

        # ── Page 9: Morphometric Results (table + narrative) ─────────
        _morphometric_results_page(
            pdf,
            grid,
            grid_data,
            buildings,
            area_label,
        )

        # ── Page 10: Morphometric Panel (figure) ────────────────────
        _fig = figure_paths.get("morphometric_panel")
        if _fig and _fig.exists():
            _figure_page(
                pdf,
                "Morphometric Indicator Panel",
                _fig,
                caption=(
                    "Six indicators at 10 m resolution: plan area density, "
                    "frontal area density, porosity, height variability, "
                    "street orientation entropy, and terrain slope."
                ),
                area_label=area_label,
            )

        # ── Page 11: Spatial Patterns (text) ─────────────────────────
        patterns_lines = _build_spatial_patterns(grid_data)
        # Prepend street SVF narrative if available
        if streets_svf is not None and not streets_svf.empty:
            street_lines = _build_street_svf_lines(streets_svf)
            patterns_lines = street_lines + patterns_lines
        _text_page(
            pdf,
            "Spatial Patterns & Correlations",
            patterns_lines,
            area_label,
        )

        # ── Page 12: SVF-Slope Scatter (figure) ─────────────────────
        _fig = figure_paths.get("svf_slope_scatter")
        if _fig and _fig.exists():
            _figure_page(
                pdf,
                "SVF vs. Terrain Slope",
                _fig,
                caption=(
                    "Each point represents a 10 m grid cell. Color encodes "
                    "plan area density. Marginal histograms show univariate "
                    "distributions."
                ),
                area_label=area_label,
            )

        # ── Page 13: Correlation Matrix (figure) ────────────────────
        _fig = figure_paths.get("correlation_matrix")
        if _fig and _fig.exists():
            _figure_page(
                pdf,
                "Indicator Correlation Matrix",
                _fig,
                caption=(
                    "Pairwise Pearson correlations. Bold values indicate "
                    "|r| > 0.7 (high collinearity flagged for predictor "
                    "selection)."
                ),
                area_label=area_label,
            )

        # ── Page 14: Distributions (figure) ──────────────────────────
        _fig = figure_paths.get("distributions")
        if _fig and _fig.exists():
            _figure_page(
                pdf,
                "Morphometric Distributions",
                _fig,
                caption=(
                    "Violin plots showing the distribution of each indicator. "
                    "Green bands indicate LCZ 3 (Compact Low-Rise) reference "
                    "ranges from Stewart & Oke (2012)."
                ),
                area_label=area_label,
            )

        # ── Page 15: CFD Campaign Implications (table + text) ────────
        _cfd_implications_page(pdf, grid, grid_data, area_label)

        n_pages = pdf.get_pagecount()

    logger.info("PDF report saved: %s (%d pages)", output_path, n_pages)
