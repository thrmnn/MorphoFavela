"""P-00 — the PI's package spec (P-01..P-09, verbatim, 2026-09-23) encoded
as data, plus a mechanical conformance check against a BUILT package
directory.

Structural fix (PI, 2026-09-27): the spec used to live only in README
prose, so nothing could tell "the package matches the spec" from "the
package happens to look similar" — conformance was invisible. ``SPEC``
below is the verbatim requirement text; each item's ``parts`` are
mechanical predicates over ``package_dir`` (a built
``outputs/_packages/mare_om2/<version>/`` directory): does a named file
exist, are named columns present with coverage read from
``p07_quality_report.json``, does the data dictionary have the right rows
and columns, does the README have the right headings, does the changelog
have a dated entry for this version.

An item is ``delivered`` if every part is, ``pending`` if none are, and
``partial`` otherwise; descoped parts are handled by the item rule in
``_item_status``. Every ``pending`` part carries ``pending_on`` — the
``tasks.json`` id(s) that unblock it — and a ``reason`` read verbatim from
the data dictionary's own ``source``/``limits`` text (never invented, per
CLAUDE.md's "never fabricate a value").

A part may instead be ``descoped``: a deliberate, decided cut (PI,
``om_v013_descope``), not a gap. It carries ``decision`` + ``reason`` and
never ``pending_on`` (nothing unblocks it in this version).

``conformance(package_dir)`` returns the full result; the build writes it
to ``p00_spec_conformance.json``/``.csv`` and renders it into the README's
"Conformance to the package spec" section
(``render_conformance_markdown``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: stated minimum non-null coverage for a numeric P-04/P-06 column to
#: count as "delivered" — the threshold is fixed here; the ACTUAL coverage
#: compared against it is always read from p07_quality_report.json, never
#: recomputed or guessed by this module.
MIN_COVERAGE_FRACTION = 0.95

BUFFER_RADII_M = (5, 10, 20, 50)

_DICT_REQUIRED_COLS = ["id", "definition", "unit", "source", "method", "limits", "status"]


@dataclass
class PartResult:
    name: str
    status: str  # "delivered" | "pending" | "descoped"
    evidence: str = ""
    pending_on: list[str] = field(default_factory=list)
    reason: str = ""
    decision: str = ""

    def __post_init__(self) -> None:
        if self.status == "descoped":
            if not self.decision:
                raise ValueError(f"part '{self.name}' is descoped without a decision id")
            if not self.reason:
                raise ValueError(f"part '{self.name}' is descoped without a reason")
            if self.pending_on:
                raise ValueError(f"part '{self.name}' is descoped but carries pending_on {self.pending_on}")
        elif self.decision:
            raise ValueError(f"part '{self.name}' has a decision id but status '{self.status}'")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
            "pending_on": list(self.pending_on),
            "reason": self.reason,
            "decision": self.decision,
        }


# ------------------------------------------------------------- file reads --

def _points_df(package_dir: Path) -> pd.DataFrame | None:
    pq = package_dir / "OM2" / "points.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = package_dir / "OM2" / "points.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return None


def _shade_df(package_dir: Path) -> pd.DataFrame | None:
    pq = package_dir / "p05_building_shade.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = package_dir / "p05_building_shade.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return None


def _quality_report(package_dir: Path) -> dict:
    p = package_dir / "OM2" / "p07_quality_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _dictionary_df(package_dir: Path) -> pd.DataFrame | None:
    p = package_dir / "p08_data_dictionary.csv"
    return pd.read_csv(p) if p.exists() else None


def _dictionary_row(package_dir: Path, var_id: str) -> dict | None:
    df = _dictionary_df(package_dir)
    if df is None or "id" not in df.columns:
        return None
    hit = df[df["id"] == var_id]
    return None if hit.empty else hit.iloc[0].to_dict()


def _readme_text(package_dir: Path) -> str:
    p = package_dir / "README.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _changelog_text(package_dir: Path) -> str:
    p = package_dir / "CHANGELOG.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _current_version(package_dir: Path) -> str:
    manifest_p = package_dir / "manifest.json"
    if manifest_p.exists():
        v = json.loads(manifest_p.read_text(encoding="utf-8")).get("package_version")
        if v:
            return v
    return package_dir.name


_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)(?:\.(\d+))?$")


def _version_key(name: str) -> tuple[int, int, int]:
    m = _VERSION_RE.match(name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) if m else (-1, -1, -1)


def _previous_version_dir(package_dir: Path) -> Path | None:
    """The next-oldest sibling version directory under the same
    mare_om2/ root, or None if this is the first version on disk (or
    package_dir's parent isn't laid out that way — e.g. a sabotage-test
    copy under tmp_path)."""
    parent = package_dir.parent
    if not parent.is_dir():
        return None
    current = _version_key(package_dir.name)
    candidates = [
        p for p in parent.iterdir()
        if p.is_dir() and not p.name.startswith("_") and _version_key(p.name) < current
    ]
    return max(candidates, key=lambda p: _version_key(p.name)) if candidates else None


# --------------------------------------------------------------- checks --

def _columns_part(name: str, package_dir: Path, cols: list[str], require_coverage: bool = True) -> PartResult:
    df = _points_df(package_dir)
    if df is None:
        return PartResult(name, "pending", evidence="OM2/points table not found")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return PartResult(name, "pending", evidence=f"OM2/points missing column(s): {', '.join(missing)}")
    if not require_coverage:
        return PartResult(name, "delivered", evidence=f"OM2/points has column(s): {', '.join(cols)}")
    q = _quality_report(package_dir)
    qcols = q.get("columns", {})
    low = {}
    covs = {}
    for c in cols:
        frac = qcols.get(c, {}).get("coverage_fraction")
        covs[c] = frac
        if frac is None or frac < MIN_COVERAGE_FRACTION:
            low[c] = frac
    if low:
        return PartResult(
            name, "pending",
            evidence=f"p07_quality_report.json coverage below {MIN_COVERAGE_FRACTION:.0%}: "
                     + ", ".join(f"{c}={low[c]}" for c in low),
        )
    evidence = (
        f"OM2/points columns {', '.join(cols)} present; p07_quality_report.json coverage >= "
        f"{MIN_COVERAGE_FRACTION:.0%}: " + ", ".join(f"{c}={covs[c]:.3f}" for c in cols)
    )
    return PartResult(name, "delivered", evidence=evidence)


def _file_exists_part(name: str, package_dir: Path, rel_paths: list[str]) -> PartResult:
    missing = [p for p in rel_paths if not (package_dir / p).exists()]
    if missing:
        return PartResult(name, "pending", evidence=f"missing file(s): {', '.join(missing)}")
    return PartResult(name, "delivered", evidence=f"present: {', '.join(rel_paths)}")


def _first_meaningful(*values: object) -> str | None:
    """First value that is neither empty nor the dictionary's own "no
    caveat" placeholder ('-'), or None if every value is empty/placeholder."""
    for v in values:
        s = "" if v is None else str(v).strip()
        if s and s != "-":
            return s
    return None


def _pending_part(name: str, package_dir: Path, dict_id: str, pending_on: list[str]) -> PartResult:
    """A part this package cannot deliver yet, by PI ruling — never a
    'maybe delivered' guess. reason is read verbatim from the data
    dictionary's own row for dict_id: whichever of limits/unit/source
    actually carries text (some rows put the caveat in 'unit', e.g. the
    shade table's 'timestamp' row; 'limits' there is just '-')."""
    row = _dictionary_row(package_dir, dict_id)
    if row is None:
        return PartResult(
            name, "pending",
            evidence=f"p08_data_dictionary.csv has no row for '{dict_id}'",
            pending_on=list(pending_on),
            reason="no dictionary row found — see tasks.json for status",
        )
    reason = _first_meaningful(row.get("limits"), row.get("unit"), row.get("source")) or "PENDING (see data dictionary)"
    evidence = f"p08_data_dictionary.csv row '{dict_id}': status={row.get('status')}"
    return PartResult(name, "pending", evidence=evidence, pending_on=list(pending_on), reason=reason)


#: PI decision (resolved 2026-10-01) that dropped the terrestrial-LiDAR
#: analysis and tree shade from v0.1.3. Both stay candidates for a later
#: version (OMPKG2); this is a cut of this version, not a cancellation.
DESCOPE_DECISION = "om_v013_descope"
_DESCOPE_REASON = (
    "Dropped from this version by PI decision; candidate for a later version (OMPKG2)."
)


def _descoped_part(name: str, evidence: str) -> PartResult:
    return PartResult(
        name, "descoped",
        evidence=f"DESCOPED ({DESCOPE_DECISION}): {evidence}",
        reason=_DESCOPE_REASON, decision=DESCOPE_DECISION,
    )


def _dictionary_rows_part(name: str, package_dir: Path, ids: list[str]) -> PartResult:
    df = _dictionary_df(package_dir)
    if df is None:
        return PartResult(name, "pending", evidence="p08_data_dictionary.csv not found")
    missing_cols = [c for c in _DICT_REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        return PartResult(name, "pending", evidence=f"p08_data_dictionary.csv missing column(s): {missing_cols}")
    present_ids = set(df["id"])
    missing_ids = [i for i in ids if i not in present_ids]
    if missing_ids:
        return PartResult(name, "pending", evidence=f"p08_data_dictionary.csv missing row(s) for: {missing_ids}")
    return PartResult(name, "delivered", evidence=f"p08_data_dictionary.csv has rows for {', '.join(ids)}")


def _readme_heading_part(name: str, package_dir: Path, heading: str) -> PartResult:
    text = _readme_text(package_dir)
    if not text:
        return PartResult(name, "pending", evidence="README.md not found")
    if heading in text:
        return PartResult(name, "delivered", evidence=f"README.md contains heading {heading!r}")
    return PartResult(name, "pending", evidence=f"README.md missing heading {heading!r}")


_CHANGELOG_ENTRY_RE = re.compile(r"^##\s+(v[\d.]+)\s+—\s+(\d{4}-\d{2}-\d{2})\s*$", re.M)


def _changelog_entry_part(name: str, package_dir: Path) -> PartResult:
    text = _changelog_text(package_dir)
    if not text:
        return PartResult(name, "pending", evidence="CHANGELOG.md not found")
    version = _current_version(package_dir)
    for v, d in _CHANGELOG_ENTRY_RE.findall(text):
        if v == version:
            return PartResult(name, "delivered", evidence=f"CHANGELOG.md has '## {version} — {d}'")
    return PartResult(name, "pending", evidence=f"CHANGELOG.md has no dated entry for {version}")


def _dictionary_required_columns_part(name: str, package_dir: Path) -> PartResult:
    df = _dictionary_df(package_dir)
    if df is None:
        return PartResult(name, "pending", evidence="p08_data_dictionary.csv not found")
    missing = [c for c in _DICT_REQUIRED_COLS if c not in df.columns]
    if missing:
        return PartResult(name, "pending", evidence=f"p08_data_dictionary.csv missing column(s): {missing}")
    if len(df) == 0:
        return PartResult(name, "pending", evidence="p08_data_dictionary.csv has the required columns but 0 rows")
    return PartResult(name, "delivered", evidence=f"{len(df)} rows, columns {_DICT_REQUIRED_COLS}")


def _ids_never_reused_part(name: str, package_dir: Path) -> PartResult:
    df = _dictionary_df(package_dir)
    if df is None or "id" not in df.columns:
        return PartResult(name, "pending", evidence="p08_data_dictionary.csv not found")
    ids = df["id"].tolist()
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        return PartResult(name, "pending", evidence=f"duplicate id(s) within p08_data_dictionary.csv: {dupes}")
    prev_dir = _previous_version_dir(package_dir)
    if prev_dir is None:
        return PartResult(
            name, "delivered",
            evidence=f"{len(ids)} unique ids in p08_data_dictionary.csv (no prior version directory to compare against)",
        )
    prev_df = _dictionary_df(prev_dir)
    if prev_df is None or "id" not in prev_df.columns:
        return PartResult(name, "delivered", evidence=f"{len(ids)} unique ids; prior version {prev_dir.name} has no dictionary to compare against")
    dropped = sorted(set(prev_df["id"]) - set(ids))
    if dropped:
        return PartResult(
            name, "pending",
            evidence=f"id(s) in {prev_dir.name}'s dictionary but absent from {package_dir.name}'s: {dropped}",
        )
    return PartResult(
        name, "delivered",
        evidence=f"{len(ids)} unique ids; every id from {prev_dir.name} ({len(prev_df)} rows) still present",
    )


def _point_id_unique_part(name: str, package_dir: Path) -> PartResult:
    df = _points_df(package_dir)
    if df is None or "point_id" not in df.columns:
        return PartResult(name, "pending", evidence="OM2/points table missing or has no point_id column")
    ok_pattern = df["point_id"].astype(str).str.match(r"^OM2-\d{6}$").all()
    if df["point_id"].is_unique and ok_pattern:
        return PartResult(name, "delivered", evidence=f"{len(df)} point_id values, unique, matching OM2-###### pattern")
    if not df["point_id"].is_unique:
        return PartResult(name, "pending", evidence="point_id column has duplicate values")
    return PartResult(name, "pending", evidence="point_id column has value(s) not matching the OM2-###### pattern")


def _no_segments_imposed_part(name: str, package_dir: Path) -> PartResult:
    df = _points_df(package_dir)
    if df is None:
        return PartResult(name, "pending", evidence="OM2/points table missing")
    if "segment_id" in df.columns:
        return PartResult(
            name, "pending",
            evidence="OM2/points table carries a segment_id column — segments belong only in "
                     "aggregate_to_segments.py's output, not the point table itself",
        )
    return PartResult(name, "delivered", evidence="OM2/points table has no segment_id column")


def _buffer_columns_part(name: str, package_dir: Path) -> PartResult:
    cols = []
    for r in BUFFER_RADII_M:
        cols += [f"lambda_p_buffer_{r}m", f"building_count_buffer_{r}m", f"building_height_mean_buffer_{r}m"]
    return _columns_part(name, package_dir, cols, require_coverage=False)


def _building_shade_table_part(name: str, package_dir: Path) -> PartResult:
    df = _shade_df(package_dir)
    if df is None:
        return PartResult(name, "pending", evidence="p05_building_shade table not found")
    required = {"point_id", "timestamp", "date", "sun_altitude_deg", "sun_azimuth_deg", "shaded", "tree_shade"}
    missing = required - set(df.columns)
    if missing:
        return PartResult(name, "pending", evidence=f"p05_building_shade missing column(s): {sorted(missing)}")
    if len(df) == 0:
        return PartResult(
            name, "pending",
            evidence="p05_building_shade has the correct schema but 0 rows (no campaign CSVs found at build time)",
        )
    return PartResult(name, "delivered", evidence=f"p05_building_shade: {len(df)} rows, schema {sorted(required)}")


def _tree_shade_part(name: str, package_dir: Path) -> PartResult:
    df = _shade_df(package_dir)
    if df is None or "tree_shade" not in df.columns:
        evidence = "tree shade not computed in this version; p05_building_shade missing or has no tree_shade column"
    else:
        all_null = bool(df["tree_shade"].isna().all()) if len(df) else True
        evidence = (
            "tree shade not computed in this version; p05_building_shade.tree_shade stays as a reserved column, "
            + ("all null" if all_null else "with non-null values (unexpected)")
            + f" ({len(df)} rows)"
        )
    return _descoped_part(name, evidence)


_VENTILATION_IDS = [
    "ventilation_wind_alignment_proxy",
    "ventilation_frontal_area_proxy",
    "ventilation_openness_proxy",
    "ventilation_dist_open_space_proxy_m",
]


def _ventilation_labelled_proxy_part(name: str, package_dir: Path) -> PartResult:
    df = _dictionary_df(package_dir)
    if df is None:
        return PartResult(name, "pending", evidence="p08_data_dictionary.csv not found")
    missing, unlabelled = [], []
    for vid in _VENTILATION_IDS:
        hit = df[df["id"] == vid] if "id" in df.columns else df.iloc[0:0]
        if hit.empty:
            missing.append(vid)
            continue
        text = " ".join(str(hit.iloc[0].get(c, "")) for c in ("definition", "limits")).upper()
        if "PROXY" not in text:
            unlabelled.append(vid)
    if missing:
        return PartResult(name, "pending", evidence=f"p08_data_dictionary.csv missing row(s): {missing}")
    if unlabelled:
        return PartResult(name, "pending", evidence=f"dictionary row(s) not labelled PROXY: {unlabelled}")
    return PartResult(name, "delivered", evidence=f"all {len(_VENTILATION_IDS)} ventilation dictionary rows contain 'PROXY'")


def _coverage_mask_part(name: str, package_dir: Path) -> PartResult:
    j = package_dir / "OM2" / "p07_quality_report.json"
    c = package_dir / "OM2" / "p07_quality_report.csv"
    missing = [p.name for p in (j, c) if not p.exists()]
    if missing:
        return PartResult(name, "pending", evidence=f"missing: {missing}")
    q = _quality_report(package_dir)
    n_cols = len(q.get("columns", {}))
    return PartResult(name, "delivered", evidence=f"p07_quality_report.json/.csv present, {n_cols} columns, n_points={q.get('n_points')}")


def _known_gaps_part(name: str, package_dir: Path) -> PartResult:
    q = _quality_report(package_dir)
    pending_items = q.get("pending_items")
    if not pending_items:
        return PartResult(name, "pending", evidence="p07_quality_report.json has no pending_items")
    return PartResult(name, "delivered", evidence=f"pending_items: {pending_items}")


# ------------------------------------------------------------------ SPEC --

SPEC: list[dict] = [
    {
        "id": "P-01",
        "title": "README",
        "requirement": "README: sources and dates, coordinate system (SIRGAS 2000, UTM zone 23 South), "
                       "methods, known limits, use terms, how to cite.",
        "parts": [
            {"name": "sources_and_dates", "check": lambda pd_: _readme_heading_part("sources_and_dates", pd_, "## Sources and dates")},
            {"name": "coordinate_system", "check": lambda pd_: _readme_heading_part("coordinate_system", pd_, "## CRS")},
            {"name": "methods", "check": lambda pd_: _readme_heading_part("methods", pd_, "## Methods")},
            {"name": "known_limits", "check": lambda pd_: _readme_heading_part("known_limits", pd_, "## Known limits")},
            {"name": "use_terms", "check": lambda pd_: _readme_heading_part("use_terms", pd_, "## Use terms")},
            {"name": "how_to_cite", "check": lambda pd_: _readme_heading_part("how_to_cite", pd_, "## How to cite")},
        ],
    },
    {
        "id": "P-02",
        "title": "Route points",
        "requirement": "OM2 sampled every 1 m at pedestrian height, with stable point IDs. No segments "
                       "imposed, so Jingxue keeps control of segment length.",
        "parts": [
            {"name": "points_table_all_formats", "check": lambda pd_: _file_exists_part(
                "points_table_all_formats", pd_, ["OM2/points.gpkg", "OM2/points.parquet", "OM2/points.csv"],
            )},
            {"name": "stable_point_id_unique", "check": lambda pd_: _point_id_unique_part("stable_point_id_unique", pd_)},
            {"name": "no_segments_imposed_on_points", "check": lambda pd_: _no_segments_imposed_part("no_segments_imposed_on_points", pd_)},
        ],
    },
    {
        "id": "P-03",
        "title": "Aggregation script",
        "requirement": "Re-aggregates any variable to whatever segment length the team chooses, at "
                       "buffers of 5, 10, 20 and 50 m.",
        "parts": [
            {"name": "buffer_columns_5_10_20_50m", "check": lambda pd_: _buffer_columns_part("buffer_columns_5_10_20_50m", pd_)},
            {"name": "segment_script_shipped_in_package", "check": lambda pd_: _file_exists_part(
                "segment_script_shipped_in_package", pd_, ["OM2/aggregate_to_segments.py"],
            )},
        ],
    },
    {
        "id": "P-04",
        "title": "Form variables",
        "requirement": "Building height, plan density, street width, height-to-width ratio, orientation, "
                       "sky-view factor from both airborne and terrestrial data.",
        "parts": [
            {"name": "airborne_building_and_canyon", "check": lambda pd_: _columns_part(
                "airborne_building_and_canyon", pd_,
                ["building_height_m", "plan_density_lambda_p", "street_width_m", "height_width_ratio"],
            )},
            {"name": "airborne_orientation", "check": lambda pd_: _columns_part("airborne_orientation", pd_, ["street_orientation_deg"])},
            {"name": "airborne_sky_view_factor", "check": lambda pd_: _columns_part("airborne_sky_view_factor", pd_, ["sky_view_factor"])},
            {"name": "terrestrial_sky_view_factor", "check": lambda pd_: _descoped_part(
                "terrestrial_sky_view_factor",
                "terrestrial sky-view factor is not part of v0.1.3 (airborne SVF only); no terrestrial column shipped",
            )},
        ],
    },
    {
        "id": "P-05",
        "title": "Shade lookup",
        "requirement": "Shaded or sunlit per point per 5 minutes for each campaign date, split into "
                       "building and tree shade, plus a short join example using Octopus timestamps.",
        "parts": [
            {"name": "building_shade_table", "check": lambda pd_: _building_shade_table_part("building_shade_table", pd_)},
            {"name": "tree_shade_column_reserved", "check": lambda pd_: _tree_shade_part("tree_shade_column_reserved", pd_)},
            {"name": "join_example_script_shipped_in_package", "check": lambda pd_: _file_exists_part(
                "join_example_script_shipped_in_package", pd_, ["OM2/join_shade_example.py"],
            )},
            {"name": "campaign_timestamp_confirmed", "check": lambda pd_: _pending_part(
                "campaign_timestamp_confirmed", pd_, "timestamp", ["OCTOPUS_CSV", "OCTOPUS_TZ"],
            )},
        ],
    },
    {
        "id": "P-06",
        "title": "Ventilation proxies",
        "requirement": "Orientation to prevailing wind, frontal area, openness, distance to open space. "
                       "Labelled as proxies.",
        "parts": [
            {"name": "wind_alignment_and_frontal_area_proxies", "check": lambda pd_: _columns_part(
                "wind_alignment_and_frontal_area_proxies", pd_,
                ["ventilation_wind_alignment_proxy", "ventilation_frontal_area_proxy"],
            )},
            {"name": "openness_and_distance_proxies", "check": lambda pd_: _columns_part(
                "openness_and_distance_proxies", pd_,
                ["ventilation_openness_proxy", "ventilation_dist_open_space_proxy_m"],
            )},
            {"name": "labelled_proxy_in_dictionary", "check": lambda pd_: _ventilation_labelled_proxy_part("labelled_proxy_in_dictionary", pd_)},
        ],
    },
    {
        "id": "P-07",
        "title": "Quality report",
        "requirement": "Airborne vs terrestrial differences on OM2 (height change 2024 to 2026, sky-view "
                       "factor error), coverage mask, known gaps.",
        "parts": [
            {"name": "coverage_mask", "check": lambda pd_: _coverage_mask_part("coverage_mask", pd_)},
            {"name": "known_gaps_listed", "check": lambda pd_: _known_gaps_part("known_gaps_listed", pd_)},
            {"name": "height_change_2024_2026", "check": lambda pd_: _descoped_part(
                "height_change_2024_2026", "2024 to 2026 height change is not part of v0.1.3",
            )},
            {"name": "airborne_vs_terrestrial_comparison", "check": lambda pd_: _descoped_part(
                "airborne_vs_terrestrial_comparison", "airborne vs terrestrial comparison is not part of v0.1.3",
            )},
        ],
    },
    {
        "id": "P-08",
        "title": "Data dictionary",
        "requirement": "One row per variable with ID, definition, unit, source, method, limits. IDs never "
                       "reused.",
        "parts": [
            {"name": "dictionary_file_with_required_columns", "check": lambda pd_: _dictionary_required_columns_part(
                "dictionary_file_with_required_columns", pd_,
            )},
            {"name": "ids_never_reused", "check": lambda pd_: _ids_never_reused_part("ids_never_reused", pd_)},
        ],
    },
    {
        "id": "P-09",
        "title": "Changelog",
        "requirement": "Versioned and dated.",
        "parts": [
            {"name": "changelog_has_dated_entry_for_version", "check": lambda pd_: _changelog_entry_part(
                "changelog_has_dated_entry_for_version", pd_,
            )},
        ],
    },
]


def _item_status(part_statuses: list[str]) -> str:
    """Item rule. Descoped parts are set aside first; the rest decide as
    before (all delivered / all pending / mixed). An item whose remaining
    parts are all delivered but that has >=1 descoped part is
    "delivered (scoped)": the qualifier keeps the cut visible in the status
    string itself (a plain "delivered" would hide it, "partial" would call
    a decided cut a gap), and the descoped parts and their decision id sit
    in the same row's evidence. Any still-pending part keeps it "partial"
    (P-05: the campaign timezone is genuinely open). An item with nothing
    but descoped parts is "descoped"."""
    live = [s for s in part_statuses if s != "descoped"]
    scoped = len(live) < len(part_statuses)
    if not live:
        return "descoped"
    if all(s == "delivered" for s in live):
        return "delivered (scoped)" if scoped else "delivered"
    if all(s == "pending" for s in live) and not scoped:
        return "pending"
    return "partial"


def conformance(package_dir: Path | str) -> dict:
    """Run every SPEC item's parts against package_dir. Never called at
    import time — always against a real, already-built package directory,
    so every number in the result is read off a file that exists."""
    package_dir = Path(package_dir)
    items_out = []
    for item in SPEC:
        parts_out = [part["check"](package_dir).to_dict() for part in item["parts"]]
        status = _item_status([p["status"] for p in parts_out])
        evidence = "; ".join(p["evidence"] for p in parts_out if p["evidence"])
        pending_on = sorted({t for p in parts_out for t in p["pending_on"]})
        reason = "; ".join(p["reason"] for p in parts_out if p["reason"])
        items_out.append({
            "id": item["id"],
            "title": item["title"],
            "requirement": item["requirement"],
            "status": status,
            "evidence": evidence,
            "pending_on": pending_on,
            "reason": reason,
            "decisions": sorted({p["decision"] for p in parts_out if p["decision"]}),
            "parts": parts_out,
        })
    return {"version": _current_version(package_dir), "items": items_out}


def conformance_rows(conf: dict) -> list[dict]:
    """Flat one-row-per-item view for p00_spec_conformance.csv."""
    return [
        {
            "id": it["id"],
            "title": it["title"],
            "requirement": it["requirement"],
            "status": it["status"],
            "evidence": it["evidence"],
            "pending_on": ", ".join(it["pending_on"]),
            "descoped_by": ", ".join(it.get("decisions", [])),
            "reason": it["reason"],
        }
        for it in conf["items"]
    ]


def write_conformance(package_dir: Path | str, conf: dict | None = None) -> dict:
    """Write p00_spec_conformance.json + .csv at package_dir's root.
    Returns the conformance dict written."""
    package_dir = Path(package_dir)
    if conf is None:
        conf = conformance(package_dir)
    (package_dir / "p00_spec_conformance.json").write_text(json.dumps(conf, indent=2), encoding="utf-8")
    pd.DataFrame(conformance_rows(conf)).to_csv(package_dir / "p00_spec_conformance.csv", index=False)
    return conf


def _md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def pending_cell(item: dict) -> str:
    """The "pending on" cell: unblocking task ids, and, for a deliberate
    cut, "descoped - <decision id>" so it reads as a decision, not a gap."""
    bits = list(item["pending_on"])
    bits += [f"descoped \u2014 {d}" for d in item.get("decisions", [])]
    return ", ".join(bits) if bits else "-"


def render_conformance_markdown(conf: dict) -> str:
    """The README's "Conformance to the package spec" section — a table
    id | requirement | status | evidence | pending on, rendered straight
    from conformance()'s own result (never re-derived by hand)."""
    lines = [
        "## Conformance to the package spec (P-01…P-09)",
        "",
        f"Computed by `src/om_package/spec.py` against this build "
        f"({conf.get('version', '?')}); see `p00_spec_conformance.json`/`.csv` "
        "at the package root for the per-part detail.",
        "",
        "A part marked *descoped* is a deliberate cut by PI decision, not a gap. "
        "*delivered (scoped)* means every other part of the item is delivered.",
        "",
        "| id | requirement | status | evidence | pending on / descoped |",
        "|---|---|---|---|---|",
    ]
    for it in conf["items"]:
        pending_on = pending_cell(it)
        lines.append(
            f"| {it['id']} | {_md_cell(it['requirement'])} | **{it['status']}** | "
            f"{_md_cell(it['evidence'])} | {_md_cell(pending_on)} |"
        )
    lines.append("")
    return "\n".join(lines)
