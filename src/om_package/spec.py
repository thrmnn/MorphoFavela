"""P-00 — the PI's package spec (P-01..P-09, verbatim, 2026-09-23; P-10/P-11
added 2026-10-01 for v0.2.0 and reworded for v0.3.0, P-12 added for v0.3.0,
worded here, not verbatim) encoded
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
to ``p00_spec_conformance.json``/``.csv`` in the INTERNAL directory beside the
package (``internal_dir_for``), not in the shipped folder, and renders it into
the README's "Conformance to the package spec" section
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


def _shade_df(package_dir: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    """The shade table is parquet only (millions of rows); read only the
    columns a check needs."""
    pq = package_dir / "p05_building_shade.parquet"
    return pd.read_parquet(pq, columns=columns) if pq.exists() else None


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


def internal_dir_for(package_dir: Path | str) -> Path:
    """Where the files that must not ship (CHANGELOG.md, the conformance
    table, the disclosure hits) are written:
    <packages>/_internal/<mare_om2>/<version>/ beside the shipped folder."""
    package_dir = Path(package_dir)
    return package_dir.parent.parent / "_internal" / package_dir.parent.name / package_dir.name


def _changelog_text(package_dir: Path) -> str:
    p = internal_dir_for(package_dir) / "CHANGELOG.md"
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


#: PI decision (resolved 2026-10-01) that dropped the terrestrial-LiDAR
#: analysis and tree shade from v0.1.3 (still out of scope in v0.2.0). Both
#: stay candidates for a later version (OMPKG2); this is a cut of this
#: version, not a cancellation.
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
        return PartResult(name, "pending", evidence="CHANGELOG.md not found in the internal directory")
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
    df = _shade_df(package_dir, ["timestamp_local", "date"]) if (package_dir / "p05_building_shade.parquet").exists() else None
    if df is None:
        return PartResult(name, "pending", evidence="p05_building_shade table not found")
    import pyarrow.parquet as pq

    have = set(pq.read_schema(package_dir / "p05_building_shade.parquet").names)
    required = {"point_id", "timestamp_local", "timestamp_utc", "date", "sun_altitude_deg", "sun_azimuth_deg", "shaded"}
    missing = required - have
    if missing:
        return PartResult(name, "pending", evidence=f"p05_building_shade missing column(s): {sorted(missing)}")
    if len(df) == 0:
        return PartResult(
            name, "pending",
            evidence="p05_building_shade has the correct schema but 0 rows",
        )
    local = pd.DatetimeIndex(df["timestamp_local"])
    offsets = {str(o) for o in (local.tz_localize(None) - local.tz_convert("UTC").tz_localize(None)).unique()}
    return PartResult(name, "delivered", evidence=f"p05_building_shade: {len(df)} rows over {df['date'].nunique()} walk dates, "
                                                  f"local-time offset(s) {sorted(offsets)}, schema {sorted(required)}")


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
    descoped_items = q.get("descoped_items")
    if pending_items is None or descoped_items is None:
        return PartResult(name, "pending", evidence="p07_quality_report.json lacks pending_items/descoped_items")
    if not pending_items and not descoped_items:
        return PartResult(name, "pending", evidence="p07_quality_report.json lists no known gaps")
    return PartResult(
        name, "delivered",
        evidence=f"pending_items: {pending_items}; descoped_items ({q.get('descoped_by')}): {descoped_items}",
    )


# ------------------------------------------------- P-10 / P-11 helpers --

P10_COLUMNS = {
    "p10_sun_envelope": ["point_id", "local_slot", "class", "sunlit_day_share", "n_days_sun_up"],
    "p10_sun_dose": ["point_id", "scope", "local_slot", "dose_1h_wh_m2", "dose_2h_wh_m2", "dose_3h_wh_m2"],
    "p10_horizon_profiles": ["point_id", "azimuth_deg", "horizon_deg"],
}
P10_CLASSES = {"always_sunlit", "always_shaded", "date_dependent", "night"}
P11_REGIME_COLUMNS = ["period", "regime_key", "name", "column_slug", "mean_direction_deg", "share", "mean_speed_ms",
                      "n_reports", "mixture_component_direction_deg", "mixture_difference_deg"]
P11_HOUR_COLUMNS = ["period", "local_hour", "regime", "regime_key", "share"]
#: point-column stems evaluated per campaign wind regime (the slug is appended)
P11_REGIME_STEMS = {
    "windward_lambda_f": "frontal_area_density_windward",
    "canyon_alignment": "canyon_alignment_deg",
    "upwind_shelter": "upwind_shelter_angle_deg",
    "roughness_z0": "z0_macdonald_m",
}
P11_STATIC_COLUMNS = {"roughness_zd": "zd_macdonald_m", "open_space_fraction": "open_space_fraction"}
P12_WALK_COLUMNS = ["walk_id", "date", "period", "start_local", "start_utc", "end_local", "end_utc", "duration_min",
                    "coverage_share", "share_on_route", "share_interpolated", "max_gap_s", "partial", "wind_regime"]
P12_POINT_COLUMNS = ["walk_id", "point_id", "distance_along_m", "t_arrival_local", "t_arrival_utc", "arrival_source",
                     "shaded_at_arrival", "dose_1h_before_wh_m2", "dose_3h_before_wh_m2"]
P12_TAUS_S = (5, 10, 30, 60)
P12_BASE_MEASURES = ["sky_view_factor", "height_width_ratio", "building_height_m", "plan_density_lambda_p",
                     "shaded_at_arrival", "dose_1h_before_wh_m2"]
P12_REGIME_MEASURE_STEMS = ["frontal_area_density_windward", "canyon_alignment_deg", "upwind_shelter_angle_deg"]
_ISO_LOCAL = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}-03:00$"
P11_PROXY_STATIC_IDS = ["zd_macdonald_m", "open_space_fraction"]


def _read_table(package_dir: Path, stem: str) -> pd.DataFrame | None:
    pq = package_dir / f"{stem}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = package_dir / f"{stem}.csv"
    return pd.read_csv(csv) if csv.exists() else None


def _table_with_columns(name: str, package_dir: Path, stem: str, cols: list[str]):
    """(df, None) or (None, PartResult pending) for a shipped package-root table."""
    df = _read_table(package_dir, stem)
    if df is None:
        return None, PartResult(name, "pending", evidence=f"{stem}.parquet/.csv not found")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None, PartResult(name, "pending", evidence=f"{stem} missing column(s): {missing}")
    if len(df) == 0:
        return None, PartResult(name, "pending", evidence=f"{stem} has the schema but 0 rows")
    return df, None


def _both_formats_part(name: str, package_dir: Path, stems: list[str]) -> PartResult | None:
    missing = [f"{s}.{ext}" for s in stems for ext in ("parquet", "csv") if not (package_dir / f"{s}.{ext}").exists()]
    return PartResult(name, "pending", evidence=f"missing file(s): {', '.join(missing)}") if missing else None


def _sun_envelope_part(name: str, package_dir: Path) -> PartResult:
    bad = _both_formats_part(name, package_dir, ["p10_sun_envelope"])
    if bad:
        return bad
    if not (package_dir / "p10_horizon_profiles.parquet").exists():
        return PartResult(name, "pending", evidence="missing file(s): p10_horizon_profiles.parquet")
    env, bad = _table_with_columns(name, package_dir, "p10_sun_envelope", P10_COLUMNS["p10_sun_envelope"])
    if bad:
        return bad
    hor, bad = _table_with_columns(name, package_dir, "p10_horizon_profiles", P10_COLUMNS["p10_horizon_profiles"])
    if bad:
        return bad
    unknown = sorted(set(env["class"].unique()) - P10_CLASSES)
    if unknown:
        return PartResult(name, "pending", evidence=f"p10_sun_envelope has unknown class value(s): {unknown}")
    pts = _points_df(package_dir)
    if pts is None:
        return PartResult(name, "pending", evidence="OM2/points table not found")
    ids = set(pts["point_id"])
    if set(env["point_id"]) != ids or set(hor["point_id"]) != ids:
        return PartResult(name, "pending", evidence="p10_sun_envelope / p10_horizon_profiles do not cover exactly the OM2 point_ids")
    day = env[env["class"] != "night"]
    share = float((day["class"] == "date_dependent").mean()) if len(day) else float("nan")
    return PartResult(
        name, "delivered",
        evidence=f"p10_sun_envelope: {len(env)} rows over {len(ids)} points, {env['local_slot'].nunique()} local slots; "
                 f"date_dependent share of daylight point-slots {share:.3f}; p10_horizon_profiles: {len(hor)} rows",
    )


def _sun_dose_part(name: str, package_dir: Path) -> PartResult:
    if not (package_dir / "p10_sun_dose.parquet").exists():
        return PartResult(name, "pending", evidence="missing file(s): p10_sun_dose.parquet")
    dose, bad = _table_with_columns(name, package_dir, "p10_sun_dose", P10_COLUMNS["p10_sun_dose"])
    if bad:
        return bad
    scopes = set(dose["scope"].astype(str).unique())
    env_scopes = {"envelope_min", "envelope_median", "envelope_max"}
    dates = sorted(scopes - env_scopes)
    if not env_scopes <= scopes or not dates:
        return PartResult(name, "pending", evidence=f"p10_sun_dose scopes {sorted(scopes)} lack the envelope statistics or a campaign date")
    dcols = [c for c in P10_COLUMNS["p10_sun_dose"] if c.startswith("dose_")]
    if (dose[dcols] < 0).any().any():
        return PartResult(name, "pending", evidence="p10_sun_dose has negative dose values")
    return PartResult(
        name, "delivered",
        evidence=f"p10_sun_dose: {len(dose)} rows; scopes = {len(dates)} campaign date(s) + {sorted(env_scopes)}; columns {dcols}",
    )


def regime_slugs(package_dir: Path) -> list[str]:
    """Slugs of the campaign-season regimes, read off p11_wind_regimes.csv."""
    p = package_dir / "p11_wind_regimes.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    return df.loc[df["period"] == "campaign", "column_slug"].tolist()


def _wind_regimes_part(name: str, package_dir: Path) -> PartResult:
    missing = [f for f in ("p11_wind_regimes.csv", "p11_regime_by_hour.csv") if not (package_dir / f).exists()]
    if missing:
        return PartResult(name, "pending", evidence=f"missing file(s): {', '.join(missing)}")
    reg = pd.read_csv(package_dir / "p11_wind_regimes.csv")
    hour = pd.read_csv(package_dir / "p11_regime_by_hour.csv")
    for tab, cols, stem in ((reg, P11_REGIME_COLUMNS, "p11_wind_regimes"), (hour, P11_HOUR_COLUMNS, "p11_regime_by_hour")):
        gone = [c for c in cols if c not in tab.columns]
        if gone:
            return PartResult(name, "pending", evidence=f"{stem} missing column(s): {gone}")
    if set(reg["period"]) != {"campaign", "climatology"} or reg.groupby("period").size().ne(2).any():
        return PartResult(name, "pending", evidence="p11_wind_regimes needs two regimes for each of campaign and climatology")
    mp = package_dir / "manifest.json"
    src = (json.loads(mp.read_text(encoding="utf-8")).get("provenance", {}).get("wind_source") if mp.exists() else None) or {}
    if not src.get("sha256") or not src.get("url"):
        return PartResult(name, "pending", evidence="manifest.json provenance.wind_source lacks the source url/sha256")
    camp = reg[reg["period"] == "campaign"]
    desc = ", ".join(f"{r['name']} {r['mean_direction_deg']:.0f} deg ({r['share']:.2f})" for _, r in camp.iterrows())
    return PartResult(
        name, "delivered",
        evidence=f"p11_wind_regimes: campaign regimes {desc}; p11_regime_by_hour: {len(hour)} rows; "
                 f"source sha256 in manifest.json provenance.wind_source ({src.get('station')}, 10 m)",
    )


def _regime_columns_part(name: str, package_dir: Path, key: str) -> PartResult:
    slugs = regime_slugs(package_dir)
    if not slugs:
        return PartResult(name, "pending", evidence="p11_wind_regimes.csv not found or has no campaign regimes")
    return _columns_part(name, package_dir, [f"{P11_REGIME_STEMS[key]}_{sl}" for sl in slugs])


def _p11_labelled_proxy_part(name: str, package_dir: Path) -> PartResult:
    df = _dictionary_df(package_dir)
    if df is None:
        return PartResult(name, "pending", evidence="p08_data_dictionary.csv not found")
    ids = [*P11_PROXY_STATIC_IDS, *(f"{stem}_{sl}" for sl in regime_slugs(package_dir) for stem in P11_REGIME_STEMS.values())]
    missing, unlabelled = [], []
    for vid in ids:
        hit = df[df["id"] == vid]
        if hit.empty:
            missing.append(vid)
        elif "PROXY" not in " ".join(str(hit.iloc[0].get(c, "")) for c in ("definition", "limits")).upper():
            unlabelled.append(vid)
    if missing:
        return PartResult(name, "pending", evidence=f"p08_data_dictionary.csv missing row(s): {missing}")
    if unlabelled:
        return PartResult(name, "pending", evidence=f"dictionary row(s) not labelled PROXY: {unlabelled}")
    return PartResult(name, "delivered", evidence=f"all {len(ids)} P-11 index dictionary rows contain 'PROXY'")


# ------------------------------------------------------------- P-12 --

def _walks_part(name: str, package_dir: Path) -> PartResult:
    bad = _both_formats_part(name, package_dir, ["p02b_walks"])
    if bad:
        return bad
    df, bad = _table_with_columns(name, package_dir, "p02b_walks", P12_WALK_COLUMNS)
    if bad:
        return bad
    if not df["walk_id"].is_unique:
        return PartResult(name, "pending", evidence="p02b_walks has duplicate walk_id values")
    for c in ("start_local", "end_local"):
        if not df[c].astype(str).str.match(_ISO_LOCAL).all():
            return PartResult(name, "pending", evidence=f"p02b_walks.{c} is not Rio local ISO time with a -03:00 offset")
    tagged = df["wind_regime"].value_counts().to_dict()
    return PartResult(
        name, "delivered",
        evidence=f"p02b_walks: {len(df)} walks on {df['date'].nunique()} dates, {int(df['partial'].sum())} partial; "
                 f"wind regime tags {tagged}",
    )


def _walk_points_part(name: str, package_dir: Path) -> PartResult:
    bad = _both_formats_part(name, package_dir, ["p12_walk_points"])
    if bad:
        return bad
    df, bad = _table_with_columns(name, package_dir, "p12_walk_points", P12_POINT_COLUMNS)
    if bad:
        return bad
    if (df["arrival_source"] == "outside_walk").any():
        return PartResult(name, "pending", evidence="p12_walk_points still holds outside_walk rows")
    if df.duplicated(["walk_id", "point_id"]).any():
        return PartResult(name, "pending", evidence="p12_walk_points has duplicate (walk_id, point_id) rows")
    if not df["t_arrival_local"].astype(str).str.match(_ISO_LOCAL).all():
        return PartResult(name, "pending", evidence="p12_walk_points.t_arrival_local is not Rio local ISO time with a -03:00 offset")
    walks = _read_table(package_dir, "p02b_walks")
    if walks is not None and not set(df["walk_id"]) <= set(walks["walk_id"]):
        return PartResult(name, "pending", evidence="p12_walk_points holds walk_id values absent from p02b_walks")
    return PartResult(
        name, "delivered",
        evidence=f"p12_walk_points: {len(df)} rows over {df['walk_id'].nunique()} walks and {df['point_id'].nunique()} points",
    )


def _sensor_matched_part(name: str, package_dir: Path) -> PartResult:
    slugs = regime_slugs(package_dir)
    measures = [*P12_BASE_MEASURES, *(f"{st}_{sl}" for sl in slugs for st in P12_REGIME_MEASURE_STEMS)]
    cols = [f"{m}_tau{t}s" for m in measures for t in P12_TAUS_S]
    df = _read_table(package_dir, "p12_walk_points")
    if df is None:
        return PartResult(name, "pending", evidence="p12_walk_points not found")
    missing = [c for c in cols if c not in df.columns]
    if missing or not slugs:
        return PartResult(name, "pending", evidence=f"p12_walk_points missing sensor-matched column(s): {missing[:6]}")
    return PartResult(
        name, "delivered",
        evidence=f"p12_walk_points has {len(cols)} sensor-matched columns ({len(measures)} measures x tau {list(P12_TAUS_S)} s)",
    )


def _walk_script_part(name: str, package_dir: Path) -> PartResult:
    p = package_dir / "OM2" / "aggregate_to_segments.py"
    if not p.exists():
        return PartResult(name, "pending", evidence="missing file(s): OM2/aggregate_to_segments.py")
    text = p.read_text(encoding="utf-8")
    if "--by" not in text or "--tau" not in text:
        return PartResult(name, "pending", evidence="OM2/aggregate_to_segments.py lacks --by or --tau")
    return PartResult(name, "delivered", evidence="OM2/aggregate_to_segments.py accepts --by (e.g. walk_id) and --tau")


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
        ],
    },
    {
        "id": "P-05",
        "title": "Shade lookup",
        "requirement": "Shaded or sunlit per point per 5 minutes for each walk date (daylight, Rio local time, "
                       "with a UTC twin), plus a short join example using Octopus timestamps.",
        "parts": [
            {"name": "building_shade_table", "check": lambda pd_: _building_shade_table_part("building_shade_table", pd_)},
            {"name": "join_example_script_shipped_in_package", "check": lambda pd_: _file_exists_part(
                "join_example_script_shipped_in_package", pd_, ["OM2/join_shade_example.py"],
            )},
        ],
    },
    {
        "id": "P-06",
        "title": "Ventilation proxies",
        "requirement": "Orientation to prevailing wind, frontal area, openness, distance to open space. "
                       "Labelled as proxies.",
        "parts": [
            {"name": "frontal_area_proxy", "check": lambda pd_: _columns_part(
                "frontal_area_proxy", pd_, ["ventilation_frontal_area_proxy"],
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
    {
        "id": "P-10",
        "title": "Sun exposure over the campaign season",
        "requirement": "Added 2026-10-01, reworded for v0.3.0. Per point and Rio local time of day, whether the point is "
                       "always sunlit, always shaded or date-dependent over the campaign season; direct-sun dose for 1, 2 and 3 h "
                       "windows (walk dates and the season envelope); annual sun hours. Geometry-derived proxies, not measured sunlight.",
        "parts": [
            {"name": "sun_envelope", "check": lambda pd_: _sun_envelope_part("sun_envelope", pd_)},
            {"name": "sun_dose", "check": lambda pd_: _sun_dose_part("sun_dose", pd_)},
            {"name": "annual_sun_hours", "check": lambda pd_: _columns_part("annual_sun_hours", pd_, ["annual_sun_hours"])},
        ],
    },
    {
        "id": "P-11",
        "title": "Two wind regimes and ventilation indices",
        "requirement": "Reworded for v0.3.0. The airport (SBGL, 10 m) wind of the campaign season and of the 2015-2024 climatology "
                       "split into two regimes, with their share by local hour; and ventilation indices derived from building geometry "
                       "at each campaign regime's mean direction: windward frontal-area density, canyon-wind alignment, upwind shelter "
                       "angle, Macdonald z0 (per regime), zd and open-space fraction. All are PROXIES (geometry), none is measured or "
                       "simulated air temperature or air movement; SBGL is not wind at the route.",
        "parts": [
            {"name": "wind_regimes", "check": lambda pd_: _wind_regimes_part("wind_regimes", pd_)},
            {"name": "windward_lambda_f", "check": lambda pd_: _regime_columns_part("windward_lambda_f", pd_, "windward_lambda_f")},
            {"name": "canyon_alignment", "check": lambda pd_: _regime_columns_part("canyon_alignment", pd_, "canyon_alignment")},
            {"name": "upwind_shelter", "check": lambda pd_: _regime_columns_part("upwind_shelter", pd_, "upwind_shelter")},
            {"name": "roughness_z0_zd", "check": lambda pd_: _columns_part(
                "roughness_z0_zd", pd_, [P11_STATIC_COLUMNS["roughness_zd"]] + [f"{P11_REGIME_STEMS['roughness_z0']}_{sl}" for sl in regime_slugs(pd_)])},
            {"name": "open_space_fraction", "check": lambda pd_: _columns_part("open_space_fraction", pd_, [P11_STATIC_COLUMNS["open_space_fraction"]])},
            {"name": "labelled_proxy_in_dictionary", "check": lambda pd_: _p11_labelled_proxy_part("labelled_proxy_in_dictionary", pd_)},
        ],
    },
    {
        "id": "P-12",
        "title": "Walks, arrival times and sensor-matched values",
        "requirement": "Added for v0.3.0. One row per logger walk (timing, coverage, wind regime tag); one row per walk and route point "
                       "(arrival time, shaded at arrival, clear-sky direct dose in the 1 h and 3 h before arrival) with sensor-matched "
                       "values of the form, shade, dose and ventilation measures at four sensor time constants; a segment script that "
                       "aggregates per walk.",
        "parts": [
            {"name": "walks_table", "check": lambda pd_: _walks_part("walks_table", pd_)},
            {"name": "walk_points_table", "check": lambda pd_: _walk_points_part("walk_points_table", pd_)},
            {"name": "sensor_matched_columns", "check": lambda pd_: _sensor_matched_part("sensor_matched_columns", pd_)},
            {"name": "segment_script_by_walk", "check": lambda pd_: _walk_script_part("segment_script_by_walk", pd_)},
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


def write_conformance(package_dir: Path | str, conf: dict | None = None, out_dir: Path | str | None = None) -> dict:
    """Write p00_spec_conformance.json + .csv into ``out_dir`` (default: the
    package's internal directory, never the shipped folder). Returns the
    conformance dict written."""
    package_dir = Path(package_dir)
    out_dir = Path(out_dir) if out_dir else internal_dir_for(package_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if conf is None:
        conf = conformance(package_dir)
    (out_dir / "p00_spec_conformance.json").write_text(json.dumps(conf, indent=2), encoding="utf-8")
    pd.DataFrame(conformance_rows(conf)).to_csv(out_dir / "p00_spec_conformance.csv", index=False)
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
        "## Conformance to the package spec (P-01…P-12)",
        "",
        f"Computed by `src/om_package/spec.py` against this build "
        f"({conf.get('version', '?')}), part by part.",
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
