"""Single source of truth for the shipped layout of the OM2 data package.

Every writer (scripts/build_om_package.py, the report, the README, the page)
and reader (spec checks, report loader, shipped scripts' defaults) takes its
relative paths from here. Paths are relative to the package directory.
"""
from __future__ import annotations

from pathlib import Path

DATA_DIR = "data"
FIGURES_DIR = "figures"
SCRIPTS_DIR = "scripts"

#: logical key -> relative stem (no extension) of a table shipped in data/
TABLES: dict[str, str] = {
    "route_points": f"{DATA_DIR}/route_points",
    "walks": f"{DATA_DIR}/walks",
    "walk_points": f"{DATA_DIR}/walk_points",
    "building_shade": f"{DATA_DIR}/building_shade",
    "sun_dose": f"{DATA_DIR}/sun_dose",
    "sun_envelope": f"{DATA_DIR}/sun_envelope",
    "horizon_profiles": f"{DATA_DIR}/horizon_profiles",
    "wind_regimes": f"{DATA_DIR}/wind_regimes",
    "wind_regime_by_hour": f"{DATA_DIR}/wind_regime_by_hour",
    "data_dictionary": f"{DATA_DIR}/data_dictionary",
    "quality_report": f"{DATA_DIR}/quality_report",
}

#: logical key -> file name of a figure in figures/ (the number is the order in report.md)
FIG: dict[str, str] = {
    "route": "fig01_route.png",
    "form": "fig02_street_form.png",
    "shade_map": "fig03_sun_share_map.png",
    "shade_calendar": "fig04_sun_calendar.png",
    "sun_dose": "fig05_sun_dose.png",
    "wind": "fig06_wind.png",
    "vent_schematic": "fig07_ventilation_schematic.png",
    "shelter_maps": "fig08_shelter_maps.png",
    "vent_profiles": "fig09_ventilation_profiles.png",
    "flags": "fig10_flagged_points.png",
    "svf_sensor": "fig11_sensor_matched.png",
}

FIGURE_FACTS = f"{FIGURES_DIR}/figure_facts.json"

#: logical key -> relative path of a script shipped in scripts/
SCRIPTS: dict[str, str] = {
    "aggregate_to_segments": f"{SCRIPTS_DIR}/aggregate_to_segments.py",
    "join_shade_example": f"{SCRIPTS_DIR}/join_shade_example.py",
}

#: key of the point table, as written for internal routes (OM1/OM3/OM4)
INTERNAL_POINTS_STEM = "points"


def stem(key: str) -> str:
    return TABLES[key]


def table(key: str, ext: str) -> str:
    return f"{TABLES[key]}.{ext}"


def table_path(package_dir: Path, key: str, ext: str) -> Path:
    return Path(package_dir) / table(key, ext)


def fig_rel(key: str) -> str:
    return f"{FIGURES_DIR}/{FIG[key]}"


def fig_path(package_dir: Path, key: str) -> Path:
    return Path(package_dir) / fig_rel(key)


def script_path(package_dir: Path, key: str) -> Path:
    return Path(package_dir) / SCRIPTS[key]


def stem_to_key(stem_str: str) -> str:
    for k, v in TABLES.items():
        if v == stem_str:
            return k
    raise KeyError(stem_str)
