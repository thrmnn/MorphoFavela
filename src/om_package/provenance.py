"""Octopus OM2 package provenance — reads the seven panel-blocker
decisions (om_use_terms, om_scope, om_shade_release, om_credit,
om_dates_tz, om_lidar, om_route_geometry) from brisaverse's own
resolved-decisions store at build time.

Audit fix (2026-09-27): the README/CHANGELOG used to cite these as
interview shorthand ("PI ruling 2026-09-24, Q6a") — a code an external
reader (the named Octopus team) cannot resolve. The decision ids and
their resolution text now travel in ``manifest.json``'s
``provenance.decisions``, and the prose cites the id instead of the
interview question code. Never invented: if brisaverse's tasks.json is
unreachable or missing an id, this raises rather than rendering a guess.
"""
from __future__ import annotations

import json
from pathlib import Path

#: brisaverse is a sibling repo under the same SCR workspace, not a
#: MorphoFavela input — same convention as scripts/build_pi_review_folder.py
#: and tests/test_om_package_spec.py's TASKS_JSON constant.
BRISAVERSE_TASKS_JSON = Path("/home/theo/SCL/SCR/brisaverse/shared/facts/tasks.json")

#: The seven Octopus package panel-blocker decisions (panel review
#: 2026-09-24), in the order the README/CHANGELOG reference them.
OM_DECISION_IDS = [
    "om_use_terms",
    "om_scope",
    "om_shade_release",
    "om_credit",
    "om_dates_tz",
    "om_lidar",
    "om_route_geometry",
]


def read_om_decisions(tasks_json: Path = BRISAVERSE_TASKS_JSON) -> list[dict]:
    """One dict per ``OM_DECISION_IDS`` entry: {id, question, resolution,
    resolved_utc}, in that order. Raises FileNotFoundError if brisaverse's
    tasks.json is unreachable at build time, and ValueError if any id is
    missing from its ``resolved_decisions`` — this provenance is read, not
    guessed."""
    data = json.loads(tasks_json.read_text(encoding="utf-8"))
    by_id = {d["id"]: d for d in data.get("resolved_decisions", []) if "id" in d}
    missing = [i for i in OM_DECISION_IDS if i not in by_id]
    if missing:
        raise ValueError(f"resolved_decisions in {tasks_json} is missing id(s): {missing}")
    return [
        {
            "id": i,
            "question": by_id[i].get("question"),
            "resolution": by_id[i].get("resolution"),
            "resolved_utc": by_id[i].get("resolved_utc"),
        }
        for i in OM_DECISION_IDS
    ]


def decision_date(decisions: list[dict], decision_id: str) -> str:
    """The YYYY-MM-DD date a named decision (from ``read_om_decisions``'s
    output) was resolved, read from its own ``resolved_utc`` — never a
    separately-typed date that could drift from the record it names."""
    for d in decisions:
        if d["id"] == decision_id and d.get("resolved_utc"):
            return str(d["resolved_utc"])[:10]
    raise ValueError(f"decision '{decision_id}' not found or has no resolved_utc in {decisions!r}")
