"""One join, one badge formatter — shared by every registry producer/consumer.

Charter phase C (O3, organization_charter.md §2 field-ownership table):
`release`, `guardian_verdict`, `paper_ref` are owned by brisaverse's
`gen_p1_artifacts.py` -> `shared/facts/p1_artifacts.json`; the registry only
ever COPIES them in, joined by `(run_of_record, filename)` with a
content-hash fallback. `build_results_registry.py` calls `join_p1_release()`
below to populate `results.json`'s `release`/`guardian_verdict`/`paper_ref`
fields at build time (this module, not that one, owns the join logic).

Corrective step 3 (docs/critic/incident_dashboard_loop_2026-09-25.md, root
cause 3 / charter phase D): every consumer that shows a caption or a badge
for a registry artifact — the review folder, the site pages, and
brisaverse's dossiers — must call `badge_text()` / `caption_text()` here,
never write its own formatting. Before this module existed,
`build_pi_review_folder.py` had its own private copy of the (run, filename)
join and the 4-word badge rule; that private copy is now this module's
`release_badge()`, imported back into build_pi_review_folder.py so there is
exactly one implementation, not two that can drift apart.

This file is mirrored byte-for-byte into brisaverse as
`shared/lib/registry_badge.py` (the two repos can't share a Python package,
so the badge/caption *formatters* — the part with no MorphoFavela-only
filesystem dependency — are kept identical by a cross-repo test in each
repo: MorphoFavela's `tests/test_registry_join.py` and brisaverse's
`shared/facts/tests/test_dossier_registry_badge.py`, both of which read the
other repo's copy by absolute path and assert the two texts match). Keep
`release_badge()`, `badge_text()` and `caption_text()` free of any
MorphoFavela-root path logic so the mirrored copy stays a straight copy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BRISAVERSE_ROOT = Path.home() / "SCL" / "SCR" / "brisaverse"


# --------------------------------------------------------------------------
# The (run_of_record, filename) / (filename, md5) join against brisaverse's
# shared/facts/p1_artifacts.json. Ported verbatim from
# build_pi_review_folder.py's private _load_p1_register / _register_index /
# _register_hash_index (2026-09-24) — moved here, not rewritten, so this
# pass changes WHO calls the join, never HOW it matches.
# --------------------------------------------------------------------------

def load_p1_register(brisaverse_root: Path = BRISAVERSE_ROOT) -> list[dict]:
    path = brisaverse_root / "shared" / "facts" / "p1_artifacts.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("artifacts", [])
    except (json.JSONDecodeError, OSError):
        return []


def register_index(register: list[dict]) -> dict[tuple[str, str], dict]:
    """(run_of_record, filename) -> register row. Filename alone collides
    across runs (every WP-07 figure family reuses f1_/f2_/f3_/f4_): the run
    each figure actually came from is what disambiguates it."""
    idx: dict[tuple[str, str], dict] = {}
    for row in register:
        run = row.get("run_of_record")
        name = Path(row.get("image_url") or "").name
        if run and name:
            idx[(run, name)] = row
    return idx


def register_hash_index(register: list[dict], runs_root: Path) -> dict[tuple[str, str], dict]:
    """(filename, md5) -> register row, resolved against the register's own
    run_of_record on THIS disk. A fallback for the case the primary
    (run, filename) key misses because the caller picked a
    differently-timestamped run of the same family — content hash still
    proves it is the figure the register describes, not a same-named one."""
    idx: dict[tuple[str, str], dict] = {}
    for row in register:
        run = row.get("run_of_record")
        name = Path(row.get("image_url") or "").name
        if not (run and name):
            continue
        run_dir = runs_root / run
        if not run_dir.is_dir():
            continue
        for src in (run_dir / name, *run_dir.rglob(name)):
            if src.exists():
                break
        else:
            continue
        try:
            h = hashlib.md5(src.read_bytes()).hexdigest()
        except OSError:
            continue
        idx[(name, h)] = row
    return idx


def release_badge(row: dict | None) -> str:
    """withheld / staged / publishable / unclassified — the only four badges
    the ruling allows (organization_charter.md §2). `state` decides first
    (staged is a state, not a class); `release_class` text decides the
    rest. `row` is a raw p1_artifacts.json row, or a registry node that has
    already been through `join_p1_release()` below (both carry `state` and
    `release_class` under those same names by construction)."""
    if row is None:
        return "unclassified"
    if row.get("state") == "staged":
        return "staged"
    rc = (row.get("release_class") or "").lower()
    if row.get("state") == "withheld" or "withheld" in rc:
        return "withheld"
    if "publishable" in rc:
        return "publishable"
    return "unclassified"


def badge_text(node: dict | None) -> str:
    """The one badge string every consumer renders for a registry figure
    node: release class + lifecycle (corrective step 3's literal wording).
    `node["release"]` is the already-joined 4-word value written by
    `join_p1_release()`; this function never re-derives it from a raw
    p1_artifacts row — a consumer with only a raw row (not yet joined into
    the registry) calls `release_badge()` above instead, same as
    `join_p1_release()` itself does."""
    if not node:
        return "unclassified"
    release = node.get("release") or "unclassified"
    lifecycle = node.get("lifecycle") or "current"
    if lifecycle in (None, "current"):
        return release
    return f"{release} · {lifecycle}"


def node_by_path(registry: dict) -> dict[str, dict]:
    """`path` (repo-root-relative, e.g. "runs/<run>/<file>.png") -> figure
    node. This is the join key every consumer already has on hand without
    hashing anything: the review folder's own `entries[i]["source"]`, a
    site page's registry node (already keyed this way), and a dossier item
    reconstructible as `f"runs/{run_of_record}/{filename}"`. Preferred over
    a fresh content-hash for THIS join (the p1-release join above still
    uses hash as its fallback, because p1_artifacts.json rows carry no
    path) since every consumer here already knows the artefact's path."""
    return {n["path"]: n for n in registry.get("nodes", {}).values()
            if n.get("kind") == "figure" and n.get("path")}


def caption_text(node: dict | None) -> str:
    """The one identifying line every consumer renders for a registry
    figure node: family, site, wp — exactly the fields corrective step 3
    names ("caption ... site and family"), never a prose caption (that
    stays each surface's own `caption_draft`/blurb field, unrelated to this
    fix)."""
    if not node:
        return "unclassified"
    bits = [node.get("family") or "?"]
    if node.get("site"):
        bits.append(str(node["site"]))
    if node.get("wp"):
        bits.append(str(node["wp"]))
    return " · ".join(bits)


# --------------------------------------------------------------------------
# Charter phase C (O3): populate a freshly-built registry's
# release/guardian_verdict/paper_ref fields from the p1_artifacts.json join.
# Called once, from build_results_registry.py's build(), never re-run per
# consumer (that would be N re-derivations again, the exact defect this
# pass closes).
# --------------------------------------------------------------------------

def join_p1_release(nodes: dict, runs_root: Path, brisaverse_root: Path = BRISAVERSE_ROOT) -> int:
    """Mutates every `kind: figure` node in `nodes` in place, setting
    `release` (the 4-word badge), `guardian_verdict`, and `paper_ref`.
    Returns the number of nodes matched. A node with no matching register
    row keeps `release: None` (rendered as "unclassified" by `badge_text`,
    per organization_charter.md §2: unclassified is a real state, never
    hidden), never null-vs-unclassified ambiguity for a caller."""
    register = load_p1_register(brisaverse_root)
    by_run_file = register_index(register)
    by_hash = register_hash_index(register, runs_root)
    hash_names = {name for (name, _h) in by_hash}
    matched = 0
    for node in nodes.values():
        if node.get("kind") != "figure":
            continue
        path = node.get("path")
        if not path:
            continue
        name = Path(path).name
        parent = node.get("parent") or ""
        run_id = parent.removeprefix("run:") if parent.startswith("run:") else None
        row = by_run_file.get((run_id, name)) if run_id else None
        if row is None and name in hash_names:
            src = runs_root.parent / path
            if src.exists():
                try:
                    h = hashlib.md5(src.read_bytes()).hexdigest()
                except OSError:
                    h = None
                if h is not None:
                    row = by_hash.get((name, h))
        if row is None:
            continue
        node["release"] = release_badge(row)
        node["guardian_verdict"] = (row.get("guardian") or {}).get("verdict")
        node["paper_ref"] = row.get("id") if row.get("cited_in_outline") else None
        matched += 1
    return matched
