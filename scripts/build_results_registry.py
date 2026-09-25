#!/usr/bin/env python3
"""Results registry — charter: docs/charter/organization_charter.md §2.

Single source of truth for MorphoFavela's results hierarchy (Output -> WP ->
Family -> Run -> Artifact), derived — never hand-registered — from metadata
that already exists on disk: `runs/*/figure_manifest.json`, the declared
`config/work_packages.yaml` (charter phase B / figure_organization_spec.md
§5), `config/sites.yaml`, and `shared/facts/tasks.json` in the sibling
brisaverse checkout for decision_ids.

Writes `outputs/_registry/results.json`. This phase (O2 / charter phase B,
"registry input") does NOT join `gen_p1_artifacts.py` -> `p1_artifacts.json`
(release/guardian_verdict/paper_ref stay null) — that join is charter phase C
(O3), which owns those three fields exclusively per the charter's field-
ownership table (§2). This generator only ever writes the fields that same
table assigns to it: `lifecycle`, `head_run`, `superseded_by`, `thumb_hash`,
`unclassified`, plus the pass-through fields whose authority is the run's own
writer (`path`, `content_hash`, `produced_utc`, `generator`, `derived_from`,
`site`) and `work_packages.yaml` (`wp`, `papers`, family membership).

ID scheme (per the O2 task ruling, correcting the round-1 prototype's bug —
see `check_registry.py`'s self-test and `tests/test_build_results_registry.py`
for the regression test): a run-backed family's artifact ids are run-scoped,
`art:<family>::<run_id>::<slug>`, so two runs of the same family that both
produce a figure with the same slug get two distinct, coexisting nodes —
never a silent last-write-wins collision (the round-1 defect: an
`art:<family>::<slug>` id with no run component meant a superseded run's
figure overwrote the current one's node in the registry dict, and the UI
could show a CURRENT badge on whichever run happened to be scanned last).
Every run-backed family ALSO gets a `art:<family>::current::<slug>` alias
node (`kind: "alias"`, `target: <the head run's real id>`) — the stable
handle any consumer (a dossier, a site page) should actually link to, always
re-pointed at whichever run is `head_run` this build. A family with no run
axis (`static_root` or `glob` in work_packages.yaml — charter §1: "Static
families ... have no L4") uses the plain two-part id `art:<family>::<slug>`,
since there is nothing to scope by.

Usage:
    python scripts/build_results_registry.py             # write outputs/_registry/results.json, print counts
    python scripts/build_results_registry.py -v           # + one line per family (placed count, source)

Zero third-party deps beyond PyYAML (already a repo dependency).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _main_checkout_root(start: Path) -> Path:
    """`runs/`/`outputs/` are gitignored data dirs that exist only in the
    main checkout, not a fresh `git worktree add`. Resolve them via
    `git rev-parse --git-common-dir`, which points at the shared `.git`
    regardless of which worktree `start` is, so this generator runs
    correctly from a worktree too."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(start), "rev-parse", "--git-common-dir"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        common_dir = Path(out)
        if not common_dir.is_absolute():
            common_dir = (start / common_dir).resolve()
        return common_dir.parent
    except Exception:
        return start


DATA_ROOT = _main_checkout_root(ROOT)
# runs/ and outputs/ are gitignored — they only physically exist in the main
# checkout, so those two resolve via DATA_ROOT. config/ and registry/ are
# tracked and travel with whichever branch/worktree this script itself is
# checked out on, so those resolve via ROOT — a worktree authoring a new
# config/work_packages.yaml must read its OWN copy, not main's.
RUNS = DATA_ROOT / "runs"
OUTPUTS = DATA_ROOT / "outputs"
CONFIG = ROOT / "config"
REGISTRY_OUT = OUTPUTS / "_registry"
BASELINE_PATH = ROOT / "registry" / "baseline.json"
WP_YAML = CONFIG / "work_packages.yaml"

BRISA = DATA_ROOT.parent / "brisaverse"
TASKS_JSON = BRISA / "shared" / "facts" / "tasks.json"

IMG_EXTS = (".png", ".svg")
RUN_SUFFIX_RE = re.compile(r"_(\d{8}T\d{6}Z)$")
# Dirs that MAY contain generated mirrors/aggregators of figures placed
# elsewhere under outputs/ (figure_organization_spec.md §1: "Hash matches
# under outputs/_hub/** and outputs/_review/** are copies") or
# internal/superseded package snapshots. The spec's exemption is scoped to
# actual hash matches, not the whole directory — "Anything else goes to
# unclassified." A file under one of these dirs is therefore only skipped
# from the unclassified walk when its content hash matches a real,
# already-registered artifact (see `_content_matches_a_placed_artifact`
# below); a file here with unique content (no registered original) is real,
# unaccounted-for content and must surface as unclassified like anything
# else — never silently dropped (organization_charter.md: "a figure with no
# register row is shown as unclassified, never hidden"). Real copy-merging
# (`copies[]`) is charter §6 / O8 (cleanup), out of scope for O2.
EXCLUDED_TOP_DIRS = {"_hub", "_review"}
EXCLUDED_SUBSTRINGS = ("/_packages/_internal/",)
# O8 cleanup (charter phase E / figure_organization_spec.md §6-§7): bytes
# moved here by the cleanup pass (never deleted — "Nothing leaves the PI's
# view. Archiving moves bytes, never rows"). Scanned separately from
# `unclassified` so a file the PI already dismissed doesn't keep inflating
# the "needs review" count every cycle; it renders in its own collapsed
# "Archived (N)" bucket instead (organization_charter.md §4 lifecycle table).
ARCHIVE_TOP_DIR = "_archive"


# --------------------------------------------------------------------------- helpers

def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _posix(path: Path, base: Path | None = None) -> str:
    # `base` defaults to the CURRENT `DATA_ROOT` global, read at call time —
    # a `base: Path = DATA_ROOT` default arg would freeze the value module
    # import captured, which breaks under `monkeypatch.setattr(brr,
    # "DATA_ROOT", ...)` in tests (default args bind once, at def time).
    base = base if base is not None else DATA_ROOT
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()
    except OSError:
        return None


def _run_utc_from_name(name: str) -> str | None:
    m = RUN_SUFFIX_RE.search(name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


def _family_of_rundir(run_dir_name: str) -> str:
    return RUN_SUFFIX_RE.sub("", run_dir_name)


def _is_excluded(rel_posix: str) -> bool:
    top = rel_posix.split("/", 2)
    if len(top) >= 2 and top[0] == "outputs" and top[1] in EXCLUDED_TOP_DIRS:
        return True
    full = "/" + rel_posix
    return any(s in full for s in EXCLUDED_SUBSTRINGS)


# --------------------------------------------------------------------------- config

def load_work_packages() -> dict:
    data = yaml.safe_load(WP_YAML.read_text())
    return data or {}


def load_sites() -> list[str]:
    data = yaml.safe_load((CONFIG / "sites.yaml").read_text())
    return sorted((data.get("sites") or {}).keys())


def load_tasks() -> dict:
    data = _read_json(TASKS_JSON)
    if not data:
        return {"open_decisions": [], "resolved_decisions": []}
    return {"open_decisions": data.get("open_decisions", []),
            "resolved_decisions": data.get("resolved_decisions", [])}


def _decision_hits(all_decisions: list[tuple[dict, str]], needle: str) -> list[str]:
    if not needle:
        return []
    hits = []
    for d, _kind in all_decisions:
        if needle in json.dumps(d, ensure_ascii=False):
            did = d.get("id")
            if did:
                hits.append(did)
    return hits


# --------------------------------------------------------------------------- run scan

class RunInfo:
    __slots__ = ("run_id", "family", "path", "run_utc", "figure_manifest", "manifest", "figures", "status")

    def __init__(self, run_id: str, path: Path):
        self.run_id = run_id
        self.path = path
        self.family = _family_of_rundir(run_id)
        self.figure_manifest = _read_json(path / "figure_manifest.json")
        self.manifest = _read_json(path / "manifest.json")
        self.run_utc = (
            _run_utc_from_name(run_id)
            or (self.figure_manifest or {}).get("_utc")
            or (self.manifest or {}).get("_utc")
        )
        self.figures = (self.figure_manifest or {}).get("figures", {}) if self.figure_manifest else {}
        self.status = (self.manifest or {}).get("status") if self.manifest else None


def _read_json_strict(path: Path):
    """Like `_read_json` but does NOT swallow a parse failure — raises
    `json.JSONDecodeError` on truncated/corrupt JSON so the caller (scan_runs,
    detecting an in-progress write) can tell "unparseable right now" apart
    from "not written yet"."""
    return json.loads(path.read_text())


def scan_runs() -> tuple[dict[str, RunInfo], list[dict]]:
    """Corrective plan step 4d (docs/critic/incident_dashboard_loop_2026-09-25.md,
    root cause 6, "the registry build failed the gate once while other
    agents were writing run directories"): a run directory a producer script
    has `mkdir`-ed but not finished writing — no manifest.json AND no
    figure_manifest.json yet, or one that is present but only half-flushed
    to disk (invalid JSON) — is a RACE, not a defect in the run itself. It
    is skipped and counted here rather than either crashing the whole
    registry build or silently being folded in as a legitimate empty/ok run
    (the pre-fix behaviour: an empty `figures` dict reads as `not r.figures`
    == True in `_build_run_backed_family`'s ok_runs filter, so a mid-write
    run could become `head_run` of its family with zero real figures).

    Returns (runs_by_id, skipped) where each `skipped` entry is
    {"run_id", "reason"} — surfaced in the registry's own output, never
    silently dropped."""
    if not RUNS.is_dir():
        return {}, []
    out: dict[str, RunInfo] = {}
    skipped: list[dict] = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        has_manifest = (d / "manifest.json").is_file()
        has_figure_manifest = (d / "figure_manifest.json").is_file()
        if not has_manifest and not has_figure_manifest:
            skipped.append({
                "run_id": d.name,
                "reason": "no manifest.json or figure_manifest.json — run directory "
                          "exists but nothing marks it complete yet (another agent may "
                          "still be writing it)",
            })
            continue
        try:
            if has_figure_manifest:
                _read_json_strict(d / "figure_manifest.json")
            if has_manifest:
                _read_json_strict(d / "manifest.json")
        except json.JSONDecodeError as exc:
            skipped.append({
                "run_id": d.name,
                "reason": f"unparseable manifest JSON ({exc}) — likely caught mid-write",
            })
            continue
        out[d.name] = RunInfo(d.name, d)
    return out, skipped


# --------------------------------------------------------------------------- static / glob family resolution

def _expand_braces(pattern: str) -> list[str]:
    """Minimal `{a,b,c}` brace expansion (stdlib has none) — handles exactly
    one brace group, which is all `work_packages.yaml` uses today."""
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    options = m.group(1).split(",")
    return [pattern[:m.start()] + opt + pattern[m.end():] for opt in options]


def _glob_files_recursive(pattern: str) -> list[Path]:
    """`pathlib.glob("dir/**")` matches directories only (including itself),
    never files, unless followed by `/*` — normalize a trailing bare `**`
    (as every `glob:` value in work_packages.yaml is written, per the
    charter's own convention) before handing it to `Path.glob`."""
    if pattern.endswith("**"):
        pattern = pattern + "/*"
    return list(DATA_ROOT.glob(pattern))


def _glob_family_files(pattern: str, sites: list[str]) -> list[Path]:
    """`pattern` is repo-relative, may contain `<site>` and one `{a,b,c}`
    brace group. `<site>` expands over every declared site AND, when the
    pattern's parent contains no `<site>` token at all (e.g. the
    presentation_figures cross_site/paper_figures branches), is dropped."""
    out: list[Path] = []
    seen: set[Path] = set()
    for expanded in _expand_braces(pattern):
        if "<site>" in expanded:
            for site in sites + ["cross_site", "paper_figures"]:
                p = expanded.replace("<site>", site)
                for f in _glob_files_recursive(p):
                    if f.is_file() and f not in seen:
                        seen.add(f)
                        out.append(f)
        else:
            for f in _glob_files_recursive(expanded):
                if f.is_file() and f not in seen:
                    seen.add(f)
                    out.append(f)
    return [f for f in out if f.suffix.lower() in IMG_EXTS]


def _static_root_files(static_root: str) -> list[Path]:
    """`static_root` is repo-relative to DATA_ROOT (`outputs/...`,
    `docs/...`) OR, when it starts with `brisaverse/`, relative to the
    sibling brisaverse checkout (DATA_ROOT.parent)."""
    if static_root.startswith("brisaverse/"):
        base = DATA_ROOT.parent / static_root
    else:
        base = DATA_ROOT / static_root
    if not base.is_dir():
        return []
    return [f for f in sorted(base.rglob("*")) if f.is_file() and f.suffix.lower() in IMG_EXTS]


def _convention_files(family: str) -> list[Path]:
    """Fallback when a family has neither `static_root` nor `glob` and no
    run-directory prefix matches: the same `outputs/<site>/<suffix>/**`
    idiom `site_paper_figures`/`site_morphometrics` already use explicitly,
    generalised for every other `site_*` / `comparative_*` family (confirmed
    against the 2026-09-24 census: site_svf_v2, site_territory, site_print,
    site_solar, comparative_health all follow this shape). Report-only at
    this phase (charter phase B) — a family this misses simply shows its
    files as unclassified rather than mis-attributing anything."""
    candidates: list[str] = [f"outputs/{family}/**/*"]
    if family.startswith("site_"):
        suffix = family[len("site_"):]
        candidates.append(f"outputs/*/{suffix}/**/*")
    if family.startswith("comparative_"):
        suffix = family[len("comparative_"):]
        candidates.append(f"outputs/comparative/{suffix}/**/*")
    if family.startswith("cross_site_"):
        suffix = family[len("cross_site_"):]
        candidates.append(f"outputs/cross_site/{suffix}/**/*")
    if family.startswith("distribution_"):
        suffix = family[len("distribution_"):]
        candidates.append(f"outputs/_distribution/{suffix}/**/*")
    out: list[Path] = []
    seen: set[Path] = set()
    for pat in candidates:
        for f in DATA_ROOT.glob(pat):
            if f.is_file() and f.suffix.lower() in IMG_EXTS and f not in seen:
                seen.add(f)
                out.append(f)
    return out


# --------------------------------------------------------------------------- registry build

def build(*, verbose: bool = False) -> dict:
    wp_cfg = load_work_packages()
    sites = load_sites()
    tasks = load_tasks()
    all_decisions = [(d, "open") for d in tasks["open_decisions"]] + \
                     [(d, "resolved") for d in tasks["resolved_decisions"]]

    runs, skipped_runs = scan_runs()
    runs_by_family: dict[str, list[RunInfo]] = defaultdict(list)
    for ri in runs.values():
        runs_by_family[ri.family].append(ri)

    nodes: dict[str, dict] = {}
    placed_paths: set[str] = set()
    declared_families: set[str] = set()

    def decision_ids_for(*needles: str) -> list[str]:
        out: list[str] = []
        for n in needles:
            for d in _decision_hits(all_decisions, n):
                if d not in out:
                    out.append(d)
        return out

    wps = wp_cfg.get("work_packages") or {}
    for wp_key, wp_row in wps.items():
        wp_node_id = f"wp:{wp_key}"
        nodes[wp_node_id] = {
            "kind": "wp", "parent": None, "title": wp_row.get("title"),
            "papers": wp_row.get("papers") or [],
        }
        families = wp_row.get("families") or {}
        for family, fam_cfg in families.items():
            declared_families.add(family)
            fam_node_id = f"fam:{family}"
            static_root = fam_cfg.get("static_root")
            glob_pat = fam_cfg.get("glob")
            run_backed = family in runs_by_family

            if run_backed:
                fam_node = _build_run_backed_family(
                    nodes, fam_node_id, wp_node_id, wp_key, family, fam_cfg,
                    runs_by_family[family], sites, decision_ids_for, placed_paths,
                )
            else:
                if static_root:
                    files = _static_root_files(static_root)
                    src_label = static_root
                elif glob_pat:
                    files = _glob_family_files(glob_pat, sites)
                    src_label = glob_pat
                else:
                    files = _convention_files(family)
                    src_label = f"outputs/{family}/** (convention fallback)"
                fam_node = _build_static_family(
                    nodes, fam_node_id, wp_node_id, wp_key, family, files, src_label,
                    decision_ids_for, placed_paths,
                )
            nodes[fam_node_id] = fam_node
            if verbose:
                print(f"  {wp_key}/{family}: {fam_node.get('_n_artifacts', 0)} artifacts "
                      f"({'run-backed' if run_backed else fam_node.get('_source')})")

    # ---- unassigned (declared, no WP — still real family membership) --------
    for family, row in (wp_cfg.get("unassigned") or {}).items():
        declared_families.add(family)
        fam_node_id = f"fam:{family}"
        wp_node_id = "wp:UNASSIGNED"
        if wp_node_id not in nodes:
            nodes[wp_node_id] = {"kind": "wp", "parent": None, "title": "Unassigned", "papers": []}
        if family in runs_by_family:
            fam_node = _build_run_backed_family(
                nodes, fam_node_id, wp_node_id, "UNASSIGNED", family, {}, runs_by_family[family],
                sites, decision_ids_for, placed_paths,
            )
        else:
            files = _convention_files(family)
            fam_node = _build_static_family(
                nodes, fam_node_id, wp_node_id, "UNASSIGNED", family, files,
                f"outputs/{family}/** (convention fallback)", decision_ids_for, placed_paths,
            )
        fam_node["reason"] = row.get("reason")
        nodes[fam_node_id] = fam_node

    # ---- orphan run families: real runs/ dirs whose family nobody declared --
    orphan_run_families = sorted(set(runs_by_family) - declared_families)

    # ---- unclassified ---------------------------------------------------
    # A hash match under an EXCLUDED_TOP_DIRS/EXCLUDED_SUBSTRINGS path is a
    # genuine copy of an already-registered artifact and is legitimately
    # skipped; anything else there is real, unregistered content and must
    # be counted (see the EXCLUDED_TOP_DIRS docstring above).
    placed_hashes: set[str] = {
        n["content_hash"] for n in nodes.values()
        if n.get("kind") == "figure" and n.get("content_hash")
    }
    # O8 cleanup (figure_organization_spec.md §1/§6): "Rows with the same
    # content_hash become one canonical row, with the other paths in
    # copies[]." Applied here to the unclassified sweep itself — a review
    # snapshot (`_review/<date>/sweep/**`) that mirrors an ALREADY-
    # unclassified original is real content seen twice, not two pieces of
    # unaccounted content, so it collapses to one counted row + a copies[]
    # entry rather than inflating the count. A candidate under _review/_hub
    # is never chosen as the canonical when a non-mirror path with the same
    # hash exists, so the canonical path a PI is shown is always the real
    # output location, never a dated snapshot.
    def _bucket_of(rel: str) -> str:
        parts = Path(rel).parts  # outputs/<top>/<sub>/.../<file>
        # Two levels when the file sits at least that deep (parts[1:3] are
        # both directories); one level — never the filename itself — when
        # the file sits directly inside a single top folder.
        return "/".join(parts[1:3]) if len(parts) > 3 else parts[1]

    def _dedup_sweep(candidates: list[tuple[str, Path]]) -> dict:
        by_hash: dict[str, list[str]] = defaultdict(list)
        hashless: list[str] = []
        for rel, f in candidates:
            h = _sha256(f)
            if h is None:
                hashless.append(rel)
                continue
            if h in placed_hashes and _is_excluded(rel):
                continue  # genuine mirror of a registered artifact — not counted at all
            by_hash[h].append(rel)

        by_folder: dict[str, int] = defaultdict(int)
        copies: dict[str, list[str]] = {}
        total = 0
        for h, rels in by_hash.items():
            non_mirror = [r for r in rels if not _is_excluded(r)]
            canonical = sorted(non_mirror or rels)[0]
            dupes = sorted(r for r in rels if r != canonical)
            by_folder[_bucket_of(canonical)] += 1
            total += 1
            if dupes:
                copies[canonical] = dupes
        for rel in hashless:
            by_folder[_bucket_of(rel)] += 1
            total += 1
        return {
            "count": total,
            "by_folder": dict(sorted(by_folder.items())),
            "copies": dict(sorted(copies.items())),
            "duplicate_files_collapsed": sum(len(v) for v in copies.values()),
        }

    unclassified_candidates: list[tuple[str, Path]] = []
    archived_candidates: list[tuple[str, Path]] = []
    if OUTPUTS.is_dir():
        for f in sorted(OUTPUTS.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in IMG_EXTS:
                continue
            rel = _posix(f)
            if rel in placed_paths:
                continue
            top = rel.split("/", 2)
            if len(top) >= 2 and top[0] == "outputs" and top[1] == ARCHIVE_TOP_DIR:
                archived_candidates.append((rel, f))
            else:
                unclassified_candidates.append((rel, f))

    unclassified_sweep = _dedup_sweep(unclassified_candidates)
    archived_sweep = _dedup_sweep(archived_candidates)

    registry = {
        "_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generator": "scripts/build_results_registry.py",
        "charter": "docs/charter/organization_charter.md §2",
        "nodes": nodes,
        "unclassified": unclassified_sweep,
        "archived": archived_sweep,
        "orphan_run_families": orphan_run_families,
        "skipped_incomplete_runs": skipped_runs,
        "counts": {
            "wp": sum(1 for n in nodes.values() if n.get("kind") == "wp"),
            "family": sum(1 for n in nodes.values() if n.get("kind") == "family"),
            "run": sum(1 for n in nodes.values() if n.get("kind") == "run"),
            "figure": sum(1 for n in nodes.values() if n.get("kind") == "figure"),
            "alias": sum(1 for n in nodes.values() if n.get("kind") == "alias"),
            "placed": len(placed_paths),
            "unclassified": unclassified_sweep["count"],
            "archived": archived_sweep["count"],
            "skipped_incomplete_runs": len(skipped_runs),
        },
    }
    return registry


def _build_run_backed_family(nodes, fam_node_id, wp_node_id, wp_key, family, fam_cfg,
                              family_runs, sites, decision_ids_for, placed_paths) -> dict:
    family_runs = sorted(family_runs, key=lambda r: r.run_utc or "")
    ok_runs = [r for r in family_runs
               if not r.figures or any(v.get("status") == "produced" for v in r.figures.values())]
    head = (ok_runs or family_runs)[-1] if family_runs else None
    head_run_id = head.run_id if head else None
    n_artifacts = 0

    for ri in family_runs:
        run_node_id = f"run:{ri.run_id}"
        is_head = ri.run_id == head_run_id
        has_produced = (not ri.figures) or any(v.get("status") == "produced" for v in ri.figures.values())
        lifecycle = "current" if is_head else ("draft" if not has_produced else "superseded")
        nodes[run_node_id] = {
            "kind": "run", "parent": fam_node_id, "run_utc": ri.run_utc,
            "lifecycle": lifecycle, "status": ri.status,
            "superseded_by": None if is_head else f"run:{head_run_id}",
            "decision_ids": decision_ids_for(ri.run_id),
        }
        for slug, fig in sorted(ri.figures.items()):
            png_path = fig.get("png_path")
            svg_path = fig.get("svg_path")
            abs_png = _posix(ri.path / png_path) if png_path else None
            abs_svg = _posix(ri.path / svg_path) if svg_path else None
            for p in (abs_png, abs_svg):
                if p:
                    placed_paths.add(p)
            content_hash = None
            for p in (abs_png, abs_svg):
                if p and (DATA_ROOT / p).exists():
                    content_hash = _sha256(DATA_ROOT / p)
                    break
            fig_lifecycle = "draft" if fig.get("status") == "skipped" else lifecycle
            site = (fig.get("window") or {}).get("id") or _match_site(slug, sites)
            art_id = f"art:{family}::{ri.run_id}::{slug}"
            nodes[art_id] = {
                "kind": "figure", "parent": run_node_id,
                "wp": wp_key, "family": family,
                "path": abs_png or abs_svg, "svg_path": abs_svg if abs_png else None,
                "content_hash": content_hash, "thumb_hash": content_hash,
                "site": site, "produced_utc": ri.run_utc,
                "generator": {"script": fam_cfg.get("script"), "git_sha": None},
                # A declared `derived_from` key names another figure by its
                # SLUG (the writer doesn't know at authoring time which run
                # will end up current) — resolve it against that slug's
                # `current` alias, not a bare 2-part id that only exists for
                # static/glob families with no run axis at all.
                "derived_from": [f"art:{family}::current::{k}" for k in (fig.get("derived_from") or [])],
                "lifecycle": fig_lifecycle, "release": None, "guardian_verdict": None,
                "paper_ref": None, "decision_ids": decision_ids_for(slug),
                "legacy_ids": [], "unclassified": False,
                "status": fig.get("status"), "reason": fig.get("reason"),
            }
            n_artifacts += 1
            if is_head and fig.get("status") != "skipped":
                alias_id = f"art:{family}::current::{slug}"
                nodes[alias_id] = {"kind": "alias", "target": art_id}

    return {"kind": "family", "parent": wp_node_id, "head_run": f"run:{head_run_id}" if head_run_id else None,
            "static_root": False, "n_runs": len(family_runs), "_n_artifacts": n_artifacts}


def _build_static_family(nodes, fam_node_id, wp_node_id, wp_key, family, files, src_label,
                          decision_ids_for, placed_paths) -> dict:
    n_artifacts = 0
    for f in sorted(files):
        rel = _posix(f)
        if rel in placed_paths:
            continue
        placed_paths.add(rel)
        slug = f.stem
        art_id = f"art:{family}::{slug}"
        nodes[art_id] = {
            "kind": "figure", "parent": fam_node_id,
            "wp": wp_key, "family": family,
            "path": rel, "svg_path": None,
            "content_hash": _sha256(f), "thumb_hash": _sha256(f),
            "site": _match_site(rel, []), "produced_utc": None,
            "generator": None, "derived_from": [],
            "lifecycle": "current", "release": None, "guardian_verdict": None,
            "paper_ref": None, "decision_ids": decision_ids_for(slug),
            "legacy_ids": [], "unclassified": False, "status": "produced", "reason": None,
        }
        n_artifacts += 1
    return {"kind": "family", "parent": wp_node_id, "head_run": None,
            "static_root": True, "source": src_label, "n_files": len(files), "_n_artifacts": n_artifacts,
            "_source": src_label}


def _match_site(text: str, sites: list[str]) -> str | None:
    t = text.lower().replace(" ", "_")
    sites = sites or load_sites()
    for s in sites:
        if s and s.lower() in t:
            return s
    return None


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", help="print a per-family placed-artifact line")
    args = ap.parse_args()

    registry = build(verbose=args.verbose)
    REGISTRY_OUT.mkdir(parents=True, exist_ok=True)
    out_path = REGISTRY_OUT / "results.json"
    out_path.write_text(json.dumps(registry, indent=1, sort_keys=True))

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps({
            "recorded_utc": registry["_utc"],
            "unclassified_count": registry["unclassified"]["count"],
            "note": "charter §3 R7 ratchet floor — recorded on the first build_results_registry.py run "
                    "(charter phase B / O2). check_registry.py R7 fails if unclassified.count rises above this.",
        }, indent=1))

    print(f"wrote {out_path}")
    print(json.dumps(registry["counts"], indent=1))
    if registry["orphan_run_families"]:
        print(f"WARNING: {len(registry['orphan_run_families'])} run families exist on disk but are not "
              f"declared in work_packages.yaml: {', '.join(registry['orphan_run_families'])}")
    if registry["skipped_incomplete_runs"]:
        print(f"WARNING: {len(registry['skipped_incomplete_runs'])} run director{'y' if len(registry['skipped_incomplete_runs']) == 1 else 'ies'} "
              "skipped as incomplete (missing/unparseable manifest — likely mid-write by another agent):")
        for s in registry["skipped_incomplete_runs"]:
            print(f"  {s['run_id']}: {s['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
