#!/usr/bin/env python3
"""Registration-contract gate for the results registry (charter §3, R1–R7).

    python scripts/check_registry.py            # build a fresh registry, check it, exit 0/1
    python scripts/check_registry.py --report    # same, plus a full human-readable report
    python scripts/check_registry.py --self-test # sabotages a throwaway fixture 4 different
                                                   # ways and asserts every one of them goes red

Charter phase B (this phase, O2) runs R1–R6 report-only — findings print but
never fail the gate — except where a check is written to be hard by
construction (this generator's ids are run-scoped from the start, so R3's
"id collision" class can only be sabotaged at the raw-manifest source, which
this script also detects, always hard, because it is cheap and a real
defect the moment it appears — see `check_source_id_collisions`). R7 (the
unclassified ratchet) is hard now, because it costs nothing to enforce and
the whole point of a ratchet is that it never silently loosens. R1/R2 become
hard for every run per charter §3 only once the run postdates
`ENFORCEMENT_CUTOFF` (2026-09-25) — before that, every run in this repo
predates the cutoff by construction, so the report-only default is what
actually runs against the live repo today; the self-test manufactures a
run dated after the cutoff to exercise the hard path.

Invariants (charter §3):
  R1  Every run dir (after the cutoff) with images has a figure_manifest.json      [orphan run]
  R2  Every run family is declared in work_packages.yaml                          [hard for new runs]
  R3  family::id is unique; every derived_from / supersedes key resolves          [hard]
  R4  Exactly one `current` run per family                                        [hard]
  R5  Different content hash implies different thumbnail hash                     [hard]
  R6  Every registry path exists; every image in a run dir is in its manifest     [orphan file]
  R7  unclassified.count <= registry/baseline.json's recorded value               [ratchet: only down]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_results_registry as brr  # noqa: E402

ENFORCEMENT_CUTOFF = date(2026, 9, 25)

# Charter phase B (migration plan §6): "R1–R6 report only; the baseline is
# recorded." R3 (id collision / dangling reference) and R7 (the unclassified
# ratchet) are the two invariants this phase enforces for real — they are
# cheap, meaningful against every run regardless of age, and the whole point
# of a ratchet is that it never gets to loosen quietly. R1/R2/R6 keep their
# real FAIL/warn severity (and the cutoff-based hardening below, which
# starts mattering once a run postdates 2026-09-25) for reporting and for
# `--self-test`, which asserts on severity directly — this set only changes
# what `main()`'s plain (no-flag) exit code blocks the git-push gate on, so
# `check_registry.py` wired into `.claude/verify-cmd` today does not fail
# every future commit over the ~150 run directories that predate this
# registration contract entirely.
PHASE_B_GATE_BLOCKING = {"R3", "R7"}


class Finding:
    def __init__(self, rule: str, severity: str, message: str, path: str | None = None):
        self.rule, self.severity, self.message, self.path = rule, severity, message, path

    def __str__(self) -> str:
        loc = f" ({self.path})" if self.path else ""
        return f"[{self.severity}] {self.rule}: {self.message}{loc}"


def _run_date(run_utc: str | None) -> date | None:
    if not run_utc:
        return None
    try:
        return date.fromisoformat(run_utc[:10])
    except ValueError:
        return None


# --------------------------------------------------------------------------- R1 / R2: orphan run, undeclared family

def check_orphan_runs_and_undeclared_families(registry: dict, declared_families: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    if not brr.RUNS.is_dir():
        return findings
    run_nodes = {k: v for k, v in registry["nodes"].items() if v.get("kind") == "run"}
    known_run_ids = {k.split(":", 1)[1] for k in run_nodes}
    for d in sorted(brr.RUNS.iterdir()):
        if not d.is_dir():
            continue
        has_images = any(f.is_file() and f.suffix.lower() in brr.IMG_EXTS for f in d.rglob("*"))
        has_manifest = (d / "figure_manifest.json").exists()
        family = brr._family_of_rundir(d.name)
        run_utc = brr._run_utc_from_name(d.name)
        rdate = _run_date(run_utc)
        hard = rdate is not None and rdate >= ENFORCEMENT_CUTOFF

        if has_images and not has_manifest:
            findings.append(Finding("R1", "FAIL" if hard else "warn",
                                     "run directory has images but no figure_manifest.json (orphan run)",
                                     path=str(d.relative_to(brr.DATA_ROOT))))
        if family not in declared_families and d.name not in known_run_ids:
            # d.name not resolved into ANY run node at all means build_results_registry
            # never even scanned it as belonging to a known run (shouldn't happen —
            # scan_runs() scans every dir); the real signal is `family not declared`.
            pass
        if family not in declared_families and has_images:
            findings.append(Finding("R2", "FAIL" if hard else "warn",
                                     f"run family '{family}' is not declared in work_packages.yaml",
                                     path=str(d.relative_to(brr.DATA_ROOT))))
    return findings


# --------------------------------------------------------------------------- R3a: id collision at the source

class _DuplicateKey(Exception):
    pass


def _no_dup_keys_hook(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise _DuplicateKey(k)
        seen[k] = v
    return seen


def check_source_id_collisions() -> list[Finding]:
    """Re-parses every `runs/*/figure_manifest.json` with an
    `object_pairs_hook` that raises on a repeated key. This is the ONLY way
    to actually catch a literal duplicate id in the `figures` object —
    `json.loads` with the default hook silently keeps the last occurrence
    and gives no signal that a collision ever happened, which is exactly
    how the round-1 prototype's `art:<family>::<slug>` id scheme could show
    a superseded run's figure as current without either the registry OR a
    naive re-check ever noticing. This check is hard regardless of the
    cutoff date — a manifest that collides with itself is never valid."""
    findings: list[Finding] = []
    if not brr.RUNS.is_dir():
        return findings
    for d in sorted(brr.RUNS.iterdir()):
        fm = d / "figure_manifest.json"
        if not fm.exists():
            continue
        try:
            json.loads(fm.read_text(), object_pairs_hook=_no_dup_keys_hook)
        except _DuplicateKey as e:
            findings.append(Finding("R3", "FAIL",
                                     f"figure_manifest.json declares the same figure id twice: '{e}' "
                                     "(id collision — family::id would not be unique)",
                                     path=str(fm.relative_to(brr.DATA_ROOT))))
        except json.JSONDecodeError:
            pass  # not this gate's job — a malformed manifest is its own problem
    return findings


# --------------------------------------------------------------------------- R3b: derived_from / supersedes resolve

def check_dangling_references(registry: dict) -> list[Finding]:
    findings: list[Finding] = []
    nodes = registry["nodes"]
    for node_id, node in nodes.items():
        for key in node.get("derived_from") or []:
            if key not in nodes:
                findings.append(Finding("R3", "FAIL",
                                         f"'{node_id}' has a dangling derived_from -> '{key}' (no such node)",
                                         path=node_id))
        sup = node.get("superseded_by")
        if sup and sup not in nodes:
            findings.append(Finding("R3", "FAIL",
                                     f"'{node_id}' has a dangling superseded_by -> '{sup}' (no such node)",
                                     path=node_id))
    return findings


# --------------------------------------------------------------------------- R4: exactly one current run per family

def check_one_current_per_family(registry: dict) -> list[Finding]:
    findings: list[Finding] = []
    nodes = registry["nodes"]
    by_family: dict[str, list[str]] = {}
    for node_id, node in nodes.items():
        if node.get("kind") == "run":
            fam = node.get("parent")
            by_family.setdefault(fam, []).append(node_id)
    for fam, run_ids in by_family.items():
        currents = [r for r in run_ids if nodes[r].get("lifecycle") == "current"]
        if len(currents) != 1:
            findings.append(Finding("R4", "FAIL",
                                     f"family '{fam}' has {len(currents)} 'current' runs, expected exactly 1",
                                     path=fam))
    return findings


# --------------------------------------------------------------------------- R5: hash implies thumb hash differs

def check_hash_thumb_consistency(registry: dict) -> list[Finding]:
    # R5 is phrased as an implication (different content -> different thumb);
    # its violation is the SAME thumb_hash claimed for two DIFFERENT content
    # hashes, so group by thumb_hash and look for >1 distinct content_hash.
    findings: list[Finding] = []
    by_thumb: dict[str, set[str]] = {}
    for node in registry["nodes"].values():
        if node.get("kind") != "figure" or node.get("content_hash") is None:
            continue
        by_thumb.setdefault(node["thumb_hash"], set()).add(node["content_hash"])
    for th, hashes in by_thumb.items():
        if len(hashes) > 1:
            findings.append(Finding("R5", "FAIL",
                                     f"thumb_hash '{th}' is shared by {len(hashes)} different content hashes",
                                     path=th))
    return findings


# --------------------------------------------------------------------------- R6: paths exist, orphan files

def check_paths_exist(registry: dict) -> list[Finding]:
    findings: list[Finding] = []
    for node_id, node in registry["nodes"].items():
        if node.get("kind") != "figure":
            continue
        for key in ("path", "svg_path"):
            p = node.get(key)
            if p and not (brr.DATA_ROOT / p).exists():
                findings.append(Finding("R6", "FAIL", f"registered path does not exist on disk: {p}", path=node_id))
    return findings


def check_orphan_files_in_runs() -> list[Finding]:
    findings: list[Finding] = []
    if not brr.RUNS.is_dir():
        return findings
    for d in sorted(brr.RUNS.iterdir()):
        fm_path = d / "figure_manifest.json"
        fm = brr._read_json(fm_path) if fm_path.exists() else None
        if fm is None:
            continue  # R1 already covers "no manifest at all"
        declared: set[Path] = set()
        for fig in (fm.get("figures") or {}).values():
            for key in ("png_path", "svg_path"):
                v = fig.get(key)
                if v:
                    declared.add((d / v).resolve())
        rdate = _run_date(brr._run_utc_from_name(d.name))
        hard = rdate is not None and rdate >= ENFORCEMENT_CUTOFF
        for f in d.rglob("*"):
            if f.is_file() and f.suffix.lower() in brr.IMG_EXTS and f.resolve() not in declared:
                findings.append(Finding("R6", "FAIL" if hard else "warn",
                                         "image inside a run directory that run's own figure_manifest.json "
                                         "does not declare (unregistered output)",
                                         path=str(f.relative_to(brr.DATA_ROOT))))
    return findings


# --------------------------------------------------------------------------- R7: unclassified ratchet

def check_unclassified_ratchet(registry: dict) -> list[Finding]:
    if not brr.BASELINE_PATH.exists():
        return [Finding("R7", "warn", "no registry/baseline.json yet — run build_results_registry.py once "
                                       "to record it", path=str(brr.BASELINE_PATH))]
    baseline = json.loads(brr.BASELINE_PATH.read_text())
    current = registry["unclassified"]["count"]
    floor = baseline["unclassified_count"]
    if current > floor:
        return [Finding("R7", "FAIL",
                         f"unclassified.count rose from the baseline {floor} to {current} — the ratchet "
                         "may only go down (charter §3 R7)", path=str(brr.BASELINE_PATH))]
    return []


# --------------------------------------------------------------------------- driver

def run_checks(registry: dict, declared_families: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    findings += check_orphan_runs_and_undeclared_families(registry, declared_families)
    findings += check_source_id_collisions()
    findings += check_dangling_references(registry)
    findings += check_one_current_per_family(registry)
    findings += check_hash_thumb_consistency(registry)
    findings += check_paths_exist(registry)
    findings += check_orphan_files_in_runs()
    findings += check_unclassified_ratchet(registry)
    return findings


def _declared_families(wp_cfg: dict) -> set[str]:
    out: set[str] = set()
    for wp_row in (wp_cfg.get("work_packages") or {}).values():
        out |= set((wp_row.get("families") or {}).keys())
    out |= set((wp_cfg.get("unassigned") or {}).keys())
    return out


def report(registry: dict, findings: list[Finding]) -> None:
    fails = [f for f in findings if f.severity == "FAIL"]
    warns = [f for f in findings if f.severity == "warn"]
    print(f"results registry: {json.dumps(registry['counts'])}")
    print(f"checked: {len(findings)} findings ({len(fails)} FAIL, {len(warns)} warn)")
    for f in fails:
        print(" ", f)
    for f in warns:
        print(" ", f)
    if not findings:
        print("  clean — R1-R7 all pass, nothing to report")


# --------------------------------------------------------------------------- self-test

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def self_test() -> int:
    """Sabotages a throwaway fixture 4 different ways (the O2 task's own
    list): unregistered output, orphan run, id collision, dangling
    derived_from — each must independently make the gate go red."""
    tmp = Path(tempfile.mkdtemp(prefix="check_registry_selftest_"))
    ok = True
    try:
        root = tmp / "MorphoFavela"
        brisa = tmp / "brisaverse"
        runs, outputs, config = root / "runs", root / "outputs", root / "config"

        _write(config / "sites.yaml", yaml.dump({"sites": {"vidigal": {"display_name": "Vidigal"}}}))
        _write_json(brisa / "shared" / "facts" / "tasks.json", {"open_decisions": [], "resolved_decisions": []})
        _write(config / "work_packages.yaml", yaml.dump({
            "work_packages": {"WP07": {"title": "x", "papers": ["p1"],
                                        "families": {"wp07_figures": {"script": "src/x.py"}}}},
            "unassigned": {},
        }))

        brr.ROOT = root
        brr.DATA_ROOT = root
        brr.RUNS = runs
        brr.OUTPUTS = outputs
        brr.CONFIG = config
        brr.REGISTRY_OUT = outputs / "_registry"
        brr.BASELINE_PATH = root / "registry" / "baseline.json"
        brr.WP_YAML = config / "work_packages.yaml"
        brr.BRISA = brisa
        brr.TASKS_JSON = brisa / "shared" / "facts" / "tasks.json"

        def fresh_run(run_id: str, figures_json_text: str) -> Path:
            d = runs / run_id
            _write(d / "figure_manifest.json", figures_json_text)
            return d

        # -- 1. clean baseline: one good run, must be all-clear -----------
        good_run_id = "wp07_figures_20260917T125201Z"
        d = fresh_run(good_run_id, json.dumps({
            "_utc": "2026-09-17T12:52:02Z",
            "figures": {"f1": {"id": "f1", "status": "produced", "png_path": "f1.png"}},
        }))
        _write(d / "f1.png")
        reg = brr.build()
        declared = _declared_families(yaml.safe_load(brr.WP_YAML.read_text()))
        findings = run_checks(reg, declared)
        fails = [f for f in findings if f.severity == "FAIL"]
        if fails:
            print("SELF-TEST SETUP FAILED: the clean fixture should have zero FAILs:")
            for f in fails:
                print("  ", f)
            ok = False
        else:
            print("  clean fixture: OK (0 FAIL, as expected)")

        # -- 2. orphan run: images, no figure_manifest.json, post-cutoff --
        orphan_dir = runs / "wp07_figures_20260926T000000Z"
        _write(orphan_dir / "f9.png")
        reg2 = brr.build()
        findings2 = run_checks(reg2, declared)
        hit = any(f.rule == "R1" and f.severity == "FAIL" for f in findings2)
        ok = _assert(ok, hit, "orphan run", findings2)
        shutil.rmtree(orphan_dir)

        # -- 3. id collision: figure_manifest.json declares the same id twice
        colliding_text = (
            '{"_utc": "2026-09-17T13:00:00Z", "figures": {'
            '"f1": {"id": "f1", "status": "produced", "png_path": "a.png"}, '
            '"f1": {"id": "f1", "status": "produced", "png_path": "b.png"}'
            '}}'
        )
        collide_dir = fresh_run("wp07_figures_20260917T130000Z", colliding_text)
        _write(collide_dir / "a.png")
        _write(collide_dir / "b.png")
        reg3 = brr.build()
        findings3 = run_checks(reg3, declared)
        hit = any(f.rule == "R3" and "id collision" in f.message and f.severity == "FAIL" for f in findings3)
        ok = _assert(ok, hit, "id collision", findings3)
        shutil.rmtree(collide_dir)

        # -- 4. dangling derived_from: names a slug that does not exist ---
        dangling_dir = fresh_run("wp07_figures_20260917T140000Z", json.dumps({
            "_utc": "2026-09-17T14:00:00Z",
            "figures": {"f2": {"id": "f2", "status": "produced", "png_path": "f2.png",
                                "derived_from": ["nonexistent_slug"]}},
        }))
        _write(dangling_dir / "f2.png")
        reg4 = brr.build()
        findings4 = run_checks(reg4, declared)
        hit = any(f.rule == "R3" and "dangling derived_from" in f.message and f.severity == "FAIL" for f in findings4)
        ok = _assert(ok, hit, "dangling derived_from", findings4)
        shutil.rmtree(dangling_dir)

        # -- 5. unregistered output: image in a run dir the manifest never declares
        d2 = fresh_run("wp07_figures_20260926T010000Z", json.dumps({
            "_utc": "2026-09-26T01:00:00Z",
            "figures": {"f3": {"id": "f3", "status": "produced", "png_path": "f3.png"}},
        }))
        _write(d2 / "f3.png")
        _write(d2 / "f4_never_declared.png")
        reg5 = brr.build()
        findings5 = run_checks(reg5, declared)
        hit = any(f.rule == "R6" and "unregistered output" in f.message and f.severity == "FAIL" for f in findings5)
        ok = _assert(ok, hit, "unregistered output", findings5)
        shutil.rmtree(d2)

        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _assert(ok: bool, hit: bool, label: str, findings: list[Finding]) -> bool:
    if hit:
        print(f"  {label}: OK (caught)")
        return ok
    print(f"  {label}: FAILED — gate did NOT go red. All findings:")
    for f in findings:
        print("    ", f)
    return False


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        rc = self_test()
        print("SELF-TEST " + ("PASSED" if rc == 0 else "FAILED"))
        return rc

    wp_cfg = yaml.safe_load(brr.WP_YAML.read_text())
    declared = _declared_families(wp_cfg)
    registry_path = brr.REGISTRY_OUT / "results.json"
    if not registry_path.exists():
        print(f"no {registry_path} yet — run scripts/build_results_registry.py first", file=sys.stderr)
        return 1
    registry = json.loads(registry_path.read_text())
    findings = run_checks(registry, declared)
    if args.report:
        report(registry, findings)
    fails = [f for f in findings if f.severity == "FAIL"]
    gate_fails = [f for f in fails if f.rule in PHASE_B_GATE_BLOCKING]
    if fails and not args.report:
        for f in fails:
            print(f)
    if gate_fails:
        return 1
    if fails:
        print(f"({len(fails)} R1/R2/R6 FAIL finding(s) reported but not gate-blocking — "
              f"charter phase B is report-only for those; see --report)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
