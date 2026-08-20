#!/usr/bin/env python3
"""Emit docs/cockpit.json — curated cockpit/v1 projection for the brisaverse hub.

THIN PROJECTION of existing repo state, no new state. Sources:
  docs/_tracks_index.json                      (track statuses; regenerate first
                                                via scripts/build_tracks_index.py)
  tests/data/lambda_f_canonical_golden.json    (pinned canonical λf snapshot)
  docs/lambda_f_fix_and_figure_regen_plan.md   (the one recorded open PI fork)
  docs/work_queue.md                           (blocked items)

Curated entries below are guarded by marker greps against their source file, so
resolving the item in the doc drops it from the cockpit at the next emit.

  python3 scripts/emit_cockpit.py                 # (re)write docs/cockpit.json
  python3 scripts/emit_cockpit.py --check         # gate mode: validate existing
                                                  # file (schema subset + fresher
                                                  # than its sources); never writes

Schema contract: ~/SCL/SCR/brisaverse/hub/cockpit_schema.json (cockpit/v1).
The bridge re-validates independently; --check embeds the same subset.
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "cockpit.json"

TRACKS = ROOT / "docs" / "_tracks_index.json"
GOLDEN = ROOT / "tests" / "data" / "lambda_f_canonical_golden.json"
LAMBDA_PLAN = ROOT / "docs" / "lambda_f_fix_and_figure_regen_plan.md"
WORK_QUEUE = ROOT / "docs" / "work_queue.md"
SOURCES = [TRACKS, GOLDEN, LAMBDA_PLAN, WORK_QUEUE]


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def status_bucket(status: str) -> str:
    return "unclear" if status.startswith("unclear") else status


def build() -> dict:
    tracks = json.loads(TRACKS.read_text())["tracks"]
    golden = json.loads(GOLDEN.read_text())
    plan_text = LAMBDA_PLAN.read_text()
    queue_text = WORK_QUEUE.read_text()

    counts = {}
    for t in tracks:
        b = status_bucket(t["status"])
        counts[b] = counts.get(b, 0) + 1
    n_stale = sum(1 for t in tracks if t.get("flag") == "STALE")
    breakdown = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))

    metrics = [
        {
            "name": "tracks_total",
            "value": len(tracks),
            "unit": "tracks",
            "kind": "declared",
            "source": rel(TRACKS),
            "formula_or_method": f"len(tracks); breakdown: {breakdown}",
        },
        {
            "name": "tracks_status_unclear",
            "value": counts.get("unclear", 0),
            "unit": "tracks",
            "kind": "declared",
            "source": rel(TRACKS),
            "formula_or_method": "count(status startswith 'unclear') — no explicit "
            "marker in the plan header; heuristic index, statuses are curated not measured",
        },
        {
            "name": "tracks_stale",
            "value": n_stale,
            "unit": "tracks",
            "kind": "declared",
            "source": rel(TRACKS),
            "formula_or_method": "count(flag=='STALE'), threshold "
            "stale_days > 14 on the plan file's last dated heading/mtime",
        },
        {
            "name": "skimming_regime_share_pooled",
            "value": round(golden["flow_regime_pooled"]["skimming"] * 100, 1),
            "unit": "% of built cells",
            "kind": "measured",
            "source": rel(GOLDEN),
            "formula_or_method": "flow_regime_pooled.skimming from the canonical "
            "dissolved-λf snapshot (2026-06-25); Oke(1988)/G&O(1999) thresholds "
            "(skimming λf ≥ 0.65)",
            "caveat": "value read from the test-pinned golden of the canonical "
            "output, not recomputed at emit time",
        },
    ]
    m = re.search(r"pooled n = ([\d,]+)", golden["canonical_denominator"])
    if m:
        metrics.append(
            {
                "name": "built_cells_pooled",
                "value": int(m.group(1).replace(",", "")),
                "unit": "grid cells",
                "kind": "declared",
                "source": rel(GOLDEN),
                "formula_or_method": "parsed from canonical_denominator prose "
                "('full built mask building_count > 0, pooled n'), 5 sites",
            }
        )

    open_decisions = []
    # A decision is live iff a "## N. DECISION NEEDED (<date>)" heading exists
    # whose heading line has not been edited to DECIDED. The 4b scope fork was
    # resolved 2026-08-20 (heading now says DECIDED); keying on heading state,
    # not on body text, prevents resurrecting superseded forks whose prose
    # survives in preserved blocks.
    live_headings = re.findall(r"^#{2,3} .*DECISION NEEDED \((\d{4}-\d{2}-\d{2})\)[^\n]*$",
                               plan_text, flags=re.M)
    if live_headings and "bootstrap-ARI stability gate" in plan_text:
        open_decisions.append(
            {
                "id": "morphofavela.ari-stability-bar",
                "question": "λf re-baseline HALTED: k=6 fit reproduces "
                "bit-for-bit, but bootstrap-ARI measures 0.843 on the correct "
                "5-site population vs the 0.90 bar TR §5.5 quotes (source of "
                "0.90 unestablished). Proceed how?",
                "owner": "PI",
                "options": [
                    {
                        "key": "A",
                        "label": "Re-pin bar to measured 0.843, cascade",
                        "detail": "Correct TR §5.5 to the honest measured "
                        "stability (0.843, sd 0.185, n=20, 5-site population, "
                        "config stated); run phases 2-3.",
                        "recommended": True,
                    },
                    {
                        "key": "B",
                        "label": "Investigate before cascading",
                        "detail": "Hold phases 2-3; probe bootstrap variance "
                        "(k, scaling, n_boot) for a defensible >=0.90 config.",
                    },
                    {
                        "key": "C",
                        "label": "Keep 06-25 fit, fix TR number only",
                        "detail": "No cascade; correct the stability claim in "
                        "place.",
                    },
                ],
                "blocks": ["morphotope", "typology_predictor", "tr_5_5", "brisa_p4_e2"],
            }
        )

    blocked_on = []
    if re.search(r"ray-caster.*BLOCKED", queue_text):
        blocked_on.append(
            "ray-caster cross-validation (work_queue Task 7): no Radiance/"
            "SOLWEIG installed locally — needs user"
        )
    for t in tracks:
        if t["track"] == "cfd_parameter_estimation" and t["status"] == "deferred":
            blocked_on.append(
                "cfd_parameter_estimation: DEFERRED — gated on the Airflow "
                "pilot recalibrating Kanda"
            )

    phase = (
        f"consolidation — {len(tracks)} tracks ({breakdown}); "
        f"{n_stale} stale >14d; statuses from a heuristic index, partly unclear"
    )
    return {
        "schema": "cockpit/v1",
        "repo": "MorphoFavela",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "status": {
            "phase": phase,
            "gate": "unknown — gate is `python -m pytest tests/ -q` "
            "(.claude/verify-cmd); no last-run state is recorded on disk",
            "source": " + ".join(rel(p) for p in SOURCES),
        },
        "headline_metrics": metrics[:5],
        "open_decisions": open_decisions[:5],
        "blocked_on": blocked_on,
        "next_wake": None,  # filled by main()
    }


# --- cockpit/v1 subset validation (mirrors brisaverse hub/cockpit_schema.json) ---

TOP_REQUIRED = [
    "schema", "repo", "generated_utc", "source_sha",
    "status", "headline_metrics", "open_decisions", "blocked_on", "next_wake",
]
METRIC_REQUIRED = ["name", "value", "unit", "kind", "source", "formula_or_method"]
DECISION_REQUIRED = ["id", "question", "owner", "options"]
KIND_ENUM = {"measured", "declared", "derived"}


def validate(doc: dict) -> list:
    errs = []
    for k in TOP_REQUIRED:
        if k not in doc:
            errs.append(f"missing top-level key: {k}")
    if errs:
        return errs
    if doc["schema"] != "cockpit/v1":
        errs.append("schema != cockpit/v1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", doc["repo"]):
        errs.append("repo fails pattern")
    if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", doc["generated_utc"]):
        errs.append("generated_utc fails pattern")
    if not re.fullmatch(r"[0-9a-f]{7,40}", doc["source_sha"]):
        errs.append("source_sha fails pattern")
    for k in ("phase", "gate", "source"):
        if not isinstance(doc["status"].get(k), str):
            errs.append(f"status.{k} missing/not string")
    if len(doc["headline_metrics"]) > 5:
        errs.append("headline_metrics > 5")
    for i, mt in enumerate(doc["headline_metrics"]):
        for k in METRIC_REQUIRED:
            if k not in mt:
                errs.append(f"metric[{i}] missing {k}")
        if mt.get("kind") not in KIND_ENUM:
            errs.append(f"metric[{i}] kind not in {sorted(KIND_ENUM)}")
        if not isinstance(mt.get("value"), (int, float, str)) or isinstance(mt.get("value"), bool):
            errs.append(f"metric[{i}] value not number/string")
    if len(doc["open_decisions"]) > 5:
        errs.append("open_decisions > 5")
    for i, d in enumerate(doc["open_decisions"]):
        for k in DECISION_REQUIRED:
            if k not in d:
                errs.append(f"decision[{i}] missing {k}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9._-]+", d.get("id", "")):
            errs.append(f"decision[{i}] id fails pattern")
        for j, o in enumerate(d.get("options", [])):
            if "key" not in o or "label" not in o:
                errs.append(f"decision[{i}].option[{j}] missing key/label")
    if not isinstance(doc["blocked_on"], list) or not all(
        isinstance(s, str) for s in doc["blocked_on"]
    ):
        errs.append("blocked_on not list[str]")
    if doc["next_wake"] is not None and not isinstance(doc["next_wake"], str):
        errs.append("next_wake not string/null")
    return errs


def check() -> int:
    if not OUT.exists():
        print(f"FAIL: {rel(OUT)} missing — run scripts/emit_cockpit.py", file=sys.stderr)
        return 1
    try:
        doc = json.loads(OUT.read_text())
    except json.JSONDecodeError as e:
        print(f"FAIL: {rel(OUT)} not valid JSON: {e}", file=sys.stderr)
        return 1
    errs = validate(doc)
    out_mtime = OUT.stat().st_mtime
    for src in SOURCES:
        if src.exists() and src.stat().st_mtime > out_mtime:
            errs.append(f"stale: {rel(src)} newer than {rel(OUT)} — re-run scripts/emit_cockpit.py")
    if errs:
        for e in errs:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"cockpit --check OK ({rel(OUT)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="validate existing file, never write")
    ap.add_argument("--next-wake", help="ISO date; default today+14d")
    args = ap.parse_args()
    if args.check:
        return check()
    doc = build()
    doc["next_wake"] = args.next_wake or (
        datetime.now(timezone.utc).date() + timedelta(days=14)
    ).isoformat()
    errs = validate(doc)
    if errs:
        for e in errs:
            print(f"FAIL (self-validation): {e}", file=sys.stderr)
        return 1
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
