# Project hub — every link must resolve under the mirror — spec (2026-09-16)

Status: implementer brief (MorphoFavela). The generated project hub
(`scripts/build_project_hub.py` + `scripts/hubkit.py` → `outputs/_hub/`) is now
served read-only at `https://brisa.theoalessandro.com/morphofavela-dash/_hub/`
from a mirror of `outputs/` whose root is `/morphofavela-dash/`. Measured with a
headless browser on 2026-09-16: 58 internal links on the landing page, 9 → 404.
Causes: (a) root-absolute hrefs (`/docs/technical_report/…` ×45 across the hub,
`/outputs/cross_site/…` ×12, `/docs/roughness_explainer/…` ×4, `/outputs/_hub/…` ×3)
that only worked when the repo root was the web root; (b) relative targets
outside the three mirrored subtrees (`comparative/maup`, `comparative/mingze_facade`,
`comparative/vidigal_vs_mingze/report`, `comparative/health`, `cross_site/roughness`,
`_distribution/site_dashboards`, `<site>/print`, `<site>/paper_figures`,
`paper_figures/exports`).

## Deliverables

1. **No root-absolute URLs anywhere in the generated hub.** In
   `build_project_hub.py` replace every `"/" + str(x.relative_to(ROOT))` (lines
   ~425–570 and ~935) and in `hubkit.py` any `/outputs/…` or `/docs/…` emission
   (`_make_thumb`, `_rel`, `md_to_html` base handling) with paths RELATIVE to the
   page being written (`os.path.relpath(target, page_dir)`). Pages live at
   `_hub/index.html`, `_hub/*.html`, `_hub/docs/*.html` — the base differs per page.
2. **Docs that live outside `outputs/`** (`docs/technical_report/technical_report.pdf`,
   `docs/roughness_explainer/*` and anything else under `docs/` the hub links):
   copy them into `outputs/_hub/docs/` at build time (only when newer) and link
   relatively. The hub must be self-contained under `outputs/`.
3. **Manifest of required subtrees**: the generator writes
   `outputs/_hub/mirror_manifest.json` — the sorted list of first-segment
   directories under `outputs/` that any generated href/src points at (computed
   from the emitted HTML, not typed). The orchestrator uses it to drive the
   rsync list and the hub route allowlist; it must never include per-cell
   withheld layers (assert no path under `<site>/morphometrics/grid`,
   `<site>/svf_v2/*.gpkg`, `runs/`, `<site>/cfd*`).
4. **Link check as a test**: `tests/test_build_project_hub.py` gains a test that
   builds the hub into a tmp tree from a SYNTHETIC outputs fixture (create the
   minimal files the generator discovers) and asserts (a) zero hrefs/srcs start
   with `/`, (b) every relative target exists under the tmp `outputs/`, (c) the
   manifest equals the set of first segments actually referenced. Existing tests
   keep passing.
5. Do NOT regenerate the real hub from the worktree (no `outputs/` there); the
   orchestrator regenerates on main. Give `build_project_hub.py` a `--root`
   flag (repo root override) so it can be run against the main checkout.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_build_project_hub.py -q
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

Commit last; paste hashes. Never `git add -A`; nothing under `outputs/`; no
edits to `outputs/_hub` by hand (generated).
