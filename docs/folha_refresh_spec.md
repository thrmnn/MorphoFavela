# Folha de Rua refresh — spec (2026-09-16, night cycle)

Status: implementer brief for task FOLHA. Branch `night-20260916/folha-refresh`.
Laptop only: ~1.5 min per A3 sheet, ~12 s per interactive dashboard.

Why: four of the five A3 sheets were rendered 2026-06-05 from
`outputs/<site>/morphometrics/svf/svf_streets_solar.gpkg` files that changed
2026-06-13/14 (Rocinha re-rendered 2026-07-02), so their numbers may be stale;
the interactive dashboards were rebuilt 2026-09-15. The PI wants the folha
refreshed and critiqued (screenshots + grading) for tomorrow morning.

Mandatory reads, then begin: `docs/critic/folha_round1.md` (the WORK LIST),
`scripts/build_site_dashboard.py`, `scripts/build_html_dashboard.py` (skim
`main`, `build_site`, `compute_site_stats` only),
`/home/theo/SCL/SCR/brisaverse/hub/ux/capture.py` (the capture pattern).

Live constants: `ROOT` is hard-coded to `/home/theo/SCL/SCR/MorphoFavela` in
both builders, so outputs are read from and written to the MAIN checkout even
from your worktree — intended tonight; your branch carries code + tests only.
Sites: `vidigal rocinha complexo_do_alemao riodaspedras maré` (folder name
carries the accent; ascii `mare` is an alias inside the builders). Interactive
dashboards: `outputs/_distribution/html_dashboards/<site>/index.html`, shared
`style.css`/`js/`/`shared/` at the directory root. Playwright chromium is
installed under `~/.cache/ms-playwright`; `from playwright.sync_api import
sync_playwright` works in this python.

## Deliverables

1. Fix the round-1 findings on the A3 sheet in `build_site_dashboard.py` —
   presentation only, no data or analysis change. Every blocker/major fixed;
   minors where cheap.
2. `--all` on both builders: loop the five sites, continue on error, print one
   summary line per site, exit non-zero if any site failed.
3. `scripts/capture_dashboards.py`: serve `outputs/_distribution/html_dashboards`
   on 127.0.0.1 (free port, `http.server` in a thread); for the landing page
   and each site capture a full-page PNG at desktop 1440×900 and tablet
   820×1180 after network-idle + 1.5 s; record console errors and failed
   requests (status ≥ 400); write
   `outputs/_distribution/audit/dashboards_<UTC>/{page}_{viewport}.png` and
   `report.json`; finish by tiling the desktop captures with
   `scripts/critic_sheet.py sheet`.
4. Rebuild all five A3 sheets (`--all`) and all five interactive dashboards
   (`--all`) into the main tree; run `capture_dashboards.py`; tile the five
   `folha_<site>_web1200.png` with `critic_sheet.py sheet --cols 5`. Read both
   sheets yourself and iterate at least once on what you see.
5. `tests/test_site_dashboard.py` — pure-function tests, no real outputs:
   `SHEET_NUMBER`, `TYPOLOGY`, `SITE_DISPLAY` cover the five sites; both
   builders accept `--all` (`--help` via subprocess is fine); one test on a
   pure helper you touched. `pytest tests/test_site_dashboard.py` must
   COLLECT tests — exit 5 ("no tests ran") is a failure, not a pass.
6. Do not change SVF/solar values, the observer network, or the audit
   findings' wording (H1–H4, M2, M3/L1); build date and sha are automatic.

## Gate (unpiped, one per line)

```
TMPDIR=/tmp python -m pytest tests/test_site_dashboard.py -q
TMPDIR=/tmp python scripts/build_site_dashboard.py --site maré
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

## Never

`git add -A`; anything under `outputs/` into git; edits to
`docs/technical_report/`; hand-editing generated HTML; new analyses;
`docs/critic/*` and `scripts/critic_sheet.py` are read-only for you.

Final message: each round-1 finding → fixed / deferred + one line why; file
list; commit hashes; gate output tails; the capture `report.json` path and
both contact-sheet paths; console errors / failed requests count per page.
