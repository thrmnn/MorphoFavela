"""Capture the interactive Folha de Rua dashboards for the critic loop.

Serves outputs/_distribution/html_dashboards on 127.0.0.1 (a free port, via
http.server in a thread — no external server needed) and screenshots the
landing page plus all five site dashboards at desktop (1440x900) and tablet
(820x1180), after network-idle + 1.5 s settle. Records console errors and
failed requests (status >= 400) per page so a page that "looks fine" but
throws JS errors or 404s its data still shows up in the report.

Pattern copied from /home/theo/SCL/SCR/brisaverse/hub/ux/capture.py
(measure-then-screenshot ordering, append-and-flush per page) and trimmed
to what this refresh needs — no perf/tap-target instrumentation, no warm
cache pass.

Run:
    python scripts/capture_dashboards.py
"""

from __future__ import annotations

import functools
import http.server
import json
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")
# Serve _distribution, not html_dashboards/ itself: each per-site page links
# its A3 print fallback via a relative "../../site_dashboards/<site>/..."
# (see build_html_dashboard.py's print-only <img>), which only resolves if
# html_dashboards/ and site_dashboards/ are both reachable as siblings
# under the served root — exactly how the page is actually opened (file://
# from its real on-disk location). Serving html_dashboards/ alone clamps
# that ".." above the root and reports a false-positive 404.
DASH_DIR = ROOT / "outputs" / "_distribution"

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]

# (name, route) — landing page first, then each site's index.
PAGES = [("landing", "/html_dashboards/")] + [(s, f"/html_dashboards/{s}/") for s in SITES]

VIEWPORTS = {
    "desktop": {"width": 1440, "height": 900},
    "tablet": {"width": 820, "height": 1180},
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(directory: Path, port: int) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def capture(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    httpd = _serve(DASH_DIR, port)
    base = f"http://127.0.0.1:{port}"

    report = {
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base": base,
        "served_from": str(DASH_DIR),
        "pages": [],
    }

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for vp_name, vp in VIEWPORTS.items():
                ctx = browser.new_context(viewport={"width": vp["width"], "height": vp["height"]})
                for name, route in PAGES:
                    page = ctx.new_page()
                    console_errors: list[str] = []
                    failed: list[str] = []
                    page.on(
                        "console",
                        lambda m: console_errors.append(f"{m.type}: {m.text}") if m.type == "error" else None,
                    )
                    page.on(
                        "response",
                        lambda r: failed.append(f"{r.status} {r.url}") if r.status >= 400 else None,
                    )
                    page.on("requestfailed", lambda r: failed.append(f"NETFAIL {r.method} {r.url} — {r.failure}"))

                    status = None
                    try:
                        resp = page.goto(base + route, wait_until="networkidle", timeout=20000)
                        page.wait_for_timeout(1500)
                        status = resp.status if resp else None
                    except Exception as exc:
                        failed.append(f"navigation: {type(exc).__name__}: {exc}")

                    shot = out_dir / f"{name}_{vp_name}.png"
                    try:
                        page.screenshot(path=str(shot), full_page=True, animations="disabled", timeout=45000)
                    except Exception as exc:
                        failed.append(f"screenshot: {exc}")

                    entry = {
                        "page": name, "route": route, "viewport": vp_name,
                        "http_status": status,
                        "console_errors": console_errors,
                        "failed_requests": failed,
                        "screenshot": shot.name,
                    }
                    report["pages"].append(entry)
                    (out_dir / "report.partial.json").write_text(json.dumps(report, indent=1))
                    page.close()
                ctx.close()
            browser.close()
    finally:
        httpd.shutdown()

    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "outputs" / "_distribution" / "audit" / f"dashboards_{stamp}"
    report = capture(out_dir)

    bad = [pg for pg in report["pages"] if pg["console_errors"] or pg["failed_requests"] or pg["http_status"] != 200]
    print(f"captured {len(report['pages'])} page-viewport pairs -> {out_dir}")
    print(f"pages with issues: {len(bad)}")
    for pg in bad:
        print(f"  {pg['viewport']}/{pg['page']}: http={pg['http_status']} "
              f"console={len(pg['console_errors'])} failed={len(pg['failed_requests'])}")
        for e in (pg["console_errors"] + pg["failed_requests"])[:3]:
            print(f"      {e[:150]}")

    desktop_pngs = [out_dir / f"{name}_desktop.png" for name, _ in PAGES]
    sheet_out = out_dir / "contact_sheet_dashboards.png"
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "critic_sheet.py"), "sheet",
         str(sheet_out), *[str(p) for p in desktop_pngs if p.exists()], "--cols", "3"],
        check=True,
    )
    print(f"report: {out_dir / 'report.json'}")
    print(f"contact sheet: {sheet_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
