#!/usr/bin/env python3
"""check_page_contrast.py — WCAG contrast gate on a RENDERED hub page.

Unlike hub/check_contrast.py (which parses design.css token hex values
statically), this renders the actual HTML file with Playwright/Chromium under
a given color-scheme preference and reads the computed styles the browser
produced — so it catches a component rule that hardcodes a colour and
silently wins the cascade over a --token, not just a bad token definition.

For every element matching a fixed selector list (p, li, td, th, code,
summary, .callout, .pill, a, h1, h2, h3) it computes the WCAG 2.1 contrast
ratio between the element's computed `color` and its nearest ancestor's
non-transparent `background-color` (walking up from the element itself,
since an element can paint its own background), and fails any pair under
4.5:1 (3:1 for text >= 18.66px that is also bold, per WCAG 1.4.3 large-text).

Usage:
    python3 scripts/check_page_contrast.py PAGE.html --scheme dark
    python3 scripts/check_page_contrast.py PAGE.html --scheme light
    python3 scripts/check_page_contrast.py --self-test

Deterministic; no LLM. Exit 0 on pass, 1 on any failure (or on --self-test
not proving both directions).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

SELECTORS = ["p", "li", "td", "th", "code", "summary", ".callout", ".pill",
             "a", "h1", "h2", "h3"]

# Runs in-page. Walks each matched element's own ancestor chain (starting at
# the element itself, since a component can paint its own background) for the
# first non-transparent background-color, parses both colors' rgba() computed
# strings, and applies the WCAG contrast formula + large-text threshold.
_JS = r"""
(selectors) => {
  function parseColor(s) {
    const m = s.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const parts = m[1].split(',').map(x => parseFloat(x.trim()));
    return {r: parts[0], g: parts[1], b: parts[2], a: parts.length > 3 ? parts[3] : 1};
  }
  function isTransparent(c) {
    return !c || c.a === 0;
  }
  function lum(c) {
    const chan = [c.r, c.g, c.b].map(v => {
      const s = v / 255;
      return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2];
  }
  function contrast(a, b) {
    const la = lum(a), lb = lum(b);
    const hi = Math.max(la, lb), lo = Math.min(la, lb);
    return (hi + 0.05) / (lo + 0.05);
  }
  function bgFor(el) {
    let node = el;
    while (node) {
      const cs = getComputedStyle(node);
      const c = parseColor(cs.backgroundColor);
      if (!isTransparent(c)) return c;
      node = node.parentElement;
    }
    return {r: 255, g: 255, b: 255, a: 1};
  }
  const results = [];
  const seen = new Set();
  for (const sel of selectors) {
    document.querySelectorAll(sel).forEach(el => {
      const text = (el.textContent || '').trim();
      if (!text) return;
      if (seen.has(el)) return;
      seen.add(el);
      const cs = getComputedStyle(el);
      const fg = parseColor(cs.color);
      if (!fg) return;
      const bg = bgFor(el);
      const ratio = contrast(fg, bg);
      const size = parseFloat(cs.fontSize);
      const weight = parseInt(cs.fontWeight, 10) || 400;
      const isBoldLarge = size >= 18.66 && weight >= 700;
      const isLarge = size >= 24;
      const floor = (isBoldLarge || isLarge) ? 3.0 : 4.5;
      results.push({
        sel, ratio, floor,
        text: text.slice(0, 50),
        fg: cs.color, bg: `rgba(${bg.r},${bg.g},${bg.b},${bg.a})`,
      });
    });
  }
  return results;
}
"""


def run_check(html_path: Path, scheme: str) -> list[dict]:
    from playwright.sync_api import sync_playwright

    url = html_path.resolve().as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(color_scheme=scheme)
        page.goto(url)
        results = page.evaluate(_JS, SELECTORS)
        browser.close()
    return results


def check(html_path: Path, scheme: str, *, quiet: bool = False) -> bool:
    results = run_check(html_path, scheme)
    fails = [r for r in results if r["ratio"] < r["floor"]]
    if not quiet:
        print("=" * 64)
        print(f"CONTRAST ({scheme}): {len(results)} element(s) checked")
        for r in fails:
            print(f"  ✗ {r['sel']} {r['ratio']:.2f}:1 (< {r['floor']}) "
                  f"fg={r['fg']} bg={r['bg']} text={r['text']!r}")
        if not fails:
            print(f"  all {len(results)} pairs OK ✓")
        print("=" * 64)
    return not fails


def _self_test() -> int:
    """(a) a tiny page built through hubkit passes in both schemes.
    (b) injecting style="color:#2a2d31" (dark ink on a light-mode-only fg)
    on a paragraph under the dark scheme must turn the check RED."""
    import scripts.hubkit as hubkit  # noqa: E402  (path set up by caller)

    body = (
        '<h1>Sample</h1><h2>Section</h2><h3>Sub</h3>'
        '<p>A body paragraph with enough words to be real text.</p>'
        '<ul><li>A list item</li></ul>'
        '<table><thead><tr><th>Col</th></tr></thead>'
        '<tbody><tr><td>Cell</td></tr></tbody></table>'
        '<p><code>inline_code()</code></p>'
        '<details><summary>Expand me</summary><p>hidden</p></details>'
        '<div class="callout"><p class="lead">Callout lead</p></div>'
        '<span class="pill ok">OK</span><span class="pill warn">WARN</span>'
        '<span class="pill amber">AMBER</span><span class="pill info">INFO</span>'
        '<span class="pill doc">DOC</span>'
        '<p><a href="#">A link</a></p>'
    )
    html = hubkit.page("Self-test", "contrast self-test", body, doc=True)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        clean = tdp / "clean.html"
        clean.write_text(html)

        ok_light = check(clean, "light", quiet=True)
        ok_dark = check(clean, "dark", quiet=True)
        print(f"self-test (a) clean page: light={'PASS' if ok_light else 'FAIL'} "
              f"dark={'PASS' if ok_dark else 'FAIL'}")
        if not (ok_light and ok_dark):
            print("self-test (a) FAILED — a clean hubkit page should pass both schemes")
            return 1

        sabotaged = html.replace(
            "<p>A body paragraph with enough words to be real text.</p>",
            '<p style="color:#2a2d31">A body paragraph with enough words to be real text.</p>',
            1,
        )
        sab = tdp / "sabotaged.html"
        sab.write_text(sabotaged)
        ok_sabotaged_dark = check(sab, "dark", quiet=True)
        print(f"self-test (b) sabotaged page under dark scheme: "
              f"{'PASS (BUG: should be RED)' if ok_sabotaged_dark else 'RED as expected'}")
        if ok_sabotaged_dark:
            print("self-test (b) FAILED — sabotage should have gone RED and did not")
            return 1

    print("self-test: PASSED (clean page green in both schemes; sabotage goes red)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("html", nargs="?", type=Path, help="rendered HTML file to check")
    ap.add_argument("--scheme", choices=["light", "dark"], help="color-scheme preference")
    ap.add_argument("--self-test", action="store_true",
                     help="run the built-in self-test instead of checking a file")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if not args.html or not args.scheme:
        ap.error("HTML file and --scheme are required unless --self-test is passed")
    if not args.html.exists():
        ap.error(f"no such file: {args.html}")

    ok = check(args.html, args.scheme)
    return 0 if ok else 1


if __name__ == "__main__":
    # Allow `import scripts.hubkit` from the self-test regardless of cwd: add
    # the repo root (parent of this file's parent) to sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
