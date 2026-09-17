#!/usr/bin/env python3
"""Reachability + link-integrity gate for the MorphoFavela project hub
(outputs/_hub/). HUBWP Phase 1 (docs/hub_wp_structure_spec.md).

2026-09-17: the hub had 33 pages and the front door (index.html) mentioned
none of `runs/`, `ledger`, `WP-0`, or `wp07_staged`. Nothing checked whether a
card actually led anywhere real — this script is that check, adapted from the
same fix applied to the brisaverse hub the same night
(brisaverse/hub/lint_pages.py, `docs/critic/nav_audit_2026-09-16.md`): a check
that tests only "does the href resolve" is a dangling-link check wearing the
name of a reachability check. This one walks the graph.

Two things it verifies, both gates:

1. REACHABILITY — breadth-first from outputs/_hub/index.html, expanding only
   NAVIGABLE edges: an anchor inside a `.card` or inside a `<nav>...</nav>`
   block (breadcrumb, sidebar toc). An anchor sitting in a running `<p>` (or
   anywhere else outside those two containers) is not navigable — it can give
   a page an inbound link without making it reachable, which is exactly the
   PROSE-ONLY failure mode from the brisaverse incident. Every `*.html` file
   under outputs/_hub/ is classified as one of:
     REACHABLE    — reached by the BFS; reported with its depth.
     PROSE-ONLY    — has an inbound link, but never a navigable one.
     ORPHAN        — no inbound link at all, navigable or not.
   A directory target (href ending "/", or resolving to a directory) is
   treated as that directory's index.html.

2. LINK INTEGRITY — every non-external href/src (and lightbox `zoom()`
   target) in every page under outputs/_hub/ must resolve to a file that
   exists on disk. A dangling link is a functional failure, reported
   separately from the reachability classes above.

Exit 1 on any ORPHAN, PROSE-ONLY page, or dangling link. Exit 0 otherwise.
The printed map (every page, sorted by depth) IS the audit mapping the PI
asked for — it is also the input `scripts/build_project_hub.py` uses to
generate outputs/_hub/map.html, so the durable map can never drift from what
this check actually walked.

Run:
    python3 scripts/audit_hub_graph.py
    python3 scripts/audit_hub_graph.py --root /path/to/repo   # e.g. a copy,
                                                               # for the red-probe
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent

import re  # noqa: E402

_EXTERNAL = ("http:", "https:", "mailto:", "tel:", "javascript:")
_A = re.compile(r'<a\b[^>]*href="([^"]*)"[^>]*>', re.I)
_ZOOM = re.compile(r"zoom\('([^']*)'", re.I)
_CARD_CLASS = re.compile(r'class="[^"]*\bcard\b', re.I)


def _resolve(page: Path, href: str) -> Path:
    """Resolve an href relative to the page that carries it. A directory
    target (explicit trailing slash, or a target that turns out to be a
    directory on disk) resolves to that directory's index.html."""
    target = (page.parent / href).resolve() if href else page
    if href.endswith("/"):
        target = target / "index.html"
    elif target.is_dir():
        target = target / "index.html"
    return target


def _in_nav(before: str) -> bool:
    """True if the nearest enclosing <nav>/</nav> pair (by last-opened,
    not-yet-closed <nav>) contains the anchor at the end of `before`."""
    last_open = before.rfind("<nav")
    last_close = before.rfind("</nav>")
    return last_open != -1 and last_open > last_close


@dataclass
class PageLinks:
    navigable: set[Path] = field(default_factory=set)   # -> other pages, card/nav only
    all_targets: list[tuple[Path, str]] = field(default_factory=list)  # (target, raw href)


def _parse_links(page: Path) -> PageLinks:
    text = page.read_text(errors="replace")
    out = PageLinks()
    for m in _A.finditer(text):
        tag, href = m.group(0), m.group(1)
        base, _, _frag = href.partition("#")
        if not base or base.startswith(_EXTERNAL):
            continue
        target = _resolve(page, base)
        target = Path(unquote(str(target)))  # hrefs percent-encode 'maré'; the disk does not
        out.all_targets.append((target, base))
        navigable = bool(_CARD_CLASS.search(tag)) or _in_nav(text[: m.start()])
        if navigable:
            out.navigable.add(target)
    for m in _ZOOM.finditer(text):
        base, _, _frag = m.group(1).partition("#")
        if base and not base.startswith(_EXTERNAL):
            out.all_targets.append((_resolve(page, base), base))
    return out


@dataclass
class AuditResult:
    pages: list[Path]              # every *.html under hub, sorted
    depth: dict[Path, int]         # reachable pages -> BFS depth
    prose_only: list[Path]
    orphan: list[Path]
    dangling: list[tuple[Path, str]]   # (source page, raw href) whose target is missing

    @property
    def ok(self) -> bool:
        return not (self.prose_only or self.orphan or self.dangling)


def audit(hub: Path) -> AuditResult:
    index = hub / "index.html"
    pages = sorted(hub.rglob("*.html"))
    links = {p: _parse_links(p) for p in pages}

    depth: dict[Path, int] = {}
    dangling: list[tuple[Path, str]] = []
    inbound: set[Path] = set()
    page_set = set(pages)

    for p, pl in links.items():
        for target, raw in pl.all_targets:
            if not target.exists():
                dangling.append((p, raw))
            elif target in page_set:
                inbound.add(target)

    if index.exists():
        depth[index] = 0
        q = deque([index])
        while q:
            cur = q.popleft()
            for nxt in sorted(links.get(cur, PageLinks()).navigable):
                if nxt not in page_set or nxt in depth:
                    continue
                depth[nxt] = depth[cur] + 1
                q.append(nxt)

    unreached = [p for p in pages if p not in depth]
    prose_only = sorted(p for p in unreached if p in inbound)
    orphan = sorted(p for p in unreached if p not in inbound)
    return AuditResult(pages=pages, depth=depth, prose_only=prose_only,
                       orphan=orphan, dangling=dangling)


def _fmt(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def report(result: AuditResult, root: Path) -> None:
    by_depth = sorted(result.depth.items(), key=lambda kv: (kv[1], _fmt(kv[0], root)))
    print(f"hub graph — {len(result.pages)} pages under outputs/_hub/")
    print(f"  reachable:  {len(result.depth)}")
    print(f"  prose-only: {len(result.prose_only)}")
    print(f"  orphan:     {len(result.orphan)}")
    print(f"  dangling:   {len(result.dangling)}")
    print()
    print("REACHABLE (sorted by depth):")
    for p, d in by_depth:
        print(f"  [{d}] {_fmt(p, root)}")
    if result.prose_only:
        print("\nPROSE-ONLY (linked, but never from a card or nav — invisible in practice):")
        for p in result.prose_only:
            print(f"  {_fmt(p, root)}")
    if result.orphan:
        print("\nORPHAN (no inbound link at all):")
        for p in result.orphan:
            print(f"  {_fmt(p, root)}")
    if result.dangling:
        print("\nDANGLING LINKS (href/src/zoom target does not resolve on disk):")
        for src, raw in result.dangling:
            print(f"  {_fmt(src, root)} -> {raw}")
    if not result.ok:
        print("\nFix each: link it from a card or a <nav> block on a page that is "
              "itself reachable, or fix the dangling target. Nothing here is "
              "quietly exempt.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="repo root whose outputs/_hub/ to audit "
                         "(default: this repo)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    hub = root / "outputs" / "_hub"
    if not hub.exists():
        print(f"FAIL: {hub} does not exist — run scripts/build_project_hub.py first",
              file=sys.stderr)
        return 1
    result = audit(hub)
    report(result, root)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
