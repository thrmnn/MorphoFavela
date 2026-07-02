"""Invariants for the project-hub generator (scripts/build_project_hub.py).

Locks the 'Latest' callout guarantees the council flagged: HTML-escaped text
(no raw & reaching the page), a real NEW pill instead of a '— NEW' text suffix,
and no hand-inlined style= (all styling flows through the hubkit token system).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_project_hub as bph  # noqa: E402


def test_latest_item_escapes_ampersand_in_label_and_gloss():
    li = bph._render_latest_item("Typology & Signature", "#x", "a & b caption")
    assert "Typology &amp; Signature" in li
    assert "a &amp; b caption" in li
    assert " & " not in li


def test_latest_item_new_suffix_becomes_pill_and_is_stripped():
    li = bph._render_latest_item("MAUP curve (§10.9) — NEW", "#maup", "sweep")
    assert '<span class="pill info">NEW</span>' in li
    assert "NEW</a>" not in li  # the suffix is not left inside the link text
    assert "MAUP curve (§10.9)" in li


def test_latest_item_without_new_has_no_pill():
    li = bph._render_latest_item("Roughness caveat", "#r", "envelope")
    assert "pill" not in li


def test_latest_item_has_no_inline_style():
    li = bph._render_latest_item("X — NEW", "#x", "d")
    assert "style=" not in li


def test_latest_item_preserves_href():
    li = bph._render_latest_item("X", "/outputs/_hub/docs/x.html#a", "d")
    assert 'href="/outputs/_hub/docs/x.html#a"' in li
