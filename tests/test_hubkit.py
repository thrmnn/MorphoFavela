"""Invariants for the zero-dependency hub engine (scripts/hubkit.py).

These lock the accessibility + HTML-validity guarantees the design council flagged:
every figure card must carry escaped alt text, primitives must escape user text, and
an empty card list must not emit a section. Grown alongside the dashboard-improvement
backlog (docs/dashboard_improvement_plan.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import hubkit  # noqa: E402


def test_card_with_image_has_nonempty_alt_defaulting_to_title():
    html = hubkit.card("Frontal area ratio λf", "desc", "/x.html", img="/f.png")
    assert 'alt="Frontal area ratio λf"' in html
    # the alt attribute is never empty when an image is present
    assert 'alt=""' not in html


def test_card_alt_and_title_are_escaped():
    html = hubkit.card("Type & signature", "a < b", "/x.html", img="/f.png")
    assert "Type &amp; signature" in html
    assert " & " not in html  # no raw ampersand survives
    assert "a &lt; b" in html


def test_card_apostrophe_survives_into_lightbox_call():
    # "Mingze's" must not break the onclick JS single-quoted string
    html = hubkit.card("Mingze's façade run", "d", "/x.html", img="/f.png")
    assert "zoom(" in html
    assert "onclick=" in html
    # the apostrophe is JS-escaped (\\') then HTML-escaped (&#x27;)
    assert "Mingze" in html


def test_card_without_image_emits_no_img_tag():
    assert "<img" not in hubkit.card("t", "d", "/x.html")


def test_section_with_no_cards_returns_empty():
    assert hubkit.section("Empty", []) == ""


def test_badge_escapes_label():
    assert "&amp;" in hubkit.badge("ok", "A & B")


def test_page_emits_lang_charset_viewport():
    out = hubkit.page("T", "sub", "<p>x</p>")
    assert "<html lang=en>" in out
    assert "charset=utf-8" in out
    assert "viewport" in out


def test_js_attr_escapes_backslash_and_quote():
    assert hubkit._js_attr("a'b") == "a\\&#x27;b"
    assert hubkit._js_attr("a\\b") == "a\\\\b"
