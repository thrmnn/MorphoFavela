"""Invariants for the project-hub generator (scripts/build_project_hub.py).

Locks the 'Latest' callout guarantees the council flagged: HTML-escaped text
(no raw & reaching the page), a real NEW pill instead of a '— NEW' text suffix,
and no hand-inlined style= (all styling flows through the hubkit token system).
"""

import re
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


def test_headline_section_heros_the_money_figure():
    html = bph.headline_section(None)
    assert 'id="headline"' in html
    assert "typology_failure_lookup.png" in html
    assert "<img" in html  # rendered as a zoomable image card, not a text link
    assert "14 % → 73 %" in html


def test_headline_figs_reference_real_gallery_anchors():
    gallery = (bph.ROOT / "outputs/cross_site/signature/figures_v2/index.html")
    if not gallery.exists():
        return  # gallery not built in this checkout; skip
    text = gallery.read_text()
    for fn, anchor, _title, _desc in bph.HEADLINE_FIGS:
        if (bph.ROOT / f"outputs/cross_site/signature/figures_v2/{fn}").exists():
            assert f'id="{anchor}"' in text, f"{anchor} missing from gallery"


# ── golden: regenerate the whole hub and assert page-level invariants (M11) ──
def _generated_hub():
    bph.main()
    return (bph.OUT / "index.html").read_text()


def test_generated_hub_has_no_raw_ampersand():
    assert " & " not in _generated_hub()


def test_generated_hub_all_images_have_alt():
    for img in re.findall(r"<img[^>]*>", _generated_hub()):
        assert "alt=" in img


def test_generated_hub_sidebar_labels_are_human_not_slugs():
    html = _generated_hub()
    labels = re.findall(r'<a class="t2" href="#[a-z-]+">([^<]+)</a>', html)
    assert labels
    for slug in ("Maup", "Facade-solar", "Figure", "Headline"):
        assert slug not in labels, f"machine slug {slug!r} leaked into the sidebar"


def test_generated_hub_every_sidebar_anchor_resolves():
    html = _generated_hub()
    for anchor in re.findall(r'<a class="t2" href="#([a-z-]+)">', html):
        assert f'id="{anchor}"' in html, f"sidebar #{anchor} has no section"


def test_generated_hub_headline_precedes_facade_solar():
    html = _generated_hub()
    assert 0 < html.find('id="headline"') < html.find('id="facade-solar"')


def test_generated_hub_has_no_enum_echo_pill():
    html = _generated_hub()
    for kind in ("ok", "info", "doc", "warn", "amber"):
        assert f'<span class="pill {kind}">{kind}</span>' not in html
