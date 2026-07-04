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


def test_generated_hub_blank_cards_all_have_noopener():
    html = _generated_hub()
    for a in re.findall(r'<a class="card"[^>]*target="_blank"[^>]*>', html):
        assert 'rel="noopener"' in a


def test_generated_hub_has_no_root_absolute_urls():
    # file:// portability: every href/src (and lightbox zoom target) is relative
    html = _generated_hub()
    assert not re.search(r'(href|src)="/[^"]*"', html)
    assert "zoom('/" not in html


def test_generated_hub_relative_targets_resolve_on_disk():
    _generated_hub()
    out = bph.OUT
    text = (out / "index.html").read_text()
    for url in re.findall(r'(?:href|src)="(\.\./[^"#]*)"', text):
        assert (out / url).exists(), f"relative target {url} does not resolve"


def test_generated_hub_latest_has_no_onpage_section_duplicate_links():
    html = _generated_hub()
    m = re.search(r'<div class="callout">.*?</ul>', html, re.DOTALL)
    assert m, "callout not found"
    # the changelog must not re-navigate to a section that sits on the same page
    assert not re.search(r'href="#[a-z-]+"', m.group(0))


def test_latest_item_renders_date_prefix():
    li = bph._render_latest_item("X", "#x", "d", date="2026-07-02")
    assert '<span class="date">2026-07-02</span>' in li


# ── planetary-health section: exposure-not-outcome discipline is load-bearing ──
def _health_page():
    bph.main()
    return (bph.OUT / "health.html").read_text()


def test_health_page_leads_with_exposure_not_outcome_disclaimer():
    html = _health_page()
    # the skeptic's disclaimer must head the page and disclaim measured outcomes
    assert "not measured" in html and "health outcomes" in html
    assert html.find("not measured") < html.find('id="health-pathways"')


def test_health_page_carries_all_four_evidence_grades():
    html = _health_page()
    for g in ("Grade A", "Grade B", "Grade C", "Grade D"):
        assert g in html, f"{g} missing — pathways must be evidence-graded"


def test_health_table_binds_to_real_cross_site_shares():
    """The compound-deprivation table is read from cross_site_stats.json, not
    hardcoded, so it can never silently drift from the taxonomy it summarises."""
    import json
    js = bph.ROOT / "outputs/paper_figures/cross_site_stats.json"
    if not js.exists():
        return  # gitignored data absent in this checkout; page degrades gracefully
    per = json.loads(js.read_text())["per_site"]
    worst = max(r["shares"]["compound_constraint"] for r in per)
    assert f'{worst*100:.0f}%' in _health_page()


def test_health_page_has_no_root_absolute_urls():
    # same file:// portability contract as the index page
    html = _health_page()
    assert not re.search(r'(href|src)="/[^"]*"', html)


def test_hub_index_exposes_health_section():
    html = (bph.OUT / "index.html").read_text() if (bph.OUT / "index.html").exists() \
        else bph.main() or (bph.OUT / "index.html").read_text()
    assert 'id="health"' in html
    assert 0 < html.find('id="facade-solar"') < html.find('id="health"')
