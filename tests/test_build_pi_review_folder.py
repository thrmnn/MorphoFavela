"""Invariants for the PI review-folder generator
(scripts/build_pi_review_folder.py), per the navigation council ruling
(docs/critic/navigation_council_2026-09-24.md), Phase 1.

G1 — rendered order (TOC and body) must equal sorted(order), never slug
position. Also covers the "New this cycle" diff, markdown/CSV rendering
fallbacks, and dated-folder retention.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_pi_review_folder as bprf  # noqa: E402


# --------------------------------------------------------------------------
# G1 — order is the only thing that decides render position
# --------------------------------------------------------------------------

def test_assert_ordered_passes_when_already_sorted():
    bprf._assert_ordered([{"order": 1}, {"order": 2}, {"order": 3}])  # no raise


def test_assert_ordered_raises_when_a_00_slug_jumps_the_queue():
    """The exact G1 red proof from the ruling: a '00_' slug given order=99
    must NOT be allowed to render first just because it sorts first
    alphabetically."""
    sections = [
        {"slug": "00_mare_territory", "title": "Z", "order": 99},  # rendered first (wrong)
        {"slug": "a_later_thing", "title": "A", "order": 2},
    ]
    with pytest.raises(AssertionError):
        bprf._assert_ordered(sections)


def test_render_index_raises_when_manifest_sections_are_out_of_order():
    """_render_index itself asserts (not only build()) — a future refactor
    that re-sorts by something else in the render step is still caught."""
    m = {
        "_utc": "2026-09-24T00:00:00Z", "cycle_date": "2026-09-24",
        "sections": [
            {"slug": "00_z", "title": "Z", "blurb": "", "provenance": "", "order": 99},
            {"slug": "a", "title": "A", "blurb": "", "provenance": "", "order": 2},
        ],
        "files": [
            {"section": "00_z", "file": "z.png", "status": "ok", "bytes": 1, "src_mtime_utc": "2026-09-24T00:00:00Z"},
            {"section": "a", "file": "a.png", "status": "ok", "bytes": 1, "src_mtime_utc": "2026-09-24T00:00:00Z"},
        ],
        "new_since": {"cutoff_utc": None, "families": [], "total": 0, "shown": 0},
    }
    with pytest.raises(AssertionError):
        bprf._render_index(m)


def test_render_index_orders_toc_and_body_by_declared_order():
    """Given sections already in the order build() must produce them in (sorted
    by their own "order" field), the render step places both the TOC and the
    body in that same order — it never falls back to slug or dict position."""
    m = {
        "_utc": "2026-09-24T00:00:00Z", "cycle_date": "2026-09-24",
        "sections": [
            {"slug": "zz_first", "title": "First Section", "blurb": "", "provenance": "", "order": 1},
            {"slug": "aa_second", "title": "Second Section", "blurb": "", "provenance": "", "order": 2},
        ],
        "files": [
            {"section": "zz_first", "file": "a.png", "status": "ok", "bytes": 1, "src_mtime_utc": "2026-09-24T00:00:00Z"},
            {"section": "aa_second", "file": "b.png", "status": "ok", "bytes": 1, "src_mtime_utc": "2026-09-24T00:00:00Z"},
        ],
        "new_since": {"cutoff_utc": None, "families": [], "total": 0, "shown": 0},
    }
    html = bprf._render_index(m)
    # Slugs are picked to sort the WRONG way alphabetically ("aa_" < "zz_"),
    # so this only passes if rendering follows "order", not the slug string.
    assert html.index("First Section") < html.index("Second Section")
    assert html.index('id="s-zz_first"') < html.index('id="s-aa_second"')


def test_records_have_unique_positive_int_orders():
    orders = [r["order"] for r in bprf.RECORDS]
    assert len(orders) == len(set(orders))
    assert all(isinstance(o, int) and o > 0 for o in orders)


def test_mare_and_octopus_records_added_at_orders_1_and_2():
    by_order = {r["order"]: r for r in bprf.RECORDS}
    assert "Maré" in by_order[1]["title"] or "Maré" in by_order[1]["title"]
    assert "territory" in by_order[1]["title"].lower()
    assert "octopus" in by_order[2]["title"].lower()
    assert "om2" in by_order[2]["title"].lower()
    assert by_order[2].get("badge") == "internal review draft — Octopus team only"


# --------------------------------------------------------------------------
# New this cycle
# --------------------------------------------------------------------------

def test_new_since_none_when_no_previous_cycle():
    out = bprf._compute_new_since([], [], None)
    assert out == {"cutoff_utc": None, "families": [], "total": 0, "shown": 0}


def test_new_since_groups_by_family_in_section_order_and_ignores_sweep():
    sections = [
        {"slug": "fam_a", "title": "Family A", "order": 1},
        {"slug": "fam_b", "title": "Family B", "order": 2},
    ]
    prev = "2026-09-17T00:00:00Z"
    entries = [
        {"section": "fam_a", "file": "old.png", "status": "ok", "src_mtime_utc": "2026-09-10T00:00:00Z"},
        {"section": "fam_a", "file": "new.png", "status": "ok", "src_mtime_utc": "2026-09-24T00:00:00Z"},
        {"section": "fam_b", "file": "new2.png", "status": "ok", "src_mtime_utc": "2026-09-20T00:00:00Z"},
        {"section": "sweep/misc", "file": "sweep.png", "status": "ok", "src_mtime_utc": "2026-09-24T00:00:00Z"},
    ]
    out = bprf._compute_new_since(entries, sections, prev)
    assert out["cutoff_utc"] == prev
    assert out["total"] == 2  # sweep entry never counts
    slugs_with_items = [f["slug"] for f in out["families"] if f["items"]]
    assert slugs_with_items == ["fam_a", "fam_b"]  # section-declared order


def test_new_since_caps_total_thumbnails_shown():
    sections = [{"slug": "fam", "title": "Family", "order": 1}]
    prev = "2026-09-17T00:00:00Z"
    entries = [
        {"section": "fam", "file": f"f{i}.png", "status": "ok", "src_mtime_utc": "2026-09-24T00:00:00Z"}
        for i in range(bprf.NEW_SINCE_CAP + 5)
    ]
    out = bprf._compute_new_since(entries, sections, prev)
    assert out["total"] == bprf.NEW_SINCE_CAP + 5
    assert out["shown"] == bprf.NEW_SINCE_CAP
    assert sum(len(f["items"]) for f in out["families"]) == bprf.NEW_SINCE_CAP


def test_render_new_since_links_to_toc_when_truncated():
    new_since = {
        "cutoff_utc": "2026-09-17T00:00:00Z", "total": 35, "shown": 30,
        "families": [{"slug": "fam", "title": "Family", "n": 35, "items": [
            {"section": "fam", "file": "f.png", "status": "ok", "bytes": 1} for _ in range(30)
        ]}],
    }
    html = bprf._render_new_since(new_since)
    assert 'id="s-new"' in html
    assert "+5 more this cycle" in html
    assert '#s-toc' in html


def test_render_index_stamp_shows_as_of_cycle_and_new_count():
    m = {
        "_utc": "2026-09-24T17:46:39Z", "cycle_date": "2026-09-24",
        "sections": [{"slug": "a", "title": "A", "blurb": "", "provenance": "", "order": 1}],
        "files": [{"section": "a", "file": "a.png", "status": "ok", "bytes": 1, "src_mtime_utc": "2026-09-24T00:00:00Z"}],
        "new_since": {"cutoff_utc": "2026-09-17T15:32:22Z", "total": 3, "shown": 3, "families": []},
    }
    html = bprf._render_index(m)
    assert "AS OF 2026-09-24T17:46:39Z" in html
    assert "cycle 2026-09-24" in html
    assert "3 new since last cycle" in html
    assert "2026-09-17T15:32:22Z" in html


# --------------------------------------------------------------------------
# "Mark this cycle reviewed" (navigation council ruling §5 phase 5)
# --------------------------------------------------------------------------

def test_render_index_includes_mark_reviewed_button_posting_review_mark():
    m = {
        "_utc": "2026-09-24T00:00:00Z", "cycle_date": "2026-09-24",
        "sections": [{"slug": "a", "title": "A", "blurb": "", "provenance": "", "order": 1}],
        "files": [{"section": "a", "file": "a.png", "status": "ok", "bytes": 1,
                   "src_mtime_utc": "2026-09-24T00:00:00Z"}],
        "new_since": {"cutoff_utc": None, "total": 0, "shown": 0, "families": []},
    }
    html = bprf._render_index(m)
    assert 'id="mark-reviewed"' in html
    assert "Mark this cycle reviewed" in html
    # Relative URL — this page's own origin becomes whatever mirrors it
    # (brisaverse's hub at /morphofavela-dash/…), never a hardcoded host.
    assert "fetch('/api/review-mark'" in html
    assert "http://" not in html and "https://" not in html


def test_mark_reviewed_button_appears_before_the_toc_and_after_the_stamp():
    m = {
        "_utc": "2026-09-24T00:00:00Z", "cycle_date": "2026-09-24",
        "sections": [{"slug": "a", "title": "A", "blurb": "", "provenance": "", "order": 1}],
        "files": [{"section": "a", "file": "a.png", "status": "ok", "bytes": 1,
                   "src_mtime_utc": "2026-09-24T00:00:00Z"}],
        "new_since": {"cutoff_utc": None, "total": 0, "shown": 0, "families": []},
    }
    html = bprf._render_index(m)
    assert html.index("AS OF") < html.index('id="mark-reviewed"') < html.index('id="s-toc"')


def test_render_index_never_leaves_a_dangling_relative_link_from_the_mark_script():
    """The mark-reviewed script's only network call is the relative fetch()
    target /api/review-mark, which is not an href/src the dangling-link
    checker would even see — this pins that it stays that way (no accidental
    <script src=...> or stylesheet link introduced alongside it)."""
    m = {
        "_utc": "2026-09-24T00:00:00Z", "cycle_date": "2026-09-24",
        "sections": [{"slug": "a", "title": "A", "blurb": "", "provenance": "", "order": 1}],
        "files": [{"section": "a", "file": "a.png", "status": "ok", "bytes": 1,
                   "src_mtime_utc": "2026-09-24T00:00:00Z"}],
        "new_since": {"cutoff_utc": None, "total": 0, "shown": 0, "families": []},
    }
    html = bprf._render_index(m)
    import re
    targets = re.findall(r'(?:href|src)="([^"#?:]+)"', html)
    assert not any("api/review-mark" in t for t in targets)


def test_render_index_shows_release_badge_without_hiding_the_section():
    """Release class never hides anything from the PI — it is a badge only."""
    m = {
        "_utc": "2026-09-24T00:00:00Z", "cycle_date": "2026-09-24",
        "sections": [{"slug": "octo", "title": "Octopus pkg", "blurb": "", "provenance": "",
                      "order": 2, "badge": "internal review draft — Octopus team only"}],
        "files": [{"section": "octo", "file": "c.png", "status": "ok", "bytes": 1, "src_mtime_utc": "2026-09-24T00:00:00Z"}],
        "new_since": {"cutoff_utc": None, "total": 0, "shown": 0, "families": []},
    }
    html = bprf._render_index(m)
    assert "internal review draft — Octopus team only" in html
    assert 'id="s-octo"' in html
    assert "c.png" in html  # the figure itself is still shown, not hidden


# --------------------------------------------------------------------------
# Markdown / CSV rendering (README, CHANGELOG, panel ruling, data dictionary)
# --------------------------------------------------------------------------

def test_render_markdown_file_writes_html_with_title_and_content(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("# Heading\n\nSome body text with a special & character.\n")
    dest = tmp_path / "out" / "doc.html"
    ok = bprf._render_markdown_file(src, dest, "My Title")
    assert ok is True
    assert dest.exists()
    html = dest.read_text()
    assert "My Title" in html
    assert "Heading" in html
    assert "body text" in html


def test_render_markdown_file_missing_source_returns_false(tmp_path):
    ok = bprf._render_markdown_file(tmp_path / "nope.md", tmp_path / "out.html", "T")
    assert ok is False


def test_render_markdown_fallback_escapes_html(tmp_path, monkeypatch):
    monkeypatch.setattr(bprf, "_has_pandoc", lambda: False)
    src = tmp_path / "doc.md"
    src.write_text("<script>alert(1)</script>")
    dest = tmp_path / "out.html"
    assert bprf._render_markdown_file(src, dest, "T") is True
    html = dest.read_text()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_csv_table_writes_header_and_rows(tmp_path):
    src = tmp_path / "dict.csv"
    src.write_text("id,definition\nfoo,bar & baz\n")
    dest = tmp_path / "out.html"
    ok = bprf._render_csv_table(src, dest, "Dictionary")
    assert ok is True
    html = dest.read_text()
    assert "<table" in html
    assert "<th>id</th>" in html
    assert "<td>foo</td>" in html
    assert "bar &amp; baz" in html


def test_render_csv_table_missing_source_returns_false(tmp_path):
    assert bprf._render_csv_table(tmp_path / "nope.csv", tmp_path / "out.html", "T") is False


def test_summarize_quality_report_missing_returns_empty_string(tmp_path):
    assert bprf._summarize_quality_report(tmp_path / "nope.json") == ""


def test_summarize_quality_report_renders_coverage_table(tmp_path):
    src = tmp_path / "p07.json"
    src.write_text(json.dumps({
        "n_points": 1559,
        "columns": {"sky_view_factor": {"n_valid": 1532, "n_total": 1559,
                                         "coverage_fraction": 0.9826812059012188}},
    }))
    html = bprf._summarize_quality_report(src)
    assert "1559" in html
    assert "sky_view_factor" in html
    assert "98.3%" in html


# --------------------------------------------------------------------------
# _copy: source mtime capture (feeds "new this cycle")
# --------------------------------------------------------------------------

def test_copy_records_src_mtime_utc(tmp_path, monkeypatch):
    monkeypatch.setattr(bprf, "ROOT", tmp_path)
    src = tmp_path / "src.txt"
    src.write_text("hello")
    entries = []
    bprf._copy(src, tmp_path / "out", {}, entries, "sec")
    assert entries[-1]["status"] == "ok"
    mtime = entries[-1]["src_mtime_utc"]
    parsed = datetime.strptime(mtime, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert parsed.year == 2026 or parsed.year >= 2020  # sanity: parses as a real UTC stamp


def test_copy_missing_source_has_no_mtime(tmp_path):
    entries = []
    bprf._copy(tmp_path / "nope.png", tmp_path / "out", {}, entries, "sec")
    assert entries[-1]["status"] == "MISSING"
    assert "src_mtime_utc" not in entries[-1]


# --------------------------------------------------------------------------
# Retention: keep newest N dated folders, own output only
# --------------------------------------------------------------------------

def _make_dated_folder(base: Path, name: str, generator: str = "scripts/build_pi_review_folder.py") -> Path:
    d = base / name
    d.mkdir(parents=True)
    (d / "MANIFEST.json").write_text(json.dumps({"generator": generator, "_utc": f"{name}T00:00:00Z"}))
    return d


def test_prune_dated_folders_keeps_only_newest_n(tmp_path):
    for name in ("2026-09-01", "2026-09-08", "2026-09-15", "2026-09-22"):
        _make_dated_folder(tmp_path, name)
    removed = bprf._prune_dated_folders(tmp_path, 3)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["2026-09-08", "2026-09-15", "2026-09-22"]
    assert [p.name for p in removed] == ["2026-09-01"]


def test_prune_dated_folders_never_touches_a_foreign_directory(tmp_path):
    _make_dated_folder(tmp_path, "2026-09-01")
    _make_dated_folder(tmp_path, "2026-09-08")
    _make_dated_folder(tmp_path, "2026-09-15")
    _make_dated_folder(tmp_path, "2026-09-22")
    foreign = tmp_path / "2026-08-01"
    foreign.mkdir()
    (foreign / "MANIFEST.json").write_text(json.dumps({"generator": "some_other_script.py"}))
    not_a_manifest_dir = tmp_path / "not_dated_at_all"
    not_a_manifest_dir.mkdir()

    bprf._prune_dated_folders(tmp_path, 3)

    assert foreign.exists()  # foreign generator's output is never touched
    assert not_a_manifest_dir.exists()  # no MANIFEST.json at all -> left alone


def test_prune_dated_folders_keep_zero_or_fewer_than_present_is_noop_safe(tmp_path):
    _make_dated_folder(tmp_path, "2026-09-01")
    _make_dated_folder(tmp_path, "2026-09-08")
    removed = bprf._prune_dated_folders(tmp_path, 3)
    assert removed == []
    assert len(list(tmp_path.iterdir())) == 2


def test_previous_cycle_utc_picks_newest_sibling_excluding_self(tmp_path):
    _make_dated_folder(tmp_path, "2026-09-17")
    today = tmp_path / "2026-09-24"
    today.mkdir()
    prev = bprf._previous_cycle_utc(today)
    assert prev == "2026-09-17T00:00:00Z"


def test_previous_cycle_utc_none_on_first_ever_cycle(tmp_path):
    today = tmp_path / "2026-09-24"
    today.mkdir()
    assert bprf._previous_cycle_utc(today) is None


def test_dangling_relative_links_catches_a_missing_target(tmp_path):
    (tmp_path / "sec").mkdir()
    (tmp_path / "sec" / "doc.html").write_text("x")
    (tmp_path / "index.html").write_text('<a href="sec/doc.html">ok</a><a href="doc.html">broken</a><a href="/ops">abs</a>')
    (tmp_path / "all.html").write_text("")
    assert bprf.dangling_relative_links(tmp_path) == ["index.html: doc.html"]
# --------------------------------------------------------------------------
# Phase 4 — join release_class from brisaverse's p1_artifacts.json register
# (never re-derived), the "Awaiting your call" block (G3), release badges.
# --------------------------------------------------------------------------

def test_release_badge_staged_wins_over_publishable_text():
    # A staged row's own release_class text says "publishable (staged; PI
    # promotes)" — state must decide first, or every staged row would render
    # as publishable and G3 would silently undercount.
    row = {"state": "staged", "release_class": "`publishable` (staged; PI promotes)"}
    assert bprf._release_badge(row) == "staged"


def test_release_badge_withheld_from_state_or_text():
    assert bprf._release_badge({"state": "withheld", "release_class": "withheld"}) == "withheld"
    assert bprf._release_badge({"state": "final", "release_class": "withheld"}) == "withheld"


def test_release_badge_publishable_from_text():
    row = {"state": "promoted", "release_class": "PNG + SVG `publishable`"}
    assert bprf._release_badge(row) == "publishable"


def test_release_badge_unclassified_when_no_row():
    assert bprf._release_badge(None) == "unclassified"


def test_release_badge_unclassified_when_row_has_no_recognizable_class():
    assert bprf._release_badge({"state": "final", "release_class": None}) == "unclassified"


def test_register_index_keys_by_run_and_filename():
    register = [
        {"id": "f1", "state": "staged", "run_of_record": "wp07_figures_X",
         "image_url": "/morphofavela-dash/outputs/_hub/wp07_staged/f1_citywide_position.png"},
    ]
    idx = bprf._register_index(register)
    assert idx[("wp07_figures_X", "f1_citywide_position.png")]["id"] == "f1"
    assert len(idx) == 1


def test_apply_release_badges_joins_by_run_and_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(bprf, "ROOT", tmp_path)
    register = [
        {"id": "f1", "state": "staged", "run_of_record": "wp07_figures_X",
         "image_url": "/x/f1_citywide_position.png"},
        {"id": "f5", "state": "withheld", "release_class": "withheld",
         "run_of_record": "wp07_map_Y", "image_url": "/x/f5_citywide_svf_map.png"},
    ]
    entries = [
        {"section": "p1_solar_figures", "file": "f1_citywide_position.png", "status": "ok",
         "source": "runs/wp07_figures_X/f1_citywide_position.png"},
        {"section": "citywide_maps", "file": "f5_citywide_svf_map.png", "status": "ok",
         "source": "runs/wp07_map_Y/f5_citywide_svf_map.png"},
        {"section": "mare_territory", "file": "mare_territory_map.png", "status": "ok",
         "source": "outputs/maré/territory/mare_territory_map.png"},
        {"section": "sweep/foo", "file": "unrelated.png", "status": "MISSING"},
    ]
    bprf._apply_release_badges(entries, register)
    assert entries[0]["release_badge"] == "staged"
    assert entries[0]["register_id"] == "f1"
    assert entries[1]["release_badge"] == "withheld"
    assert entries[2]["release_badge"] == "unclassified"
    assert "register_id" not in entries[2]
    assert "release_badge" not in entries[3]  # MISSING entries are never joined


def test_compute_awaiting_selects_only_staged_ok_entries_sorted():
    entries = [
        {"section": "b", "file": "z.png", "status": "ok", "register_state": "staged"},
        {"section": "a", "file": "y.png", "status": "ok", "register_state": "staged"},
        {"section": "a", "file": "x.png", "status": "ok", "register_state": "withheld"},
        {"section": "a", "file": "w.png", "status": "MISSING", "register_state": "staged"},
    ]
    out = bprf._compute_awaiting(entries)
    assert [(e["section"], e["file"]) for e in out] == [("a", "y.png"), ("b", "z.png")]


def test_render_awaiting_links_badge_to_ops_promotion_card():
    # No governing_decision on this entry -> falls back to the pre-split id,
    # never a fabricated guess (see test_release_badge_tag_* below for the
    # per-family routing this fell back from).
    awaiting = [{"section": "p1_solar_figures", "file": "f1.png", "status": "ok", "bytes": 1,
                 "release_badge": "staged", "register_id": "f1_citywide_position"}]
    html = bprf._render_awaiting(awaiting)
    assert 'id="s-awaiting"' in html
    assert f"/ops#dec-{bprf._FALLBACK_PROMOTION_DECISION}" in html
    assert "Awaiting your call <span class=\"n\">(1)</span>" in html


def test_governing_decision_routes_by_run_of_record_prefix():
    assert bprf._governing_decision("wp07_figures_20260924T212023Z") == "wp07_promote_solar"
    assert bprf._governing_decision("wp07_method_20260917T125027Z") == "wp07_promote_method"
    assert bprf._governing_decision("terrain_split_full_20260917T130045Z") == "wp07_promote_terrain"
    # unknown / missing run -> the old pre-split id, never a guess
    assert bprf._governing_decision("some_other_run_20260101T000000Z") == bprf._FALLBACK_PROMOTION_DECISION
    assert bprf._governing_decision(None) == bprf._FALLBACK_PROMOTION_DECISION


def test_release_badge_tag_routes_staged_figure_to_its_own_decision_card():
    # round-1 critic finding (criterion 4/6): every staged figure's badge
    # used to point at the single, now-resolved wp07_figure_promotion id
    # regardless of which of the three post-split decisions actually
    # governs it — the tag must now carry the figure's own family.
    e = {"release_badge": "staged", "governing_decision": "wp07_promote_terrain"}
    html = bprf._release_badge_tag(e)
    assert "/ops#dec-wp07_promote_terrain" in html
    assert "wp07_figure_promotion" not in html


def test_render_awaiting_renders_zero_state_without_omitting_section():
    html = bprf._render_awaiting([])
    assert 'id="s-awaiting"' in html
    assert "(0)" in html


def test_render_index_includes_awaiting_section_between_new_and_toc():
    m = {
        "_utc": "2026-09-24T00:00:00Z", "cycle_date": "2026-09-24",
        "sections": [{"slug": "a", "title": "A", "blurb": "", "provenance": "", "order": 1}],
        "files": [{"section": "a", "file": "a.png", "status": "ok", "bytes": 1,
                   "src_mtime_utc": "2026-09-24T00:00:00Z"}],
        "new_since": {"cutoff_utc": None, "total": 0, "shown": 0, "families": []},
        "awaiting": [{"section": "a", "file": "a.png", "status": "ok", "bytes": 1,
                      "release_badge": "staged"}],
    }
    html = bprf._render_index(m)
    assert html.index('id="s-new"') < html.index('id="s-awaiting"') < html.index('id="s-toc"')


def test_figure_card_release_badge_is_shown_and_never_hides_the_figure():
    e = {"status": "ok", "file": "f.png", "bytes": 1, "release_badge": "unclassified"}
    html = bprf._figure_card(e, "sec")
    assert "unclassified" in html
    assert "f.png" in html


def test_every_figure_card_names_its_file_for_the_gates():
    card = bprf._figure_card({"status": "OK", "file": "f9_x.png", "bytes": 1000, "thumb": None}, "sec")
    assert 'data-file="f9_x.png"' in card


# --------------------------------------------------------------------------
# O7 — "WP chips" (charter phase D / figure_organization_spec.md §7): the
# sweep grouped by the registry's WP -> family -> run, and index.html's chip
# row linking all.html#wp-<KEY>.
# --------------------------------------------------------------------------

def test_wp_eligible_entries_includes_curated_sections_not_only_sweep():
    """2026-09-25 live-round-1 finding 3: a curated RECORDS entry (e.g.
    section="wp07_staged" or "site_territory") is exactly as WP-eligible as
    a sweep/-prefixed one — build()'s wp_tree must not silently see only
    the leftover catch-all sweep while the main page (and /figures, and a
    site page) show the curated figure. Only a failed copy is excluded."""
    entries = [
        {"status": "ok", "section": "wp07_staged", "file": "f1_citywide_position.png",
         "source": "outputs/_hub/wp07_staged/f1_citywide_position.png"},
        {"status": "ok", "section": "site_territory", "file": "mare_territory_map.png",
         "source": "outputs/mare/territory/mare_territory_map.png"},
        {"status": "ok", "section": "sweep/misc", "file": "other.png",
         "source": "outputs/misc/other.png"},
        {"status": "MISSING", "section": "wp07_staged", "file": "gone.png",
         "source": "outputs/_hub/wp07_staged/gone.png"},
    ]
    files = {e["file"] for e in bprf._wp_eligible_entries(entries)}
    assert files == {"f1_citywide_position.png", "mare_territory_map.png", "other.png"}


def _fake_registry():
    """One run-backed family (WP07/wp07_figures) with a current + a
    superseded run, and one static family (WP04F/site_print) with no run
    axis — enough to exercise every branch of _group_sweep_by_wp."""
    return {
        "nodes": {
            "wp:WP07": {"kind": "wp", "parent": None, "title": "P1 solar results"},
            "wp:WP04F": {"kind": "wp", "parent": None, "title": "Facade solar"},
            "fam:wp07_figures": {"kind": "family", "parent": "wp:WP07"},
            "fam:site_print": {"kind": "family", "parent": "wp:WP04F"},
            "run:wp07_figures_20260917T125201Z": {
                "kind": "run", "parent": "fam:wp07_figures", "lifecycle": "current",
                "run_utc": "2026-09-17T12:52:01Z"},
            "run:wp07_figures_20260910T000000Z": {
                "kind": "run", "parent": "fam:wp07_figures", "lifecycle": "superseded",
                "run_utc": "2026-09-10T00:00:00Z"},
            "art:wp07_figures::wp07_figures_20260917T125201Z::f1": {
                "kind": "figure", "parent": "run:wp07_figures_20260917T125201Z",
                "wp": "WP07", "family": "wp07_figures",
                "path": "runs/wp07_figures_20260917T125201Z/f1.png"},
            "art:wp07_figures::wp07_figures_20260910T000000Z::f1": {
                "kind": "figure", "parent": "run:wp07_figures_20260910T000000Z",
                "wp": "WP07", "family": "wp07_figures",
                "path": "runs/wp07_figures_20260910T000000Z/f1.png"},
            "art:site_print::plate": {
                "kind": "figure", "parent": "fam:site_print",
                "wp": "WP04F", "family": "site_print",
                "path": "outputs/mare/print/plate.png"},
        },
    }


def test_group_sweep_by_wp_buckets_current_and_superseded_runs(monkeypatch):
    monkeypatch.setattr(bprf, "_wp_key_order", lambda: [
        ("WP07", "P1 solar results"), ("WP04F", "Facade solar"), ("WP01", "Footprint repair")])
    entries = [
        {"status": "ok", "section": "sweep/a", "file": "f1_new.png",
         "source": "runs/wp07_figures_20260917T125201Z/f1.png"},
        {"status": "ok", "section": "sweep/a", "file": "f1_old.png",
         "source": "runs/wp07_figures_20260910T000000Z/f1.png"},
        {"status": "ok", "section": "sweep/b", "file": "plate.png",
         "source": "outputs/mare/print/plate.png"},
        {"status": "ok", "section": "sweep/c", "file": "mystery.png",
         "source": "outputs/some/untracked/mystery.png"},
    ]
    tree = bprf._group_sweep_by_wp(entries, _fake_registry())

    by_key = {wp["key"]: wp for wp in tree["wps"]}
    assert set(by_key) == {"WP07", "WP04F", "WP01"}
    assert by_key["WP01"]["n"] == 0 and by_key["WP01"]["families"] == []  # declared, nothing swept

    wp07 = by_key["WP07"]
    assert wp07["n"] == 2
    fam = wp07["families"][0]
    assert fam["key"] == "wp07_figures"
    runs_by_lifecycle = {r["lifecycle"]: r for r in fam["runs"]}
    assert runs_by_lifecycle["current"]["items"][0]["file"] == "f1_new.png"
    assert runs_by_lifecycle["superseded"]["items"][0]["file"] == "f1_old.png"  # kept, not dropped

    wp04f = by_key["WP04F"]
    assert wp04f["n"] == 1
    assert wp04f["families"][0]["runs"][0]["run_id"] is None  # static family, no run axis

    assert [e["file"] for e in tree["unmatched"]] == ["mystery.png"]  # shown, never hidden


def test_group_sweep_by_wp_degrades_to_all_unmatched_when_registry_unavailable(monkeypatch):
    monkeypatch.setattr(bprf, "_wp_key_order", lambda: [("WP07", "P1 solar results")])
    entries = [{"status": "ok", "section": "sweep/a", "file": "x.png", "source": "outputs/x.png"}]
    tree = bprf._group_sweep_by_wp(entries, {})
    assert tree["wps"] == [{"key": "WP07", "title": "P1 solar results", "families": [], "n": 0}]
    assert [e["file"] for e in tree["unmatched"]] == ["x.png"]


def _wp_tree_fixture():
    return {
        "wps": [
            {"key": "WP07", "title": "P1 solar results", "n": 2, "families": [
                {"key": "wp07_figures", "n": 2, "runs": [
                    {"run_id": "wp07_figures_20260917T125201Z", "lifecycle": "current",
                     "run_utc": "2026-09-17T12:52:01Z",
                     "items": [{"section": "sweep/a", "file": "f1_new.png", "status": "ok", "bytes": 1}]},
                    {"run_id": "wp07_figures_20260910T000000Z", "lifecycle": "superseded",
                     "run_utc": "2026-09-10T00:00:00Z",
                     "items": [{"section": "sweep/a", "file": "f1_old.png", "status": "ok", "bytes": 1}]},
                ]},
            ]},
            {"key": "WP01", "title": "Footprint repair", "n": 0, "families": []},
        ],
        "unmatched": [{"section": "sweep/c", "file": "mystery.png", "status": "ok", "bytes": 1}],
    }


def _base_manifest(**extra):
    m = {
        "_utc": "2026-09-24T00:00:00Z", "cycle_date": "2026-09-24",
        "sections": [{"slug": "sweep/c", "title": "some/untracked", "group": "other"}],
        "files": [], "new_since": {"cutoff_utc": None, "total": 0, "shown": 0, "families": []},
        "awaiting": [],
    }
    m.update(extra)
    return m


def test_every_wp_chip_on_index_opens_a_matching_anchor_in_all_html():
    """The O7 test: every chip index.html renders must resolve to a real
    id="wp-<KEY>" node in all.html — including WP01, which has nothing
    swept this cycle (muted, never omitted)."""
    m = _base_manifest(wp_tree=_wp_tree_fixture())
    index_html = bprf._render_index(m)
    all_html = bprf._render_all(m)

    chips = re.findall(r'<a class="(chip[^"]*)" href="(all\.html#[^"]+)">', index_html)
    assert len(chips) == 2  # WP07 + WP01, both declared
    for cls, href in chips:
        anchor_id = href.split("#", 1)[1]
        assert f'id="{anchor_id}"' in all_html, f"chip {href!r} has no matching anchor in all.html"
    muted = {href for cls, href in chips if "chip-muted" in cls}
    assert muted == {"all.html#wp-WP01"}  # the empty WP is muted, not hidden


def test_all_html_collapses_a_superseded_run_but_keeps_its_figure():
    m = _base_manifest(wp_tree=_wp_tree_fixture())
    html = bprf._render_all(m)
    assert 'id="wp-WP07"' in html
    assert "f1_new.png" in html
    assert "<details" in html and "f1_old.png" in html  # collapsed, never deleted
    # the current run's figure is NOT inside a <details> (open by default)
    assert html.index("f1_new.png") < html.index("<details")


def test_all_html_shows_unmatched_sweep_entries_grouped_by_folder_never_hidden():
    m = _base_manifest(wp_tree=_wp_tree_fixture())
    html = bprf._render_all(m)
    assert 'id="wp-UNCLASSIFIED"' in html
    assert "mystery.png" in html
    assert "some/untracked" in html  # folder title, from the sweep section metadata


def test_render_all_with_no_wp_tree_at_all_still_renders_a_valid_page():
    m = _base_manifest()
    html = bprf._render_all(m)
    assert "<!doctype html>" in html.lower()


def test_all_html_groups_many_superseded_runs_into_one_collapsed_entry():
    """O8 cleanup (organization_charter.md §4 / figure_organization_spec.md
    §7): a family with many non-current runs (wp02_horizon has ~95) must
    render ONE collapsed 'N earlier runs' entry, never one <details> per
    run — that is exactly the card-noise the rubric's criterion 4 forbids.
    Every run and figure still appears once the group is opened; nothing
    is deleted, only nested one level deeper."""
    runs = [{"run_id": "wp07_figures_20260917T125201Z", "lifecycle": "current",
             "run_utc": "2026-09-17T12:52:01Z",
             "items": [{"section": "sweep/a", "file": "f1_new.png", "status": "ok", "bytes": 1}]}]
    for i in range(5):
        runs.append({"run_id": f"wp07_figures_2026091{i}T000000Z", "lifecycle": "superseded",
                      "run_utc": f"2026-09-1{i}T00:00:00Z",
                      "items": [{"section": "sweep/a", "file": f"f1_old_{i}.png", "status": "ok", "bytes": 1}]})
    runs.append({"run_id": "wp07_figures_20260901T000000Z", "lifecycle": "draft",
                 "run_utc": "2026-09-01T00:00:00Z",
                 "items": [{"section": "sweep/a", "file": "f1_draft.png", "status": "ok", "bytes": 1}]})
    wp_tree = {"wps": [{"key": "WP07", "title": "P1 solar results", "n": 7, "families": [
        {"key": "wp07_figures", "n": 7, "runs": runs}]}], "unmatched": []}
    m = _base_manifest(wp_tree=wp_tree)
    html = bprf._render_all(m)

    # every superseded figure still appears, and the group summary names the count
    for i in range(5):
        assert f"f1_old_{i}.png" in html
    assert "f1_draft.png" in html
    assert "5 earlier runs — superseded" in html
    assert "1 run — draft" in html
    # exactly one TOP-LEVEL <details> for superseded and one for draft, not
    # six — the 5 per-run <details> are nested one level inside it
    top_level = re.findall(r'<details class="run-details"><summary>(\d+) (?:earlier run|run)', html)
    assert top_level == ["5", "1"]


# --------------------------------------------------------------------------
# Corrective plan step 4 (docs/critic/incident_dashboard_loop_2026-09-25.md,
# root cause 4): _load_fresh_registry must fail loudly, never silently
# degrade to {} — that silence is exactly how "the WP chips shipped empty
# because the review folder was built while the registry build failed"
# reached the PI undetected.
# --------------------------------------------------------------------------

def test_load_fresh_registry_raises_when_build_results_registry_errors(monkeypatch):
    class _FakeBRR:
        @staticmethod
        def build():
            raise RuntimeError("simulated registry build failure")

    monkeypatch.setitem(sys.modules, "build_results_registry", _FakeBRR())
    with pytest.raises(RuntimeError, match="results registry could not be built"):
        bprf._load_fresh_registry()


def test_load_fresh_registry_raises_when_import_fails(monkeypatch):
    real_import = __import__

    def _blocking_import(name, *args, **kwargs):
        if name == "build_results_registry":
            raise ImportError("simulated: module not importable")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "build_results_registry", raising=False)
    monkeypatch.setattr("builtins.__import__", _blocking_import)
    with pytest.raises(RuntimeError, match="cannot import build_results_registry"):
        bprf._load_fresh_registry()


def test_load_fresh_registry_returns_real_registry_on_success(monkeypatch):
    """Green counterpart to the two red proofs above: a working registry
    module still returns its built dict, unchanged."""
    class _FakeBRR:
        @staticmethod
        def build():
            return {"nodes": {}, "counts": {"figure": 0}}

    monkeypatch.setitem(sys.modules, "build_results_registry", _FakeBRR())
    result = bprf._load_fresh_registry()
    assert result == {"nodes": {}, "counts": {"figure": 0}}
