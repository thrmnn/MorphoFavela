"""Tests for scripts/audit_hub_graph.py — the reachability + link-integrity
gate for outputs/_hub/ (HUBWP Phase 1, docs/hub_wp_structure_spec.md).

Every fixture is a throwaway tree under tmp_path (outputs/_hub/ is gitignored
and not checked out in a fresh worktree, so these tests never depend on a
real pipeline run) mirroring the pattern in test_build_project_hub.py's
synthetic-fixture test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_hub_graph as ahg  # noqa: E402


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _hub(tmp_path: Path) -> Path:
    return tmp_path / "outputs" / "_hub"


def test_card_link_is_reachable_at_depth_one(tmp_path):
    hub = _hub(tmp_path)
    _write(hub / "index.html",
          '<div class="grid"><a class="card" href="a.html">'
          '<div class="cap"><h3>A</h3></div></a></div>')
    _write(hub / "a.html", "<h1>A</h1>")
    result = ahg.audit(hub)
    assert result.ok
    assert result.depth[hub / "index.html"] == 0
    assert result.depth[hub / "a.html"] == 1


def test_breadcrumb_back_link_is_navigable(tmp_path):
    hub = _hub(tmp_path)
    _write(hub / "index.html",
          '<nav class="toc"><a href="docs/a.html">A</a></nav>')
    _write(hub / "docs" / "a.html",
          '<nav class="crumb"><a href="../index.html">Hub</a></nav>')
    result = ahg.audit(hub)
    assert result.ok
    assert result.depth[hub / "docs" / "a.html"] == 1


def test_prose_link_is_not_navigable_but_not_orphan(tmp_path):
    """A link sitting in running prose (no card, no nav) gives the target an
    inbound link without making it reachable — this is the PROSE-ONLY class,
    the exact failure mode that hid /roadmap in the sibling brisaverse hub."""
    hub = _hub(tmp_path)
    _write(hub / "index.html", '<p>See also <a href="a.html">A</a>.</p>')
    _write(hub / "a.html", "<h1>A</h1>")
    result = ahg.audit(hub)
    assert not result.ok
    assert hub / "a.html" in result.prose_only
    assert hub / "a.html" not in result.orphan
    assert (hub / "a.html") not in result.depth


def test_page_with_no_inbound_link_at_all_is_orphan(tmp_path):
    hub = _hub(tmp_path)
    _write(hub / "index.html", "<h1>Hub</h1>")
    _write(hub / "a.html", "<h1>A, linked from nowhere</h1>")
    result = ahg.audit(hub)
    assert not result.ok
    assert hub / "a.html" in result.orphan
    assert hub / "a.html" not in result.prose_only


def test_reachable_only_through_an_unreachable_page_does_not_count(tmp_path):
    """Reaching a page only via a page that is itself unreachable is not
    reaching it — the BFS must not just union every edge into one flat set.
    b.html has a real inbound card link, but its only source (orphan.html)
    is never reached from the index, so b.html must not be REACHABLE either
    (it lands in prose-only: linked, but not navigable-from-the-index)."""
    hub = _hub(tmp_path)
    _write(hub / "index.html", "<h1>Hub, no cards</h1>")
    _write(hub / "orphan.html",
          '<div class="grid"><a class="card" href="b.html">'
          '<div class="cap"><h3>B</h3></div></a></div>')
    _write(hub / "b.html", "<h1>B</h1>")
    result = ahg.audit(hub)
    assert hub / "orphan.html" in result.orphan
    assert (hub / "b.html") not in result.depth
    assert hub / "b.html" in result.prose_only


def test_directory_target_resolves_to_its_index(tmp_path):
    hub = _hub(tmp_path)
    _write(hub / "index.html",
          '<div class="grid"><a class="card" href="sub/">'
          '<div class="cap"><h3>Sub</h3></div></a></div>')
    _write(hub / "sub" / "index.html", "<h1>Sub</h1>")
    result = ahg.audit(hub)
    assert result.ok
    assert hub / "sub" / "index.html" in result.depth


def test_dangling_link_is_reported_and_fails_the_gate(tmp_path):
    hub = _hub(tmp_path)
    _write(hub / "index.html",
          '<div class="grid"><a class="card" href="missing.html">'
          '<div class="cap"><h3>Gone</h3></div></a></div>')
    result = ahg.audit(hub)
    assert not result.ok
    assert result.dangling == [(hub / "index.html", "missing.html")]


def test_dangling_zoom_target_is_reported(tmp_path):
    hub = _hub(tmp_path)
    _write(hub / "index.html", "onclick=\"zoom('missing.png','cap')\"")
    result = ahg.audit(hub)
    assert not result.ok
    assert (hub / "index.html", "missing.png") in result.dangling


def test_external_and_anchor_only_links_are_never_dangling(tmp_path):
    hub = _hub(tmp_path)
    _write(hub / "index.html",
          '<a href="https://example.com/x">ext</a>'
          '<a href="mailto:a@b.com">mail</a>'
          '<a href="#section">on-page</a>')
    result = ahg.audit(hub)
    assert result.dangling == []


def test_ok_property_false_when_any_failure_class_present(tmp_path):
    hub = _hub(tmp_path)
    _write(hub / "index.html", "<h1>Hub</h1>")
    _write(hub / "orphan.html", "<h1>orphan</h1>")
    result = ahg.audit(hub)
    assert not result.ok


def test_main_exit_code_matches_ok(tmp_path, capsys):
    hub = _hub(tmp_path)
    _write(hub / "index.html",
          '<div class="grid"><a class="card" href="a.html">'
          '<div class="cap"><h3>A</h3></div></a></div>')
    _write(hub / "a.html", "<h1>A</h1>")
    assert ahg.main(["--root", str(tmp_path)]) == 0

    _write(hub / "orphan.html", "<h1>orphan, no inbound link</h1>")
    assert ahg.main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "orphan.html" in out


def test_main_fails_loudly_when_hub_dir_missing(tmp_path, capsys):
    assert ahg.main(["--root", str(tmp_path)]) == 1


def test_report_prints_every_page_sorted_by_depth(tmp_path, capsys):
    hub = _hub(tmp_path)
    _write(hub / "index.html",
          '<div class="grid"><a class="card" href="a.html">'
          '<div class="cap"><h3>A</h3></div></a></div>')
    _write(hub / "a.html",
          '<div class="grid"><a class="card" href="b.html">'
          '<div class="cap"><h3>B</h3></div></a></div>')
    _write(hub / "b.html", "<h1>B</h1>")
    result = ahg.audit(hub)
    ahg.report(result, tmp_path)
    out = capsys.readouterr().out
    assert out.find("[0]") < out.find("[1]") < out.find("[2]")
