"""Pure-function tests for the Maré distributions module (FOLHA4). No real
pipeline outputs are read — compute_distributions/draw_distributions need
the WP-05 parquet + Maré data files and are exercised instead by
scripts/build_site_dashboard.py --folha4-mare (a real-data smoke run, not a
fast unit test)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import render_mare_irradiation_distributions as mid  # noqa: E402


def test_distributions_rc_disables_top_and_right_spines():
    # These three panels' own local style, applied via rc_context by every
    # caller (this module's main() and build_site_dashboard.py's
    # build_folha4_mare) — never a global rcParams mutation that would leak
    # into a neighbouring panel's style.
    assert mid.DISTRIBUTIONS_RC["axes.spines.top"] is False
    assert mid.DISTRIBUTIONS_RC["axes.spines.right"] is False


def test_provenance_note_states_run_of_record_and_release_status():
    note = mid.provenance_note({"run_of_record": "wp05_full_20260101T000000Z"})
    assert "wp05_full_20260101T000000Z" in note
    assert "ethics-gate" in note
    assert "PI review" in note


def test_provenance_note_glosses_the_stats_terms_the_panels_use():
    # round-2/3 council finding: decile/IQR/length-weighted appear on the
    # panels with no other gloss anywhere on a print page (no hover the
    # way the interactive twin's <dfn> tooltips have) — this note is where
    # FOLHA4 puts the plain-language definitions.
    note = mid.provenance_note({"run_of_record": "wp05_full_20260101T000000Z"})
    assert "decile" in note
    assert "interquartile range" in note
    assert "length-weighted" in note


def test_draw_distributions_top_and_bottom_are_split_out():
    # FOLHA4's hero/no-hero variants call these directly (not the combined
    # draw_distributions wrapper) so the host sheet can place its own
    # fixed-ratio spacer between them — see build_site_dashboard.py.
    assert callable(mid.draw_distributions_top)
    assert callable(mid.draw_distributions_bottom)


def test_between_label_is_the_module_constant_not_a_retyped_string():
    assert mid.BETWEEN == "between communities"


# ---------------------------------------------------------------------------
# compute_community_stats (FOLHA4 round 4): the ONE place panel 3 and
# build_site_dashboard.py's community choropleth get their per-community
# numbers from — see the function's own docstring for why. No real gpkg/
# parquet data needed: these are plain name/label/value arrays.
# ---------------------------------------------------------------------------

def _stub_cells():
    # 3 named communities with distinct medians (A > C > B) + one with a
    # single cell, so median is trivially defined but n_cells is small.
    names = ["A", "B", "C"]
    sub = np.array(["A", "A", "B", "B", "B", "C", "C", "D"], dtype=object)
    p = np.array([90.0, 80.0, 10.0, 20.0, 30.0, 55.0, 65.0, 5.0])
    return names, sub, p


def test_compute_community_stats_ranks_by_descending_median():
    names, sub, p = _stub_cells()
    stats = mid.compute_community_stats(names, sub, p)
    assert [r["name"] for r in stats] == ["A", "C", "B"]
    assert [r["rank"] for r in stats] == [1, 2, 3]
    assert [r["number"] for r in stats] == [1, 2, 3]
    a = next(r for r in stats if r["name"] == "A")
    assert a["n_cells"] == 2
    assert a["median_percentile"] == 85.0


def test_compute_community_stats_zero_cell_community_is_flagged_not_silently_zero():
    # The exact real-data shape this guards: Marcílio Dias has zero cells
    # matching `sub == "Marcílio Dias"` (excluded from the study area by
    # geometry) — compute_distributions()'s own `order` list silently
    # drops it (`if (e['sub'] == c).any()`); this function must not.
    names, sub, p = _stub_cells()
    names = names + ["Zero"]  # "Zero" has no matching cells in `sub` at all
    stats = mid.compute_community_stats(names, sub, p)
    zero = next(r for r in stats if r["name"] == "Zero")
    assert zero["n_cells"] == 0
    assert zero["median_percentile"] is None
    assert zero["rank"] is None
    # still gets a display number (the map/panel-3 axis needs one for
    # every named community, not just the ranked ones)
    assert zero["number"] == 4
    # never silently ranked/sorted as if its median were 0 — it must not
    # appear among the ranked (rank is not None) rows at all
    ranked_names = [r["name"] for r in stats if r["rank"] is not None]
    assert "Zero" not in ranked_names
    assert len(ranked_names) == 3


def test_compute_community_stats_is_invariant_to_shuffled_name_order():
    # Shuffling the `names` input must not change any single community's
    # own median/n_cells/rank — only which order the OUTPUT rows come
    # back in is order-dependent (sorted by rank, always).
    names, sub, p = _stub_cells()
    stats_a = {r["name"]: r for r in mid.compute_community_stats(names, sub, p)}
    stats_b = {r["name"]: r for r in mid.compute_community_stats(list(reversed(names)), sub, p)}
    for name in names:
        assert stats_a[name]["median_percentile"] == stats_b[name]["median_percentile"]
        assert stats_a[name]["n_cells"] == stats_b[name]["n_cells"]
        assert stats_a[name]["rank"] == stats_b[name]["rank"]


def test_compute_community_stats_duplicated_name_produces_two_matching_rows():
    # A duplicated name in `names` must not be silently deduplicated or
    # averaged into one row — the caller (a gpkg with an accidental
    # repeated community row) gets two rows back, both with the SAME
    # n_cells/median (proving no partial-input aggregation happened), so
    # the bug is visible (two rows sharing a name) rather than hidden.
    names, sub, p = _stub_cells()
    dup_names = names + ["A"]
    stats = mid.compute_community_stats(dup_names, sub, p)
    a_rows = [r for r in stats if r["name"] == "A"]
    assert len(a_rows) == 2
    assert a_rows[0]["n_cells"] == a_rows[1]["n_cells"] == 2
    assert a_rows[0]["median_percentile"] == a_rows[1]["median_percentile"] == 85.0
    assert len(stats) == len(dup_names)


def test_draw_distributions_bottom_reads_community_stats_verbatim_not_recomputed():
    # F5b: the sidecar JSON build_site_dashboard.py writes IS
    # data['community_stats'] (json.dump'd directly) — this proves panel 3
    # draws its x-axis numbers from that exact same object too, so the two
    # can never independently drift apart. No real gpkg/parquet data
    # needed: draw_distributions_bottom only reads data['community_stats']
    # and data['e'] (here a minimal stand-in with a 'sub'/'p' column pair).
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, sub, p = _stub_cells()
    community_stats = mid.compute_community_stats(names, sub, p)
    e = pd.DataFrame({"sub": sub, "p": p})
    data = dict(community_stats=community_stats, e=e)

    fig = plt.figure()
    spec = fig.add_gridspec(1, 1)[0, 0]
    ax, = mid.draw_distributions_bottom(fig, spec, data)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    plt.close(fig)

    # Ranked order (A=1, C=2, B=3), each label carrying the SAME number
    # compute_community_stats assigned — not a re-derived index.
    assert labels[0].startswith("1 A")
    assert labels[1].startswith("2 C")
    assert labels[2].startswith("3 B")
