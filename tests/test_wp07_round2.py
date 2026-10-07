"""WP-07 round-2 analyses (src/brisa_solar/wp07_round2.py): estimator unit
tests on toy data, plus run-of-record checks that skip when the run is absent."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.brisa_solar import wp07_round2 as r2
from src.brisa_solar.wp07_crosstab import crosstab
from src.brisa_solar.wp07_ledger import RUN_OF_RECORD

ROOT = Path(__file__).resolve().parents[1]
ROUND2 = ROOT / "runs" / RUN_OF_RECORD["round2"] / "summary.json"
XTAB = ROOT / "runs" / RUN_OF_RECORD["crosstab"] / "summary.json"


def _toy_cells():
    return pd.DataFrame({
        "share_ge_2h_winter": [1.0, 0.5, 0.0, 0.25, 0.0, None],
        "sun_h_winter_p50": [5.0, 2.0, 0.0, 1.0, 0.0, None],
        "constraint_vertical": [0, 0, 1, 1, 1, 1],
        "constraint_lateral": [0, 1, 0, 1, 1, 0],
        "constraint_directional": [0, 0, 0, 0, 1, 0],
        "n_constraints": [0, 1, 1, 2, 3, 1],
    })


def test_stratified_crosstab_classes():
    s = r2.stratified_crosstab(_toy_cells())
    assert s["v0"]["o0"] == {"cells": 1, "point_deficit": 0.0, "cell_deficit": 0.0}
    assert s["v0"]["o1"]["point_deficit"] == 0.5
    assert s["v1"]["o0"]["cells"] == 1  # the NaN row drops out, as in wp07_crosstab
    assert s["v1"]["o1"]["point_deficit"] == 0.75
    assert s["v1"]["o2"]["cell_deficit"] == 1.0
    assert s["v0"]["o2"]["cells"] == 0


def test_stratified_cells_partition_table1_cells():
    t = _toy_cells()
    s = r2.stratified_crosstab(t)
    total = sum(s[v][o]["cells"] for v in s for o in s[v])
    assert total == sum(r["cells"] for r in crosstab(t))


def test_block_bootstrap_degenerate_and_joint():
    rng = np.random.default_rng(0)
    codes = np.array([0, 0, 1, 1, 2, 2])
    vals = np.c_[np.array([1, 0, 1, 0, 1, 0], float), np.ones(6)]
    rs = r2.resampled_sums(r2.block_sums(codes, vals), 500, rng)
    lo, hi = r2.interval(rs[:, 0], rs[:, 1])
    assert lo == hi == 0.5  # every block has share 0.5, so every replicate does
    assert np.all(rs[:, 1] == 6)  # n_blocks draws of 2-point blocks


def test_block_codes_square_grid():
    x = np.array([0.0, 99.9, 100.0, 250.0])
    y = np.array([0.0, 0.0, 0.0, 0.0])
    c = r2.block_codes(x, y, 100.0)
    assert c[0] == c[1] and c[1] != c[2] and c[2] != c[3]


def test_practical_range_iid_vs_blocky():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1500, 20000)
    y = rng.uniform(0, 1500, 20000)
    iid = rng.random(20000) < 0.5
    assert r2.practical_range_m(x, y, iid)["range_m"] == r2.VARIO_LAG_M
    field = rng.random((10, 10)) < 0.5
    blocky = field[(x // 150).astype(int), (y // 150).astype(int)]
    rng_m = r2.practical_range_m(x, y, blocky)["range_m"]
    assert 100.0 <= rng_m <= 300.0


def test_december_metrics():
    m = r2.december_metrics(np.array([0.0, 1.0, 6.0, 8.0]), np.array([True, True, True, False]))
    assert m["share_ge_6h"] == 0.5 and m["share_ge_8h"] == 0.25
    assert m["share_lt2h_june_and_ge6h_december"] == 0.25
    assert m["share_lt2h_june_and_lt2h_december"] == 0.5
    assert m["p50"] == 3.5


@pytest.mark.skipif(not (ROUND2.exists() and XTAB.exists()), reason="round-2 or crosstab run absent")
def test_round2_run_agrees_with_table1_and_reproduces_stored_hours():
    s = json.loads(ROUND2.read_text())
    xt = json.loads(XTAB.read_text())["per_site"]
    for k, row in enumerate(xt["pooled"]["rows"]):
        assert s["bootstrap"]["pooled"]["n"][f"n{k}"]["point_deficit"] == pytest.approx(row["point_deficit"], rel=1e-12)
    for slug, d in r2.SITE_DIRS.items():
        strat = s["stratified"][slug]
        assert sum(strat[v][o]["cells"] for v in strat for o in strat[v]) == sum(r["cells"] for r in xt[d]["rows"])
        assert strat["v0"]["o0"]["cells"] == xt[d]["rows"][0]["cells"]
        assert strat["v1"]["o2"]["cells"] == xt[d]["rows"][3]["cells"]
        for day in ("winter", "equinox"):
            b = s["bootstrap"]["per_site"][slug][f"share_below_2h_{day}"]
            assert b["lo"] <= b["estimate"] <= b["hi"]
        assert s["bootstrap"]["per_site"][slug]["block_m"] >= s["variogram"]["per_site"][slug]["range_m"]
        check = s["december"]["per_site"][slug]["remarch_check_vs_stored"]
        assert check["winter_solstice"]["n_mismatch"] == 0 and check["equinox"]["n_mismatch"] == 0
    assert s["bootstrap"]["replicates"] >= 1000


@pytest.mark.skipif(not ROUND2.exists(), reason="round-2 run absent")
def test_render_additions_reports_changed_keys():
    from src.brisa_solar import wp07_ledger as w
    new = w.build_ledger(ROOT)
    old = {"entries": {k: dict(v) for k, v in list(new["entries"].items())[:5]}}
    assert "changed: 0 — every pre-existing key holds" in r2.render_additions(new, old, "a", "b")
    first = next(iter(old["entries"]))
    old["entries"][first] = {**old["entries"][first], "value": "drifted"}
    assert f"changed: 1 ({first})" in r2.render_additions(new, old, "a", "b")
