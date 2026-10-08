"""lambda_f sensitivity: the 0.65 column must reproduce the runs of record exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import wp07_lambda_sensitivity as ls  # noqa: E402
from src.brisa_solar.wp07_ledger import RUN_OF_RECORD  # noqa: E402

needs_runs = pytest.mark.skipif(
    not (ROOT / "runs" / ls.LEDGER_RUN / "ledger.json").exists()
    or not (ROOT / "outputs" / "vidigal" / "geometry_indicators" / ls.CROSSTAB_IN_NAME).exists(),
    reason="runs of record absent",
)


def _toy():
    return pd.DataFrame({
        "lambda_f_mean": [0.4, 0.55, 0.7, 0.9, float("nan")],
        "share_ge_2h_winter": [1.0, 0.5, 0.25, 0.0, 1.0],
        "sun_h_winter_p50": [5.0, 3.0, 1.0, 0.0, 4.0],
        "constraint_vertical": [0, 0, 1, 1, 0],
        "constraint_lateral": [0, 1, 0, 1, 0],
        "constraint_directional": [0, 0, 0, 0, 0],
        "n_constraints": [0, 1, 1, 2, 0],
    })


def test_threshold_moves_only_vertical_flag():
    t = ls.with_threshold(_toy(), 0.5)
    assert t["constraint_vertical"].tolist() == [0, 1, 1, 1, 0]
    assert t["n_constraints"].tolist() == [0, 2, 1, 2, 0]
    t8 = ls.with_threshold(_toy(), 0.8)
    assert t8["constraint_vertical"].tolist() == [0, 0, 0, 1, 0]
    assert t8["constraint_lateral"].tolist() == _toy()["constraint_lateral"].tolist()


def test_toy_at_065_keeps_input_counts():
    assert ls.with_threshold(_toy(), 0.65)["n_constraints"].tolist() == _toy()["n_constraints"].tolist()


@needs_runs
def test_065_reproduces_runs_of_record():
    result = ls.compute(ls.load_tables(ROOT))
    check = ls.verify_record(result, ROOT)
    assert check["values_checked"] > 300
    assert RUN_OF_RECORD["crosstab"] in check["against"]
