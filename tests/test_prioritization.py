"""WS-B prioritization — components, composite, worst-decile aggregation."""

import numpy as np
import pandas as pd

from src.morphometry.prioritization import (
    aggregate_priority_to_cells,
    priority_components,
    priority_score,
    quantile_classes,
)


def _obs():
    # observer 0: bad (no sun, enclosed, skimming); observer 2: good (sunny, open)
    return pd.DataFrame({
        "zone_id": ["A", "A", "B"],
        "solar_hours_winter": [0.0, 1.0, 6.0],
        "svf": [0.1, 0.3, 0.9],
        "lambda_f_mean": [0.7, 0.4, 0.05],
    })


def test_components_in_unit_range_and_oriented():
    c = priority_components(_obs())
    assert ((c >= 0) & (c <= 1)).all().all()
    # the sun-starved enclosed observer scores worse on every component than the open one
    assert (c.iloc[0] > c.iloc[2]).all()


def test_priority_score_monotone():
    s = priority_score(_obs())
    assert s.iloc[0] > s.iloc[1] > s.iloc[2]
    assert (s >= 0).all() and (s <= 1).all()


def test_quantile_classes_label_set():
    s = pd.Series(np.linspace(0, 1, 30))
    cls = quantile_classes(s)
    assert set(cls.unique()) <= {"lower", "elevated", "highest"}
    assert cls.iloc[0] == "lower" and cls.iloc[-1] == "highest"


def test_aggregate_uses_worst_decile_and_nulls_unsupported():
    obs = _obs()
    grid = pd.DataFrame({"zone_id": ["A", "B", "C"]})  # C has no observers
    agg = aggregate_priority_to_cells(obs, grid, priority_score(obs)).set_index("zone_id")
    assert bool(agg.loc["A", "has_priority"]) and not bool(agg.loc["C", "has_priority"])
    assert agg.loc["A", "n_obs"] == 2 and agg.loc["C", "n_obs"] == 0
    assert pd.isna(agg.loc["C", "priority_p90"])
    # p90 of cell A sits near its worse observer
    assert agg.loc["A", "priority_p90"] >= agg.loc["A", "priority_p50"]
