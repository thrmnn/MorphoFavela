import math

import pandas as pd

from src.brisa_solar.wp07_crosstab import crosstab


def test_crosstab_classes_and_deficits():
    table = pd.DataFrame({
        "share_ge_2h_winter": [1.0, 0.0, 0.5, 0.25, None],
        "sun_h_winter_p50": [5.0, 0.0, 2.0, 1.0, None],
        "n_constraints": [0, 0, 1, 3, 2],
    })
    rows = {r["n_constraints"]: r for r in crosstab(table)}
    assert [rows[k]["cells"] for k in range(4)] == [2, 1, 0, 1]
    assert rows[0]["point_deficit"] == 0.5
    assert rows[0]["cell_deficit"] == 0.5
    assert rows[1]["cell_deficit"] == 0.0
    assert rows[3]["point_deficit"] == 0.75
    assert math.isnan(rows[2]["point_deficit"])
    assert sum(r["share_of_cells"] for r in rows.values()) == 1.0
