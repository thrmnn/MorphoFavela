"""Guard the resolution-suffix → grid-path mapping in the dissolve migration.

The 20 m MAUP grids were once left on summed λf because the migration only ever
targeted the 10 m path, which made the MAUP A/B confound cell size with the
summed→dissolved over-count. This pins the suffix→path contract so that bug
cannot return silently.
"""

from scripts.migrate_lambda_f_dissolve import grid_path


def test_default_suffix_targets_10m_grid():
    p = grid_path("vidigal")
    assert p.parts[-3:] == ("morphometrics", "grid", "grid_metrics.gpkg")


def test_20m_suffix_targets_20m_grid():
    p = grid_path("vidigal", "_20m")
    assert p.parts[-3:] == ("morphometrics_20m", "grid", "grid_metrics.gpkg")


def test_suffix_changes_only_the_morphometrics_dir():
    base, twenty = grid_path("maré"), grid_path("maré", "_20m")
    assert base.parent.parent.name == "morphometrics"
    assert twenty.parent.parent.name == "morphometrics_20m"
    assert base.parents[2] == twenty.parents[2]  # same site dir
