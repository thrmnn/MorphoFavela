"""WS-A signature engine — feature assembly, k-elbow, labelling, recurrence."""

import numpy as np
import pandas as pd
import pytest

from src.morphometry.signature import (
    GRAIN_COL,
    SHAPE_DESCRIPTORS,
    SIGNATURE_FEATURES,
    _area_entropy,
    aggregate_shape_to_grid,
    assemble_signature_matrix,
    choose_k_elbow,
    fit_morphotypes,
    recurrence_flags,
    recurrence_matrix,
    standardize,
    vif_screen,
)

# Canonical 6-feature fabric vector — pinned as an exact ordered literal so the
# production GMM fit cannot silently drift (added/removed/reordered feature).
# Changing this requires a documented A/B in docs/morpho_signature_decisions.md.
CANONICAL_SIGNATURE_FEATURES = (
    "lambda_p",
    "H_mean",
    "sigma_h",
    "lambda_f_mean",
    "lambda_f_aniso",
    "slope_deg",
)


def test_signature_features_exact_order_pin():
    assert tuple(SIGNATURE_FEATURES) == CANONICAL_SIGNATURE_FEATURES


def _pooled():
    # 3 built cells (one single-building → sigma_h NaN) + 1 unbuilt (filtered out)
    return pd.DataFrame({
        "site": ["vidigal"] * 3 + ["rocinha"],
        "zone_id": [1, 2, 3, 4],
        "built_mask": [True, True, True, False],
        "lambda_p": [0.5, 0.7, 0.2, 0.0],
        "H_mean": [9.0, 12.0, 5.0, 0.0],
        "sigma_h": [2.0, np.nan, 1.0, np.nan],   # row 2 single-building
        "lambda_f_mean": [0.4, 0.6, 0.2, 0.0],
        "lambda_f_max": [0.9, 0.7, 0.5, 0.0],
        "slope_deg": [15.0, 20.0, 5.0, 0.0],
    })


def test_assemble_filters_built_derives_aniso_imputes_sigma():
    mat = assemble_signature_matrix(_pooled())
    assert len(mat) == 3                       # unbuilt cell dropped
    assert set(SIGNATURE_FEATURES) <= set(mat.columns)
    # lambda_f_aniso = max - mean, clipped >= 0
    assert np.isclose(mat.iloc[0]["lambda_f_aniso"], 0.5)
    # single-building sigma_h imputed to 0, so no rows dropped for NaN
    assert mat.iloc[1]["sigma_h"] == 0.0
    assert mat.attrs["n_dropped"] == 0


def test_standardize_zeromean_unitstd():
    mat = assemble_signature_matrix(_pooled())
    Xz, stats = standardize(mat)
    assert np.allclose(Xz.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Xz.std(axis=0), 1, atol=1e-9)
    assert list(stats["feature"]) == SIGNATURE_FEATURES


def test_choose_k_elbow_finds_the_knee():
    # BIC drops steeply to k=4 then flattens → elbow at 4
    sel = pd.DataFrame({
        "k": [2, 3, 4, 5, 6, 7],
        "bic": [100, 60, 40, 38, 37, 36.5],
    })
    assert choose_k_elbow(sel) == 4


def test_fit_morphotypes_separable_and_density_ordered():
    rng = np.random.default_rng(0)
    # three blobs separated on column 0 (lambda_p axis), ascending
    blobs = [rng.normal(m, 0.05, size=(50, 6)) for m in (-3, 0, 3)]
    for i, b in enumerate(blobs):
        b[:, 0] = rng.normal(i, 0.05, size=50)  # col0 ascending by group
    Xz = np.vstack(blobs)
    labels = fit_morphotypes(Xz, k=3, random_state=0)
    assert set(labels) == {0, 1, 2}
    # label 0 should be the lowest-col0 (sparsest) group
    means = [Xz[labels == c, 0].mean() for c in range(3)]
    assert means[0] < means[1] < means[2]


def test_recurrence_shares_sum_to_one_and_flags():
    mat = pd.DataFrame({
        "site": ["vidigal", "vidigal", "rocinha", "rocinha", "maré", "maré"],
        "zone_id": range(6),
    })
    labels = np.array([0, 1, 0, 1, 0, 0])
    shares = recurrence_matrix(mat, labels, sites=["vidigal", "rocinha", "maré"])
    assert np.allclose(shares.sum(axis=1), 1.0)
    flags = recurrence_flags(shares, min_share=0.05, min_sites=3)
    assert bool(flags.loc[0, "recurs"])          # type 0 present in all 3 sites
    assert not bool(flags.loc[1, "recurs"])       # type 1 only in 2 sites


# --- shape/grain re-fit (#3, additive) -------------------------------------


def test_area_entropy_uniform_is_zero_mixed_is_positive():
    import numpy as np
    assert _area_entropy(np.array([10.0, 10.0, 10.0]), n_bins=8) == 0.0
    assert _area_entropy(np.array([5.0, 50.0, 500.0]), n_bins=8) > 0.4
    assert _area_entropy(np.array([7.0]), n_bins=8) == 0.0   # <2 buildings → 0


def test_vif_screen_flags_collinear_feature():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    b = rng.normal(size=400)
    c = a + 0.001 * rng.normal(size=400)   # near-perfect copy of a → high VIF
    mat = pd.DataFrame({"a": a, "b": b, "c": c})
    vif = vif_screen(mat, ["a", "b", "c"], threshold=10.0)
    flagged = set(vif.loc[vif["drop"], "feature"])
    assert {"a", "c"} & flagged          # the collinear pair is caught
    assert "b" not in flagged            # the independent feature is kept


def test_aggregate_shape_area_weighted_and_support_flag():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point, box

    grid = gpd.GeoDataFrame(
        {"zone_id": [0, 1]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs="EPSG:31983",
    )
    # cell 0: two buildings (areas 1 and 9) → area-weighted shape_index
    # cell 1: one building → has_shape_support False (< MIN_BUILDINGS_FOR_SHAPE)
    blds = gpd.GeoDataFrame(
        {
            "shape_index": [2.0, 6.0, 4.0],
            "convexity": [1.0, 1.0, 1.0],
            "building_adjacency": [0.0, 0.0, 0.0],
            "elongation": [1.0, 1.0, 1.0],
            "tessellation_neighbors": [3.0, 3.0, 3.0],
            "fractal_dimension": [1.5, 1.5, 1.5],
        },
        geometry=[
            Point(5, 5).buffer(0.5642),   # area ≈ 1
            Point(5, 5).buffer(1.6926),   # area ≈ 9
            Point(15, 5).buffer(1.0),
        ],
        crs="EPSG:31983",
    )
    agg = aggregate_shape_to_grid(blds, grid)
    assert set(agg["zone_id"]) == {0, 1}
    c0 = agg[agg["zone_id"] == 0].iloc[0]
    # area-weighted mean of shape_index: (2*1 + 6*9)/10 = 5.6, well above the 4.0 mean
    assert c0["shape_index_mean"] > 5.0
    assert bool(c0["has_shape_support"])             # 2 buildings
    assert not bool(agg[agg["zone_id"] == 1].iloc[0]["has_shape_support"])  # 1 building
    assert GRAIN_COL in agg.columns
    assert set(f"{c}_mean" for c in SHAPE_DESCRIPTORS) <= set(agg.columns)
