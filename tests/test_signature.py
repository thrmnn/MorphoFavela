"""WS-A signature engine — feature assembly, k-elbow, labelling, recurrence."""

import numpy as np
import pandas as pd

from src.morphometry.signature import (
    SIGNATURE_FEATURES,
    assemble_signature_matrix,
    choose_k_elbow,
    fit_morphotypes,
    recurrence_flags,
    recurrence_matrix,
    standardize,
)


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
