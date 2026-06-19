"""R-A roughness wrapper — UMEP binding, NaN guard, flags, vectorization."""

import numpy as np
import pandas as pd

from src.morphometry.roughness import (
    DIRS,
    extrapolation_flags,
    patch_mean_lambda_f,
    roughness,
    roughness_vec,
)


def test_kanda_returns_physical_values():
    zd, z0 = roughness("Kan", zH=6, fai=0.4, pai=0.6, zMax=15, zSdev=2)
    assert np.isfinite(zd) and np.isfinite(z0)
    assert z0 > 0 and zd > 0


def test_kanda_displacement_can_exceed_mean_height():
    # the heterogeneity feature: tall outliers push zd above H_mean
    zd, _ = roughness("Kan", zH=6, fai=0.4, pai=0.6, zMax=20, zSdev=4)
    assert zd > 6


def test_z0_increases_with_frontal_area():
    z0_lo = roughness("Kan", 6, 0.1, 0.4, 15, 2)[1]
    z0_hi = roughness("Kan", 6, 0.5, 0.4, 15, 2)[1]
    assert z0_hi > z0_lo


def test_nan_guard():
    assert np.isnan(roughness("Kan", np.nan, 0.4, 0.6, 15, 2)[0])
    assert np.isnan(roughness("Kan", 6, 0.4, 0.0, 15, 2)[1])  # pai<=0 unphysical


def test_vectorized_matches_scalar():
    zH = np.array([6.0, 9.0])
    fai = np.array([0.4, 0.3])
    pai = np.array([0.6, 0.5])
    zMax = np.array([15.0, 20.0])
    sdev = np.array([2.0, 3.0])
    zd, z0 = roughness_vec("Kan", zH, fai, pai, zMax, sdev)
    for i in range(2):
        zd_i, z0_i = roughness("Kan", zH[i], fai[i], pai[i], zMax[i], sdev[i])
        assert np.isclose(zd[i], zd_i) and np.isclose(z0[i], z0_i)


def test_patch_mean_lambda_f_selects_radius():
    cells = pd.DataFrame({"centroid_x": [0.0, 10.0, 100.0],
                          "centroid_y": [0.0, 0.0, 0.0],
                          **{f"lambda_f_{d}": [0.2, 0.4, 9.0] for d in DIRS}})
    out = patch_mean_lambda_f(cells, cx=0, cy=0, radius=50)  # excludes the x=100 cell
    assert out["n_cells"] == 2
    assert abs(out["N"] - 0.3) < 1e-9  # mean of 0.2, 0.4


def test_extrapolation_flags():
    f = extrapolation_flags(zH=np.array([6.0, 6.0]), pai=np.array([0.6, 0.3]),
                            zMax=np.array([15.0, 15.0]), zSdev=np.array([2.0, 2.0]))
    assert bool(f["flag_pai_over_envelope"][0]) and not bool(f["flag_pai_over_envelope"][1])
    assert np.isfinite(f["kanda_X"]).all()
