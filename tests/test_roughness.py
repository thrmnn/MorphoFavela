"""R-A roughness wrapper — UMEP binding, NaN guard, flags, vectorization."""

import numpy as np
import pandas as pd

from src.morphometry.roughness import (
    DIRS,
    Z0_FLOOR_M,
    extrapolation_flags,
    floor_z0,
    patch_mean_lambda_f,
    roughness,
    roughness_vec,
    z0_was_floored,
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


# --- z0 floor (validate-first CFD-inlet core) ---
# The floor must be NaN-safe: it exists to rescue the empty/skimming case where the
# raw estimate is NaN or ~0, and np.maximum(nan, floor) would silently return nan.

def test_floor_rescues_nan():
    assert floor_z0(np.nan) == Z0_FLOOR_M          # the np.maximum trap
    assert z0_was_floored(np.nan) is True


def test_floor_rescues_skimming_collapse():
    assert floor_z0(0.0) == Z0_FLOOR_M             # z0→0 at λp>0.5
    assert floor_z0(0.001) == Z0_FLOOR_M
    assert z0_was_floored(0.001) is True


def test_floor_passes_through_valid_z0():
    assert floor_z0(0.25) == 0.25                  # a rough, valid patch is untouched
    assert z0_was_floored(0.25) is False


def test_floor_is_scalar_in_scalar_out_array_in_array_out():
    assert isinstance(floor_z0(0.5), float)
    out = floor_z0(np.array([np.nan, 0.0, 0.25]))
    assert out.shape == (3,)
    assert list(out) == [Z0_FLOOR_M, Z0_FLOOR_M, 0.25]
    flags = z0_was_floored(np.array([np.nan, 0.0, 0.25]))
    assert list(flags) == [True, True, False]
