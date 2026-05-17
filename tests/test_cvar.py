"""Tests for the exact-CVaR (Rockafellar-Uryasev) implementation.

The crucial check: the closed-form mixed-quantile result must equal the
numerical minimum of nu + 1/((1-alpha)M) sum max(J - nu, 0) over nu.
"""

import numpy as np
import pytest

from gtsim.risk import cvar_alpha, mean_plus_cvar, mean_plus_cvar_batch


def _rockafellar_numeric(J, alpha):
    """Brute-force minimisation over nu, used as a reference."""
    J = np.asarray(J, dtype=float)
    M = J.size
    grid = np.linspace(float(J.min()) - 5.0, float(J.max()) + 5.0, 5001)
    vals = grid + (1.0 / ((1.0 - alpha) * M)) * np.maximum(J[None, :] - grid[:, None], 0.0).sum(axis=1)
    return float(vals.min())


def test_cvar_alpha_integer_top_k():
    """When (1-alpha)*M is integer, CVaR_alpha equals top-k average."""
    J = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # alpha=0.7 -> (1-alpha)M = 3 -> mean of [8, 9, 10] = 9.
    val = cvar_alpha(J, alpha=0.7)
    assert val == pytest.approx(9.0, abs=1e-9)


def test_cvar_alpha_mixed_quantile():
    """When (1-alpha)*M is not integer, naive top-k mean would be wrong;
    the Rockafellar form gives the exact answer."""
    J = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # alpha = 0.75 -> aM = 7.5, k* = 8, (1-alpha)M = 2.5
    # CVaR = 1/2.5 * ((8 - 7.5) * J_(8) + J_(9) + J_(10))
    #      = 1/2.5 * (0.5 * 8 + 9 + 10) = 1/2.5 * 23 = 9.2
    val = cvar_alpha(J, alpha=0.75)
    assert val == pytest.approx(9.2, abs=1e-9)


def test_cvar_alpha_matches_numerical_rockafellar():
    rng = np.random.default_rng(0)
    for trial in range(15):
        M = rng.integers(8, 32)
        J = rng.uniform(0.0, 50.0, size=M)
        alpha = float(rng.uniform(0.1, 0.95))
        val = cvar_alpha(J, alpha)
        ref = _rockafellar_numeric(J, alpha)
        # Numerical reference is on a 5001-point grid: tolerance ~0.05.
        assert abs(val - ref) < 0.05, (
            f"trial {trial}: exact={val:.4f}, numeric={ref:.4f}, alpha={alpha}"
        )


def test_cvar_alpha_alpha_zero_is_mean():
    J = np.array([1.0, 2.0, 3.0, 4.0])
    assert cvar_alpha(J, alpha=0.0) == pytest.approx(2.5, abs=1e-9)


def test_cvar_alpha_alpha_one_is_max():
    J = np.array([1.0, 2.0, 7.0, 4.0])
    val = cvar_alpha(J, alpha=0.999999)
    assert val == pytest.approx(7.0, abs=1e-3)


def test_mean_plus_cvar_convex_combo():
    J = np.array([1.0, 2.0, 3.0, 4.0])
    # alpha=0.5 -> aM=2, k*=2, CVaR = 1/2 * ((2-2)*J_(2) + J_(3)+J_(4)) = 1/2*(3+4)=3.5
    # mean = 2.5; kappa=0.4 -> R = 0.6*2.5 + 0.4*3.5 = 1.5 + 1.4 = 2.9
    R = mean_plus_cvar(J, kappa=0.4, alpha=0.5)
    assert R == pytest.approx(2.9, abs=1e-9)


def test_mean_plus_cvar_batch_matches_loop():
    rng = np.random.default_rng(2)
    K, M = 7, 13
    J = rng.uniform(0.0, 20.0, size=(K, M))
    kappa, alpha = 0.6, 0.7
    R_batch = mean_plus_cvar_batch(J, kappa, alpha)
    R_loop = np.array([mean_plus_cvar(J[i], kappa, alpha) for i in range(K)])
    np.testing.assert_allclose(R_batch, R_loop, atol=1e-9)
