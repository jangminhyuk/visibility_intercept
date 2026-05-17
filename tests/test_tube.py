"""Tests for the intruder prediction tube (Eqs. 11-15)."""

import numpy as np
import pytest

from gtsim.config import E3, TubeParams
from gtsim.tube import (
    attitude_consistent_accel,
    initial_sigma_est,
    nominal_thrust_axis,
    propagate_sigma_est,
    sample_accelerations,
    tube_covariance,
)


def test_nominal_thrust_axis_identity_attitude():
    np.testing.assert_allclose(nominal_thrust_axis(np.eye(3)), E3, atol=1e-12)


def test_attitude_consistent_accel_recovers_scalar():
    """If a_hat = -g e_3 + sigma R e_3, the projection recovers sigma exactly."""
    g = 9.81
    sigma_true = 12.0
    R = np.eye(3)
    a_hat = -g * E3 + sigma_true * (R @ E3)
    sigma_bar, a_bar, w_perp = attitude_consistent_accel(
        a_hat, R, g, sigma_A_max=30.0
    )
    assert sigma_bar == pytest.approx(sigma_true, abs=1e-9)
    np.testing.assert_allclose(a_bar, a_hat, atol=1e-9)
    np.testing.assert_allclose(w_perp, np.zeros(3), atol=1e-9)


def test_attitude_consistent_accel_orthogonal_residual_recovered():
    """When a_hat has an off-axis component, w_perp must contain it (so we can
    fold it back into Sigma^est instead of throwing it away)."""
    g = 9.81
    R = np.eye(3)
    # Aligned thrust 10 + extra horizontal accel.
    extra = np.array([3.0, 0.0, 0.0])
    a_hat = -g * E3 + 10.0 * E3 + extra
    sigma_bar, a_bar, w_perp = attitude_consistent_accel(
        a_hat, R, g, sigma_A_max=30.0
    )
    assert sigma_bar == pytest.approx(10.0, abs=1e-9)
    np.testing.assert_allclose(w_perp, extra, atol=1e-9)


def test_attitude_consistent_accel_clip_low():
    """Negative inferred sigma must clip to zero (no negative thrust)."""
    g = 9.81
    R = np.eye(3)
    a_hat = -2.0 * g * E3  # implies sigma = -g, must clip to 0
    sigma_bar, a_bar, _ = attitude_consistent_accel(
        a_hat, R, g, sigma_A_max=30.0
    )
    assert sigma_bar == 0.0
    np.testing.assert_allclose(a_bar, -g * E3, atol=1e-9)


def test_tube_covariance_inflation_perp_dominates():
    tp = TubeParams(sigma_perp=5.0, sigma_par=1.0, sigma_est_floor=0.0)
    Sigma_est = np.zeros((3, 3))
    r_bar = np.array([1.0, 0.0, 0.0])
    S = tube_covariance(Sigma_est, r_bar, tp)
    # Diagonal: along r_bar = 1; perpendicular = 25.
    np.testing.assert_allclose(np.diag(S), [1.0, 25.0, 25.0], atol=1e-9)


def test_initial_sigma_est_folds_w_perp():
    tp = TubeParams(sigma_est_floor=0.5)
    w_perp = np.array([3.0, 4.0, 0.0])  # |w|^2 = 25
    S = initial_sigma_est(tp, w_perp)
    np.testing.assert_allclose(np.diag(S), [25.5, 25.5, 25.5], atol=1e-9)


def test_sample_accelerations_respect_a_max():
    rng = np.random.default_rng(0)
    a_bar = np.array([1.0, 0.0, 0.0])
    Sigma_seq = [10.0 * np.eye(3) for _ in range(5)]
    out = sample_accelerations(
        a_bar=a_bar, Sigma_seq=Sigma_seq,
        a_A_max=5.0, chi_a=2.0, M=20, rng=rng,
    )
    mags = np.linalg.norm(out, axis=-1)
    assert (mags <= 5.0 + 1e-6).all(), f"max |a|={mags.max()}"


def test_propagate_sigma_est_grows_diagonally():
    Sigma0 = np.eye(3)
    S1 = propagate_sigma_est(Sigma0, k=10, dt=0.05, Q_white=4.0)
    # +k*dt*Q = +10*0.05*4 = 2 on each diagonal.
    np.testing.assert_allclose(np.diag(S1), [3.0, 3.0, 3.0], atol=1e-9)
