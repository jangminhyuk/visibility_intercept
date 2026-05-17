"""Tests for the visibility geometry.

Most important: the finite-difference check that Eq. (8) is implemented
consistently with the analytical c_Omega and c_v terms.
"""

import numpy as np
import pytest

from gtsim.config import VisibilityParams
from gtsim.so3 import exp_so3
from gtsim.visibility import boresight_world, c_Omega, c_v, h_V, mu_V, eta_V


COS_TF = float(np.cos(np.deg2rad(30.0)))


def test_h_V_full_lock_when_centered():
    # If boresight points exactly at the target, h_V = 1 - cos(theta_F) > 0
    R = np.eye(3)
    b_c = np.array([1.0, 0.0, 0.0])
    p_D = np.zeros(3)
    p_A = np.array([5.0, 0.0, 0.0])
    val = h_V(R, b_c, p_D, p_A, COS_TF)
    assert val == pytest.approx(1.0 - COS_TF, abs=1e-9)


def test_h_V_negative_when_target_behind():
    R = np.eye(3)
    b_c = np.array([1.0, 0.0, 0.0])
    p_D = np.zeros(3)
    p_A = np.array([-5.0, 0.0, 0.0])
    val = h_V(R, b_c, p_D, p_A, COS_TF)
    assert val == pytest.approx(-1.0 - COS_TF, abs=1e-9)


def test_c_Omega_zero_when_target_on_boresight():
    R = np.eye(3)
    b_c = np.array([1.0, 0.0, 0.0])
    p_D = np.zeros(3)
    p_A = np.array([5.0, 0.0, 0.0])
    np.testing.assert_allclose(c_Omega(R, b_c, p_D, p_A), np.zeros(3), atol=1e-12)


def test_c_v_zero_when_relative_velocity_along_LOS():
    R = np.eye(3)
    b_c = np.array([1.0, 0.0, 0.0])
    p_D = np.zeros(3)
    p_A = np.array([5.0, 0.0, 0.0])
    v_D = np.array([1.0, 0.0, 0.0])
    v_A = np.array([3.0, 0.0, 0.0])
    # v = v_A - v_D = (2, 0, 0); aligned with r_hat, so P_perp v = 0.
    assert c_v(R, b_c, p_D, p_A, v_D, v_A) == pytest.approx(0.0, abs=1e-12)


def test_h_V_finite_difference_matches_analytical():
    """The headline correctness check: numerical h_V_dot ≈ c_Omega^T Omega + c_v.

    We pick a state with non-trivial bore-sight error, lateral relative
    velocity, and non-zero body rate, then propagate dt=1e-5 forward.
    """
    rng = np.random.default_rng(0)
    for trial in range(20):
        # Random small rotation.
        R = exp_so3(rng.normal(size=3) * 0.3)
        b_c = np.array([1.0, 0.0, 0.0])
        p_D = rng.normal(size=3) * 2.0
        # Place attacker at modest distance with random direction.
        rel = rng.normal(size=3)
        rel = rel * (5.0 / max(1e-9, np.linalg.norm(rel)))
        p_A = p_D + rel
        v_D = rng.normal(size=3) * 0.5
        v_A = rng.normal(size=3) * 0.5
        Omega = rng.normal(size=3) * 1.5
        # Analytical h_V_dot
        co = c_Omega(R, b_c, p_D, p_A)
        cv = c_v(R, b_c, p_D, p_A, v_D, v_A)
        h_dot_analytic = float(co @ Omega + cv)
        # Finite-difference: propagate state by tiny dt with body-frame
        # body rate Omega (so R' = R exp(S(Omega) dt)) and translation
        # by v_D dt for defender, v_A dt for attacker.
        dt = 1e-6
        R_new = R @ exp_so3(Omega * dt)
        p_D_new = p_D + v_D * dt
        p_A_new = p_A + v_A * dt
        h0 = h_V(R, b_c, p_D, p_A, COS_TF)
        h1 = h_V(R_new, b_c, p_D_new, p_A_new, COS_TF)
        h_dot_fd = (h1 - h0) / dt
        # Should agree to ~5 decimal places.
        assert abs(h_dot_fd - h_dot_analytic) < 5e-4, (
            f"trial {trial}: fd={h_dot_fd:.6e}, analytic={h_dot_analytic:.6e}"
        )


def test_mu_V_signs():
    vp = VisibilityParams(alpha_V=2.0, zeta_V=0.05)
    R = np.eye(3)
    b_c = np.array([1.0, 0.0, 0.0])
    p_D = np.zeros(3)
    p_A = np.array([5.0, 0.0, 0.0])
    v_D = np.zeros(3)
    v_A = np.zeros(3)
    Omega_max = 8.0
    val = mu_V(R, b_c, p_D, p_A, v_D, v_A, vp, Omega_max, COS_TF)
    expected = 2.0 * (1.0 - COS_TF) - 0.05
    assert val == pytest.approx(expected, abs=1e-9)


def test_eta_V_is_mu_V_dot_h_V_substitute():
    """eta_V at a sampled Omega is c_Omega^T Omega + c_v + alpha_V h_V - zeta_V,
    which is the same expression as mu_V except the support-fn term is
    replaced by the actual <c_Omega, Omega>.  When ||Omega||_inf = Omega_max
    and Omega is aligned with sign(c_Omega), eta_V = mu_V exactly."""
    vp = VisibilityParams(alpha_V=3.0, zeta_V=0.02)
    rng = np.random.default_rng(7)
    Omega_max = 6.0
    for _ in range(10):
        R = np.eye(3)  # boresight = b_c
        b_c = rng.normal(size=3)
        b_c /= np.linalg.norm(b_c)
        p_D = rng.normal(size=3)
        p_A = p_D + 5.0 * rng.normal(size=3)
        v_D = rng.normal(size=3) * 0.3
        v_A = rng.normal(size=3) * 0.3
        co = c_Omega(R, b_c, p_D, p_A)
        # Best Omega: Omega_max * sign(co)  (l-infinity support).
        Omega_best = Omega_max * np.sign(co)
        mu = mu_V(R, b_c, p_D, p_A, v_D, v_A, vp, Omega_max, COS_TF)
        eta = eta_V(R, b_c, p_D, p_A, v_D, v_A, Omega_best, vp, COS_TF)
        assert eta == pytest.approx(mu, abs=1e-9)
