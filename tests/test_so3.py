"""Tests for SO(3) ops in gtsim.so3."""

import numpy as np
import pytest

from gtsim.so3 import (
    exp_so3,
    integrate_R,
    is_rotation,
    log_so3,
    project_so3,
    skew,
    vee,
)


def test_skew_vee_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        w = rng.normal(size=3)
        np.testing.assert_allclose(vee(skew(w)), w, atol=1e-12)


def test_skew_correctness():
    # S(a) b == a x b
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, -1.0, 0.5])
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-12)


def test_exp_so3_zero():
    R = exp_so3(np.zeros(3))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_exp_so3_is_rotation():
    rng = np.random.default_rng(1)
    for _ in range(20):
        w = rng.normal(size=3) * 0.7
        R = exp_so3(w)
        assert is_rotation(R, atol=1e-9)


def test_exp_so3_known_axis():
    # Rotate pi/2 around +z: e_1 -> e_2
    R = exp_so3(np.array([0.0, 0.0, np.pi / 2]))
    e1 = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(R @ e1, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_exp_log_roundtrip_small():
    rng = np.random.default_rng(2)
    for _ in range(30):
        w = rng.normal(size=3) * 0.5  # bounded away from pi
        R = exp_so3(w)
        w2 = log_so3(R)
        np.testing.assert_allclose(w, w2, atol=1e-9)


def test_project_so3_recovers_R():
    rng = np.random.default_rng(3)
    R = exp_so3(rng.normal(size=3) * 0.4)
    R_noisy = R + 1e-3 * rng.normal(size=(3, 3))
    R_proj = project_so3(R_noisy)
    assert is_rotation(R_proj, atol=1e-9)
    # Should be close (but not identical) to R.
    err = np.linalg.norm(R_proj - R, ord="fro")
    assert err < 5e-3


def test_integrate_R_small_step():
    # Rotating a unit body-x vector around body-z by w*dt should match
    # a direct rotation matrix.
    R0 = np.eye(3)
    w = np.array([0.0, 0.0, 2.0])
    dt = 0.1
    R1 = integrate_R(R0, w, dt)
    expected = exp_so3(w * dt)
    np.testing.assert_allclose(R1, expected, atol=1e-12)
