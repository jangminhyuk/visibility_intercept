"""Smoke tests for the MPPI planner and a short sim run."""

import numpy as np

from gtsim.config import SimConfig
from gtsim.dynamics import DefenderState
from gtsim.mppi import PlannerState, plan
from gtsim.world import run_headless


def test_planner_returns_admissible_command():
    cfg = SimConfig()
    cfg.mppi.K = 16
    cfg.mppi.M = 4
    cfg.mppi.N = 6
    state = PlannerState.from_config(cfg)
    defender = DefenderState(
        p=np.array([5.0, 0.0, 4.0]),
        v=np.zeros(3),
        R=np.eye(3),
    )
    p_A = np.array([15.0, 0.0, 4.0])
    v_A = np.array([-5.0, 0.0, 0.0])
    R_A = np.eye(3)
    a_A = np.zeros(3)
    rng = np.random.default_rng(0)
    sigma_0, Omega_0, new_state, dbg, stressed = plan(
        cfg, state, defender, p_A, v_A, R_A, a_A, rng, collect_debug=True
    )
    assert 0.0 <= sigma_0 <= cfg.defender.sigma_max + 1e-9
    assert np.max(np.abs(Omega_0)) <= cfg.defender.Omega_max + 1e-9
    assert dbg is not None
    assert dbg.p_D.shape == (cfg.mppi.K, cfg.mppi.N + 1, 3)
    assert dbg.p_A.shape == (cfg.mppi.M, cfg.mppi.N + 1, 3)
    assert dbg.weights.shape == (cfg.mppi.K,)
    np.testing.assert_allclose(dbg.weights.sum(), 1.0, atol=1e-6)


def test_short_headless_run_terminates():
    """A short straight-line scenario should run to completion without
    crashing and produce a known outcome."""
    cfg = SimConfig()
    cfg.sim.t_end = 1.0       # short
    cfg.mppi.K = 16
    cfg.mppi.M = 4
    cfg.mppi.N = 6
    cfg.scenario.attacker_mode = "straight"
    cfg.scenario.use_mppi = True
    cfg.scenario.planner_method = "full"
    world = run_headless(cfg, collect_plan_debug_steps=0)
    assert world.outcome in {"intercept", "breach", "visual_loss", "timeout"}
    assert len(world.metrics.records) > 0
