"""Per-step records + per-run summary for the v2 visibility-feasibility sim.

Adds eta_V, mu_V, the planner-stressed flag, and the event-time fields
tau_I, tau_B, tau_L.  Computes the per-run aggregates used by the paper:
    h_min, mu_min, eta_min, eta_viol_integral, frac time h_V<0, mu_V<0,
    eta_V<0, time-to-first-loss, time-to-intercept.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepRecord:
    t: float = 0.0
    p_D: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    v_D: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    p_A: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    v_A: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    a_A_true: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    a_A_hat: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    sigma_mppi: float = 0.0
    Omega_mppi: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    Omega_applied: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rho: float = 0.0
    h_V: float = 0.0
    mu_V: float = 0.0
    eta_V: float = 0.0
    attacker_hvu_dist: float = 0.0
    outcome: str = "in_progress"
    R_D_flat: list = field(default_factory=lambda: [1, 0, 0, 0, 1, 0, 0, 0, 1])
    plan_debug_idx: int = -1
    planner_stressed: bool = False


@dataclass
class RunSummary:
    outcome: str = "in_progress"
    end_time: float = 0.0
    n_steps: int = 0
    # Range metrics
    min_rho: float = float("inf")
    min_attacker_hvu_dist: float = float("inf")
    # Visibility metrics (paper v2 Eq. 29)
    min_h_V: float = float("inf")
    min_mu_V: float = float("inf")
    min_eta_V: float = float("inf")
    mean_h_V: float = 0.0
    mean_mu_V: float = 0.0
    mean_eta_V: float = 0.0
    eta_viol_integral: float = 0.0          # int [-eta_V]_+ dt
    fraction_h_V_negative: float = 0.0
    fraction_mu_V_negative: float = 0.0
    fraction_eta_V_negative: float = 0.0
    fraction_planner_stressed: float = 0.0
    # Terminal events (paper v2 Eq. 6).  NaN if never occurred.
    tau_I: float = float("nan")
    tau_B: float = float("nan")
    tau_L: float = float("nan")
    success: bool = False                   # tau_I <= min(tau_B, tau_L)


class MetricsLog:
    def __init__(self) -> None:
        self.records: list[StepRecord] = []
        self.summary: RunSummary = RunSummary()

    def record(self, rec: StepRecord) -> None:
        self.records.append(rec)
        s = self.summary
        s.min_rho = min(s.min_rho, rec.rho)
        s.min_h_V = min(s.min_h_V, rec.h_V)
        s.min_mu_V = min(s.min_mu_V, rec.mu_V)
        s.min_eta_V = min(s.min_eta_V, rec.eta_V)
        s.min_attacker_hvu_dist = min(s.min_attacker_hvu_dist,
                                      rec.attacker_hvu_dist)

    def finalize(self, outcome: str, end_time: float,
                 tau_I: float = float("nan"),
                 tau_B: float = float("nan"),
                 tau_L: float = float("nan")) -> RunSummary:
        s = self.summary
        s.outcome = outcome
        s.end_time = end_time
        s.n_steps = len(self.records)
        s.tau_I = tau_I
        s.tau_B = tau_B
        s.tau_L = tau_L

        # success := tau_I <= min(tau_B, tau_L)
        finite = [t for t in (tau_B, tau_L)
                  if not (t is None or math.isnan(t))]
        min_adverse = min(finite) if finite else math.inf
        if not math.isnan(tau_I) and tau_I <= min_adverse:
            s.success = True

        if self.records:
            n = len(self.records)
            h = [r.h_V for r in self.records]
            mu = [r.mu_V for r in self.records]
            eta = [r.eta_V for r in self.records]
            stressed = [1.0 if r.planner_stressed else 0.0
                        for r in self.records]
            s.mean_h_V = float(sum(h) / n)
            s.mean_mu_V = float(sum(mu) / n)
            s.mean_eta_V = float(sum(eta) / n)
            s.fraction_h_V_negative = float(
                sum(1 for x in h if x < 0.0) / n)
            s.fraction_mu_V_negative = float(
                sum(1 for x in mu if x < 0.0) / n)
            s.fraction_eta_V_negative = float(
                sum(1 for x in eta if x < 0.0) / n)
            s.fraction_planner_stressed = float(sum(stressed) / n)
            # Trapezoidal-ish: dt is constant dt_sim.
            if n >= 2:
                dt = self.records[1].t - self.records[0].t
            else:
                dt = 0.0
            s.eta_viol_integral = float(sum(max(-x, 0.0) for x in eta) * dt)
        return s

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": asdict(self.summary),
            "records": [asdict(r) for r in self.records],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        s = self.summary
        print("--- Run summary ---")
        print(f"  outcome:              {s.outcome}")
        print(f"  success (tau_I min):  {s.success}")
        print(f"  end_time (s):         {s.end_time:.3f}")
        print(f"  n_steps:              {s.n_steps}")
        print(f"  min rho (m):          {s.min_rho:.3f}")
        print(f"  min attacker->HVU(m): {s.min_attacker_hvu_dist:.3f}")
        print(f"  min h_V:              {s.min_h_V:+.3f}")
        print(f"  min mu_V:             {s.min_mu_V:+.3f}")
        print(f"  min eta_V:            {s.min_eta_V:+.3f}")
        print(f"  mean h_V:             {s.mean_h_V:+.3f}")
        print(f"  eta_viol integral:    {s.eta_viol_integral:.4f}")
        print(f"  frac h_V<0:           {s.fraction_h_V_negative:.3f}")
        print(f"  frac mu_V<0:          {s.fraction_mu_V_negative:.3f}")
        print(f"  frac eta_V<0:         {s.fraction_eta_V_negative:.3f}")
        print(f"  frac planner stress:  {s.fraction_planner_stressed:.3f}")
        if not math.isnan(s.tau_I):
            print(f"  tau_I (s):            {s.tau_I:.3f}")
        if not math.isnan(s.tau_L):
            print(f"  tau_L (s):            {s.tau_L:.3f}")
        if not math.isnan(s.tau_B):
            print(f"  tau_B (s):            {s.tau_B:.3f}")
