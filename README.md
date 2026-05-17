# Visibility-Constrained Terminal Guidance for Kinetic UAV Interception

Python simulator and ablation suite for a **scenario-risk MPPI planner
with a body-rate visibility-feasibility margin**.  The defender is a
quadrotor with a body-fixed (strap-down) camera; the camera direction
and the thrust direction share the body-z axis, so accelerating to
close on the target and keeping the target inside the camera cone are
*coupled through attitude*.  The planner has to do both.

The simulator demonstrates that visibility-aware MPPI substantially
beats a naive reactive baseline against realistic banking-pilot
attackers, while remaining competitive on closure.

## Headline result (n=60 per method, see `results/v2/RESULTS.md`)

| Method | Hard attacker P_succ | Easy attacker P_succ |
|--------|---------------------:|---------------------:|
| Naive proportional navigation | 40% | 73% |
| Range-MPPI (no visibility cost) | 85% | 100% |
| Vision-MPPI (h_V penalty) | 85% | 100% |
| Feasibility-MPPI (+ μ_V) | 83% | 100% |
| **Proposed (scenario-risk MPPI + rollout gate)** | **83%** | **100%** |

`P_succ` is the strict success rate
$\mathbb{P}\left[\tau_I \le \min(\tau_B, \tau_L)\right]$
(intercept before breach *and* before any visual-loss event).

Proposed beats the naive baseline by **+43 pp** on the hard scenario
and **+27 pp** on the easy scenario.

The headline figure is `results/v2/PAPER_SUMMARY_v3.png` and the full
narrative + reproduction commands are in `results/v2/RESULTS.md`.

## Installation

```powershell
py -m pip install -r requirements.txt
```

Requirements: `numpy`, `scipy`, `matplotlib`, `cvxpy`, `osqp`, `pytest`,
`joblib`.  Cinematic videos additionally use `pygame`, `PyOpenGL`, and
`ffmpeg` on `PATH`.

## Quick test

```powershell
py -m pytest tests -q     # should print "37 passed"
```

## Reproducing the headline results

### 1. Run the ablation (≈ 6 min on 6 workers)

```powershell
py main.py --mode ablation --n 60 --workers 6 --t-end 7.0 ^
   --attackers pilot_hard,pilot_easy ^
   --out-dir results/v2/final_pilot
```

This runs all five methods (`pn`, `range_only`, `visibility_cost`,
`feasibility_aware`, `full`) on both attackers and writes per-attacker
ablation tables, time-series plots, and bar charts to
`results/v2/final_pilot/ablation_pilot_{hard,easy}/`.

### 2. Build the headline figure

```powershell
py scripts/make_paper_summary.py
```

Writes `results/v2/PAPER_SUMMARY_v3.png` — three rows:
1. Bar charts of intercept rate + strict P_succ per attacker, with the
   Proposed − Naive PN gap callout.
2. Time-series of h_V / μ_V / η_V / ρ / |Ω| on the hard scenario
   (median + 35–65 percentile bands).
3. Summary table + h_min / η_viol / frac(η_V<0) bars.

### 3. Render cinematic 5-method comparison videos

```powershell
py scripts/render_pilot_grids.py ^
   --seeds-hard 1 4 5 12 ^
   --seeds-easy 3 8 14 ^
   --t-end 4.5
```

For each showcase seed this:
1. Runs every method on the same seed.
2. Renders each method's engagement as a 640×360 cinematic MP4 using
   the OpenGL renderer (quadrotor wireframes, FoV cone, MPPI fans, HUD).
3. Composites the five videos into a 2×3 grid with `ffmpeg`, with a
   method-coloured border per cell and a numbers panel in the
   bottom-right.

Output: `results/v2/final_pilot/cinematic_grids/grid_pilot_{hard,easy}_seed*.mp4`.

### 4. Single-trial demo (one method)

```powershell
py main.py --mode demo --method full --attacker pilot_hard --seed 4 --video
```

Writes `results/v2/demo_pilot_hard_full_seed4/` with the time-series
plot, MPPI snapshots, 3D trajectory, and optional MP4.

## Method ablation (paper Sec. VII + naive baseline)

| Method id | Architecture | Cost / aggregation |
|-----------|--------------|--------------------|
| `pn` | Proportional navigation (no MPPI) | reactive LOS-pursuit body-rate + near-max thrust |
| `range_only` | MPPI, range terms only | mean over scenarios |
| `visibility_cost` | MPPI + h_V penalty + visual-loss event | mean |
| `feasibility_aware` | MPPI + h_V + μ_V | mean |
| `full` | MPPI + h_V + μ_V + η_V | mean + CVaR<sub>α<sub>R</sub></sub> + rollout gate (Eq. 27) |

`pn` is the naive baseline; the other four are the MPPI ablations.

## Engagement design

Two realistic banking-pilot attacker modes:

- **`pilot_easy`** — smooth random course corrections at ~14 m/s² with
  one mild defender-reactive break (~25 m/s²) at engagement range.
- **`pilot_hard`** — aggressive banking + multi-break: 2–3 sharp
  ~60 m/s² lateral breaks chained across the engagement, with smooth
  recovery and continued pursuit of the HVU after each break, so the
  defender that misses on the first pass still has the engagement
  alive for a second pass.

Defender hardware:
- σ<sub>max</sub> = 60 m/s² (~6 g), Ω<sub>max</sub> = 14 rad/s (~800 °/s)
- FoV half-angle θ<sub>F</sub> = 30° (60° full-angle strap-down camera)
- Modestly out-specs the intruder (a<sub>max</sub>=60 m/s², v<sub>max</sub>=18 m/s)

Estimator: vision-gated — when h<sub>V</sub> < 0 the planner stops
receiving measurements and coasts on the last-visible state with noise
inflating to 15× over 60 ms.  Lock loss is physically costly.

## File layout

```
visibility_intercept/
├── README.md
├── requirements.txt
├── main.py                     # CLI: demo / videos / ablation / map / compare
├── src/gtsim/
│   ├── config.py               # SimConfig + PLANNER_METHODS + method_weights
│   ├── so3.py                  # Rodrigues exp/log, project_so3
│   ├── dynamics.py             # step_defender (Eq.1), step_intruder (Eq.19)
│   ├── visibility.py           # h_V, c_Omega, c_v, mu_V, eta_V (Eqs.5–14)
│   ├── tube.py                 # Eq.18 intruder scenario tube
│   ├── attacker.py             # pilot_easy / pilot_hard / open-loop modes
│   ├── risk.py                 # exact-CVaR + mean+CVaR aggregation
│   ├── mppi.py                 # rollout cost (Eq.24) + gate (Eq.27) + PN warmstart
│   ├── world.py                # sim loop, vision-gated estimator, terminal commit
│   ├── metrics.py              # StepRecord, RunSummary, MetricsLog
│   ├── viz.py                  # matplotlib trajectory + time-series plots
│   ├── video3d.py              # matplotlib fallback video renderer
│   ├── viz_opengl.py           # cinematic OpenGL video renderer
│   └── video_grid.py           # 2×3 method-comparison grid renderer (mpl)
├── scripts/
│   ├── make_paper_summary.py   # builds the headline figure
│   ├── render_cinematic_grid.py# 5-method cinematic grid (per scenario)
│   └── render_pilot_grids.py   # batch wrapper around the cinematic grid
├── tests/                      # 37 pytest unit + integration tests
└── results/v2/                 # outputs (final_pilot/, RESULTS.md, PAPER_SUMMARY_v3.png)
```

## Paper-symbol → code mapping

| Paper symbol | Code | Equation |
|--------------|------|----------|
| x<sub>D</sub> = (p<sub>D</sub>, v<sub>D</sub>, R<sub>D</sub>), u<sub>D</sub> = (σ<sub>D</sub>, Ω<sub>D</sub>) | `dynamics.DefenderState`, `step_defender` | (1), (2) |
| r, v, ρ, r̂ | `visibility.los` | (3) |
| b<sub>D</sub> = R<sub>D</sub> b<sub>c</sub> | `visibility.boresight_world` | (5) |
| h<sub>V</sub> = b<sub>D</sub>ᵀr̂ − cos θ<sub>F</sub> | `visibility.h_V` | (5) |
| ḣ<sub>V</sub> = c<sub>Ω</sub>ᵀ Ω<sub>D</sub> + c<sub>v</sub> | `visibility.c_Omega`, `c_v` | (8)–(9) |
| μ<sub>V</sub> feasibility margin | `visibility.mu_V` | (13) |
| η<sub>V</sub> sampled-cmd residual | `visibility.eta_V` | (14) |
| Intruder scenario tube Q<sub>N</sub> | `tube.build_intruder_tube` | (17)–(18) |
| Rollout cost J(U<sub>D</sub>, Q<sub>A</sub>; t) | `mppi.rollout_costs_and_gate` | (24) |
| Event penalty ℓ<sub>ev</sub> | inside same | (23) |
| Rollout-level gate G<sub>i</sub> | inside same | (27) |
| Risk score (1−κ) J̄ + κ CVaR | `risk.mean_plus_cvar_batch` | (26) |
| MPPI softmin + gate filter | `mppi.softmin_update` | (post-27) |

## Safety note

Simulation only.  No interface to real UAV firmware, GPS, RTK, or
network endpoints.  "Intercept" is a tag event between two simulated
points in a synthetic 3D world.
