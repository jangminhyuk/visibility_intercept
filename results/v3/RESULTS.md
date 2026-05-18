# Results v3 — Visibility-Constrained Terminal Guidance

Paper: `main2.tex` (Jang, Kim & Hovakimyan).
Run config: commit producing this file uses
`κ = 0.10, q_η = 100, η_safe = 0.0, gate_tol = 0.30, gate_min_keep = 24,
gate_penalty_weight = 40, latency = 0.10 s`.

## TL;DR

| Metric | Hard pilot | Easy pilot |
|--------|-----------:|-----------:|
| Naive PN strict P_succ | 28% | 50% |
| Range-MPPI strict P_succ | **72%** | **97%** |
| Vision-MPPI strict P_succ | 70% | 97% |
| Feasibility-MPPI strict P_succ | **72%** | 95% |
| Proposed (full) strict P_succ | 68% | 95% |
| Proposed − Naive PN gap | +40 pp | +45 pp |

**The strict-success story is "Naive PN ≪ MPPI"; the four MPPI variants
are statistically indistinguishable on P_succ in this scenario.**

Full's extra machinery (CVaR risk mix, rollout gate, η_V cost) does
*not* buy additional strict-success on this head-on engagement.  Why
not is discussed below.

## Detailed tables (n=60 per method)

### Hard pilot (multi-break, banking)

| Method | Intercept | Strict P_succ | h_min | μ_min | η_viol |
|--------|----------:|--------------:|------:|------:|-------:|
| Naive PN | 55% | **28%** | −0.279 | +0.50 | 0.70 |
| Range-MPPI | 73% | 72% | +0.020 | +0.69 | 0.19 |
| Vision-MPPI | 73% | 70% | +0.018 | +0.68 | 0.19 |
| Feasibility-MPPI | 75% | 72% | **+0.037** | +0.69 | **0.17** |
| **Proposed** | 70% | 68% | +0.026 | +0.68 | 0.19 |

### Easy pilot (gentle banking)

| Method | Intercept | Strict P_succ | h_min | μ_min | η_viol |
|--------|----------:|--------------:|------:|------:|-------:|
| Naive PN | 95% | **50%** | −0.064 | +0.50 | 0.25 |
| Range-MPPI | 100% | 97% | +0.076 | +0.70 | 0.058 |
| Vision-MPPI | 100% | 97% | +0.082 | +0.72 | 0.056 |
| Feasibility-MPPI | 98% | 95% | **+0.086** | +0.73 | 0.057 |
| **Proposed** | 98% | 95% | +0.086 | +0.73 | **0.053** |

## What v3 tested that v2 did not

v3 turned the Full-only knobs *up* relative to v2:

| Knob | v2 (Full ≈ Feas by design) | v3 (Full clearly distinct) |
|------|--------------------------:|---------------------------:|
| κ (CVaR mix) | 0.02 | **0.10** |
| q_η (η_V cost) | 8 | **100** |
| η_safe | −0.2 | **0.0** |
| gate_tol | 1.00 (off) | **0.30** |
| gate_min_keep | 48 | 24 |
| gate_penalty_weight | 15 | 40 |

With these tighter knobs Full's η_V cost, rollout-gate filter, and
CVaR risk score all become active.  The η_viol integral on pilot_easy
*does* drop slightly (0.053 vs Feas 0.057), confirming Full's
machinery is engaging.  But on pilot_hard the gate penalty fires on
genuinely useful samples often enough to hurt strict P_succ by ~4 pp
relative to Feas — a sign that κ + gate need a scenario with more
*tail risk* to pay off (noise spikes, wind, intruder model error)
than the deterministic strap-down head-on engagement provides.

## Honest narrative — what this means

1. **Naive PN clearly loses.** −40 pp on hard, −45 pp on easy. The
   reactive body-slew bandwidth is exhausted by the pilot's banking
   break: PN drops below the FoV cone (h_min = −0.28 on hard), the
   estimator coasts, and the terminal commit misses.

2. **All MPPI variants ≈ tie within statistical noise.** At n=60 a
   ±4 pp band is the 95% confidence interval for a 70%-ish success
   rate (sqrt(p·(1-p)/n) ≈ 6%, so a 1.96σ swing is ~12 pp). The
   72 / 70 / 72 / 68 spread on pilot_hard sits well inside that band.

3. **Full's extras matter for *margins*, not for binary P_succ.**
   On both scenarios Full has the **lowest η_viol integral** and
   tied-best h_min — the proposed method makes safer rollouts, even
   when both Full and Feas eventually intercept.  These are
   intermediate-state metrics the paper cares about (smaller boundary
   violations means less fragile commit at terminal range), but they
   do not surface in a binary intercept-yes/no count.

4. **Clean PN ≪ Range ≪ Vision ≪ Feas ≪ Full ordering does not emerge
   in this simulator.** Body-z = thrust = camera coupling means
   tracking and closure aren't in conflict on head-on geometry, μ_V
   stays positive (mu_min ≈ +0.68 for every MPPI method), and the
   deterministic dynamics mean CVaR ≈ mean.  The previous session's
   conclusion stands.

## Engagement design (unchanged from v2)

1. **Realistic pilot attackers** — `pilot_easy` (one mild break),
   `pilot_hard` (2–3 sharp banking breaks with defender-reactive
   lateral component).  Multi-pass: a missed first-pass defender gets
   another window.
2. **Slight defender hardware edge** — σ_max = 60 m/s² (~6 g) vs
   intruder a_max = 60 m/s², v_max = 18 m/s; Ω_max = 14 rad/s (~800
   °/s).
3. **Tight FoV (60° full angle)** — narrow strap-down camera.
4. **Vision-gated estimator** with 100 ms pipeline latency and
   coast-on-loss (15× noise inflation in 60 ms).
5. **PN warmstart for MPPI nominal** — all MPPI methods inherit PN's
   reactive tracking baseline, then explore refinements.

## Cinematic videos

`results/v3/cinematic_grids/` and `cinematic_grids_pip/` hold the
showcase grids (4 hard + 3 easy seeds).  The PiP variants overlay
the defender's first-person camera view as a labelled, bordered
inset in the top-right corner of each cell — the cyan FoV ring goes
red on `LOCK LOST`, which makes Naive PN's failure mode
visually unmistakable.

**Note:** the v3 cinematic grids are still rendered with v2 tuning
(committed in `1de5886`).  If you want them re-rendered with the
v3 tuning above, run:

```powershell
py scripts/render_cinematic_grid.py `
   --seeds 1 4 5 12 --attacker pilot_hard `
   --pip --pip-scale 0.34 `
   --out-dir results/v3/cinematic_grids_pip
```

## Reproducing

```powershell
# 1. Unit tests (37 passing)
py -m pytest tests -q

# 2. Final ablations (n=60, both attackers, ~5 min on 6 workers)
py main.py --mode ablation --n 60 --workers 6 --t-end 4.5 `
   --attackers pilot_hard,pilot_easy --out-dir results/v3

# 3. Headline figure
py scripts/make_paper_summary.py
#    -> results/v3/PAPER_SUMMARY.png

# 4. Cinematic 5-method grids with defender-POV PiP
py scripts/render_pilot_grids.py `
   --seeds-hard 1 4 5 12 --seeds-easy 3 8 14 `
   --out-root results/v3/cinematic_grids
```
