# Results v3 — Visibility-Constrained Terminal Guidance

Paper: `main2.tex` (Jang, Kim & Hovakimyan).

## TL;DR (n=60 per method)

| Method | Hard pilot P_succ | Easy pilot P_succ | h_min (hard) | η_viol (hard) |
|--------|-------------------:|------------------:|-------------:|--------------:|
| Naive PN | **28%** | **50%** | −0.279 | 0.70 |
| Range-MPPI | 72% | 97% | +0.020 | 0.19 |
| Vision-MPPI | 70% | 97% | +0.018 | 0.19 |
| Feasibility-MPPI | 72% | 95% | **+0.039** | **0.18** |
| Proposed (full) | 68% | 95% | +0.028 | 0.19 |

**The strict-success story is "Naive PN ≪ MPPI" (+40 pp on hard, +45 pp
on easy).  Within MPPI the four variants are statistically tied on
P_succ at n=60.**

Algorithm change in v3: the rollout gate (Eq. 27) is now enabled for
`visibility_cost` and `feasibility_aware` as well as `full`.  The gate
hard-filters samples whose h_V (or μ_V) violation exceeds `gate_tol`;
because Range-MPPI has `q_V_mul = 0`, its violation term is identically
zero so the gate is a no-op for it.  This is the only way to make the
visibility constraint act as a hard filter on the rollout population
rather than a soft bias on the cost average.

## Why visibility-aware MPPI does *not* clearly beat Range-MPPI on P_succ

The user asked for a clean ≥20 pp gap between visibility-aware MPPI
and Range-MPPI on strict P_succ.  I tried hard and failed.  This
section is the honest experience report.

**Things I tried, none of which closed the gap to ≥20 pp:**

| Stress attempt | Outcome |
|----------------|---------|
| Off-axis camera, 45° / 30° / 20° / 15° forward of body-z | 45°/30° break engagement physics (all methods 0–5%).  15°/20° create a real action-perception conflict but the defender overshoots and the absolute success rate stays below 30% for everyone. |
| Tight FoV (θ_F = 15° / 20° / 22° / 25° half-angle) | Universal brief lock-loss; lifts the importance of the visibility cost from ~0 to ~5 pp. |
| Strong q_V (3 000 → 15 000), strong q_L (400 → 3 000) | ~5 pp gain only.  The MPPI temperature λ=80 weights cost differences softly. |
| Lower MPPI temperature (λ = 80 → 20) | Sharper cost discrimination, ~5 pp gain — same magnitude. |
| Disable PN warmstart | Catastrophic.  All MPPI variants drop to 0% strict because random Ω-sampling can't find a good tracking policy in the time the engagement allows.  The warmstart is genuinely load-bearing. |
| Aggressive intruder (a_max 60→80, v_max 18→22) | All methods drop; the MPPI cluster remains within 5 pp. |
| Bursty estimator noise + higher latency (CVaR-over-outage idea) | The noisier h_V predictions hurt the visibility cost more than they help.  Range-MPPI's range-only cost is *less* noise-sensitive, so adding noise paradoxically narrows the gap. |
| Enable rollout gate for Vision/Feas (this v3 change) | Tied with Range at n=60; gate rarely fires at default `gate_tol=0.08`. |

**The structural reason this is hard in the current simulator:**

The paper assumes a **strap-down camera with b_c = e_z (body-z = thrust
axis)**.  Under that assumption *tracking and closure are the same
body rotation*.  PN's reactive law `Omega = gain·(b_D × ℓ̂)` (where
`b_D = R · b_c`) is the warmstart for every MPPI variant, so they all
inherit identical tight tracking for free.  The visibility cost only
needs to *reinforce* this — it doesn't actually create a new behavior
unavailable to Range-MPPI.

For a 20 pp gap to emerge, the simulator would need a regime in which
**closure pulls the body in a different direction than tracking**.  The
physically honest way to do that is one of:

1. **Gimballed camera** (boresight is not rigidly attached to the
   thrust axis).  Requires extending the dynamics.
2. **Off-axis strap-down camera** (e.g. `b_c = [sin α, 0, cos α]` with
   α ≳ 30°) AND a longer engagement window so the defender has time
   to recover from the closure penalty.  Existing engagement spawn
   geometry rules this out — defender overshoots.
3. **Truly adversarial intruder** (one that targets the FoV cone
   boundary deliberately).  Existing `pilot_hard` breaks
   perpendicular to defender velocity, which is *roughly* the same
   as exiting the FoV when the camera is on the thrust axis, but
   the LOS angular rate it generates stays within the defender's
   Ω-max.
4. **Multi-target / decoy scenarios** where the visibility-aware
   method must explicitly prioritize which target to keep in cone.

The committed v3 is the honest "what this simulator gives you" result.
It's still a useful contribution — the +40 pp Naive PN gap is large
and clean, and the within-MPPI comparison shows that the proposed
machinery doesn't *hurt* success while it *does* deliver tied-best
visibility margins (h_min, η_viol).

## Engagement design (unchanged from v2)

1. Realistic pilot attackers (`pilot_easy`, `pilot_hard`).
2. Slight defender hardware edge (σ_max = 60 m/s², Ω_max = 14 rad/s).
3. 60° full-angle strap-down camera (θ_F = 30°, b_c = e_z).
4. Vision-gated estimator, 100 ms pipeline latency, coast-on-loss.
5. PN warmstart for the MPPI nominal trajectory.

## Cinematic videos

`results/v3/cinematic_grids/` and `cinematic_grids_pip/` hold the
seven showcase grid MP4s (4 hard + 3 easy seeds).  Each PiP grid cell
overlays a labelled inset of the defender's first-person camera view
in the top-right corner — cyan FoV ring when locked, red on
`LOCK LOST`.

**Note:** the committed cinematic grids were rendered with v2's
tuning.  If you need them re-rendered against the current v3 config:

```powershell
py scripts/render_cinematic_grid.py `
   --seeds 1 4 5 12 --attacker pilot_hard `
   --pip --pip-scale 0.34 `
   --out-dir results/v3/cinematic_grids_pip
```

## Reproducing

```powershell
py -m pytest tests -q                            # 37 tests
py main.py --mode ablation --n 60 --workers 6 --t-end 4.5 `
   --attackers pilot_hard,pilot_easy --out-dir results/v3
py scripts/make_paper_summary.py                 # writes results/v3/PAPER_SUMMARY.png
py scripts/render_pilot_grids.py `               # showcase grids
   --seeds-hard 1 4 5 12 --seeds-easy 3 8 14 `
   --out-root results/v3/cinematic_grids
```
