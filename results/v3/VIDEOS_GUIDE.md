# v3 video gallery — where to look

All videos sit under `results/v3/`.  Each cell of the grid videos
also includes a labelled defender-camera POV inset (cyan FoV ring
when locked, red on LOCK LOST).

## Cinematic 5-method comparison grids (with PiP defender POV)

These are the showcase videos for the paper.  Each is a 2×3 grid of
the five methods on the same seed of the same attacker, plus a
bottom-right summary panel.  Each cell has the defender's
first-person camera view as a labelled inset.

### Original "pilot" attackers

**Hard pilot** (multi-break banking; v2 tuning):
```
results/v3/cinematic_grids_pip/cinematic_grid_pip_seed1.mp4    (~2.6 MB)
results/v3/cinematic_grids_pip/cinematic_grid_pip_seed4.mp4    (~1.9 MB)   <- best demo
results/v3/cinematic_grids_pip/cinematic_grid_pip_seed5.mp4    (~1.1 MB)
results/v3/cinematic_grids_pip/cinematic_grid_pip_seed12.mp4   (~1.4 MB)
```

**Easy pilot** (gentle banking; v2 tuning):
```
results/v3/cinematic_grids_pip/cinematic_grid_pip_seed3.mp4    (~1.4 MB)
results/v3/cinematic_grids_pip/cinematic_grid_pip_seed8.mp4    (~2.0 MB)
results/v3/cinematic_grids_pip/cinematic_grid_pip_seed14.mp4   (~1.1 MB)
```

### New v3 FoV-exit attacker (camera-aware break, v3 tuning)

**Best paper figure candidate** — Naive PN and Range-MPPI both
breach (visible red LOCK LOST rings in their POV insets) while
Vision / Feas / Proposed all intercept:
```
results/v3/cinematic_grids_pip_fov_exit/cinematic_grid_pip_seed1.mp4
results/v3/cinematic_grids_pip_fov_exit/cinematic_grid_pip_seed4.mp4    <- recommended showcase
results/v3/cinematic_grids_pip_fov_exit/cinematic_grid_pip_seed5.mp4
```

## Third-person only (no PiP)

If you want a cleaner third-person view without the POV inset:
```
results/v3/cinematic_grids/grid_pilot_hard_seed{1,4,5,12}.mp4
results/v3/cinematic_grids/grid_pilot_easy_seed{3,8,14}.mp4
```

## How to render more

Single demo with PiP:
```powershell
py main.py --mode demo --method full --attacker pilot_fov_exit --seed 7 `
   --video --pip --pip-scale 0.34 --out-dir results/v3/demo_my_pick
```
Writes `run.mp4` (third-person), `run_pov.mp4` (POV only), and
`run_pip.mp4` (composited).

Full 5-method grid with PiP for a different seed set:
```powershell
py scripts/render_cinematic_grid.py `
   --seeds 0 2 6 10 --attacker pilot_fov_exit `
   --pip --pip-scale 0.34 `
   --out-dir results/v3/cinematic_grids_pip_fov_exit_more
```

## Key plots accompanying the videos

| File | What it shows |
|------|---------------|
| `HERO_SUMMARY.png` | Three-panel headline: P_succ across scenarios, lock-loss -> failure chain, mean lock-loss per method |
| `PAPER_SUMMARY.png` | Detailed multi-panel summary across attackers |
| `lock_loss_analysis_ablation_pilot_fov_exit.png` | The causal-chain story: lock-loss time vs intercept |
| `clean_dist_ablation_pilot_fov_exit.png` | Outcome breakdown + miss-distance (intercepts only) + lock-loss distribution |
| `variance_ablation_pilot_fov_exit.png` | Wilson 95% CI on P_succ + reliability metrics |
| `sweeps/fov_pilot_hard/sweep_visibility_theta_F_deg.png` | P_succ vs camera FoV half-angle |
| `sweeps/latency_pilot_hard/sweep_estimator_latency.png` | P_succ vs sensor latency |
| `sweeps/r_c_pilot_hard/sweep_geom_r_c.png` | P_succ vs intercept envelope |
