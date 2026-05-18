"""Render cinematic 5-method comparison grids for the pilot attacker
scenarios (pilot_hard and pilot_easy), across multiple showcase seeds.

For each (attacker, seed) pair, renders 5 individual cinematic OpenGL
videos at 640x360 and stitches them into a 2x3 grid with a per-method
numbers panel.  Output: results/v3/cinematic_grids/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import main as M  # noqa: E402
from scripts.render_cinematic_grid import (  # noqa: E402
    render_per_method_videos,
    composite_grid,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-hard", type=int, nargs="+",
                        default=[1, 4, 7, 11])
    parser.add_argument("--seeds-easy", type=int, nargs="+",
                        default=[2, 6, 9])
    parser.add_argument("--t-end", type=float, default=5.0)
    parser.add_argument("--cell-width", type=int, default=640)
    parser.add_argument("--cell-height", type=int, default=360)
    parser.add_argument("--out-root", type=str,
                        default=str(ROOT / "results/v3/cinematic_grids"))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    plans = [
        ("pilot_hard", args.seeds_hard),
        ("pilot_easy", args.seeds_easy),
    ]
    for attacker, seeds in plans:
        cell_dir = out_root / "cells" / attacker
        cell_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            print(f"\n=== {attacker} seed {seed} ===")
            t0 = time.time()
            per_method = render_per_method_videos(
                seed, cell_dir,
                attacker=attacker, t_end=args.t_end,
                width=args.cell_width, height=args.cell_height,
            )
            out = out_root / f"grid_{attacker}_seed{seed}.mp4"
            composite_grid(per_method, out,
                           cell_w=args.cell_width,
                           cell_h=args.cell_height,
                           seed=seed,
                           attacker_label=M.ATTACKER_LABEL.get(attacker, attacker))
            print(f"  total: {time.time() - t0:.1f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
