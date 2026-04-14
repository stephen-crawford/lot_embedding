#!/usr/bin/env python3
"""
Sweep LOT + RC settings until VELOCITY map RMSE < POSITIONS (or exhaust grid).

Usage:
  python scripts/sweep_velocity_wins.py --mode synthetic
  python scripts/sweep_velocity_wins.py --mode goes --goes-dir goes_data

Requires: venv, pip install -r requirements.txt (and realworld for GOES download).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pipeline():
    path = REPO_ROOT / "scripts" / "realworld_experiment_pipeline.py"
    spec = importlib.util.spec_from_file_location("realworld_experiment_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")[:80]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "goes"], default="synthetic")
    ap.add_argument("--goes-dir", type=Path, default=REPO_ROOT / "goes_data")
    ap.add_argument(
        "--fetch-goes",
        action="store_true",
        help="GOES mode: download particles.npy if missing (goes2go + network)",
    )
    ap.add_argument("--goes-hours", type=int, default=24)
    ap.add_argument("--goes-date", default="2024-07-15")
    ap.add_argument("--goes-hour-utc", type=int, default=18)
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--steps", type=int, default=260, help="Synthetic trajectory length")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--rc-backend", default="numpy")
    ap.add_argument(
        "--full-grid",
        action="store_true",
        help="Try every combo; default is stop after first velocity win",
    )
    args = ap.parse_args()
    stop_on_first_win = not args.full_grid

    mod = _load_pipeline()
    wins: list[dict] = []
    tried: list[dict] = []

    kinds = ["gaussian_iso", "uniform_square", "snapshot_begin"]
    assignments = ["fixed", "per_frame"]
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    vel_hps = [
        (None, None, None),
        (0.85, 0.85, 0.02),
        (0.75, 0.9, 0.05),
        (0.9, 0.7, 0.005),
    ]

    goes_dir = Path(args.goes_dir) if args.mode == "goes" else None
    fetch_kw = {}
    if args.mode == "goes":
        fetch_kw = dict(
            fetch_goes_if_missing=args.fetch_goes,
            goes_hours=args.goes_hours,
            goes_date=args.goes_date,
            goes_hour_utc=args.goes_hour_utc,
        )
        if not args.fetch_goes and not (goes_dir / "particles.npy").is_file():
            print(
                f"[ERROR] No particles at {goes_dir / 'particles.npy'}. "
                "Pass --fetch-goes or run scripts/download_realworld_data.py --source goes"
            )
            return 1

    done = False
    for kind in kinds:
        if done:
            break
        for assignment in assignments:
            if done:
                break
            for scale in scales:
                if done:
                    break
                for vhp_i, (sr_v, lk_v, rd_v) in enumerate(vel_hps):
                    sys_name = _slug(
                        f"sw_{args.mode}_{args.N}_{kind}_{assignment}_sc{scale}_vhp{vhp_i}"
                    )
                    tag = f"{sys_name}"
                    try:
                        s = mod.run_pipeline(
                            system=sys_name,
                            n_particles=args.N,
                            n_steps=args.steps if args.mode == "synthetic" else 10**9,
                            n_cycles=args.cycles,
                            goes_dir=goes_dir,
                            quick=False,
                            ot_method="emd",
                            ot_device="cpu",
                            rc_backend=args.rc_backend,
                            lot_kind=kind,
                            lot_frac=100,
                            assignment=assignment,
                            reservoir_scale=scale,
                            run_tag_velocity=f"v_{tag}",
                            run_tag_positions=f"p_{tag}",
                            spectral_radius_vel=sr_v,
                            leak_rate_vel=lk_v,
                            ridge_vel=rd_v,
                            **fetch_kw,
                        )
                    except Exception as e:
                        print(f"[SKIP] {sys_name}: {e}")
                        continue
                    tried.append(s)
                    vr, pr = s["forecast_map_rmse_velocity"], s["forecast_map_rmse_positions"]
                    ok = vr < pr
                    print(
                        f"{'WIN ' if ok else 'loss'} {sys_name} | "
                        f"vel={vr:.5f} pos={pr:.5f} | kind={kind} {assignment} scale={scale} vhp={vhp_i}"
                    )
                    if ok:
                        wins.append(s)
                        if stop_on_first_win:
                            done = True
                            break

    out = REPO_ROOT / "experiment_velocity_wins.json"
    with open(out, "w") as f:
        json.dump({"wins": wins, "n_tried": len(tried), "n_wins": len(wins)}, f, indent=2)
    print(f"\nSaved {out} | wins={len(wins)} / tried={len(tried)}")

    if wins:
        best = min(wins, key=lambda x: x["forecast_map_rmse_velocity"])
        print("\nBest VELOCITY RMSE among wins:")
        print(json.dumps(best, indent=2))
        return 0

    print("\nNo winning config in this grid; extend grid or increase --steps / GOES hours.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
