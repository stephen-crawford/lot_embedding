#!/usr/bin/env python3
"""
Run velocity vs positions experiments on 180-day real SST data,
then generate plots for winning configurations.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def l2_sigma_per_step(pred, true):
    diff = pred - true
    sq_norms = np.sum(diff ** 2, axis=-1)
    return np.sqrt(np.mean(sq_norms, axis=-1))


def l2_sigma_rmse(pred, true):
    per_step = l2_sigma_per_step(pred, true)
    return float(np.sqrt(np.mean(per_step ** 2)))


def _load_pipeline():
    path = REPO_ROOT / "scripts" / "realworld_experiment_pipeline.py"
    spec = importlib.util.spec_from_file_location("realworld_experiment_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    sst_dir = REPO_ROOT / "sst_data_180d"
    traj = np.load(sst_dir / "particles.npy")
    T, N, d = traj.shape
    print(f"SST 180-day data: T={T}, N={N}, d={d}")

    mod = _load_pipeline()

    configs = [
        # (lot_kind, assignment, scale, sr_v, lk_v, rd_v, n_cycles)
        ("gaussian_iso", "fixed",     1.0,  None, None, None, 3),
        ("gaussian_iso", "per_frame", 1.0,  None, None, None, 3),
        ("gaussian_iso", "fixed",     0.5,  None, None, None, 3),
        ("gaussian_iso", "fixed",     0.75, None, None, None, 3),
        ("gaussian_iso", "fixed",     1.0,  0.85, 0.85, 0.02, 3),
        ("gaussian_iso", "fixed",     1.0,  0.9,  0.7,  0.005, 3),
        ("gaussian_iso", "fixed",     1.0,  0.75, 0.9,  0.05, 3),
        ("uniform_square", "fixed",   1.0,  None, None, None, 3),
        ("uniform_square", "per_frame", 1.0, None, None, None, 3),
        ("snapshot_begin", "fixed",   1.0,  None, None, None, 3),
        # Try different cycle estimates
        ("gaussian_iso", "fixed",     1.0,  None, None, None, 5),
        ("gaussian_iso", "fixed",     1.0,  None, None, None, 6),
        ("gaussian_iso", "per_frame", 1.0,  None, None, None, 5),
        ("gaussian_iso", "fixed",     1.0,  0.85, 0.85, 0.02, 5),
        ("gaussian_iso", "fixed",     1.0,  0.9,  0.7,  0.005, 5),
        ("gaussian_iso", "fixed",     0.5,  None, None, None, 5),
        ("gaussian_iso", "fixed",     0.5,  0.9,  0.7,  0.005, 5),
        ("gaussian_iso", "per_frame", 0.5,  None, None, None, 5),
    ]

    all_results = []
    wins = []

    for kind, assign, scale, sr, lk, rd, nc in configs:
        tag = f"sst180_{kind}_{assign}_sc{scale}_nc{nc}"
        if sr is not None:
            tag += f"_sr{sr}_lk{lk}_rd{rd}"

        print(f"\n{'='*60}")
        print(f"Running: {tag}")
        print(f"{'='*60}")

        try:
            summary = mod.run_pipeline(
                system=tag,
                n_particles=N,
                n_steps=T,
                n_cycles=nc,
                goes_dir=sst_dir,
                quick=False,
                ot_method="emd",
                ot_device="cpu",
                rc_backend="numpy",
                lot_kind=kind,
                lot_frac=100,
                assignment=assign,
                reservoir_scale=scale,
                run_tag_velocity=f"v_{tag}",
                run_tag_positions=f"p_{tag}",
                spectral_radius_vel=sr,
                leak_rate_vel=lk,
                ridge_vel=rd,
            )
        except Exception as e:
            print(f"  [SKIP] {tag}: {e}")
            continue

        vel_dir = Path(summary["velocity_forecast_dir"])
        pos_dir = Path(summary["positions_forecast_dir"])

        vel_pred = np.load(vel_dir / "predicted_maps.npy")
        vel_true = np.load(vel_dir / "true_future_maps.npy")
        pos_pred = np.load(pos_dir / "predicted_maps.npy")
        pos_true = np.load(pos_dir / "true_future_maps.npy")

        if vel_pred.shape[0] == vel_true.shape[0] + 1:
            vel_pred = vel_pred[1:]

        Tf = min(vel_pred.shape[0], pos_pred.shape[0])
        vel_rmse = l2_sigma_rmse(vel_pred[:Tf], vel_true[:Tf])
        pos_rmse = l2_sigma_rmse(pos_pred[:Tf], pos_true[:Tf])
        improvement = (pos_rmse - vel_rmse) / pos_rmse * 100 if pos_rmse > 0 else 0
        velocity_wins = vel_rmse < pos_rmse

        result = {
            "tag": tag,
            "vel_l2_rmse": vel_rmse,
            "pos_l2_rmse": pos_rmse,
            "velocity_wins": velocity_wins,
            "improvement_pct": improvement,
            "vel_dir": str(vel_dir),
            "pos_dir": str(pos_dir),
            "forecast_steps": Tf,
            "config": {
                "kind": kind, "assignment": assign, "scale": scale,
                "sr": sr, "lk": lk, "rd": rd, "n_cycles": nc,
            },
        }
        all_results.append(result)

        status = "WIN" if velocity_wins else "loss"
        print(f"  [{status}] vel={vel_rmse:.6f} pos={pos_rmse:.6f} improvement={improvement:+.1f}%")

        if velocity_wins:
            wins.append(result)

    # Save report
    report = {
        "data_source": "NOAA OISST (Gulf of Mexico)",
        "data_shape": list(traj.shape),
        "metric": "L2_sigma_RMSE (paper empirical-L2)",
        "n_configs_tried": len(all_results),
        "n_velocity_wins": len(wins),
        "all_results": all_results,
    }
    report_path = REPO_ROOT / "sst180_experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SST 180-DAY RESULTS: {len(wins)}/{len(all_results)} velocity wins")
    if wins:
        best = min(wins, key=lambda x: x["vel_l2_rmse"])
        print(f"Best: vel={best['vel_l2_rmse']:.6f} pos={best['pos_l2_rmse']:.6f} "
              f"improvement={best['improvement_pct']:+.1f}% ({best['tag']})")
    print(f"Report: {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
