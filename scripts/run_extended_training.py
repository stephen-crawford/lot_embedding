#!/usr/bin/env python3
"""
Train RC on extended multi-season GOES data (many diurnal cycles)
and compare velocity vs positions forecasting.

Tests:
  1. Short training: 1 cycle warm-up, 1 cycle forecast (existing approach)
  2. Extended training: 10+ cycles warm-up, 2 cycle forecast
  3. Multi-year: concatenate 2023+2024 for even more training data
"""
import sys, json, os, time
import numpy as np
from pathlib import Path
from dataclasses import dataclass

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from data_utils.simulation.generate_lot_embeddings import generate_lot_embeddings
from run_forecast.velocity_forecast import SystemConfig as VelCfg, run_single as vel_run
from run_forecast.positions_forecast import SystemConfig as PosCfg, run_single as pos_run


def _normalize01(traj, margin=0.02):
    lo = traj.min(axis=(0, 1), keepdims=True)
    hi = traj.max(axis=(0, 1), keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    u = (traj - lo) / span
    return np.clip(u * (1 - 2 * margin) + margin, 0.0, 1.0)


def l2_sigma_rmse(pred, true):
    diff = pred - true
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))


def run_comparison(traj, tag, cycle_length, sr=0.9, lk=0.9, ridge=0.01):
    """Run velocity vs positions on given trajectory with explicit cycle_length."""
    T, N, d = traj.shape
    results_root = str(REPO / "results")
    lot_root = str(REPO / "lot_maps")
    forecast_root = str(REPO / "forecast_output")

    traj_dir = f"{results_root}/{tag}_N_{N}"
    os.makedirs(traj_dir, exist_ok=True)
    np.save(f"{traj_dir}/trajectory.npy", traj)

    generate_lot_embeddings(
        tag, N_list=[N], kinds=["gaussian_iso"], fractions=[100],
        assignment="per_frame", results_root=results_root, lot_root=lot_root,
        verbose=False, ot_method="emd", ot_device="cpu",
    )

    lot_dir = Path(f"{lot_root}/{tag}_N_{N}/gaussian_iso/frac100")

    common = dict(
        name=tag, N_list=[N], kinds=["gaussian_iso"], fractions=[100],
        reservoir_scales=[1.0], results_root=results_root,
        forecast_root=forecast_root, lot_root=lot_root,
        cycle_length=cycle_length, boundary_mode="clip",
        rc_backend="numpy", rc_device="cpu",
        spectral_radius=sr, leak_rate=lk, ridge_param=ridge,
    )

    vel_cfg = VelCfg(**common, run_tag=f"ext_vel_{tag}")
    pos_cfg = PosCfg(**common, run_tag=f"ext_pos_{tag}")

    vel_out = vel_cfg.output_dir(N, "gaussian_iso", 100, 1.0)
    pos_out = pos_cfg.output_dir(N, "gaussian_iso", 100, 1.0)

    rv = vel_run(lot_dir, vel_out, N, "gaussian_iso", 100, 1.0, vel_cfg, skip_existing=False)
    rp = pos_run(lot_dir, pos_out, N, "gaussian_iso", 100, 1.0, pos_cfg, skip_existing=False)

    if rv is None or rp is None:
        return None

    vel_pred = np.load(vel_out / "predicted_maps.npy")
    vel_true = np.load(vel_out / "true_future_maps.npy")
    pos_pred = np.load(pos_out / "predicted_maps.npy")
    pos_true = np.load(pos_out / "true_future_maps.npy")

    if vel_pred.shape[0] == vel_true.shape[0] + 1:
        vel_pred = vel_pred[1:]

    Tf = min(vel_pred.shape[0], pos_pred.shape[0])

    v_rmse = l2_sigma_rmse(vel_pred[:Tf], vel_true[:Tf])
    p_rmse = l2_sigma_rmse(pos_pred[:Tf], pos_true[:Tf])
    impr = (p_rmse - v_rmse) / p_rmse * 100 if p_rmse > 0 else 0

    warm = cycle_length - 1
    forecast = T - 2 - warm

    return {
        "tag": tag, "T": T, "N": N, "cycle_length": cycle_length,
        "warm_steps": warm, "forecast_steps": forecast,
        "sr": sr, "lk": lk, "ridge": ridge,
        "v_rmse": v_rmse, "p_rmse": p_rmse,
        "improvement": impr,
        "velocity_wins": v_rmse < p_rmse,
    }


def main():
    results = []

    # ── Load extended data ──
    p2024 = np.load(REPO / "goes_data_july_2024" / "particles.npy")
    print(f"2024 data: {p2024.shape}")  # (360, 200, 2) at 1h cadence

    try:
        p2023 = np.load(REPO / "goes_data_july_2023" / "particles.npy")
        print(f"2023 data: {p2023.shape}")
    except FileNotFoundError:
        p2023 = None
        print("2023 data not available yet")

    # At 1h cadence, cycle_length = 24 (24h = 24 steps)
    CYCLE = 24  # diurnal cycle at 1h cadence

    # ── Test 1: Short training (1 cycle warm-up, ~1 cycle forecast) ──
    # Use first 48h (48 steps at 1h)
    print("\n" + "="*70)
    print("TEST 1: Short training — 1 cycle warm-up (24h), 1 cycle forecast")
    print("="*70)
    traj_short = _normalize01(p2024[:48].astype(np.float32))
    for sr, lk, ridge in [(0.9, 0.9, 0.01), (0.95, 0.9, 0.01), (0.8, 0.9, 0.01)]:
        tag = f"short_1cyc_sr{sr}_lk{lk}"
        r = run_comparison(traj_short, tag, cycle_length=CYCLE, sr=sr, lk=lk, ridge=ridge)
        if r:
            w = "VEL" if r["velocity_wins"] else "POS"
            print(f"  {tag}: V={r['v_rmse']:.4f} P={r['p_rmse']:.4f} impr={r['improvement']:+.1f}% [{w}]"
                  f" (warm={r['warm_steps']}, fc={r['forecast_steps']})")
            results.append(r)

    # ── Test 2: Extended training — 5 cycles warm-up ──
    print("\n" + "="*70)
    print("TEST 2: Extended training — 5 cycles warm-up (5 days)")
    print("="*70)
    traj_5day = _normalize01(p2024[:168].astype(np.float32))  # 7 days
    cl_5day = CYCLE * 5  # 5-day cycle → warm_steps = 119
    for sr, lk, ridge in [(0.9, 0.9, 0.01), (0.95, 0.9, 0.01), (0.8, 0.9, 0.01)]:
        tag = f"ext_5cyc_sr{sr}_lk{lk}"
        r = run_comparison(traj_5day, tag, cycle_length=cl_5day, sr=sr, lk=lk, ridge=ridge)
        if r:
            w = "VEL" if r["velocity_wins"] else "POS"
            print(f"  {tag}: V={r['v_rmse']:.4f} P={r['p_rmse']:.4f} impr={r['improvement']:+.1f}% [{w}]"
                  f" (warm={r['warm_steps']}, fc={r['forecast_steps']})")
            results.append(r)

    # ── Test 3: Extended training — 10 cycles warm-up ──
    print("\n" + "="*70)
    print("TEST 3: Extended training — 10 cycles warm-up (10 days)")
    print("="*70)
    traj_12day = _normalize01(p2024[:288].astype(np.float32))  # 12 days
    cl_10day = CYCLE * 10  # 10-day cycle → warm_steps = 239
    for sr, lk, ridge in [(0.9, 0.9, 0.01), (0.95, 0.9, 0.01), (0.8, 0.9, 0.01)]:
        tag = f"ext_10cyc_sr{sr}_lk{lk}"
        r = run_comparison(traj_12day, tag, cycle_length=cl_10day, sr=sr, lk=lk, ridge=ridge)
        if r:
            w = "VEL" if r["velocity_wins"] else "POS"
            print(f"  {tag}: V={r['v_rmse']:.4f} P={r['p_rmse']:.4f} impr={r['improvement']:+.1f}% [{w}]"
                  f" (warm={r['warm_steps']}, fc={r['forecast_steps']})")
            results.append(r)

    # ── Test 4: Full 14 days, 12 cycles warm-up ──
    print("\n" + "="*70)
    print("TEST 4: Full 14-day training — 12 cycles warm-up")
    print("="*70)
    traj_full = _normalize01(p2024.astype(np.float32))
    cl_12day = CYCLE * 12  # warm_steps = 287
    for sr, lk, ridge in [(0.9, 0.9, 0.01), (0.95, 0.9, 0.01)]:
        tag = f"ext_12cyc_sr{sr}_lk{lk}"
        r = run_comparison(traj_full, tag, cycle_length=cl_12day, sr=sr, lk=lk, ridge=ridge)
        if r:
            w = "VEL" if r["velocity_wins"] else "POS"
            print(f"  {tag}: V={r['v_rmse']:.4f} P={r['p_rmse']:.4f} impr={r['improvement']:+.1f}% [{w}]"
                  f" (warm={r['warm_steps']}, fc={r['forecast_steps']})")
            results.append(r)

    # ── Test 5: Multi-year (2023+2024 concatenated) ──
    if p2023 is not None:
        print("\n" + "="*70)
        print("TEST 5: Multi-year — 2023+2024 concatenated (28+ days)")
        print("="*70)
        # Concatenate: 2023 data first, then 2024
        traj_multi = _normalize01(np.concatenate([p2023, p2024], axis=0).astype(np.float32))
        T_multi = traj_multi.shape[0]
        # Train on most of it, forecast last ~2 days
        cl_multi = T_multi - 2 - 48  # leave 48 steps (2 days) for forecast
        for sr, lk, ridge in [(0.9, 0.9, 0.01), (0.95, 0.9, 0.01)]:
            tag = f"multi_yr_sr{sr}_lk{lk}"
            r = run_comparison(traj_multi, tag, cycle_length=cl_multi, sr=sr, lk=lk, ridge=ridge)
            if r:
                w = "VEL" if r["velocity_wins"] else "POS"
                print(f"  {tag}: V={r['v_rmse']:.4f} P={r['p_rmse']:.4f} impr={r['improvement']:+.1f}% [{w}]"
                      f" (warm={r['warm_steps']}, fc={r['forecast_steps']})")
                results.append(r)

    # ── Summary ──
    print(f"\n\n{'='*100}")
    print(f"{'EXTENDED TRAINING RESULTS':^100}")
    print(f"{'='*100}")
    print(f"{'Tag':<30} {'T':>4} {'Warm':>5} {'FC':>4} {'sr':>5} {'V-RMSE':>8} {'P-RMSE':>8} {'Impr%':>7} {'Win':>4}")
    print(f"{'-'*100}")
    for r in results:
        w = "VEL" if r["velocity_wins"] else "POS"
        print(f"{r['tag']:<30} {r['T']:>4} {r['warm_steps']:>5} {r['forecast_steps']:>4} "
              f"{r['sr']:>5.2f} {r['v_rmse']:>8.4f} {r['p_rmse']:>8.4f} {r['improvement']:>+7.1f} {w:>4}")

    vel_wins = sum(1 for r in results if r["velocity_wins"])
    print(f"\nVelocity wins: {vel_wins}/{len(results)}")

    with open(REPO / "extended_training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved: extended_training_results.json")


if __name__ == "__main__":
    main()
