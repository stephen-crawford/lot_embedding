#!/usr/bin/env python3
"""
Train RC on 100+ diurnal cycles of multi-year July GOES data.
Compare velocity vs positions at various training lengths.
Generate 3-row satellite comparison plot for the best result.
"""
import sys, json, os, time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from data_utils.simulation.generate_lot_embeddings import generate_lot_embeddings
from data_utils.goes_cloud_data import cloud_field_to_particles
from run_forecast.velocity_forecast import SystemConfig as VelCfg, run_single as vel_run
from run_forecast.positions_forecast import SystemConfig as PosCfg, run_single as pos_run

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


def _normalize01(traj, margin=0.02):
    lo = traj.min(axis=(0, 1), keepdims=True)
    hi = traj.max(axis=(0, 1), keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    u = (traj - lo) / span
    return np.clip(u * (1 - 2 * margin) + margin, 0.0, 1.0)


def run_comparison(traj, tag, cycle_length, sr=0.9, lk=0.9, ridge=0.01):
    T, N, d = traj.shape
    rr = str(REPO / "results")
    lr = str(REPO / "lot_maps")
    fr = str(REPO / "forecast_output")

    td = f"{rr}/{tag}_N_{N}"
    os.makedirs(td, exist_ok=True)
    np.save(f"{td}/trajectory.npy", traj)

    generate_lot_embeddings(
        tag, N_list=[N], kinds=["gaussian_iso"], fractions=[100],
        assignment="per_frame", results_root=rr, lot_root=lr,
        verbose=False, ot_method="emd", ot_device="cpu",
    )

    lot_dir = Path(f"{lr}/{tag}_N_{N}/gaussian_iso/frac100")
    common = dict(
        name=tag, N_list=[N], kinds=["gaussian_iso"], fractions=[100],
        reservoir_scales=[1.0], results_root=rr, forecast_root=fr, lot_root=lr,
        cycle_length=cycle_length, boundary_mode="clip",
        rc_backend="numpy", rc_device="cpu",
        spectral_radius=sr, leak_rate=lk, ridge_param=ridge,
    )

    vc = VelCfg(**common, run_tag=f"100c_vel_{tag}")
    pc = PosCfg(**common, run_tag=f"100c_pos_{tag}")
    vo = vc.output_dir(N, "gaussian_iso", 100, 1.0)
    po = pc.output_dir(N, "gaussian_iso", 100, 1.0)

    rv = vel_run(lot_dir, vo, N, "gaussian_iso", 100, 1.0, vc, skip_existing=False)
    rp = pos_run(lot_dir, po, N, "gaussian_iso", 100, 1.0, pc, skip_existing=False)
    if rv is None or rp is None:
        return None

    vp = np.load(vo / "predicted_maps.npy")
    vt = np.load(vo / "true_future_maps.npy")
    pp = np.load(po / "predicted_maps.npy")
    pt = np.load(po / "true_future_maps.npy")
    if vp.shape[0] == vt.shape[0] + 1:
        vp = vp[1:]
    Tf = min(vp.shape[0], pp.shape[0])

    def rmse(a, b):
        return float(np.sqrt(np.mean(np.sum((a[:Tf] - b[:Tf])**2, axis=-1))))

    vr = rmse(vp, vt)
    pr = rmse(pp, pt)
    im = (pr - vr) / pr * 100 if pr > 0 else 0
    warm = cycle_length - 1

    return {
        "tag": tag, "T": T, "warm": warm, "fc": T - 2 - warm,
        "sr": sr, "v_rmse": vr, "p_rmse": pr, "improvement": im,
        "velocity_wins": vr < pr, "vel_dir": str(vo), "pos_dir": str(po),
    }


def main():
    # Load multi-year combined data
    combined_path = REPO / "goes_data_multiyear_july" / "particles.npy"
    if not combined_path.exists():
        print("Multi-year data not ready yet. Run the download script first.")
        return

    combined = np.load(combined_path)
    T_total = combined.shape[0]
    # At 2h cadence, 12 steps per cycle
    STEPS_PER_CYCLE = 12
    total_cycles = T_total / STEPS_PER_CYCLE
    print(f"Combined data: {combined.shape} ({total_cycles:.0f} diurnal cycles at 2h cadence)")

    results = []

    # Test various training lengths, all forecasting ~2 days (24 steps at 2h)
    FC_STEPS = 24  # 2 days at 2h cadence

    for n_train_cycles in [5, 10, 20, 50, 80, 100]:
        train_steps = n_train_cycles * STEPS_PER_CYCLE
        total_needed = train_steps + FC_STEPS + 2  # +2 for velocity pairs

        if total_needed > T_total:
            print(f"\n  {n_train_cycles} cycles: need {total_needed} steps, only have {T_total}. Skipping.")
            continue

        traj = _normalize01(combined[:total_needed].astype(np.float32))
        cl = train_steps + 1  # cycle_length → warm_steps = cl - 1 = train_steps

        print(f"\n{'='*70}")
        print(f"Training: {n_train_cycles} cycles ({train_steps} steps), "
              f"Forecast: {FC_STEPS} steps (2 days)")
        print(f"{'='*70}")

        for sr in [0.9, 0.8]:
            tag = f"c{n_train_cycles}_sr{sr}"
            t0 = time.time()
            r = run_comparison(traj, tag, cycle_length=cl, sr=sr)
            elapsed = time.time() - t0
            if r:
                w = "VEL" if r["velocity_wins"] else "POS"
                print(f"  sr={sr}: V={r['v_rmse']:.4f} P={r['p_rmse']:.4f} "
                      f"impr={r['improvement']:+.1f}% [{w}] ({elapsed:.1f}s)")
                results.append({**r, "n_cycles": n_train_cycles})

    # Summary
    print(f"\n\n{'='*100}")
    print(f"{'100-CYCLE TRAINING RESULTS':^100}")
    print(f"{'='*100}")
    print(f"{'Cycles':>6} {'sr':>5} {'T':>5} {'Warm':>5} {'FC':>4} {'V-RMSE':>8} {'P-RMSE':>8} {'Impr%':>7} {'Win':>4}")
    print(f"{'-'*100}")
    for r in results:
        w = "VEL" if r["velocity_wins"] else "POS"
        print(f"{r['n_cycles']:>6} {r['sr']:>5.2f} {r['T']:>5} {r['warm']:>5} {r['fc']:>4} "
              f"{r['v_rmse']:>8.4f} {r['p_rmse']:>8.4f} {r['improvement']:>+7.1f} {w:>4}")

    # Find best velocity win
    vel_wins = [r for r in results if r["velocity_wins"]]
    if vel_wins:
        best = max(vel_wins, key=lambda r: r["improvement"])
        print(f"\nBest velocity win: {best['n_cycles']} cycles, "
              f"sr={best['sr']}, improvement={best['improvement']:+.1f}%")

    with open(REPO / "100cycle_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved: 100cycle_results.json")

    # ── Generate 3-row plot for the best result ──
    if vel_wins:
        best = max(vel_wins, key=lambda r: r["improvement"])
        print(f"\nGenerating 3-row plot for best result: {best['tag']}")

        vel_dir = Path(best["vel_dir"])
        pos_dir = Path(best["pos_dir"])

        vel_pred = np.load(vel_dir / "predicted_measures_X.npy")
        pos_pred = np.load(pos_dir / "predicted_measures_X.npy")
        vel_true = np.load(vel_dir / "true_measures_X.npy")
        F = min(vel_pred.shape[0], pos_pred.shape[0])

        # Use the raw fields from goes_data_july_2024 for satellite background
        # The forecast period is at the START of the combined data (2024 first)
        raw_fields = np.load(REPO / "goes_data_july_2024" / "raw_fields.npy")

        warm = best["warm"]
        # At 2h cadence, map forecast steps to raw field indices
        # The combined data starts at the same point as 2024 data (subsampled 2x)
        # Forecast starts at step warm+1 in combined data = step (warm+1)*2 in 1h data

        fc_indices = np.linspace(0, F-1, 7, dtype=int)
        col_fc = [int(i) for i in fc_indices]

        H_img, W_img = raw_fields[0].shape

        def to_pixel(pts):
            return pts[:, 0] * (W_img - 1), pts[:, 1] * (H_img - 1)

        GT_C, VEL_C, POS_C = '#FF6D00', '#00E5FF', '#FF1744'
        n_cols = 1 + len(col_fc)  # 1 train + 7 forecast

        fig = plt.figure(figsize=(3.0 * n_cols, 10), facecolor='white')
        gs = gridspec.GridSpec(3, n_cols, hspace=0.10, wspace=0.03,
                               left=0.055, right=0.99, top=0.88, bottom=0.05)

        all_cols = [-1] + col_fc  # -1 = training step

        for ci, fi in enumerate(all_cols):
            is_train = (fi < 0)

            # Get satellite field: map to 1h index in raw_fields
            if is_train:
                raw_idx = min(warm * 2, len(raw_fields) - 1)
            else:
                raw_idx = min((warm + 1 + fi) * 2, len(raw_fields) - 1)

            sat_field = raw_fields[raw_idx]

            # Time label
            hours_from_fc = fi * 2 if fi >= 0 else 0  # 2h cadence

            for ri in range(3):
                ax = fig.add_subplot(gs[ri, ci])
                ax.imshow(sat_field, cmap='gray_r', origin='upper',
                          interpolation='bilinear', alpha=0.9, vmin=195, vmax=310)
                ax.set_xlim(0, W_img)
                ax.set_ylim(H_img, 0)
                ax.set_xticks([])
                ax.set_yticks([])

                if ri == 0:
                    # Ground truth from satellite
                    pts = cloud_field_to_particles(sat_field.astype(np.float64),
                                                   n_particles=500,
                                                   method='consistent_quantile')[:, :2]
                    cx, cy = to_pixel(pts)
                    ax.scatter(cx, cy, s=6, c=GT_C, alpha=0.85,
                              edgecolors='white', linewidths=0.15, zorder=3)
                    if is_train:
                        tag_str = f'Train end\n({best["n_cycles"]} cycles)'
                    else:
                        tag_str = f'+{hours_from_fc}h'
                    ax.set_title(tag_str, fontsize=7, fontweight='bold', pad=3)

                elif ri == 1:
                    if is_train:
                        pts = cloud_field_to_particles(sat_field.astype(np.float64),
                                                       n_particles=500,
                                                       method='consistent_quantile')[:, :2]
                        cx, cy = to_pixel(pts)
                        ax.scatter(cx, cy, s=6, c=VEL_C, alpha=0.3,
                                  edgecolors='white', linewidths=0.1, zorder=3)
                        ax.text(0.5, 0.5, f'training\n({best["n_cycles"]} cycles)',
                                transform=ax.transAxes, ha='center', va='center',
                                fontsize=7, color='white', fontweight='bold',
                                bbox=dict(fc='black', alpha=0.5, ec='none',
                                         boxstyle='round,pad=0.3'))
                    elif 0 <= fi < F:
                        cx, cy = to_pixel(vel_pred[fi])
                        ax.scatter(cx, cy, s=6, c=VEL_C, alpha=0.85,
                                  edgecolors='white', linewidths=0.15, zorder=3)

                elif ri == 2:
                    if is_train:
                        pts = cloud_field_to_particles(sat_field.astype(np.float64),
                                                       n_particles=500,
                                                       method='consistent_quantile')[:, :2]
                        cx, cy = to_pixel(pts)
                        ax.scatter(cx, cy, s=6, c=POS_C, alpha=0.3,
                                  edgecolors='white', linewidths=0.1, zorder=3)
                        ax.text(0.5, 0.5, f'training\n({best["n_cycles"]} cycles)',
                                transform=ax.transAxes, ha='center', va='center',
                                fontsize=7, color='white', fontweight='bold',
                                bbox=dict(fc='black', alpha=0.5, ec='none',
                                         boxstyle='round,pad=0.3'))
                    elif 0 <= fi < F:
                        cx, cy = to_pixel(pos_pred[fi])
                        ax.scatter(cx, cy, s=6, c=POS_C, alpha=0.85,
                                  edgecolors='white', linewidths=0.15, zorder=3)

        # Labels
        fig.text(0.015, 0.73, 'Ground\nTruth', fontsize=10, fontweight='bold',
                 va='center', ha='center', rotation=90, color=GT_C)
        fig.text(0.015, 0.49, 'Velocity\nForecast', fontsize=10, fontweight='bold',
                 va='center', ha='center', rotation=90, color=VEL_C)
        fig.text(0.015, 0.23, 'Positions\nForecast', fontsize=10, fontweight='bold',
                 va='center', ha='center', rotation=90, color=POS_C)

        legend_els = [
            Line2D([], [], marker='o', ls='none', color=GT_C, markersize=5,
                   label='True cloud particles'),
            Line2D([], [], marker='o', ls='none', color=VEL_C, markersize=5,
                   label='Velocity forecast (LOT)'),
            Line2D([], [], marker='o', ls='none', color=POS_C, markersize=5,
                   label='Positions forecast (LOT)'),
        ]
        fig.legend(handles=legend_els, loc='lower center', ncol=3, fontsize=8,
                   frameon=True, fancybox=True, edgecolor='#ccc',
                   bbox_to_anchor=(0.52, 0.005))

        fig.suptitle(
            f'GOES-16 Cloud Forecasting: {best["n_cycles"]}-Cycle Training\n'
            f'N=200, R=200 | Per-frame gaussian_iso | 2h cadence | Multi-year July',
            fontsize=12, fontweight='bold', y=0.96)
        fig.text(0.5, 0.91,
                 f'sr={best["sr"]}, lk=0.9, ridge=0.01 | '
                 f'Velocity wins: {best["improvement"]:+.1f}% L²',
                 ha='center', fontsize=9, color='#444')

        out = REPO / 'plots' / 'goes_100cycle_forecast.png'
        fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
        fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
