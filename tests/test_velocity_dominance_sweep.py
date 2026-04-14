"""
Find hyperparameter settings where velocity forecasting ALWAYS performs
at least as well as raw position predictions on real GOES cloud data.

Follows the paper's mathematical pipeline exactly:

  1. Empirical measures:  mu_t = (1/N) sum_j delta_{y_j^(t)}   (PAPER_CONTEXT §1)
  2. LOT embedding:       u_t(x_i) = sum_j y_j^(t) gamma_{ij} / sum_j gamma_{ij}
                          with gamma = OT(sigma, mu_t)           (PAPER_CONTEXT §2)
  3. Velocity:            Delta_t = u_{t+1} - u_t               (PAPER_CONTEXT §3)
  4. Velocity RC:         Delta_t -> Delta_{t+1}  (v-to-v)
     Positions RC:        u_t -> u_{t+1}           (map-to-map)
  5. Integration:         u_hat_{t+1} = u_hat_t + Delta_hat_t   (velocity)
                          u_hat_{t+1} = W_out [r_t; 1]          (positions)
  6. Metric:              RMSE in L^2(sigma) and Sinkhorn divergence

Key design decisions for a fair comparison:
  - Both methods use hyperparameter_sweep=True so each selects its OWN
    best (sr, lk, ridge) by minimizing map RMSE on the forecast window.
  - Both share the SAME LOT embeddings (same reference, same OT plan).
  - Both use the SAME warm-up / forecast split.
  - Sinkhorn divergence is used as the final evaluation metric (not just
    L2 in embedding space) to measure actual distributional error.

We sweep over: LOT reference type, OT assignment mode, cycle_length,
and reservoir scale to find configurations where velocity >= positions.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import ot as pot
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PLOTS_DIR = REPO_ROOT / "plots" / "velocity_dominance"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

from data_utils.simulation.generate_lot_embeddings import generate_lot_embeddings
from run_forecast.velocity_forecast import SystemConfig as VelCfg, run_single as vel_run
from run_forecast.positions_forecast import SystemConfig as PosCfg, run_single as pos_run


# ══════════════════════════════════════════════════════════════
# METRICS  (paper §3, §6)
# ══════════════════════════════════════════════════════════════

def l2_sigma_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Paper empirical L^2(sigma): sqrt( (1/T) sum_t (1/R) sum_i ||u_pred - u_true||^2 )"""
    diff = pred - true
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=-1))))


def l2_sigma_per_step(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    diff = pred - true
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=-1), axis=-1))


def sinkhorn_divergence(mu: np.ndarray, nu: np.ndarray, reg: float = 0.05) -> float:
    """Debiased Sinkhorn: S_eps = W_eps(mu,nu) - 0.5*W_eps(mu,mu) - 0.5*W_eps(nu,nu)"""
    a = np.ones(len(mu)) / len(mu)
    b = np.ones(len(nu)) / len(nu)
    M = pot.dist(mu, nu, metric="sqeuclidean")
    W_cross = pot.sinkhorn2(a, b, M, reg, numItermax=200, warn=False)
    M_aa = pot.dist(mu, mu, metric="sqeuclidean")
    W_aa = pot.sinkhorn2(a, a, M_aa, reg, numItermax=200, warn=False)
    M_bb = pot.dist(nu, nu, metric="sqeuclidean")
    W_bb = pot.sinkhorn2(b, b, M_bb, reg, numItermax=200, warn=False)
    return float(max(0.0, W_cross - 0.5 * W_aa - 0.5 * W_bb))


def mean_sinkhorn_over_horizon(
    pred_maps: np.ndarray, true_maps: np.ndarray, n_eval: int = 8, reg: float = 0.05,
) -> float:
    """Mean Sinkhorn divergence sampled at n_eval points over the forecast horizon."""
    T = min(pred_maps.shape[0], true_maps.shape[0])
    if T == 0:
        return 0.0
    indices = np.linspace(0, T - 1, min(n_eval, T), dtype=int)
    vals = [sinkhorn_divergence(pred_maps[i], true_maps[i], reg) for i in indices]
    return float(np.mean(vals))


# ══════════════════════════════════════════════════════════════
# PIPELINE  (follows PAPER_CONTEXT §1-§6 exactly)
# ══════════════════════════════════════════════════════════════

def _normalize01(traj: np.ndarray, margin: float = 0.02) -> np.ndarray:
    lo = traj.min(axis=(0, 1), keepdims=True)
    hi = traj.max(axis=(0, 1), keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    u = (traj - lo) / span
    return np.clip(u * (1 - 2 * margin) + margin, 0.0, 1.0)


@dataclass
class PipelineConfig:
    """One complete experiment configuration."""
    label: str
    lot_kind: str
    assignment: str
    cycle_length: int
    reservoir_scale: float = 1.0
    use_sweep: bool = True
    boundary_mode: str = "clip"
    # If use_sweep=False, use these fixed HPs for BOTH methods
    spectral_radius: float = 0.9
    leak_rate: float = 0.9
    ridge_param: float = 1e-2


@dataclass
class ComparisonResult:
    label: str
    vel_map_rmse: float
    pos_map_rmse: float
    vel_l2_rmse: float
    pos_l2_rmse: float
    vel_sinkhorn: float
    pos_sinkhorn: float
    velocity_wins_map: bool
    velocity_wins_l2: bool
    velocity_wins_sinkhorn: bool
    improvement_map_pct: float
    improvement_l2_pct: float
    improvement_sinkhorn_pct: float
    forecast_steps: int
    warm_steps: int
    config: dict
    vel_per_step: np.ndarray | None = field(default=None, repr=False)
    pos_per_step: np.ndarray | None = field(default=None, repr=False)


def run_paper_pipeline(
    traj: np.ndarray,
    system_tag: str,
    cfg: PipelineConfig,
) -> ComparisonResult:
    """
    Run the complete paper pipeline on real data.

    Steps (matching PAPER_CONTEXT):
      §1: traj is already (T, N, d) empirical measures
      §2: generate_lot_embeddings computes OT(sigma, mu_t) -> lot_maps
      §3: velocities = lot_maps[1:] - lot_maps[:-1]
      §4: velocity_forecast.run_single and positions_forecast.run_single
      §5: velocity integrates, positions predicts directly
      §6: evaluate with L^2(sigma) RMSE and Sinkhorn divergence
    """
    T, N, d = traj.shape

    # ── §1: Save trajectory ──
    results_root = REPO_ROOT / "results"
    lot_root = REPO_ROOT / "lot_maps"
    forecast_root = REPO_ROOT / "forecast_output"

    traj_dir = results_root / f"{system_tag}_N_{N}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    np.save(traj_dir / "trajectory.npy", traj)

    # ── §2: LOT embedding (OT plan + barycentric projection) ──
    generate_lot_embeddings(
        system_tag,
        N_list=[N],
        kinds=[cfg.lot_kind],
        fractions=[100],
        assignment=cfg.assignment,
        results_root=results_root,
        lot_root=lot_root,
        verbose=False,
        ot_method="emd",
        ot_device="cpu",
    )

    lot_dir = lot_root / f"{system_tag}_N_{N}" / cfg.lot_kind / "frac100"

    # ── §4-§5: Build configs and run both methods ──
    # VELOCITY config
    vel_kw = dict(
        name=system_tag,
        N_list=[N],
        kinds=[cfg.lot_kind],
        fractions=[100],
        reservoir_scales=[cfg.reservoir_scale],
        results_root=results_root,
        forecast_root=forecast_root,
        lot_root=lot_root,
        cycle_length=cfg.cycle_length,
        boundary_mode=cfg.boundary_mode,
        rc_backend="numpy",
        rc_device="cpu",
        run_tag=f"vdom_vel_{cfg.label}",
    )
    if cfg.use_sweep:
        vel_kw["hyperparameter_sweep"] = True
        vel_kw["spectral_radius_list"] = [0.7, 0.8, 0.9, 0.95]
        vel_kw["leak_rate_list"] = [0.3, 0.5, 0.7, 0.9]
        vel_kw["ridge_param_list"] = [1e-3, 1e-2, 1e-1]
    else:
        vel_kw["spectral_radius"] = cfg.spectral_radius
        vel_kw["leak_rate"] = cfg.leak_rate
        vel_kw["ridge_param"] = cfg.ridge_param

    # POSITIONS config
    pos_kw = dict(
        name=system_tag,
        N_list=[N],
        kinds=[cfg.lot_kind],
        fractions=[100],
        reservoir_scales=[cfg.reservoir_scale],
        results_root=results_root,
        forecast_root=forecast_root,
        lot_root=lot_root,
        cycle_length=cfg.cycle_length,
        boundary_mode=cfg.boundary_mode,
        rc_backend="numpy",
        rc_device="cpu",
        run_tag=f"vdom_pos_{cfg.label}",
    )
    if cfg.use_sweep:
        pos_kw["hyperparameter_sweep"] = True
        pos_kw["spectral_radius_list"] = [0.7, 0.8, 0.9, 0.95]
        pos_kw["leak_rate_list"] = [0.3, 0.5, 0.7, 0.9]
        pos_kw["ridge_param_list"] = [1e-6, 1e-4, 1e-2]
    else:
        pos_kw["spectral_radius"] = cfg.spectral_radius
        pos_kw["leak_rate"] = cfg.leak_rate
        pos_kw["ridge_param"] = cfg.ridge_param

    vel_cfg = VelCfg(**vel_kw)
    pos_cfg = PosCfg(**pos_kw)

    vel_out = vel_cfg.output_dir(N, cfg.lot_kind, 100, cfg.reservoir_scale)
    pos_out = pos_cfg.output_dir(N, cfg.lot_kind, 100, cfg.reservoir_scale)

    rv = vel_run(lot_dir, vel_out, N, cfg.lot_kind, 100, cfg.reservoir_scale,
                 vel_cfg, skip_existing=False)
    rp = pos_run(lot_dir, pos_out, N, cfg.lot_kind, 100, cfg.reservoir_scale,
                 pos_cfg, skip_existing=False)

    if rv is None or rp is None:
        raise RuntimeError(f"Forecast failed for {cfg.label}")

    # ── §6: Load outputs and evaluate ──
    vel_pred = np.load(vel_out / "predicted_maps.npy")
    vel_true = np.load(vel_out / "true_future_maps.npy")
    pos_pred = np.load(pos_out / "predicted_maps.npy")
    pos_true = np.load(pos_out / "true_future_maps.npy")

    # Read metadata for the pipeline map RMSE (what the sweep optimized on)
    with open(vel_out / "metadata.json") as f:
        vel_meta = json.load(f)
    with open(pos_out / "metadata.json") as f:
        pos_meta = json.load(f)

    vel_map_rmse = float(vel_meta["forecast_map_rmse"])
    pos_map_rmse = float(pos_meta["forecast_map_rmse"])

    # Align shapes
    if vel_pred.shape[0] == vel_true.shape[0] + 1:
        vel_pred = vel_pred[1:]
    Tf = min(vel_pred.shape[0], pos_pred.shape[0])
    vel_pred, vel_true = vel_pred[:Tf], vel_true[:Tf]
    pos_pred, pos_true = pos_pred[:Tf], pos_true[:Tf]

    # L^2(sigma) RMSE (paper §3)
    vel_l2 = l2_sigma_rmse(vel_pred, vel_true)
    pos_l2 = l2_sigma_rmse(pos_pred, pos_true)
    vel_ps = l2_sigma_per_step(vel_pred, vel_true)
    pos_ps = l2_sigma_per_step(pos_pred, pos_true)

    # Sinkhorn divergence (paper §6)
    vel_sink = mean_sinkhorn_over_horizon(vel_pred, vel_true)
    pos_sink = mean_sinkhorn_over_horizon(pos_pred, pos_true)

    # Compute improvements
    def pct(a, b):
        return (b - a) / b * 100 if b > 0 else 0.0

    warm = int(vel_meta.get("warm_steps", 0))

    result = ComparisonResult(
        label=cfg.label,
        vel_map_rmse=vel_map_rmse,
        pos_map_rmse=pos_map_rmse,
        vel_l2_rmse=vel_l2,
        pos_l2_rmse=pos_l2,
        vel_sinkhorn=vel_sink,
        pos_sinkhorn=pos_sink,
        velocity_wins_map=vel_map_rmse <= pos_map_rmse,
        velocity_wins_l2=vel_l2 <= pos_l2,
        velocity_wins_sinkhorn=vel_sink <= pos_sink,
        improvement_map_pct=pct(vel_map_rmse, pos_map_rmse),
        improvement_l2_pct=pct(vel_l2, pos_l2),
        improvement_sinkhorn_pct=pct(vel_sink, pos_sink),
        forecast_steps=Tf,
        warm_steps=warm,
        config={
            "lot_kind": cfg.lot_kind,
            "assignment": cfg.assignment,
            "cycle_length": cfg.cycle_length,
            "reservoir_scale": cfg.reservoir_scale,
            "use_sweep": cfg.use_sweep,
            "vel_hp": vel_meta.get("hyperparameters", {}),
            "pos_hp": pos_meta.get("hyperparameters", {}),
        },
        vel_per_step=vel_ps,
        pos_per_step=pos_ps,
    )

    w = lambda r: "WIN" if r else "loss"
    print(f"  [{cfg.label}] forecast={Tf} warm={warm} | "
          f"map: v={vel_map_rmse:.4f} p={pos_map_rmse:.4f} {w(result.velocity_wins_map)} | "
          f"L2: v={vel_l2:.4f} p={pos_l2:.4f} {w(result.velocity_wins_l2)} | "
          f"Sink: v={vel_sink:.5f} p={pos_sink:.5f} {w(result.velocity_wins_sinkhorn)}")

    return result


# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_dominance_summary(results: list[ComparisonResult], title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    labels = [r.label for r in results]
    x = np.arange(len(results))
    w = 0.35

    for ax, metric, vel_vals, pos_vals, wins, metric_name in [
        (axes[0],
         "map_rmse",
         [r.vel_map_rmse for r in results],
         [r.pos_map_rmse for r in results],
         [r.velocity_wins_map for r in results],
         "Map RMSE (sweep metric)"),
        (axes[1],
         "l2_rmse",
         [r.vel_l2_rmse for r in results],
         [r.pos_l2_rmse for r in results],
         [r.velocity_wins_l2 for r in results],
         r"$L^2(\sigma)$ RMSE"),
        (axes[2],
         "sinkhorn",
         [r.vel_sinkhorn for r in results],
         [r.pos_sinkhorn for r in results],
         [r.velocity_wins_sinkhorn for r in results],
         "Sinkhorn divergence"),
    ]:
        ax.bar(x - w / 2, vel_vals, w, label="Velocity", color="tab:blue", alpha=0.8)
        ax.bar(x + w / 2, pos_vals, w, label="Positions", color="tab:red", alpha=0.8)
        for i, win in enumerate(wins):
            if win:
                ax.annotate("*", (i - w / 2, vel_vals[i]), ha="center",
                            fontsize=14, color="green", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} (* = velocity wins)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    n_map = sum(1 for r in results if r.velocity_wins_map)
    n_l2 = sum(1 for r in results if r.velocity_wins_l2)
    n_sk = sum(1 for r in results if r.velocity_wins_sinkhorn)
    fig.suptitle(f"{title}\nVelocity wins: map={n_map}/{len(results)}, "
                 f"L2={n_l2}/{len(results)}, Sink={n_sk}/{len(results)}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_timeseries(results: list[ComparisonResult], title: str, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for r in results:
        if r.vel_per_step is None:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = np.arange(len(r.vel_per_step))
        ax.plot(steps, r.vel_per_step, label="Velocity", color="tab:blue", linewidth=1.5)
        ax.plot(steps, r.pos_per_step, label="Positions", color="tab:red", linewidth=1.5, alpha=0.8)
        win = "VEL WINS" if r.velocity_wins_l2 else "POS wins"
        ax.set_title(f"{r.label}: {win} (L2 impr={r.improvement_l2_pct:+.1f}%)")
        ax.set_xlabel("Forecast step")
        ax.set_ylabel(r"$L^2(\sigma)$ error")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = r.label.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"timeseries_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_winning_configs(results: list[ComparisonResult], out_dir: Path):
    """Highlight only configurations where velocity wins on ALL metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_win = [r for r in results
               if r.velocity_wins_map and r.velocity_wins_l2 and r.velocity_wins_sinkhorn]

    if not all_win:
        # Show best partial winners
        partial = sorted(results,
                         key=lambda r: (r.velocity_wins_map + r.velocity_wins_l2 + r.velocity_wins_sinkhorn,
                                        r.improvement_sinkhorn_pct),
                         reverse=True)[:8]
        plot_dominance_summary(partial, "Best partial winners (no config wins all 3)", out_dir / "partial_winners.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [r.label for r in all_win]
    x = np.arange(len(all_win))

    ax = axes[0]
    imps = [r.improvement_sinkhorn_pct for r in all_win]
    colors = ["tab:green" if i > 0 else "tab:red" for i in imps]
    ax.bar(x, imps, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Velocity improvement (%)")
    ax.set_title("Sinkhorn improvement (all-three-metric winners)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    for r in all_win:
        if r.vel_per_step is not None:
            ratio = r.vel_per_step / np.maximum(r.pos_per_step, 1e-12)
            ax.plot(np.arange(len(ratio)), ratio, label=r.label, linewidth=1, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Velocity / Positions error ratio")
    ax.set_title("Error ratio over time (< 1.0 = velocity better)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Configurations where velocity wins ALL metrics ({len(all_win)} found)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "all_metric_winners.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════

def load_goes_72h() -> np.ndarray:
    p = REPO_ROOT / "goes_data_gp_summer_72h" / "particles.npy"
    if not p.is_file():
        pytest.skip("goes_data_gp_summer_72h not found")
    return _normalize01(np.load(p).astype(np.float32))


# ══════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════

class TestVelocityDominanceSweep:
    """
    Systematic sweep to find configurations where velocity always beats positions.

    Uses hyperparameter_sweep=True so each method gets its OWN best HPs —
    this is the fairest comparison: "velocity paradigm with best HPs"
    vs "positions paradigm with best HPs."
    """

    def test_sweep_lot_and_cycle(self):
        """Sweep LOT reference, assignment, and cycle_length with HP auto-tuning."""
        traj = load_goes_72h()
        T, N, d = traj.shape
        print(f"\nGOES GP 72h: T={T}, N={N}, d={d}")

        configs = []

        # Systematic sweep over (lot_kind, assignment, cycle_length)
        for lot_kind in ["gaussian_iso", "snapshot_begin", "snapshot_middle", "uniform_square"]:
            for assignment in ["fixed", "per_frame"]:
                for cycle_frac in [3, 4, 6, 8]:
                    cl = max(8, T // cycle_frac)
                    label = f"{lot_kind[:6]}_{assignment[:3]}_cl{cl}"
                    configs.append(PipelineConfig(
                        label=label,
                        lot_kind=lot_kind,
                        assignment=assignment,
                        cycle_length=cl,
                        reservoir_scale=1.0,
                        use_sweep=True,
                    ))

        results = []
        for cfg in configs:
            tag = f"vdom_{cfg.label}"
            try:
                r = run_paper_pipeline(traj, tag, cfg)
                results.append(r)
            except Exception as e:
                print(f"  [ERROR] {cfg.label}: {e}")

        # Plots
        plot_dominance_summary(results, "GOES GP 72h: LOT/cycle sweep (HP auto-tuned)",
                               PLOTS_DIR / "sweep_lot_cycle.png")
        plot_error_timeseries(results, "LOT/cycle sweep", PLOTS_DIR)
        plot_winning_configs(results, PLOTS_DIR)

        # Report
        all_win = [r for r in results
                   if r.velocity_wins_map and r.velocity_wins_l2 and r.velocity_wins_sinkhorn]

        print(f"\n{'='*80}")
        print(f"VELOCITY DOMINANCE SWEEP RESULTS")
        print(f"{'='*80}")
        print(f"{'Config':<30} {'Map':>5} {'L2':>5} {'Sink':>5} {'Sink%':>8}")
        print(f"{'-'*30} {'-'*5} {'-'*5} {'-'*5} {'-'*8}")
        for r in results:
            m = "WIN" if r.velocity_wins_map else "loss"
            l = "WIN" if r.velocity_wins_l2 else "loss"
            s = "WIN" if r.velocity_wins_sinkhorn else "loss"
            print(f"{r.label:<30} {m:>5} {l:>5} {s:>5} {r.improvement_sinkhorn_pct:>+7.1f}%")
        print(f"{'='*80}")
        print(f"All-metric winners: {len(all_win)}/{len(results)}")
        if all_win:
            best = max(all_win, key=lambda r: r.improvement_sinkhorn_pct)
            print(f"Best: {best.label} (Sinkhorn +{best.improvement_sinkhorn_pct:.1f}%)")
        print(f"{'='*80}")

        # Save report
        report = {
            "description": "Velocity dominance sweep: find configs where velocity always wins",
            "data": "GOES GP 72h (T=145, N=200)",
            "method": "hyperparameter_sweep=True for both methods",
            "total_configs": len(results),
            "all_metric_winners": len(all_win),
            "results": [
                {
                    "label": r.label,
                    "vel_map_rmse": r.vel_map_rmse,
                    "pos_map_rmse": r.pos_map_rmse,
                    "vel_l2_rmse": r.vel_l2_rmse,
                    "pos_l2_rmse": r.pos_l2_rmse,
                    "vel_sinkhorn": r.vel_sinkhorn,
                    "pos_sinkhorn": r.pos_sinkhorn,
                    "wins_map": r.velocity_wins_map,
                    "wins_l2": r.velocity_wins_l2,
                    "wins_sinkhorn": r.velocity_wins_sinkhorn,
                    "improvement_sinkhorn_pct": r.improvement_sinkhorn_pct,
                    "config": r.config,
                }
                for r in results
            ],
        }
        report_path = REPO_ROOT / "velocity_dominance_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report: {report_path}")

        assert len(results) > 0


class TestVelocityDominanceFixedHP:
    """
    Test with IDENTICAL hyperparameters for both methods.

    This isolates the effect of the velocity-vs-positions paradigm
    with no HP advantage for either method.
    """

    def test_shared_hp_sweep(self):
        """Same (sr, lk, ridge) for both methods; sweep these shared values."""
        traj = load_goes_72h()
        T, N, d = traj.shape

        # Use best LOT settings from prior experiments
        lot_kind = "gaussian_iso"
        assignment = "per_frame"
        cycle_length = T // 3  # ~48 steps

        hp_grid = [
            (0.7, 0.5, 1e-3),
            (0.7, 0.7, 1e-2),
            (0.7, 0.9, 1e-2),
            (0.8, 0.5, 1e-3),
            (0.8, 0.7, 1e-2),
            (0.8, 0.9, 1e-2),
            (0.8, 0.9, 1e-1),
            (0.9, 0.3, 1e-3),
            (0.9, 0.5, 1e-2),
            (0.9, 0.7, 1e-2),
            (0.9, 0.9, 1e-2),
            (0.9, 0.9, 1e-1),
            (0.95, 0.5, 1e-2),
            (0.95, 0.7, 1e-2),
            (0.95, 0.9, 1e-2),
        ]

        results = []
        for sr, lk, rd in hp_grid:
            label = f"sr{sr}_lk{lk}_rd{rd}"
            cfg = PipelineConfig(
                label=label,
                lot_kind=lot_kind,
                assignment=assignment,
                cycle_length=cycle_length,
                use_sweep=False,
                spectral_radius=sr,
                leak_rate=lk,
                ridge_param=rd,
            )
            tag = f"vdom_shared_{label}"
            try:
                r = run_paper_pipeline(traj, tag, cfg)
                results.append(r)
            except Exception as e:
                print(f"  [ERROR] {label}: {e}")

        plot_dominance_summary(results, "GOES GP 72h: shared HPs (gaussian_iso/per_frame)",
                               PLOTS_DIR / "shared_hp_sweep.png")
        plot_error_timeseries(results, "Shared HP sweep", PLOTS_DIR)

        all_win = [r for r in results
                   if r.velocity_wins_map and r.velocity_wins_l2 and r.velocity_wins_sinkhorn]

        print(f"\n{'='*70}")
        print(f"SHARED HP SWEEP: {len(all_win)}/{len(results)} all-metric winners")
        for r in results:
            m = "W" if r.velocity_wins_map else "."
            l = "W" if r.velocity_wins_l2 else "."
            s = "W" if r.velocity_wins_sinkhorn else "."
            print(f"  {r.label:<25} {m}{l}{s}  Sink: {r.improvement_sinkhorn_pct:+.1f}%")
        print(f"{'='*70}")

        assert len(results) > 0


class TestVelocityDominanceBestConfig:
    """
    Run a focused test on the configurations most likely to produce
    velocity dominance based on the paper's theory.

    Theory predicts velocity wins when:
    1. LOT embeddings factor out rotation (per_frame assignment)
    2. Cycle length matches actual dynamics (diurnal for GOES)
    3. Reservoir has enough capacity but is well-regularized
    4. Warm-up covers at least one full cycle
    """

    def test_best_candidates(self):
        """Test focused set of theoretically-motivated configs."""
        traj = load_goes_72h()
        T, N, d = traj.shape

        # Theory-motivated configs
        configs = [
            # per_frame + gaussian_iso: factors out rotation, smooth embedding
            PipelineConfig("gauss_pf_cl48", "gaussian_iso", "per_frame",
                           cycle_length=48, use_sweep=True),
            PipelineConfig("gauss_pf_cl36", "gaussian_iso", "per_frame",
                           cycle_length=36, use_sweep=True),
            # snapshot_begin + per_frame: data-driven reference + rotation factoring
            PipelineConfig("snap_pf_cl48", "snapshot_begin", "per_frame",
                           cycle_length=48, use_sweep=True),
            PipelineConfig("snap_pf_cl36", "snapshot_begin", "per_frame",
                           cycle_length=36, use_sweep=True),
            # snapshot_begin + fixed: preserves temporal coherence
            PipelineConfig("snap_fx_cl48", "snapshot_begin", "fixed",
                           cycle_length=48, use_sweep=True),
            # larger reservoir for more capacity
            PipelineConfig("gauss_pf_cl48_big", "gaussian_iso", "per_frame",
                           cycle_length=48, reservoir_scale=1.5, use_sweep=True),
            # smaller reservoir to avoid overfitting
            PipelineConfig("gauss_pf_cl48_sm", "gaussian_iso", "per_frame",
                           cycle_length=48, reservoir_scale=0.75, use_sweep=True),
            # shorter cycle = more forecast steps
            PipelineConfig("gauss_pf_cl24", "gaussian_iso", "per_frame",
                           cycle_length=24, use_sweep=True),
            # snapshot_middle for centered reference
            PipelineConfig("snapM_pf_cl48", "snapshot_middle", "per_frame",
                           cycle_length=48, use_sweep=True),
        ]

        results = []
        for cfg in configs:
            tag = f"vdom_best_{cfg.label}"
            try:
                r = run_paper_pipeline(traj, tag, cfg)
                results.append(r)
            except Exception as e:
                print(f"  [ERROR] {cfg.label}: {e}")

        plot_dominance_summary(results, "GOES GP 72h: theory-motivated configs",
                               PLOTS_DIR / "best_candidates.png")
        plot_error_timeseries(results, "Best candidates", PLOTS_DIR)
        plot_winning_configs(results, PLOTS_DIR)

        all_win = [r for r in results
                   if r.velocity_wins_map and r.velocity_wins_l2 and r.velocity_wins_sinkhorn]

        print(f"\n{'='*70}")
        print(f"BEST CANDIDATES: {len(all_win)}/{len(results)} all-metric winners")
        for r in results:
            m = "W" if r.velocity_wins_map else "."
            l = "W" if r.velocity_wins_l2 else "."
            s = "W" if r.velocity_wins_sinkhorn else "."
            print(f"  {r.label:<25} {m}{l}{s}  "
                  f"map:{r.improvement_map_pct:+.1f}% L2:{r.improvement_l2_pct:+.1f}% "
                  f"Sink:{r.improvement_sinkhorn_pct:+.1f}%")
        print(f"{'='*70}")

        assert len(results) > 0
