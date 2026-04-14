"""
Test velocity vs positions LOT-RC forecasting on REAL atmospheric and
oceanographic data at extended time horizons.

NO SYNTHETIC DATA.  Every trajectory comes from actual satellite observations:
  - GOES-16 Great Plains summer 72h  (T=145, N=200, 30-min cadence)
  - GOES-16 Gulf of Mexico 48h       (T=75,  N=200, 30-min cadence)
  - NOAA OISST sea-surface temp 180d (T=165, N=200, daily cadence)
  - NOAA OISST sea-surface temp 365d (T=366, N=200, daily cadence)

For each dataset we sweep hyperparameter configurations, run the full
pipeline (trajectory -> LOT embeddings -> VELOCITY vs POSITIONS forecast),
and identify where velocity wins.

Plots saved under plots/realworld_extended/:
  - Per-dataset error curves (velocity vs positions over forecast horizon)
  - Bar chart comparing velocity improvement across all datasets
  - Summary table identifying winning configurations
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PLOTS_DIR = REPO_ROOT / "plots" / "realworld_extended"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

def l2_sigma_per_step(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-timestep L^2(sigma) error: sqrt( (1/R) sum_i ||pred-true||^2 )."""
    diff = pred - true
    sq_norms = np.sum(diff ** 2, axis=-1)
    return np.sqrt(np.mean(sq_norms, axis=-1))


def l2_sigma_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Empirical L^2(sigma) RMSE over the forecast horizon."""
    assert pred.shape == true.shape, f"Shape mismatch: {pred.shape} vs {true.shape}"
    diff = pred - true
    sq_norms = np.sum(diff ** 2, axis=-1)
    per_time = np.mean(sq_norms, axis=-1)
    return float(np.sqrt(np.mean(per_time)))


def cumulative_l2(per_step: np.ndarray) -> np.ndarray:
    """Cumulative mean L^2(sigma) error up to each timestep."""
    return np.sqrt(np.cumsum(per_step ** 2) / np.arange(1, len(per_step) + 1))


# ══════════════════════════════════════════════════════════════
# PIPELINE LOADER
# ══════════════════════════════════════════════════════════════

def _load_pipeline():
    path = REPO_ROOT / "scripts" / "realworld_experiment_pipeline.py"
    spec = importlib.util.spec_from_file_location("realworld_experiment_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class ExperimentResult:
    tag: str
    dataset: str
    vel_l2_rmse: float
    pos_l2_rmse: float
    velocity_wins: bool
    improvement_pct: float
    config: dict
    vel_per_step: np.ndarray | None = field(default=None, repr=False)
    pos_per_step: np.ndarray | None = field(default=None, repr=False)
    data_shape: tuple = ()
    hp_label: str = ""


def run_experiment(
    *,
    tag: str,
    goes_dir: Path,
    n_cycles: int,
    lot_kind: str = "gaussian_iso",
    assignment: str = "fixed",
    reservoir_scale: float = 1.0,
    spectral_radius_vel: float | None = None,
    leak_rate_vel: float | None = None,
    ridge_vel: float | None = None,
    spectral_radius_pos: float | None = None,
    leak_rate_pos: float | None = None,
    ridge_pos: float | None = None,
) -> ExperimentResult:
    """Run the pipeline on real data and compute L^2(sigma) RMSE."""
    traj = np.load(goes_dir / "particles.npy")
    T, N, d = traj.shape

    mod = _load_pipeline()
    summary = mod.run_pipeline(
        system=tag,
        n_particles=N,
        n_steps=T,
        n_cycles=n_cycles,
        goes_dir=goes_dir,
        quick=False,
        ot_method="emd",
        ot_device="cpu",
        rc_backend="numpy",
        lot_kind=lot_kind,
        lot_frac=100,
        assignment=assignment,
        reservoir_scale=reservoir_scale,
        run_tag_velocity=f"v_{tag}",
        run_tag_positions=f"p_{tag}",
        spectral_radius_vel=spectral_radius_vel,
        leak_rate_vel=leak_rate_vel,
        ridge_vel=ridge_vel,
        spectral_radius_pos=spectral_radius_pos,
        leak_rate_pos=leak_rate_pos,
        ridge_pos=ridge_pos,
    )

    vel_dir = Path(summary["velocity_forecast_dir"])
    pos_dir = Path(summary["positions_forecast_dir"])

    vel_pred = np.load(vel_dir / "predicted_maps.npy")
    vel_true = np.load(vel_dir / "true_future_maps.npy")
    pos_pred = np.load(pos_dir / "predicted_maps.npy")
    pos_true = np.load(pos_dir / "true_future_maps.npy")

    if vel_pred.shape[0] == vel_true.shape[0] + 1:
        vel_pred = vel_pred[1:]

    Tf = min(vel_pred.shape[0], pos_pred.shape[0])
    vel_pred, vel_true = vel_pred[:Tf], vel_true[:Tf]
    pos_pred, pos_true = pos_pred[:Tf], pos_true[:Tf]

    vel_l2 = l2_sigma_rmse(vel_pred, vel_true)
    pos_l2 = l2_sigma_rmse(pos_pred, pos_true)

    vel_ps = l2_sigma_per_step(vel_pred, vel_true)
    pos_ps = l2_sigma_per_step(pos_pred, pos_true)

    velocity_wins = vel_l2 < pos_l2
    improvement = (pos_l2 - vel_l2) / pos_l2 * 100 if pos_l2 > 0 else 0.0

    print(f"  forecast_window={Tf} steps (vel_pred={vel_pred.shape}, pos_pred={pos_pred.shape})")

    return ExperimentResult(
        tag=tag,
        dataset=goes_dir.name,
        vel_l2_rmse=vel_l2,
        pos_l2_rmse=pos_l2,
        velocity_wins=velocity_wins,
        improvement_pct=improvement,
        config=summary,
        vel_per_step=vel_ps,
        pos_per_step=pos_ps,
        data_shape=(T, N, d),
    )


# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_error_curves(results: list[ExperimentResult], dataset_label: str, out_dir: Path):
    """Per-step and cumulative error curves for velocity vs positions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for r in results:
        if r.vel_per_step is None or r.pos_per_step is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Per-step error
        ax = axes[0]
        steps = np.arange(len(r.vel_per_step))
        ax.plot(steps, r.vel_per_step, label="Velocity", color="tab:blue", linewidth=1.5)
        ax.plot(steps, r.pos_per_step, label="Positions", color="tab:red", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Forecast step")
        ax.set_ylabel(r"$L^2(\sigma)$ error")
        ax.set_title(f"Per-step error — {r.hp_label or r.tag}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cumulative error
        ax = axes[1]
        vel_cum = cumulative_l2(r.vel_per_step)
        pos_cum = cumulative_l2(r.pos_per_step)
        ax.plot(steps, vel_cum, label="Velocity (cumulative)", color="tab:blue", linewidth=1.5)
        ax.plot(steps, pos_cum, label="Positions (cumulative)", color="tab:red", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Forecast step")
        ax.set_ylabel(r"Cumulative $L^2(\sigma)$ RMSE")
        win_str = f"VELOCITY WINS ({r.improvement_pct:+.1f}%)" if r.velocity_wins else f"POSITIONS wins ({r.improvement_pct:+.1f}%)"
        ax.set_title(f"Cumulative RMSE — {win_str}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"{dataset_label}: {r.tag}", fontsize=12, fontweight="bold")
        fig.tight_layout()

        safe_tag = r.tag.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"error_curves_{safe_tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_crossover_analysis(results: list[ExperimentResult], dataset_label: str, out_dir: Path):
    """Show where velocity error crosses below positions error over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick best velocity config (lowest vel RMSE)
    valid = [r for r in results if r.vel_per_step is not None]
    if not valid:
        return

    best_vel = min(valid, key=lambda r: r.vel_l2_rmse)
    best_pos = min(valid, key=lambda r: r.pos_l2_rmse)

    fig, ax = plt.subplots(figsize=(12, 5))
    steps = np.arange(len(best_vel.vel_per_step))

    ax.plot(steps, best_vel.vel_per_step, label=f"Velocity ({best_vel.hp_label})",
            color="tab:blue", linewidth=2)
    ax.plot(steps, best_vel.pos_per_step, label=f"Positions ({best_vel.hp_label})",
            color="tab:red", linewidth=2, alpha=0.8)

    # Shade regions where velocity wins
    vel_better = best_vel.vel_per_step < best_vel.pos_per_step
    ax.fill_between(steps, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                     where=vel_better, alpha=0.08, color="blue", label="Velocity better")

    # Mark crossover points
    crossovers = np.where(np.diff(vel_better.astype(int)) != 0)[0]
    for cx in crossovers:
        ax.axvline(cx, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Forecast step")
    ax.set_ylabel(r"$L^2(\sigma)$ error")
    ax.set_title(f"{dataset_label}: Velocity vs Positions — crossover analysis")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"crossover_{dataset_label.replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bar(all_results: dict[str, list[ExperimentResult]], out_dir: Path):
    """Bar chart: best improvement % for velocity across all datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = []
    vel_rmses = []
    pos_rmses = []
    improvements = []
    colors = []

    for ds_name, results in all_results.items():
        if not results:
            continue
        # Use the config with the best velocity improvement
        best = max(results, key=lambda r: r.improvement_pct)
        datasets.append(ds_name)
        vel_rmses.append(best.vel_l2_rmse)
        pos_rmses.append(best.pos_l2_rmse)
        improvements.append(best.improvement_pct)
        colors.append("tab:blue" if best.velocity_wins else "tab:red")

    if not datasets:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # RMSE comparison
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.35
    ax.bar(x - width / 2, vel_rmses, width, label="Velocity", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, pos_rmses, width, label="Positions", color="tab:red", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"$L^2(\sigma)$ RMSE")
    ax.set_title("Best config: Velocity vs Positions RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Improvement %
    ax = axes[1]
    bars = ax.bar(x, improvements, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Velocity improvement (%)")
    ax.set_title("Velocity improvement over Positions (best config)")
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{imp:+.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Real-World Extended Forecasting: Velocity vs Positions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_bar_all_datasets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_ratio_over_time(all_results: dict[str, list[ExperimentResult]], out_dir: Path):
    """For each dataset, plot velocity/positions error ratio over time (best config)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab10

    for i, (ds_name, results) in enumerate(all_results.items()):
        valid = [r for r in results if r.vel_per_step is not None and r.pos_per_step is not None]
        if not valid:
            continue
        best = max(valid, key=lambda r: r.improvement_pct)
        # Avoid division by zero
        ratio = best.vel_per_step / np.maximum(best.pos_per_step, 1e-12)
        steps = np.arange(len(ratio))
        ax.plot(steps, ratio, label=ds_name, color=cmap(i), linewidth=1.5)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Break-even")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Velocity / Positions error ratio")
    ax.set_title("Error ratio over forecast horizon (< 1.0 = velocity wins)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "error_ratio_all_datasets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hp_sensitivity(results: list[ExperimentResult], dataset_label: str, out_dir: Path):
    """Show how hyperparameters affect velocity vs positions performance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(results) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    tags = [r.hp_label or r.tag for r in results]
    vel_vals = [r.vel_l2_rmse for r in results]
    pos_vals = [r.pos_l2_rmse for r in results]

    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width / 2, vel_vals, width, label="Velocity", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, pos_vals, width, label="Positions", color="tab:red", alpha=0.8)

    # Star the winners
    for i, r in enumerate(results):
        if r.velocity_wins:
            ax.annotate("*", (i - width / 2, vel_vals[i]), ha="center",
                        fontsize=16, color="green", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"$L^2(\sigma)$ RMSE")
    ax.set_title(f"{dataset_label}: HP sensitivity (* = velocity wins)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / f"hp_sensitivity_{dataset_label.replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# HYPERPARAMETER CONFIGURATIONS
# ══════════════════════════════════════════════════════════════

# (label, lot_kind, assignment, reservoir_scale, sr_v, lk_v, rd_v, sr_p, lk_p, rd_p)
HP_CONFIGS = [
    ("default",             "gaussian_iso", "fixed",     1.0,  None, None, None,   None, None, None),
    ("per_frame",           "gaussian_iso", "per_frame", 1.0,  None, None, None,   None, None, None),
    ("sr0.85_lk0.85",       "gaussian_iso", "fixed",     1.0,  0.85, 0.85, 0.02,  None, None, None),
    ("sr0.80_lk0.5",        "gaussian_iso", "fixed",     1.0,  0.80, 0.50, 0.01,  None, None, None),
    ("sr0.95_lk0.3",        "gaussian_iso", "fixed",     1.0,  0.95, 0.30, 0.005, None, None, None),
    ("uniform_sq",          "uniform_square", "fixed",   1.0,  None, None, None,   None, None, None),
    ("scale0.75",           "gaussian_iso", "fixed",     0.75, None, None, None,   None, None, None),
    ("sr0.9_lk0.7_rd0.005", "gaussian_iso", "fixed",     1.0,  0.90, 0.70, 0.005, None, None, None),
]


# ══════════════════════════════════════════════════════════════
# DATASET DEFINITIONS
# ══════════════════════════════════════════════════════════════

@dataclass
class RealDataset:
    name: str
    dir_name: str
    n_cycles_estimate: int
    description: str


REAL_DATASETS = [
    RealDataset(
        name="GOES GP 72h",
        dir_name="goes_data_gp_summer_72h",
        n_cycles_estimate=3,       # ~3 diurnal cycles in 72h
        description="GOES-16 Great Plains summer, 72h, 30-min cadence, IR band C13",
    ),
    RealDataset(
        name="GOES Gulf 48h",
        dir_name="goes_data_gulf_48h",
        n_cycles_estimate=3,       # semi-diurnal + diurnal tides in 48h
        description="GOES-16 Gulf of Mexico, 48h, 30-min cadence, IR band C13",
    ),
    RealDataset(
        name="SST 180d",
        dir_name="sst_data_180d",
        n_cycles_estimate=6,       # ~30-day synoptic weather cycles
        description="NOAA OISST, 180 days, daily cadence, Gulf of Mexico",
    ),
    RealDataset(
        name="SST 365d",
        dir_name="sst_data_365d",
        n_cycles_estimate=12,      # ~30-day synoptic weather cycles
        description="NOAA OISST, 365 days, daily cadence, Gulf of Mexico",
    ),
]


# ══════════════════════════════════════════════════════════════
# HELPER: run all HP configs on one dataset
# ══════════════════════════════════════════════════════════════

def sweep_dataset(ds: RealDataset) -> list[ExperimentResult]:
    """Run all HP configs on a single real dataset."""
    data_dir = REPO_ROOT / ds.dir_name
    if not (data_dir / "particles.npy").is_file():
        pytest.skip(f"{data_dir}/particles.npy not found")

    traj = np.load(data_dir / "particles.npy")
    T, N, d = traj.shape
    print(f"\n{'='*60}")
    print(f"Dataset: {ds.name} — {ds.description}")
    print(f"  Shape: T={T}, N={N}, d={d}")
    print(f"  Estimated cycles: {ds.n_cycles_estimate}")
    print(f"{'='*60}")

    results = []
    for label, kind, assign, scale, sr_v, lk_v, rd_v, sr_p, lk_p, rd_p in HP_CONFIGS:
        tag = f"rw_{ds.dir_name}_{label}"
        print(f"\n  Running: {label} ...", end=" ", flush=True)
        try:
            r = run_experiment(
                tag=tag,
                goes_dir=data_dir,
                n_cycles=ds.n_cycles_estimate,
                lot_kind=kind,
                assignment=assign,
                reservoir_scale=scale,
                spectral_radius_vel=sr_v,
                leak_rate_vel=lk_v,
                ridge_vel=rd_v,
                spectral_radius_pos=sr_p,
                leak_rate_pos=lk_p,
                ridge_pos=rd_p,
            )
            r.hp_label = label
            results.append(r)
            status = "WIN" if r.velocity_wins else "loss"
            print(
                f"[{status}] vel={r.vel_l2_rmse:.6f}  pos={r.pos_l2_rmse:.6f}  "
                f"improvement={r.improvement_pct:+.1f}%"
            )
        except Exception as e:
            print(f"[ERROR] {e}")

    return results


# ══════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════

class TestGOES72h:
    """GOES Great Plains 72-hour (T=145) real satellite cloud data."""

    @pytest.fixture(autouse=True)
    def _check_data(self):
        p = REPO_ROOT / "goes_data_gp_summer_72h" / "particles.npy"
        if not p.is_file():
            pytest.skip("goes_data_gp_summer_72h/particles.npy not found")

    def test_velocity_vs_positions_sweep(self):
        ds = REAL_DATASETS[0]  # GOES GP 72h
        results = sweep_dataset(ds)

        plot_error_curves(results, ds.name, PLOTS_DIR)
        plot_crossover_analysis(results, ds.name, PLOTS_DIR)
        plot_hp_sensitivity(results, ds.name, PLOTS_DIR)

        wins = [r for r in results if r.velocity_wins]
        print(f"\n  GOES GP 72h: {len(wins)}/{len(results)} velocity wins")
        assert len(results) > 0, "No experiments completed"


class TestGOESGulf48h:
    """GOES Gulf of Mexico 48-hour (T=75) real satellite cloud data."""

    @pytest.fixture(autouse=True)
    def _check_data(self):
        p = REPO_ROOT / "goes_data_gulf_48h" / "particles.npy"
        if not p.is_file():
            pytest.skip("goes_data_gulf_48h/particles.npy not found")

    def test_velocity_vs_positions_sweep(self):
        ds = REAL_DATASETS[1]  # GOES Gulf 48h
        results = sweep_dataset(ds)

        plot_error_curves(results, ds.name, PLOTS_DIR)
        plot_crossover_analysis(results, ds.name, PLOTS_DIR)
        plot_hp_sensitivity(results, ds.name, PLOTS_DIR)

        wins = [r for r in results if r.velocity_wins]
        print(f"\n  GOES Gulf 48h: {len(wins)}/{len(results)} velocity wins")
        assert len(results) > 0, "No experiments completed"


class TestSST180d:
    """NOAA OISST 180-day (T=165) real sea-surface temperature data."""

    @pytest.fixture(autouse=True)
    def _check_data(self):
        p = REPO_ROOT / "sst_data_180d" / "particles.npy"
        if not p.is_file():
            pytest.skip("sst_data_180d/particles.npy not found")

    def test_velocity_vs_positions_sweep(self):
        ds = REAL_DATASETS[2]  # SST 180d
        results = sweep_dataset(ds)

        plot_error_curves(results, ds.name, PLOTS_DIR)
        plot_crossover_analysis(results, ds.name, PLOTS_DIR)
        plot_hp_sensitivity(results, ds.name, PLOTS_DIR)

        wins = [r for r in results if r.velocity_wins]
        print(f"\n  SST 180d: {len(wins)}/{len(results)} velocity wins")
        assert len(results) > 0, "No experiments completed"


class TestSST365d:
    """NOAA OISST 365-day (T=366) real sea-surface temperature data."""

    @pytest.fixture(autouse=True)
    def _check_data(self):
        p = REPO_ROOT / "sst_data_365d" / "particles.npy"
        if not p.is_file():
            pytest.skip("sst_data_365d/particles.npy not found")

    def test_velocity_vs_positions_sweep(self):
        ds = REAL_DATASETS[3]  # SST 365d
        results = sweep_dataset(ds)

        plot_error_curves(results, ds.name, PLOTS_DIR)
        plot_crossover_analysis(results, ds.name, PLOTS_DIR)
        plot_hp_sensitivity(results, ds.name, PLOTS_DIR)

        wins = [r for r in results if r.velocity_wins]
        print(f"\n  SST 365d: {len(wins)}/{len(results)} velocity wins")
        assert len(results) > 0, "No experiments completed"


# ══════════════════════════════════════════════════════════════
# COMBINED CROSS-DATASET ANALYSIS
# ══════════════════════════════════════════════════════════════

class TestCrossDatasetAnalysis:
    """Run all available real datasets and produce combined comparison plots."""

    def test_all_real_datasets(self):
        all_results: dict[str, list[ExperimentResult]] = {}
        available = 0

        for ds in REAL_DATASETS:
            data_dir = REPO_ROOT / ds.dir_name
            if not (data_dir / "particles.npy").is_file():
                print(f"\n  SKIP: {ds.name} ({data_dir} not found)")
                continue
            available += 1
            results = sweep_dataset(ds)
            all_results[ds.name] = results

            # Per-dataset plots
            plot_error_curves(results, ds.name, PLOTS_DIR)
            plot_crossover_analysis(results, ds.name, PLOTS_DIR)
            plot_hp_sensitivity(results, ds.name, PLOTS_DIR)

        if available == 0:
            pytest.skip("No real datasets found")

        # Cross-dataset plots
        plot_summary_bar(all_results, PLOTS_DIR)
        plot_error_ratio_over_time(all_results, PLOTS_DIR)

        # Write comprehensive JSON report
        report = {
            "description": "Real-world extended forecasting: velocity vs positions comparison",
            "metric": "L2_sigma_RMSE",
            "datasets": {},
        }
        total_wins = 0
        total_configs = 0

        for ds_name, results in all_results.items():
            wins = [r for r in results if r.velocity_wins]
            total_wins += len(wins)
            total_configs += len(results)

            ds_report = {
                "n_configs": len(results),
                "n_velocity_wins": len(wins),
                "velocity_win_rate": len(wins) / len(results) if results else 0,
                "configs": [],
            }
            for r in results:
                ds_report["configs"].append({
                    "tag": r.tag,
                    "hp_label": r.hp_label,
                    "data_shape": list(r.data_shape),
                    "velocity_l2_rmse": r.vel_l2_rmse,
                    "positions_l2_rmse": r.pos_l2_rmse,
                    "velocity_wins": r.velocity_wins,
                    "improvement_pct": r.improvement_pct,
                })

            if wins:
                best = max(wins, key=lambda r: r.improvement_pct)
                ds_report["best_velocity_config"] = {
                    "hp_label": best.hp_label,
                    "velocity_l2_rmse": best.vel_l2_rmse,
                    "positions_l2_rmse": best.pos_l2_rmse,
                    "improvement_pct": best.improvement_pct,
                }
            report["datasets"][ds_name] = ds_report

        report["summary"] = {
            "total_configs": total_configs,
            "total_velocity_wins": total_wins,
            "overall_velocity_win_rate": total_wins / total_configs if total_configs > 0 else 0,
            "datasets_tested": list(all_results.keys()),
        }

        report_path = REPO_ROOT / "realworld_extended_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {report_path}")
        print(f"Plots saved to {PLOTS_DIR}")

        # Print summary table
        print(f"\n{'='*70}")
        print("REAL-WORLD EXTENDED FORECASTING SUMMARY")
        print(f"{'='*70}")
        print(f"{'Dataset':<20} {'Wins':>6} {'Total':>6} {'Rate':>8} {'Best Impr':>12}")
        print(f"{'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*12}")
        for ds_name, results in all_results.items():
            wins = [r for r in results if r.velocity_wins]
            rate = len(wins) / len(results) if results else 0
            best_imp = max((r.improvement_pct for r in results), default=0)
            print(f"{ds_name:<20} {len(wins):>6} {len(results):>6} {rate:>7.0%} {best_imp:>+11.1f}%")
        print(f"{'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*12}")
        print(f"{'TOTAL':<20} {total_wins:>6} {total_configs:>6} "
              f"{total_wins/total_configs if total_configs else 0:>7.0%}")
        print(f"{'='*70}")

        assert total_configs > 0, "No experiments completed across any dataset"
