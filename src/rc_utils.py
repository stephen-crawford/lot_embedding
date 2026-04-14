#!/usr/bin/env python3
"""
RC Hyperparameter Utilities

Helper functions to determine optimal Reservoir Computing hyperparameters
for LOT+RC forecasting.

Key hyperparameters:
    - spectral_radius: Controls memory/stability trade-off (0.1 to 1.0)
    - input_scaling: Scales input weights (0.01 to 1.0)
    - leak_rate: Leaky integration parameter (0.1 to 1.0)
    - ridge_param: Regularization strength (1e-8 to 1e-1)
    - reservoir_size: Number of reservoir neurons

Usage:
    from rc_utils import (
        suggest_hyperparameters,
        grid_search_cv,
        analyze_dynamics,
        RCHyperparameterSearch,
    )

    # Quick suggestion based on data
    params = suggest_hyperparameters(velocities, method="velocity")

    # Full grid search with cross-validation
    searcher = RCHyperparameterSearch(input_data, target_data)
    best_params, results = searcher.grid_search()
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import json
import time

try:
    from .rc_computer import ReservoirComputer, ReservoirConfig, InitScheme
except ImportError:
    from rc_computer import ReservoirComputer, ReservoirConfig, InitScheme


# ═══════════════════════════════════════════════════════════════
# DEFAULT PARAMETER GRIDS
# ═══════════════════════════════════════════════════════════════

# For VELOCITY forecasting (v_t → v_{t+1})
VELOCITY_DEFAULTS = {
    "spectral_radius": 0.7,
    "input_scaling": 0.1,
    "leak_rate": 0.7,
    "ridge_param": 1e-6,
}

# For POSITIONS forecasting (T_t → T_{t+1})
POSITIONS_DEFAULTS = {
    "spectral_radius": 0.95,
    "input_scaling": 0.1,
    "leak_rate": 0.9,
    "ridge_param": 1e-6,
}

# Standard grid for hyperparameter search
STANDARD_GRID = {
    "spectral_radius": [0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
    "input_scaling": [0.01, 0.05, 0.1, 0.2, 0.5],
    "leak_rate": [0.3, 0.5, 0.7, 0.9, 1.0],
    "ridge_param": [1e-8, 1e-6, 1e-4, 1e-2],
}

# Coarse grid for quick search
COARSE_GRID = {
    "spectral_radius": [0.7, 0.9, 0.99],
    "input_scaling": [0.1, 0.5],
    "leak_rate": [0.5, 0.9],
    "ridge_param": [1e-6, 1e-2],
}

# Fine grid for refinement
FINE_GRID = {
    "spectral_radius": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    "input_scaling": [0.05, 0.1, 0.15, 0.2],
    "leak_rate": [0.6, 0.7, 0.8, 0.9],
    "ridge_param": [1e-7, 1e-6, 1e-5, 1e-4],
}


# ═══════════════════════════════════════════════════════════════
# DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════

@dataclass
class DynamicsAnalysis:
    """Analysis of input dynamics to inform hyperparameter selection."""

    # Basic statistics
    mean_magnitude: float
    std_magnitude: float
    max_magnitude: float

    # Temporal characteristics
    autocorrelation_lag1: float
    autocorrelation_lag5: float
    stationarity_score: float  # 0 = non-stationary, 1 = stationary

    # Spectral properties
    dominant_frequency: Optional[float]
    spectral_entropy: float

    # Recommendations
    suggested_spectral_radius: float
    suggested_leak_rate: float
    suggested_input_scaling: float
    suggested_ridge_param: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_magnitude": self.mean_magnitude,
            "std_magnitude": self.std_magnitude,
            "max_magnitude": self.max_magnitude,
            "autocorrelation_lag1": self.autocorrelation_lag1,
            "autocorrelation_lag5": self.autocorrelation_lag5,
            "stationarity_score": self.stationarity_score,
            "dominant_frequency": self.dominant_frequency,
            "spectral_entropy": self.spectral_entropy,
            "suggested_spectral_radius": self.suggested_spectral_radius,
            "suggested_leak_rate": self.suggested_leak_rate,
            "suggested_input_scaling": self.suggested_input_scaling,
            "suggested_ridge_param": self.suggested_ridge_param,
        }


def analyze_dynamics(data: np.ndarray, verbose: bool = False) -> DynamicsAnalysis:
    """
    Analyze input dynamics to suggest hyperparameters.

    Args:
        data: Input time series, shape (T, D) or (T, N, d)
        verbose: Print analysis details

    Returns:
        DynamicsAnalysis with statistics and recommendations
    """
    # Flatten if needed
    if data.ndim == 3:
        T, N, d = data.shape
        data_flat = data.reshape(T, N * d)
    else:
        data_flat = data
        T, D = data.shape

    # Basic statistics
    magnitudes = np.linalg.norm(data_flat, axis=1)
    mean_mag = float(np.mean(magnitudes))
    std_mag = float(np.std(magnitudes))
    max_mag = float(np.max(magnitudes))

    # Autocorrelation
    def autocorr(x, lag):
        if lag >= len(x):
            return 0.0
        n = len(x)
        mean = np.mean(x)
        var = np.var(x)
        if var < 1e-10:
            return 1.0
        return np.mean((x[:-lag] - mean) * (x[lag:] - mean)) / var

    ac_lag1 = autocorr(magnitudes, 1)
    ac_lag5 = autocorr(magnitudes, 5)

    # Stationarity score (simple rolling mean variance check)
    window = max(10, T // 10)
    if T > 2 * window:
        rolling_means = [np.mean(magnitudes[i:i+window]) for i in range(0, T - window, window)]
        stationarity = 1.0 - min(1.0, np.std(rolling_means) / (mean_mag + 1e-10))
    else:
        stationarity = 0.5  # Unknown

    # Spectral analysis
    try:
        fft = np.fft.rfft(magnitudes - np.mean(magnitudes))
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(T)

        # Dominant frequency (excluding DC)
        if len(power) > 1:
            peak_idx = np.argmax(power[1:]) + 1
            dominant_freq = float(freqs[peak_idx])
        else:
            dominant_freq = None

        # Spectral entropy
        power_norm = power / (np.sum(power) + 1e-10)
        spectral_entropy = float(-np.sum(power_norm * np.log(power_norm + 1e-10)))
    except Exception:
        dominant_freq = None
        spectral_entropy = 0.0

    # Generate recommendations based on analysis

    # Spectral radius: higher for more stationary/predictable dynamics
    if stationarity > 0.8 and ac_lag1 > 0.9:
        sr = 0.95  # Very stable dynamics
    elif stationarity > 0.5 and ac_lag1 > 0.7:
        sr = 0.8   # Moderately stable
    elif ac_lag1 > 0.5:
        sr = 0.7   # Some structure
    else:
        sr = 0.6   # Chaotic/noisy

    # Leak rate: higher for slower dynamics
    if ac_lag5 > 0.8:
        lr = 0.9   # Slow dynamics, need memory
    elif ac_lag5 > 0.5:
        lr = 0.7   # Moderate dynamics
    else:
        lr = 0.5   # Fast dynamics

    # Input scaling: inversely related to magnitude
    if max_mag > 10:
        ins = 0.01
    elif max_mag > 1:
        ins = 0.1
    else:
        ins = 0.5

    # Ridge param: higher for noisier data
    if stationarity > 0.8:
        rp = 1e-6
    elif stationarity > 0.5:
        rp = 1e-4
    else:
        rp = 1e-2

    analysis = DynamicsAnalysis(
        mean_magnitude=mean_mag,
        std_magnitude=std_mag,
        max_magnitude=max_mag,
        autocorrelation_lag1=ac_lag1,
        autocorrelation_lag5=ac_lag5,
        stationarity_score=stationarity,
        dominant_frequency=dominant_freq,
        spectral_entropy=spectral_entropy,
        suggested_spectral_radius=sr,
        suggested_leak_rate=lr,
        suggested_input_scaling=ins,
        suggested_ridge_param=rp,
    )

    if verbose:
        print(f"\n{'='*50}")
        print("DYNAMICS ANALYSIS")
        print(f"{'='*50}")
        print(f"Mean magnitude:     {mean_mag:.4f}")
        print(f"Std magnitude:      {std_mag:.4f}")
        print(f"Max magnitude:      {max_mag:.4f}")
        print(f"Autocorr (lag 1):   {ac_lag1:.4f}")
        print(f"Autocorr (lag 5):   {ac_lag5:.4f}")
        print(f"Stationarity:       {stationarity:.4f}")
        print(f"Spectral entropy:   {spectral_entropy:.4f}")
        print(f"\nSUGGESTED PARAMETERS:")
        print(f"  spectral_radius:  {sr}")
        print(f"  leak_rate:        {lr}")
        print(f"  input_scaling:    {ins}")
        print(f"  ridge_param:      {rp}")
        print(f"{'='*50}\n")

    return analysis


def suggest_hyperparameters(
    data: np.ndarray,
    method: str = "velocity",
    analyze: bool = True,
) -> Dict[str, float]:
    """
    Suggest hyperparameters based on data and forecasting method.

    Args:
        data: Input time series (velocities or maps)
        method: "velocity" or "positions"
        analyze: If True, analyze data; if False, use method defaults

    Returns:
        Dictionary of suggested hyperparameters
    """
    if method == "velocity":
        defaults = VELOCITY_DEFAULTS.copy()
    elif method == "positions":
        defaults = POSITIONS_DEFAULTS.copy()
    else:
        raise ValueError(f"Unknown method: {method}")

    if analyze and data is not None:
        analysis = analyze_dynamics(data, verbose=False)
        return {
            "spectral_radius": analysis.suggested_spectral_radius,
            "input_scaling": analysis.suggested_input_scaling,
            "leak_rate": analysis.suggested_leak_rate,
            "ridge_param": analysis.suggested_ridge_param,
        }

    return defaults


# ═══════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════

def time_series_cv_splits(
    T: int,
    n_splits: int = 5,
    min_train_size: int = 50,
    test_size: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate time-series cross-validation splits.

    Uses expanding window: train on [0, split_point], test on [split_point, split_point + test_size]

    Args:
        T: Total number of time steps
        n_splits: Number of CV splits
        min_train_size: Minimum training set size
        test_size: Size of each test set (default: T // (n_splits + 1))

    Returns:
        List of (train_indices, test_indices) tuples
    """
    if test_size is None:
        test_size = max(10, T // (n_splits + 1))

    splits = []
    step = (T - min_train_size - test_size) // n_splits

    for i in range(n_splits):
        train_end = min_train_size + i * step
        test_start = train_end
        test_end = min(test_start + test_size, T)

        if test_end <= test_start:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


def evaluate_rc_config(
    input_data: np.ndarray,
    target_data: np.ndarray,
    config: ReservoirConfig,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate a single RC configuration on train/test split.

    Args:
        input_data: Input sequences, shape (T, D)
        target_data: Target sequences, shape (T, D)
        config: RC configuration to evaluate
        train_idx: Training indices
        test_idx: Test indices

    Returns:
        Dictionary with train_rmse, test_rmse, and other metrics
    """
    rc = ReservoirComputer(config)

    # Train
    train_input = input_data[train_idx]
    train_target = target_data[train_idx]

    states = rc.run(train_input)
    rc.train(states, train_target)

    train_pred = rc.predict(states)
    train_rmse = float(np.sqrt(np.mean((train_pred - train_target) ** 2)))

    # Test (autonomous rollout)
    test_steps = len(test_idx)
    _, pred_flat = rc.run_autonomous(train_input, test_steps)

    test_target = target_data[test_idx]
    test_rmse = float(np.sqrt(np.mean((pred_flat - test_target) ** 2)))

    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "test_steps": test_steps,
    }


# ═══════════════════════════════════════════════════════════════
# GRID SEARCH
# ═══════════════════════════════════════════════════════════════

@dataclass
class GridSearchResult:
    """Result of hyperparameter grid search."""

    best_params: Dict[str, float]
    best_score: float
    all_results: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_configs": len(self.all_results),
        }

    def save(self, path: Path):
        """Save results to JSON."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump({
                "best_params": self.best_params,
                "best_score": self.best_score,
                "all_results": self.all_results,
            }, f, indent=2)


class RCHyperparameterSearch:
    """
    Hyperparameter search for Reservoir Computing.

    Example:
        searcher = RCHyperparameterSearch(
            input_data=velocities[:-1],
            target_data=velocities[1:],
            reservoir_size=500,
        )

        # Coarse search first
        result = searcher.grid_search(grid=COARSE_GRID)
        print(f"Best params: {result.best_params}")

        # Fine search around best
        fine_result = searcher.refine_search(result.best_params)
    """

    def __init__(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        reservoir_size: int = 500,
        n_cv_splits: int = 3,
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize hyperparameter search.

        Args:
            input_data: Input sequences, shape (T, D) or (T, N, d)
            target_data: Target sequences, shape (T, D) or (T, N, d)
            reservoir_size: Size of reservoir
            n_cv_splits: Number of cross-validation splits
            random_seed: Random seed for reproducibility
            verbose: Print progress
        """
        # Flatten if needed
        if input_data.ndim == 3:
            T, N, d = input_data.shape
            self.input_data = input_data.reshape(T, N * d)
            self.target_data = target_data.reshape(T, N * d)
            self.input_size = N * d
        else:
            self.input_data = input_data
            self.target_data = target_data
            self.input_size = input_data.shape[1]

        self.reservoir_size = reservoir_size
        self.n_cv_splits = n_cv_splits
        self.random_seed = random_seed
        self.verbose = verbose

        # Generate CV splits
        T = len(self.input_data)
        self.cv_splits = time_series_cv_splits(T, n_cv_splits)

        if verbose:
            print(f"[RCHyperparameterSearch] T={T}, D={self.input_size}, "
                  f"reservoir_size={reservoir_size}, cv_splits={len(self.cv_splits)}")

    def _create_config(self, params: Dict[str, float]) -> ReservoirConfig:
        """Create RC config from parameters."""
        return ReservoirConfig(
            input_size=self.input_size,
            reservoir_size=self.reservoir_size,
            output_size=self.input_size,
            spectral_radius=params.get("spectral_radius", 0.9),
            input_scaling=params.get("input_scaling", 0.1),
            leak_rate=params.get("leak_rate", 1.0),
            ridge_param=params.get("ridge_param", 1e-6),
            random_seed=self.random_seed,
        )

    def evaluate_params(self, params: Dict[str, float]) -> float:
        """
        Evaluate parameters using cross-validation.

        Returns mean test RMSE across CV splits.
        """
        config = self._create_config(params)

        test_rmses = []
        for train_idx, test_idx in self.cv_splits:
            try:
                result = evaluate_rc_config(
                    self.input_data, self.target_data,
                    config, train_idx, test_idx
                )
                test_rmses.append(result["test_rmse"])
            except Exception as e:
                # Return large value if evaluation fails
                test_rmses.append(float('inf'))

        return float(np.mean(test_rmses))

    def grid_search(
        self,
        grid: Optional[Dict[str, List[float]]] = None,
        metric: str = "test_rmse",
    ) -> GridSearchResult:
        """
        Perform grid search over hyperparameters.

        Args:
            grid: Dictionary mapping parameter names to lists of values
            metric: Metric to optimize ("test_rmse")

        Returns:
            GridSearchResult with best parameters and all results
        """
        if grid is None:
            grid = COARSE_GRID

        # Generate all combinations
        from itertools import product

        param_names = list(grid.keys())
        param_values = list(grid.values())

        all_results = []
        best_score = float('inf')
        best_params = None

        total_configs = 1
        for v in param_values:
            total_configs *= len(v)

        if self.verbose:
            print(f"\n[GridSearch] Testing {total_configs} configurations...")

        t0 = time.perf_counter()

        for i, values in enumerate(product(*param_values)):
            params = dict(zip(param_names, values))

            score = self.evaluate_params(params)

            result = {
                **params,
                "score": score,
            }
            all_results.append(result)

            if score < best_score:
                best_score = score
                best_params = params.copy()

            if self.verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{total_configs}] best_score={best_score:.6f}")

        elapsed = time.perf_counter() - t0

        if self.verbose:
            print(f"\n[GridSearch] Complete in {elapsed:.1f}s")
            print(f"  Best score: {best_score:.6f}")
            print(f"  Best params: {best_params}")

        return GridSearchResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
        )

    def refine_search(
        self,
        center_params: Dict[str, float],
        n_steps: int = 5,
        shrink_factor: float = 0.5,
    ) -> GridSearchResult:
        """
        Refine search around a center point.

        Creates a fine grid around the center parameters.

        Args:
            center_params: Center point for refinement
            n_steps: Number of steps in each direction
            shrink_factor: How much to shrink the range

        Returns:
            GridSearchResult with refined best parameters
        """
        # Create fine grid around center
        fine_grid = {}

        for param, center_val in center_params.items():
            if param == "ridge_param":
                # Log scale for ridge
                log_center = np.log10(center_val)
                log_range = 1.0 * shrink_factor
                log_vals = np.linspace(log_center - log_range, log_center + log_range, n_steps)
                fine_grid[param] = [10**v for v in log_vals]
            else:
                # Linear scale for others
                if param == "spectral_radius":
                    delta = 0.1 * shrink_factor
                    vals = np.linspace(max(0.1, center_val - delta),
                                       min(0.99, center_val + delta), n_steps)
                elif param == "leak_rate":
                    delta = 0.2 * shrink_factor
                    vals = np.linspace(max(0.1, center_val - delta),
                                       min(1.0, center_val + delta), n_steps)
                else:  # input_scaling
                    delta = center_val * 0.5 * shrink_factor
                    vals = np.linspace(max(0.001, center_val - delta),
                                       center_val + delta, n_steps)
                fine_grid[param] = list(vals)

        if self.verbose:
            print(f"\n[RefineSearch] Around {center_params}")

        return self.grid_search(fine_grid)


# ═══════════════════════════════════════════════════════════════
# QUICK UTILITIES
# ═══════════════════════════════════════════════════════════════

def quick_tune(
    input_data: np.ndarray,
    target_data: np.ndarray,
    reservoir_size: int = 500,
    method: str = "velocity",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Quick hyperparameter tuning with sensible defaults.

    Does a coarse search followed by refinement.

    Args:
        input_data: Input sequences
        target_data: Target sequences
        reservoir_size: Reservoir size
        method: "velocity" or "positions"
        verbose: Print progress

    Returns:
        Best hyperparameters
    """
    # Start with data analysis
    analysis = analyze_dynamics(input_data, verbose=verbose)

    # Use analysis to create focused grid
    sr_center = analysis.suggested_spectral_radius
    lr_center = analysis.suggested_leak_rate

    focused_grid = {
        "spectral_radius": [max(0.5, sr_center - 0.15), sr_center, min(0.99, sr_center + 0.15)],
        "input_scaling": [0.05, 0.1, 0.2],
        "leak_rate": [max(0.3, lr_center - 0.2), lr_center, min(1.0, lr_center + 0.1)],
        "ridge_param": [1e-6, 1e-4, 1e-2],
    }

    # Run search
    searcher = RCHyperparameterSearch(
        input_data, target_data,
        reservoir_size=reservoir_size,
        n_cv_splits=3,
        verbose=verbose,
    )

    result = searcher.grid_search(focused_grid)

    # Refine
    refined = searcher.refine_search(result.best_params)

    return refined.best_params


def compare_configurations(
    input_data: np.ndarray,
    target_data: np.ndarray,
    configs: Dict[str, Dict[str, float]],
    reservoir_size: int = 500,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compare multiple named configurations.

    Args:
        input_data: Input sequences
        target_data: Target sequences
        configs: Dictionary mapping config names to parameter dicts
        reservoir_size: Reservoir size
        verbose: Print results

    Returns:
        Dictionary mapping config names to test RMSE scores
    """
    searcher = RCHyperparameterSearch(
        input_data, target_data,
        reservoir_size=reservoir_size,
        verbose=False,
    )

    results = {}

    if verbose:
        print(f"\n{'='*50}")
        print("CONFIGURATION COMPARISON")
        print(f"{'='*50}")

    for name, params in configs.items():
        score = searcher.evaluate_params(params)
        results[name] = score

        if verbose:
            print(f"  {name:20s}: RMSE = {score:.6f}")

    if verbose:
        best_name = min(results, key=results.get)
        print(f"\n  BEST: {best_name} (RMSE = {results[best_name]:.6f})")
        print(f"{'='*50}\n")

    return results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    """CLI for RC hyperparameter utilities."""
    import argparse

    parser = argparse.ArgumentParser(description="RC Hyperparameter Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze dynamics")
    analyze_parser.add_argument("data_path", type=Path, help="Path to .npy data file")

    # search command
    search_parser = subparsers.add_parser("search", help="Run grid search")
    search_parser.add_argument("input_path", type=Path, help="Input data .npy")
    search_parser.add_argument("target_path", type=Path, help="Target data .npy")
    search_parser.add_argument("--reservoir-size", type=int, default=500)
    search_parser.add_argument("--output", type=Path, help="Save results to JSON")
    search_parser.add_argument("--grid", choices=["coarse", "standard", "fine"], default="coarse")

    args = parser.parse_args()

    if args.command == "analyze":
        data = np.load(args.data_path)
        analyze_dynamics(data, verbose=True)

    elif args.command == "search":
        input_data = np.load(args.input_path)
        target_data = np.load(args.target_path)

        grid = {"coarse": COARSE_GRID, "standard": STANDARD_GRID, "fine": FINE_GRID}[args.grid]

        searcher = RCHyperparameterSearch(
            input_data, target_data,
            reservoir_size=args.reservoir_size,
        )

        result = searcher.grid_search(grid)

        if args.output:
            result.save(args.output)
            print(f"Saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()