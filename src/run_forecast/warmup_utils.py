"""
Warm-up computation utilities for LOT+RC forecasting.

This module provides correct warm-up step calculation that ensures
the reservoir sees at least one complete oscillation before autonomous rollout.

Key insight: Training on only half a cycle (e.g., expansion but not contraction)
means the reservoir never sees the reversal dynamics, leading to poor predictions
when it encounters contraction for the first time during autonomous rollout.

Features:
    - Automatic cycle detection from velocity autocorrelation
    - Cycle-aligned warm-up computation
    - Adaptive hyperparameter selection based on cycle characteristics
    - Optional cyclicity validation
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, NamedTuple
import json
import numpy as np


# ═══════════════════════════════════════════════════════════════
# SYSTEM DEFAULTS
# ═══════════════════════════════════════════════════════════════

# Default trajectory parameters established by generate_trajectories.py
SYSTEM_DEFAULTS = {
    # ═══════════════════════════════════════════════════════════════
    # SYNTHETIC SYSTEMS
    # ═══════════════════════════════════════════════════════════════
    "geodesic_transport": {
        "n_steps": 400,
        "n_cycles": 2,
        "cycle_length": 200,  # One full circle→triangle→circle oscillation
    },
    "swirling_cluster": {
        "n_steps": 250,
        "n_cycles": 2,
        "cycle_length": 125,  # One full r=1→max→1 breathing oscillation
    },
    "lorenz_gmm": {
        "n_steps": 400,
        "n_cycles": 2,
        "cycle_length": 50,  # Lorenz is chaotic, use shorter warm-up
    },
    # ═══════════════════════════════════════════════════════════════
    # REAL-WORLD SYSTEMS
    # ═══════════════════════════════════════════════════════════════
    "era5_wind": {
        "n_steps": 240,
        "n_cycles": 12,
        "cycle_length": 20,  # ~5 days at 6-hourly (synoptic patterns) - surface wind
    },
    "era5_500hPa": {
        "n_steps": 100,
        "n_cycles": 5,
        "cycle_length": 20,  # ~5 days at 6-hourly (Rossby wave period) - jet stream
    },
    "hycom_gulf": {
        "n_steps": 80,
        "n_cycles": 4,
        "cycle_length": 20,  # ~10 days at 3-hourly (mesoscale eddy rotation)
    },
    "oscar_gulf": {
        "n_steps": 80,
        "n_cycles": 8,
        "cycle_length": 10,  # ~10 days at daily (eddy rotation period)
    },
    "sst_ocean": {
        "n_steps": 365,
        "n_cycles": 1,
        "cycle_length": 365,  # Annual cycle for SST
    },
}


# ═══════════════════════════════════════════════════════════════
# CYCLE DETECTION RESULTS
# ═══════════════════════════════════════════════════════════════

class CycleDetectionResult(NamedTuple):
    """Result of automatic cycle detection."""
    cycle_length: int
    confidence: float  # 0-1, higher = more confident
    autocorr_peak: float  # Autocorrelation at detected cycle
    method: str  # "autocorr", "fft", or "fallback"
    is_cyclic: bool  # Whether system appears cyclic




# ═══════════════════════════════════════════════════════════════
# 1. AUTOMATIC CYCLE DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_cycle_from_maps(
    maps: np.ndarray,
    min_cycle: int = 10,
    max_cycle: Optional[int] = None,
) -> CycleDetectionResult:
    """
    Detect cycle length from LOT maps by finding when positions return to start.

    For cyclic systems, positions return to similar values after one cycle.
    This is more reliable than velocity autocorrelation for systems with
    reversal dynamics (like geodesic transport).

    Args:
        maps: LOT maps array of shape (T, R, d)
        min_cycle: Minimum cycle length to consider
        max_cycle: Maximum cycle length (default: T//2)

    Returns:
        CycleDetectionResult with detected cycle length and confidence
    """
    T, R, d = maps.shape
    maps_flat = maps.reshape(T, R * d)

    if max_cycle is None:
        max_cycle = T // 2
    max_cycle = min(max_cycle, T - 1)

    # Compute distance from initial position at each timestep
    initial = maps_flat[0]
    distances = np.array([np.linalg.norm(maps_flat[t] - initial) for t in range(T)])

    # Normalize distances
    max_dist = np.max(distances)
    if max_dist > 0:
        distances = distances / max_dist

    # Find local minima in distance (returns to start)
    # A cycle is when we return close to the initial position
    minima = []
    for t in range(min_cycle, min(max_cycle + 1, T - 1)):
        if distances[t] < distances[t-1] and distances[t] < distances[t+1]:
            minima.append((t, distances[t]))

    if minima:
        # Take the first significant minimum (first return to start)
        # Sort by distance (closest to start)
        minima.sort(key=lambda x: x[1])
        best_t, best_dist = minima[0]

        # Confidence: how close did we get to the start?
        # dist=0 means perfect return, dist=1 means no return
        confidence = 1.0 - best_dist
        is_cyclic = best_dist < 0.3  # Return to within 30% of max distance

        return CycleDetectionResult(
            cycle_length=best_t,
            confidence=float(confidence),
            autocorr_peak=float(1.0 - best_dist),
            method="map_return",
            is_cyclic=is_cyclic,
        )

    # Fallback: use FFT on distance signal
    return _detect_cycle_fft_distance(distances, min_cycle, max_cycle)


def _detect_cycle_fft_distance(
    distances: np.ndarray,
    min_cycle: int,
    max_cycle: int,
) -> CycleDetectionResult:
    """Detect cycle from FFT of distance-from-start signal."""
    T = len(distances)

    # FFT of distance signal (should show periodicity)
    fft = np.fft.rfft(distances - np.mean(distances))
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(T)

    # Find peak in valid frequency range
    min_freq = 1.0 / max_cycle if max_cycle > 0 else 0
    max_freq = 1.0 / min_cycle if min_cycle > 0 else 0.5

    valid_mask = (freqs >= min_freq) & (freqs <= max_freq) & (freqs > 0)
    if not np.any(valid_mask):
        return CycleDetectionResult(
            cycle_length=min_cycle,
            confidence=0.0,
            autocorr_peak=0.0,
            method="fallback",
            is_cyclic=False,
        )

    valid_power = power.copy()
    valid_power[~valid_mask] = 0

    peak_idx = np.argmax(valid_power)
    peak_freq = freqs[peak_idx]

    if peak_freq == 0:
        cycle_length = max_cycle
    else:
        cycle_length = int(round(1.0 / peak_freq))

    cycle_length = max(min_cycle, min(max_cycle, cycle_length))

    # Confidence from peak prominence
    total_power = np.sum(power[valid_mask])
    confidence = float(power[peak_idx] / total_power) if total_power > 0 else 0
    confidence = min(1.0, confidence * 2)

    return CycleDetectionResult(
        cycle_length=cycle_length,
        confidence=confidence,
        autocorr_peak=confidence,
        method="fft_distance",
        is_cyclic=confidence > 0.3,
    )


def detect_cycle_from_velocities(
    velocities: np.ndarray,
    min_cycle: int = 10,
    max_cycle: Optional[int] = None,
) -> CycleDetectionResult:
    """
    Detect cycle length from velocity time series.

    Note: For systems with reversal dynamics (like geodesic transport),
    use detect_cycle_from_maps() instead, which is more reliable.

    Args:
        velocities: Velocity array of shape (T-1, R, d) or (T-1, R*d)
        min_cycle: Minimum cycle length to consider
        max_cycle: Maximum cycle length (default: T//2)

    Returns:
        CycleDetectionResult with detected cycle length and confidence
    """
    # Flatten to (T-1, features)
    if velocities.ndim == 3:
        T_minus_1, R, d = velocities.shape
        vel_flat = velocities.reshape(T_minus_1, R * d)
    else:
        vel_flat = velocities
        T_minus_1 = vel_flat.shape[0]

    if max_cycle is None:
        max_cycle = T_minus_1 // 2

    max_cycle = min(max_cycle, T_minus_1 - 1)

    return _detect_cycle_autocorr(vel_flat, min_cycle, max_cycle)


def _detect_cycle_autocorr(
    vel_flat: np.ndarray,
    min_cycle: int,
    max_cycle: int,
) -> CycleDetectionResult:
    """
    Detect cycle using autocorrelation peak finding.

    Uses multiple signals:
    1. Standard autocorrelation (for systems where velocity repeats)
    2. Absolute autocorrelation (for reversal dynamics like geodesic)
    3. Velocity magnitude autocorrelation (for oscillating systems)
    """
    T = vel_flat.shape[0]

    # Compute different velocity representations
    vel_mean = np.mean(vel_flat, axis=1)  # Mean velocity per timestep
    vel_mag = np.linalg.norm(vel_flat, axis=1)  # Velocity magnitude per timestep

    # Sample features for per-particle autocorrelation
    n_features = vel_flat.shape[1]
    n_sample = min(100, n_features)
    np.random.seed(42)  # Reproducible
    sample_idx = np.random.choice(n_features, n_sample, replace=False)

    # Compute autocorrelation for each lag
    autocorrs_standard = []
    autocorrs_abs = []
    autocorrs_mag = []

    for lag in range(min_cycle, max_cycle + 1):
        if T <= lag:
            continue

        # Standard autocorrelation (velocity repeats)
        corrs_std = []
        corrs_abs_list = []
        for idx in sample_idx:
            feat = vel_flat[:, idx]
            c = np.corrcoef(feat[:-lag], feat[lag:])[0, 1]
            if not np.isnan(c):
                corrs_std.append(c)
                corrs_abs_list.append(abs(c))

        corr_std = np.mean(corrs_std) if corrs_std else 0
        corr_abs = np.mean(corrs_abs_list) if corrs_abs_list else 0

        # Magnitude autocorrelation (speed pattern repeats)
        if len(vel_mag) > lag:
            corr_mag = np.corrcoef(vel_mag[:-lag], vel_mag[lag:])[0, 1]
            if np.isnan(corr_mag):
                corr_mag = 0
        else:
            corr_mag = 0

        autocorrs_standard.append((lag, corr_std))
        autocorrs_abs.append((lag, corr_abs))
        autocorrs_mag.append((lag, corr_mag))

    if not autocorrs_standard:
        return CycleDetectionResult(
            cycle_length=min_cycle,
            confidence=0.0,
            autocorr_peak=0.0,
            method="fallback",
            is_cyclic=False,
        )

    # Convert to arrays
    lags = np.array([a[0] for a in autocorrs_standard])
    corrs_std = np.nan_to_num(np.array([a[1] for a in autocorrs_standard]), nan=0.0)
    corrs_abs = np.nan_to_num(np.array([a[1] for a in autocorrs_abs]), nan=0.0)
    corrs_mag = np.nan_to_num(np.array([a[1] for a in autocorrs_mag]), nan=0.0)

    # Combined signal: weight different methods
    # - Standard: good for simple repeating patterns
    # - Absolute: good for reversal dynamics (geodesic)
    # - Magnitude: good for breathing/oscillating patterns (swirling)
    combined = 0.3 * corrs_std + 0.4 * corrs_abs + 0.3 * corrs_mag

    # Find peaks in combined signal
    def find_peaks(arr, threshold=0.05):
        peaks = []
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > threshold:
                peaks.append((lags[i], arr[i]))
        return peaks

    peaks = find_peaks(combined, threshold=0.03)

    if not peaks:
        # No peaks - use maximum
        best_idx = np.argmax(combined)
        best_lag = lags[best_idx]
        best_corr = combined[best_idx]
        is_cyclic = best_corr > 0.15

        return CycleDetectionResult(
            cycle_length=int(best_lag),
            confidence=max(0, min(1, best_corr * 2)),
            autocorr_peak=float(best_corr),
            method="autocorr_max",
            is_cyclic=is_cyclic,
        )

    # Find fundamental frequency (earliest significant peak)
    peaks.sort(key=lambda x: x[0])  # Sort by lag
    max_peak_val = max(p[1] for p in peaks)

    # Take earliest peak that's at least 50% of max
    best_lag, best_corr = peaks[0]
    for lag, corr in peaks:
        if corr >= 0.5 * max_peak_val:
            best_lag, best_corr = lag, corr
            break

    # Confidence based on peak strength
    confidence = min(1.0, best_corr * 3)  # Scale up
    is_cyclic = best_corr > 0.1

    return CycleDetectionResult(
        cycle_length=int(best_lag),
        confidence=float(confidence),
        autocorr_peak=float(best_corr),
        method="autocorr",
        is_cyclic=is_cyclic,
    )


def _detect_cycle_fft(
    vel_flat: np.ndarray,
    min_cycle: int,
    max_cycle: int,
) -> CycleDetectionResult:
    """Detect cycle using FFT dominant frequency."""
    T = vel_flat.shape[0]

    # Use mean velocity for FFT
    vel_mean = np.mean(vel_flat, axis=1)

    # Remove DC component
    vel_centered = vel_mean - np.mean(vel_mean)

    # FFT
    fft = np.fft.rfft(vel_centered)
    freqs = np.fft.rfftfreq(T)
    power = np.abs(fft) ** 2

    # Convert frequency bounds to indices
    # freq = 1/cycle, so min_freq = 1/max_cycle, max_freq = 1/min_cycle
    min_freq = 1.0 / max_cycle if max_cycle > 0 else 0
    max_freq = 1.0 / min_cycle if min_cycle > 0 else 0.5

    # Find dominant frequency in range
    valid_mask = (freqs >= min_freq) & (freqs <= max_freq) & (freqs > 0)
    if not np.any(valid_mask):
        return CycleDetectionResult(
            cycle_length=min_cycle,
            confidence=0.0,
            autocorr_peak=0.0,
            method="fallback",
            is_cyclic=False,
        )

    valid_power = power.copy()
    valid_power[~valid_mask] = 0

    peak_idx = np.argmax(valid_power)
    peak_freq = freqs[peak_idx]
    peak_power = power[peak_idx]

    if peak_freq == 0:
        cycle_length = max_cycle
    else:
        cycle_length = int(round(1.0 / peak_freq))

    # Clamp to valid range
    cycle_length = max(min_cycle, min(max_cycle, cycle_length))

    # Confidence based on peak prominence
    total_power = np.sum(power[valid_mask])
    confidence = peak_power / total_power if total_power > 0 else 0
    confidence = min(1.0, confidence * 2)  # Scale up

    is_cyclic = confidence > 0.2

    return CycleDetectionResult(
        cycle_length=cycle_length,
        confidence=float(confidence),
        autocorr_peak=float(confidence),  # Use same value for consistency
        method="fft",
        is_cyclic=is_cyclic,
    )


# ═══════════════════════════════════════════════════════════════
# 2. CYCLE-ALIGNED WARM-UP
# ═══════════════════════════════════════════════════════════════

def compute_cycle_aligned_warm_steps(
    T: int,
    cycle_length: int,
    n_cycles: int = 1,
    min_forecast_steps: int = 10,
    method: str = "velocity",
) -> Tuple[int, int]:
    """
    Compute warm-up steps aligned to exact cycle boundaries.

    Ensures training covers exactly n_cycles complete cycles, which is
    critical for the reservoir to learn the full dynamics.

    Args:
        T: Total trajectory length (number of maps)
        cycle_length: Detected or known cycle length
        n_cycles: Number of complete cycles to use for training
        min_forecast_steps: Minimum steps to reserve for forecasting
        method: "velocity" or "positions"

    Returns:
        (warm_steps, actual_cycles): Aligned warm-up steps and cycles used

    Example:
        >>> warm_steps, n_cyc = compute_cycle_aligned_warm_steps(400, 200, n_cycles=1)
        >>> print(f"Warm steps: {warm_steps}, cycles: {n_cyc}")
        Warm steps: 199, cycles: 1
    """
    if method == "velocity":
        max_steps = T - 2  # v[0]→v[1], ..., v[T-3]→v[T-2]
    else:
        max_steps = T - 1  # map[0]→map[1], ..., map[T-2]→map[T-1]

    # Compute warm_steps for requested cycles
    # For velocity: warm_steps covers transitions, so n_cycles * cycle_length - 1
    target_warm = n_cycles * cycle_length - 1

    # Check if we have enough data
    available_for_forecast = max_steps - target_warm

    if available_for_forecast < min_forecast_steps:
        # Reduce cycles until we have enough forecast room
        actual_cycles = n_cycles
        while actual_cycles > 0:
            target_warm = actual_cycles * cycle_length - 1
            available_for_forecast = max_steps - target_warm
            if available_for_forecast >= min_forecast_steps:
                break
            actual_cycles -= 1

        if actual_cycles == 0:
            # Can't even fit one cycle - use half the data
            target_warm = max_steps // 2
            actual_cycles = target_warm / cycle_length
    else:
        actual_cycles = n_cycles

    # Ensure bounds
    warm_steps = max(1, min(target_warm, max_steps - min_forecast_steps))

    return warm_steps, actual_cycles


# ═══════════════════════════════════════════════════════════════
# 3. CYCLICITY VALIDATION (OPTIONAL)
# ═══════════════════════════════════════════════════════════════

def validate_cyclicity(
    velocities: np.ndarray,
    cycle_length: int,
    confidence_threshold: float = 0.3,
    autocorr_threshold: float = 0.2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate that a system appears cyclic and suitable for velocity forecasting.

    Performs several checks:
        1. Autocorrelation at cycle length is positive and significant
        2. Velocity pattern repeats approximately every cycle
        3. No strong drift that would break stationarity

    Args:
        velocities: Velocity array (T-1, R, d) or (T-1, R*d)
        cycle_length: Proposed cycle length
        confidence_threshold: Minimum confidence for cycle detection
        autocorr_threshold: Minimum autocorrelation at cycle length
        verbose: Print warnings if validation fails

    Returns:
        Dictionary with validation results:
            - is_valid: bool
            - is_cyclic: bool
            - warnings: list of warning messages
            - metrics: dict of computed metrics
    """
    # Detect cycle to compare
    detection = detect_cycle_from_velocities(velocities, min_cycle=5)

    warnings = []
    metrics = {
        "detected_cycle": detection.cycle_length,
        "proposed_cycle": cycle_length,
        "detection_confidence": detection.confidence,
        "autocorr_peak": detection.autocorr_peak,
    }

    # Check 1: Is the system cyclic at all?
    if not detection.is_cyclic:
        warnings.append(
            f"System does not appear cyclic (autocorr={detection.autocorr_peak:.3f} < {autocorr_threshold}). "
            f"Consider using POSITIONS forecasting instead."
        )

    # Check 2: Does detected cycle match proposed?
    cycle_mismatch = abs(detection.cycle_length - cycle_length) / max(cycle_length, 1)
    if cycle_mismatch > 0.2 and detection.confidence > 0.5:
        warnings.append(
            f"Detected cycle ({detection.cycle_length}) differs from proposed ({cycle_length}) by {cycle_mismatch*100:.0f}%. "
            f"Consider using detected value."
        )
    metrics["cycle_mismatch"] = cycle_mismatch

    # Check 3: Low confidence
    if detection.confidence < confidence_threshold:
        warnings.append(
            f"Low confidence in cycle detection ({detection.confidence:.2f} < {confidence_threshold}). "
            f"Results may be unreliable."
        )

    # Check 4: Drift detection
    if velocities.ndim == 3:
        vel_flat = velocities.reshape(velocities.shape[0], -1)
    else:
        vel_flat = velocities

    mean_vel_per_step = np.mean(vel_flat, axis=1)
    drift = np.mean(mean_vel_per_step)
    drift_std = np.std(mean_vel_per_step)

    if abs(drift) > 0.1 * drift_std and abs(drift) > 1e-4:
        warnings.append(
            f"Detected net drift (mean={drift:.6f}). "
            f"Velocity forecasting assumes zero-mean cyclic dynamics."
        )
    metrics["drift"] = float(drift)
    metrics["drift_std"] = float(drift_std)

    # Determine validity
    is_cyclic = detection.is_cyclic and detection.confidence >= confidence_threshold
    is_valid = is_cyclic and cycle_mismatch <= 0.3

    if verbose and warnings:
        for w in warnings:
            print(f"[WARN] {w}")

    return {
        "is_valid": is_valid,
        "is_cyclic": is_cyclic,
        "warnings": warnings,
        "metrics": metrics,
        "detection": detection,
    }


# ═══════════════════════════════════════════════════════════════
# COMBINED UTILITY
# ═══════════════════════════════════════════════════════════════

def analyze_and_configure(
    velocities: np.ndarray,
    system_name: Optional[str] = None,
    T: Optional[int] = None,
    n_cycles: int = 1,
    validate: bool = True,
    verbose: bool = True,
    prefer_system_defaults: bool = True,
) -> Dict[str, Any]:
    """
    Analyze velocities and configure cycle-related settings.

    Combines cycle detection, warm-up computation, and optional validation.
    For hyperparameter tuning, use rc_utils.suggest_hyperparameters() separately.

    Args:
        velocities: Velocity array (T-1, R, d)
        system_name: Optional system name for defaults lookup
        T: Total maps (if not provided, inferred as velocities.shape[0] + 1)
        n_cycles: Number of cycles to train on
        validate: Whether to run cyclicity validation
        verbose: Print diagnostics
        prefer_system_defaults: If True, use system defaults when available

    Returns:
        Configuration dictionary with cycle_length, warm_steps, validation, etc.
    """
    if T is None:
        T = velocities.shape[0] + 1

    # Always run detection for diagnostics
    detection = detect_cycle_from_velocities(velocities)

    # 1. Determine cycle length
    if system_name and system_name in SYSTEM_DEFAULTS and prefer_system_defaults:
        # Known system - use defaults (they're based on how trajectories were generated)
        default_cycle = SYSTEM_DEFAULTS[system_name]["cycle_length"]
        cycle_length = default_cycle
        cycle_source = "system_default"

        if verbose and detection.confidence > 0.5:
            detected_diff = abs(detection.cycle_length - default_cycle) / default_cycle
            if detected_diff > 0.3:
                print(f"[INFO] Detection suggests cycle={detection.cycle_length}, "
                      f"using system default={default_cycle}")
    elif detection.confidence > 0.3 and detection.is_cyclic:
        # Unknown system but confident detection
        cycle_length = detection.cycle_length
        cycle_source = "detected"
    else:
        # Low confidence - use detection anyway but warn
        cycle_length = detection.cycle_length
        cycle_source = "detected_low_confidence"
        if verbose:
            print(f"[WARN] Low confidence cycle detection (conf={detection.confidence:.2f})")

    # 2. Compute warm-up
    warm_steps, actual_cycles = compute_cycle_aligned_warm_steps(
        T=T,
        cycle_length=cycle_length,
        n_cycles=n_cycles,
        method="velocity",
    )

    # 3. Optional validation
    if validate:
        validation = validate_cyclicity(
            velocities=velocities,
            cycle_length=cycle_length,
            verbose=verbose,
        )
    else:
        validation = None

    config = {
        "cycle_length": cycle_length,
        "cycle_source": cycle_source,
        "cycle_detection": detection._asdict(),
        "warm_steps": warm_steps,
        "actual_cycles": actual_cycles,
        "forecast_steps": T - 2 - warm_steps,
        "validation": validation,
    }

    if verbose:
        print(f"[CONFIG] cycle={cycle_length} ({cycle_source}), "
              f"warm={warm_steps}, forecast={config['forecast_steps']}")

    return config


def get_default_cycle_length(system_name: str) -> int:
    """
    Get the default cycle length for a system.
    
    Returns the number of frames in one complete oscillation based on
    default trajectory generation parameters.
    
    Args:
        system_name: "geodesic_transport" or "swirling_cluster"
        
    Returns:
        cycle_length: Default frames per oscillation
        
    Raises:
        ValueError: If system_name is not recognized
    """
    if system_name not in SYSTEM_DEFAULTS:
        raise ValueError(
            f"Unknown system: {system_name}. "
            f"Expected one of: {list(SYSTEM_DEFAULTS.keys())}"
        )
    return SYSTEM_DEFAULTS[system_name]["cycle_length"]


def detect_cycle_length_from_metadata(lot_dir: Path) -> Optional[int]:
    """
    Attempt to detect cycle length from LOT metadata.
    
    Looks for metadata.json in the LOT directory and extracts trajectory info.
    
    Args:
        lot_dir: Path to LOT embedding directory
        
    Returns:
        cycle_length if detected, None otherwise
    """
    meta_path = lot_dir / "metadata.json"
    if not meta_path.exists():
        return None
    
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        
        # Check if trajectory source has info
        traj_source = meta.get("trajectory_source", "")
        
        # Infer from system name and standard parameters
        system = meta.get("system", "")
        if system in SYSTEM_DEFAULTS:
            return SYSTEM_DEFAULTS[system]["cycle_length"]
        
        # Check for explicit cycle info (future-proofing)
        if "cycle_length" in meta:
            return meta["cycle_length"]
        if "n_steps" in meta and "n_cycles" in meta:
            return meta["n_steps"] // meta["n_cycles"]
        
        return None
        
    except (json.JSONDecodeError, KeyError):
        return None


def compute_warm_steps(
    max_steps: int,
    system_name: str,
    cycle_length: Optional[int] = None,
    align_to_boundary: bool = True,
    fallback_steps: int = 50,
    lot_dir: Optional[Path] = None,
    verbose: bool = False,
) -> int:
    """
    Compute warm-up steps for reservoir training.
    
    Ensures the reservoir sees at least one complete oscillation before
    autonomous rollout begins. This is critical for learning both phases
    of the dynamics (e.g., expansion AND contraction).
    
    Priority for determining cycle_length:
        1. Explicit cycle_length argument
        2. Detected from lot_dir metadata
        3. System defaults based on system_name
        4. Fallback to fallback_steps (with warning)
    
    Args:
        max_steps: Maximum available training steps (T-1 for velocities, T-1 for positions)
        system_name: "geodesic_transport" or "swirling_cluster"
        cycle_length: Explicit cycle length override
        align_to_boundary: If True, align to cycle boundary; if False, use cycle_length directly
        fallback_steps: Fallback if cycle length cannot be determined
        lot_dir: Optional path to LOT directory for metadata detection
        verbose: Print diagnostic info
        
    Returns:
        warm_steps: Number of training steps (ensures full cycle coverage)
        
    Example:
        >>> compute_warm_steps(399, "geodesic_transport")
        200  # One full cycle for geodesic transport
        
        >>> compute_warm_steps(249, "swirling_cluster")  
        125  # One full cycle for swirling cluster
        
    Note on velocity vs position forecasting:
        - For velocity forecasting: max_steps = T - 2 (velocity pairs)
        - For position forecasting: max_steps = T - 1 (position pairs)
        
        The warm_steps returned is the number of training INPUT steps.
        This corresponds to:
        - Velocity: seeing warm_steps velocity vectors
        - Position: seeing warm_steps position vectors
    """
    # Priority 1: Explicit override
    if cycle_length is not None:
        effective_cycle = cycle_length
        source = "explicit"
    
    # Priority 2: Detect from metadata
    elif lot_dir is not None:
        detected = detect_cycle_length_from_metadata(lot_dir)
        if detected is not None:
            effective_cycle = detected
            source = "metadata"
        else:
            effective_cycle = None
            source = None
    else:
        effective_cycle = None
        source = None
    
    # Priority 3: System defaults
    if effective_cycle is None and system_name in SYSTEM_DEFAULTS:
        effective_cycle = SYSTEM_DEFAULTS[system_name]["cycle_length"]
        source = "system_default"
    
    # Priority 4: Fallback
    if effective_cycle is None:
        if verbose:
            print(f"[WARN] Could not determine cycle length for {system_name}, "
                  f"using fallback={fallback_steps}")
        return min(fallback_steps, max_steps - 1)
    
    # Compute warm_steps
    if align_to_boundary:
        # Train on exactly one full cycle
        # For maps: cycle_length maps → cycle_length - 1 transitions
        # So warm_steps = cycle_length - 1 for velocity, cycle_length - 1 for position
        warm_steps = effective_cycle - 1
    else:
        # Use cycle_length directly (may not align to cycle boundary)
        warm_steps = effective_cycle
    
    # Ensure within bounds
    warm_steps = max(1, min(warm_steps, max_steps - 1))
    
    if verbose:
        print(f"[WARM] {system_name}: cycle_length={effective_cycle} ({source}) "
              f"→ warm_steps={warm_steps}")
    
    return warm_steps


def compute_warm_steps_for_velocity(
    T: int,
    system_name: str,
    cycle_length: Optional[int] = None,
    **kwargs
) -> int:
    """
    Compute warm-up steps for VELOCITY forecasting.
    
    In velocity forecasting:
        - We have T maps → T-1 velocities
        - We predict vel[t+1] from vel[t] → T-2 training pairs
        - max_steps = T - 2
        
    After warm_steps training inputs, autonomous rollout predicts the remaining
    velocities, which are integrated to reconstruct maps.
    
    Args:
        T: Number of maps in trajectory
        system_name: "geodesic_transport" or "swirling_cluster"
        cycle_length: Optional explicit cycle length
        **kwargs: Additional args passed to compute_warm_steps
        
    Returns:
        warm_steps: Training steps for velocity pairs
    """
    max_steps = T - 2  # vel[0]→vel[1], ..., vel[T-3]→vel[T-2]
    return compute_warm_steps(max_steps, system_name, cycle_length, **kwargs)


def compute_warm_steps_for_positions(
    T: int,
    system_name: str,
    cycle_length: Optional[int] = None,
    **kwargs
) -> int:
    """
    Compute warm-up steps for POSITIONS forecasting.
    
    In positions forecasting:
        - We have T maps
        - We predict map[t+1] from map[t] → T-1 training pairs  
        - max_steps = T - 1
        
    After warm_steps training inputs, autonomous rollout predicts the remaining
    maps directly.
    
    Args:
        T: Number of maps in trajectory
        system_name: "geodesic_transport" or "swirling_cluster"
        cycle_length: Optional explicit cycle length
        **kwargs: Additional args passed to compute_warm_steps
        
    Returns:
        warm_steps: Training steps for position pairs
    """
    max_steps = T - 1  # map[0]→map[1], ..., map[T-2]→map[T-1]
    return compute_warm_steps(max_steps, system_name, cycle_length, **kwargs)


# ═══════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_warm_steps(
    warm_steps: int,
    system_name: str,
    T: int,
    method: str = "velocity",
) -> Dict[str, Any]:
    """
    Validate warm-up configuration and return diagnostic info.
    
    Checks:
        1. warm_steps covers at least one full cycle
        2. Sufficient forecast horizon remains
        3. Alignment to cycle boundary
        
    Args:
        warm_steps: Proposed warm-up steps
        system_name: System name
        T: Total trajectory length (maps)
        method: "velocity" or "positions"
        
    Returns:
        Dictionary with validation results and diagnostics
    """
    if system_name not in SYSTEM_DEFAULTS:
        return {
            "valid": False,
            "error": f"Unknown system: {system_name}",
        }
    
    defaults = SYSTEM_DEFAULTS[system_name]
    cycle_length = defaults["cycle_length"]
    
    # Compute expected values
    if method == "velocity":
        max_steps = T - 2
        expected_warm = cycle_length - 1
    else:  # positions
        max_steps = T - 1
        expected_warm = cycle_length - 1
    
    forecast_steps = max_steps - warm_steps
    cycles_trained = (warm_steps + 1) / cycle_length  # +1 because warm_steps = cycle - 1
    
    # Validation checks
    issues = []
    
    if warm_steps < expected_warm:
        issues.append(
            f"warm_steps={warm_steps} < recommended={expected_warm} "
            f"(less than one full cycle)"
        )
    
    if forecast_steps < cycle_length // 2:
        issues.append(
            f"forecast_steps={forecast_steps} < {cycle_length // 2} "
            f"(very short forecast horizon)"
        )
    
    if (warm_steps + 1) % cycle_length != 0:
        issues.append(
            f"warm_steps={warm_steps} not aligned to cycle boundary "
            f"(cycle_length={cycle_length})"
        )
    
    return {
        "valid": len(issues) == 0,
        "warm_steps": warm_steps,
        "expected_warm_steps": expected_warm,
        "forecast_steps": forecast_steps,
        "cycle_length": cycle_length,
        "cycles_trained": cycles_trained,
        "issues": issues,
        "system": system_name,
        "method": method,
        "T": T,
    }