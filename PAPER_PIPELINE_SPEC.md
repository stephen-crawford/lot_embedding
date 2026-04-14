# Paper Pipeline Specification (Context Lock)

Source: `paper.tex` — "Reservoir Computing on LOT-Embedded Measure-Valued Dynamical Systems"

This file is a machine-readable reference for verifying code against the paper.
Each section maps a paper equation/specification to the corresponding code and flags
discrepancies.

---

## 1. LOT Embedding (Paper §2, Eq. lot-coords-prelim, baryproj-prelim)

**Paper specification:**
- Fix reference σ ∈ P_ac(R^d) with m samples {x_i}
- For each frame t, solve discrete OT plan γ ∈ R^{m×N} minimizing Σ_{i,j} ||x_i - y_j^(t)||² γ_{ij}
  subject to row sums = 1/m, column sums = 1/N
- Barycentric projection: û_t(x_i) = (Σ_j y_j^(t) γ_{ij}) / (Σ_j γ_{ij})
- Flatten to R^{dm}: stack û_t(x_i) for i=1..m

**Code:** `src/data_utils/simulation/generate_lot_embeddings.py`
- `compute_ot_plan()` → POT `ot.emd()` or Sinkhorn
- `plan_to_barycentric_weights()` → γ_{ij} / Σ_j γ_{ij}
- `apply_barycentric_projection()` → weights @ targets
- Output: `lot_maps.npy` shape (T, R, d)

**Status: ✅ MATCHES**

---

## 2. LOT Velocity (Paper §2, Eq. lot-vel-discrete)

**Paper specification:**
- Δ_t := T_σ^{μ_{t+1}} - T_σ^{μ_t} ∈ R^{R×2}
- Simple temporal difference of LOT maps

**Code:** `generate_lot_embeddings.py`
```python
velocities = lot_maps[1:] - lot_maps[:-1]  # (T-1, R, d)
```

**Status: ✅ MATCHES**

---

## 3. RC Input/Output — Velocity Paradigm (Paper §4.2 Eq. rc-update-discrete, §5.3)

**Paper specification (Eq. rc-update-discrete):**
```
r_{t+1} = φ(A r_t + W_in Δ_t + b)
ŷ_t = W_out r_t
```
- **INPUT**: LOT velocity Δ_t
- **TARGET**: Next LOT velocity Δ_{t+1}
- The readout ŷ_t = W_out r_t predicts Δ_{t+1} from state after processing Δ_t

**Code:** `src/run_forecast/velocity_forecast.py` lines 1054-1058:
```python
vel_flat = vels.reshape(T - 1, R * d)
input_seq = vel_flat[:-1]   # v_0, ..., v_{T-3}
target_seq = vel_flat[1:]   # v_1, ..., v_{T-2}
```

**Status: ✅ MATCHES** — Code feeds velocities and predicts next velocities.

---

## 4. RC Input/Output — RAW/Positions Paradigm (Paper §5.3)

**Paper specification:**
- **RAW Forecasting**: Input = flattened particle positions vec(x(t)), d_in = 2N
- Operates on RAW particle positions, NOT LOT maps
- Ridge = 1e-4

**Code:** `src/run_forecast/positions_forecast.py` lines 304-308:
```python
map_flat = maps.reshape(T, R * d)  # LOT maps, not raw particles
input_seq = map_flat[:-1]   # T_0, ..., T_{T-2}
target_seq = map_flat[1:]   # T_1, ..., T_{T-1}
```

**Status: ⚠️ DISCREPANCY** — Code's POSITIONS method predicts LOT maps → next LOT maps
(d_in = 2R), NOT raw particle positions (d_in = 2N). The code baseline is stronger than
the paper's RAW baseline because it benefits from LOT embedding.

**Impact**: Velocity method looks relatively worse in code vs paper comparisons since
the baseline is stronger. Not a bug per se, but changes the comparison semantics.

---

## 5. Autonomous Forecasting (Paper §5.4, Eq. autonomous-vel through reconstruct)

**Paper specification:**
```
Δ̂_{T+k} = W_out · r_{T+k-1}                          (autonomous-vel)
r_{T+k}  = φ(A r_{T+k-1} + W_in Δ̂_{T+k} + b)         (autonomous-state)
û_{T+k}  = û_T + Σ_{j=1}^k Δ̂_{T+j}                    (integrate)
μ̂_{T+k}  = (û_{T+k})_# σ̂_m                             (reconstruct)
```
- Predicted velocity fed back as RC input (closed loop on velocities)
- Integration by cumulative sum of predicted velocities

**Code:** `rc_computer.py` `run_autonomous()`:
```python
for _ in range(n_steps):
    state_aug = np.append(self.state.flatten(), 1.0)
    pred = state_aug @ self.W_out       # predict Δ̂_{t+1}
    predictions.append(pred)
    self._update(pred)                   # feed Δ̂_{t+1} back as input
```
Then `velocity_forecast.py` line 1270:
```python
pred_maps[t + 1] = pred_maps[t] + pred_vel_unscaled[t]
```

**Status: ✅ MATCHES** — Velocity closed-loop + cumulative integration.

---

## 6. Reservoir Architecture (Paper §5.2)

### 6a. Spectral Radius Normalization

**Paper (Eq. in §5.2):**
```
W ← 0.7 · W / max_i |λ_i(W)|
```
Uses **spectral radius** ρ(W) = max eigenvalue magnitude.

**Code:** `rc_computer.py` line 310:
```python
use_operator_norm: bool = True  # scales by ||A||_2 (max singular value)
```

**Status: ❌ DISCREPANCY** — Paper uses spectral radius (eigenvalues), code defaults to
operator norm (singular values). For sparse random matrices the gap can be significant.
The operator norm is ALWAYS ≥ spectral radius, so `use_operator_norm=True` with
`spectral_radius=0.7` produces a LESS active reservoir than the paper intends.

### 6b. Default Spectral Radius

**Paper:** ρ(W) = 0.7
**Code (velocity_forecast.py SystemConfig):** spectral_radius = 0.8

**Status: ❌ DISCREPANCY** — Code uses 0.8 instead of 0.7.

### 6c. Leak Rate

**Paper:** α_leak = 0.7
**Code (velocity_forecast.py SystemConfig):** leak_rate = 0.8

**Status: ❌ DISCREPANCY** — Code uses 0.8 instead of 0.7.

### 6d. Ridge Regularization

**Paper:** λ_LOT = 1e-6, λ_RAW = 1e-4
**Code (velocity_forecast.py SystemConfig):** ridge_param = 1e-2

**Status: ❌ CRITICAL DISCREPANCY** — Code uses ridge 1e-2, which is 10,000x larger
than the paper's 1e-6 for LOT velocity. The GOES experiments show ridge=0.1 kills
performance; ridge=0.01 works but is still 10,000x above the paper value. The paper
explicitly justifies small ridge: "The LOT representation works in L²(σ), which yields
a smoother, more regular velocity field. Smoother targets require less regularization."

### 6e. Bias

**Paper:** "No additive bias (zero vector)"
**Code:** `bias_scale: float = 0.1` (default in ReservoirConfig)

**Status: ❌ DISCREPANCY** — Code injects random bias with scale 0.1.
However, `velocity_forecast.py` does NOT override this, so the RC gets
bias_scale=0.1 instead of the paper's 0.

### 6f. Init Scheme

**Paper:** Sparse, 10% density, normal entries
**Code (velocity_forecast.py):** `init_scheme=InitScheme.SPARSE`

**Status: ✅ MATCHES** — SPARSE uses normal entries with 10% sparsity mask.

---

## 7. Training Protocol (Paper §5.3)

### 7a. Training Window

**Paper:** T_train = 50 steps (combined washout + readout training)
**Code:** Uses `compute_warm_steps_for_velocity()` which returns cycle-length-based
warm-up (e.g., 48 for GOES diurnal, 125 for swirling cluster).

**Status: ⚠️ DISCREPANCY** — The paper uses a fixed 50-step window. The code uses
variable cycle-length-based warm-up. For GOES with cycle_length=48, the code uses
~48 warm-up steps, which is close but semantically different.

### 7b. Forecast Horizon

**Paper:** H = 48 steps
**Code:** `forecast_steps = T - 1 - warm_steps` (all remaining data)

**Status: ⚠️ DISCREPANCY** — Code uses variable horizon. For GOES GP 72h (T=145)
with warm=48, forecast is 96 steps, twice the paper's H=48.

---

## 8. Reference Measure — Gaussian (Paper §5.1)

**Paper:** gaussian_iso = N(0, I_2) (isotropic Gaussian, std=1.0, centered at origin)
**Code:** `make_gaussian_iso(R, std=0.5, center=(0, 0))`

**Status: ⚠️ MINOR DISCREPANCY** — Code uses std=0.5 vs paper std=1.0. Both centered
at origin. For GOES data normalized to [0,1]^2, the reference is far from data centroid
(0.5, 0.5), creating large baseline OT distances that reduce velocity signal-to-noise.

---

## 9. Pipeline Default — OT Assignment

**Paper:** Does not specify per_frame vs fixed (synthetic benchmarks only)
**GOES experiments (GOES_CLOUD_EXPERIMENTS.md):** per_frame + gaussian_iso is
the winning config (+32.2%). Fixed + gaussian_iso gives only +0.7%.

**Code pipeline default:** `assignment="fixed"` (realworld_experiment_pipeline.py line 317)

**Status: ❌ DISCREPANCY** — Pipeline defaults to fixed assignment, but all evidence
shows per_frame is critical for real-world GOES data.

---

## Summary of Discrepancies

| # | Parameter | Paper | Code (before) | Code (after fix) | Status |
|---|-----------|-------|---------------|------------------|--------|
| 1 | Ridge regularization | 1e-6 | 1e-2 | 1e-6 | ✅ FIXED |
| 2 | Norm scaling | spectral radius | operator norm | spectral radius | ✅ FIXED |
| 3 | Spectral radius value | 0.7 | 0.8 | 0.7 | ✅ FIXED |
| 4 | Leak rate | 0.7 | 0.8 | 0.7 | ✅ FIXED |
| 5 | Bias | 0 (none) | 0.1 (random) | 0 | ✅ FIXED |
| 6 | Init scheme | SPARSE | UNIFORM | SPARSE | ✅ FIXED |
| 7 | Pipeline assignment default | (n/a) | fixed | per_frame | ✅ FIXED |
| 8 | Positions baseline | RAW particles (2N) | LOT maps (2R) | LOT maps (2R) | Semantic (unchanged) |
| 9 | Training window | 50 fixed | Variable cycle-based | Variable cycle-based | LOW (unchanged) |
| 10 | Forecast horizon | 48 fixed | Variable (rest of data) | Variable (rest of data) | LOW (unchanged) |

**Fixes applied (this session):**
1. ✅ Ridge default → 1e-6 (velocity), 1e-4 (positions) — matches paper
2. ✅ use_operator_norm → False — matches paper's spectral radius scaling
3. ✅ bias_scale → 0.0 — matches paper's "no additive bias"
4. ✅ spectral_radius → 0.7, leak_rate → 0.7 — matches paper
5. ✅ init_scheme → SPARSE — matches paper
6. ✅ Pipeline default assignment → per_frame — critical for GOES
7. ✅ Added `consistent_quantile` particle sampling (deterministic CDF-quantile)
8. ✅ Added `temporal_smooth_maps()` for post-hoc denoising
9. ✅ Added `gaussian_data_centered` reference type for [0,1]^2 data

---

## GOES Cloud Data: Root Cause and Best Practices

### Root cause of poor predictions
The GOES particles were independently resampled each frame via `np.random.choice`
(weighted_sample), creating LOT velocities dominated by **sampling noise**:
- Velocity autocorrelation at lag 1 = **-0.45** (anti-correlated — noise signature)
- Signal-to-noise ratio ≈ 1.0 (temporal std ≈ spatial std)

### Best configuration for GOES GP 72h (empirically validated)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Reference | gaussian_iso | Rotationally symmetric |
| Reference fraction | frac25 (R=50) | Noise averaging: each ref point averages over more target particles |
| Assignment | per_frame | Factors out rotation |
| Temporal smoothing | window=5 | Reduces sampling noise, autocorr -0.44 → +0.47 |
| spectral_radius | 0.9 | Higher capacity for complex dynamics |
| leak_rate | 0.9 | Retains more memory |
| ridge_param | 1e-2 | Prevents overfitting on noisy real data |

### GOES results comparison

| Configuration | V-RMSE | Improvement |
|---|---:|---:|
| Paper defaults (sr=0.7, lk=0.7, ridge=1e-6, R=200) | 0.404 | +31.9% |
| Best HP (sr=0.9, lk=0.9, ridge=1e-2, R=200) | 0.311 | +71.3% |
| Best HP + R=50 | 0.233 | +17.4% |
| **Best HP + R=50 + smooth5** | **0.212** | **best absolute** |

### For future GOES data collection
Use `consistent_quantile` sampling method instead of `weighted_sample`:
```python
timeseries_to_particles_safe(datasets, n_particles, rng, method="consistent_quantile")
```
This produces deterministic pseudo-Lagrangian particles that track the cloud
evolution via CDF-quantile inversion, eliminating sampling noise at the source.
