# Paper Pipeline Lock File

This document specifies the exact LOT+RC forecasting pipeline described in the paper
"Reservoir Computing on LOT-Embedded Measure-Valued Dynamical Systems" and maps each
stage to its code implementation, noting any discrepancies.

---

## 1. Pipeline Overview (Paper Eq. 5, Section 4.1)

```
mu_t  -->  {y_j^(t)}  -->  T_sigma^{mu_t}  -->  Delta_t  -->  ESN  -->  Delta_hat_{t+1}  -->  u_hat_{t+H}
      sample           OT + bary proj       LOT vel diff     predict     integrate
```

Five stages converting a time series of probability measures into a forecast:

| Stage | Operation | Input | Output | Error source |
|-------|-----------|-------|--------|-------------|
| 1 | Sample | mu_t | {y_j^(t)}, j=1..N | Finite-sample |
| 2 | OT + barycentric projection | sigma, mu_t | T_sigma^{mu_t} in R^{R x d} | Compatibility eps |
| 3 | LOT velocity | T_t, T_{t+1} | Delta_t = T_{t+1} - T_t | Time discretization |
| 4 | ESN prediction | Delta_t history | Delta_hat_{t+1} | RC approximation eta |
| 5 | Integration + pushforward | Delta_hat, T_t0 | u_hat_{t+H}, mu_hat_{t+H} | Accumulation |

**Code**: `scripts/realworld_experiment_pipeline.py` orchestrates the full pipeline.
LOT embedding: `src/data_utils/simulation/generate_lot_embeddings.py`
Velocity forecast: `src/run_forecast/velocity_forecast.py` (active code starts line 624)
Position forecast: `src/run_forecast/positions_forecast.py`

---

## 2. LOT Embedding (Paper Section 2, Eq. 3-4)

### 2.1 LOT Coordinates
For reference sigma in P_ac(R^d) and target mu_t in P_2(R^d):

```
u_t := T_sigma^{mu_t}  in  H = L^2(sigma; R^d)
```

The transport map T_sigma^{mu_t} pushes sigma onto mu_t. In the discrete setting
with m reference points and N target points, this is computed via:

1. Solve discrete OT plan gamma in R^{m x N} between sigma_hat and mu_hat_t
2. Apply barycentric projection: u_t(x_i) = sum_j y_j * gamma_ij / sum_j gamma_ij

**Code**: `generate_lot_embeddings.py` function `generate_lot_embeddings()`
- Output: `lot_maps.npy` shape (T, R, d), `velocities.npy` shape (T-1, R, d)
- Also saves `reference.npy` (the sigma points)

### 2.2 LOT Velocity
```
Delta_t := u_{t+1} - u_t = T_sigma^{mu_{t+1}} - T_sigma^{mu_t}  in  R^{R x d}
```

Flattened to R^{dm} for ESN input (d=2 gives R^{2R}).

### 2.3 Reference Measure Types (Paper Section 5.1)
| Kind | Description | Code function |
|------|-------------|---------------|
| gaussian_iso | N(0, I_2) isotropic Gaussian | `make_gaussian_iso()` |
| snapshot_begin | Empirical mu at t=0 | trajectory[0] |
| snapshot_middle | Empirical mu at t=T/2 | trajectory[T//2] |
| snapshot_end | Empirical mu at t=T | trajectory[-1] |
| clean_circle | Uniform on circle r=0.8 | `make_circle()` |
| uniform_square | Uniform on [-2,2]^2 | `make_uniform_square()` |

### 2.4 Reference Discretization
Reference resolution: R = alpha_ref * N, with alpha_ref in {0.10, 0.25, 0.50, 0.75, 1.00}

### 2.5 OT Computation
**Paper** (Section 5.1): Sinkhorn with epsilon = 0.01
**Code**: Pipeline defaults to EMD (exact OT); Sinkhorn available with `--ot-method sinkhorn`

**DISCREPANCY**: Paper says Sinkhorn eps=0.01. Code pipeline defaults to EMD and uses
sinkhorn_reg=0.06 when Sinkhorn is selected. For the barycentric projection used to
compute LOT maps, EMD gives exact results while Sinkhorn is an approximation, so
EMD is actually more faithful to the theory (though slower).

---

## 3. Echo State Network (Paper Section 2, Eq. 6-7; Section 5.2)

### 3.1 State Update (Paper Eq. 9, Section 5.2)
```
r_{t+1} = (1 - alpha) * r_t + alpha * tanh(W * r_t + W_in * Delta_t + b)
```

With alpha = 0.7 (leaky integrator).

**Code**: `src/rc_computer.py` class `ReservoirComputer._update()` (line 451)

### 3.2 Architecture Parameters (Paper Section 5.2)

| Parameter | Paper Value | Code Default | Match? |
|-----------|-------------|--------------|--------|
| Spectral radius rho | 0.7 | 0.7 | YES |
| Input scaling sigma_in | 0.1 | 0.1 | YES |
| Leak rate alpha | 0.7 | 0.7 | YES |
| Bias | Zero vector | bias_scale=0.0 | YES |
| Activation | tanh (L=1) | "tanh" | YES |
| Sparsity | 10% density | sparsity=0.1 | YES |
| Init scheme | Sparse Gaussian | InitScheme.SPARSE | YES |
| Ridge lambda (LOT) | 1e-6 | 1e-6 | YES |
| Ridge lambda (RAW) | 1e-4 | 1e-4 | YES |
| Reservoir size | n_r = alpha_res * N | reservoir_scale * N | YES |
| Scaling method | Spectral radius | use_operator_norm=False | YES |

### 3.3 Readout (Paper Eq. 11, Section 5.2)
```
y_t = W_out^T [r_t; 1]
```

Affine readout with learned bias via ridge regression:
```
W_out = (X^T X + lambda I)^{-1} X^T Y
```

**Code**: `rc_computer.py` `train()` and `predict()` methods.

---

## 4. Training Protocol (Paper Section 5.3)

### 4.1 Velocity Forecasting (LOT - the proposed method)

**Paper**: Train RC on v_t -> v_{t+1} (velocity autoregressive)
- Input: LOT velocity Delta_t
- Target: next velocity Delta_{t+1}
- T_train = 50 steps (combined washout + readout)

**Code** (`velocity_forecast.py` lines 1053-1057):
```python
vel_flat = vels.reshape(T - 1, R * d)
input_seq = vel_flat[:-1]   # v_0, ..., v_{T-3}
target_seq = vel_flat[1:]   # v_1, ..., v_{T-2}
```

This correctly implements v_t -> v_{t+1} matching the paper.

**DISCREPANCY (training window)**: Paper says T_train=50 for all systems. Code uses
`compute_warm_steps_for_velocity()` which returns cycle_length-1 for known systems
(199 for geodesic, 124 for swirling_cluster). For unknown systems, falls back to 50.
This is intentional flexibility but deviates from the paper's fixed T_train=50.

### 4.2 Position Forecasting (RAW - the baseline)

**Paper**: Train RC on T_t -> T_{t+1} (direct map prediction)
- Input: LOT map T_t
- Target: next map T_{t+1}

**Code** (`positions_forecast.py` lines 306-309):
```python
map_flat = maps.reshape(T, R * d)
input_seq = map_flat[:-1]   # T_0, ..., T_{T-2}
target_seq = map_flat[1:]   # T_1, ..., T_{T-1}
```

Matches the paper.

---

## 5. Autonomous Forecasting (Paper Section 5.4)

### 5.1 Velocity Forecast Rollout

**Paper** (Eq. 14-17):
```
Delta_hat_{T+k} = W_out * r_{T+k-1}           (predict velocity)
r_{T+k} = phi(A * r_{T+k-1} + W_in * Delta_hat_{T+k} + b)  (advance reservoir)
u_hat_{T+k} = u_T + sum_{j=1}^{k} Delta_hat_{T+j}          (integrate)
mu_hat_{T+k} = (u_hat_{T+k})_# sigma_hat                    (reconstruct)
```

**Code** (`velocity_forecast.py` lines 1252-1253 + 1277-1287):
```python
_, pred_vel_flat = rc.run_autonomous(train_input, forecast_steps)
# Integration:
for t in range(forecast_steps):
    next_pos = pred_maps[t] + pred_vel_unscaled[t]
    pred_maps[t + 1] = next_pos
```

Matches the paper's integration scheme.

### 5.2 Forecast Horizon

**Paper**: H = 48 autonomous steps
**Code**: forecast_steps = T - 1 - warm_steps (varies with trajectory length)

**DISCREPANCY**: Paper fixes H=48. Code computes forecast horizon from remaining
trajectory length. This allows flexibility for different trajectory lengths.

---

## 6. Evaluation Metrics (Paper Section 6)

### 6.1 Point-wise Wasserstein Distance (Section 6.1)
```
E_W2(t) = W_2(mu_hat_t, mu_t^true)
```
Approximated via debiased Sinkhorn divergence with adaptive epsilon.

### 6.2 LOT Velocity Error (Section 6.2)
```
E_LOT(t) = ||Delta_hat_t - Delta_t^true||_{L^2(sigma)}
```

### 6.3 Temporal Alignment (Section 6.3)
Soft-DTW with adaptive gamma for phase-shift-aware evaluation.

### 6.4 Backend Validation (Section 6.4)
Two backends: LLT+L2 (fast, LOT-based) and debiased Sinkhorn (direct W2).
Agreement validated via Pearson correlation on validation window.

---

## 7. Known Discrepancies Summary

| Item | Paper | Code | Severity |
|------|-------|------|----------|
| OT method | Sinkhorn eps=0.01 | EMD default (exact) | LOW - EMD is more accurate |
| Sinkhorn reg | 0.01 | 0.06 (when used) | MEDIUM - affects LOT map quality |
| Training window | T_train=50 fixed | cycle_length-1 (system-specific) | LOW - flexibility |
| Forecast horizon | H=48 fixed | T-1-warm_steps (flexible) | LOW - flexibility |
| paradigm config field | v_t -> v_{t+1} | Defaults to "position_to_velocity" (misleading) | MEDIUM - misleading name |

## 8. Reconstruction: predicted_measures_X (Critical)

The paper (Section 4.3, Eq. 17) specifies reconstruction via pushforward:

```
mu_hat_{t+k} = (u_hat_{t+k})_# sigma_hat_m
```

In the code, `predicted_measures_X.npy` stores the LOT map values
`u_hat_{t+k}(x_i)` for each reference point `x_i` in sigma. Since the
pushforward of sigma through the map u gives exactly the points `u(x_i)`,
**the LOT map values ARE the reconstructed particle positions**.

This means:
- `predicted_measures_X[t]` has shape `(R, d)` — R particles, not N
- These R particles represent the predicted empirical measure
- They should be overlaid on satellite imagery for visualization
- They are in whatever coordinate space the original particles were extracted in

**Visualization rule**: To overlay on satellite images, particles must be in
the SAME coordinate space as the image. For GOES:
- If particles extracted in pixel space -> plot in pixel space with `imshow(origin='upper')`
- If particles extracted in lon/lat -> must convert back to pixel via inverse projection
- NEVER use `imshow(extent=[lon_min, lon_max, ...])` — geostationary projection is non-linear

## 9. File Map

| Paper Section | Code File | Function/Class |
|--------------|-----------|----------------|
| Section 2 (ESN) | `src/rc_computer.py` | `ReservoirComputer` |
| Section 4.1 (Pipeline) | `scripts/realworld_experiment_pipeline.py` | `run_pipeline()` |
| Section 5.1 (References) | `src/data_utils/simulation/generate_lot_embeddings.py` | `make_*()` functions |
| Section 5.1 (LOT maps) | `src/data_utils/simulation/generate_lot_embeddings.py` | `generate_lot_embeddings()` |
| Section 5.3 (Velocity) | `src/run_forecast/velocity_forecast.py` | `run_single()` |
| Section 5.3 (Positions) | `src/run_forecast/positions_forecast.py` | `run_single()` |
| Section 5.3 (Warmup) | `src/run_forecast/warmup_utils.py` | `compute_warm_steps()` |
| Section 6 (Metrics) | `src/lotrc_eval/block_f_metrics.py` | Evaluation blocks |
| GOES data | `src/data_utils/goes_cloud_data.py` | `fetch_goes_particles_bundle()` |
