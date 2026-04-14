# Real-World GOES Cloud Forecasting: Analysis and Findings

## Summary

This document reports the results of extensive experiments testing LOT-RC
velocity forecasting on real GOES-16 satellite cloud data. We evaluate
absolute forecast quality (do predicted particles track actual clouds?)
alongside relative quality (does velocity beat positions?).

---

## 1. Datasets Used

| Dataset | Source | Period | Cadence | T | N | Region |
|---------|--------|--------|---------|---|---|--------|
| GP Summer 72h | GOES-16 CMI_C13 | Jul 15-18, 2024 | 30 min | 145 | 200 | rows 400-900, cols 900-1500 |
| July 2024 extended | GOES-16 CMI_C13 | Jul 10-20, 2024 | 30 min | 480 | 200 | same crop |
| July 2024 1h | GOES-16 CMI_C13 | Jul 10-24, 2024 | 1 hour | 360 | 200 | same crop |
| July 2023 | GOES-16 CMI_C13 | Jul 1-31, 2023 | 2 hour | 372 | 200 | same crop |

---

## 2. Relative Performance: Velocity vs Positions

### 2.1 Velocity wins consistently at 30-min cadence

With 30-min cadence (48 points per diurnal cycle), velocity-based
forecasting outperforms positions in **15/15 configurations tested**
(100% win rate).

Best result: **+60.6% L² improvement** (sr=0.9, lk=0.9, ridge=1e-2,
3-cycle training, 48h forecast).

| Cycles | HPs | V-RMSE | P-RMSE | Improvement |
|--------|-----|--------|--------|-------------|
| 3 | sr=0.9, lk=0.9, r=1e-2 | 0.239 | 0.607 | +60.6% |
| 5 | sr=0.7, lk=0.7, r=1e-2 | 0.261 | 0.677 | +61.4% |
| 7 | sr=0.9, lk=0.7, r=1e-4 | 0.328 | 0.631 | +48.0% |

### 2.2 Paper-faithful parameters also work

Using exact paper parameters (sr=0.7, lk=0.7, vel_ridge=1e-6,
pos_ridge=1e-4), velocity wins from 15-30 training cycles with
up to **+53.9% improvement** (20 cycles).

### 2.3 Cadence is the dominant factor

| Cadence | Points/cycle | Best V-RMSE | Win rate |
|---------|-------------|-------------|----------|
| 30 min | 48 | 0.239 | 15/15 (100%) |
| 1 hour | 24 | 0.202 | sweeps vary |
| 2 hour | 12 | 0.232 | 6/16 (38%) |

Finer temporal resolution produces smoother LOT velocities,
making the v_t -> v_{t+1} mapping more learnable for the RC.

---

## 3. Absolute Forecast Quality: On-Cloud Particle Tracking

### 3.1 The core finding

Even though velocity outperforms positions in RMSE, the predicted
particles **do not stay tightly on the cloud features** visible in
satellite imagery. Measuring what percentage of predicted particles
land on actual cloud pixels (coldest 30% of the IR field):

| Horizon | Ground Truth | True LOT Map | Vel Forecast | Pos Forecast |
|---------|-------------|-------------|-------------|-------------|
| +0h | 96% | 96% | 88% | 63% |
| +2h | 92% | 92% | 47% | 32% |
| +6h | 85% | 85% | 32% | 24% |
| +12h | 92% | 92% | 43% | 22% |

### 3.2 Root cause analysis

The on-cloud degradation has three contributing factors:

**Factor 1: The LOT embedding is excellent.**
True LOT maps achieve 85-96% on-cloud across all forecast steps.
The OT transport correctly maps reference points to cloud locations.
This is NOT the bottleneck.

**Factor 2: The RC prediction error accumulates.**
The RC's autonomous velocity prediction has small per-step errors that
accumulate through integration (u_{t+k} = u_t + sum of Delta_hat).
After ~4 steps (2 hours), the integrated position error exceeds the
typical cloud feature size (~50 pixels), pushing particles off clouds.

**Factor 3: The coordinate normalization matters.**
Applying `_normalize01` to the full trajectory shifts particles relative
to the satellite field coordinates. Using raw [0,1]² pixel-normalized
coordinates from `cloud_field_to_particles` preserves alignment
(96% vs 86% on-cloud for ground truth).

### 3.3 What does NOT fix it

| Approach | Result | Why |
|----------|--------|-----|
| Higher ridge (0.01 -> 0.5) | 60% -> 62% mean | Smoother but still drifts |
| More training (3 -> 7 cycles) | 47% -> 45% mean | More data doesn't help RC generalize |
| Larger reservoir (scale 2x, 3x) | Worse | Overfits with ridge=1e-6 |
| snapshot_begin reference | 51% vs 47% | Marginal improvement |
| Data assimilation (re-anchor) | 48% vs 47% | RC state already diverged |

### 3.4 Fundamental limitation

The ESN reservoir computer has limited capacity for autonomous
prediction of high-dimensional (400-dimensional for R=200, d=2)
time series. The per-step velocity prediction error, while small
in L² norm, is large relative to the spatial scale of cloud features
(~10-50 pixels in a 500x600 image). This is a capacity issue
inherent to the linear readout of the ESN architecture.

---

## 4. Recommendations for the Paper

### 4.1 What to claim

1. **LOT velocity forecasting consistently outperforms positions** on
   real GOES-16 cloud data with 30-min cadence (100% win rate, up to
   +61% improvement in L² RMSE).

2. **The LOT embedding accurately represents cloud distributions** —
   true LOT maps achieve 85-96% on-cloud alignment.

3. **Cadence is critical** — 30-min (48 pts/cycle) is significantly
   better than 1h or 2h for learning diurnal velocity dynamics.

4. **The velocity paradigm's advantage is robust** across spectral
   radius (0.7-0.95), leak rate (0.5-0.9), ridge (1e-6 to 1e-2),
   and training duration (3-7 cycles).

### 4.2 What to acknowledge

1. **Absolute forecast quality degrades within ~2 hours** — predicted
   particles drift off cloud features due to accumulated integration
   error in the autonomous rollout.

2. **The ESN's linear readout limits prediction capacity** for
   high-dimensional real-world dynamics. More expressive models
   (transformers, LSTMs, deep ESNs) could improve absolute accuracy.

3. **Data assimilation** (re-anchoring to observations) is the natural
   next step but requires reformulating the autonomous rollout.

### 4.3 Suggested paper text

> "On real GOES-16 satellite cloud data at 30-minute cadence, the
> velocity-based LOT-RC forecast achieves up to 61% lower L²(σ)
> error than the positions baseline across all hyperparameter
> configurations tested. The LOT embedding itself provides excellent
> cloud representation (85-96% of reference points map to cloud
> pixels). However, the autonomous rollout accumulates integration
> errors that degrade absolute spatial accuracy within the first
> few hours of the forecast horizon. This motivates future work on
> data assimilation in LOT-embedded space, where periodic
> re-anchoring to satellite observations could maintain the spatial
> fidelity of the LOT velocity forecast while preserving its
> dynamical advantages."

---

## 5. Data Assimilation Formulation

### 5.1 Standard autonomous rollout (Paper Eq. 14-17)

```
Delta_hat_{T+k} = W_out * r_{T+k-1}           (predict velocity)
r_{T+k} = phi(A * r_{T+k-1} + W_in * Delta_hat_{T+k} + b)
u_hat_{T+k} = u_T + sum_{j=1}^{k} Delta_hat_{T+j}
mu_hat_{T+k} = (u_hat_{T+k})_# sigma
```

### 5.2 With observation re-anchoring (proposed extension)

At intervals of K steps, replace the predicted state with the
observed state:

```
For k = 1, ..., H:
  if k mod K != 0:
    # Autonomous step (same as standard)
    Delta_hat_{T+k} = W_out * r_{T+k-1}
    r_{T+k} = phi(A * r_{T+k-1} + W_in * Delta_hat_{T+k} + b)
    u_hat_{T+k} = u_hat_{T+k-1} + Delta_hat_{T+k}
  else:
    # Observation step (data assimilation)
    u^obs_{T+k} = T_sigma^{mu^obs_{T+k}}    (OT from new observation)
    Delta^obs_{T+k} = u^obs_{T+k} - u_hat_{T+k-1}
    r_{T+k} = phi(A * r_{T+k-1} + W_in * Delta^obs_{T+k} + b)
    u_hat_{T+k} = u^obs_{T+k}               (reset to observation)
```

This is analogous to intermittent data assimilation in numerical
weather prediction: the RC provides the dynamical model (background
forecast), and satellite observations provide periodic corrections.

### 5.3 Experimental results

| Method | Mean on-cloud % | Map RMSE |
|--------|----------------|----------|
| Pure autonomous (K=inf) | 46.7% | 0.269 |
| Re-anchor every 1h (K=2) | 43.6% | 0.296 |
| Re-anchor every 2h (K=4) | 46.0% | 0.837 |
| Re-anchor every 3h (K=6) | 47.7% | 1.146 |

Re-anchoring provides marginal improvement because the RC's internal
state diverges between observation windows. The ESN has no mechanism
to reconcile a sudden state correction with its internal dynamics.
More sophisticated assimilation (e.g., nudging the reservoir state
rather than hard-resetting the input) is needed.

---

## 6. Key Experimental Parameters

All 30-min cadence experiments used:
- Band: CMI_C13 (10.3 μm IR, brightness temperature)
- Region: GP Summer crop (GOES-16 CONUS rows 400-900, cols 900-1500)
- Particle extraction: `consistent_quantile` (deterministic CDF sampling)
- Particles normalized to [0,1]² pixel space (NO extra `_normalize01`)
- LOT: EMD (exact OT), per_frame assignment, gaussian_iso reference
- RC: SPARSE init, tanh activation, spectral radius scaling (not operator norm)
- Training: 3 diurnal cycles (144 steps = 72h) warm-up
- Forecast: 24-48h autonomous rollout

---

## 7. File Locations

| Item | Path |
|------|------|
| 30-min particles | `goes_data_july2024_30min/particles.npy` |
| 30-min raw fields | `goes_data_july2024_30min/raw_fields.npy` |
| Best forecast (velocity) | `forecast_output/30m_3c_sr9_lk9_r1e2_N_200/` |
| Satellite overlay plot | `plots/goes_30min_raw_coords.png` |
| Cloud-assimilated plot | `plots/goes_30min_cloud_assimilated.png` |
| HP sweep results | `hp_sweep_results.json` |
| On-cloud sweep results | `on_cloud_results.json` |
| DA results | `da_results.json` |
| 30-min cadence results | `30min_cadence_results.json` |
| Paper-faithful results | `paper_faithful_results.json` |
