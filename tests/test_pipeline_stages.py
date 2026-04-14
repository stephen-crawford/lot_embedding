"""
Tests for each stage of the LOT-RC forecasting pipeline.

Verifies that each stage:
  (a) produces correct shapes and types
  (b) satisfies mathematical invariants from the paper
  (c) feeds correctly into the next stage

Paper: "Reservoir Computing on LOT-Embedded Measure-Valued Dynamical Systems"

Run:  pytest tests/test_pipeline_stages.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# Paper parameters (§5.2 – §5.4)
PAPER_SR = 0.7
PAPER_INPUT_SCALE = 0.1
PAPER_LEAK = 0.7
PAPER_RIDGE_LOT = 1e-6
PAPER_RIDGE_RAW = 1e-4
PAPER_T_TRAIN = 50
PAPER_H = 48


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def swirling_trajectory():
    """Swirling cluster: N=200, 2 cycles, 250 steps, radius 1–6, NOT normalized."""
    from data_utils.simulation.measure_dynamical_systems import SwirlingClusterSystem
    sw = SwirlingClusterSystem(N=200, noise_scale=0.02, seed=42)
    return sw.get_cyclical_trajectory(n_cycles=2, n_steps=250, max_radius=6.0)


@pytest.fixture(scope="module")
def geodesic_trajectory():
    """Geodesic transport: circle → triangle, 2 cycles, 400 steps, NOT normalized."""
    from data_utils.simulation.measure_dynamical_systems import GeodesicTransportSimulator
    N = 200
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    source = np.column_stack([np.cos(theta), np.sin(theta)]) * 0.8
    v = np.array([[0, 1.0], [np.sin(2 * np.pi / 3), -0.5], [-np.sin(2 * np.pi / 3), -0.5]])
    target = []
    per_side = [N // 3, N // 3, N - 2 * (N // 3)]
    for i, ns in enumerate(per_side):
        a, b = v[i], v[(i + 1) % 3]
        for t in np.linspace(0, 1, ns, endpoint=False):
            target.append((1 - t) * a + t * b)
    target = np.array(target)
    geo = GeodesicTransportSimulator(source, target)
    return geo.get_cyclical_trajectory(n_cycles=2, n_steps=400)


@pytest.fixture(scope="module")
def lot_embedding_swirl(swirling_trajectory):
    """Compute LOT embedding for the swirling cluster."""
    from data_utils.simulation.generate_lot_embeddings import (
        generate_reference, compute_ot_plan, plan_to_barycentric_weights,
        apply_barycentric_projection,
    )
    traj = swirling_trajectory
    T, N, d = traj.shape
    R = N
    sigma = generate_reference("gaussian_iso", R, trajectory=traj, d=d)
    lot_maps = np.empty((T, R, d))
    for t in range(T):
        plan = compute_ot_plan(sigma, traj[t], ot_method="emd")
        weights = plan_to_barycentric_weights(plan)
        lot_maps[t] = apply_barycentric_projection(weights, traj[t])
    velocities = lot_maps[1:] - lot_maps[:-1]
    return sigma, lot_maps, velocities


@pytest.fixture(scope="module")
def lot_embedding_geodesic(geodesic_trajectory):
    """Compute LOT embedding for geodesic transport."""
    from data_utils.simulation.generate_lot_embeddings import (
        generate_reference, compute_ot_plan, plan_to_barycentric_weights,
        apply_barycentric_projection,
    )
    traj = geodesic_trajectory
    T, N, d = traj.shape
    R = N
    sigma = generate_reference("gaussian_iso", R, trajectory=traj, d=d)
    lot_maps = np.empty((T, R, d))
    for t in range(T):
        plan = compute_ot_plan(sigma, traj[t], ot_method="emd")
        weights = plan_to_barycentric_weights(plan)
        lot_maps[t] = apply_barycentric_projection(weights, traj[t])
    velocities = lot_maps[1:] - lot_maps[:-1]
    return sigma, lot_maps, velocities


# ═══════════════════════════════════════════════════════════════
# STAGE 1: TRAJECTORY GENERATION
# ═══════════════════════════════════════════════════════════════

class TestStage1Trajectory:
    """Stage 1: Particle trajectory generation."""

    def test_swirling_shape(self, swirling_trajectory):
        traj = swirling_trajectory
        assert traj.shape == (250, 200, 2), f"Expected (250, 200, 2), got {traj.shape}"

    def test_swirling_not_normalized(self, swirling_trajectory):
        """Paper uses raw coordinates; radius spans 1 to 6."""
        traj = swirling_trajectory
        extent = traj.max() - traj.min()
        assert extent > 5.0, (
            f"Swirling cluster should span radius 1–6 (extent ~12), got {extent:.2f}. "
            "Do NOT normalize synthetic data to [0,1]²."
        )

    def test_swirling_periodic(self, swirling_trajectory):
        """Trajectory should be approximately periodic (n_cycles=2, cycle_length=125)."""
        traj = swirling_trajectory
        cycle_len = 125
        # Frame 0 and frame cycle_len should have similar radii
        r0 = np.linalg.norm(traj[0], axis=-1).mean()
        r_cycle = np.linalg.norm(traj[cycle_len], axis=-1).mean()
        assert abs(r0 - r_cycle) < 1.0, f"Not periodic: r(0)={r0:.2f}, r({cycle_len})={r_cycle:.2f}"

    def test_geodesic_shape(self, geodesic_trajectory):
        traj = geodesic_trajectory
        assert traj.shape == (400, 200, 2), f"Expected (400, 200, 2), got {traj.shape}"

    def test_geodesic_not_normalized(self, geodesic_trajectory):
        """Geodesic coordinates span ~[-1.5, 1.5], not [0,1]."""
        traj = geodesic_trajectory
        assert traj.min() < 0, f"Geodesic should have negative coords, min={traj.min():.2f}"

    def test_geodesic_starts_on_circle(self, geodesic_trajectory):
        """Source is a circle of radius 0.8."""
        radii = np.linalg.norm(geodesic_trajectory[0], axis=-1)
        np.testing.assert_allclose(radii, 0.8, atol=0.05)


# ═══════════════════════════════════════════════════════════════
# STAGE 2: LOT EMBEDDING
# ═══════════════════════════════════════════════════════════════

class TestStage2LOTEmbedding:
    """Stage 2: LOT embedding — OT plan, barycentric projection, velocities."""

    def test_lot_maps_shape(self, lot_embedding_swirl, swirling_trajectory):
        _, lot_maps, _ = lot_embedding_swirl
        T, N, d = swirling_trajectory.shape
        assert lot_maps.shape == (T, N, d)

    def test_velocities_shape(self, lot_embedding_swirl, swirling_trajectory):
        _, _, velocities = lot_embedding_swirl
        T, N, d = swirling_trajectory.shape
        assert velocities.shape == (T - 1, N, d)

    def test_velocity_equals_map_difference(self, lot_embedding_swirl):
        """Paper Eq. (lot-vel-discrete): Δ_t = u_{t+1} - u_t."""
        _, lot_maps, velocities = lot_embedding_swirl
        expected = lot_maps[1:] - lot_maps[:-1]
        np.testing.assert_array_almost_equal(velocities, expected, decimal=10)

    def test_ot_plan_is_valid_coupling(self, swirling_trajectory):
        """OT plan must have correct marginals: row sums = 1/R, col sums = 1/N."""
        from data_utils.simulation.generate_lot_embeddings import (
            generate_reference, compute_ot_plan,
        )
        traj = swirling_trajectory
        T, N, d = traj.shape
        R = N
        sigma = generate_reference("gaussian_iso", R, trajectory=traj, d=d)
        plan = compute_ot_plan(sigma, traj[0], ot_method="emd")

        np.testing.assert_allclose(plan.sum(axis=1), 1.0 / R, atol=1e-8,
                                   err_msg="Row sums should be 1/R")
        np.testing.assert_allclose(plan.sum(axis=0), 1.0 / N, atol=1e-8,
                                   err_msg="Col sums should be 1/N")

    def test_barycentric_weights_sum_to_one(self, swirling_trajectory):
        """After normalization, each row of weights should sum to 1."""
        from data_utils.simulation.generate_lot_embeddings import (
            generate_reference, compute_ot_plan, plan_to_barycentric_weights,
        )
        traj = swirling_trajectory
        R = N = traj.shape[1]
        sigma = generate_reference("gaussian_iso", R, trajectory=traj, d=2)
        plan = compute_ot_plan(sigma, traj[0], ot_method="emd")
        weights = plan_to_barycentric_weights(plan)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-8)

    def test_lot_maps_near_trajectory_centroid(self, lot_embedding_swirl, swirling_trajectory):
        """LOT map centroids should approximately track trajectory centroids."""
        _, lot_maps, _ = lot_embedding_swirl
        traj = swirling_trajectory
        for t in [0, 50, 100]:
            traj_centroid = traj[t].mean(axis=0)
            map_centroid = lot_maps[t].mean(axis=0)
            dist = np.linalg.norm(traj_centroid - map_centroid)
            assert dist < 2.0, (
                f"LOT map centroid at t={t} too far from trajectory: {dist:.2f}"
            )

    def test_geodesic_lot_maps(self, lot_embedding_geodesic, geodesic_trajectory):
        """Basic shape check for geodesic LOT embedding."""
        _, lot_maps, velocities = lot_embedding_geodesic
        T, N, d = geodesic_trajectory.shape
        assert lot_maps.shape == (T, N, d)
        assert velocities.shape == (T - 1, N, d)


# ═══════════════════════════════════════════════════════════════
# STAGE 3: RC TRAINING
# ═══════════════════════════════════════════════════════════════

class TestStage3RCTraining:
    """Stage 3: Reservoir computer — architecture and training."""

    def _make_rc(self, R, d, ridge):
        from rc_computer import ReservoirComputer, ReservoirConfig, InitScheme
        return ReservoirComputer(ReservoirConfig(
            input_size=R * d,
            reservoir_size=R,
            output_size=R * d,
            spectral_radius=PAPER_SR,
            input_scaling=PAPER_INPUT_SCALE,
            leak_rate=PAPER_LEAK,
            ridge_param=ridge,
            bias_scale=0.0,
            activation="tanh",
            init_scheme=InitScheme.SPARSE,
            sparsity=0.1,
            random_seed=42,
            use_operator_norm=False,
        ))

    def test_velocity_input_target_alignment(self, lot_embedding_swirl):
        """VELOCITY: input is Δ_t, target is Δ_{t+1} (one-step shift)."""
        _, _, velocities = lot_embedding_swirl
        R, d = velocities.shape[1], velocities.shape[2]
        vel_flat = velocities.reshape(-1, R * d)
        inp = vel_flat[:-1]
        tgt = vel_flat[1:]
        assert inp.shape[0] == tgt.shape[0]
        # First target should equal second velocity
        np.testing.assert_array_equal(tgt[0], vel_flat[1])

    def test_positions_input_target_alignment(self, lot_embedding_swirl):
        """POSITIONS: input is u_t, target is u_{t+1}."""
        _, lot_maps, _ = lot_embedding_swirl
        R, d = lot_maps.shape[1], lot_maps.shape[2]
        map_flat = lot_maps.reshape(-1, R * d)
        inp = map_flat[:-1]
        tgt = map_flat[1:]
        assert inp.shape[0] == tgt.shape[0]
        np.testing.assert_array_equal(tgt[0], map_flat[1])

    def test_rc_spectral_radius(self):
        """Paper §5.2: W is scaled so max |eigenvalue| = 0.7."""
        rc = self._make_rc(50, 2, PAPER_RIDGE_LOT)
        eigenvalues = np.linalg.eigvals(rc.W)
        actual_sr = np.max(np.abs(eigenvalues))
        np.testing.assert_allclose(actual_sr, PAPER_SR, atol=0.05,
                                   err_msg=f"Spectral radius should be ~{PAPER_SR}")

    def test_rc_no_bias(self):
        """Paper §5.2: no additive bias."""
        rc = self._make_rc(50, 2, PAPER_RIDGE_LOT)
        np.testing.assert_array_equal(rc.bias, np.zeros_like(rc.bias))

    def test_rc_sparsity(self):
        """Paper §5.2: 10% density in W."""
        rc = self._make_rc(100, 2, PAPER_RIDGE_LOT)
        density = np.count_nonzero(rc.W) / rc.W.size
        assert 0.05 < density < 0.20, f"Density should be ~0.1, got {density:.3f}"

    def test_teacher_forcing_shape(self, lot_embedding_swirl):
        """Teacher forcing produces (T_train, reservoir_size) states."""
        _, _, velocities = lot_embedding_swirl
        R, d = velocities.shape[1], velocities.shape[2]
        vel_flat = velocities.reshape(-1, R * d)
        train_input = vel_flat[:PAPER_T_TRAIN]

        rc = self._make_rc(R, d, PAPER_RIDGE_LOT)
        states = rc.run(train_input)
        assert states.shape == (PAPER_T_TRAIN, R)

    def test_ridge_regression_produces_wout(self, lot_embedding_swirl):
        """Training should produce W_out with shape (n_r + 1, output_size)."""
        _, _, velocities = lot_embedding_swirl
        R, d = velocities.shape[1], velocities.shape[2]
        vel_flat = velocities.reshape(-1, R * d)
        train_input = vel_flat[:PAPER_T_TRAIN]
        train_target = vel_flat[1:PAPER_T_TRAIN + 1]

        rc = self._make_rc(R, d, PAPER_RIDGE_LOT)
        states = rc.run(train_input)
        rc.train(states, train_target)
        assert rc.W_out is not None
        assert rc.W_out.shape == (R + 1, R * d), (
            f"W_out shape should be ({R+1}, {R*d}), got {rc.W_out.shape}"
        )

    def test_train_rmse_is_small(self, lot_embedding_swirl):
        """Training error should be small (the RC can fit the training data)."""
        _, _, velocities = lot_embedding_swirl
        R, d = velocities.shape[1], velocities.shape[2]
        vel_flat = velocities.reshape(-1, R * d)
        train_input = vel_flat[:PAPER_T_TRAIN]
        train_target = vel_flat[1:PAPER_T_TRAIN + 1]

        rc = self._make_rc(R, d, PAPER_RIDGE_LOT)
        states = rc.run(train_input)
        rc.train(states, train_target)
        pred = rc.predict(states)
        rmse = np.sqrt(np.mean((pred - train_target) ** 2))
        assert rmse < 0.1, f"Training RMSE too high: {rmse:.4f}"


# ═══════════════════════════════════════════════════════════════
# STAGE 4: AUTONOMOUS FORECASTING
# ═══════════════════════════════════════════════════════════════

class TestStage4AutonomousForecasting:
    """Stage 4: Closed-loop autonomous prediction."""

    def _train_rc(self, velocities, ridge):
        from rc_computer import ReservoirComputer, ReservoirConfig, InitScheme
        R, d = velocities.shape[1], velocities.shape[2]
        vel_flat = velocities.reshape(-1, R * d)
        rc = ReservoirComputer(ReservoirConfig(
            input_size=R * d, reservoir_size=R, output_size=R * d,
            spectral_radius=PAPER_SR, input_scaling=PAPER_INPUT_SCALE,
            leak_rate=PAPER_LEAK, ridge_param=ridge, bias_scale=0.0,
            activation="tanh", init_scheme=InitScheme.SPARSE,
            sparsity=0.1, random_seed=42, use_operator_norm=False,
        ))
        train_input = vel_flat[:PAPER_T_TRAIN]
        train_target = vel_flat[1:PAPER_T_TRAIN + 1]
        states = rc.run(train_input)
        rc.train(states, train_target)
        return rc, train_input, vel_flat

    def test_autonomous_output_shape(self, lot_embedding_swirl):
        _, _, velocities = lot_embedding_swirl
        rc, train_input, _ = self._train_rc(velocities, PAPER_RIDGE_LOT)
        R, d = velocities.shape[1], velocities.shape[2]
        states, predictions = rc.run_autonomous(train_input, PAPER_H)
        assert predictions.shape == (PAPER_H, R * d)
        assert states.shape == (PAPER_T_TRAIN + PAPER_H, R)

    def test_autonomous_predictions_finite(self, lot_embedding_swirl):
        """Predictions should not blow up (no NaN/Inf)."""
        _, _, velocities = lot_embedding_swirl
        rc, train_input, _ = self._train_rc(velocities, PAPER_RIDGE_LOT)
        _, predictions = rc.run_autonomous(train_input, PAPER_H)
        assert np.all(np.isfinite(predictions)), "Autonomous predictions contain NaN/Inf"

    def test_damped_rollout_shape(self, lot_embedding_swirl):
        """run_autonomous_damped returns (states, predictions) with correct shapes."""
        _, _, velocities = lot_embedding_swirl
        rc, train_input, _ = self._train_rc(velocities, PAPER_RIDGE_LOT)
        R, d = velocities.shape[1], velocities.shape[2]
        states, predictions = rc.run_autonomous_damped(
            train_input, PAPER_H, damping=0.95, decay=0.99
        )
        assert predictions.shape == (PAPER_H, R * d)
        assert states.shape == (PAPER_T_TRAIN + PAPER_H, R)

    def test_damped_reduces_prediction_magnitude(self, lot_embedding_swirl):
        """Damped predictions should have smaller magnitude than standard."""
        _, _, velocities = lot_embedding_swirl
        rc, train_input, _ = self._train_rc(velocities, PAPER_RIDGE_LOT)
        _, pred_std = rc.run_autonomous(train_input, PAPER_H)
        _, pred_damp = rc.run_autonomous_damped(train_input, PAPER_H, damping=0.5, decay=0.9)
        assert np.linalg.norm(pred_damp) < np.linalg.norm(pred_std), (
            "Damped predictions should have smaller norm than standard"
        )

    def test_nudged_rollout_shape(self, lot_embedding_swirl):
        """run_autonomous_nudged returns correct shapes."""
        _, _, velocities = lot_embedding_swirl
        rc, train_input, vel_flat = self._train_rc(velocities, PAPER_RIDGE_LOT)
        R, d = velocities.shape[1], velocities.shape[2]
        obs = vel_flat[PAPER_T_TRAIN:PAPER_T_TRAIN + PAPER_H]
        states, predictions = rc.run_autonomous_nudged(
            train_input, PAPER_H, observations=obs,
            nudge_interval=4, nudge_strength=0.5,
        )
        assert predictions.shape == (PAPER_H, R * d)


# ═══════════════════════════════════════════════════════════════
# STAGE 5: INTEGRATION & RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════

class TestStage5Integration:
    """Stage 5: Velocity integration to recover LOT maps."""

    def test_integration_from_seed(self, lot_embedding_swirl):
        """Integrating true velocities from seed should recover the true maps."""
        _, lot_maps, velocities = lot_embedding_swirl
        seed = lot_maps[0]
        reconstructed = np.empty_like(lot_maps)
        reconstructed[0] = seed
        for t in range(len(velocities)):
            reconstructed[t + 1] = reconstructed[t] + velocities[t]
        np.testing.assert_allclose(reconstructed, lot_maps, atol=1e-10,
                                   err_msg="Integration of true velocities should exactly recover maps")

    def test_forecast_seed_is_last_training_map(self, lot_embedding_swirl):
        """The seed for forecasting should be lot_maps[T_TRAIN], not lot_maps[0]."""
        _, lot_maps, _ = lot_embedding_swirl
        seed = lot_maps[PAPER_T_TRAIN]
        assert seed.shape == (lot_maps.shape[1], lot_maps.shape[2])

    def test_predicted_measures_are_map_values(self, lot_embedding_swirl):
        """The predicted measures ARE the integrated map values (pushforward of σ)."""
        sigma, lot_maps, velocities = lot_embedding_swirl
        R, d = lot_maps.shape[1], lot_maps.shape[2]
        # pred_measures_X = pred_maps[1:] in the code
        # These are u_hat(x_i) values = the predicted particle positions
        seed = lot_maps[PAPER_T_TRAIN]
        # Simulate a trivial 1-step integration with the true velocity
        true_vel = velocities[PAPER_T_TRAIN]
        pred_map_1 = seed + true_vel
        expected_true = lot_maps[PAPER_T_TRAIN + 1]
        np.testing.assert_allclose(pred_map_1, expected_true, atol=1e-10)


# ═══════════════════════════════════════════════════════════════
# STAGE 6: EVALUATION
# ═══════════════════════════════════════════════════════════════

class TestStage6Evaluation:
    """Stage 6: L²(σ) RMSE evaluation."""

    def test_l2_sigma_rmse_zero_for_identical(self):
        """RMSE should be zero when pred == true."""
        x = np.random.randn(10, 50, 2)
        rmse = _l2_sigma_rmse(x, x)
        assert rmse < 1e-12

    def test_l2_sigma_rmse_positive(self):
        """RMSE should be positive when pred != true."""
        x = np.random.randn(10, 50, 2)
        y = x + 0.1
        rmse = _l2_sigma_rmse(x, y)
        assert rmse > 0

    def test_l2_sigma_rmse_scale(self):
        """Shifting all predictions by offset should give RMSE = offset magnitude."""
        x = np.zeros((10, 50, 2))
        y = np.ones((10, 50, 2)) * 0.5
        rmse = _l2_sigma_rmse(x, y)
        expected = np.sqrt(0.5 ** 2 + 0.5 ** 2)  # ||[0.5, 0.5]|| = sqrt(0.5)
        np.testing.assert_allclose(rmse, expected, atol=1e-10)


# ═══════════════════════════════════════════════════════════════
# END-TO-END: VELOCITY BEATS POSITIONS
# ═══════════════════════════════════════════════════════════════

class TestEndToEndVelocityWins:
    """
    End-to-end test: velocity forecasting should outperform positions
    on the swirling cluster with RAW (unnormalized) coordinates.

    This is the paper's central claim.

    NOTE: The swirling cluster has cycle_length=125. The RC needs at
    least one full cycle of warm-up to learn the periodic velocity field.
    The paper's T_train=50 is for the ablation grid with shorter cycles;
    velocity_forecast.py uses cycle_length-1 = 124 warm-up steps for
    the swirling cluster (see warmup_utils.py).
    """

    def test_velocity_beats_positions_swirling(self, lot_embedding_swirl):
        """On unnormalized swirling cluster, velocity RMSE < positions RMSE."""
        from rc_computer import ReservoirComputer, ReservoirConfig, InitScheme

        _, lot_maps, velocities = lot_embedding_swirl
        T, R, d = lot_maps.shape

        # Swirling cluster cycle_length = 125, so warm-up = 124
        # (matches velocity_forecast.py compute_warm_steps_for_velocity)
        cycle_length = 125
        warm_steps = cycle_length - 1  # 124

        def _make_rc(ridge):
            return ReservoirComputer(ReservoirConfig(
                input_size=R * d, reservoir_size=R, output_size=R * d,
                spectral_radius=PAPER_SR, input_scaling=PAPER_INPUT_SCALE,
                leak_rate=PAPER_LEAK, ridge_param=ridge,
                bias_scale=0.0, activation="tanh", init_scheme=InitScheme.SPARSE,
                sparsity=0.1, random_seed=42, use_operator_norm=False,
            ))

        # ── Velocity RC ──
        vel_flat = velocities.reshape(T - 1, R * d)
        rc_vel = _make_rc(PAPER_RIDGE_LOT)
        train_in_vel = vel_flat[:warm_steps]
        train_tgt_vel = vel_flat[1:warm_steps + 1]
        states = rc_vel.run(train_in_vel)
        rc_vel.train(states, train_tgt_vel)

        H = min(PAPER_H, T - warm_steps - 2)
        _, pred_vel_flat = rc_vel.run_autonomous(train_in_vel, H)
        pred_vel = pred_vel_flat.reshape(H, R, d)

        seed = lot_maps[warm_steps]
        pred_maps_vel = np.empty((H + 1, R, d))
        pred_maps_vel[0] = seed
        for k in range(H):
            pred_maps_vel[k + 1] = pred_maps_vel[k] + pred_vel[k]

        # ── Positions RC ──
        map_flat = lot_maps.reshape(T, R * d)
        rc_pos = _make_rc(PAPER_RIDGE_RAW)
        train_in_pos = map_flat[:warm_steps]
        train_tgt_pos = map_flat[1:warm_steps + 1]
        states = rc_pos.run(train_in_pos)
        rc_pos.train(states, train_tgt_pos)

        _, pred_pos_flat = rc_pos.run_autonomous(train_in_pos, H)
        pred_maps_pos = pred_pos_flat.reshape(H, R, d)

        # ── Compare ──
        true_maps = lot_maps[warm_steps + 1:warm_steps + 1 + H]
        H_cmp = min(H, true_maps.shape[0])

        rmse_vel = _l2_sigma_rmse(pred_maps_vel[1:H_cmp + 1], true_maps[:H_cmp])
        rmse_pos = _l2_sigma_rmse(pred_maps_pos[:H_cmp], true_maps[:H_cmp])

        improvement = (rmse_pos - rmse_vel) / rmse_pos * 100
        print(f"\n  Swirling cluster (raw coords, warm={warm_steps}):")
        print(f"    Velocity RMSE: {rmse_vel:.6f}")
        print(f"    Positions RMSE: {rmse_pos:.6f}")
        print(f"    Improvement: {improvement:+.1f}%")

        assert rmse_vel < rmse_pos, (
            f"Velocity ({rmse_vel:.4f}) should beat positions ({rmse_pos:.4f}) "
            f"on unnormalized swirling cluster with {warm_steps}-step warm-up"
        )


# ═══════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════

def _l2_sigma_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """L²(σ) RMSE: sqrt(mean over time and ref points of ||pred - true||²)."""
    diff = pred - true
    sq_norms = np.sum(diff ** 2, axis=-1)
    per_time = np.mean(sq_norms, axis=-1)
    return float(np.sqrt(np.mean(per_time)))
