"""
Tests aligned with PAPER_CONTEXT / paper.tex (OT marginals, barycentric map,
empirical L^2(sigma), discrete LOT velocity, ridge readout, ESP scaling).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_emd_plan_uniform_marginals():
    """Discrete OT: row sums 1/m, column sums 1/N (paper Sec. 2.1.4)."""
    ot = pytest.importorskip("ot")
    from data_utils.simulation.generate_lot_embeddings import compute_ot_plan

    np.random.seed(0)
    m, N, d = 5, 7, 2
    source = np.random.rand(m, d)
    target = np.random.rand(N, d)
    gamma = compute_ot_plan(source, target, ot_method="emd")
    assert gamma.shape == (m, N)
    np.testing.assert_allclose(gamma.sum(axis=1), np.ones(m) / m, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(gamma.sum(axis=0), np.ones(N) / N, rtol=1e-10, atol=1e-10)


def test_barycentric_projection_weights_sum_to_one():
    from data_utils.simulation.generate_lot_embeddings import (
        apply_barycentric_projection,
        plan_to_barycentric_weights,
    )

    rng = np.random.default_rng(1)
    plan = rng.random((4, 6))
    plan = plan / plan.sum()  # dummy nonnegative — not necessarily OT
    w = plan_to_barycentric_weights(plan)
    assert w.shape == plan.shape
    np.testing.assert_allclose(w.sum(axis=1), np.ones(4), rtol=1e-10, atol=1e-10)


def test_barycentric_matches_paper_formula_uniform_marginals():
    """û(x_i) = m * sum_j y_j * γ_ij when Σ_j γ_ij = 1/m."""
    from data_utils.simulation.generate_lot_embeddings import (
        apply_barycentric_projection,
        plan_to_barycentric_weights,
    )

    m, N, d = 3, 4, 2
    gamma = np.full((m, N), 1.0 / (m * N))
    y = np.arange(N * d, dtype=np.float64).reshape(N, d)
    weights = plan_to_barycentric_weights(gamma)
    u = apply_barycentric_projection(weights, y)
    for i in range(m):
        manual = m * (gamma[i : i + 1] @ y).ravel()
        np.testing.assert_allclose(u[i], manual, rtol=1e-12)


def test_empirical_L2_sigma_norm():
    """‖u‖^2_L2(σ) ≈ (1/m) Σ_i ‖u(x_i)‖^2 for uniform m atoms."""
    m = 10
    u = np.random.randn(m, 2)
    norm_sq = (np.linalg.norm(u, axis=1) ** 2).mean()
    assert norm_sq >= 0
    flat = u.reshape(-1)
    alt = np.dot(flat, flat) / m
    np.testing.assert_allclose(norm_sq, alt, rtol=1e-12)


def test_discrete_lot_velocity_is_temporal_difference():
    from data_utils.simulation.generate_lot_embeddings import compute_lot_embedding

    T, N, d = 8, 12, 2
    traj = np.cumsum(np.random.randn(T, N, d) * 0.05, axis=0) % 1.0
    ref = np.random.rand(6, d)
    maps, vel, _ = compute_lot_embedding(
        traj.astype(np.float32),
        ref.astype(np.float32),
        assignment="fixed",
        ot_method="emd",
    )
    diff = maps[1:] - maps[:-1]
    np.testing.assert_allclose(vel, diff, rtol=1e-5, atol=1e-5)


def test_ridge_readout_matches_normal_equations():
    from rc_computer import InitScheme, ReservoirComputer, ReservoirConfig

    rng = np.random.default_rng(42)
    n_in, n_res, n_out, T = 4, 30, 4, 60
    cfg = ReservoirConfig(
        input_size=n_in,
        reservoir_size=n_res,
        output_size=n_out,
        ridge_param=1e-3,
        random_seed=7,
        init_scheme=InitScheme.UNIFORM,
    )
    rc = ReservoirComputer(cfg)
    U = rng.standard_normal((T, n_in))
    Y = rng.standard_normal((T, n_out))
    states = rc.run(U)
    rc.train(states, Y, washout=0)
    R_aug = np.hstack([states, np.ones((T, 1))])
    ridge = cfg.ridge_param * np.eye(R_aug.shape[1])
    w_manual = np.linalg.solve(R_aug.T @ R_aug + ridge, R_aug.T @ Y)
    np.testing.assert_allclose(rc.W_out, w_manual, rtol=1e-10, atol=1e-10)


def test_operator_norm_scaling_matches_esp_convention():
    """After init with use_operator_norm, ‖A‖_2 ≈ spectral_radius (Prop. ESP)."""
    from rc_computer import InitScheme, ReservoirComputer, ReservoirConfig

    rho = 0.85
    cfg = ReservoirConfig(
        input_size=8,
        reservoir_size=40,
        output_size=8,
        spectral_radius=rho,
        use_operator_norm=True,
        random_seed=123,
        init_scheme=InitScheme.SPARSE,
    )
    rc = ReservoirComputer(cfg)
    op_norm = np.linalg.svd(rc.W, compute_uv=False)[0]
    np.testing.assert_allclose(op_norm, rho, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_sinkhorn_plan_approximate_marginals():
    import torch
    from sinkhorn_ot import sinkhorn_transport_plan

    B, m, N = 2, 5, 6
    d = 2
    rng = np.random.default_rng(3)
    cost = torch.tensor(rng.random((B, m, N)), dtype=torch.float32).square() * 2.0
    a = torch.full((B, m), 1.0 / m)
    b = torch.full((B, N), 1.0 / N)
    P = sinkhorn_transport_plan(cost, a, b, reg=0.08, num_iter=400)
    rs = P.sum(dim=-1)
    cs = P.sum(dim=-2)
    np.testing.assert_allclose(rs.numpy(), a.numpy(), rtol=0.02, atol=0.02)
    np.testing.assert_allclose(cs.numpy(), b.numpy(), rtol=0.02, atol=0.02)
