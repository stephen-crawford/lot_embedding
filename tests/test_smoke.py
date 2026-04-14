"""
Smoke tests for reservoir backends, Sinkhorn LOT, and imports.

Run from repo root (with venv activated):
    pip install pytest
    pytest
"""

from __future__ import annotations

import importlib.util
import numpy as np
import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_numpy_reservoir_run_train_predict():
    from rc_computer import ReservoirConfig, InitScheme, ReservoirComputer

    cfg = ReservoirConfig(
        input_size=5,
        reservoir_size=24,
        output_size=5,
        random_seed=0,
        init_scheme=InitScheme.SPARSE,
    )
    rc = ReservoirComputer(cfg)
    rng = np.random.default_rng(1)
    U = rng.standard_normal((25, 5))
    Y = np.roll(U, -1, axis=0)[:-1]
    U = U[:-1]
    states = rc.run(U)
    rc.train(states, Y, washout=2)
    pred = rc.predict(states)
    assert pred.shape == U.shape


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_torch_reservoir_matches_shapes():
    from forecast_backend import create_reservoir
    from rc_computer import ReservoirConfig, InitScheme

    cfg = ReservoirConfig(
        input_size=5,
        reservoir_size=24,
        output_size=5,
        random_seed=0,
        init_scheme=InitScheme.SPARSE,
    )
    rc, backend, dev = create_reservoir(cfg, "torch", "cpu")
    rng = np.random.default_rng(2)
    U = rng.standard_normal((25, 5))
    Y = np.roll(U, -1, axis=0)[:-1]
    U = U[:-1]
    states = rc.run(U)
    rc.train(states, Y, washout=2)
    pred = rc.predict(states)
    _, aut = rc.run_autonomous(U[:6], 4)
    assert pred.shape == U.shape
    assert aut.shape == (4, 5)
    assert backend == "torch"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_sinkhorn_lot_batched():
    import torch
    from sinkhorn_ot import lot_maps_from_reference_sinkhorn_batched

    R, N, d, B = 6, 10, 2, 4
    ref = torch.randn(R, d)
    tgt = torch.randn(B, N, d)
    m = lot_maps_from_reference_sinkhorn_batched(ref, tgt, 0.07, 80)
    assert m.shape == (B, R, d)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_generate_lot_embedding_sinkhorn_cpu():
    from data_utils.simulation.generate_lot_embeddings import compute_lot_embedding

    T, N, d = 12, 16, 2
    traj = np.random.rand(T, N, d).astype(np.float32)
    ref = np.random.rand(8, d).astype(np.float32)
    maps, vel, _ = compute_lot_embedding(
        traj,
        ref,
        assignment="per_frame",
        ot_method="sinkhorn",
        ot_device="cpu",
        sinkhorn_reg=0.1,
        sinkhorn_iters=60,
        sinkhorn_batch_frames=4,
    )
    assert maps.shape == (T, 8, d)
    assert vel.shape == (T - 1, 8, d)
