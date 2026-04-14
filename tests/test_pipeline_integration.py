"""Load the experiment pipeline module and ensure a minimal run produces valid metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pipeline_module():
    path = REPO_ROOT / "scripts" / "realworld_experiment_pipeline.py"
    spec = importlib.util.spec_from_file_location("realworld_experiment_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_produces_summary_metrics():
    mod = _load_pipeline_module()
    summary = mod.run_pipeline(
        system="pytest_pipeline_demo",
        n_particles=80,
        n_steps=200,
        n_cycles=2,
        goes_dir=None,
        quick=True,
        ot_method="emd",
        ot_device="cpu",
        rc_backend="numpy",
    )
    assert "forecast_map_rmse_velocity" in summary
    assert "forecast_map_rmse_positions" in summary
    assert summary["comparison_metric"] == "lot_embedding_map_rmse"
    assert summary["velocity_wins"] == summary["velocity_wins_map_rmse"]
    assert summary["forecast_map_rmse_velocity"] >= 0
    assert summary["forecast_map_rmse_positions"] >= 0
    summary_path = REPO_ROOT / "pipeline_run_summary.json"
    assert summary_path.is_file()
    with open(summary_path) as f:
        disk = json.load(f)
    assert disk["system"] == "pytest_pipeline_demo"
