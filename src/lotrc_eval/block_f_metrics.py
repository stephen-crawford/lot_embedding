#!/usr/bin/env python3
"""
Block F — Metrics & diagnostics (backend agreement, run summaries).

Compatible with velocity_forecast.py and positions_forecast.py outputs.

Public API
----------
- plot_backend_agreement(D_ref, D_alt, title, savepath=None, idx=None) -> dict
- pack_run_alignment_summary(run_name, D, metrics_json_path, path_json_path=None, extra=None) -> dict
- summarize_runs_and_backends(out_dir, runs, deltas, eval_dirs, window_deltas=None, window_indices=None)

Notes
-----
- This block is backend-agnostic: it just consumes Δ matrices and the
  alignment artifacts written by Block E (alignment_metrics.json, path.json).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any
import json
import csv
import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _write_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _append_csv_row(csv_path: Path, header: list, row: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


# ═══════════════════════════════════════════════════════════════
# BACKEND AGREEMENT
# ═══════════════════════════════════════════════════════════════

def plot_backend_agreement(
    D_ref: np.ndarray,
    D_alt: np.ndarray,
    title: str,
    savepath: Optional[Path] = None,
    idx: Optional[np.ndarray] = None,
) -> dict:
    """
    Scatter plot comparing two Δ matrices (reference vs alternate backend).
    
    If idx is provided and one matrix is windowed (H'×H'), compare the submatrix
    D_ref[np.ix_(idx, idx)] to D_alt (or also subselect D_alt if it is full).
    """
    D_ref = np.asarray(D_ref, dtype=np.float64)
    D_alt = np.asarray(D_alt, dtype=np.float64)

    if idx is not None:
        Hsub = len(idx)
        ref_sub = D_ref[np.ix_(idx, idx)]
        if D_alt.shape == (Hsub, Hsub):
            x = ref_sub.reshape(-1)
            y = D_alt.reshape(-1)
        else:
            alt_sub = D_alt[np.ix_(idx, idx)]
            x = ref_sub.reshape(-1)
            y = alt_sub.reshape(-1)
    else:
        if D_ref.shape != D_alt.shape:
            raise ValueError(f"Shape mismatch without idx: {D_ref.shape} vs {D_alt.shape}")
        x = D_ref.reshape(-1)
        y = D_alt.reshape(-1)

    # Robust linear fit y ≈ a + b x (least squares)
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        b, a = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        a, b = np.nan, np.nan

    # Correlations
    try:
        from scipy.stats import pearsonr, spearmanr
        r = float(pearsonr(x, y)[0])
        rho = float(spearmanr(x, y).correlation)
    except Exception:
        xm, ym = x - x.mean(), y - y.mean()
        denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
        r = float((xm * ym).sum() / denom) if denom > 0 else float("nan")
        rho = float("nan")

    # Plot
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    ax.scatter(x, y, s=6, alpha=0.4)
    if np.isfinite(a) and np.isfinite(b):
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, a + b * xs, lw=1.5, label=f"fit: y≈{a:.3g}+{b:.3g}x")
    ax.set_title(title)
    ax.set_xlabel("Δ (reference backend)")
    ax.set_ylabel("Δ (alternate backend)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return dict(pearson_r=r, spearman_rho=rho, slope=b, intercept=a)


# ═══════════════════════════════════════════════════════════════
# RUN SUMMARY
# ═══════════════════════════════════════════════════════════════

def pack_run_alignment_summary(
    run_name: str,
    D: np.ndarray,
    metrics_json_path: Path,
    path_json_path: Optional[Path] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Load alignment_metrics.json (from Block E), augment with Δ-level stats and optional extras.
    """
    with open(metrics_json_path, "r") as f:
        metr = json.load(f)
    
    if extra:
        metr.update(extra)
    
    if path_json_path and path_json_path.exists():
        try:
            metr["path"] = json.loads(path_json_path.read_text())
        except Exception:
            pass
    
    metr["H"] = int(D.shape[0])
    metr["median_delta"] = float(np.median(D))
    metr["run"] = run_name
    
    return metr


# ═══════════════════════════════════════════════════════════════
# TOP-LEVEL AGGREGATOR
# ═══════════════════════════════════════════════════════════════

def summarize_runs_and_backends(
    out_dir: Path,
    runs: Dict[str, Any],
    deltas: Dict[str, np.ndarray],
    eval_dirs: Dict[str, Path],
    window_deltas: Optional[Dict[str, np.ndarray]] = None,
    window_indices: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """
    Aggregate metrics across runs and compare backends (optional).

    Parameters
    ----------
    out_dir : Path
        Where to write summary.json, summary.csv, and *_agreement.png plots.
    runs : dict
        Labels → RunData (only used for keys and logging).
    deltas : dict
        Keys like 'VELOCITY_llt_l2', 'POSITIONS_llt_l2' → full/window Δ for the *reference* backend.
    eval_dirs : dict
        Labels → directory where Block E wrote 'alignment_metrics.json' and 'path.json'.
        Example: {'VELOCITY': EVAL_OUT/'velocity_llt_l2', 'POSITIONS': EVAL_OUT/'positions_llt_l2'}
    window_deltas : dict | None
        Optional alternate-backend windowed Δs, keys like 'VELOCITY_sink_win', 'POSITIONS_sink_win'.
    window_indices : dict | None
        Optional mapping of the same keys in window_deltas to their index arrays.

    Writes
    ------
    - summary.json
    - summary.csv
    - {TAG}_backend_agreement.png (if window_deltas are provided)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-run summaries pulled from Block E artifacts
    run_summaries = {}
    for tag in runs.keys():
        eval_dir = eval_dirs.get(tag)
        if not eval_dir:
            print(f"[warn] No eval dir for {tag}; skipping.")
            continue
        
        metrics_json = Path(eval_dir) / "alignment_metrics.json"
        if not metrics_json.exists():
            print(f"[warn] Missing alignment_metrics.json in {eval_dir}; skipping {tag}.")
            continue

        # Choose a matching Δ (prefer '<tag>_llt_l2', fallback to any with prefix)
        D_key = f"{tag}_llt_l2"
        if D_key not in deltas:
            # try any delta with this tag as prefix
            candidates = [k for k in deltas if k.startswith(tag)]
            if not candidates:
                print(f"[warn] No Δ found for {tag}; skipping.")
                continue
            D_key = candidates[0]
        
        D = deltas[D_key]
        path_json = Path(eval_dir) / "path.json"
        packed = pack_run_alignment_summary(tag, D, metrics_json, path_json_path=path_json)
        run_summaries[tag] = dict(packed, eval_dir=str(eval_dir))

    # 2) Backend agreement (optional) — compare LLT+L2 vs Sinkhorn window
    backend_agreements = {}
    if window_deltas:
        for tag in runs.keys():
            key_ref = f"{tag}_llt_l2"
            key_alt = f"{tag}_sink_win"
            if key_ref in deltas and key_alt in window_deltas:
                D_ref = deltas[key_ref]
                D_alt = window_deltas[key_alt]
                idx = window_indices.get(key_alt) if window_indices else None
                png = out_dir / f"{tag}_backend_agreement.png"
                stats = plot_backend_agreement(
                    D_ref, D_alt, f"{tag}: LLT+L² vs Sinkhorn (window)", png, idx
                )
                backend_agreements[tag] = stats

    # 3) Save JSON summary
    summary = dict(
        runs=list(runs.keys()),
        run_summaries=run_summaries,
        backend_agreements=backend_agreements,
    )
    _write_json(summary, out_dir / "summary.json")

    # 4) Save compact CSV
    header = [
        "tag", "H",
        "unaligned_mean", "aligned_mean", "improve_pct",
        "dilation_ratio", "soft_dtw_value", "median_delta",
        "pearson_r_win", "spearman_rho_win",
    ]
    
    for tag in runs.keys():
        rs = run_summaries.get(tag)
        if not rs:
            continue
        agree = backend_agreements.get(tag, {})
        row = [
            tag,
            rs.get("H", ""),
            f"{rs.get('unaligned_mean', float('nan')):.6g}",
            f"{rs.get('aligned_mean', float('nan')):.6g}",
            f"{rs.get('improve_pct', float('nan')):.4g}",
            f"{rs.get('dilation_ratio', float('nan')):.4g}",
            f"{rs.get('soft_dtw_value', float('nan')):.6g}",
            f"{rs.get('median_delta', float('nan')):.6g}",
            f"{agree.get('pearson_r', float('nan')):.4g}" if agree else "",
            f"{agree.get('spearman_rho', float('nan')):.4g}" if agree else "",
        ]
        _append_csv_row(out_dir / "summary.csv", header, row)

    print(f"[summary] wrote: {out_dir/'summary.json'} and {out_dir/'summary.csv'}")