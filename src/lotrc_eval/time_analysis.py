#!/usr/bin/env python3
"""
time_analysis_geodesic.py

Summarize wall-clock times for geodesic RC runs (RAW and LOT variants).

What it does
------------
- Reads per-run artifacts to extract:
  * H, N, d from arrays (predicted_measures_X.npy / true_measures_X.npy).
  * warm/autonomous steps if available in warm_start_info.npy.
  * timing fields from metadata.json/metrics.json, e.g.:
        RAW: {"time_train_s": ..., "time_rollout_s": ..., "time_total_s": ...}
        LOT: {"time_train": ..., "time_rollout": ..., "time_total": ...}
  * fallback timing = (max mtime - min mtime) across files under run_dir.

- Emits a CSV with one row per run:
    run_name, system, domain, H, N, d, warm_steps, autonomous_steps,
    train_time_sec, rollout_time_sec, total_time_sec, timing_source, run_dir

- Prints a compact summary to stdout.

Typical usage
-------------
python time_analysis_geodesic.py \
  --runs forecast_output/geodesic_transport_N_500_raw/res_100pct_of_N \
         forecast_output/geodesic_transport_N_500/clean_circle/frac100/lot_velocity_resFromOrigN_100pct \
  --out results/geodesic_time_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np


# Files that indicate a valid run directory
REQ_FILES = [
    "predicted_measures_X.npy",
    "true_measures_X.npy",
    "measures_w.npy",
]

# Additional files to check for (not all required)
OPT_FILES = [
    "reference_sigma_X.npy",
    "warm_start_info.npy",
    "metadata.json",
    "metrics.json",
    "rc_config.json",
]


def _safe_load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _safe_load_npy(p: Path) -> Optional[np.ndarray]:
    try:
        return np.load(p)
    except Exception:
        return None


def _gather_files_recursive(run_dir: Path) -> List[Path]:
    return [q for q in run_dir.rglob("*") if q.is_file()]


def _mtime_span_seconds(files: List[Path]) -> Optional[float]:
    if not files:
        return None
    try:
        mtimes = [f.stat().st_mtime for f in files]
        return float(max(mtimes) - min(mtimes))
    except Exception:
        return None


def _infer_system_and_domain(run_dir: Path, meta: Optional[dict] = None) -> Tuple[str, str]:
    """
    Infer system name and domain from path and metadata.
    Returns (system, domain) tuple.
    """
    s = str(run_dir.as_posix()).lower()
    
    # Check metadata first
    if meta:
        system = meta.get("system", "")
        domain = meta.get("domain", "")
        if system and domain:
            return str(system), str(domain)
    
    # Fallback to path-based inference
    system = "unknown"
    domain = "unknown"
    
    if "geodesic" in s:
        system = "geodesic_transport"
    elif "swirl" in s:
        system = "swirling_cluster"
    
    if "_raw" in s or "raw_" in s:
        domain = "raw_positions"
    elif "lot_velocity" in s or "velocity" in s:
        domain = "lot_velocity"
    elif "lot_" in s:
        domain = "lot_positions"
    
    return system, domain


def analyze_one_run(run_dir: Path) -> Dict[str, object]:
    """
    Return a dict with:
      run_name, system, domain, H, N, d, warm_steps, autonomous_steps,
      train_time_sec, rollout_time_sec, total_time_sec, timing_source, run_dir
    """
    run_dir = run_dir.resolve()
    out: Dict[str, object] = {
        "run_name": run_dir.name,
        "system": "unknown",
        "domain": "unknown", 
        "H": "",
        "N": "",
        "d": "",
        "warm_steps": "",
        "autonomous_steps": "",
        "train_time_sec": "",
        "rollout_time_sec": "",
        "total_time_sec": "",
        "timing_source": "unknown",
        "run_dir": str(run_dir),
    }

    # ---- Load metadata first for system/domain info ----
    meta = _safe_load_json(run_dir / "metadata.json")
    if meta is None:
        meta = _safe_load_json(run_dir / "metrics.json")
    
    system, domain = _infer_system_and_domain(run_dir, meta)
    out.update({"system": system, "domain": domain})

    # ---- Load shapes (H,N,d) from measures arrays if present ----
    pred = _safe_load_npy(run_dir / "predicted_measures_X.npy")
    true = _safe_load_npy(run_dir / "true_measures_X.npy")
    if pred is not None and pred.ndim == 3:
        H, N, d = pred.shape
        out.update({"H": int(H), "N": int(N), "d": int(d)})
    elif true is not None and true.ndim == 3:
        H, N, d = true.shape
        out.update({"H": int(H), "N": int(N), "d": int(d)})

    # ---- Warm/autonomous steps (if saved) ----
    warm_info = _safe_load_npy(run_dir / "warm_start_info.npy")
    if warm_info is not None:
        try:
            if warm_info.size >= 2:
                warm_steps = int(warm_info[0])
                autonomous_steps = int(warm_info[1])
                out.update({"warm_steps": warm_steps, "autonomous_steps": autonomous_steps})
        except Exception:
            pass

    # ---- Extract timing information from CSV summaries first, then metadata ----
    used_meta = False
    
    # Try CSV summaries first (more reliable)
    try:
        import pandas as pd
        run_dir_str = str(run_dir)
        
        # RAW CSV lookup
        raw_csv = Path("results/raw_rc_runtime_summary.csv")
        if raw_csv.exists() and "_raw" in run_dir_str:
            df = pd.read_csv(raw_csv, on_bad_lines='skip', engine='python')
            matches = df[df['output_dir'].str.contains(run_dir.name, na=False)]
            if not matches.empty:
                row = matches.iloc[0]
                out.update({
                    "train_time_sec": float(row.get("time_train_s", 0)),
                    "rollout_time_sec": float(row.get("time_rollout_s", 0)),
                    "total_time_sec": float(row.get("time_total_s", 0)),
                    "timing_source": "raw_csv_summary"
                })
                used_meta = True
        
        # LOT CSV lookup  
        lot_csv = Path("results/rc_runtime_summary_geodesic_lot_velocity.csv")
        if lot_csv.exists() and not used_meta and ("lot_velocity" in run_dir_str or "velocity" in run_dir_str):
            df = pd.read_csv(lot_csv, on_bad_lines='skip', engine='python')
            matches = df[df['output_dir'].str.contains(run_dir.name, na=False)]
            if not matches.empty:
                row = matches.iloc[0]
                out.update({
                    "train_time_sec": float(row.get("time_train", 0)),
                    "rollout_time_sec": float(row.get("time_rollout", 0)),
                    "total_time_sec": float(row.get("time_total", 0)),
                    "timing_source": "lot_csv_summary"
                })
                used_meta = True
                
    except Exception:
        pass  # Fall back to metadata extraction
    
    # Fallback: extract from metadata/metrics files
    if not used_meta and meta:
        # RAW format: time_train_s, time_rollout_s, time_total_s
        # LOT format: time_train, time_rollout, time_total
        tr = (meta.get("time_train_s") or meta.get("time_train") or 
              meta.get("train_time_sec") or meta.get("training_time_sec"))
        ro = (meta.get("time_rollout_s") or meta.get("time_rollout") or 
              meta.get("rollout_time_sec") or meta.get("inference_time_sec") or
              meta.get("autonomous_time_sec"))
        tt = (meta.get("time_total_s") or meta.get("time_total") or 
              meta.get("total_time_sec") or meta.get("elapsed_time_sec"))
        
        # If total is missing but train/rollout exist, compute it
        if tt is None and (tr is not None or ro is not None):
            tt = (float(tr) if tr is not None else 0.0) + (float(ro) if ro is not None else 0.0)
        
        # If any were found, record and mark source
        have_any = any(v is not None for v in (tr, ro, tt))
        if have_any:
            if tr is not None:
                out["train_time_sec"] = float(tr)
            if ro is not None:
                out["rollout_time_sec"] = float(ro)
            if tt is not None:
                out["total_time_sec"] = float(tt)
            out["timing_source"] = "metadata.json" if (run_dir / "metadata.json").exists() else "metrics.json"
            used_meta = True

    # ---- Fallback: filesystem mtime span ----
    if not used_meta:
        files = _gather_files_recursive(run_dir)
        span = _mtime_span_seconds(files)
        if span is not None:
            out["total_time_sec"] = float(span)
            out["timing_source"] = "mtime_span"

    return out


def discover_runs(base: Path, pattern: Optional[str]) -> List[Path]:
    """
    If base is a directory containing one run, return [base].
    If --glob is provided, scan base.glob(pattern).
    If multiple bases provided on CLI, we handle that outside.
    """
    base = base.resolve()
    if pattern:
        matches = sorted([p.resolve() for p in base.glob(pattern) if p.is_dir()])
        return matches
    
    # If directory has required files, treat as a single run
    if all((base / f).exists() for f in REQ_FILES):
        return [base]
    
    # Otherwise, try immediate subdirs that look like runs
    cands = [p for p in base.iterdir() if p.is_dir()]
    runs = []
    for p in cands:
        if any((p / f).exists() for f in REQ_FILES):
            runs.append(p.resolve())
    return sorted(runs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time analysis for geodesic RC runs.")
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more run roots. Each may be a run dir or a parent dir containing runs.",
    )
    p.add_argument(
        "--glob",
        default=None,
        help="Optional glob pattern (relative to each --runs base) to select runs, e.g., '*/res_*' or 'clean_circle/*'.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path, e.g., results/geodesic_time_summary.csv",
    )
    return p.parse_args()


def analyze_runs(run_dirs: List[str], output_path: str, glob_pattern: Optional[str] = None):
    """
    Notebook-friendly version that takes direct parameters instead of using argparse.
    
    Args:
        run_dirs: List of directory paths to analyze
        output_path: Path for output CSV
        glob_pattern: Optional glob pattern for subdirectory matching
    """
    all_runs: List[Path] = []
    
    for b in run_dirs:
        base = Path(b)
        if not base.exists():
            print(f"[warn] missing base: {base}", file=sys.stderr)
            continue
        runs = discover_runs(base, glob_pattern)
        if not runs:
            print(f"[warn] no runs found under: {base} (glob={glob_pattern})", file=sys.stderr)
        all_runs.extend(runs)

    rows: List[Dict[str, object]] = []
    for rdir in all_runs:
        try:
            row = analyze_one_run(rdir)
            rows.append(row)
        except Exception as e:
            print(f"[error] failed on {rdir}: {e}", file=sys.stderr)

    # Write CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "run_name",
        "system", 
        "domain",
        "H",
        "N",
        "d",
        "warm_steps",
        "autonomous_steps",
        "train_time_sec",
        "rollout_time_sec",
        "total_time_sec",
        "timing_source",
        "run_dir",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in header})

    # Console summary
    print(f"[ok] wrote {len(rows)} rows → {out_path}")
    
    # Group by domain for cleaner display
    raw_runs = [r for r in rows if r.get("domain") == "raw_positions"]
    lot_runs = [r for r in rows if r.get("domain") == "lot_velocity"]
    
    if raw_runs:
        print(f"\nRAW runs ({len(raw_runs)}):")
        for r in raw_runs:
            print(f"  • {r['run_name']:>30s} | H={r.get('H','')} N={r.get('N','')} "
                  f"| total={r.get('total_time_sec',''):>8} s | source={r.get('timing_source','')}")
    
    if lot_runs:
        print(f"\nLOT runs ({len(lot_runs)}):")
        for r in lot_runs:
            print(f"  • {r['run_name']:>30s} | H={r.get('H','')} N={r.get('N','')} "
                  f"| total={r.get('total_time_sec',''):>8} s | source={r.get('timing_source','')}")
    
    return rows


def main():
    args = parse_args()
    analyze_runs(args.runs, str(args.out), args.glob)


if __name__ == "__main__":
    main()