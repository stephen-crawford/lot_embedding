#!/usr/bin/env python3
"""
Download data for real-world LOT+RC pipelines.

  GOES-16: cloud proxy (IR band) → particles.npy + times.npy + config.json
  SST:     NOAA OISST or synthetic fallback → same layout under sst_data/

After GOES download, run the experiment pipeline:
  python scripts/realworld_experiment_pipeline.py --goes-dir goes_data --N 500

Use the same --N as n_particles below (default 500).

Usage:
  pip install -r requirements-realworld.txt
  python scripts/download_realworld_data.py
  python scripts/download_realworld_data.py --source goes --hours 6 --n-particles 500
  python scripts/download_realworld_data.py --source sst --days 30
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


def download_goes_bundle(
    output_dir: Path,
    *,
    hours: int,
    n_particles: int,
    date: str,
    hour_utc: int,
) -> bool:
    try:
        from data_utils.goes_cloud_data import GOES2GO_AVAILABLE, fetch_goes_particles_bundle
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

    if not GOES2GO_AVAILABLE:
        print("[ERROR] goes2go not installed. Run: pip install -r requirements-realworld.txt")
        return False

    try:
        fetch_goes_particles_bundle(
            output_dir,
            hours=hours,
            n_particles=n_particles,
            date=date,
            hour_utc=hour_utc,
            verbose=True,
        )
    except Exception as e:
        print(f"[ERROR] GOES fetch failed: {e}")
        return False
    return True


def download_sst_bundle(output_dir: Path, *, days: int, n_particles: int) -> bool:
    from datetime import datetime as dt

    try:
        from data_utils.generate_sst_data import download_and_process_sst
    except ImportError as e:
        print(f"[ERROR] {e}")
        return False

    end = dt.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    output_dir = Path(output_dir)
    print(f"SST: {start.date()} → {end.date()} ({days} days), N={n_particles}")
    try:
        download_and_process_sst(
            start,
            end,
            n_particles=n_particles,
            output_dir=output_dir,
            use_alternative=False,
            ensure_cyclicality=False,
        )
    except Exception as e:
        print(f"[WARN] NOAA download failed ({e}); using synthetic SST-like series.")
        download_and_process_sst(
            start,
            end,
            n_particles=n_particles,
            output_dir=output_dir,
            use_alternative=True,
            ensure_cyclicality=False,
        )
    p = output_dir / "particles.npy"
    if not p.is_file():
        print("[ERROR] SST export missing particles.npy")
        return False
    print(f"\n[OK] SST data ready: {p}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Download GOES / SST data for LOT pipelines")
    ap.add_argument(
        "--source",
        choices=["goes", "sst", "both"],
        default="both",
        help="Which dataset to fetch (default: both)",
    )
    ap.add_argument(
        "--output-goes",
        type=Path,
        default=REPO_ROOT / "goes_data",
        help="Directory for GOES particles.npy",
    )
    ap.add_argument(
        "--output-sst",
        type=Path,
        default=REPO_ROOT / "sst_data",
        help="Directory for SST particles.npy",
    )
    ap.add_argument("--hours", type=int, default=6, help="GOES: duration in hours (default 6)")
    ap.add_argument("--days", type=int, default=60, help="SST: day span (default 60)")
    ap.add_argument("--n-particles", type=int, default=500, help="Particles per frame")
    ap.add_argument(
        "--date",
        default="2024-07-15",
        help="GOES: UTC date YYYY-MM-DD (historical CONUS weather)",
    )
    ap.add_argument("--hour-utc", type=int, default=18, help="GOES: start hour UTC")
    args = ap.parse_args()

    ok = True
    if args.source in ("goes", "both"):
        g = download_goes_bundle(
            args.output_goes,
            hours=args.hours,
            n_particles=args.n_particles,
            date=args.date,
            hour_utc=args.hour_utc,
        )
        ok = ok and g
    if args.source in ("sst", "both"):
        s = download_sst_bundle(
            args.output_sst,
            days=args.days,
            n_particles=args.n_particles,
        )
        ok = ok and s

    if ok:
        print("\nNext steps:")
        print(
            f"  python scripts/realworld_experiment_pipeline.py "
            f"--goes-dir {args.output_goes} --N {args.n_particles}"
        )
        print("  (or --goes-dir sst_data after SST-only download)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
