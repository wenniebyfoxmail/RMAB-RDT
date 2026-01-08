#!/usr/bin/env python3
"""
Road DT AoII-ARD RMAB - Run All Experiments
===========================================
Unified entry point for all experiments.

Usage:
    python run_all.py --quick      # Quick test (~15 min)
    python run_all.py --full       # Full experiments (~1-2 hours)
"""

import subprocess
import sys
import os
import time
import multiprocessing as mp
from pathlib import Path


def get_cpu_info():
    """Get CPU information for parallel execution."""
    total_cores = mp.cpu_count()
    available_cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else total_cores
    recommended = max(1, min(available_cores - 1, 8))
    return total_cores, available_cores, recommended


def run_script(script_name, args, description):
    """Run a Python script with given arguments."""
    cmd = [sys.executable, script_name] + args
    print(f"\nCommand: python {script_name} {' '.join(args)}")
    print()

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✅ {description} completed ({elapsed:.1f}s)")
    else:
        print(f"\n❌ {description} FAILED (exit code {result.returncode})")

    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run all RMAB experiments')
    parser.add_argument('--quick', action='store_true', help='Quick mode (reduced parameters)')
    parser.add_argument('--full', action='store_true', help='Full experiments')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel execution')
    args = parser.parse_args()

    if not args.quick and not args.full:
        args.quick = True  # Default to quick

    # Get CPU info
    total_cores, available_cores, recommended = get_cpu_info()
    n_workers = args.workers if args.workers else recommended

    if args.no_parallel:
        n_workers = 1

    print("=" * 60)
    print("🔬 ROAD DT AoII-ARD RMAB EXPERIMENTS", "(QUICK MODE)" if args.quick else "(FULL MODE)")
    print("=" * 60)
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"CPU cores (total): {total_cores}")
    print(f"CPU cores (available): {available_cores}")
    print(f"Workers: {n_workers}")
    print("=" * 50)
    print()

    mode_args = ['--quick'] if args.quick else []
    output_args = ['--output', args.output]

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Define experiments with their specific arguments
    # Format: (script, description, supports_workers)
    experiments = [
        ('01_main_experiments.py', 'Main Experiments (Fig1-3 + Table1)', False),
        ('02_regime_map.py', 'Regime Map (Fig4)', True),
        ('03_time_varying.py', 'Time-Varying (Fig5)', True),
        ('04_indexability.py', 'Indexability Verification', False),
        ('05_noise_sensitivity.py', 'Noise Sensitivity', True),
        ('lp_comparison.py', 'LP Comparison (Fig6)', False),
        ('ltpp_calibration.py', 'LTPP Calibration', False),
    ]

    results = []
    total_start = time.time()

    for i, (script, description, supports_workers) in enumerate(experiments, 1):
        print()
        print("=" * 60)
        print(f"🚀 [{i:02d}] {description}")
        print("=" * 60)

        if not os.path.exists(script):
            print(f"⚠️ Script not found: {script}, skipping...")
            results.append((description, None))
            continue

        # Build arguments
        script_args = mode_args + output_args

        # Only add --workers for scripts that support it
        if supports_workers and n_workers > 1:
            script_args += ['--workers', str(n_workers)]

        success = run_script(script, script_args, description)
        results.append((description, success))

    # Summary
    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 60)

    for desc, success in results:
        if success is None:
            status = "⏭️ SKIPPED"
        elif success:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  {status}  {desc}")

    print()
    print(f"Total time: {total_elapsed / 60:.1f} minutes")
    print(f"Results saved to: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()