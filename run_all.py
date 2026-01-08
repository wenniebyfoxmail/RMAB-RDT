#!/usr/bin/env python3
"""
Road DT AoII-ARD RMAB - Run All Experiments
===========================================
Unified entry point for all experiments.

Usage:
    python run_all.py --quick      # Quick test (~15 min)
    python run_all.py --full       # Full experiments (~1-2 hours)
    python run_all.py --quick --no-parallel  # Safe mode for Colab
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def get_cpu_info():
    """Get CPU information for parallel execution."""
    try:
        import multiprocessing as mp
        total_cores = mp.cpu_count()
    except:
        total_cores = 2

    try:
        available_cores = len(os.sched_getaffinity(0))
    except:
        available_cores = total_cores

    recommended = max(1, min(available_cores - 1, 8))
    return total_cores, available_cores, recommended


def run_script(script_name, args, description):
    """Run a Python script with given arguments."""
    cmd = [sys.executable, script_name] + args
    cmd_str = f"python {script_name} {' '.join(args)}"
    print(f"\nCommand: {cmd_str}")
    print()

    start_time = time.time()
    try:
        result = subprocess.run(cmd, timeout=1800)  # 30 min timeout
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ {description} completed ({elapsed:.1f}s)")
            return True
        else:
            print(f"\n❌ {description} FAILED (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n⏱️ {description} TIMEOUT (>30min)")
        return False
    except Exception as e:
        print(f"\n❌ {description} ERROR: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run all RMAB experiments')
    parser.add_argument('--quick', action='store_true', help='Quick mode (reduced parameters)')
    parser.add_argument('--full', action='store_true', help='Full experiments')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel (safe for Colab)')
    args = parser.parse_args()

    if not args.quick and not args.full:
        args.quick = True

    # CPU info
    total_cores, available_cores, recommended = get_cpu_info()

    # Determine workers
    if args.no_parallel:
        n_workers = 1
    elif args.workers:
        n_workers = args.workers
    else:
        n_workers = recommended

    mode_str = "QUICK MODE" if args.quick else "FULL MODE"
    print("=" * 60)
    print(f"🔬 ROAD DT AoII-ARD RMAB EXPERIMENTS ({mode_str})")
    print("=" * 60)
    print(f"CPU cores: {total_cores} total, {available_cores} available")
    print(f"Workers: {n_workers}" + (" (parallel disabled)" if args.no_parallel else ""))
    print(f"Output: {args.output}/")
    print("=" * 60)

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Build common args
    mode_args = ['--quick'] if args.quick else []
    output_args = ['--output', args.output]
    worker_args = ['--workers', str(n_workers)] if n_workers > 1 else []

    # Define experiments: (script, description, supports_workers)
    experiments = [
        ('01_main_experiments.py', 'Main Experiments (Fig1-3, Table1)', False),
        ('02_regime_map.py', 'Regime Map (Fig4)', True),
        ('03_time_varying.py', 'Time-Varying (Fig5)', True),
        ('04_indexability.py', 'Indexability Verification', False),
        ('05_noise_sensitivity.py', 'Noise Sensitivity', True),
        ('lp_comparison.py', 'LP Comparison (Fig6)', False),
        ('ltpp_calibration.py', 'LTPP Calibration', False),
        ('ontario_calibration.py', 'Ontario Real Data Validation ⭐', False),  # 真实数据验证
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

        # Build script-specific arguments
        script_args = output_args.copy()  # Start with output args
        
        # Add mode args only for scripts that support it
        # ontario_calibration.py doesn't support --quick
        if script != 'ontario_calibration.py':
            script_args = mode_args + script_args

        # Only add --workers if script supports it AND parallel is enabled
        if supports_workers and n_workers > 1:
            script_args += worker_args
        
        # Special handling for ontario_calibration.py
        if script == 'ontario_calibration.py':
            # Check common data directory locations
            for data_dir in ['data/ontario', '../data/ontario', './data/ontario']:
                if os.path.exists(data_dir):
                    script_args += ['--data-dir', data_dir]
                    print(f"   Found Ontario data at: {data_dir}")
                    break
            else:
                print(f"   ⚠️ Ontario data directory not found, using default")

        success = run_script(script, script_args, description)
        results.append((description, success))

    # Summary
    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for desc, success in results:
        if success is None:
            status = "⏭️ SKIPPED"
            skipped += 1
        elif success:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        print(f"  {status}  {desc}")

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Total time: {total_elapsed / 60:.1f} minutes")
    print(f"Output directory: {args.output}/")
    print("=" * 60)

    # List outputs
    print("\n📁 Generated files:")
    for subdir in ['data', 'figures', 'indexability']:
        subpath = Path(args.output) / subdir
        if subpath.exists():
            files = list(subpath.glob('*'))
            if files:
                print(f"  {subdir}/")
                for f in files[:5]:
                    print(f"    - {f.name}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")


if __name__ == "__main__":
    main()