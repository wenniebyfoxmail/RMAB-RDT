#!/usr/bin/env python3
"""
Road DT AoII-ARD RMAB: One-Click Experiment Runner
===================================================

Implements all experiments with automatic parallel execution.

Script Execution Order:
  01_main_experiments.py  - P0: Fig1-3, Table1 (main results)
  02_regime_map.py        - P1: Fig4 (policy boundary)
  03_time_varying.py      - P1: Fig5 (seasonal variation)
  04_indexability.py      - P1: Appendix (verification)
  05_noise_sensitivity.py - P2: Appendix (Q_R channel)

Usage:
    python run_all.py --quick    # Quick test (~15-20 min)
    python run_all.py --full     # Full experiments (~1-1.5 hours with parallel)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from parallel_utils import print_system_info, get_optimal_workers, get_cpu_count


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print("\n" + "=" * 60)
    print(f"🚀 {description}")
    print("=" * 60)
    print(f"Command: {cmd}\n")
    
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"\n✅ {description} completed in {elapsed/60:.1f} min")
        return True
    else:
        print(f"\n❌ {description} FAILED (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all RMAB experiments')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (~15-20 min)')
    parser.add_argument('--full', action='store_true', help='Full experiments (~1-1.5 hours)')
    parser.add_argument('--skip-main', action='store_true', help='Skip 01_main_experiments')
    parser.add_argument('--skip-regime', action='store_true', help='Skip 02_regime_map')
    parser.add_argument('--skip-time', action='store_true', help='Skip 03_time_varying')
    parser.add_argument('--skip-index', action='store_true', help='Skip 04_indexability')
    parser.add_argument('--skip-noise', action='store_true', help='Skip 05_noise_sensitivity')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: auto)')
    args = parser.parse_args()
    
    if not args.quick and not args.full:
        print("Please specify --quick or --full mode")
        print("  --quick: Fast validation (~15-20 min)")
        print("  --full:  Complete experiments (~1-1.5 hours with parallel)")
        sys.exit(1)
    
    mode = "quick" if args.quick else "full"
    output_dir = args.output
    quick_flag = "--quick" if args.quick else ""
    
    # Determine workers
    n_workers = args.workers if args.workers else get_optimal_workers()
    worker_flag = f"--workers {n_workers}"
    
    # Create output directories
    Path(f"{output_dir}/data").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/indexability").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"🔬 ROAD DT AoII-ARD RMAB EXPERIMENTS ({mode.upper()} MODE)")
    print("=" * 60)
    
    # Print system info
    print_system_info()
    
    if args.quick:
        print("\n⏱️  Estimated time: 15-20 minutes (with parallel)")
    else:
        print("\n⏱️  Estimated time: 1-1.5 hours (with parallel)")
    
    print(f"📁 Output directory: {output_dir}/")
    print(f"🔧 Parallel workers: {n_workers}")
    
    print("\n📋 Execution Order:")
    print("  [01] Main Experiments: Fig1-3, Table1")
    print("  [02] Regime Map: Fig4 (Whittle vs Myopic)")
    print("  [03] Time-Varying: Fig5 (seasonal)")
    print("  [04] Indexability: Appendix verification")
    print("  [05] Noise Sensitivity: Appendix Q_R")
    print("  [06] LP Comparison: Fig6 (Whittle vs LP + wall-clock) ⭐")
    print("  [07] LTPP Calibration: Parameter validation ⭐")
    print("=" * 60)
    
    total_start = time.time()
    results = []
    
    # ========== 01: Main Experiments ==========
    if not args.skip_main:
        cmd = f"python 01_main_experiments.py {quick_flag} {worker_flag} --output {output_dir}"
        success = run_command(cmd, "[01] Main Experiments (Fig1-3 + Table1)")
        results.append(("01_main_experiments", success))
    
    # ========== 02: Regime Map ==========
    if not args.skip_regime:
        cmd = f"python 02_regime_map.py {quick_flag} {worker_flag} --output {output_dir}"
        success = run_command(cmd, "[02] Regime Map (Fig4)")
        results.append(("02_regime_map", success))
    
    # ========== 03: Time-Varying ==========
    if not args.skip_time:
        cmd = f"python 03_time_varying.py {quick_flag} {worker_flag} --output {output_dir}"
        success = run_command(cmd, "[03] Time-Varying (Fig5)")
        results.append(("03_time_varying", success))
    
    # ========== 04: Indexability ==========
    if not args.skip_index:
        cmd = f"python 04_indexability.py --output {output_dir}/indexability"
        success = run_command(cmd, "[04] Indexability Verification")
        results.append(("04_indexability", success))
    
    # ========== 05: Noise Sensitivity ==========
    if not args.skip_noise:
        cmd = f"python 05_noise_sensitivity.py {quick_flag} {worker_flag} --output {output_dir}"
        success = run_command(cmd, "[05] Noise Sensitivity")
        results.append(("05_noise_sensitivity", success))
    
    # ========== 06: LP Comparison ==========
    cmd = f"python lp_comparison.py {quick_flag} --output {output_dir}"
    success = run_command(cmd, "[06] LP vs Whittle Comparison (Fig6)")
    results.append(("06_lp_comparison", success))
    
    # ========== 07: LTPP Calibration ==========
    cmd = f"python ltpp_calibration.py --output {output_dir}"
    success = run_command(cmd, "[07] LTPP Calibration Validation")
    results.append(("07_ltpp_calibration", success))
    
    # ========== Summary ==========
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    all_success = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
        if not success:
            all_success = False
    
    print(f"\n⏱️  Total time: {total_elapsed/60:.1f} minutes")
    print(f"🔧 Workers used: {n_workers}")
    
    if all_success:
        print("\n" + "=" * 60)
        print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\n📁 Output files in {output_dir}/:")
        print("\n  [P0] Main Results (论文主图):")
        print("    data/fig1_n_sweep.csv")
        print("    data/fig2_m_sweep.csv")
        print("    data/fig3_ps_sweep.csv")
        print("    data/table1_summary.csv")
        print("    figures/fig1_*.pdf, fig2_*.pdf, fig3_*.pdf")
        print("\n  [P1] Novelty Evidence (顶刊防守):")
        print("    data/fig4_regime_map.csv, figures/fig4_regime_map.pdf")
        print("    data/fig5_time_varying.csv, figures/fig5_*.pdf")
        print("    data/fig6_lp_comparison.csv, figures/fig6_lp_comparison.pdf ⭐")
        print("    indexability/*.png")
        print("\n  [P2] Calibration & Appendix:")
        print("    data/ltpp_calibration.csv, figures/ltpp_calibration.pdf ⭐")
        print("    data/noise_sensitivity.csv, figures/noise_sensitivity.pdf")
    else:
        print("\n⚠️  Some experiments failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
