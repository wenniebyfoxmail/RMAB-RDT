"""
Regime Map: Whittle vs Myopic Policy Dominance Boundary
=========================================================

This script generates the regime map showing where Whittle outperforms Myopic
and vice versa, as required by advisor feedback (P1 task).

Output: Fig4 - Heatmap of (p_s × M/N) showing policy gap

Key insight: In high-reliability (p_s→1) regimes, Myopic is near-optimal;
Whittle's advantage emerges in resource-constrained, unreliable settings.

Parallel execution: Auto-detects CPU cores for faster computation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Fix multiprocessing for Colab/Jupyter
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

from concurrent.futures import ProcessPoolExecutor, as_completed

def setup_ieee_style():
    """Configure matplotlib for IEEE publication format."""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.figsize': (3.5, 3.0),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def run_single_config(N: int, M: int, p_s: float, T: int, 
                      n_seeds: int, delta_max: int = 100,
                      verbose: bool = False) -> Dict[str, float]:
    """
    Run experiment for a single (N, M, p_s) configuration.
    
    Returns:
        Dict with 'whittle_aoii', 'myopic_aoii', 'gap_pct'
    """
    # Create config
    config = SimulationConfig()
    config.experiment.N = N
    config.experiment.M = M
    config.experiment.delta_max = delta_max
    config.arm_classes = get_nhgp_arm_classes(J=5, R=8)
    
    # Override p_s for all arm classes
    for ac in config.arm_classes:
        ac.p_s = p_s
    
    # Compute Whittle tables
    solver = WhittleSolver(config.whittle)
    index_tables = solver.compute_all_tables(config.arm_classes, delta_max, verbose=False)
    
    # Create policies
    whittle_policy = WhittlePolicy(index_tables)
    myopic_policy = MyopicPolicy()
    
    burn_in_ratio = 0.5
    seeds = list(range(42, 42 + n_seeds))
    
    results = {'Whittle': [], 'Myopic': []}
    
    for seed in seeds:
        for name, policy in [('Whittle', whittle_policy), ('Myopic', myopic_policy)]:
            env = RMABEnvironment(config, seed=seed)
            env.reset(seed=seed)
            
            trajectory = []
            for t in range(T):
                obs = env._get_observations()
                actions = policy.select_arms(obs, env)
                result = env.step(actions)
                trajectory.append(result.info['mean_oracle_aoii'])
            
            # Evaluate on last 50%
            burn_in = int(T * burn_in_ratio)
            eval_aoii = np.mean(trajectory[burn_in:])
            results[name].append(eval_aoii)
    
    whittle_mean = np.mean(results['Whittle'])
    myopic_mean = np.mean(results['Myopic'])
    
    # Gap: positive means Whittle is better (lower AoII)
    gap_pct = (myopic_mean - whittle_mean) / whittle_mean * 100 if whittle_mean > 0 else 0
    
    if verbose:
        print(f"  N={N}, M={M}, p_s={p_s:.2f}: Whittle={whittle_mean:.3f}, "
              f"Myopic={myopic_mean:.3f}, Gap={gap_pct:+.1f}%")
    
    return {
        'whittle_aoii': whittle_mean,
        'myopic_aoii': myopic_mean,
        'gap_pct': gap_pct,
        'N': N,
        'M': M,
        'p_s': p_s
    }


def _run_config_wrapper(args: tuple) -> Dict[str, float]:
    """Wrapper for parallel execution."""
    N, M, p_s, T, n_seeds = args
    return run_single_config(N, M, p_s, T, n_seeds, verbose=False)


def generate_regime_map(output_dir: str = "results",
                        N: int = 50,
                        T: int = 1000,
                        n_seeds: int = 5,
                        quick_test: bool = False):
    """
    Generate regime map: p_s × (M/N) heatmap.
    
    Args:
        output_dir: Output directory
        N: Number of arms
        T: Simulation horizon
        n_seeds: Number of random seeds
        quick_test: If True, use reduced grid
    """
    setup_ieee_style()
    
    Path(f"{output_dir}/data").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
    
    # Define sweep grid
    if quick_test:
        ps_values = [0.70, 0.90, 0.996]
        budget_ratios = [0.04, 0.10, 0.20]
        T = 500
        n_seeds = 3
    else:
        ps_values = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.996]
        budget_ratios = [0.02, 0.04, 0.06, 0.10, 0.15, 0.20]
    
    print("=" * 60)
    print("REGIME MAP: Whittle vs Myopic Policy Dominance")
    print("=" * 60)
    print(f"N={N}, T={T}, seeds={n_seeds}")
    print(f"p_s grid: {ps_values}")
    print(f"M/N grid: {budget_ratios}")
    
    # Determine parallelization
    total_configs = len(ps_values) * len(budget_ratios)
    n_workers = get_optimal_workers(total_configs)
    print(f"Parallel execution: {n_workers} workers, {total_configs} configs")
    print("=" * 60)
    
    # Build configuration list
    config_list = []
    config_indices = []  # Store (i, j) for each config
    for i, p_s in enumerate(ps_values):
        for j, ratio in enumerate(budget_ratios):
            M = max(1, int(N * ratio))
            config_list.append((N, M, p_s, T, n_seeds))
            config_indices.append((i, j, p_s, ratio, M))
    
    start_time = time.time()
    
    # Run in parallel
    if n_workers > 1:
        print(f"\nRunning {total_configs} configurations in parallel...")
        all_results = parallel_map(_run_config_wrapper, config_list, n_workers=n_workers)
    else:
        print(f"\nRunning {total_configs} configurations sequentially...")
        all_results = [_run_config_wrapper(c) for c in config_list]
    
    # Process results
    results_data = []
    gap_matrix = np.zeros((len(ps_values), len(budget_ratios)))
    
    for idx, (result, (i, j, p_s, ratio, M)) in enumerate(zip(all_results, config_indices)):
        gap_matrix[i, j] = result['gap_pct']
        results_data.append({
            'p_s': p_s,
            'budget_ratio': ratio,
            'M': M,
            'N': N,
            'whittle_aoii': result['whittle_aoii'],
            'myopic_aoii': result['myopic_aoii'],
            'gap_pct': result['gap_pct']
        })
        print(f"  [{idx+1}/{total_configs}] p_s={p_s:.2f}, M/N={ratio:.0%}: gap={result['gap_pct']:+.1f}%")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    
    # Save data
    df = pd.DataFrame(results_data)
    df.to_csv(f"{output_dir}/data/fig4_regime_map.csv", index=False)
    print(f"Saved: {output_dir}/data/fig4_regime_map.csv")
    
    # Generate heatmap
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    
    # Custom colormap: red (Myopic better) -> white (tie) -> blue (Whittle better)
    cmap = plt.cm.RdBu
    
    # Determine symmetric color scale
    max_abs = max(abs(gap_matrix.min()), abs(gap_matrix.max()), 5)
    
    im = ax.imshow(gap_matrix, aspect='auto', cmap=cmap, 
                   vmin=-max_abs, vmax=max_abs, origin='lower')
    
    # Labels
    ax.set_xticks(range(len(budget_ratios)))
    ax.set_xticklabels([f'{r:.0%}' for r in budget_ratios])
    ax.set_yticks(range(len(ps_values)))
    ax.set_yticklabels([f'{p:.2f}' for p in ps_values])
    
    ax.set_xlabel('Budget Ratio $M/N$')
    ax.set_ylabel('Success Probability $p_s$')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Gap (%) = (Myopic - Whittle) / Whittle')
    
    # Annotate cells
    for i in range(len(ps_values)):
        for j in range(len(budget_ratios)):
            val = gap_matrix[i, j]
            color = 'white' if abs(val) > max_abs * 0.5 else 'black'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center', 
                    fontsize=6, color=color)
    
    # Add interpretation annotation
    ax.text(0.02, 0.98, 'Blue: Whittle better\nRed: Myopic better',
            transform=ax.transAxes, fontsize=6, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    for ext in ['pdf', 'png']:
        fig.savefig(f"{output_dir}/figures/fig4_regime_map.{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/figures/fig4_regime_map.pdf/png")
    
    plt.close(fig)
    
    # Print summary
    print("\n" + "=" * 60)
    print("REGIME MAP SUMMARY")
    print("=" * 60)
    print(f"Whittle dominance (gap > 2%): {(gap_matrix > 2).sum()} cells")
    print(f"Near-tie (|gap| < 2%): {(np.abs(gap_matrix) < 2).sum()} cells")
    print(f"Myopic better (gap < -2%): {(gap_matrix < -2).sum()} cells")
    
    # Find max advantages
    max_idx = np.unravel_index(gap_matrix.argmax(), gap_matrix.shape)
    min_idx = np.unravel_index(gap_matrix.argmin(), gap_matrix.shape)
    print(f"\nMax Whittle advantage: {gap_matrix.max():+.1f}% at p_s={ps_values[max_idx[0]]:.2f}, M/N={budget_ratios[max_idx[1]]:.0%}")
    print(f"Max Myopic advantage: {gap_matrix.min():+.1f}% at p_s={ps_values[min_idx[0]]:.2f}, M/N={budget_ratios[min_idx[1]]:.0%}")
    
    return df, gap_matrix


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Regime Map (Fig4)')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--N', type=int, default=50, help='Number of arms')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: auto)')
    args = parser.parse_args()
    
    generate_regime_map(
        output_dir=args.output,
        N=args.N,
        quick_test=args.quick
    )
