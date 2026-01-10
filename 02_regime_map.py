"""
Regime Map: Whittle vs Myopic Policy Dominance (v3 - Heterogeneous)
====================================================================

UPDATED v3: Sweeps over heterogeneity levels instead of fixed p_s.
This reveals the TRUE boundary conditions for Whittle advantage.

Key insight: Whittle advantage emerges when:
1. Channel heterogeneity is HIGH (σ(p_s) > 0.2)
2. Budget is tight (M/N ≤ 10%)

Output: Fig4 - Heatmap of (heterogeneity × M/N) showing policy gap
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import time
from copy import deepcopy

from config import (
    SimulationConfig, get_nhgp_arm_classes, HeterogeneousConfig,
    generate_heterogeneous_p_s, compute_tail_metrics
)
from environment import RMABEnvironment
from whittle_solver import WhittleSolver
from policies import WhittlePolicy, MyopicPolicy, MaxAgePolicy, WorstStatePolicy, RandomPolicy
from parallel_utils import get_optimal_workers, parallel_map, get_cpu_count


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
        'figure.figsize': (4.5, 3.5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def run_single_config(N: int, M: int, heterogeneity: str, T: int, 
                      n_seeds: int, delta_max: int = 100,
                      verbose: bool = False) -> Dict[str, float]:
    """
    Run experiment for a single (N, M, heterogeneity) configuration.
    
    Uses per-arm heterogeneous p_s automatically.
    """
    # Create config with heterogeneity setting
    config = SimulationConfig()
    config.experiment.N = N
    config.experiment.M = M
    config.experiment.delta_max = delta_max
    config.experiment.heterogeneous.level = heterogeneity
    config.arm_classes = get_nhgp_arm_classes(J=5, R=8, recovery_prob=0.02)  # Required for non-zero AoII

    burn_in_ratio = 0.3
    seeds = list(range(42, 42 + n_seeds))

    results = {
        'Whittle': [], 'Myopic': [], 'MaxAge': [],
        'WorstState': [], 'Random': [],
        'Whittle_p95': [], 'Myopic_p95': []
    }

    for seed in seeds:
        # Generate p_s for this seed
        p_s_per_arm = generate_heterogeneous_p_s(N, config.experiment.heterogeneous, seed)

        # Compute Whittle tables with heterogeneous support
        solver = WhittleSolver(config.whittle)
        index_tables = solver.compute_all_tables(config.arm_classes, delta_max, verbose=False)

        # Create policies
        whittle_policy = WhittlePolicy(
            index_tables=index_tables,
            arm_classes=config.arm_classes,
            p_s_per_arm=p_s_per_arm,
            delta_max=delta_max,
            whittle_config=config.whittle
        )
        myopic_policy = MyopicPolicy()
        maxage_policy = MaxAgePolicy()
        worststate_policy = WorstStatePolicy()
        random_policy = RandomPolicy()

        policies = [
            ('Whittle', whittle_policy),
            ('Myopic', myopic_policy),
            ('MaxAge', maxage_policy),
            ('WorstState', worststate_policy),
            ('Random', random_policy),
        ]

        for name, policy in policies:
            env = RMABEnvironment(config, seed=seed)
            env.reset(seed=seed)

            trajectory = []
            for t in range(T):
                obs = env._get_observations()
                actions = policy.select_arms(obs, env)
                result = env.step(actions)
                trajectory.append(result.info['mean_oracle_aoii'])

            burn_in = int(T * burn_in_ratio)
            eval_trajectory = trajectory[burn_in:]
            eval_aoii = np.mean(eval_trajectory)
            results[name].append(eval_aoii)

            # P95 for Whittle and Myopic
            if name in ['Whittle', 'Myopic']:
                p95 = np.percentile(eval_trajectory, 95)
                results[f'{name}_p95'].append(p95)

    whittle_mean = np.mean(results['Whittle'])
    myopic_mean = np.mean(results['Myopic'])

    gap_pct = (myopic_mean - whittle_mean) / myopic_mean * 100 if myopic_mean > 0 else 0

    # P95 gap
    whittle_p95 = np.mean(results['Whittle_p95'])
    myopic_p95 = np.mean(results['Myopic_p95'])
    p95_gap = (myopic_p95 - whittle_p95) / myopic_p95 * 100 if myopic_p95 > 0 else 0

    if verbose:
        print(f"  N={N}, M={M}, het={heterogeneity}: Whittle={whittle_mean:.3f}, "
              f"Myopic={myopic_mean:.3f}, Gap={gap_pct:+.1f}%")

    return {
        'whittle_aoii': whittle_mean,
        'myopic_aoii': myopic_mean,
        'maxage_aoii': np.mean(results['MaxAge']),
        'worststate_aoii': np.mean(results['WorstState']),
        'random_aoii': np.mean(results['Random']),
        'gap_pct': gap_pct,
        'whittle_p95': whittle_p95,
        'myopic_p95': myopic_p95,
        'p95_gap': p95_gap,
        'N': N,
        'M': M,
        'heterogeneity': heterogeneity
    }


def _run_config_wrapper(args: tuple) -> Dict[str, float]:
    """Wrapper for parallel execution."""
    N, M, heterogeneity, T, n_seeds = args
    return run_single_config(N, M, heterogeneity, T, n_seeds, verbose=False)


def generate_regime_map(output_dir: str = "results",
                        N: int = 80,            # 改为 80 以匹配 Fig 2
                        T: int = 500,
                        n_seeds: int = 5,
                        quick_test: bool = False):
    """
    Generate regime map: heterogeneity × (M/N) heatmap.
    """
    setup_ieee_style()

    Path(f"{output_dir}/data").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)

    # Define sweep grid
    if quick_test:
        heterogeneities = ["homogeneous", "medium", "high"]
        budget_ratios = [0.02, 0.05, 0.10]     # 聚焦紧预算
        T = 300
        n_seeds = 3
    else:
        heterogeneities = ["homogeneous", "low", "medium", "high"]
        budget_ratios = [0.01, 0.02, 0.05, 0.10, 0.15]  # 增加 1% 极紧预算

    print("=" * 70)
    print("REGIME MAP v3: Whittle vs Myopic (Heterogeneous p_s)")
    print("=" * 70)
    print(f"N={N}, T={T}, seeds={n_seeds}")
    print(f"Heterogeneity grid: {heterogeneities}")
    print(f"M/N grid: {budget_ratios}")

    total_configs = len(heterogeneities) * len(budget_ratios)
    n_workers = get_optimal_workers(total_configs)
    print(f"Parallel execution: {n_workers} workers, {total_configs} configs")
    print("=" * 70)

    # Build configuration list
    config_list = []
    config_indices = []
    for i, het in enumerate(heterogeneities):
        for j, ratio in enumerate(budget_ratios):
            M = max(1, int(N * ratio))
            config_list.append((N, M, het, T, n_seeds))
            config_indices.append((i, j, het, ratio, M))

    start_time = time.time()

    # Run
    if n_workers > 1:
        print(f"\nRunning {total_configs} configurations in parallel...")
        all_results = parallel_map(_run_config_wrapper, config_list, n_workers=n_workers)
    else:
        print(f"\nRunning {total_configs} configurations sequentially...")
        all_results = []
        for idx, c in enumerate(config_list):
            print(f"  [{idx+1}/{total_configs}] het={c[2]}, M/N={c[1]/c[0]:.0%}")
            all_results.append(_run_config_wrapper(c))

    # Process results
    results_data = []
    gap_matrix = np.zeros((len(heterogeneities), len(budget_ratios)))

    for idx, (result, (i, j, het, ratio, M)) in enumerate(zip(all_results, config_indices)):
        gap_matrix[i, j] = result['gap_pct']
        results_data.append({
            'heterogeneity': het,
            'budget_ratio': ratio,
            'M': M,
            'N': N,
            'whittle_aoii': result['whittle_aoii'],
            'myopic_aoii': result['myopic_aoii'],
            'maxage_aoii': result['maxage_aoii'],
            'worststate_aoii': result['worststate_aoii'],
            'random_aoii': result['random_aoii'],
            'gap_pct': result['gap_pct'],
            'whittle_p95': result['whittle_p95'],
            'myopic_p95': result['myopic_p95'],
            'p95_gap': result['p95_gap'],
        })
        print(f"  [{idx+1}/{total_configs}] het={het}, M/N={ratio:.0%}: gap={result['gap_pct']:+.1f}%")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Save data
    df = pd.DataFrame(results_data)
    df.to_csv(f"{output_dir}/data/fig4_regime_map.csv", index=False)
    print(f"Saved: {output_dir}/data/fig4_regime_map.csv")

    # Generate heatmap
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    cmap = plt.cm.RdBu
    max_abs = max(abs(gap_matrix.min()), abs(gap_matrix.max()), 5)

    im = ax.imshow(gap_matrix, aspect='auto', cmap=cmap,
                   vmin=-max_abs, vmax=max_abs, origin='lower')

    ax.set_xticks(range(len(budget_ratios)))
    ax.set_xticklabels([f'{r:.0%}' for r in budget_ratios])
    ax.set_yticks(range(len(heterogeneities)))
    ax.set_yticklabels(heterogeneities)

    ax.set_xlabel('Budget Ratio $M/N$')
    ax.set_ylabel('Channel Heterogeneity Level')
    ax.set_title('Whittle Advantage over Myopic (%)')

    cbar = plt.colorbar(im, ax=ax, label='Gap (%) = (Myopic - Whittle) / Myopic')

    # Annotate cells
    for i in range(len(heterogeneities)):
        for j in range(len(budget_ratios)):
            val = gap_matrix[i, j]
            color = 'white' if abs(val) > max_abs * 0.5 else 'black'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold' if val > 10 else 'normal')

    ax.text(0.02, 0.98, 'Blue: Whittle better\nRed: Myopic better',
            transform=ax.transAxes, fontsize=6, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(f"{output_dir}/figures/fig4_regime_map.{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/figures/fig4_regime_map.pdf/png")

    plt.close(fig)

    # Print summary
    print("\n" + "=" * 70)
    print("REGIME MAP SUMMARY")
    print("=" * 70)
    print(f"Whittle dominance (gap > 5%): {(gap_matrix > 5).sum()} cells")
    print(f"Strong Whittle (gap > 10%): {(gap_matrix > 10).sum()} cells")
    print(f"Near-tie (|gap| < 3%): {(np.abs(gap_matrix) < 3).sum()} cells")

    max_idx = np.unravel_index(gap_matrix.argmax(), gap_matrix.shape)
    print(f"\n✅ BEST CONFIG: het={heterogeneities[max_idx[0]]}, M/N={budget_ratios[max_idx[1]]:.0%}")
    print(f"   Whittle advantage: {gap_matrix.max():+.1f}%")

    # Policy ranking table
    print("\n" + "=" * 70)
    print("POLICY RANKING (Mean AoII - lower is better)")
    print("=" * 70)
    print(f"{'Het':<12} {'M/N':<6} {'Whittle':<9} {'Myopic':<9} {'MaxAge':<9} {'WorstSt':<9} {'Random':<9}")
    print("-" * 70)
    for r in sorted(results_data, key=lambda x: -x['gap_pct'])[:5]:
        print(f"{r['heterogeneity']:<12} {r['budget_ratio']:.0%:<6} "
              f"{r['whittle_aoii']:<9.2f} {r['myopic_aoii']:<9.2f} "
              f"{r['maxage_aoii']:<9.2f} {r['worststate_aoii']:<9.2f} "
              f"{r['random_aoii']:<9.2f}")

    return df, gap_matrix


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Regime Map (Fig4) - v3 Heterogeneous')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--N', type=int, default=50, help='Number of arms')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    generate_regime_map(
        output_dir=args.output,
        N=args.N,
        quick_test=args.quick
    )