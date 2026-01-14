"""
Regime Map: Whittle vs Myopic Policy Dominance (v4 - Fixed p_s Matching)
========================================================================

CRITICAL FIX v4:
- Use arm_class.p_s for BOTH Whittle index computation AND environment
- Disable environment-level heterogeneous to prevent p_s mismatch
- Heterogeneity is achieved via different arm_class.p_s values

Key insight: Whittle advantage emerges when:
1. Budget is tight (M/N <= 5%)
2. p_s values are correctly matched between index tables and environment

Output: Fig4 - Heatmap of (p_s heterogeneity x M/N) showing policy gap
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from typing import Dict, List
from pathlib import Path
import time

from config import SimulationConfig, get_nhgp_arm_classes, ArmClassConfig
from environment import RMABEnvironment
from whittle_solver import WhittleSolver
from policies import WhittlePolicy, MyopicPolicy, MaxAgePolicy, WorstStatePolicy, RandomPolicy
from parallel_utils import get_optimal_workers, parallel_map


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


def create_arm_classes_with_heterogeneity(level: str, J: int = 5, R: int = 8,
                                          recovery_prob: float = 0.02) -> List[ArmClassConfig]:
    """
    Create arm classes with specific p_s heterogeneity levels.

    This is the CORRECT way to achieve heterogeneity:
    - Different arm classes have different p_s values
    - Environment uses these p_s values directly (no random generation)
    - Whittle index tables are computed for each p_s
    """
    p_s_configs = {
        'homogeneous': [0.90, 0.90],
        'low':         [0.80, 0.95],
        'medium':      [0.60, 0.90],
        'high':        [0.30, 0.70],
    }

    p_s_values = p_s_configs.get(level, [0.90, 0.90])
    base_classes = get_nhgp_arm_classes(J=J, R=R, recovery_prob=recovery_prob, heterogeneous=False)

    for i, ac in enumerate(base_classes):
        ac.p_s = p_s_values[i % len(p_s_values)]

    return base_classes


def run_single_config(N: int, M: int, heterogeneity: str, T: int,
                      n_seeds: int, delta_max: int = 100,
                      verbose: bool = False) -> Dict[str, float]:
    """Run experiment for a single configuration with FIXED p_s matching."""
    config = SimulationConfig()
    config.experiment.N = N
    config.experiment.M = M
    config.experiment.delta_max = delta_max
    config.experiment.heterogeneous.enabled = False  # CRITICAL FIX
    config.arm_classes = create_arm_classes_with_heterogeneity(heterogeneity, J=5, R=8, recovery_prob=0.02)

    burn_in_ratio = 0.5
    seeds = list(range(42, 42 + n_seeds))

    results = {'Whittle': [], 'Myopic': [], 'MaxAge': [], 'WorstState': [], 'Random': [],
               'Whittle_p95': [], 'Myopic_p95': []}

    solver = WhittleSolver(config.whittle)
    index_tables = solver.compute_all_tables(config.arm_classes, delta_max, verbose=False)

    whittle_policy = WhittlePolicy(index_tables)
    myopic_policy = MyopicPolicy()
    maxage_policy = MaxAgePolicy()
    worststate_policy = WorstStatePolicy()
    random_policy = RandomPolicy()

    policies = [('Whittle', whittle_policy), ('Myopic', myopic_policy),
                ('MaxAge', maxage_policy), ('WorstState', worststate_policy), ('Random', random_policy)]

    for seed in seeds:
        for name, policy in policies:
            env = RMABEnvironment(config, seed=seed)
            env.reset(seed=seed)
            policy.reset()

            trajectory = []
            for t in range(T):
                obs = env._get_observations()
                actions = policy.select_arms(obs, env)
                result = env.step(actions)
                trajectory.append(result.info['mean_oracle_aoii'])

            burn_in = int(T * burn_in_ratio)
            eval_trajectory = trajectory[burn_in:]
            results[name].append(np.mean(eval_trajectory))

            if name in ['Whittle', 'Myopic']:
                results[f'{name}_p95'].append(np.percentile(eval_trajectory, 95))

    whittle_mean = np.mean(results['Whittle'])
    myopic_mean = np.mean(results['Myopic'])
    gap_pct = (myopic_mean - whittle_mean) / myopic_mean * 100 if myopic_mean > 0 else 0

    whittle_p95 = np.mean(results['Whittle_p95'])
    myopic_p95 = np.mean(results['Myopic_p95'])
    p95_gap = (myopic_p95 - whittle_p95) / myopic_p95 * 100 if myopic_p95 > 0 else 0

    return {
        'whittle_aoii': whittle_mean, 'whittle_std': np.std(results['Whittle']),
        'myopic_aoii': myopic_mean, 'myopic_std': np.std(results['Myopic']),
        'maxage_aoii': np.mean(results['MaxAge']),
        'worststate_aoii': np.mean(results['WorstState']),
        'random_aoii': np.mean(results['Random']),
        'gap_pct': gap_pct, 'whittle_p95': whittle_p95, 'myopic_p95': myopic_p95, 'p95_gap': p95_gap,
        'N': N, 'M': M, 'heterogeneity': heterogeneity
    }


def _run_config_wrapper(args: tuple) -> Dict[str, float]:
    N, M, heterogeneity, T, n_seeds, delta_max = args
    return run_single_config(N, M, heterogeneity, T, n_seeds, delta_max, verbose=False)


def generate_regime_map(output_dir: str = "results", N: int = 80, T: int = 500,
                        n_seeds: int = 5, delta_max: int = 100,
                        quick_test: bool = False, n_workers: int = None):
    """Generate regime map with FIXED p_s matching."""
    setup_ieee_style()

    Path(f"{output_dir}/data").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)

    if quick_test:
        heterogeneities = ["homogeneous", "medium", "high"]
        budget_ratios = [0.01, 0.02, 0.05, 0.10]
        T, n_seeds = 300, 3
    else:
        heterogeneities = ["homogeneous", "low", "medium", "high"]
        budget_ratios = [0.01, 0.02, 0.05, 0.10, 0.15]

    print("=" * 70)
    print("REGIME MAP v4: Whittle vs Myopic (Fixed p_s Matching)")
    print("=" * 70)
    print(f"N={N}, T={T}, seeds={n_seeds}, delta_max={delta_max}")
    print(f"Heterogeneity grid: {heterogeneities}")
    print(f"M/N grid: {budget_ratios}")
    print(f"Key fix: heterogeneous.enabled=False, using arm_class.p_s")

    total_configs = len(heterogeneities) * len(budget_ratios)

    if n_workers is None:
        n_workers = get_optimal_workers(total_configs)
    else:
        n_workers = max(1, int(n_workers))

    print(f"Parallel execution: {n_workers} workers, {total_configs} configs")
    print("=" * 70)

    config_list = []
    config_indices = []
    for i, het in enumerate(heterogeneities):
        for j, ratio in enumerate(budget_ratios):
            M = max(1, int(N * ratio))
            config_list.append((N, M, het, T, n_seeds, delta_max))
            config_indices.append((i, j, het, ratio, M))

    start_time = time.time()

    if n_workers > 1:
        print(f"\nRunning {total_configs} configurations in parallel...")
        all_results = parallel_map(_run_config_wrapper, config_list, n_workers=n_workers)
    else:
        print(f"\nRunning {total_configs} configurations sequentially...")
        all_results = []
        for idx, c in enumerate(config_list):
            het, ratio, M = config_indices[idx][2], config_indices[idx][3], config_indices[idx][4]
            print(f"  [{idx+1}/{total_configs}] het={het}, M/N={ratio:.0%}, M={M}")
            result = _run_config_wrapper(c)
            all_results.append(result)
            print(f"    -> Whittle={result['whittle_aoii']:.1f}, Myopic={result['myopic_aoii']:.1f}, Gap={result['gap_pct']:+.1f}%")

    results_data = []
    gap_matrix = np.zeros((len(heterogeneities), len(budget_ratios)))

    for idx, (result, (i, j, het, ratio, M)) in enumerate(zip(all_results, config_indices)):
        gap_matrix[i, j] = result['gap_pct']
        results_data.append({
            'heterogeneity': het, 'budget_ratio': ratio, 'M': M, 'N': N,
            'whittle_aoii': result['whittle_aoii'], 'whittle_std': result.get('whittle_std', 0),
            'myopic_aoii': result['myopic_aoii'], 'myopic_std': result.get('myopic_std', 0),
            'maxage_aoii': result['maxage_aoii'], 'worststate_aoii': result['worststate_aoii'],
            'random_aoii': result['random_aoii'], 'gap_pct': result['gap_pct'],
            'whittle_p95': result['whittle_p95'], 'myopic_p95': result['myopic_p95'], 'p95_gap': result['p95_gap'],
        })

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    df = pd.DataFrame(results_data)
    df.to_csv(f"{output_dir}/data/fig4_regime_map.csv", index=False)
    print(f"Saved: {output_dir}/data/fig4_regime_map.csv")

    # Generate heatmap
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    cmap = plt.cm.RdBu
    max_abs = max(abs(gap_matrix.min()), abs(gap_matrix.max()), 10)

    im = ax.imshow(gap_matrix, aspect='auto', cmap=cmap, vmin=-max_abs, vmax=max_abs, origin='lower')

    ax.set_xticks(range(len(budget_ratios)))
    ax.set_xticklabels([f'{r:.0%}' for r in budget_ratios])
    ax.set_yticks(range(len(heterogeneities)))
    ax.set_yticklabels(heterogeneities)

    ax.set_xlabel('Budget Ratio $M/N$')
    ax.set_ylabel('Channel Heterogeneity Level')
    ax.set_title('Whittle Advantage over Myopic (%)')

    plt.colorbar(im, ax=ax, label='Gap (%) = (Myopic - Whittle) / Myopic')

    for i in range(len(heterogeneities)):
        for j in range(len(budget_ratios)):
            val = gap_matrix[i, j]
            color = 'white' if abs(val) > max_abs * 0.5 else 'black'
            fontweight = 'bold' if val > 10 else 'normal'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center', fontsize=7, color=color, fontweight=fontweight)

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
    print(f"Whittle dominance (gap > 5%): {(gap_matrix > 5).sum()} / {gap_matrix.size} cells")
    print(f"Strong Whittle (gap > 10%): {(gap_matrix > 10).sum()} / {gap_matrix.size} cells")
    print(f"Strong Whittle (gap > 15%): {(gap_matrix > 15).sum()} / {gap_matrix.size} cells")

    max_idx = np.unravel_index(gap_matrix.argmax(), gap_matrix.shape)
    print(f"\nBEST CONFIG: het={heterogeneities[max_idx[0]]}, M/N={budget_ratios[max_idx[1]]:.0%}")
    print(f"   Whittle advantage: {gap_matrix.max():+.1f}%")

    print("\n" + "=" * 70)
    print("TOP 5 CONFIGS (by Whittle advantage)")
    print("=" * 70)
    print(f"{'Het':<12} {'M/N':<8} {'Whittle':<10} {'Myopic':<10} {'Gap':<10}")
    print("-" * 50)
    for r in sorted(results_data, key=lambda x: -x['gap_pct'])[:5]:
        ratio_str = f"{r['budget_ratio']*100:.0f}%"
        print(f"{r['heterogeneity']:<12} {ratio_str:<8} {r['whittle_aoii']:<10.2f} {r['myopic_aoii']:<10.2f} {r['gap_pct']:+.1f}%")

    return df, gap_matrix


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Regime Map (Fig4) - v4 Fixed')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--N', type=int, default=80, help='Number of arms')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (1 for Colab)')
    parser.add_argument('--delta-max', type=int, default=100, help='Maximum delta value')
    args = parser.parse_args()

    generate_regime_map(output_dir=args.output, N=args.N, delta_max=args.delta_max,
                        quick_test=args.quick, n_workers=args.workers)