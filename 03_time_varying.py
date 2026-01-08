"""
Time-Nonhomogeneous RMAB Experiment
====================================

Implements seasonal degradation variation and windowed P̄ estimation,
as specified by advisor decision (Q1).

Key comparisons:
1. Fixed-P̄: Use initial P̄ throughout (baseline)
2. Windowed-P̄_w: Update every W epochs (practical)
3. Oracle-P(t): Know true P(t) at each step (upper bound)

Seasonal model: c(t) = c_0 * (1 + a * sin(2πt / T_season))
where a = 0.3 (±30% variation), T_season = 12 months
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import time
from copy import deepcopy

from config import SimulationConfig, ArmClassConfig, get_nhgp_arm_classes
from nhgp_builder import (
    NHGPParams, BinConfig, default_bin_config,
    compute_time_averaged_transition_matrix
)
from environment import RMABEnvironment, ArmState
from whittle_solver import WhittleSolver
from policies import WhittlePolicy, MyopicPolicy


def build_P_bar_with_c(J: int, c: float, b: float = 2.0) -> np.ndarray:
    """
    Build upper-triangular P̄ matrix with specified NHGP parameters.
    
    Args:
        J: Number of states
        c: Degradation rate coefficient
        b: Shape exponent
        
    Returns:
        P̄: J×J upper-triangular stochastic matrix
    """
    bins = default_bin_config(J=J)
    nhgp = NHGPParams(c=c, b=b)
    P = compute_time_averaged_transition_matrix(nhgp, bins, t_start=0.0, t_end=1.0, n_samples=5)
    
    # Ensure upper triangular
    P = np.triu(P)
    for j in range(J):
        row_sum = P[j, :].sum()
        if row_sum > 0:
            P[j, :] /= row_sum
        else:
            P[j, j] = 1.0
    
    return P


@dataclass
class SeasonalConfig:
    """Configuration for seasonal degradation variation."""
    T_season: int = 12          # Season period (epochs), 12 months = 1 year
    amplitude: float = 0.3      # ±30% variation in degradation rate
    window_size: int = 3        # Update P̄ every W epochs (quarterly)
    

def seasonal_c(t: int, c_0: float, config: SeasonalConfig) -> float:
    """
    Compute seasonal degradation rate coefficient.
    
    c(t) = c_0 * (1 + a * sin(2πt / T_season))
    
    Args:
        t: Current epoch (0-indexed)
        c_0: Base degradation rate
        config: Seasonal configuration
        
    Returns:
        Time-varying degradation rate c(t)
    """
    return c_0 * (1 + config.amplitude * np.sin(2 * np.pi * t / config.T_season))


def build_seasonal_transition_matrix(t: int, c_0: float, b: float, J: int,
                                     seasonal_config: SeasonalConfig) -> np.ndarray:
    """
    Build transition matrix with seasonal variation.
    
    Args:
        t: Current epoch
        c_0: Base degradation rate  
        b: Shape parameter (fixed)
        J: Number of states
        seasonal_config: Seasonal parameters
        
    Returns:
        P(t): J×J transition matrix at time t
    """
    c_t = seasonal_c(t, c_0, seasonal_config)
    
    # Use NHGP builder with seasonal c
    bins = default_bin_config(J=J)
    nhgp = NHGPParams(c=c_t, b=b)
    
    # Compute single-time transition matrix
    P = compute_time_averaged_transition_matrix(nhgp, bins, t_start=0.0, t_end=1.0, n_samples=5)
    
    # Ensure upper triangular (no recovery)
    P = np.triu(P)
    for j in range(J):
        row_sum = P[j, :].sum()
        if row_sum > 0:
            P[j, :] /= row_sum
        else:
            P[j, j] = 1.0
    
    return P


class SeasonalRMABEnvironment:
    """
    RMAB Environment with time-varying transition matrices.
    
    Extends base environment to support:
    - True P(t) that varies seasonally
    - Different P̄ estimation strategies (fixed/windowed/oracle)
    """
    
    def __init__(self, config: SimulationConfig, seasonal_config: SeasonalConfig,
                 seed: int = 42):
        self.config = config
        self.seasonal_config = seasonal_config
        self.rng = np.random.default_rng(seed)
        
        self.N = config.experiment.N
        self.M = config.experiment.M
        self.J = config.arm_classes[0].P_bar.shape[0]
        self.delta_max = config.experiment.delta_max
        
        # Store base parameters for each arm class
        self.arm_class_params = []
        for ac in config.arm_classes:
            # Extract c from the transition matrix (approximate)
            # For NHGP, c controls the speed of degradation
            self.arm_class_params.append({
                'name': ac.name,
                'c_0': 0.01 if 'slow' in ac.name.lower() else 0.02,  # Base rates
                'b': 2.0,
                'p_s': ac.p_s,
                'D': ac.D,
            })
        
        self.arms: List[Arm] = []
        self.t = 0
        
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset environment with new seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.t = 0
        self.arms = []
        
        # Assign arms to classes
        n_classes = len(self.arm_class_params)
        for i in range(self.N):
            class_idx = i % n_classes
            params = self.arm_class_params[class_idx]
            
            arm = ArmState(
                arm_id=i,
                class_name=params['name'],
                s_true=0,  # Start in best state
                h=0,
                delta=1,
                p_s=params['p_s'],
            )
            # Store parameters for P(t) computation
            arm.c_0 = params['c_0']
            arm.b = params['b']
            self.arms.append(arm)
    
    def get_true_P(self, arm: ArmState, t: int) -> np.ndarray:
        """Get true transition matrix P(t) for an arm at time t."""
        return build_seasonal_transition_matrix(
            t=t,
            c_0=arm.c_0,
            b=arm.b,
            J=self.J,
            seasonal_config=self.seasonal_config
        )
    
    def step(self, actions: np.ndarray) -> Dict:
        """
        Execute one step with time-varying dynamics.
        
        Args:
            actions: Binary array of length N
            
        Returns:
            Dict with step results
        """
        # Step 1: Evolve all true states using P(t)
        for arm in self.arms:
            P_t = self.get_true_P(arm, self.t)
            probs = P_t[arm.s_true, :]
            arm.s_true = self.rng.choice(self.J, p=probs)
        
        # Step 2: Process actions
        successes = []
        for i, arm in enumerate(self.arms):
            if actions[i] == 1:
                success = self.rng.random() < arm.p_s
                successes.append(success)
                if success:
                    arm.h = arm.s_true
                    arm.delta = 1
                else:
                    arm.delta = min(arm.delta + 1, self.delta_max)
            else:
                successes.append(False)
                arm.delta = min(arm.delta + 1, self.delta_max)
        
        # Compute metrics
        oracle_aoii = np.mean([
            self._compute_aoii(arm.s_true, arm.h, arm.delta)
            for arm in self.arms
        ])
        
        self.t += 1
        
        return {
            'mean_oracle_aoii': oracle_aoii,
            'successes': successes,
        }
    
    def _compute_aoii(self, s_true: int, h: int, delta: int) -> float:
        """Compute Age of Incorrect Information."""
        if s_true == h:
            return 0.0
        else:
            return float(delta)
    
    def get_observations(self) -> List[Tuple[int, int, int]]:
        """Get current observations for all arms."""
        return [(arm.arm_id, arm.h, arm.delta) for arm in self.arms]


def run_nonhomogeneous_experiment(
    output_dir: str = "results",
    N: int = 50,
    M: int = 5,
    T: int = 120,  # 10 years of monthly data
    n_seeds: int = 10,
    quick_test: bool = False
) -> pd.DataFrame:
    """
    Run time-nonhomogeneous experiment comparing Fixed/Windowed/Oracle strategies.
    
    Args:
        output_dir: Output directory
        N: Number of arms
        M: Budget per epoch
        T: Horizon (epochs = months)
        n_seeds: Number of random seeds
        quick_test: Use reduced parameters if True
    """
    
    Path(f"{output_dir}/data").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
    
    if quick_test:
        T = 48  # 4 years
        n_seeds = 3
        window_sizes = [3, 6]
    else:
        window_sizes = [3, 6, 12]
    
    seasonal_config = SeasonalConfig(
        T_season=12,      # 12 months = 1 year
        amplitude=0.3,    # ±30%
        window_size=3,    # Default, will vary
    )
    
    print("=" * 60)
    print("TIME-NONHOMOGENEOUS EXPERIMENT")
    print("=" * 60)
    print(f"N={N}, M={M}, T={T} months ({T/12:.1f} years)")
    print(f"Seasonal period: {seasonal_config.T_season} months")
    print(f"Amplitude: ±{seasonal_config.amplitude*100:.0f}%")
    print(f"Window sizes: {window_sizes}")
    print("=" * 60)
    
    # Create base config
    config = SimulationConfig()
    config.experiment.N = N
    config.experiment.M = M
    config.experiment.delta_max = 100
    config.arm_classes = get_nhgp_arm_classes(J=5, R=8)
    
    # Precompute Whittle tables for different P̄ scenarios
    solver = WhittleSolver(config.whittle)
    
    results_data = []
    seeds = list(range(42, 42 + n_seeds))
    
    for seed in seeds:
        print(f"\nSeed {seed}:")
        
        # Strategy 1: Fixed-P̄ (use initial P̄ throughout)
        print("  Running Fixed-P̄...")
        env = SeasonalRMABEnvironment(config, seasonal_config, seed=seed)
        env.reset(seed=seed)
        
        # Compute Whittle tables using base (t=0) parameters
        base_arm_classes = get_nhgp_arm_classes(J=5, R=8)
        fixed_tables = solver.compute_all_tables(base_arm_classes, config.experiment.delta_max, verbose=False)
        fixed_policy = WhittlePolicy(fixed_tables)
        
        fixed_aoii = []
        for t in range(T):
            obs = env.get_observations()
            actions = fixed_policy.select_arms(obs, env)
            result = env.step(actions)
            fixed_aoii.append(result['mean_oracle_aoii'])
        
        # Strategy 2: Windowed-P̄_w (update every W epochs)
        for W in window_sizes:
            print(f"  Running Windowed-P̄ (W={W})...")
            env.reset(seed=seed)
            
            windowed_aoii = []
            current_tables = fixed_tables  # Start with fixed
            
            for t in range(T):
                # Update tables every W epochs
                if t > 0 and t % W == 0:
                    # Recompute using average c over recent window
                    avg_c_slow = np.mean([seasonal_c(t-w, 0.01, seasonal_config) for w in range(W)])
                    avg_c_fast = np.mean([seasonal_c(t-w, 0.02, seasonal_config) for w in range(W)])
                    
                    # Build updated arm classes
                    updated_classes = []
                    for ac in base_arm_classes:
                        c = avg_c_slow if 'slow' in ac.name.lower() else avg_c_fast
                        P_bar = build_P_bar_with_c(J=5, c=c)
                        updated_classes.append(ArmClassConfig(
                            name=ac.name, P_bar=P_bar, p_s=ac.p_s, D=ac.D
                        ))
                    
                    current_tables = solver.compute_all_tables(updated_classes, config.experiment.delta_max, verbose=False)
                
                windowed_policy = WhittlePolicy(current_tables)
                obs = env.get_observations()
                actions = windowed_policy.select_arms(obs, env)
                result = env.step(actions)
                windowed_aoii.append(result['mean_oracle_aoii'])
            
            results_data.append({
                'seed': seed,
                'strategy': f'Windowed-W{W}',
                'mean_aoii': np.mean(windowed_aoii[T//2:]),  # Last 50%
                'std_aoii': np.std(windowed_aoii[T//2:]),
                'trajectory': windowed_aoii,
            })
        
        # Strategy 3: Oracle-P(t) (know true P at each step)
        print("  Running Oracle-P(t)...")
        env.reset(seed=seed)
        
        oracle_aoii = []
        for t in range(T):
            # Compute exact tables for current P(t)
            oracle_classes = []
            for ac in base_arm_classes:
                c_t = seasonal_c(t, 0.01 if 'slow' in ac.name.lower() else 0.02, seasonal_config)
                P_bar = build_P_bar_with_c(J=5, c=c_t)
                oracle_classes.append(ArmClassConfig(
                    name=ac.name, P_bar=P_bar, p_s=ac.p_s, D=ac.D
                ))
            
            oracle_tables = solver.compute_all_tables(oracle_classes, config.experiment.delta_max, verbose=False)
            oracle_policy = WhittlePolicy(oracle_tables)
            
            obs = env.get_observations()
            actions = oracle_policy.select_arms(obs, env)
            result = env.step(actions)
            oracle_aoii.append(result['mean_oracle_aoii'])
        
        results_data.append({
            'seed': seed,
            'strategy': 'Fixed',
            'mean_aoii': np.mean(fixed_aoii[T//2:]),
            'std_aoii': np.std(fixed_aoii[T//2:]),
            'trajectory': fixed_aoii,
        })
        
        results_data.append({
            'seed': seed,
            'strategy': 'Oracle',
            'mean_aoii': np.mean(oracle_aoii[T//2:]),
            'std_aoii': np.std(oracle_aoii[T//2:]),
            'trajectory': oracle_aoii,
        })
    
    # Aggregate results
    df = pd.DataFrame(results_data)
    
    # Compute summary statistics
    summary = df.groupby('strategy').agg({
        'mean_aoii': ['mean', 'std'],
    }).round(4)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(summary)
    
    # Save data
    summary_df = df.groupby('strategy')['mean_aoii'].agg(['mean', 'std']).reset_index()
    summary_df.to_csv(f"{output_dir}/data/fig5_nonhomog.csv", index=False)
    print(f"\nSaved: {output_dir}/data/fig5_nonhomog.csv")
    
    # Generate figure
    _plot_nonhomog_results(df, output_dir, T, seasonal_config)
    
    return df


def _plot_nonhomog_results(df: pd.DataFrame, output_dir: str, T: int,
                           seasonal_config: SeasonalConfig):
    """Generate publication-quality figure for nonhomogeneous results."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 8,
        'figure.figsize': (3.5, 2.5),
        'figure.dpi': 300,
    })
    
    fig, ax = plt.subplots()
    
    # Get unique strategies and compute means
    strategies = df['strategy'].unique()
    strategy_order = ['Fixed'] + [s for s in strategies if 'Windowed' in s] + ['Oracle']
    
    colors = {'Fixed': '#d62728', 'Oracle': '#2ca02c'}
    for s in strategies:
        if 'Windowed' in s:
            colors[s] = '#1f77b4'
    
    x_pos = np.arange(len(strategy_order))
    means = []
    stds = []
    
    for s in strategy_order:
        s_data = df[df['strategy'] == s]['mean_aoii']
        means.append(s_data.mean())
        stds.append(s_data.std())
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=3,
                  color=[colors.get(s, '#1f77b4') for s in strategy_order],
                  edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategy_order, rotation=15, ha='right')
    ax.set_ylabel('Mean AoII')
    ax.set_xlabel('Update Strategy')
    
    # Add gap annotations
    oracle_mean = means[strategy_order.index('Oracle')]
    for i, (s, m) in enumerate(zip(strategy_order, means)):
        if s != 'Oracle':
            gap = (m - oracle_mean) / oracle_mean * 100
            ax.annotate(f'+{gap:.1f}%', xy=(i, m), xytext=(0, 5),
                       textcoords='offset points', ha='center', fontsize=6)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        fig.savefig(f"{output_dir}/figures/fig5_nonhomog.{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/figures/fig5_nonhomog.pdf/png")
    
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Time-Nonhomogeneous Experiment')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: auto)')
    args = parser.parse_args()
    
    run_nonhomogeneous_experiment(
        output_dir=args.output,
        quick_test=args.quick
    )
