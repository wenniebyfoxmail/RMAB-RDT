"""
Experiments for Road Digital Twin AoII-ARD RMAB Simulation (v2.3)
==================================================================

按导师要求实现的主实验:
- Fig 1: N sweep (N ∈ {20, 50, 100, 200}, 固定M/N比例)
- Fig 2: M sweep (N=100, M ∈ {1, 2, 5, 10, 15, 20})
- Fig 3: p_s sweep (可靠性敏感性: 0.70, 0.80, 0.90, 0.95, 0.996)
- Table 1: 统计摘要 + Δ分布统计
- P1: 小规模真最优对照 (N=2, M=1, J=3, Δ_max=5)

IEEE Format:
- Width: 3.5 inches
- No titles
- Font size: 8pt
- PNG + PDF output

实验参数:
- T=2000, 10 seeds
- 后50% horizon统计
- γ=0.999

并行计算:
- 自动检测CPU核心数
- 多seed并行执行
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
import os
import warnings
from pathlib import Path
from copy import deepcopy
from itertools import product
from functools import partial

# Import project modules
from config import (
    SimulationConfig, ExperimentConfig, WhittleConfig,
    ArmClassConfig, get_nhgp_arm_classes, get_channel_params, R_TABLE,
    compute_control_cost
)
from environment import RMABEnvironment, EpisodeLogger
from whittle_solver import WhittleSolver, WhittleIndexTable
from policies import (
    BasePolicy, WhittlePolicy, MaxAgePolicy, 
    MyopicPolicy, RandomPolicy
)
from parallel_utils import get_optimal_workers, parallel_map, get_cpu_count

warnings.filterwarnings('ignore')


# =============================================================================
# IEEE Figure Style Configuration
# =============================================================================

def setup_ieee_style():
    """Configure matplotlib for IEEE publication format."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.figsize': (3.5, 2.5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'axes.linewidth': 0.5,
        'axes.grid': True,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.5,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })


POLICY_STYLES = {
    'Whittle': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
    'MaxAge': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
    'Myopic': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
    'Random': {'color': '#d62728', 'marker': 'x', 'linestyle': ':'},
}


# =============================================================================
# Extended Experiment Results with Δ Statistics
# =============================================================================

@dataclass
class ExperimentResults:
    """Container for experiment results with Δ distribution statistics."""
    policy_name: str
    seeds: List[int]
    trajectories: np.ndarray      # Shape: (n_seeds, T)
    eval_means: np.ndarray        # Shape: (n_seeds,)
    eval_stds: np.ndarray
    delta_means: np.ndarray       # Shape: (n_seeds,) - Mean Δ over eval period
    delta_p90: np.ndarray         # Shape: (n_seeds,) - 90th percentile Δ
    computation_time: float
    
    @property
    def mean(self) -> float:
        return float(np.mean(self.eval_means))
    
    @property
    def std(self) -> float:
        return float(np.std(self.eval_means))
    
    @property
    def mean_delta(self) -> float:
        return float(np.mean(self.delta_means))


# =============================================================================
# Parallel Execution Helper Functions
# =============================================================================

def _config_to_dict(config: SimulationConfig) -> dict:
    """Convert SimulationConfig to picklable dict."""
    return {
        'N': config.experiment.N,
        'M': config.experiment.M,
        'J': config.experiment.J,
        'delta_max': config.experiment.delta_max,
        'T': config.experiment.T,
        'R': config.experiment.R,
        'burn_in_ratio': config.experiment.burn_in_ratio,
        'arm_classes': [(ac.name, ac.P_bar.tolist(), ac.p_s, ac.D) 
                        for ac in config.arm_classes],
    }


def _dict_to_config(d: dict) -> SimulationConfig:
    """Reconstruct SimulationConfig from dict."""
    config = SimulationConfig()
    config.experiment.N = d['N']
    config.experiment.M = d['M']
    config.experiment.J = d['J']
    config.experiment.delta_max = d['delta_max']
    config.experiment.T = d['T']
    config.experiment.R = d['R']
    config.experiment.burn_in_ratio = d['burn_in_ratio']
    
    config.arm_classes = [
        ArmClassConfig(name=name, P_bar=np.array(P_bar), p_s=p_s, D=D)
        for name, P_bar, p_s, D in d['arm_classes']
    ]
    
    return config


def _serialize_index_tables(tables: Dict[str, 'WhittleIndexTable']) -> dict:
    """Serialize Whittle index tables for pickling."""
    return {
        name: {
            'indices': table.indices.tolist(),
            'J': table.J,
            'delta_max': table.delta_max
        }
        for name, table in tables.items()
    }


def _deserialize_index_tables(data: dict) -> Dict[str, 'WhittleIndexTable']:
    """Deserialize Whittle index tables."""
    tables = {}
    for name, table_data in data.items():
        table = WhittleIndexTable(
            J=table_data['J'],
            delta_max=table_data['delta_max']
        )
        table.indices = np.array(table_data['indices'])
        tables[name] = table
    return tables


def _run_single_seed(seed: int, config_dict: dict, policy_name: str,
                     index_tables_data: dict, T: int, delta_max: int,
                     burn_in_ratio: float) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Run a single seed for a given policy. Designed for parallel execution.
    
    Returns:
        (trajectory, eval_mean, eval_std, delta_mean, delta_p90)
    """
    # Reconstruct config
    config = _dict_to_config(config_dict)
    
    # Create environment
    env = RMABEnvironment(config, seed=seed)
    env.delta_max = delta_max
    
    # Create policy
    if policy_name == 'Whittle':
        index_tables = _deserialize_index_tables(index_tables_data)
        policy = WhittlePolicy(index_tables)
    elif policy_name == 'MaxAge':
        policy = MaxAgePolicy()
    elif policy_name == 'Myopic':
        policy = MyopicPolicy()
    elif policy_name == 'Random':
        policy = RandomPolicy(seed=seed)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    policy.reset()
    
    # Run episode
    obs = env._get_observations()
    trajectory = np.zeros(T)
    delta_history = []
    
    for t in range(T):
        actions = policy.select_arms(obs, env)
        result = env.step(actions)
        trajectory[t] = result.info['mean_oracle_aoii']
        
        # Record Δ distribution
        if t >= int(T * burn_in_ratio):
            deltas = result.observations[:, 1]
            delta_history.extend(deltas.tolist())
        
        obs = result.observations
    
    # Compute statistics
    burn_in = int(T * burn_in_ratio)
    eval_period = trajectory[burn_in:]
    delta_array = np.array(delta_history)
    
    return (trajectory, 
            eval_period.mean(), 
            eval_period.std(),
            delta_array.mean() if len(delta_array) > 0 else 0,
            np.percentile(delta_array, 90) if len(delta_array) > 0 else 0)


# =============================================================================
# Core Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Runs RMAB experiments with comprehensive metrics."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / "data"
        self.fig_dir = self.output_dir / "figures"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        
        setup_ieee_style()
        
        # Cache for Whittle tables
        self._whittle_cache = {}
    
    def _get_whittle_tables(self, arm_classes: List[ArmClassConfig], 
                            delta_max: int, whittle_config: WhittleConfig) -> Dict[str, WhittleIndexTable]:
        """Get or compute Whittle tables with caching."""
        # Create cache key
        key = (tuple(ac.name for ac in arm_classes), 
               tuple(ac.p_s for ac in arm_classes),
               delta_max)
        
        if key not in self._whittle_cache:
            solver = WhittleSolver(whittle_config)
            self._whittle_cache[key] = solver.compute_all_tables(arm_classes, delta_max, verbose=False)
        
        return self._whittle_cache[key]
    
    def run_single_episode(self, env: RMABEnvironment, policy: BasePolicy,
                           T: int, burn_in_ratio: float = 0.5) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Run a single episode with comprehensive metrics.
        
        Returns:
            trajectory, eval_mean, eval_std, delta_mean, delta_p90
        """
        obs = env._get_observations()
        trajectory = np.zeros(T)
        delta_history = []
        
        for t in range(T):
            actions = policy.select_arms(obs, env)
            result = env.step(actions)
            trajectory[t] = result.info['mean_oracle_aoii']
            
            # Record Δ distribution
            if t >= int(T * burn_in_ratio):
                deltas = result.observations[:, 1]
                delta_history.extend(deltas.tolist())
            
            obs = result.observations
        
        # Compute statistics over evaluation period
        burn_in = int(T * burn_in_ratio)
        eval_period = trajectory[burn_in:]
        
        delta_array = np.array(delta_history)
        
        return (trajectory, 
                eval_period.mean(), 
                eval_period.std(),
                delta_array.mean() if len(delta_array) > 0 else 0,
                np.percentile(delta_array, 90) if len(delta_array) > 0 else 0)
    
    def run_experiment(self, N: int, M: int, J: int = 5, delta_max: int = 100,
                       T: int = 2000, R: int = 8, n_seeds: int = 10,
                       p_s_override: float = None,
                       verbose: bool = True,
                       parallel: bool = True) -> Dict[str, ExperimentResults]:
        """
        Run experiment with all policies.
        
        Args:
            parallel: If True, run seeds in parallel (auto-detect cores)
        """
        
        # Create configuration
        config = SimulationConfig()
        config.experiment.N = N
        config.experiment.M = M
        config.experiment.J = J
        config.experiment.delta_max = delta_max
        config.experiment.T = T
        config.experiment.R = R
        config.arm_classes = get_nhgp_arm_classes(J=J, R=R)
        
        # Override p_s if specified (for stress testing)
        if p_s_override is not None:
            for ac in config.arm_classes:
                ac.p_s = p_s_override
        
        seeds = list(range(42, 42 + n_seeds))
        burn_in_ratio = config.experiment.burn_in_ratio
        
        # Get Whittle tables
        if verbose:
            print(f"  Computing Whittle indices...")
        index_tables = self._get_whittle_tables(config.arm_classes, delta_max, config.whittle)
        
        # Determine parallelization
        n_workers = get_optimal_workers(n_seeds) if parallel else 1
        use_parallel = n_workers > 1 and n_seeds > 1
        
        if verbose and use_parallel:
            print(f"  Parallel execution: {n_workers} workers, {n_seeds} seeds")
        
        results = {}
        
        for name in ['Whittle', 'MaxAge', 'Myopic', 'Random']:
            if verbose:
                print(f"  Running {name}...", end=" ", flush=True)
            
            start_time = time.time()
            
            if use_parallel:
                # Parallel execution
                seed_results = parallel_map(
                    _run_single_seed,
                    seeds,
                    n_workers=n_workers,
                    config_dict=_config_to_dict(config),
                    policy_name=name,
                    index_tables_data=_serialize_index_tables(index_tables) if name == 'Whittle' else None,
                    T=T,
                    delta_max=delta_max,
                    burn_in_ratio=burn_in_ratio
                )
            else:
                # Sequential execution
                seed_results = []
                for seed in seeds:
                    result = _run_single_seed(
                        seed,
                        config_dict=_config_to_dict(config),
                        policy_name=name,
                        index_tables_data=_serialize_index_tables(index_tables) if name == 'Whittle' else None,
                        T=T,
                        delta_max=delta_max,
                        burn_in_ratio=burn_in_ratio
                    )
                    seed_results.append(result)
            
            # Aggregate results
            trajectories = np.array([r[0] for r in seed_results])
            eval_means = np.array([r[1] for r in seed_results])
            eval_stds = np.array([r[2] for r in seed_results])
            delta_means = np.array([r[3] for r in seed_results])
            delta_p90s = np.array([r[4] for r in seed_results])
            
            comp_time = time.time() - start_time
            
            results[name] = ExperimentResults(
                policy_name=name,
                seeds=seeds,
                trajectories=trajectories,
                eval_means=eval_means,
                eval_stds=eval_stds,
                delta_means=delta_means,
                delta_p90=delta_p90s,
                computation_time=comp_time
            )
            
            if verbose:
                print(f"AoII={results[name].mean:.3f}, Δ_avg={results[name].mean_delta:.1f}")
        
        return results


# =============================================================================
# Plotting Functions
# =============================================================================

class IEEEPlotter:
    """Generates IEEE-format publication figures."""
    
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_ieee_style()
    
    def save_figure(self, fig: plt.Figure, name: str) -> Tuple[str, str]:
        """Save figure in PNG and PDF formats."""
        png_path = self.output_dir / f"{name}.png"
        pdf_path = self.output_dir / f"{name}.pdf"
        
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        
        print(f"Saved: {png_path}")
        return str(png_path), str(pdf_path)
    
    def plot_fig1_n_sweep(self, results: Dict[int, Dict[str, ExperimentResults]],
                          budget_ratio: float = 0.1) -> plt.Figure:
        """Fig 1: AoII vs N (scale sweep)."""
        fig, ax = plt.subplots(figsize=(3.5, 2.3))
        
        N_values = sorted(results.keys())
        
        for name in ['Whittle', 'MaxAge', 'Myopic', 'Random']:
            style = POLICY_STYLES[name]
            
            means = [results[N][name].mean for N in N_values]
            stds = [results[N][name].std for N in N_values]
            
            ax.errorbar(N_values, means, yerr=stds,
                        label=name, color=style['color'],
                        marker=style['marker'], linestyle=style['linestyle'],
                        markersize=4, capsize=2, linewidth=1.0)
        
        ax.set_xlabel('Number of Arms $N$')
        ax.set_ylabel('Mean Oracle AoII')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_xscale('log')
        ax.set_xticks(N_values)
        ax.set_xticklabels([str(n) for n in N_values])
        
        # Add M/N ratio annotation
        ax.text(0.02, 0.98, f'$M/N = {budget_ratio}$', 
                transform=ax.transAxes, fontsize=7, va='top')
        
        plt.tight_layout()
        self.save_figure(fig, 'fig1_n_sweep')
        return fig
    
    def plot_fig2_m_sweep(self, results: Dict[int, Dict[str, ExperimentResults]],
                          N: int = 100) -> plt.Figure:
        """Fig 2: AoII vs M (budget sweep)."""
        fig, ax = plt.subplots(figsize=(3.5, 2.3))
        
        M_values = sorted(results.keys())
        
        for name in ['Whittle', 'MaxAge', 'Myopic', 'Random']:
            style = POLICY_STYLES[name]
            
            means = [results[M][name].mean for M in M_values]
            stds = [results[M][name].std for M in M_values]
            
            ax.errorbar(M_values, means, yerr=stds,
                        label=name, color=style['color'],
                        marker=style['marker'], linestyle=style['linestyle'],
                        markersize=4, capsize=2, linewidth=1.0)
        
        ax.set_xlabel('Budget $M$')
        ax.set_ylabel('Mean Oracle AoII')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_xticks(M_values)
        
        # Add N annotation
        ax.text(0.02, 0.98, f'$N = {N}$', 
                transform=ax.transAxes, fontsize=7, va='top')
        
        plt.tight_layout()
        self.save_figure(fig, 'fig2_m_sweep')
        return fig
    
    def plot_fig3_ps_sweep(self, results: Dict[float, Dict[str, ExperimentResults]]) -> plt.Figure:
        """Fig 3: AoII vs p_s (reliability sweep)."""
        fig, ax = plt.subplots(figsize=(3.5, 2.3))
        
        ps_values = sorted(results.keys())
        
        for name in ['Whittle', 'MaxAge', 'Myopic', 'Random']:
            style = POLICY_STYLES[name]
            
            means = [results[ps][name].mean for ps in ps_values]
            stds = [results[ps][name].std for ps in ps_values]
            
            ax.errorbar(ps_values, means, yerr=stds,
                        label=name, color=style['color'],
                        marker=style['marker'], linestyle=style['linestyle'],
                        markersize=4, capsize=2, linewidth=1.0)
        
        ax.set_xlabel('Success Probability $p_s$')
        ax.set_ylabel('Mean Oracle AoII')
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Highlight R-table values
        ax.axvline(x=0.996, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
        ax.text(0.996, ax.get_ylim()[1]*0.95, 'R=8', fontsize=6, ha='center')
        
        plt.tight_layout()
        self.save_figure(fig, 'fig3_ps_sweep')
        return fig
    
    def plot_delta_distribution(self, results: Dict[str, ExperimentResults]) -> plt.Figure:
        """Plot Δ distribution comparison."""
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        
        policies = ['Whittle', 'MaxAge', 'Myopic', 'Random']
        delta_means = [results[p].mean_delta for p in policies]
        colors = [POLICY_STYLES[p]['color'] for p in policies]
        
        x = np.arange(len(policies))
        bars = ax.bar(x, delta_means, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels(policies)
        ax.set_ylabel('Mean Age $\\Delta$')
        
        for bar, val in zip(bars, delta_means):
            ax.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=6)
        
        plt.tight_layout()
        self.save_figure(fig, 'fig_delta_dist')
        return fig


# =============================================================================
# Table Generation
# =============================================================================

def generate_table1(results: Dict[str, ExperimentResults], 
                    output_path: str,
                    config_info: str = "") -> pd.DataFrame:
    """
    Generate Table 1: Comprehensive statistics with Δ distribution.
    """
    whittle_mean = results['Whittle'].mean
    
    data = []
    for name in ['Whittle', 'MaxAge', 'Myopic', 'Random']:
        if name not in results:
            continue
        
        r = results[name]
        gap = (r.mean - whittle_mean) / whittle_mean * 100 if whittle_mean > 0 else 0
        
        data.append({
            'Policy': name,
            'Mean AoII': f'{r.mean:.4f}',
            'Std': f'{r.std:.4f}',
            'Gap (%)': f'{gap:+.1f}' if name != 'Whittle' else '—',
            'Mean Δ': f'{r.mean_delta:.1f}',
            'Δ P90': f'{np.mean(r.delta_p90):.1f}',
            'Time (s)': f'{r.computation_time:.1f}'
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    return df


# =============================================================================
# P1: Small-Scale Optimal Benchmark (真最优对照)
# =============================================================================

class SmallScaleOptimalSolver:
    """
    Compute optimal policy for small-scale RMAB via value iteration.
    
    For N=2, M=1, J=3, Δ_max=5: State space is manageable (~225 joint states).
    Uses γ-discounted value iteration on the joint state space.
    """
    
    def __init__(self, N: int, M: int, J: int, delta_max: int,
                 P_bar: np.ndarray, p_s: float, gamma: float = 0.999):
        self.N = N
        self.M = M
        self.J = J
        self.delta_max = delta_max
        self.P_bar = P_bar
        self.p_s = p_s
        self.gamma = gamma
        
        # Generate state space: each arm state is (h, delta)
        self.arm_states = [(h, d) for h in range(J) for d in range(1, delta_max + 1)]
        self.n_arm_states = len(self.arm_states)
        
        # Joint state space: tuple of N arm states
        self.joint_states = list(product(range(self.n_arm_states), repeat=N))
        self.n_joint_states = len(self.joint_states)
        
        # Action space: which M arms to schedule (combinations)
        from itertools import combinations
        self.actions = list(combinations(range(N), M))
        self.n_actions = len(self.actions)
        
        # Precompute costs
        self._precompute_costs()
    
    def _arm_idx_to_state(self, idx: int) -> Tuple[int, int]:
        """Convert arm state index to (h, delta)."""
        return self.arm_states[idx]
    
    def _state_to_arm_idx(self, h: int, delta: int) -> int:
        """Convert (h, delta) to arm state index."""
        delta = max(1, min(delta, self.delta_max))
        return h * self.delta_max + (delta - 1)
    
    def _precompute_costs(self):
        """Precompute per-arm costs."""
        self.arm_costs = np.zeros(self.n_arm_states)
        for idx, (h, delta) in enumerate(self.arm_states):
            self.arm_costs[idx] = compute_control_cost(h, delta, self.P_bar)
    
    def _compute_joint_cost(self, joint_state: Tuple[int, ...]) -> float:
        """Compute total cost for a joint state."""
        return sum(self.arm_costs[s] for s in joint_state)
    
    def _next_state_passive(self, arm_idx: int) -> int:
        """Next arm state if not scheduled (passive)."""
        h, delta = self._arm_idx_to_state(arm_idx)
        new_delta = min(delta + 1, self.delta_max)
        return self._state_to_arm_idx(h, new_delta)
    
    def _next_state_active_success(self, arm_idx: int) -> List[Tuple[int, float]]:
        """
        Next arm states after successful update (active success).
        Returns list of (next_arm_idx, probability) pairs.
        """
        h, delta = self._arm_idx_to_state(arm_idx)
        
        # Compute belief after evolution
        from config import compute_belief_after_evolution
        belief_evolved = compute_belief_after_evolution(h, delta, self.P_bar)
        
        # Transition to (j, 1) with probability belief_evolved[j]
        transitions = []
        for j in range(self.J):
            if belief_evolved[j] > 1e-10:
                next_idx = self._state_to_arm_idx(j, 1)
                transitions.append((next_idx, belief_evolved[j]))
        
        return transitions
    
    def _next_state_active_fail(self, arm_idx: int) -> int:
        """Next arm state after failed update (same as passive)."""
        return self._next_state_passive(arm_idx)
    
    def compute_optimal_value(self, max_iter: int = 500, 
                               tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal value function via value iteration.
        
        Returns:
            V: Optimal value function
            policy: Optimal action for each joint state
        """
        print(f"  Computing optimal policy for N={self.N}, M={self.M}...")
        print(f"  State space: {self.n_joint_states} joint states, {self.n_actions} actions")
        
        V = np.zeros(self.n_joint_states)
        policy = np.zeros(self.n_joint_states, dtype=int)
        
        for iteration in range(max_iter):
            V_old = V.copy()
            
            for s_idx, joint_state in enumerate(self.joint_states):
                best_Q = np.inf
                best_action = 0
                
                # Current cost
                cost = self._compute_joint_cost(joint_state)
                
                for a_idx, action in enumerate(self.actions):
                    # Expected future value
                    EV = 0.0
                    
                    # For each arm, compute transition
                    # Action specifies which arms are scheduled
                    scheduled_set = set(action)
                    
                    # Build transition distribution
                    # Start with all arms being passive
                    arm_transitions = []
                    for arm in range(self.N):
                        if arm in scheduled_set:
                            # Active: success or fail
                            arm_idx = joint_state[arm]
                            success_trans = self._next_state_active_success(arm_idx)
                            fail_next = self._next_state_active_fail(arm_idx)
                            
                            # Combine success (p_s) and fail (1-p_s)
                            combined = []
                            for next_idx, prob in success_trans:
                                combined.append((next_idx, self.p_s * prob))
                            combined.append((fail_next, 1 - self.p_s))
                            arm_transitions.append(combined)
                        else:
                            # Passive
                            arm_idx = joint_state[arm]
                            next_idx = self._next_state_passive(arm_idx)
                            arm_transitions.append([(next_idx, 1.0)])
                    
                    # Compute expected future value (product of independent transitions)
                    # For small N, enumerate all combinations
                    def enumerate_transitions(arm_idx: int, current_state: List[int], 
                                             current_prob: float):
                        nonlocal EV
                        if arm_idx == self.N:
                            # Terminal: convert to joint state index
                            next_joint = tuple(current_state)
                            next_s_idx = self.joint_states.index(next_joint)
                            EV += current_prob * V_old[next_s_idx]
                            return
                        
                        for next_arm_idx, prob in arm_transitions[arm_idx]:
                            enumerate_transitions(
                                arm_idx + 1,
                                current_state + [next_arm_idx],
                                current_prob * prob
                            )
                    
                    enumerate_transitions(0, [], 1.0)
                    
                    Q = cost + self.gamma * EV
                    
                    if Q < best_Q:
                        best_Q = Q
                        best_action = a_idx
                
                V[s_idx] = best_Q
                policy[s_idx] = best_action
            
            # Check convergence
            diff = np.max(np.abs(V - V_old))
            if diff < tol:
                print(f"  Converged at iteration {iteration}")
                break
        
        return V, policy
    
    def evaluate_policy(self, policy_fn, T: int = 1000, n_seeds: int = 5) -> float:
        """
        Evaluate a policy by simulation.
        
        Args:
            policy_fn: Function that takes joint state and returns action index
            T: Number of time steps
            n_seeds: Number of random seeds
        
        Returns:
            Mean cost per step
        """
        total_cost = 0.0
        
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            
            # Initial state: all arms at (h=0, delta=1)
            current_state = tuple([self._state_to_arm_idx(0, 1)] * self.N)
            
            episode_cost = 0.0
            for t in range(T):
                # Get action
                s_idx = self.joint_states.index(current_state)
                a_idx = policy_fn(s_idx)
                action = self.actions[a_idx]
                
                # Cost
                episode_cost += self._compute_joint_cost(current_state)
                
                # Transition
                scheduled_set = set(action)
                next_state = []
                
                for arm in range(self.N):
                    arm_idx = current_state[arm]
                    
                    if arm in scheduled_set:
                        # Active
                        if rng.random() < self.p_s:
                            # Success
                            h, delta = self._arm_idx_to_state(arm_idx)
                            from config import compute_belief_after_evolution
                            belief_evolved = compute_belief_after_evolution(h, delta, self.P_bar)
                            new_h = rng.choice(self.J, p=belief_evolved)
                            next_arm_idx = self._state_to_arm_idx(new_h, 1)
                        else:
                            # Fail
                            next_arm_idx = self._next_state_passive(arm_idx)
                    else:
                        # Passive
                        next_arm_idx = self._next_state_passive(arm_idx)
                    
                    next_state.append(next_arm_idx)
                
                current_state = tuple(next_state)
            
            total_cost += episode_cost / T
        
        return total_cost / n_seeds


def run_small_scale_benchmark(output_dir: str = "results") -> pd.DataFrame:
    """
    Run P1: Small-scale optimal benchmark.
    
    Note: With J=3 and high p_s, ties are common and results are 
    dominated by tie-breaking randomness. Results should be interpreted
    with this caveat.
    """
    print("\n" + "=" * 60)
    print("P1: SMALL-SCALE OPTIMAL BENCHMARK")
    print("N=2, M=1, J=3, Δ_max=5")
    print("=" * 60)
    
    # Parameters
    N, M, J, delta_max = 2, 1, 3, 5
    T_sim = 2000
    n_seeds = 10
    
    # Get arm class (use slow class)
    arm_classes = get_nhgp_arm_classes(J=J, R=8)
    arm_class = arm_classes[0]  # slow class
    P_bar = arm_class.P_bar
    p_s = arm_class.p_s
    
    print(f"\np_s = {p_s}")
    print(f"P_bar:\n{np.round(P_bar, 4)}")
    
    # Compute optimal policy
    solver = SmallScaleOptimalSolver(N, M, J, delta_max, P_bar, p_s)
    V_opt, policy_opt = solver.compute_optimal_value()
    
    # Evaluate optimal policy
    opt_cost = solver.evaluate_policy(lambda s: policy_opt[s], T_sim, n_seeds)
    print(f"\nOptimal policy average cost: {opt_cost:.4f}")
    
    # Create comparison environment
    # Use ONLY slow class (single class for homogeneous benchmark)
    config = SimulationConfig()
    config.experiment.N = N
    config.experiment.M = M
    config.experiment.J = J
    config.experiment.delta_max = delta_max
    # Only ONE arm class to ensure homogeneous environment
    config.arm_classes = [arm_classes[0]]  # Only slow class
    
    # Run heuristic policies
    results = {}
    
    # Compute Whittle tables
    whittle_config = config.whittle
    whittle_solver = WhittleSolver(whittle_config)
    table_slow = whittle_solver.compute_index_table(arm_classes[0], delta_max, verbose=False)
    whittle_tables = {'slow': table_slow}
    
    policies = {
        'Whittle': WhittlePolicy(whittle_tables),
        'MaxAge': MaxAgePolicy(),
        'Myopic': MyopicPolicy(),
        'Random': RandomPolicy(seed=42),
    }
    
    for name, policy in policies.items():
        total_cost = 0.0
        
        for seed in range(42, 42 + n_seeds):
            env = RMABEnvironment(config, seed=seed)
            env.delta_max = delta_max
            policy.reset()
            
            episode_cost = 0.0
            for t in range(T_sim):
                obs = env._get_observations()
                # Use deterministic tie-breaking based on arm index
                np.random.seed(seed * 10000 + t)
                actions = policy.select_arms(obs, env)
                result = env.step(actions)
                # Use CONTROL COST (same as optimal solver) for fair comparison
                episode_cost += np.sum(result.control_costs)
            
            total_cost += episode_cost / T_sim
        
        results[name] = total_cost / n_seeds
    
    # Summary
    print("\n--- Results ---")
    print(f"{'Policy':<10} {'Avg Cost':<12} {'Gap to Opt':<12}")
    print("-" * 35)
    print(f"{'Optimal':<10} {opt_cost:<12.4f} {'—':<12}")
    
    for name, cost in results.items():
        gap = (cost - opt_cost) / opt_cost * 100 if opt_cost > 0 else 0
        print(f"{name:<10} {cost:<12.4f} {gap:+.1f}%")
    
    print("\nNote: With J=3 and high p_s=0.996, most states are ties.")
    print("Whittle and MaxAge make identical decisions when tie-breaking is fixed.")
    
    # Save results
    Path(f"{output_dir}/data").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {'Policy': 'Optimal', 'Avg_Cost': opt_cost, 'Gap_pct': 0.0},
        *[{'Policy': name, 'Avg_Cost': cost, 
           'Gap_pct': (cost - opt_cost) / opt_cost * 100 if opt_cost > 0 else 0} 
          for name, cost in results.items()]
    ])
    df.to_csv(f"{output_dir}/data/p1_optimal_benchmark.csv", index=False)
    print(f"\nSaved: {output_dir}/data/p1_optimal_benchmark.csv")
    
    return df


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_experiments(output_dir: str = "results", quick_test: bool = False):
    """
    Run all experiments per advisor requirements.
    
    Args:
        output_dir: Output directory
        quick_test: If True, use reduced parameters
    """
    if quick_test:
        T, n_seeds = 500, 3
        N_values = [20, 50]
        M_values = [2, 5, 10]
        ps_values = [0.90, 0.996]
        N_base = 50
    else:
        T, n_seeds = 2000, 10
        N_values = [20, 50, 100, 200]
        M_values = [1, 2, 5, 10, 15, 20]
        ps_values = [0.70, 0.80, 0.90, 0.95, 0.996]
        N_base = 100
    
    print("\n" + "=" * 70)
    print("  ROAD DT AoII-ARD RMAB EXPERIMENTS v2.2")
    print("  (按导师要求: Fig1-N sweep, Fig2-M sweep, Fig3-p_s sweep)")
    print("=" * 70)
    print(f"T={T}, seeds={n_seeds}, burn-in=50%")
    print("=" * 70 + "\n")
    
    runner = ExperimentRunner(output_dir)
    plotter = IEEEPlotter(f"{output_dir}/figures")
    
    total_start = time.time()
    
    # =========================================================================
    # Fig 1: N Sweep
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: N SWEEP (Fig 1)")
    print("=" * 60)
    
    fig1_results = {}
    budget_ratio = 0.1
    
    for N in N_values:
        M = max(1, int(N * budget_ratio))
        print(f"\n--- N={N}, M={M} ---")
        fig1_results[N] = runner.run_experiment(N=N, M=M, T=T, n_seeds=n_seeds)
    
    # Save data
    fig1_data = []
    for N, results in fig1_results.items():
        for name, r in results.items():
            fig1_data.append({
                'N': N, 'M': int(N * budget_ratio), 'policy': name,
                'mean_aoii': r.mean, 'std_aoii': r.std, 'mean_delta': r.mean_delta
            })
    pd.DataFrame(fig1_data).to_csv(f"{output_dir}/data/fig1_n_sweep.csv", index=False)
    
    # Plot
    plotter.plot_fig1_n_sweep(fig1_results, budget_ratio)
    plt.close('all')
    
    # =========================================================================
    # Fig 2: M Sweep
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: M SWEEP (Fig 2)")
    print("=" * 60)
    
    fig2_results = {}
    
    for M in M_values:
        print(f"\n--- M={M} (M/N={M/N_base:.1%}) ---")
        fig2_results[M] = runner.run_experiment(N=N_base, M=M, T=T, n_seeds=n_seeds)
    
    # Save data
    fig2_data = []
    for M, results in fig2_results.items():
        for name, r in results.items():
            fig2_data.append({
                'N': N_base, 'M': M, 'policy': name,
                'mean_aoii': r.mean, 'std_aoii': r.std, 'mean_delta': r.mean_delta
            })
    pd.DataFrame(fig2_data).to_csv(f"{output_dir}/data/fig2_m_sweep.csv", index=False)
    
    # Plot
    plotter.plot_fig2_m_sweep(fig2_results, N_base)
    plt.close('all')
    
    # =========================================================================
    # Fig 3: p_s Sweep
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: RELIABILITY SWEEP (Fig 3)")
    print("=" * 60)
    
    fig3_results = {}
    M_fixed = max(1, int(N_base * 0.1))
    
    for p_s in ps_values:
        print(f"\n--- p_s={p_s} ---")
        fig3_results[p_s] = runner.run_experiment(
            N=N_base, M=M_fixed, T=T, n_seeds=n_seeds, p_s_override=p_s
        )
    
    # Save data
    fig3_data = []
    for p_s, results in fig3_results.items():
        for name, r in results.items():
            fig3_data.append({
                'p_s': p_s, 'N': N_base, 'M': M_fixed, 'policy': name,
                'mean_aoii': r.mean, 'std_aoii': r.std, 'mean_delta': r.mean_delta
            })
    pd.DataFrame(fig3_data).to_csv(f"{output_dir}/data/fig3_ps_sweep.csv", index=False)
    
    # Plot
    plotter.plot_fig3_ps_sweep(fig3_results)
    plt.close('all')
    
    # =========================================================================
    # Table 1: Summary (using baseline N=50 or 100, M=5 or 10)
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING TABLE 1")
    print("=" * 60)
    
    # Use middle point from M sweep as baseline
    baseline_M = M_values[len(M_values)//2]
    baseline_results = fig2_results.get(baseline_M, fig2_results[M_values[0]])
    
    table1 = generate_table1(baseline_results, f"{output_dir}/data/table1_summary.csv",
                            f"N={N_base}, M={baseline_M}")
    print("\n" + table1.to_string(index=False))
    
    # Also plot Δ distribution
    plotter.plot_delta_distribution(baseline_results)
    plt.close('all')
    
    # =========================================================================
    # P1: Small-Scale Optimal Benchmark (if not quick test)
    # =========================================================================
    p1_results = None
    if not quick_test:
        try:
            p1_results = run_small_scale_benchmark(output_dir)
        except Exception as e:
            print(f"\nWarning: P1 benchmark failed: {e}")
            print("Continuing with other experiments...")
    
    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("  EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nOutput: {output_dir}/")
    print("  Figures: fig1_n_sweep, fig2_m_sweep, fig3_ps_sweep, fig_delta_dist")
    print("  Data: fig1_n_sweep.csv, fig2_m_sweep.csv, fig3_ps_sweep.csv, table1_summary.csv")
    if p1_results is not None:
        print("  P1 Benchmark: p1_optimal_benchmark.csv")
    print("=" * 70)
    
    return {
        'fig1_results': fig1_results,
        'fig2_results': fig2_results,
        'fig3_results': fig3_results,
        'table1': table1,
        'p1_benchmark': p1_results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AoII-ARD RMAB Experiments v2.3')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--p1-only', action='store_true', help='Run only P1 benchmark')
    args = parser.parse_args()
    
    if args.p1_only:
        run_small_scale_benchmark(args.output)
    else:
        run_all_experiments(output_dir=args.output, quick_test=args.quick)
