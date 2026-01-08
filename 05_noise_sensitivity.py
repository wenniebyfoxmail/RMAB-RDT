"""
Observation Noise Sensitivity Analysis: Q_R Channel
=====================================================

Implements the observation noise channel Q_R(ŝ|s) as specified by advisor (Q2).

Main experiment: D=0 (perfect observation)
This module: Sensitivity analysis for D>0 (imperfect observation)

Noise model:
    Q_R(ŝ|s) = {
        1 - q_err           if ŝ = s
        q_err / 2           if |ŝ - s| = 1 (first-order neighbors)
        0                   otherwise
    }
    
    where q_err(R) = min(η · σ(R) / Δ_bin, q_max)
    and σ(R) = √D(R), q_max = 0.3, η ∈ [1, 2]

Engineering interpretation:
    - D(R) is the MSE from rate-distortion theory
    - σ(R) = √D(R) is the typical error magnitude (RMS)
    - Δ_bin is the bin width in the continuous PCI scale
    - q_err represents the probability of misclassifying by ±1 state
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

from config import SimulationConfig, get_nhgp_arm_classes, R_TABLE
from environment import RMABEnvironment
from whittle_solver import WhittleSolver
from policies import WhittlePolicy, MyopicPolicy


# =============================================================================
# Q_R Observation Noise Configuration
# =============================================================================

@dataclass
class ObservationNoiseConfig:
    """Configuration for observation noise channel Q_R."""
    
    # PCI-based bin width (100-point scale divided into J=5 bins)
    # State 0: [85,100], State 1: [70,85], etc.
    delta_bin: float = 15.0  # Approximate bin width in PCI units
    
    # Noise scaling parameters
    eta: float = 1.0         # Scaling factor (1.0 = conservative, 2.0 = aggressive)
    q_max: float = 0.3       # Maximum misclassification probability
    
    # Second-order neighbor probability (optional)
    include_second_order: bool = False
    second_order_factor: float = 0.1  # q_err_2 = second_order_factor * q_err


def compute_q_err(R: int, config: ObservationNoiseConfig) -> float:
    """
    Compute observation error probability from rate R.
    
    Formula: q_err = min(η · σ(R) / Δ_bin, q_max)
    where σ(R) = √D(R)
    
    Args:
        R: Rate level (bits)
        config: Noise configuration
        
    Returns:
        q_err: Misclassification probability
    """
    if R not in R_TABLE:
        raise ValueError(f"R must be in {list(R_TABLE.keys())}, got {R}")
    
    D = R_TABLE[R]['D']
    sigma = np.sqrt(D)
    
    q_err = config.eta * sigma / config.delta_bin
    q_err = min(q_err, config.q_max)
    
    return q_err


def build_Q_R_matrix(J: int, q_err: float, 
                     config: ObservationNoiseConfig) -> np.ndarray:
    """
    Build the observation noise channel matrix Q_R(ŝ|s).
    
    Args:
        J: Number of states
        q_err: Misclassification probability
        config: Noise configuration
        
    Returns:
        Q_R: J×J matrix where Q_R[s, s_hat] = P(observe ŝ | true state s)
    """
    Q_R = np.zeros((J, J))
    
    for s in range(J):
        # Probability of correct observation
        Q_R[s, s] = 1 - q_err
        
        # First-order neighbors
        if s > 0:
            Q_R[s, s-1] = q_err / 2
        if s < J - 1:
            Q_R[s, s+1] = q_err / 2
        
        # Handle boundary states (redistribute probability)
        if s == 0:
            Q_R[s, s] += q_err / 2  # No left neighbor
        if s == J - 1:
            Q_R[s, s] += q_err / 2  # No right neighbor
        
        # Optional: second-order neighbors
        if config.include_second_order:
            q_err_2 = config.second_order_factor * q_err
            if s > 1:
                Q_R[s, s-2] = q_err_2 / 2
                Q_R[s, s] -= q_err_2 / 2
            if s < J - 2:
                Q_R[s, s+2] = q_err_2 / 2
                Q_R[s, s] -= q_err_2 / 2
    
    # Verify row stochastic
    row_sums = Q_R.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"Q_R rows must sum to 1, got {row_sums}"
    
    return Q_R


def sample_noisy_observation(s_true: int, Q_R: np.ndarray, 
                             rng: np.random.Generator) -> int:
    """
    Sample a noisy observation given the true state.
    
    Args:
        s_true: True state
        Q_R: Observation noise matrix
        rng: Random number generator
        
    Returns:
        s_hat: Observed (possibly noisy) state
    """
    J = Q_R.shape[0]
    probs = Q_R[s_true, :]
    return rng.choice(J, p=probs)


# =============================================================================
# Modified Environment with Observation Noise
# =============================================================================

class NoisyRMABEnvironment(RMABEnvironment):
    """
    RMAB Environment with observation noise on successful sync.
    
    When sync succeeds, DT receives ŝ ~ Q_R(·|s_true) instead of s_true.
    """
    
    def __init__(self, config: SimulationConfig, 
                 noise_config: ObservationNoiseConfig,
                 seed: int = 42):
        super().__init__(config, seed=seed)
        self.noise_config = noise_config
        self.t = 0  # Initialize time step counter
        
        # Build Q_R matrix
        R = config.experiment.R
        self.q_err = compute_q_err(R, noise_config)
        self.Q_R = build_Q_R_matrix(self.J, self.q_err, noise_config)
    
    def reset(self, seed: Optional[int] = None):
        """Reset environment with new seed."""
        result = super().reset(seed)
        self.t = 0
        return result
        
    def step(self, actions: np.ndarray):
        """
        Execute one step with noisy observations.
        
        Modified from base: on success, h = sample(Q_R(·|s_true))
        instead of h = s_true.
        """
        # Step 1: Evolve all true states (same as base)
        for arm in self.arms:
            P = self.arm_classes[arm.class_idx].P_bar
            probs = P[arm.s_true, :]
            arm.s_true = self.rng_physics.choice(self.J, p=probs)
        
        # Step 2: Process actions with noisy observations
        total_reward = 0.0
        successes = []
        
        for i, arm in enumerate(self.arms):
            if actions[i] == 1:
                # Attempt sync
                success = self.rng_channel.random() < self.arm_classes[arm.class_idx].p_s

                successes.append(success)
                
                if success:
                    # MODIFIED: Apply observation noise
                    h_observed = sample_noisy_observation(
                        arm.s_true, self.Q_R, self.rng_channel
                    )
                    arm.h = h_observed  # DT receives noisy observation
                    arm.delta = 1
                else:
                    arm.delta = min(arm.delta + 1, self.delta_max)
            else:
                successes.append(False)
                arm.delta = min(arm.delta + 1, self.delta_max)
        
        # Compute metrics
        observations = self._get_observations()
        oracle_aoii = np.array([
            self._compute_oracle_aoii(arm) for arm in self.arms
        ])
        control_costs = np.array([
            self._compute_control_cost(arm) for arm in self.arms
        ])
        
        self.t += 1
        
        # Build result using parent's StepResult format
        from environment import StepResult
        return StepResult(
            observations=observations,
            oracle_aoii=oracle_aoii,
            control_costs=control_costs,
            successes=np.array(successes),
            rewards=-oracle_aoii.sum(),
            info={
                'epoch': self.t,
                'mean_oracle_aoii': oracle_aoii.mean(),
                'mean_control_cost': control_costs.mean(),
                'q_err': self.q_err,
                'n_scheduled': actions.sum(),
                'n_success': np.sum(successes),
            }
        )
    
    def _compute_oracle_aoii(self, arm) -> float:
        """Compute oracle AoII for an arm."""
        if arm.s_true == arm.h:
            return 0.0
        else:
            return float(arm.delta)
    
    def _compute_control_cost(self, arm) -> float:
        """Compute control cost for an arm."""
        from config import compute_control_cost
        arm_config = self.arm_classes[arm.class_idx]
        return compute_control_cost(arm.h, arm.delta, arm_config.P_bar)


# =============================================================================
# Sensitivity Analysis Experiment
# =============================================================================

def run_noise_sensitivity(
    output_dir: str = "results",
    N: int = 50,
    M: int = 5,
    T: int = 1000,
    n_seeds: int = 5,
    quick_test: bool = False
) -> pd.DataFrame:
    """
    Run sensitivity analysis for observation noise Q_R.
    
    Compares:
    1. D=0 (perfect observation, baseline)
    2. D>0 with varying η values
    
    Args:
        output_dir: Output directory
        N, M, T: Experiment parameters
        n_seeds: Number of random seeds
        quick_test: Use reduced parameters
    """
    
    Path(f"{output_dir}/data").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
    
    if quick_test:
        T = 500
        n_seeds = 3
        eta_values = [0.0, 1.0, 2.0]
    else:
        eta_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    print("=" * 60)
    print("OBSERVATION NOISE SENSITIVITY ANALYSIS (Q_R)")
    print("=" * 60)
    print(f"N={N}, M={M}, T={T}, seeds={n_seeds}")
    print(f"η values: {eta_values}")
    print("=" * 60)
    
    # Create base config
    config = SimulationConfig()
    config.experiment.N = N
    config.experiment.M = M
    config.experiment.delta_max = 100
    config.experiment.R = 8
    config.arm_classes = get_nhgp_arm_classes(J=5, R=8)
    
    # Compute Whittle tables (same for all η since we use belief-based scheduling)
    solver = WhittleSolver(config.whittle)
    tables = solver.compute_all_tables(config.arm_classes, config.experiment.delta_max, verbose=False)
    whittle_policy = WhittlePolicy(tables)
    
    results_data = []
    seeds = list(range(42, 42 + n_seeds))
    
    for eta in eta_values:
        print(f"\nRunning η={eta:.1f}...")
        
        noise_config = ObservationNoiseConfig(eta=eta)
        q_err = compute_q_err(8, noise_config) if eta > 0 else 0.0
        
        for seed in seeds:
            if eta == 0:
                # Perfect observation (baseline)
                env = RMABEnvironment(config, seed=seed)
            else:
                # Noisy observation
                env = NoisyRMABEnvironment(config, noise_config, seed=seed)
            
            env.reset(seed=seed)
            
            trajectory = []
            for t in range(T):
                obs = env._get_observations()
                actions = whittle_policy.select_arms(obs, env)
                result = env.step(actions)
                trajectory.append(result.info['mean_oracle_aoii'])
            
            # Evaluate on last 50%
            burn_in = T // 2
            mean_aoii = np.mean(trajectory[burn_in:])
            
            results_data.append({
                'eta': eta,
                'q_err': q_err,
                'seed': seed,
                'mean_aoii': mean_aoii,
            })
            
            print(f"  seed={seed}, η={eta:.1f}, q_err={q_err:.4f}, AoII={mean_aoii:.4f}")
    
    # Aggregate results
    df = pd.DataFrame(results_data)
    
    # Summary
    summary = df.groupby('eta').agg({
        'mean_aoii': ['mean', 'std'],
        'q_err': 'first'
    }).round(4)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(summary)
    
    # Compute degradation from baseline
    baseline_aoii = df[df['eta'] == 0]['mean_aoii'].mean()
    print(f"\nBaseline (D=0): {baseline_aoii:.4f}")
    for eta in eta_values:
        if eta > 0:
            eta_aoii = df[df['eta'] == eta]['mean_aoii'].mean()
            degradation = (eta_aoii - baseline_aoii) / baseline_aoii * 100
            print(f"η={eta:.1f}: AoII={eta_aoii:.4f}, degradation=+{degradation:.1f}%")
    
    # Save results
    df.to_csv(f"{output_dir}/data/noise_sensitivity.csv", index=False)
    print(f"\nSaved: {output_dir}/data/noise_sensitivity.csv")
    
    # Generate figure
    _plot_noise_sensitivity(df, output_dir)
    
    return df


def _plot_noise_sensitivity(df: pd.DataFrame, output_dir: str):
    """Generate figure for noise sensitivity analysis."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'figure.figsize': (3.5, 2.5),
        'figure.dpi': 300,
    })
    
    fig, ax = plt.subplots()
    
    # Aggregate by eta
    agg = df.groupby('eta')['mean_aoii'].agg(['mean', 'std']).reset_index()
    
    ax.errorbar(agg['eta'], agg['mean'], yerr=agg['std'], 
                marker='o', capsize=3, linewidth=1.5, markersize=5)
    
    ax.set_xlabel('Noise Scaling Factor $\\eta$')
    ax.set_ylabel('Mean AoII')
    ax.set_title('Observation Noise Sensitivity')
    
    # Add baseline reference
    baseline = agg[agg['eta'] == 0]['mean'].values[0]
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=0.8, label='Baseline (D=0)')
    ax.legend(loc='upper left', fontsize=7)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        fig.savefig(f"{output_dir}/figures/noise_sensitivity.{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/figures/noise_sensitivity.pdf/png")
    
    plt.close(fig)


# =============================================================================
# Print Q_R Matrix for Documentation
# =============================================================================

def print_Q_R_info():
    """Print Q_R matrix information for different R values."""
    
    print("=" * 60)
    print("Q_R OBSERVATION NOISE CHANNEL PARAMETERS")
    print("=" * 60)
    print("\nFormula: q_err = min(η · σ(R) / Δ_bin, q_max)")
    print("         where σ(R) = √D(R), Δ_bin=15 (PCI units), q_max=0.3\n")
    
    config = ObservationNoiseConfig()
    
    for R in [4, 8, 16]:
        D = R_TABLE[R]['D']
        sigma = np.sqrt(D)
        
        for eta in [1.0, 2.0]:
            config.eta = eta
            q_err = compute_q_err(R, config)
            
            print(f"R={R:2d}: D={D:.2e}, σ={sigma:.2e}")
            print(f"       η={eta:.1f} → q_err={q_err:.4f} ({q_err*100:.2f}%)")
        print()
    
    # Example Q_R matrix for R=8, η=1
    config.eta = 1.0
    q_err = compute_q_err(8, config)
    Q_R = build_Q_R_matrix(J=5, q_err=q_err, config=config)
    
    print("Example Q_R matrix (R=8, η=1.0, J=5):")
    print("       ŝ=0    ŝ=1    ŝ=2    ŝ=3    ŝ=4")
    for s in range(5):
        row_str = f"s={s}:  " + "  ".join([f"{Q_R[s,j]:.4f}" for j in range(5)])
        print(row_str)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Observation Noise Sensitivity')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--info', action='store_true', help='Print Q_R info only')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: auto)')
    args = parser.parse_args()
    
    if args.info:
        print_Q_R_info()
    else:
        run_noise_sensitivity(
            output_dir=args.output,
            quick_test=args.quick
        )