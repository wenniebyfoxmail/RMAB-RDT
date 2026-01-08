"""
Indexability Empirical Verifier for AoII-Aware RMAB
=====================================================

This module provides empirical verification of the indexability condition
required for Whittle Index policy to be well-defined.

Indexability Condition:
-----------------------
For each state z=(h,Δ), as λ increases from -∞ to +∞, the optimal action
should monotonically transition from Active (a=1) to Passive (a=0).
Equivalently, the "passive set" P(λ) = {z: a*(z,λ)=0} should be monotonically
non-decreasing in λ.

Usage:
------
    from indexability_verifier import IndexabilityVerifier
    
    verifier = IndexabilityVerifier(arm_class, delta_max, whittle_config)
    is_indexable, report = verifier.verify()

Output for Paper:
-----------------
- Violation rate (% of states violating monotonicity)
- Example violating states (if any)
- Appendix-ready figure showing passive set expansion

Reference:
----------
DR-06C §B.3: "Indexability 的经验验证器"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import time

from config import (
    WhittleConfig, ArmClassConfig, 
    BeliefCache, compute_control_cost
)


@dataclass
class IndexabilityReport:
    """Report from indexability verification."""
    is_indexable: bool
    violation_rate: float  # Fraction of states violating monotonicity
    violating_states: List[Tuple[int, int]]  # List of (h, Δ) that violate
    lambda_samples: np.ndarray
    passive_counts: np.ndarray  # Number of passive states at each λ
    action_matrix: np.ndarray  # Shape (n_lambda, J, delta_max): optimal action
    computation_time: float
    
    def __str__(self):
        status = "✅ INDEXABLE" if self.is_indexable else "❌ NOT INDEXABLE"
        return (
            f"Indexability Verification Report\n"
            f"================================\n"
            f"Status: {status}\n"
            f"Violation Rate: {self.violation_rate:.2%}\n"
            f"Violating States: {len(self.violating_states)}\n"
            f"Computation Time: {self.computation_time:.2f}s\n"
        )


class IndexabilityVerifier:
    """
    Empirical verifier for indexability condition.
    
    Algorithm:
    1. For each λ in a fine grid, solve single-arm MDP via VI
    2. Record optimal action for each state z=(h,Δ)
    3. Check monotonicity: as λ↑, optimal action should flip 0→1 at most once
    """
    
    def __init__(self, arm_class: ArmClassConfig, delta_max: int,
                 whittle_config: WhittleConfig):
        self.arm_class = arm_class
        self.delta_max = delta_max
        self.config = whittle_config
        
        self.J = arm_class.P_bar.shape[0]
        self.p_s = arm_class.p_s
        self.gamma = whittle_config.gamma
        
        # Precompute costs and beliefs
        self._precompute()
    
    def _precompute(self):
        """Precompute costs and evolved beliefs."""
        self.costs = np.zeros((self.J, self.delta_max))
        self.beliefs_evolved = np.zeros((self.J, self.delta_max, self.J))
        
        belief_cache = BeliefCache(self.arm_class.P_bar, self.delta_max)
        
        for h in range(self.J):
            for d_idx in range(self.delta_max):
                delta = d_idx + 1
                self.costs[h, d_idx] = compute_control_cost(
                    h, delta, self.arm_class.P_bar
                )
                self.beliefs_evolved[h, d_idx, :] = belief_cache.get_belief_evolved(
                    h, delta
                )
    
    def solve_vi(self, lam: float, 
                 V_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve single-arm MDP via Value Iteration for fixed λ.
        
        Returns:
            V: Value function (J, delta_max)
            optimal_action: Optimal action matrix (J, delta_max), 0=passive, 1=active
        """
        J, delta_max = self.J, self.delta_max
        
        V = V_init.copy() if V_init is not None else np.zeros((J, delta_max))
        
        d_indices = np.arange(delta_max)
        next_d_idx = np.minimum(d_indices + 1, delta_max - 1)
        
        for _ in range(self.config.vi_max_iter):
            V_old = V.copy()
            
            # Q_passive(h, d) = C(h,d) + γ * V(h, next_d)
            Q_passive = self.costs + self.gamma * V_old[np.arange(J)[:, None], next_d_idx]
            
            # Q_active: with subsidy λ
            V_success = np.einsum('hdj,j->hd', self.beliefs_evolved, V_old[:, 0])
            V_fail = V_old[np.arange(J)[:, None], next_d_idx]
            
            Q_active = self.costs + lam + self.gamma * (
                self.p_s * V_success + (1 - self.p_s) * V_fail
            )
            
            # Bellman update
            V = np.minimum(Q_passive, Q_active)
            
            # Relative normalization
            V = V - V[0, 0]
            
            if np.max(np.abs(V - V_old)) < self.config.vi_threshold:
                break
        
        # Compute optimal action: 1 if Q_active < Q_passive, else 0
        # Note: active is optimal when Q_active < Q_passive (i.e., Q_diff < 0)
        Q_diff = Q_active - Q_passive
        optimal_action = (Q_diff < 0).astype(int)
        
        return V, optimal_action
    
    def verify(self, n_lambda: int = 50,
               lambda_min: float = -100.0,
               lambda_max: float = 500.0,
               verbose: bool = True) -> Tuple[bool, IndexabilityReport]:
        """
        Verify indexability by checking passive set monotonicity.
        
        Args:
            n_lambda: Number of λ samples
            lambda_min, lambda_max: Range of λ to test
            verbose: Print progress
        
        Returns:
            is_indexable: True if indexability holds
            report: Detailed verification report
        """
        start_time = time.time()
        
        if verbose:
            print(f"Verifying indexability for class '{self.arm_class.name}'...")
            print(f"  λ range: [{lambda_min}, {lambda_max}], samples: {n_lambda}")
        
        lambdas = np.linspace(lambda_min, lambda_max, n_lambda)
        
        # Store optimal action at each λ
        action_history = np.zeros((n_lambda, self.J, self.delta_max), dtype=int)
        passive_counts = np.zeros(n_lambda, dtype=int)
        
        V_prev = None
        for i, lam in enumerate(lambdas):
            V, actions = self.solve_vi(lam, V_prev)
            V_prev = V
            
            action_history[i] = actions
            passive_counts[i] = np.sum(actions == 0)  # Count passive states
        
        # Check monotonicity for each state
        violating_states = []
        
        for h in range(self.J):
            for d_idx in range(self.delta_max):
                actions_seq = action_history[:, h, d_idx]
                
                # Find transitions: should be 1→0 (at most once), not 0→1
                transitions = np.diff(actions_seq)
                
                # Violation: any transition 0→1 (value +1)
                if np.any(transitions > 0):
                    violating_states.append((h, d_idx + 1))
        
        # Check overall monotonicity of passive set
        is_monotone = all(
            passive_counts[i] <= passive_counts[i+1] 
            for i in range(len(passive_counts) - 1)
        )
        
        is_indexable = len(violating_states) == 0 and is_monotone
        violation_rate = len(violating_states) / (self.J * self.delta_max)
        
        computation_time = time.time() - start_time
        
        if verbose:
            status = "✅ INDEXABLE" if is_indexable else "❌ VIOLATIONS FOUND"
            print(f"  Result: {status}")
            print(f"  Violation rate: {violation_rate:.2%}")
            print(f"  Time: {computation_time:.2f}s")
        
        report = IndexabilityReport(
            is_indexable=is_indexable,
            violation_rate=violation_rate,
            violating_states=violating_states,
            lambda_samples=lambdas,
            passive_counts=passive_counts,
            action_matrix=action_history,
            computation_time=computation_time
        )
        
        return is_indexable, report
    
    def plot_passive_set_evolution(self, report: IndexabilityReport,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot passive set size vs λ (for appendix).
        
        Args:
            report: Indexability report from verify()
            save_path: Path to save figure (optional)
        
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        
        ax.plot(report.lambda_samples, report.passive_counts, 
                'b-', linewidth=1.5, marker='o', markersize=2)
        
        ax.set_xlabel('Subsidy $\\lambda$', fontsize=8)
        ax.set_ylabel('Passive Set Size $|P(\\lambda)|$', fontsize=8)
        
        # Mark expected monotonic increase
        if report.is_indexable:
            ax.set_title(f'Indexability Verified (Class: {self.arm_class.name})', 
                        fontsize=8, color='green')
        else:
            ax.set_title(f'Indexability Violation (Class: {self.arm_class.name})', 
                        fontsize=8, color='red')
        
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_action_heatmap(self, report: IndexabilityReport,
                            lambda_index: int = -1,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot optimal action heatmap for a specific λ.
        
        Args:
            report: Indexability report
            lambda_index: Index of λ to visualize (-1 for last)
            save_path: Path to save figure
        
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(4, 3))
        
        lam = report.lambda_samples[lambda_index]
        actions = report.action_matrix[lambda_index]
        
        # Limit Δ for visualization
        delta_vis = min(50, self.delta_max)
        actions_vis = actions[:, :delta_vis]
        
        im = ax.imshow(actions_vis, aspect='auto', cmap='RdYlGn', 
                       origin='lower', vmin=0, vmax=1)
        
        ax.set_xlabel('Age $\\Delta$', fontsize=8)
        ax.set_ylabel('Last State $h$', fontsize=8)
        ax.set_title(f'Optimal Action ($\\lambda$={lam:.1f}): Green=Active, Red=Passive', 
                    fontsize=8)
        
        ax.set_yticks(range(self.J))
        
        plt.colorbar(im, ax=ax, label='Action', ticks=[0, 1])
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig


def verify_all_classes(arm_classes: List[ArmClassConfig],
                       delta_max: int,
                       whittle_config: WhittleConfig,
                       output_dir: str = None,
                       verbose: bool = True) -> Dict[str, IndexabilityReport]:
    """
    Verify indexability for all arm classes.
    
    Args:
        arm_classes: List of arm class configurations
        delta_max: Maximum age
        whittle_config: Whittle solver configuration
        output_dir: Directory to save figures (optional)
        verbose: Print progress
    
    Returns:
        Dictionary mapping class name to report
    """
    reports = {}
    
    for arm_class in arm_classes:
        verifier = IndexabilityVerifier(arm_class, delta_max, whittle_config)
        is_indexable, report = verifier.verify(verbose=verbose)
        reports[arm_class.name] = report
        
        if output_dir:
            from pathlib import Path
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save passive set evolution plot
            fig1 = verifier.plot_passive_set_evolution(
                report, f"{output_dir}/indexability_{arm_class.name}.png"
            )
            plt.close(fig1)
            
            # Save action heatmap for high λ
            fig2 = verifier.plot_action_heatmap(
                report, lambda_index=-1,
                save_path=f"{output_dir}/actions_{arm_class.name}.png"
            )
            plt.close(fig2)
    
    return reports


def generate_indexability_summary(reports: Dict[str, IndexabilityReport]) -> str:
    """
    Generate summary table for paper appendix.
    
    Args:
        reports: Dictionary of indexability reports
    
    Returns:
        Markdown-formatted summary table
    """
    lines = [
        "| Class | Indexable | Violation Rate | Violating States |",
        "|-------|-----------|----------------|------------------|"
    ]
    
    for name, report in reports.items():
        status = "✅ Yes" if report.is_indexable else "❌ No"
        viol_rate = f"{report.violation_rate:.2%}"
        viol_states = len(report.violating_states)
        lines.append(f"| {name} | {status} | {viol_rate} | {viol_states} |")
    
    return "\n".join(lines)


if __name__ == "__main__":
    from config import SimulationConfig
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    print("=== Indexability Verification Test ===\n")
    
    config = SimulationConfig()
    
    # Verify all classes
    reports = verify_all_classes(
        config.arm_classes,
        delta_max=100,
        whittle_config=config.whittle,
        output_dir="results/indexability",
        verbose=True
    )
    
    # Print summary
    print("\n=== Summary ===")
    print(generate_indexability_summary(reports))
    
    # Detailed report for each class
    for name, report in reports.items():
        print(f"\n{report}")
        
        if not report.is_indexable:
            print(f"Violating states (h, Δ): {report.violating_states[:10]}...")
