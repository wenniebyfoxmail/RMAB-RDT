"""
NHGP-based Transition Matrix Builder (v3 - Fixed)
==================================================

FIXED: Removed recovery_prob to maintain upper triangular structure per DR-06A.

This module constructs the time-homogeneous transition matrix P̄ from NHGP 
(Non-Homogeneous Gamma Process) degradation model following DR-06A specifications.

Key Algorithm:
1. NHGP continuous degradation: X(t) with increments ΔX ~ Gamma(Δα(t), β)
2. Discretize into J bins with boundaries [b_0, b_1, ..., b_J]
3. Compute transition probabilities using Gamma CDF differences
4. Average over time window for time-homogeneous P̄

IMPORTANT (DR-06A Compliance):
- P̄ MUST be upper triangular (irreversible degradation)
- recovery_prob = 0 for main experiments (no self-recovery)
- If recovery is needed, model as external maintenance intervention (扩展实验)
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class NHGPParams:
    """
    Parameters for Non-Homogeneous Gamma Process degradation model.
    
    The shape function is: α(t) = c * t^b
    The increment over [t, t+ΔT] has shape: Δα = c * [(t+ΔT)^b - t^b]
    
    Attributes:
        c: Scale coefficient (higher = faster degradation)
        b: Power exponent (typically 1.0 for linear, >1 for accelerating)
        beta: Rate parameter of Gamma distribution
        delta_T: Epoch duration (time step)
    
    Parameter Sources:
        - c: Synthetic, physics-consistent (可由LTPP校准)
        - b: Typically 1.0 (linear) or >1 (accelerating)
        - beta: Synthetic (影响退化跳跃幅度的随机性)
        - delta_T: 1.0 (one epoch)
    """
    c: float = 0.05        # Degradation rate coefficient [SOURCE: Synthetic]
    b: float = 1.0         # Power exponent [SOURCE: Linear degradation]
    beta: float = 1.0      # Gamma rate parameter [SOURCE: Synthetic]
    delta_T: float = 1.0   # Epoch duration [SOURCE: DR-07]
    
    def compute_delta_alpha(self, t: float) -> float:
        """Compute shape increment Δα(t) for one epoch starting at time t."""
        return self.c * ((t + self.delta_T) ** self.b - t ** self.b)


@dataclass 
class BinConfig:
    """
    Configuration for state space discretization.
    
    Attributes:
        J: Number of bins (states)
        boundaries: Bin boundaries [b_0, b_1, ..., b_J] where b_J = infinity
        representatives: Representative point for each bin (default: left boundary)
    """
    J: int
    boundaries: np.ndarray
    representatives: np.ndarray = None
    
    def __post_init__(self):
        assert len(self.boundaries) == self.J + 1, "Need J+1 boundaries for J bins"
        if self.representatives is None:
            # Default: use left boundary as representative
            self.representatives = self.boundaries[:-1].copy()


def default_bin_config(J: int = 5, max_degradation: float = 10.0) -> BinConfig:
    """
    Create default evenly-spaced bin configuration.
    
    Args:
        J: Number of bins
        max_degradation: Upper bound of non-absorbing region
    
    Returns:
        BinConfig with evenly spaced bins, last bin is absorbing
    """
    # Evenly spaced boundaries, last boundary is infinity (absorbing)
    boundaries = np.linspace(0, max_degradation, J)
    boundaries = np.append(boundaries, np.inf)
    
    return BinConfig(J=J, boundaries=boundaries)


def compute_transition_matrix_at_time(nhgp: NHGPParams, bins: BinConfig, 
                                       t: float) -> np.ndarray:
    """
    Compute transition matrix P(t) at a specific time point.
    
    For current bin j with representative x_j:
    P_{j→k} = F_Gamma(b_{k+1} - x_j; Δα, β) - F_Gamma(b_k - x_j; Δα, β)
    
    IMPORTANT: Returns UPPER TRIANGULAR matrix (irreversible degradation)
    
    Args:
        nhgp: NHGP parameters
        bins: Bin configuration
        t: Current time
    
    Returns:
        P: J x J transition matrix (upper triangular, row stochastic)
    """
    J = bins.J
    P = np.zeros((J, J))
    
    delta_alpha = nhgp.compute_delta_alpha(t)
    
    for j in range(J):
        x_j = bins.representatives[j]
        
        # Last bin is absorbing
        if j == J - 1:
            P[j, j] = 1.0
            continue
        
        # Compute transition probabilities to each possible next bin (k >= j only!)
        for k in range(j, J):  # Only forward transitions
            b_k = bins.boundaries[k]
            b_k_plus_1 = bins.boundaries[k + 1]
            
            # Degradation needed to reach bin k from x_j
            delta_lower = max(0, b_k - x_j)
            delta_upper = b_k_plus_1 - x_j if not np.isinf(b_k_plus_1) else np.inf
            
            # Gamma CDF difference
            if delta_alpha > 0:
                if np.isinf(delta_upper):
                    prob = 1.0 - gamma_dist.cdf(delta_lower, a=delta_alpha, scale=1/nhgp.beta)
                else:
                    prob = (gamma_dist.cdf(delta_upper, a=delta_alpha, scale=1/nhgp.beta) -
                           gamma_dist.cdf(delta_lower, a=delta_alpha, scale=1/nhgp.beta))
            else:
                # No degradation increment: stay in current bin
                prob = 1.0 if k == j else 0.0
            
            P[j, k] = max(0, prob)  # Ensure non-negative
        
        # Normalize row to ensure stochastic
        row_sum = P[j, :].sum()
        if row_sum > 0:
            P[j, :] /= row_sum
        else:
            P[j, j] = 1.0  # Fallback: stay in current bin
    
    return P


def compute_time_averaged_transition_matrix(nhgp: NHGPParams, bins: BinConfig,
                                            t_start: float, t_end: float,
                                            n_samples: int = 10) -> np.ndarray:
    """
    Compute time-averaged transition matrix P̄ over a time window.
    
    P̄ = (1/K) * Σ_k P(t_k)
    
    Args:
        nhgp: NHGP parameters
        bins: Bin configuration
        t_start: Start of time window
        t_end: End of time window
        n_samples: Number of sample points
    
    Returns:
        P_bar: Time-averaged J x J transition matrix (upper triangular)
    """
    sample_times = np.linspace(t_start, t_end, n_samples)
    
    P_sum = np.zeros((bins.J, bins.J))
    for t in sample_times:
        P_sum += compute_transition_matrix_at_time(nhgp, bins, t)
    
    P_bar = P_sum / n_samples
    
    # Ensure row stochastic after averaging
    for j in range(bins.J):
        row_sum = P_bar[j, :].sum()
        if row_sum > 0:
            P_bar[j, :] /= row_sum
    
    return P_bar


def create_heterogeneous_classes(base_nhgp: NHGPParams, bins: BinConfig,
                                  c_ratios: List[float],
                                  class_names: List[str],
                                  t_start: float = 0.0,
                                  t_end: float = 100.0) -> List[Tuple[str, np.ndarray]]:
    """
    Create heterogeneous arm classes by scaling the degradation rate c.
    
    Args:
        base_nhgp: Base NHGP parameters
        bins: Bin configuration
        c_ratios: List of scaling factors for c (e.g., [1.0, 2.0] for slow/fast)
        class_names: Names for each class
        t_start, t_end: Time window for averaging
    
    Returns:
        List of (class_name, P_bar) tuples
    """
    classes = []
    
    for name, ratio in zip(class_names, c_ratios):
        # Create scaled NHGP
        scaled_nhgp = NHGPParams(
            c=base_nhgp.c * ratio,
            b=base_nhgp.b,
            beta=base_nhgp.beta,
            delta_T=base_nhgp.delta_T
        )
        
        # Compute time-averaged transition matrix
        P_bar = compute_time_averaged_transition_matrix(
            scaled_nhgp, bins, t_start, t_end
        )
        
        classes.append((name, P_bar))
    
    return classes


# Default configurations for the simulation
def get_default_nhgp_classes(J: int = 5, 
                              recovery_prob: float = 0.0,  # FIXED: Default to 0
                              ) -> List[Tuple[str, np.ndarray, float]]:
    """
    Get default NHGP-based arm classes for simulation.
    
    Args:
        J: Number of degradation bins
        recovery_prob: Recovery probability (DEFAULT: 0.0 for main experiments)
                      Set > 0 only for sensitivity analysis (外生维护事件)
    
    Returns:
        List of (name, P_bar, c_ratio) tuples
    
    IMPORTANT (DR-06A Compliance):
        - Main experiments: recovery_prob = 0.0 (upper triangular P̄)
        - Sensitivity analysis: recovery_prob > 0 as "external maintenance"
    """
    # Base NHGP parameters (CALIBRATED for meaningful experiments)
    # SOURCE: Synthetic but physics-consistent
    base_nhgp = NHGPParams(
        c=0.20,      # ← 退化率提高4倍
        b=1.0,
        beta=1.0,
        delta_T=1.0
    )
    
    # Bin configuration
    bins = default_bin_config(J=J, max_degradation=5.0)
    
    # Two classes: slow (baseline) and fast (2x degradation rate)
    classes = create_heterogeneous_classes(
        base_nhgp, bins,
        c_ratios=[1.0, 2.0],
        class_names=["slow", "fast"],
        t_start=0.0,
        t_end=500.0
    )
    
    result = []
    for (name, P_bar), ratio in zip(classes, [1.0, 2.0]):
        # FIXED: Only add recovery if explicitly requested (for sensitivity analysis)
        # Main line: recovery_prob = 0.0 (maintain upper triangular structure)
        if recovery_prob > 0:
            # This models external maintenance intervention (NOT self-recovery)
            # WARNING: This breaks strict upper triangular structure
            import warnings
            warnings.warn(
                f"recovery_prob={recovery_prob} > 0 breaks upper triangular structure. "
                "Only use for sensitivity analysis (labeled as 'external maintenance')."
            )
            J_local = P_bar.shape[0]
            P_bar[J_local-1, J_local-1] = 1.0 - recovery_prob
            P_bar[J_local-1, J_local-2] = recovery_prob
        
        result.append((name, P_bar, ratio))
    
    return result


def verify_upper_triangular(P_bar: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Verify that transition matrix is upper triangular (irreversible degradation).
    
    Args:
        P_bar: Transition matrix
        tolerance: Numerical tolerance for lower triangular entries
    
    Returns:
        True if upper triangular (within tolerance)
    """
    lower_tri = np.tril(P_bar, -1)
    return np.all(np.abs(lower_tri) < tolerance)


if __name__ == "__main__":
    print("=== NHGP Transition Matrix Builder Test (v3 - Fixed) ===\n")
    
    # Test basic NHGP parameters
    nhgp = NHGPParams(c=0.05, b=1.0, beta=1.0, delta_T=1.0)
    bins = default_bin_config(J=5, max_degradation=5.0)
    
    print(f"NHGP params: c={nhgp.c}, b={nhgp.b}, β={nhgp.beta}")
    print(f"Bins: {bins.boundaries}")
    print(f"Representatives: {bins.representatives}")
    
    # Test transition matrix at different times
    print("\n=== Transition Matrices at Different Times ===")
    for t in [0, 50, 100, 200]:
        P = compute_transition_matrix_at_time(nhgp, bins, t)
        print(f"\nP(t={t}):")
        print(np.round(P, 4))
        print(f"Upper triangular: {verify_upper_triangular(P)}")
    
    # Test time-averaged matrix
    print("\n=== Time-Averaged Transition Matrix ===")
    P_bar = compute_time_averaged_transition_matrix(nhgp, bins, 0, 500, n_samples=20)
    print("P̄ (averaged over t=0 to 500):")
    print(np.round(P_bar, 4))
    
    # Test heterogeneous classes (MAIN LINE: no recovery)
    print("\n=== Heterogeneous Classes (Main Line: recovery_prob=0) ===")
    classes = get_default_nhgp_classes(J=5, recovery_prob=0.0)
    for name, P_bar, c_ratio in classes:
        print(f"\nClass '{name}' (c_ratio={c_ratio}):")
        print(np.round(P_bar, 4))
        
        # Verify properties
        print(f"  Row sums: {P_bar.sum(axis=1).round(6)}")
        is_upper = verify_upper_triangular(P_bar)
        print(f"  Upper triangular: {is_upper}")
        if not is_upper:
            print("  ⚠️ WARNING: Matrix is NOT upper triangular!")
    
    # Test with recovery (sensitivity analysis only)
    print("\n=== Sensitivity Analysis: With External Maintenance (recovery=0.01) ===")
    classes_recovery = get_default_nhgp_classes(J=5, recovery_prob=0.01)
    for name, P_bar, c_ratio in classes_recovery:
        print(f"\nClass '{name}' (with maintenance):")
        print(np.round(P_bar, 4))
        print(f"  Upper triangular: {verify_upper_triangular(P_bar)}")
