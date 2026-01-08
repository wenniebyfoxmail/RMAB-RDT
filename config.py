"""
Configuration for Road Digital Twin AoII-ARD RMAB Simulation (v2)
==================================================================

UPDATED based on advisor feedback to fix:
1. P̄ now constructed from NHGP (not hand-crafted)
2. p_s uses DR-06B R-table values (R=8: 0.996)
3. Belief uses P^(Δ-1) so Δ=1 gives one-hot
4. Active success distribution uses belief @ P (evolved state)

Key Assumptions (updated):
- A1: Observable state z=(h,Δ), true state s is latent
- A2: Belief formula π = e_h^T P^(Δ-1) (Δ=1 → one-hot)  [FIXED]
- A3: f(Δ)=Δ (linear), g(d)=d (quantized)
- A4: After success Δ=1 (not 0)
- A5: λ is active penalty, higher W means more urgent
- A6: First evolve s, then action, success syncs to s(t+1)
- A7: Mainline D=0 (perfect observation)

=============================================================================
ENGINEERING SEMANTICS (Advisor Decisions Q3-Q5)
=============================================================================

TIME MAPPING (Q3):
    1 epoch = 1 month (maintenance planning interval)
    - Δ_max=100 corresponds to ~8 years without update
    - T=2000 epochs = 167 years of simulation (for Monte Carlo averaging)
    - Typical inspection/update cycles: 1-3 months for critical corridors
    
STATE MAPPING (Q4) - PCI-based categorization:
    One commonly used categorization in asset management practice.
    Reference: FHWA LTPP User Guide, Pima County AZ Pavement Management.
    
    | State j | Physical Meaning | Typical PCI Range |
    |---------|------------------|-------------------|
    |    0    | Good             | 85 - 100          |
    |    1    | Fair             | 70 - 84           |
    |    2    | Poor             | 55 - 69           |
    |    3    | Very Poor        | 40 - 54           |
    |    4    | Failed           | < 40              |
    
    Note: Thresholds are adjustable per agency standards. The model is
    invariant to the specific numeric thresholds; only the discrete state
    transitions matter.

ARM CLASS MAPPING (Q5) - Traffic load differentiation:
    Class-H (High-load corridor / "fast" degradation):
        - Major arterials, freight routes, high ESAL (heavy truck traffic)
        - c_fast ≈ 0.02 (faster NHGP degradation rate)
        - Typical: Interstate highways, container port access roads
        
    Class-L (Light-load local roads / "slow" degradation):
        - Residential streets, rural collectors, low ESAL
        - c_slow ≈ 0.01 (slower NHGP degradation rate)
        - Typical: Neighborhood streets, rural county roads
        
    Scaling: c_fast = 2 × c_slow (factor of 2, as per advisor guidance)
    
    Alternative interpretation (climate-driven):
        - Fast: Extreme climate zones (freeze-thaw cycles, high temperature range)
        - Slow: Temperate/stable climate zones

LTPP CALIBRATION NOTE (Q6):
    The synthetic parameters are physics-consistent with LTPP/InfoPave data:
    - P_ii ≈ 0.94 implies ~6% monthly degradation probability
    - This corresponds to ~50% remaining in "Good" state after 1 year
    - Consistent with published deterioration curves in pavement engineering
    Reference: FHWA-HRT-21-038 (LTPP User Guide)
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from numpy.random import Generator, PCG64, SeedSequence

from nhgp_builder import (
    NHGPParams, BinConfig, default_bin_config,
    compute_time_averaged_transition_matrix,
    get_default_nhgp_classes
)


# =============================================================================
# DR-06B R-Table Interface
# =============================================================================

# DR-06B Table: Rate level R → (p_s, D)
R_TABLE = {
    4:  {'p_s': 0.998, 'D': 0.0039},
    8:  {'p_s': 0.996, 'D': 1.5e-5},
    16: {'p_s': 0.989, 'D': 2.3e-10},
    # 添加以下行 ↓
    32: {'p_s': 0.90, 'D': 1e-6},   # 新增：90%成功率
    64: {'p_s': 0.80, 'D': 1e-6},   # 新增：80%成功率
}

def get_channel_params(R: int) -> Tuple[float, float]:
    """Get (p_s, D) from DR-06B R-table."""
    if R not in R_TABLE:
        raise ValueError(f"R must be in {list(R_TABLE.keys())}, got {R}")
    return R_TABLE[R]['p_s'], R_TABLE[R]['D']


# =============================================================================
# Arm Class Configuration
# =============================================================================

@dataclass
class ArmClassConfig:
    """Configuration for a class of arms (road segments)."""
    name: str
    P_bar: np.ndarray  # J x J transition matrix (upper triangular, stochastic)
    p_s: float         # Success probability from DR-06B R-table
    D: float = 0.0     # Observation distortion (mainline: 0)
    R: int = 32         # Rate level (for documentation)
    c_ratio: float = 1.0  # NHGP c scaling ratio (for documentation)
    
    def __post_init__(self):
        """Validate transition matrix."""
        J = self.P_bar.shape[0]
        assert self.P_bar.shape == (J, J), "P_bar must be square"
        # Check row stochastic
        row_sums = self.P_bar.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"P_bar rows must sum to 1, got {row_sums}"
        # Check mostly upper triangular (allow small recovery probability)
        lower_tri = np.tril(self.P_bar, -1)
        lower_sum = lower_tri.sum()
        if lower_sum > 0.1:  # Allow up to 10% total recovery probability
            import warnings
            warnings.warn(f"P_bar has {lower_sum:.2%} lower-triangular mass (recovery)")


# =============================================================================
# Whittle Solver Configuration
# =============================================================================

@dataclass
class WhittleConfig:
    """Configuration for Whittle Index computation."""
    gamma: float = 0.999           # Discount factor for VI
    epsilon: float = 1e-3          # Binary search precision
    vi_threshold: float = 1e-5     # VI convergence threshold
    vi_max_iter: int = 1000        # Maximum VI iterations
    use_hot_start: bool = True     # Use hot-start for VI
    ref_state: Tuple[int, int] = (0, 1)  # Reference state for normalization (h, Δ)
    
    # Adaptive bracket for λ search (replaces hardcoded bounds)
    lambda_init: float = 10.0      # Initial bracket half-width
    lambda_max: float = 10000.0    # Maximum bracket (safety limit)
    bracket_expand_factor: float = 2.0  # Expansion factor for bracket


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass 
class ExperimentConfig:
    """Configuration for experiments."""
    N: int = 50                    # Number of arms
    M: int = 5                     # Number of arms to schedule per epoch
    J: int = 5                     # Number of degradation bins
    delta_max: int = 100           # Maximum age
    T: int = 2000                  # Total epochs (reduced from 5000 per advisor)
    n_seeds: int = 10              # Number of random seeds
    burn_in_ratio: float = 0.5     # Burn-in ratio for evaluation
    
    # Arm class distribution
    class_distribution: List[int] = field(default_factory=lambda: [25, 25])
    
    # Random seeds for reproducibility
    master_seed: int = 42
    
    # DR-06B channel configuration
    R: int = 8                     # Default rate level (main experiment)


# =============================================================================
# Master Simulation Configuration
# =============================================================================

@dataclass
class SimulationConfig:
    """Master configuration combining all settings."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    whittle: WhittleConfig = field(default_factory=WhittleConfig)
    arm_classes: List[ArmClassConfig] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default arm classes if not provided."""
        if not self.arm_classes:
            self.arm_classes = get_nhgp_arm_classes(
                J=self.experiment.J,
                R=self.experiment.R
            )


def get_nhgp_arm_classes(J: int = 5, R: int = 8, 
                         recovery_prob: float = 0.0) -> List[ArmClassConfig]:
    """
    Get arm classes with NHGP-derived transition matrices.
    
    Args:
        J: Number of degradation bins
        R: Rate level from DR-06B table
        recovery_prob: Recovery probability (DEFAULT: 0.0 for main experiments)
                      Set > 0 only for sensitivity analysis (labeled as 'external maintenance')
    
    Returns:
        List of ArmClassConfig with physics-based P̄
    
    IMPORTANT (DR-06A Compliance):
        - Main experiments: recovery_prob = 0.0 (upper triangular P̄)
        - Sensitivity analysis only: recovery_prob > 0
    """
    # Get channel parameters from DR-06B
    p_s, D = get_channel_params(R)
    
    # Get NHGP-based transition matrices (with recovery_prob parameter)
    nhgp_classes = get_default_nhgp_classes(J=J, recovery_prob=recovery_prob)
    
    arm_classes = []
    for name, P_bar, c_ratio in nhgp_classes:
        arm_classes.append(ArmClassConfig(
            name=name,
            P_bar=P_bar,
            p_s=p_s,
            D=D,
            R=R,
            c_ratio=c_ratio
        ))
    
    return arm_classes


# =============================================================================
# Cost Functions (following DR-06C)
# =============================================================================

def f_age(delta: int) -> float:
    """Time penalty function f(Δ) = Δ (linear)."""
    return float(delta)


def g_semantic(d: int) -> float:
    """Semantic error function g(d) = |d| (quantized AoII)."""
    return float(abs(d))


def g_binary(d: int) -> float:
    """Binary semantic error function g(d) = 1{d ≠ 0}."""
    return 1.0 if d != 0 else 0.0


# =============================================================================
# Belief Computation (FIXED: uses P^(Δ-1) per advisor feedback)
# =============================================================================

def compute_belief(h: int, delta: int, P_bar: np.ndarray) -> np.ndarray:
    """
    Compute belief distribution over true state s given (h, Δ).
    
    IMPORTANT: Uses P^(Δ-1) so that Δ=1 gives one-hot (just synchronized).
    
    π_{h,Δ} = e_h^T @ P_bar^(Δ-1)
    
    Args:
        h: Last synchronized state (bin index, 0-indexed)
        delta: Age since last sync (1-indexed, Δ≥1)
        P_bar: Transition matrix (J x J)
    
    Returns:
        belief: Probability distribution over J states
    """
    J = P_bar.shape[0]
    e_h = np.zeros(J)
    e_h[h] = 1.0
    
    # FIXED: Use delta-1 so Δ=1 gives e_h (one-hot)
    power = max(0, delta - 1)
    
    if power == 0:
        return e_h
    
    P_power = np.linalg.matrix_power(P_bar, power)
    belief = e_h @ P_power
    
    return belief


def compute_belief_after_evolution(h: int, delta: int, P_bar: np.ndarray) -> np.ndarray:
    """
    Compute belief distribution after one more evolution step.
    
    This is used for Myopic policy and Whittle active-success transition.
    
    π^+ = π_{h,Δ} @ P_bar
    
    Args:
        h: Last synchronized state
        delta: Current age
        P_bar: Transition matrix
    
    Returns:
        belief_evolved: Distribution over s(t+1)
    """
    belief_now = compute_belief(h, delta, P_bar)
    return belief_now @ P_bar


# =============================================================================
# Control Cost Computation
# =============================================================================

def compute_control_cost(h: int, delta: int, P_bar: np.ndarray,
                         f_func=f_age, g_func=g_semantic) -> float:
    """
    Compute control cost C(h, Δ) for Whittle Index optimization.
    
    C(h, Δ) = f(Δ) * Σ_j g(|j - h|) * π_{h,Δ}(j)
    
    Args:
        h: Last synchronized state
        delta: Age since last sync
        P_bar: Transition matrix
        f_func: Time penalty function
        g_func: Semantic error function
    
    Returns:
        cost: Control cost value
    """
    J = P_bar.shape[0]
    belief = compute_belief(h, delta, P_bar)
    
    expected_g = sum(g_func(j - h) * belief[j] for j in range(J))
    cost = f_func(delta) * expected_g
    
    return cost


def compute_oracle_aoii(s: int, h: int, delta: int, 
                        g_func=g_semantic) -> float:
    """
    Compute Oracle AoII for evaluation (requires true state s).
    
    AoII_oracle = Δ * g(|s - h|)
    
    Args:
        s: True physical state
        h: DT estimated state
        delta: Age since last sync
        g_func: Semantic error function
    
    Returns:
        aoii: Oracle AoII value
    """
    return float(delta) * g_func(s - h)


# =============================================================================
# Precomputation Utilities
# =============================================================================

class BeliefCache:
    """
    Cache for precomputed beliefs to avoid redundant matrix powers.
    """
    
    def __init__(self, P_bar: np.ndarray, delta_max: int):
        self.J = P_bar.shape[0]
        self.delta_max = delta_max
        self.P_bar = P_bar
        
        # Precompute all beliefs: shape (J, delta_max, J)
        # beliefs[h, delta-1, :] = π_{h,Δ}
        self._beliefs = np.zeros((self.J, delta_max, self.J))
        self._beliefs_evolved = np.zeros((self.J, delta_max, self.J))
        
        self._precompute()
    
    def _precompute(self):
        """Precompute all belief distributions."""
        for h in range(self.J):
            for d_idx in range(self.delta_max):
                delta = d_idx + 1
                self._beliefs[h, d_idx, :] = compute_belief(h, delta, self.P_bar)
                self._beliefs_evolved[h, d_idx, :] = compute_belief_after_evolution(
                    h, delta, self.P_bar
                )
    
    def get_belief(self, h: int, delta: int) -> np.ndarray:
        """Get cached belief π_{h,Δ}."""
        delta = max(1, min(delta, self.delta_max))
        return self._beliefs[h, delta - 1, :]
    
    def get_belief_evolved(self, h: int, delta: int) -> np.ndarray:
        """Get cached belief after evolution π_{h,Δ} @ P."""
        delta = max(1, min(delta, self.delta_max))
        return self._beliefs_evolved[h, delta - 1, :]


class CostCache:
    """
    Cache for precomputed control costs.
    """
    
    def __init__(self, P_bar: np.ndarray, delta_max: int):
        self.J = P_bar.shape[0]
        self.delta_max = delta_max
        
        # Precompute all costs: shape (J, delta_max)
        self._costs = np.zeros((self.J, delta_max))
        
        for h in range(self.J):
            for d_idx in range(delta_max):
                delta = d_idx + 1
                self._costs[h, d_idx] = compute_control_cost(h, delta, P_bar)
    
    def get_cost(self, h: int, delta: int) -> float:
        """Get cached control cost C(h, Δ)."""
        delta = max(1, min(delta, self.delta_max))
        return self._costs[h, delta - 1]


# =============================================================================
# Random State Manager
# =============================================================================

class RandomStateManager:
    """
    Manages random number generators for reproducible simulations.
    
    Uses NumPy's SeedSequence to spawn independent generators for different
    components, ensuring that adding/removing components doesn't affect
    the randomness of others.
    """
    
    def __init__(self, master_seed: int = 42):
        self.master_seed = master_seed
        self.seed_sequence = SeedSequence(master_seed)
        self._generators: Dict[str, Generator] = {}
        
    def get_generator(self, name: str) -> Generator:
        """Get or create a generator for the given component name."""
        if name not in self._generators:
            child_seed = self.seed_sequence.spawn(1)[0]
            self._generators[name] = Generator(PCG64(child_seed))
        return self._generators[name]
    
    def reset(self, master_seed: Optional[int] = None):
        """Reset all generators with a new master seed."""
        if master_seed is not None:
            self.master_seed = master_seed
        self.seed_sequence = SeedSequence(self.master_seed)
        self._generators.clear()


# Default configuration instance
DEFAULT_CONFIG = SimulationConfig()


if __name__ == "__main__":
    print("=== Configuration Test (v2 - NHGP Based) ===\n")
    
    config = SimulationConfig()
    print(f"N={config.experiment.N}, M={config.experiment.M}, J={config.experiment.J}")
    print(f"Δ_max={config.experiment.delta_max}, T={config.experiment.T}")
    print(f"R={config.experiment.R} (from DR-06B)")
    
    print(f"\nNumber of arm classes: {len(config.arm_classes)}")
    for i, arm_class in enumerate(config.arm_classes):
        print(f"\nClass {i+1} ({arm_class.name}):")
        print(f"  p_s = {arm_class.p_s} (R={arm_class.R})")
        print(f"  D = {arm_class.D}")
        print(f"  c_ratio = {arm_class.c_ratio}")
        print(f"  P_bar =\n{np.round(arm_class.P_bar, 4)}")
    
    # Test belief computation (verify Δ=1 gives one-hot)
    print("\n=== Belief Test (FIXED: Δ=1 should give one-hot) ===")
    P = config.arm_classes[0].P_bar
    
    for h in [0, 2, 4]:
        belief_d1 = compute_belief(h, delta=1, P_bar=P)
        print(f"Belief(h={h}, Δ=1): {belief_d1.round(4)}")
        assert np.argmax(belief_d1) == h, f"Δ=1 should give one-hot at h={h}"
    
    print("\nBeliefs for h=0, various Δ:")
    for delta in [1, 2, 5, 10, 20]:
        belief = compute_belief(h=0, delta=delta, P_bar=P)
        print(f"  Δ={delta:2d}: {belief.round(4)}")
    
    # Test belief after evolution
    print("\n=== Belief After Evolution ===")
    belief_now = compute_belief(0, 5, P)
    belief_evolved = compute_belief_after_evolution(0, 5, P)
    print(f"π(h=0,Δ=5):     {belief_now.round(4)}")
    print(f"π(h=0,Δ=5) @ P: {belief_evolved.round(4)}")
    
    # Test cost computation
    print("\n=== Control Cost Test ===")
    for h in [0, 2, 4]:
        for delta in [1, 10, 50]:
            cost = compute_control_cost(h, delta, P)
            print(f"C(h={h}, Δ={delta}) = {cost:.4f}")
    
    # Test cache
    print("\n=== Cache Test ===")
    belief_cache = BeliefCache(P, delta_max=100)
    cost_cache = CostCache(P, delta_max=100)
    
    for h, delta in [(0, 1), (0, 10), (2, 5)]:
        b = belief_cache.get_belief(h, delta)
        c = cost_cache.get_cost(h, delta)
        print(f"Cached: belief({h},{delta})={b.round(3)}, cost={c:.3f}")
