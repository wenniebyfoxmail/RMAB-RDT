"""
Configuration for Road Digital Twin AoII-ARD RMAB Simulation (v3 - Unified)
============================================================================

KEY CHANGES (v3):
1. Per-arm heterogeneous p_s: Each arm has its own success probability
2. This breaks Liu-Weber-Zhao degeneracy, enabling 15%+ Whittle advantage
3. All original scripts work automatically with new settings

Theory Reference:
- Liu-Weber-Zhao theorem: Homogeneous arms → Whittle ≈ Myopic
- Breaking condition: Heterogeneous p_{s,i} + tight budget M/N ≤ 10%

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

R_TABLE = {
    4:  {'p_s': 0.50, 'D': 0.0039},
    8:  {'p_s': 0.50, 'D': 1.5e-5},
    16: {'p_s': 0.50, 'D': 2.3e-10},
    32: {'p_s': 0.50, 'D': 1e-6},
    64: {'p_s': 0.50, 'D': 1e-6},
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
    P_bar: np.ndarray
    p_s: float         # Base p_s (used as reference, actual p_s is per-arm)
    D: float = 0.0
    R: int = 32
    c_ratio: float = 1.0
    
    def __post_init__(self):
        J = self.P_bar.shape[0]
        assert self.P_bar.shape == (J, J), "P_bar must be square"
        row_sums = self.P_bar.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"P_bar rows must sum to 1"


# =============================================================================
# Whittle Solver Configuration
# =============================================================================

@dataclass
class WhittleConfig:
    """Configuration for Whittle Index computation."""
    gamma: float = 0.999
    epsilon: float = 1e-3
    vi_threshold: float = 1e-5
    vi_max_iter: int = 1000
    use_hot_start: bool = True
    ref_state: Tuple[int, int] = (0, 1)
    lambda_init: float = 10.0
    lambda_max: float = 10000.0
    bracket_expand_factor: float = 2.0
    # 加速参数
    n_lambda_coarse: int = 200  # Reduce to 50-100 for faster computation


# =============================================================================
# HETEROGENEOUS P_S CONFIGURATION (KEY ADDITION)
# =============================================================================

@dataclass
class HeterogeneousConfig:
    """
    Configuration for per-arm heterogeneous p_s.

    This is the KEY mechanism for Whittle advantage over Myopic.
    """
    enabled: bool = True  # DEFAULT: enabled for v3

    # Heterogeneity level: "homogeneous", "low", "medium", "high"
    level: str = "high"

    # p_s ranges by level
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "homogeneous": (0.50, 0.50),
        "low": (0.35, 0.55),
        "medium": (0.25, 0.70),
        "high": (0.20, 0.85),
    })

    def get_range(self) -> Tuple[float, float]:
        return self.ranges.get(self.level, (0.20, 0.85))


def generate_heterogeneous_p_s(N: int, config: HeterogeneousConfig,
                                seed: int = 42) -> np.ndarray:
    """
    Generate per-arm heterogeneous p_s values.

    Args:
        N: Number of arms
        config: Heterogeneous configuration
        seed: Random seed

    Returns:
        Array of p_s values, one per arm
    """
    rng = np.random.default_rng(seed)
    p_min, p_max = config.get_range()

    if config.level == "homogeneous":
        return np.full(N, 0.50)
    else:
        return rng.uniform(p_min, p_max, N)


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    N: int = 50
    M: int = 5  # M/N = 10% by default
    J: int = 5
    delta_max: int = 100
    T: int = 2000
    n_seeds: int = 10
    burn_in_ratio: float = 0.5
    class_distribution: List[int] = field(default_factory=lambda: [25, 25])
    master_seed: int = 42
    R: int = 8

    # NEW: Heterogeneous p_s configuration
    heterogeneous: HeterogeneousConfig = field(default_factory=HeterogeneousConfig)


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
        if not self.arm_classes:
            self.arm_classes = get_nhgp_arm_classes(
                J=self.experiment.J,
                R=self.experiment.R
            )


def get_nhgp_arm_classes(J: int = 5, R: int = 8,
                         recovery_prob: float = 0.02,
                         heterogeneous: bool = False) -> List[ArmClassConfig]:
    """
    Get arm classes with NHGP-derived transition matrices.

    Args:
        J: Number of states
        R: Channel parameter (affects base p_s)
        recovery_prob: Recovery probability (default 0.02, REQUIRED for non-zero AoII)
        heterogeneous: If True, use heterogeneous p_s values [0.3, 0.7]
                      If False, use homogeneous p_s (default)

    NOTE: heterogeneous=True is recommended for demonstrating Whittle vs Myopic gap.
    """
    p_s_base, D = get_channel_params(R)

    # NHGP classes from nhgp_builder
    nhgp_classes = get_default_nhgp_classes(J=J, recovery_prob=recovery_prob)

    # Define p_s values based on heterogeneity setting
    if heterogeneous:
        # Heterogeneous: different p_s per class
        # slow class: harder to sync (low p_s)
        # fast class: easier to sync (high p_s)
        p_s_values = [0.30, 0.70]  # Conservative heterogeneity
        print(f"  📊 Using heterogeneous p_s: {p_s_values}")
    else:
        p_s_values = [p_s_base, p_s_base]  # Homogeneous

    arm_classes = []
    for (name, P_bar, c_ratio), p_s in zip(nhgp_classes, p_s_values):
        arm_classes.append(ArmClassConfig(
            name=name,
            P_bar=P_bar,
            p_s=p_s,
            D=D,
            R=R,
            c_ratio=c_ratio
        ))

    return arm_classes


def get_ontario_arm_classes(data_dir: str = "data/ontario",
                            heterogeneous_p_s: bool = True) -> List[ArmClassConfig]:
    """
    从 Ontario 真实路面数据创建 arm classes.

    使用真实的年度转移矩阵（2022→2023）和基于路段响应性的 p_s 分布。

    Args:
        data_dir: Ontario 数据目录路径
        heterogeneous_p_s: 是否使用异质性 p_s（3种类型）

    Returns:
        基于真实数据的 ArmClassConfig 列表

    数据来源:
        https://data.ontario.ca/dataset/pavement-condition-for-provincial-highways
    """
    from ontario_data_loader import OntarioDataLoader
    from pathlib import Path

    # 构建正确的文件路径
    data_path = Path(data_dir)
    path_2022 = data_path / "ontario_2022.csv"
    path_2023 = data_path / "ontario_2023.csv"

    # 检查文件是否存在
    if not path_2022.exists():
        raise FileNotFoundError(f"Ontario 2022 data not found: {path_2022}")
    if not path_2023.exists():
        raise FileNotFoundError(f"Ontario 2023 data not found: {path_2023}")

    loader = OntarioDataLoader(str(path_2022), str(path_2023))
    result = loader.load_and_process()

    # 获取真实转移矩阵（年度）
    P_bar_annual = loader.transition_matrices['overall']

    # 转换为月度转移矩阵（假设一年12次转移）
    # P_monthly ≈ P_annual^(1/12)，这里简化处理
    # 实际使用年度矩阵，因为主实验 T 通常对应年度时间步
    P_bar = P_bar_annual

    arm_classes = []

    if heterogeneous_p_s:
        # 基于 Ontario 数据的路段响应性分类
        # responsive: 维护后快速改善
        # moderate: 中等响应
        # unresponsive: 维护效果差

        p_s_configs = [
            ('ontario_responsive', 0.70, 1.0),    # 30% 路段，高 p_s
            ('ontario_moderate', 0.50, 1.0),      # 50% 路段，中 p_s
            ('ontario_unresponsive', 0.30, 1.0),  # 20% 路段，低 p_s
        ]

        for name, p_s, c_ratio in p_s_configs:
            arm_classes.append(ArmClassConfig(
                name=name,
                P_bar=P_bar.copy(),
                p_s=p_s,
                D=0,
                R=8,
                c_ratio=c_ratio
            ))
    else:
        # 单一类型，使用平均 p_s
        arm_classes.append(ArmClassConfig(
            name='ontario_avg',
            P_bar=P_bar,
            p_s=0.50,
            D=0,
            R=8,
            c_ratio=1.0
        ))

    print(f"✅ Loaded Ontario real data from {data_dir}")
    print(f"   Transition matrix shape: {P_bar.shape}")
    print(f"   Arm classes: {[ac.name for ac in arm_classes]}")

    return arm_classes


# =============================================================================
# Random State Management
# =============================================================================

class RandomStateManager:
    """Manage multiple independent random streams."""

    def __init__(self, master_seed: int = 42):
        self.master_seed = master_seed
        self.ss = SeedSequence(master_seed)
        self.generators: Dict[str, Generator] = {}
        self._spawn_count = 0

    def get_generator(self, name: str) -> Generator:
        if name not in self.generators:
            child_seed = self.ss.spawn(1)[0]
            self.generators[name] = Generator(PCG64(child_seed))
            self._spawn_count += 1
        return self.generators[name]

    def reset(self, new_seed: int):
        self.master_seed = new_seed
        self.ss = SeedSequence(new_seed)
        self.generators = {}
        self._spawn_count = 0


# =============================================================================
# Belief and Cost Functions
# =============================================================================

class BeliefCache:
    """Cache for belief computations."""

    def __init__(self, P_bar: np.ndarray, delta_max: int = 100):
        self.P_bar = P_bar
        self.J = P_bar.shape[0]
        self.delta_max = delta_max
        self._P_powers = self._precompute_powers()

    def _precompute_powers(self) -> List[np.ndarray]:
        powers = [np.eye(self.J)]
        P_power = np.eye(self.J)
        for _ in range(self.delta_max):
            P_power = P_power @ self.P_bar
            powers.append(P_power.copy())
        return powers

    def get_P_power(self, n: int) -> np.ndarray:
        n = min(n, self.delta_max)
        return self._P_powers[n]

    def get_belief(self, h: int, delta: int) -> np.ndarray:
        power = min(delta - 1, self.delta_max - 1)
        e_h = np.zeros(self.J)
        e_h[h] = 1.0
        return e_h @ self.get_P_power(power)

    def get_belief_evolved(self, h: int, delta: int) -> np.ndarray:
        return self.get_belief(h, delta) @ self.P_bar


def g_semantic(d: int) -> float:
    """Semantic distance function g(d) = d."""
    return float(d)


# =============================================================================
# Cost Function Options (NEW - Per Report Recommendations)
# =============================================================================

@dataclass
class CostFunctionConfig:
    """
    Configuration for AoII cost function f(Δ).

    Report insight: Linear costs don't differentiate planning horizons.
    Convex costs (Δ², exp) magnify Whittle advantage by 5-15%.
    """
    cost_type: str = "linear"  # "linear", "quadratic", "exponential", "threshold"
    exp_alpha: float = 0.05    # For exponential: e^(αΔ)
    threshold: int = 50        # For threshold: extra penalty when Δ > threshold
    threshold_penalty: float = 10.0

    def get_f_age(self):
        """Return the age cost function f(Δ)."""
        if self.cost_type == "linear":
            return lambda x: float(x)
        elif self.cost_type == "quadratic":
            return lambda x: float(x ** 2)
        elif self.cost_type == "exponential":
            alpha = self.exp_alpha
            return lambda x: float(np.exp(alpha * x))
        elif self.cost_type == "threshold":
            threshold = self.threshold
            penalty = self.threshold_penalty
            return lambda x: float(x + penalty * (x > threshold))
        else:
            raise ValueError(f"Unknown cost type: {self.cost_type}")


def get_cost_function(cost_type: str = "linear", **kwargs) -> callable:
    """
    Factory function for cost functions.

    Args:
        cost_type: One of "linear", "quadratic", "exponential", "threshold"
        **kwargs: Additional parameters (exp_alpha, threshold, etc.)

    Returns:
        f_age function: Δ → cost

    Usage:
        f_linear = get_cost_function("linear")
        f_quadratic = get_cost_function("quadratic")
        f_exp = get_cost_function("exponential", exp_alpha=0.1)
    """
    config = CostFunctionConfig(cost_type=cost_type, **kwargs)
    return config.get_f_age()


def compute_belief(h: int, delta: int, P_bar: np.ndarray) -> np.ndarray:
    """Compute belief distribution π(h, Δ) = e_h^T P^{Δ-1}."""
    J = P_bar.shape[0]
    e_h = np.zeros(J)
    e_h[h] = 1.0

    power = max(0, delta - 1)
    P_power = np.linalg.matrix_power(P_bar, power)

    return e_h @ P_power


def compute_belief_after_evolution(h: int, delta: int, P_bar: np.ndarray) -> np.ndarray:
    """Compute evolved belief π^+ = π @ P."""
    return compute_belief(h, delta, P_bar) @ P_bar


def compute_control_cost(h: int, delta: int, P_bar: np.ndarray,
                         f_age=None, g_sem=None) -> float:
    """
    Compute control cost C(h, Δ).

    C(h, Δ) = E_{s~π}[f(Δ) * g(|s - h|)]

    Default: f(Δ) = Δ (linear), g(d) = d
    """
    if f_age is None:
        f_age = lambda x: x
    if g_sem is None:
        g_sem = g_semantic

    belief = compute_belief(h, delta, P_bar)
    J = P_bar.shape[0]

    cost = 0.0
    for s in range(J):
        semantic_dist = abs(g_sem(s) - g_sem(h))
        cost += belief[s] * f_age(delta) * semantic_dist

    return cost


def compute_oracle_aoii(s_true: int, h: int, delta: int,
                        f_age=None, g_sem=None) -> float:
    """
    Compute Oracle AoII (ground truth).

    Oracle_AoII = f(Δ) * g(|s - h|) if s ≠ h, else 0
    """
    if s_true == h:
        return 0.0

    if f_age is None:
        f_age = lambda x: x
    if g_sem is None:
        g_sem = g_semantic

    return f_age(delta) * abs(g_sem(s_true) - g_sem(h))


# =============================================================================
# Tail Metrics (for publication)
# =============================================================================

def compute_tail_metrics(values: np.ndarray) -> dict:
    """Compute tail risk metrics for AoII distribution."""
    if len(values) == 0:
        return {'mean': 0, 'std': 0, 'P50': 0, 'P90': 0, 'P95': 0, 'P99': 0, 'max': 0}

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'P50': float(np.percentile(values, 50)),
        'P90': float(np.percentile(values, 90)),
        'P95': float(np.percentile(values, 95)),
        'P99': float(np.percentile(values, 99)),
        'max': float(np.max(values)),
    }


class CostCache:
    """Cache for precomputed control costs."""

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


# Default configuration instance
DEFAULT_CONFIG = SimulationConfig()


if __name__ == "__main__":
    print("=== Config v3 Test (Heterogeneous p_s) ===")

    config = SimulationConfig()
    print(f"N={config.experiment.N}, M={config.experiment.M}")
    print(f"Heterogeneous: {config.experiment.heterogeneous.enabled}")
    print(f"Level: {config.experiment.heterogeneous.level}")
    print(f"p_s range: {config.experiment.heterogeneous.get_range()}")

    # Test p_s generation
    p_s_values = generate_heterogeneous_p_s(
        50, config.experiment.heterogeneous, seed=42
    )
    print(f"\nGenerated p_s: [{p_s_values.min():.3f}, {p_s_values.max():.3f}]")
    print(f"Mean: {p_s_values.mean():.3f}, Std: {p_s_values.std():.3f}")