"""
Scheduling Policies for AoII-Aware RMAB (v3 - Unified Heterogeneous)
=====================================================================

KEY CHANGES (v3):
1. WhittlePolicy automatically handles per-arm heterogeneous p_s
2. Computes separate index tables for different p_s levels
3. Myopic policy uses per-arm p_s for gain calculation

Policies:
1. WhittlePolicy: Per-arm p_s aware Whittle indices
2. MyopicPolicy: Greedy single-step (with per-arm p_s)
3. MaxAgePolicy: Only looks at age Δ
4. WorstStatePolicy: Only looks at state h (ablation)
5. RandomPolicy: Uniform random selection
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from config import (
    ArmClassConfig, compute_control_cost, 
    compute_belief, compute_belief_after_evolution,
    g_semantic
)
from whittle_solver import WhittleIndexTable


class BasePolicy(ABC):
    """Abstract base class for scheduling policies."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        pass
    
    def reset(self):
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class WhittlePolicy(BasePolicy):
    """
    Whittle Index Policy (v3 - Heterogeneous p_s Support).
    
    KEY FEATURE: Automatically handles per-arm p_s by:
    1. Discretizing p_s into n_levels
    2. Computing index table for each (class, p_s_level) pair
    3. Looking up appropriate table for each arm
    """
    
    def __init__(self, index_tables: Dict[str, WhittleIndexTable], 
                 name: str = "Whittle",
                 arm_classes: List[ArmClassConfig] = None,
                 p_s_per_arm: np.ndarray = None,
                 delta_max: int = 100,
                 whittle_config = None,
                 n_p_s_levels: int = 10):
        """
        Initialize Whittle policy.
        
        Two modes:
        1. Simple mode: Just pass index_tables (original behavior)
        2. Heterogeneous mode: Pass arm_classes + p_s_per_arm to auto-build tables
        """
        super().__init__(name)
        
        if p_s_per_arm is not None and arm_classes is not None and whittle_config is not None:
            # Heterogeneous mode: build tables for different p_s levels
            self._build_heterogeneous_tables(
                arm_classes, p_s_per_arm, delta_max, whittle_config, n_p_s_levels
            )
            self.heterogeneous_mode = True
        else:
            # Simple mode: use provided tables
            self.index_tables = index_tables
            self.heterogeneous_mode = False
            self.p_s_per_arm = None
            self.arm_p_s_level_idx = None
    
    def _build_heterogeneous_tables(self, arm_classes, p_s_per_arm, 
                                     delta_max, whittle_config, n_p_s_levels):
        """Build index tables for heterogeneous p_s."""
        from whittle_solver import WhittleSolver
        
        self.p_s_per_arm = p_s_per_arm
        
        # Discretize p_s
        p_s_min, p_s_max = p_s_per_arm.min(), p_s_per_arm.max()
        if p_s_max - p_s_min < 0.01:
            self.p_s_levels = np.array([p_s_per_arm.mean()])
        else:
            self.p_s_levels = np.linspace(p_s_min, p_s_max, n_p_s_levels)
        
        # Map each arm to nearest p_s level
        self.arm_p_s_level_idx = np.array([
            np.argmin(np.abs(self.p_s_levels - p_s))
            for p_s in p_s_per_arm
        ])
        
        # Build tables
        self.index_tables = {}
        self.het_tables = {}  # (class_name, level_idx) -> table
        
        solver = WhittleSolver(whittle_config)
        
        for arm_class in arm_classes:
            # Standard table (for backward compatibility)
            self.index_tables[arm_class.name] = solver.compute_index_table(
                arm_class, delta_max
            )
            
            # Per-level tables
            for level_idx, p_s in enumerate(self.p_s_levels):
                modified_class = ArmClassConfig(
                    name=f"{arm_class.name}_ps{level_idx}",
                    P_bar=arm_class.P_bar.copy(),
                    p_s=p_s,
                    D=arm_class.D,
                    R=arm_class.R,
                    c_ratio=arm_class.c_ratio
                )
                table = solver.compute_index_table(modified_class, delta_max)
                self.het_tables[(arm_class.name, level_idx)] = table
    
    def get_index(self, h: int, delta: int, class_name: str, 
                  arm_idx: int = None) -> float:
        """Get Whittle index for a state."""
        if self.heterogeneous_mode and arm_idx is not None:
            # Use per-arm p_s level table
            level_idx = self.arm_p_s_level_idx[arm_idx]
            table = self.het_tables.get((class_name, level_idx))
            if table is not None:
                return table.get_index(h, delta)
        
        # Fallback to standard table
        return self.index_tables[class_name].get_index(h, delta)
    
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        """Select arms with highest Whittle indices."""
        N = observations.shape[0]
        M = env.M
        
        # Auto-enable heterogeneous mode if environment has per-arm p_s
        if not self.heterogeneous_mode and hasattr(env, 'p_s_per_arm'):
            p_s_per_arm = env.p_s_per_arm
            # Check if actually heterogeneous (std > 0.05)
            if p_s_per_arm.std() > 0.05:
                self._auto_build_heterogeneous_tables(env, p_s_per_arm)
        
        # Update p_s mapping if environment changed
        if self.heterogeneous_mode and hasattr(env, 'p_s_per_arm'):
            if self.p_s_per_arm is None or not np.array_equal(self.p_s_per_arm, env.p_s_per_arm):
                self.p_s_per_arm = env.p_s_per_arm
                self.arm_p_s_level_idx = np.array([
                    np.argmin(np.abs(self.p_s_levels - p_s))
                    for p_s in self.p_s_per_arm
                ])
        
        # Compute indices
        indices = np.zeros(N)
        for i in range(N):
            h, delta = observations[i]
            class_name = env.arm_classes[env.arm_class_indices[i]].name
            indices[i] = self.get_index(h, delta, class_name, arm_idx=i)
        
        # Select top M
        actions = np.zeros(N, dtype=np.int32)
        if M > 0:
            noise = np.random.uniform(0, 1e-10, N)
            top_indices = np.argsort(indices + noise)[-M:]
            actions[top_indices] = 1
        
        return actions
    
    def _auto_build_heterogeneous_tables(self, env, p_s_per_arm: np.ndarray):
        """Auto-build heterogeneous tables when detected."""
        from whittle_solver import WhittleSolver, WhittleConfig
        
        self.p_s_per_arm = p_s_per_arm
        self.heterogeneous_mode = True
        
        # Discretize p_s into 5 levels for efficiency
        p_s_min, p_s_max = p_s_per_arm.min(), p_s_per_arm.max()
        n_levels = 5  # Reduced from 10 for speed
        self.p_s_levels = np.linspace(p_s_min, p_s_max, n_levels)
        
        # Map arms to levels
        self.arm_p_s_level_idx = np.array([
            np.argmin(np.abs(self.p_s_levels - p_s))
            for p_s in p_s_per_arm
        ])
        
        # Build tables
        self.het_tables = {}
        whittle_config = WhittleConfig()
        solver = WhittleSolver(whittle_config)
        delta_max = env.delta_max
        
        for arm_class in env.arm_classes:
            for level_idx, p_s in enumerate(self.p_s_levels):
                modified_class = ArmClassConfig(
                    name=f"{arm_class.name}_ps{level_idx}",
                    P_bar=arm_class.P_bar.copy(),
                    p_s=p_s,
                    D=arm_class.D,
                    R=arm_class.R,
                    c_ratio=arm_class.c_ratio
                )
                table = solver.compute_index_table(modified_class, delta_max)
                self.het_tables[(arm_class.name, level_idx)] = table


class MyopicPolicy(BasePolicy):
    """
    Myopic (Greedy) Policy (v3 - Per-arm p_s Support).
    
    Computes single-step gain using per-arm p_s.
    """
    
    def __init__(self, name: str = "Myopic"):
        super().__init__(name)
    
    def compute_gain(self, h: int, delta: int, p_s: float, 
                     P_bar: np.ndarray, delta_max: int) -> float:
        """Compute single-step expected gain with per-arm p_s."""
        J = P_bar.shape[0]
        
        current_cost = compute_control_cost(h, delta, P_bar)
        
        # Success case
        belief_evolved = compute_belief_after_evolution(h, delta, P_bar)
        success_cost = sum(
            belief_evolved[j] * compute_control_cost(j, 1, P_bar)
            for j in range(J)
        )
        
        # Failure case
        fail_cost = compute_control_cost(h, min(delta + 1, delta_max), P_bar)
        
        expected_cost = p_s * success_cost + (1 - p_s) * fail_cost
        
        return current_cost - expected_cost
    
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        """Select arms with highest myopic gain."""
        N = observations.shape[0]
        M = env.M
        
        gains = np.zeros(N)
        for i in range(N):
            h, delta = observations[i]
            class_idx = env.arm_class_indices[i]
            P_bar = env.arm_classes[class_idx].P_bar
            
            # Use per-arm p_s if available
            if hasattr(env, 'p_s_per_arm'):
                p_s = env.p_s_per_arm[i]
            else:
                p_s = env.arm_classes[class_idx].p_s
            
            gains[i] = self.compute_gain(h, delta, p_s, P_bar, env.delta_max)
        
        actions = np.zeros(N, dtype=np.int32)
        if M > 0:
            noise = np.random.uniform(0, 1e-10, N)
            top_indices = np.argsort(gains + noise)[-M:]
            actions[top_indices] = 1
        
        return actions


class MaxAgePolicy(BasePolicy):
    """MaxAge Policy: Select arms with highest age Δ."""
    
    def __init__(self, name: str = "MaxAge"):
        super().__init__(name)
    
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        N = observations.shape[0]
        M = env.M
        
        ages = observations[:, 1].astype(float)
        noise = np.random.uniform(0, 1e-10, N)
        
        actions = np.zeros(N, dtype=np.int32)
        if M > 0:
            top_indices = np.argsort(ages + noise)[-M:]
            actions[top_indices] = 1
        
        return actions


class WorstStatePolicy(BasePolicy):
    """
    WorstState Policy: Select arms with highest (worst) state h.
    
    Ablation study: proves that ignoring age Δ is suboptimal.
    """
    
    def __init__(self, name: str = "WorstState"):
        super().__init__(name)
    
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        N = observations.shape[0]
        M = env.M
        
        h_values = observations[:, 0].astype(float)
        noise = np.random.uniform(0, 1e-10, N)
        
        actions = np.zeros(N, dtype=np.int32)
        if M > 0:
            top_indices = np.argsort(h_values + noise)[-M:]
            actions[top_indices] = 1
        
        return actions


class RandomPolicy(BasePolicy):
    """Random Policy: Uniformly random selection."""
    
    def __init__(self, seed: int = 42, name: str = "Random"):
        super().__init__(name)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self._initial_seed = seed
    
    def reset(self):
        """Reset random generator."""
        self.rng = np.random.Generator(np.random.PCG64(self._initial_seed))
    
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        N = observations.shape[0]
        M = env.M
        
        actions = np.zeros(N, dtype=np.int32)
        if M > 0:
            selected = self.rng.choice(N, M, replace=False)
            actions[selected] = 1
        
        return actions


class MaxControlCostPolicy(BasePolicy):
    """MaxControlCost Policy: Select arms with highest expected control cost."""
    
    def __init__(self, name: str = "MaxControlCost"):
        super().__init__(name)
    
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        N = observations.shape[0]
        M = env.M
        
        costs = np.zeros(N)
        for i in range(N):
            h, delta = observations[i]
            class_idx = env.arm_class_indices[i]
            P_bar = env.arm_classes[class_idx].P_bar
            costs[i] = compute_control_cost(h, delta, P_bar)
        
        actions = np.zeros(N, dtype=np.int32)
        if M > 0:
            noise = np.random.uniform(0, 1e-10, N)
            top_indices = np.argsort(costs + noise)[-M:]
            actions[top_indices] = 1
        
        return actions


def get_all_policies(index_tables: Dict[str, WhittleIndexTable],
                     arm_classes: List[ArmClassConfig] = None,
                     p_s_per_arm: np.ndarray = None,
                     delta_max: int = 100,
                     whittle_config = None) -> Dict[str, BasePolicy]:
    """
    Get all available policies.
    
    If p_s_per_arm is provided, creates heterogeneous Whittle policy.
    """
    policies = {
        'Myopic': MyopicPolicy(),
        'MaxAge': MaxAgePolicy(),
        'WorstState': WorstStatePolicy(),
        'Random': RandomPolicy(),
        'MaxControlCost': MaxControlCostPolicy(),
    }
    
    # Create appropriate Whittle policy
    if p_s_per_arm is not None and arm_classes is not None and whittle_config is not None:
        policies['Whittle'] = WhittlePolicy(
            index_tables=index_tables,
            arm_classes=arm_classes,
            p_s_per_arm=p_s_per_arm,
            delta_max=delta_max,
            whittle_config=whittle_config
        )
    else:
        policies['Whittle'] = WhittlePolicy(index_tables)
    
    return policies


if __name__ == "__main__":
    print("=== Policies v3 Test ===")
    
    from config import SimulationConfig, generate_heterogeneous_p_s
    from whittle_solver import WhittleSolver
    
    config = SimulationConfig()
    
    # Generate heterogeneous p_s
    p_s_per_arm = generate_heterogeneous_p_s(
        config.experiment.N, 
        config.experiment.heterogeneous,
        seed=42
    )
    
    print(f"p_s range: [{p_s_per_arm.min():.3f}, {p_s_per_arm.max():.3f}]")
    
    # Build Whittle tables
    solver = WhittleSolver(config.whittle)
    index_tables = solver.compute_all_tables(
        config.arm_classes, config.experiment.delta_max
    )
    
    # Create heterogeneous Whittle policy
    whittle = WhittlePolicy(
        index_tables=index_tables,
        arm_classes=config.arm_classes,
        p_s_per_arm=p_s_per_arm,
        delta_max=config.experiment.delta_max,
        whittle_config=config.whittle
    )
    
    print(f"Heterogeneous mode: {whittle.heterogeneous_mode}")
    print(f"p_s levels: {len(whittle.p_s_levels)}")
