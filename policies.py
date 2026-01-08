"""
Scheduling Policies for AoII-Aware RMAB (v2)
=============================================

UPDATED based on advisor feedback:
- Myopic policy now uses belief_after_evolution for success distribution

Policies:
1. WhittlePolicy: Select arms with highest Whittle indices
2. MaxAgePolicy: Select arms with highest age Δ (AoI-optimal baseline)
3. MyopicPolicy: Select arms with highest single-step expected gain (FIXED)
4. RandomPolicy: Uniformly random selection
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
        """
        Select which arms to schedule.
        
        Args:
            observations: Shape (N, 2) with (h, Δ) for each arm
            env: Environment instance (for accessing arm configs)
        
        Returns:
            actions: Binary array of shape (N,), 1=schedule, 0=idle
        """
        pass
    
    def reset(self):
        """Reset policy state (if any)."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class WhittlePolicy(BasePolicy):
    """
    Whittle Index Policy.
    
    Selects the M arms with highest Whittle indices W(h, Δ).
    Requires pre-computed index tables for each arm class.
    """
    
    def __init__(self, index_tables: Dict[str, WhittleIndexTable], name: str = "Whittle"):
        super().__init__(name)
        self.index_tables = index_tables
    
    def get_index(self, h: int, delta: int, class_name: str) -> float:
        """Get Whittle index for a state."""
        return self.index_tables[class_name].get_index(h, delta)
    
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        """Select arms with highest Whittle indices."""
        N = observations.shape[0]
        M = env.M
        
        # Compute index for each arm
        indices = np.zeros(N)
        for i in range(N):
            h, delta = observations[i]
            class_name = env.arm_classes[env.arm_class_indices[i]].name
            indices[i] = self.get_index(h, delta, class_name)
        
        # Select top M (break ties randomly for fairness)
        actions = np.zeros(N, dtype=np.int32)
        if M > 0:
            # Add small random noise for tie-breaking
            noise = np.random.uniform(0, 1e-10, N)
            top_indices = np.argsort(indices + noise)[-M:]
            actions[top_indices] = 1
        
        return actions


class MaxAgePolicy(BasePolicy):
    """
    Maximum Age Policy (AoI-optimal baseline).
    
    Selects the M arms with highest age Δ.
    """
    
    def __init__(self, name: str = "MaxAge"):
        super().__init__(name)
    
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        """Select arms with highest age."""
        N = observations.shape[0]
        M = env.M
        
        ages = observations[:, 1].astype(float)
        
        # Add small noise for tie-breaking
        noise = np.random.uniform(0, 1e-10, N)
        
        actions = np.zeros(N, dtype=np.int32)
        if M > 0:
            top_indices = np.argsort(ages + noise)[-M:]
            actions[top_indices] = 1
        
        return actions


class MyopicPolicy(BasePolicy):
    """
    Myopic (Greedy) Policy (FIXED).
    
    Selects arms based on single-step expected improvement in control cost.
    
    FIXED: Uses belief_after_evolution for success distribution, consistent
    with DR-07 semantics (success syncs to s(t+1)).
    
    Gain(i) = C(h_i, Δ_i) - E[C(h', Δ') | a_i = 1]
    
    For active action:
    - With prob p_s: h' = j where j ~ π^+ = π @ P (evolved belief), Δ' = 1
    - With prob 1-p_s: h' = h, Δ' = Δ + 1
    """
    
    def __init__(self, name: str = "Myopic"):
        super().__init__(name)
    
    def compute_gain(self, h: int, delta: int, p_s: float, 
                     P_bar: np.ndarray, delta_max: int) -> float:
        """
        Compute single-step expected gain from scheduling this arm.
        
        FIXED: Uses evolved belief π^+ = π @ P for success distribution.
        
        Gain = C(h, Δ) - [p_s * E_{j~π^+}[C(j, 1)] + (1-p_s) * C(h, min(Δ+1, Δ_max))]
        """
        J = P_bar.shape[0]
        
        # Current cost
        current_cost = compute_control_cost(h, delta, P_bar)
        
        # FIXED: Use evolved belief for success distribution
        # This is consistent with env semantics: first evolve s, then sync
        belief_evolved = compute_belief_after_evolution(h, delta, P_bar)
        
        # Expected cost after success: new state (j, 1) where j ~ π^+
        expected_cost_success = 0.0
        for j in range(J):
            if belief_evolved[j] > 1e-10:
                cost_j = compute_control_cost(j, 1, P_bar)
                expected_cost_success += belief_evolved[j] * cost_j
        
        # Expected cost after failure: state (h, min(Δ+1, Δ_max))
        next_delta = min(delta + 1, delta_max)
        cost_fail = compute_control_cost(h, next_delta, P_bar)
        
        # Expected cost after action
        expected_cost_active = p_s * expected_cost_success + (1 - p_s) * cost_fail
        
        # Gain = current cost - expected cost after action
        gain = current_cost - expected_cost_active
        
        return gain
    
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        """Select arms with highest myopic gain."""
        N = observations.shape[0]
        M = env.M
        
        # Compute gain for each arm
        gains = np.zeros(N)
        for i in range(N):
            h, delta = observations[i]
            arm_config = env.arm_classes[env.arm_class_indices[i]]
            gains[i] = self.compute_gain(
                h, delta, arm_config.p_s, arm_config.P_bar, env.delta_max
            )
        
        # Select top M
        actions = np.zeros(N, dtype=np.int32)
        if M > 0:
            noise = np.random.uniform(0, 1e-10, N)
            top_indices = np.argsort(gains + noise)[-M:]
            actions[top_indices] = 1
        
        return actions


class RandomPolicy(BasePolicy):
    """
    Random Policy (baseline).
    
    Uniformly selects M arms at random.
    """
    
    def __init__(self, seed: int = 42, name: str = "Random"):
        super().__init__(name)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self._initial_seed = seed
    
    def reset(self):
        """Reset random generator."""
        self.rng = np.random.Generator(np.random.PCG64(self._initial_seed))
    
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        """Select M arms uniformly at random."""
        N = observations.shape[0]
        M = env.M
        
        actions = np.zeros(N, dtype=np.int32)
        if M > 0:
            selected = self.rng.choice(N, size=min(M, N), replace=False)
            actions[selected] = 1
        
        return actions


class MaxControlCostPolicy(BasePolicy):
    """
    Maximum Control Cost Policy.
    
    Selects arms with highest current control cost C(h, Δ).
    """
    
    def __init__(self, name: str = "MaxCost"):
        super().__init__(name)
    
    def select_arms(self, observations: np.ndarray, env) -> np.ndarray:
        """Select arms with highest control cost."""
        N = observations.shape[0]
        M = env.M
        
        costs = np.zeros(N)
        for i in range(N):
            h, delta = observations[i]
            arm_config = env.arm_classes[env.arm_class_indices[i]]
            costs[i] = compute_control_cost(h, delta, arm_config.P_bar)
        
        actions = np.zeros(N, dtype=np.int32)
        if M > 0:
            noise = np.random.uniform(0, 1e-10, N)
            top_indices = np.argsort(costs + noise)[-M:]
            actions[top_indices] = 1
        
        return actions


def create_standard_policies(index_tables: Dict[str, WhittleIndexTable],
                             seed: int = 42) -> Dict[str, BasePolicy]:
    """
    Create standard policies for experiments.
    
    Args:
        index_tables: Pre-computed Whittle index tables
        seed: Random seed for Random policy
    
    Returns:
        Dictionary of policy name -> policy instance
    """
    return {
        'Whittle': WhittlePolicy(index_tables),
        'MaxAge': MaxAgePolicy(),
        'Myopic': MyopicPolicy(),
        'Random': RandomPolicy(seed=seed),
    }


if __name__ == "__main__":
    from config import SimulationConfig
    from whittle_v2 import WhittleSolver
    from environment import RMABEnvironment
    
    print("=== Policy Test (v2 - Fixed) ===\n")
    
    # Setup
    config = SimulationConfig()
    config.experiment.N = 20
    config.experiment.M = 4
    test_delta_max = 30
    
    # Create environment
    env = RMABEnvironment(config, seed=42)
    
    # Compute Whittle indices
    print("Computing Whittle indices...")
    solver = WhittleSolver(config.whittle)
    index_tables = solver.compute_all_tables(config.arm_classes, test_delta_max)
    
    # Create policies
    policies = create_standard_policies(index_tables, seed=42)
    
    # Test Myopic gain computation (verify it uses evolved belief)
    print("\n=== Myopic Gain Analysis (FIXED) ===")
    myopic = MyopicPolicy()
    
    arm_config = config.arm_classes[0]
    print(f"Class: {arm_config.name}, p_s={arm_config.p_s}")
    print("\nGain(h, Δ) for various states:")
    print("      Δ=1    Δ=5    Δ=10   Δ=20")
    
    for h in range(min(5, config.experiment.J)):
        gains = []
        for delta in [1, 5, 10, 20]:
            gain = myopic.compute_gain(
                h, delta, arm_config.p_s, arm_config.P_bar, 
                config.experiment.delta_max
            )
            gains.append(gain)
        print(f"h={h}: " + "  ".join(f"{g:6.3f}" for g in gains))
    
    # Run short episodes with each policy
    print("\n=== Short Episode Test (200 epochs) ===")
    T = 200
    
    for name, policy in policies.items():
        env.reset(seed=42)
        policy.reset()
        
        total_aoii = 0.0
        for t in range(T):
            obs = env._get_observations()
            actions = policy.select_arms(obs, env)
            result = env.step(actions)
            total_aoii += result.info['mean_oracle_aoii']
        
        mean_aoii = total_aoii / T
        print(f"  {name:12s}: Mean Oracle AoII = {mean_aoii:.3f}")
