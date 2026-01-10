"""
Environment for Road Digital Twin AoII-ARD RMAB Simulation (v3 - Unified)
=========================================================================

KEY CHANGES (v3):
1. Per-arm heterogeneous p_s: self.p_s_per_arm[i] for each arm
2. Automatically generates heterogeneous p_s from config
3. Maintains backward compatibility with original interface

Environment Step Order (per DR-07):
1. For all arms: s_i(t+1) ~ Categorical(P_bar[s_i(t),:])
2. If a_i=0: Δ ← min(Δ+1, Δ_max)
   If a_i=1:
     - Success w.p. p_s_i: h ← s_i(t+1), Δ ← 1
     - Failure: Δ ← min(Δ+1, Δ_max)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from numpy.random import Generator, PCG64, SeedSequence

from config import (
    SimulationConfig, ArmClassConfig, RandomStateManager,
    compute_oracle_aoii, compute_control_cost, g_semantic,
    generate_heterogeneous_p_s
)


@dataclass
class ArmState:
    """State of a single arm (road segment)."""
    s_true: int      # True physical state (latent)
    h: int           # DT estimated state
    delta: int       # Age since last sync
    class_idx: int   # Arm class index
    p_s: float       # Per-arm success probability (NEW in v3)
    
    def get_observable(self) -> Tuple[int, int]:
        return (self.h, self.delta)


@dataclass
class StepResult:
    """Result of a single environment step."""
    observations: np.ndarray
    oracle_aoii: np.ndarray
    control_costs: np.ndarray
    successes: np.ndarray
    rewards: float
    info: Dict[str, Any]


class RMABEnvironment:
    """
    Restless Multi-Armed Bandit Environment (v3 - Heterogeneous p_s).
    
    KEY CHANGE: Each arm has its own p_s value (self.p_s_per_arm).
    """
    
    def __init__(self, config: SimulationConfig, seed: int = 42):
        self.config = config
        self.N = config.experiment.N
        self.M = config.experiment.M
        self.J = config.experiment.J
        self.delta_max = config.experiment.delta_max
        self.arm_classes = config.arm_classes
        
        # Initialize random state
        self.rng_manager = RandomStateManager(seed)
        self.rng_physics = self.rng_manager.get_generator("physics")
        self.rng_channel = self.rng_manager.get_generator("channel")
        self.rng_structure = self.rng_manager.get_generator("structure")
        
        # Assign arm classes
        self.arm_class_indices = self._assign_arm_classes()
        
        # NEW v3: Generate per-arm heterogeneous p_s
        self.p_s_per_arm = self._generate_p_s(seed)
        
        # Initialize arm states
        self.arms: List[ArmState] = []
        self.epoch = 0
        
        self._initialize_arms()
    
    def _generate_p_s(self, seed: int) -> np.ndarray:
        """Generate per-arm heterogeneous p_s values."""
        het_config = self.config.experiment.heterogeneous
        
        if het_config.enabled:
            return generate_heterogeneous_p_s(self.N, het_config, seed)
        else:
            # Fallback: use class-level p_s
            p_s_values = np.zeros(self.N)
            for i in range(self.N):
                class_idx = self.arm_class_indices[i]
                p_s_values[i] = self.arm_classes[class_idx].p_s
            return p_s_values
    
    def _assign_arm_classes(self) -> np.ndarray:
        """Assign each arm to a class (30% slow, 70% fast)."""
        slow_ratio = 0.30
        n_slow = int(self.N * slow_ratio)
        
        indices = [0] * n_slow + [1] * (self.N - n_slow)
        indices = np.array(indices)
        self.rng_structure.shuffle(indices)
        
        return indices
    
    def _initialize_arms(self):
        """Initialize arm states with per-arm p_s."""
        self.arms = []
        for i in range(self.N):
            class_idx = self.arm_class_indices[i]
            self.arms.append(ArmState(
                s_true=0,
                h=0,
                delta=1,
                class_idx=class_idx,
                p_s=self.p_s_per_arm[i]  # Per-arm p_s
            ))
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment."""
        if seed is not None:
            self.rng_manager.reset(seed)
            self.rng_physics = self.rng_manager.get_generator("physics")
            self.rng_channel = self.rng_manager.get_generator("channel")
            self.rng_structure = self.rng_manager.get_generator("structure")
            
            self.arm_class_indices = self._assign_arm_classes()
            self.p_s_per_arm = self._generate_p_s(seed)
        
        self.epoch = 0
        self._initialize_arms()
        
        return self._get_observations()
    
    def _get_observations(self) -> np.ndarray:
        """Get observable states (h, Δ) for all arms."""
        obs = np.zeros((self.N, 2), dtype=np.int32)
        for i, arm in enumerate(self.arms):
            obs[i] = [arm.h, arm.delta]
        return obs
    
    def step(self, actions: np.ndarray) -> StepResult:
        """
        Execute one step.
        
        Uses per-arm p_s for success probability.
        """
        assert len(actions) == self.N
        assert actions.sum() <= self.M, f"Budget violation: {actions.sum()} > {self.M}"
        
        oracle_aoii = np.zeros(self.N)
        control_costs = np.zeros(self.N)
        successes = np.zeros(self.N, dtype=bool)
        
        for i, arm in enumerate(self.arms):
            class_idx = arm.class_idx
            P_bar = self.arm_classes[class_idx].P_bar
            
            # 1. State evolution (restless)
            probs = P_bar[arm.s_true, :]
            probs = probs / probs.sum()  # Normalize
            arm.s_true = self.rng_physics.choice(self.J, p=probs)
            
            # 2. Action processing with PER-ARM p_s
            if actions[i] == 1:
                if self.rng_channel.random() < arm.p_s:  # Use arm.p_s
                    arm.h = arm.s_true
                    arm.delta = 1
                    successes[i] = True
                else:
                    arm.delta = min(arm.delta + 1, self.delta_max)
            else:
                arm.delta = min(arm.delta + 1, self.delta_max)
            
            # 3. Compute costs
            oracle_aoii[i] = compute_oracle_aoii(arm.s_true, arm.h, arm.delta)
            control_costs[i] = compute_control_cost(arm.h, arm.delta, P_bar)
        
        self.epoch += 1
        
        return StepResult(
            observations=self._get_observations(),
            oracle_aoii=oracle_aoii,
            control_costs=control_costs,
            successes=successes,
            rewards=-oracle_aoii.sum(),
            info={
                'epoch': self.epoch,
                'mean_oracle_aoii': oracle_aoii.mean(),
                'mean_control_cost': control_costs.mean(),
                'success_rate': successes.mean() if actions.sum() > 0 else 0,
                'scheduled': actions.sum(),
            }
        )
    
    def get_arm_p_s(self, arm_idx: int) -> float:
        """Get p_s for a specific arm."""
        return self.p_s_per_arm[arm_idx]
    
    def get_all_p_s(self) -> np.ndarray:
        """Get p_s values for all arms."""
        return self.p_s_per_arm.copy()


class EpisodeLogger:
    """Logs episode data for analysis."""
    
    def __init__(self, N: int, T: int, detailed: bool = False):
        self.N = N
        self.T = T
        self.detailed = detailed
        
        self.epoch_oracle_aoii = np.zeros(T)
        self.epoch_control_cost = np.zeros(T)
        self.epoch_n_scheduled = np.zeros(T, dtype=np.int32)
        self.epoch_n_success = np.zeros(T, dtype=np.int32)
        
        if detailed:
            self.arm_oracle_aoii = np.zeros((T, N))
            self.arm_h = np.zeros((T, N), dtype=np.int32)
            self.arm_delta = np.zeros((T, N), dtype=np.int32)
            self.arm_actions = np.zeros((T, N), dtype=np.int32)
            self.arm_success = np.zeros((T, N), dtype=np.int32)
    
    def log(self, epoch: int, result: StepResult, actions: np.ndarray):
        """Log data for one epoch."""
        self.epoch_oracle_aoii[epoch] = result.oracle_aoii.mean()
        self.epoch_control_cost[epoch] = result.control_costs.mean()
        self.epoch_n_scheduled[epoch] = actions.sum()
        self.epoch_n_success[epoch] = result.successes.sum()
        
        if self.detailed:
            self.arm_oracle_aoii[epoch] = result.oracle_aoii
            self.arm_h[epoch] = result.observations[:, 0]
            self.arm_delta[epoch] = result.observations[:, 1]
            self.arm_actions[epoch] = actions
            self.arm_success[epoch] = result.successes
    
    def get_summary(self, burn_in_ratio: float = 0.5) -> Dict[str, float]:
        """Get summary statistics over evaluation period."""
        burn_in = int(len(self.epoch_oracle_aoii) * burn_in_ratio)
        eval_aoii = self.epoch_oracle_aoii[burn_in:]
        eval_cost = self.epoch_control_cost[burn_in:]
        
        return {
            'mean_oracle_aoii': eval_aoii.mean(),
            'std_oracle_aoii': eval_aoii.std(),
            'mean_control_cost': eval_cost.mean(),
            'std_control_cost': eval_cost.std(),
            'total_scheduled': self.epoch_n_scheduled.sum(),
            'total_success': self.epoch_n_success.sum(),
            'avg_success_rate': self.epoch_n_success.sum() / max(self.epoch_n_scheduled.sum(), 1)
        }
    
    def get_trajectory(self) -> np.ndarray:
        """Get full AoII trajectory."""
        return self.epoch_oracle_aoii.copy()


class EpisodeLogger:
    """Logs episode data for analysis."""
    
    def __init__(self, N: int, T: int, detailed: bool = False):
        self.N = N
        self.T = T
        self.detailed = detailed
        
        self.epoch_oracle_aoii = np.zeros(T)
        self.epoch_control_cost = np.zeros(T)
        self.epoch_n_scheduled = np.zeros(T, dtype=np.int32)
        self.epoch_n_success = np.zeros(T, dtype=np.int32)
        
        if detailed:
            self.arm_oracle_aoii = np.zeros((T, N))
            self.arm_h = np.zeros((T, N), dtype=np.int32)
            self.arm_delta = np.zeros((T, N), dtype=np.int32)
            self.arm_actions = np.zeros((T, N), dtype=np.int32)
            self.arm_success = np.zeros((T, N), dtype=np.int32)
    
    def log(self, epoch: int, result: StepResult, actions: np.ndarray):
        """Log data for one epoch."""
        self.epoch_oracle_aoii[epoch] = result.oracle_aoii.mean()
        self.epoch_control_cost[epoch] = result.control_costs.mean()
        self.epoch_n_scheduled[epoch] = actions.sum()
        self.epoch_n_success[epoch] = result.successes.sum()
        
        if self.detailed:
            self.arm_oracle_aoii[epoch] = result.oracle_aoii
            self.arm_h[epoch] = result.observations[:, 0]
            self.arm_delta[epoch] = result.observations[:, 1]
            self.arm_actions[epoch] = actions
            self.arm_success[epoch] = result.successes
    
    def get_summary(self, burn_in_ratio: float = 0.5) -> Dict[str, float]:
        """Get summary statistics over evaluation period."""
        burn_in = int(len(self.epoch_oracle_aoii) * burn_in_ratio)
        eval_aoii = self.epoch_oracle_aoii[burn_in:]
        eval_cost = self.epoch_control_cost[burn_in:]
        
        return {
            'mean_oracle_aoii': eval_aoii.mean(),
            'std_oracle_aoii': eval_aoii.std(),
            'mean_control_cost': eval_cost.mean(),
            'std_control_cost': eval_cost.std(),
            'total_scheduled': self.epoch_n_scheduled.sum(),
            'total_success': self.epoch_n_success.sum(),
            'avg_success_rate': self.epoch_n_success.sum() / max(self.epoch_n_scheduled.sum(), 1)
        }
    
    def get_trajectory(self) -> np.ndarray:
        """Get full AoII trajectory."""
        return self.epoch_oracle_aoii.copy()


if __name__ == "__main__":
    print("=== Environment v3 Test (Heterogeneous p_s) ===")
    
    config = SimulationConfig()
    env = RMABEnvironment(config, seed=42)
    
    print(f"N={env.N}, M={env.M}")
    print(f"p_s range: [{env.p_s_per_arm.min():.3f}, {env.p_s_per_arm.max():.3f}]")
    print(f"p_s std: {env.p_s_per_arm.std():.3f}")
    
    # Test step
    obs = env.reset()
    actions = np.zeros(env.N, dtype=np.int32)
    actions[:env.M] = 1
    
    result = env.step(actions)
    print(f"\nStep result: mean_aoii={result.info['mean_oracle_aoii']:.3f}")
