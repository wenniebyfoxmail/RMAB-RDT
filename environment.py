"""
Environment for Road Digital Twin AoII-ARD RMAB Simulation (v2)
================================================================

UPDATED based on advisor feedback:
- reset(new_seed) now reassigns arm-class mapping for proper Monte Carlo

Environment Step Order (per DR-07):
1. For all arms: s_i(t+1) ~ Categorical(P_bar[s_i(t),:])  # Restless evolution FIRST
2. If a_i=0: Δ ← min(Δ+1, Δ_max)
   If a_i=1:
     - Success w.p. p_s: h ← s_i(t+1), Δ ← 1  # Sync to EVOLVED state
     - Failure: Δ ← min(Δ+1, Δ_max)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from numpy.random import Generator, PCG64, SeedSequence

from config import (
    SimulationConfig, ArmClassConfig, RandomStateManager,
    compute_oracle_aoii, compute_control_cost, g_semantic
)


@dataclass
class ArmState:
    """State of a single arm (road segment)."""
    s_true: int      # True physical state (latent, for Oracle)
    h: int           # DT estimated state (last sync value)
    delta: int       # Age since last successful update
    class_idx: int   # Index of arm class
    
    def get_observable(self) -> Tuple[int, int]:
        """Return observable state (h, Δ)."""
        return (self.h, self.delta)


@dataclass
class StepResult:
    """Result of a single environment step."""
    observations: np.ndarray      # Shape (N, 2): (h, Δ) for each arm
    oracle_aoii: np.ndarray       # Shape (N,): Oracle AoII for each arm
    control_costs: np.ndarray     # Shape (N,): Control costs for each arm
    successes: np.ndarray         # Shape (N,): Success flags for scheduled arms
    rewards: float                # Negative sum of Oracle AoII (for RL interface)
    info: Dict[str, Any]          # Additional info


class RMABEnvironment:
    """
    Restless Multi-Armed Bandit Environment for Road Digital Twin.
    
    Maintains N arms, each with:
    - True state s (latent, for Oracle evaluation)
    - Observable state (h, Δ) for scheduling
    """
    
    def __init__(self, config: SimulationConfig, seed: int = 42):
        """
        Initialize the RMAB environment.
        
        Args:
            config: Simulation configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.N = config.experiment.N
        self.M = config.experiment.M
        self.J = config.experiment.J
        self.delta_max = config.experiment.delta_max
        self.arm_classes = config.arm_classes
        
        # Initialize random state manager
        self.rng_manager = RandomStateManager(seed)
        self.rng_physics = self.rng_manager.get_generator("physics")
        self.rng_channel = self.rng_manager.get_generator("channel")
        self.rng_structure = self.rng_manager.get_generator("structure")
        
        # Assign arms to classes based on distribution
        self.arm_class_indices = self._assign_arm_classes()
        
        # Initialize arm states
        self.arms: List[ArmState] = []
        self.epoch = 0
        
        self._initialize_arms()
    
    def _assign_arm_classes(self) -> np.ndarray:
        """Assign each arm to a class based on configuration."""
        distribution = self.config.experiment.class_distribution
        n_classes = len(self.arm_classes)
        
        # Always recalculate distribution based on current N
        # Split evenly among classes, with remainder going to last class
        per_class = self.N // n_classes
        distribution = [per_class] * n_classes
        distribution[-1] = self.N - sum(distribution[:-1])
        
        indices = []
        for class_idx, count in enumerate(distribution):
            indices.extend([class_idx] * count)
        
        indices = np.array(indices)
        self.rng_structure.shuffle(indices)
        
        return indices
    
    def _initialize_arms(self):
        """Initialize arm states."""
        self.arms = []
        for i in range(self.N):
            class_idx = self.arm_class_indices[i]
            self.arms.append(ArmState(
                s_true=0,
                h=0,
                delta=1,  # Start with Δ=1 (just synchronized)
                class_idx=class_idx
            ))
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment to initial state.
        
        UPDATED: If new seed is provided, reassign arm-class mapping
        for proper Monte Carlo across seeds.
        
        Args:
            seed: Optional new random seed
            
        Returns:
            observations: Initial observable states (N, 2)
        """
        if seed is not None:
            self.rng_manager.reset(seed)
            self.rng_physics = self.rng_manager.get_generator("physics")
            self.rng_channel = self.rng_manager.get_generator("channel")
            self.rng_structure = self.rng_manager.get_generator("structure")
            
            # UPDATED: Reassign arm-class mapping for new seed
            self.arm_class_indices = self._assign_arm_classes()
        
        self.epoch = 0
        self._initialize_arms()
        
        return self._get_observations()
    
    def _get_observations(self) -> np.ndarray:
        """Get observable states for all arms."""
        obs = np.zeros((self.N, 2), dtype=np.int32)
        for i, arm in enumerate(self.arms):
            obs[i, 0] = arm.h
            obs[i, 1] = arm.delta
        return obs
    
    def _get_arm_config(self, arm_idx: int) -> ArmClassConfig:
        """Get configuration for a specific arm."""
        class_idx = self.arm_class_indices[arm_idx]
        return self.arm_classes[class_idx]
    
    def step(self, actions: np.ndarray) -> StepResult:
        """
        Execute one epoch step.
        
        Step Order (per DR-07):
        1. Evolve all true states s(t) → s(t+1) FIRST
        2. Process actions: success syncs to s(t+1)
        3. Compute metrics
        
        Args:
            actions: Binary array of shape (N,), 1=schedule, 0=idle
                    Sum must be <= M
        
        Returns:
            StepResult with observations, costs, and metrics
        """
        actions = np.asarray(actions, dtype=np.int32)
        assert actions.shape == (self.N,), f"Expected shape ({self.N},), got {actions.shape}"
        assert actions.sum() <= self.M, f"Too many actions: {actions.sum()} > {self.M}"
        
        successes = np.zeros(self.N, dtype=np.int32)
        
        # Step 1: Evolve true states FIRST (restless dynamics)
        for i, arm in enumerate(self.arms):
            arm_config = self._get_arm_config(i)
            P = arm_config.P_bar
            
            # Sample next state from transition distribution
            next_state = self.rng_physics.choice(
                self.J, 
                p=P[arm.s_true, :]
            )
            arm.s_true = next_state
        
        # Step 2: Process actions
        for i, arm in enumerate(self.arms):
            if actions[i] == 0:
                # Idle: age increases
                arm.delta = min(arm.delta + 1, self.delta_max)
            else:
                # Update attempt
                arm_config = self._get_arm_config(i)
                
                # Success with probability p_s
                if self.rng_channel.random() < arm_config.p_s:
                    # Success: sync to EVOLVED true state s(t+1)
                    arm.h = arm.s_true
                    arm.delta = 1
                    successes[i] = 1
                else:
                    # Failure: age increases
                    arm.delta = min(arm.delta + 1, self.delta_max)
        
        # Step 3: Compute metrics
        oracle_aoii = np.zeros(self.N)
        control_costs = np.zeros(self.N)
        
        for i, arm in enumerate(self.arms):
            arm_config = self._get_arm_config(i)
            
            oracle_aoii[i] = compute_oracle_aoii(
                arm.s_true, arm.h, arm.delta, g_func=g_semantic
            )
            
            control_costs[i] = compute_control_cost(
                arm.h, arm.delta, arm_config.P_bar
            )
        
        self.epoch += 1
        
        observations = self._get_observations()
        total_aoii = oracle_aoii.sum()
        
        return StepResult(
            observations=observations,
            oracle_aoii=oracle_aoii,
            control_costs=control_costs,
            successes=successes,
            rewards=-total_aoii,
            info={
                'epoch': self.epoch,
                'total_oracle_aoii': total_aoii,
                'mean_oracle_aoii': oracle_aoii.mean(),
                'total_control_cost': control_costs.sum(),
                'mean_control_cost': control_costs.mean(),
                'n_scheduled': actions.sum(),
                'n_success': successes.sum(),
                'success_rate': successes.sum() / max(actions.sum(), 1)
            }
        )
    
    def get_arm_info(self, arm_idx: int) -> Dict[str, Any]:
        """Get detailed information about a specific arm."""
        arm = self.arms[arm_idx]
        arm_config = self._get_arm_config(arm_idx)
        
        return {
            's_true': arm.s_true,
            'h': arm.h,
            'delta': arm.delta,
            'class_name': arm_config.name,
            'class_idx': arm.class_idx,
            'p_s': arm_config.p_s,
            'oracle_aoii': compute_oracle_aoii(arm.s_true, arm.h, arm.delta),
            'control_cost': compute_control_cost(arm.h, arm.delta, arm_config.P_bar)
        }


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
    from config import SimulationConfig
    
    print("=== Environment Test (v2) ===")
    config = SimulationConfig()
    env = RMABEnvironment(config, seed=42)
    
    print(f"N={env.N}, M={env.M}, J={env.J}")
    print(f"p_s={config.arm_classes[0].p_s} (from DR-06B R={config.experiment.R})")
    
    # Test reset with new seed reassigns arm-class
    print("\n=== Testing reset with new seed ===")
    old_mapping = env.arm_class_indices.copy()
    env.reset(seed=123)
    new_mapping = env.arm_class_indices
    print(f"Mapping changed: {not np.array_equal(old_mapping, new_mapping)}")
    
    # Test episode
    print("\n=== Short Episode Test ===")
    env.reset(seed=42)
    
    for t in range(10):
        actions = np.zeros(env.N, dtype=np.int32)
        selected = np.random.choice(env.N, env.M, replace=False)
        actions[selected] = 1
        
        result = env.step(actions)
        print(f"Epoch {t+1}: Mean AoII={result.info['mean_oracle_aoii']:.3f}, "
              f"Successes={result.info['n_success']}/{result.info['n_scheduled']}")
