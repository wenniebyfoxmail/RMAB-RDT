"""
Parallel Computing Utilities
=============================

Automatic CPU core detection and parallel experiment execution.
Provides significant speedup for Monte Carlo simulations across seeds.

Usage:
    from parallel_utils import ParallelRunner, get_optimal_workers
    
    runner = ParallelRunner(n_workers='auto')
    results = runner.run_parallel(run_single_seed, seeds)
"""

import os
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from typing import Callable, List, Any, Optional, Dict, Union
from dataclasses import dataclass
import time
import numpy as np


def get_cpu_count() -> int:
    """Get the number of available CPU cores."""
    try:
        # Try to get the number of available CPUs (respects cgroup limits in containers)
        if hasattr(os, 'sched_getaffinity'):
            return len(os.sched_getaffinity(0))
        else:
            return cpu_count() or 1
    except:
        return cpu_count() or 1


def get_optimal_workers(task_count: int = None) -> int:
    """
    Get optimal number of worker processes.
    
    Args:
        task_count: Number of tasks to run (optional)
        
    Returns:
        Optimal number of workers
    """
    n_cores = get_cpu_count()
    
    # Leave 1 core free for system tasks if we have many cores
    if n_cores > 4:
        n_workers = n_cores - 1
    elif n_cores > 1:
        n_workers = n_cores
    else:
        n_workers = 1
    
    # Don't use more workers than tasks
    if task_count is not None:
        n_workers = min(n_workers, task_count)
    
    return max(1, n_workers)


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    n_workers: int = None       # Number of workers (None = auto)
    verbose: bool = True        # Print progress
    chunk_size: int = 1         # Chunk size for pool.map
    
    def __post_init__(self):
        if self.n_workers is None:
            self.n_workers = get_optimal_workers()


class ParallelRunner:
    """
    Parallel task runner with automatic core detection.
    
    Example:
        runner = ParallelRunner(n_workers='auto')
        
        def run_seed(seed):
            # ... run experiment
            return result
        
        results = runner.run_parallel(run_seed, [42, 43, 44, 45])
    """
    
    def __init__(self, n_workers: Union[str, int] = 'auto', verbose: bool = True):
        """
        Initialize parallel runner.
        
        Args:
            n_workers: Number of workers. 'auto' for automatic detection,
                      or specific integer.
            verbose: Whether to print progress
        """
        self.verbose = verbose
        
        if n_workers == 'auto':
            self.n_workers = get_optimal_workers()
        else:
            self.n_workers = int(n_workers)
        
        if self.verbose:
            print(f"🔧 Parallel Runner initialized:")
            print(f"   CPU cores detected: {get_cpu_count()}")
            print(f"   Workers to use: {self.n_workers}")
    
    def run_parallel(self, func: Callable, args_list: List[Any],
                     desc: str = "Running tasks") -> List[Any]:
        """
        Run function in parallel across argument list.
        
        Args:
            func: Function to run (must be picklable)
            args_list: List of arguments to pass to func
            desc: Description for progress display
            
        Returns:
            List of results in same order as args_list
        """
        n_tasks = len(args_list)
        
        if self.verbose:
            print(f"\n⚡ {desc}")
            print(f"   Tasks: {n_tasks}, Workers: {self.n_workers}")
        
        start_time = time.time()
        
        if self.n_workers == 1 or n_tasks == 1:
            # Sequential execution
            results = []
            for i, arg in enumerate(args_list):
                if self.verbose:
                    print(f"   [{i+1}/{n_tasks}] Processing...")
                results.append(func(arg))
        else:
            # Parallel execution
            with Pool(processes=self.n_workers) as pool:
                results = pool.map(func, args_list)
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"   ✅ Completed in {elapsed:.1f}s")
            if self.n_workers > 1 and n_tasks > 1:
                print(f"   📈 Estimated speedup: ~{min(self.n_workers, n_tasks):.1f}x")
        
        return results


def print_system_info():
    """Print system information for debugging."""
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"CPU cores (total): {cpu_count()}")
    print(f"CPU cores (available): {get_cpu_count()}")
    print(f"Recommended workers: {get_optimal_workers()}")
    try:
        print(f"Multiprocessing method: {mp.get_start_method()}")
    except:
        pass
    print("=" * 50)


def parallel_map(func: Callable, args_list: List[Any], 
                 n_workers: Union[str, int] = 'auto',
                 desc: str = "Running tasks",
                 verbose: bool = True) -> List[Any]:
    """
    Convenience function for parallel mapping.
    
    Args:
        func: Function to apply (must be picklable, defined at module level)
        args_list: List of arguments
        n_workers: Number of workers ('auto' or int)
        desc: Description for progress
        verbose: Whether to print progress
        
    Returns:
        List of results
    """
    if n_workers == 'auto':
        n_workers = get_optimal_workers(len(args_list))
    else:
        n_workers = int(n_workers)
    
    n_tasks = len(args_list)
    
    if verbose:
        print(f"\n⚡ {desc}")
        print(f"   Tasks: {n_tasks}, Workers: {n_workers}")
    
    start_time = time.time()
    
    if n_workers == 1 or n_tasks == 1:
        # Sequential execution
        results = [func(arg) for arg in args_list]
    else:
        # Parallel execution
        with Pool(processes=n_workers) as pool:
            results = pool.map(func, args_list)
    
    elapsed = time.time() - start_time
    
    if verbose:
        speedup = min(n_workers, n_tasks)
        print(f"   ✅ Completed in {elapsed:.1f}s (estimated {speedup:.0f}x speedup)")
    
    return results


if __name__ == "__main__":
    # Test parallel execution
    print_system_info()
    
    def dummy_task(x):
        """Dummy task for testing."""
        import time
        time.sleep(0.1)  # Simulate work
        return x * 2
    
    runner = ParallelRunner(n_workers='auto')
    results = runner.run_parallel(dummy_task, list(range(8)), "Testing parallel execution")
    print(f"Results: {results}")
