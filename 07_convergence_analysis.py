"""
收敛性分析脚本 (Convergence Analysis)
=====================================

功能:
1. 运行实验并记录完整轨迹 (每个 timestep 的 AoII)
2. 绘制收敛曲线 (移动平均 + 置信带)
3. 分析收敛速度和稳态性能
4. 计算 burn-in 前后的统计差异

使用方法:
    python 07_convergence_analysis.py --output results/convergence
    python 07_convergence_analysis.py --output results/convergence --quick
    python 07_convergence_analysis.py --output results/convergence --heterogeneous

输出:
    - convergence_curves.png/pdf (收敛曲线图)
    - convergence_data.csv (原始轨迹数据)
    - convergence_stats.csv (收敛统计)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import (
    SimulationConfig, get_nhgp_arm_classes, get_ontario_arm_classes
)
from environment import RMABEnvironment
from whittle_solver import WhittleSolver
from policies import (
    WhittlePolicy, MyopicPolicy, MaxAgePolicy, 
    RandomPolicy, WorstStatePolicy
)


# =============================================================================
# IEEE 风格设置
# =============================================================================

def setup_ieee_style():
    """Configure matplotlib for IEEE publication format."""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.figsize': (3.5, 2.5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# =============================================================================
# 策略样式
# =============================================================================

POLICY_STYLES = {
    'Whittle': {'color': '#2ecc71', 'marker': 'o', 'linestyle': '-'},
    'Myopic': {'color': '#e74c3c', 'marker': 's', 'linestyle': '--'},
    'MaxAge': {'color': '#3498db', 'marker': '^', 'linestyle': '-.'},
    'Random': {'color': '#95a5a6', 'marker': 'd', 'linestyle': ':'},
    'WorstState': {'color': '#9b59b6', 'marker': 'v', 'linestyle': '-'},
}


# =============================================================================
# 实验运行函数
# =============================================================================

def run_single_trajectory(config: SimulationConfig, 
                          policy_name: str,
                          index_tables: Dict,
                          T: int,
                          seed: int,
                          delta_max: int = 100) -> np.ndarray:
    """
    运行单个轨迹，返回每个 timestep 的 AoII
    
    Returns:
        trajectory: shape (T,) 每个时间步的平均 AoII
    """
    env = RMABEnvironment(config, seed=seed)
    env.delta_max = delta_max
    
    # 创建策略
    if policy_name == 'Whittle':
        policy = WhittlePolicy(index_tables)
    elif policy_name == 'Myopic':
        policy = MyopicPolicy()
    elif policy_name == 'MaxAge':
        policy = MaxAgePolicy()
    elif policy_name == 'Random':
        policy = RandomPolicy(seed=seed)
    elif policy_name == 'WorstState':
        policy = WorstStatePolicy()
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    policy.reset()
    
    # 运行
    obs = env._get_observations()
    trajectory = np.zeros(T)
    
    for t in range(T):
        actions = policy.select_arms(obs, env)
        result = env.step(actions)
        trajectory[t] = result.info['mean_oracle_aoii']
        obs = result.observations
    
    return trajectory


def run_convergence_experiment(N: int, M: int, T: int, n_seeds: int,
                               arm_classes, delta_max: int = 100,
                               verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    运行收敛性实验
    
    Returns:
        trajectories: {policy_name: array of shape (n_seeds, T)}
    """
    config = SimulationConfig()
    config.experiment.N = N
    config.experiment.M = M
    config.experiment.delta_max = delta_max
    config.arm_classes = arm_classes
    
    # 计算 Whittle 表
    if verbose:
        print("  Computing Whittle indices...")
    solver = WhittleSolver(config.whittle)
    index_tables = solver.compute_all_tables(arm_classes, delta_max, verbose=False)
    
    policies = ['Whittle', 'Myopic', 'MaxAge', 'Random']
    seeds = list(range(42, 42 + n_seeds))
    
    trajectories = {}
    
    for policy_name in policies:
        if verbose:
            print(f"  Running {policy_name}...", end=" ", flush=True)
        
        start_time = time.time()
        policy_trajectories = []
        
        for seed in seeds:
            traj = run_single_trajectory(
                config, policy_name, index_tables, T, seed, delta_max
            )
            policy_trajectories.append(traj)
        
        trajectories[policy_name] = np.array(policy_trajectories)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"{elapsed:.1f}s")
    
    return trajectories


# =============================================================================
# 分析函数
# =============================================================================

@dataclass
class ConvergenceStats:
    """收敛统计结果"""
    steady_state_mean: float
    steady_state_std: float
    convergence_time: int
    burn_in_mean: float
    post_burn_in_mean: float
    improvement_after_burn_in: float


def analyze_convergence(trajectory: np.ndarray, 
                        burn_in_ratio: float = 0.5,
                        window: int = 50) -> ConvergenceStats:
    """
    分析单个策略的收敛特性
    
    Args:
        trajectory: shape (n_seeds, T) 或 (T,)
        burn_in_ratio: burn-in 比例
        window: 移动平均窗口
    """
    if trajectory.ndim > 1:
        mean_traj = trajectory.mean(axis=0)
    else:
        mean_traj = trajectory
    
    T = len(mean_traj)
    burn_in = int(T * burn_in_ratio)
    
    # 稳态统计 (最后 20%)
    steady_start = int(T * 0.8)
    steady_state_mean = mean_traj[steady_start:].mean()
    steady_state_std = mean_traj[steady_start:].std()
    
    # 收敛时间 (第一次进入稳态 ±10% 的时间)
    threshold = steady_state_mean * 0.1
    smoothed = np.convolve(mean_traj, np.ones(window)/window, mode='valid')
    
    convergence_time = T
    for t, val in enumerate(smoothed):
        if abs(val - steady_state_mean) < threshold:
            convergence_time = t + window // 2
            break
    
    # Burn-in 前后对比
    burn_in_mean = mean_traj[:burn_in].mean()
    post_burn_in_mean = mean_traj[burn_in:].mean()
    improvement = (burn_in_mean - post_burn_in_mean) / burn_in_mean * 100 if burn_in_mean > 0 else 0
    
    return ConvergenceStats(
        steady_state_mean=steady_state_mean,
        steady_state_std=steady_state_std,
        convergence_time=convergence_time,
        burn_in_mean=burn_in_mean,
        post_burn_in_mean=post_burn_in_mean,
        improvement_after_burn_in=improvement
    )


# =============================================================================
# 可视化函数
# =============================================================================

def plot_convergence_curves(trajectories: Dict[str, np.ndarray],
                            output_path: str,
                            burn_in_ratio: float = 0.5,
                            window: int = 50,
                            title: str = None) -> plt.Figure:
    """
    绘制收敛曲线
    
    Args:
        trajectories: {policy_name: array of shape (n_seeds, T)}
        output_path: 保存路径
        burn_in_ratio: burn-in 比例
        window: 移动平均窗口
    """
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    T = None
    
    for name, traj in trajectories.items():
        if name not in POLICY_STYLES:
            continue
        
        style = POLICY_STYLES[name]
        
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        
        T = traj.shape[1]
        
        # 计算均值和标准差
        mean_traj = traj.mean(axis=0)
        std_traj = traj.std(axis=0)
        
        # 移动平均平滑
        smoothed_mean = np.convolve(mean_traj, np.ones(window)/window, mode='valid')
        smoothed_std = np.convolve(std_traj, np.ones(window)/window, mode='valid')
        
        x = np.arange(len(smoothed_mean)) + window // 2
        
        # 绘制均值曲线
        ax.plot(x, smoothed_mean, label=name, 
               color=style['color'], linestyle=style['linestyle'],
               linewidth=1.5)
        
        # 绘制置信带 (±1 std)
        ax.fill_between(x, 
                       smoothed_mean - smoothed_std,
                       smoothed_mean + smoothed_std,
                       alpha=0.15, color=style['color'])
    
    # 添加 burn-in 标记线
    if T is not None:
        burn_in = int(T * burn_in_ratio)
        ax.axvline(x=burn_in, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(burn_in + T*0.02, ax.get_ylim()[1] * 0.95, 
               'Burn-in', fontsize=7, color='gray', va='top')
    
    ax.set_xlabel('Time Step $t$')
    ax.set_ylabel('Average AoII')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=8)
    
    plt.tight_layout()
    
    # 保存
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    return fig


def plot_convergence_comparison(trajectories_list: List[Dict[str, np.ndarray]],
                                 labels: List[str],
                                 output_path: str,
                                 policy: str = 'Whittle') -> plt.Figure:
    """
    对比不同配置下同一策略的收敛曲线
    """
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(trajectories_list)))
    
    for i, (traj_dict, label) in enumerate(zip(trajectories_list, labels)):
        if policy not in traj_dict:
            continue
        
        traj = traj_dict[policy]
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        
        mean_traj = traj.mean(axis=0)
        window = 50
        smoothed = np.convolve(mean_traj, np.ones(window)/window, mode='valid')
        x = np.arange(len(smoothed)) + window // 2
        
        ax.plot(x, smoothed, label=label, color=colors[i], linewidth=1.5)
    
    ax.set_xlabel('Time Step $t$')
    ax.set_ylabel(f'Average AoII ({policy})')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    return fig


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convergence Analysis')
    parser.add_argument('--output', type=str, default='results/convergence',
                       help='Output directory')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced parameters')
    parser.add_argument('--heterogeneous', action='store_true',
                       help='Use heterogeneous p_s')
    parser.add_argument('--use-ontario', action='store_true',
                       help='Use Ontario real data')
    parser.add_argument('--ontario-dir', type=str, default='data/ontario',
                       help='Ontario data directory')
    parser.add_argument('--delta-max', type=int, default=100,
                       help='Maximum delta value')
    args = parser.parse_args()
    
    # 创建输出目录
    Path(f"{args.output}/data").mkdir(parents=True, exist_ok=True)
    Path(f"{args.output}/figures").mkdir(parents=True, exist_ok=True)
    
    # 参数设置
    if args.quick:
        T = 500
        n_seeds = 3
        N_values = [50]
        M_values = [2, 5]
    else:
        T = 1000
        n_seeds = 5
        N_values = [50, 100]
        M_values = [2, 5, 10]
    
    print("=" * 60)
    print("📈 CONVERGENCE ANALYSIS")
    print("=" * 60)
    print(f"T = {T}, n_seeds = {n_seeds}")
    print(f"N_values = {N_values}")
    print(f"M_values = {M_values}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # 加载 arm classes
    if args.use_ontario:
        try:
            arm_classes = get_ontario_arm_classes(args.ontario_dir)
            print("✅ Using Ontario data")
        except Exception as e:
            print(f"⚠️ Ontario load failed: {e}, using NHGP")
            arm_classes = get_nhgp_arm_classes(J=5, R=8, heterogeneous=args.heterogeneous)
    else:
        arm_classes = get_nhgp_arm_classes(J=5, R=8, heterogeneous=args.heterogeneous)
        print(f"✅ Using NHGP data (heterogeneous={args.heterogeneous})")
    
    all_stats = []
    all_trajectories = {}
    
    # 运行实验
    for N in N_values:
        for M in M_values:
            print(f"\n{'='*40}")
            print(f"Running N={N}, M={M}")
            print(f"{'='*40}")
            
            trajectories = run_convergence_experiment(
                N=N, M=M, T=T, n_seeds=n_seeds,
                arm_classes=arm_classes,
                delta_max=args.delta_max
            )
            
            key = f"N{N}_M{M}"
            all_trajectories[key] = trajectories
            
            # 绘制收敛曲线
            plot_convergence_curves(
                trajectories,
                f"{args.output}/figures/convergence_{key}.png",
                burn_in_ratio=0.5,
                title=f"N={N}, M={M}"
            )
            
            # 分析收敛统计
            for policy_name, traj in trajectories.items():
                stats = analyze_convergence(traj)
                all_stats.append({
                    'N': N,
                    'M': M,
                    'policy': policy_name,
                    'steady_state_mean': stats.steady_state_mean,
                    'steady_state_std': stats.steady_state_std,
                    'convergence_time': stats.convergence_time,
                    'burn_in_mean': stats.burn_in_mean,
                    'post_burn_in_mean': stats.post_burn_in_mean,
                    'improvement_pct': stats.improvement_after_burn_in
                })
            
            # 保存原始轨迹数据
            traj_data = []
            for policy_name, traj in trajectories.items():
                for seed_idx in range(traj.shape[0]):
                    for t in range(traj.shape[1]):
                        traj_data.append({
                            'N': N, 'M': M, 'policy': policy_name,
                            'seed': seed_idx, 't': t, 'aoii': traj[seed_idx, t]
                        })
            
            traj_df = pd.DataFrame(traj_data)
            traj_df.to_csv(f"{args.output}/data/trajectories_{key}.csv", index=False)
            print(f"✅ Saved trajectories: {args.output}/data/trajectories_{key}.csv")
    
    # 保存汇总统计
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(f"{args.output}/data/convergence_stats.csv", index=False)
    print(f"\n✅ Saved: {args.output}/data/convergence_stats.csv")
    
    # 绘制综合对比图 (不同 M 值的 Whittle 收敛对比)
    if len(M_values) > 1:
        for N in N_values:
            traj_list = [all_trajectories[f"N{N}_M{M}"] for M in M_values 
                        if f"N{N}_M{M}" in all_trajectories]
            labels = [f"M={M}" for M in M_values if f"N{N}_M{M}" in all_trajectories]
            
            if traj_list:
                plot_convergence_comparison(
                    traj_list, labels,
                    f"{args.output}/figures/convergence_whittle_N{N}_comparison.png",
                    policy='Whittle'
                )
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("📊 CONVERGENCE SUMMARY")
    print("=" * 60)
    
    # 按策略分组显示平均收敛时间
    print("\nAverage Convergence Time (timesteps):")
    conv_summary = stats_df.groupby('policy')['convergence_time'].mean()
    for policy, conv_time in conv_summary.items():
        print(f"  {policy}: {conv_time:.0f}")
    
    # Whittle vs Myopic 稳态对比
    print("\nSteady-State Performance (Post Burn-in):")
    for (N, M), group in stats_df.groupby(['N', 'M']):
        whittle = group[group['policy'] == 'Whittle']['post_burn_in_mean'].values
        myopic = group[group['policy'] == 'Myopic']['post_burn_in_mean'].values
        if len(whittle) > 0 and len(myopic) > 0:
            gap = (myopic[0] - whittle[0]) / myopic[0] * 100
            print(f"  N={N}, M={M}: Whittle={whittle[0]:.2f}, Myopic={myopic[0]:.2f}, Gap={gap:+.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ CONVERGENCE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  📁 {args.output}/data/")
    print(f"     - convergence_stats.csv")
    print(f"     - trajectories_*.csv")
    print(f"  📁 {args.output}/figures/")
    print(f"     - convergence_*.png/pdf")


if __name__ == "__main__":
    main()
