"""
统计显著性分析脚本 (Statistical Significance Analysis)
======================================================

功能:
1. 读取现有实验 CSV 文件
2. 计算 95% 置信区间 (CI)
3. 计算 Whittle vs Myopic 的 t-test p-value
4. 生成增强的 CSV 和可视化图表

使用方法:
    python 06_statistical_analysis.py --input results/data --output results/statistics
    python 06_statistical_analysis.py --input results/data --output results/statistics --n-seeds 5

输出:
    - fig1_n_sweep_stats.csv (带 CI 的数据)
    - fig2_m_sweep_stats.csv
    - fig3_ps_sweep_stats.csv
    - significance_summary.csv (显著性汇总)
    - fig_ci_comparison.png/pdf (带误差棒的对比图)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
import warnings
warnings.filterwarnings('ignore')


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
# 统计分析函数
# =============================================================================

def compute_confidence_interval(mean: float, std: float, n: int, 
                                 confidence: float = 0.95) -> Tuple[float, float]:
    """
    计算置信区间
    
    Args:
        mean: 样本均值
        std: 样本标准差
        n: 样本数 (seeds)
        confidence: 置信水平
    
    Returns:
        (ci_lower, ci_upper)
    """
    if n <= 1:
        return mean, mean
    
    se = std / np.sqrt(n)  # Standard Error
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_critical * se
    
    return mean - margin, mean + margin


def compute_significance(mean1: float, std1: float, 
                         mean2: float, std2: float,
                         n: int) -> Dict[str, float]:
    """
    计算两组数据的显著性 (基于汇总统计量的近似 t-test)
    
    注意: 这是基于汇总数据的近似方法，完整的 t-test 需要原始数据
    """
    if n <= 1:
        return {'t_stat': 0, 'p_value': 1.0, 'significant': False}
    
    # Welch's t-test approximation
    se1 = std1 / np.sqrt(n)
    se2 = std2 / np.sqrt(n)
    se_diff = np.sqrt(se1**2 + se2**2)
    
    if se_diff < 1e-10:
        return {'t_stat': 0, 'p_value': 1.0, 'significant': False}
    
    t_stat = (mean1 - mean2) / se_diff
    
    # Welch-Satterthwaite degrees of freedom
    df = (se1**2 + se2**2)**2 / (se1**4/(n-1) + se2**4/(n-1))
    df = max(1, df)
    
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'highly_significant': p_value < 0.01
    }


def add_ci_to_dataframe(df: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """为 DataFrame 添加置信区间列"""
    df = df.copy()
    
    ci_results = []
    for idx, row in df.iterrows():
        ci_lower, ci_upper = compute_confidence_interval(
            row['mean_aoii'], row['std_aoii'], n_seeds
        )
        ci_results.append({
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'ci_pct': (ci_upper - ci_lower) / row['mean_aoii'] * 100 if row['mean_aoii'] > 0 else 0
        })
    
    ci_df = pd.DataFrame(ci_results)
    return pd.concat([df, ci_df], axis=1)


def compute_pairwise_significance(df: pd.DataFrame, n_seeds: int,
                                   baseline: str = 'Whittle',
                                   compare_to: str = 'Myopic') -> pd.DataFrame:
    """
    计算每个实验点上 Whittle vs Myopic 的显著性
    """
    results = []
    
    # 确定分组列 (排除 policy 和统计列)
    stat_cols = ['policy', 'mean_aoii', 'std_aoii', 'mean_delta', 
                 'ci_lower', 'ci_upper', 'ci_width', 'ci_pct']
    group_cols = [c for c in df.columns if c not in stat_cols]
    
    if not group_cols:
        # 如果没有分组列，按行处理
        group_cols = ['N', 'M'] if 'N' in df.columns else ['p_s']
    
    # 过滤只保留存在的列
    group_cols = [c for c in group_cols if c in df.columns]
    
    if not group_cols:
        print("Warning: No grouping columns found")
        return pd.DataFrame()
    
    for group_vals, group_df in df.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        
        baseline_row = group_df[group_df['policy'] == baseline]
        compare_row = group_df[group_df['policy'] == compare_to]
        
        if len(baseline_row) == 0 or len(compare_row) == 0:
            continue
        
        baseline_mean = baseline_row['mean_aoii'].values[0]
        baseline_std = baseline_row['std_aoii'].values[0]
        compare_mean = compare_row['mean_aoii'].values[0]
        compare_std = compare_row['std_aoii'].values[0]
        
        sig_result = compute_significance(
            baseline_mean, baseline_std,
            compare_mean, compare_std,
            n_seeds
        )
        
        # 计算改进幅度
        improvement = (compare_mean - baseline_mean) / compare_mean * 100 if compare_mean > 0 else 0
        
        result = dict(zip(group_cols, group_vals))
        result.update({
            f'{baseline}_mean': baseline_mean,
            f'{compare_to}_mean': compare_mean,
            'improvement_pct': improvement,
            't_statistic': sig_result['t_stat'],
            'p_value': sig_result['p_value'],
            'significant_0.05': sig_result['significant'],
            'significant_0.01': sig_result['highly_significant'],
            'significance_level': '***' if sig_result['p_value'] < 0.001 else 
                                  '**' if sig_result['p_value'] < 0.01 else
                                  '*' if sig_result['p_value'] < 0.05 else 'ns'
        })
        results.append(result)
    
    return pd.DataFrame(results)


# =============================================================================
# 可视化函数
# =============================================================================

def plot_ci_comparison(df: pd.DataFrame, x_col: str, 
                       output_path: str,
                       xlabel: str = None,
                       title: str = None) -> plt.Figure:
    """
    绘制带置信区间的策略对比图
    """
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    policies = ['Whittle', 'Myopic', 'MaxAge', 'Random']
    colors = {
        'Whittle': '#2ecc71',
        'Myopic': '#e74c3c',
        'MaxAge': '#3498db',
        'Random': '#95a5a6'
    }
    markers = {
        'Whittle': 'o',
        'Myopic': 's',
        'MaxAge': '^',
        'Random': 'd'
    }
    
    x_values = sorted(df[x_col].unique())
    
    for policy in policies:
        if policy not in df['policy'].values:
            continue
            
        policy_df = df[df['policy'] == policy].sort_values(x_col)
        
        means = policy_df['mean_aoii'].values
        ci_lower = policy_df['ci_lower'].values
        ci_upper = policy_df['ci_upper'].values
        x = policy_df[x_col].values
        
        yerr = np.array([means - ci_lower, ci_upper - means])
        
        ax.errorbar(x, means, yerr=yerr,
                   label=policy, color=colors.get(policy, 'gray'),
                   marker=markers.get(policy, 'o'),
                   markersize=5, capsize=3, capthick=1,
                   linewidth=1.5, linestyle='-')
    
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel('Average AoII')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    return fig


def plot_significance_heatmap(sig_df: pd.DataFrame, 
                               x_col: str, y_col: str,
                               output_path: str) -> plt.Figure:
    """
    绘制显著性热力图
    """
    setup_ieee_style()
    
    if x_col not in sig_df.columns or y_col not in sig_df.columns:
        print(f"Warning: {x_col} or {y_col} not in dataframe")
        return None
    
    pivot = sig_df.pivot(index=y_col, columns=x_col, values='improvement_pct')
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', 
                   vmin=-10, vmax=20)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{v}' for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{v}' for v in pivot.index])
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Whittle Improvement (%)')
    
    # 添加数值标注
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            sig_row = sig_df[(sig_df[y_col] == pivot.index[i]) & 
                            (sig_df[x_col] == pivot.columns[j])]
            if len(sig_row) > 0:
                sig_level = sig_row['significance_level'].values[0]
                text = f'{val:.1f}\n{sig_level}'
            else:
                text = f'{val:.1f}'
            
            color = 'white' if abs(val) > 10 else 'black'
            ax.text(j, i, text, ha='center', va='center', 
                   fontsize=6, color=color)
    
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    return fig


def generate_latex_table(sig_df: pd.DataFrame, output_path: str):
    """
    生成 LaTeX 格式的显著性表格
    """
    # 格式化数据
    formatted = sig_df.copy()
    
    if 'Whittle_mean' in formatted.columns:
        formatted['Whittle_mean'] = formatted['Whittle_mean'].apply(lambda x: f'{x:.2f}')
    if 'Myopic_mean' in formatted.columns:
        formatted['Myopic_mean'] = formatted['Myopic_mean'].apply(lambda x: f'{x:.2f}')
    if 'improvement_pct' in formatted.columns:
        formatted['improvement_pct'] = formatted.apply(
            lambda row: f"{row['improvement_pct']:.1f}\\%{row['significance_level']}", axis=1
        )
    if 'p_value' in formatted.columns:
        formatted['p_value'] = formatted['p_value'].apply(
            lambda x: f'{x:.4f}' if x >= 0.001 else '<0.001'
        )
    
    # 选择输出列
    output_cols = [c for c in ['N', 'M', 'p_s', 'Whittle_mean', 'Myopic_mean', 
                               'improvement_pct', 'p_value'] if c in formatted.columns]
    
    latex = formatted[output_cols].to_latex(index=False, escape=False)
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ Saved: {output_path}")


# =============================================================================
# 主函数
# =============================================================================

def analyze_experiment_file(input_path: str, output_dir: str, 
                            n_seeds: int, file_prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """分析单个实验文件"""
    
    print(f"\n📊 Analyzing: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} rows")
    
    # 添加置信区间
    df_with_ci = add_ci_to_dataframe(df, n_seeds)
    
    # 保存增强数据
    output_csv = f"{output_dir}/data/{file_prefix}_stats.csv"
    df_with_ci.to_csv(output_csv, index=False)
    print(f"   ✅ Saved: {output_csv}")
    
    # 计算显著性
    sig_df = compute_pairwise_significance(df_with_ci, n_seeds)
    
    if len(sig_df) > 0:
        sig_csv = f"{output_dir}/data/{file_prefix}_significance.csv"
        sig_df.to_csv(sig_csv, index=False)
        print(f"   ✅ Saved: {sig_csv}")
    
    return df_with_ci, sig_df


def main():
    parser = argparse.ArgumentParser(description='Statistical Significance Analysis')
    parser.add_argument('--input', type=str, default='results/data',
                       help='Input directory containing CSV files')
    parser.add_argument('--output', type=str, default='results/statistics',
                       help='Output directory')
    parser.add_argument('--n-seeds', type=int, default=5,
                       help='Number of seeds used in experiments')
    args = parser.parse_args()
    
    # 创建输出目录
    Path(f"{args.output}/data").mkdir(parents=True, exist_ok=True)
    Path(f"{args.output}/figures").mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📊 STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"N seeds: {args.n_seeds}")
    print("=" * 60)
    
    setup_ieee_style()
    
    all_significance = []
    
    # 分析 Fig 1: N Sweep
    fig1_files = list(Path(args.input).glob('fig1_n_sweep*.csv'))
    for f in fig1_files:
        prefix = f.stem
        df, sig_df = analyze_experiment_file(str(f), args.output, args.n_seeds, prefix)
        
        if len(df) > 0:
            plot_ci_comparison(df, 'N', 
                              f"{args.output}/figures/{prefix}_ci.png",
                              xlabel='Number of Arms $N$')
        
        if len(sig_df) > 0:
            sig_df['experiment'] = 'N_sweep'
            all_significance.append(sig_df)
    
    # 分析 Fig 2: M Sweep
    fig2_files = list(Path(args.input).glob('fig2_m_sweep*.csv'))
    for f in fig2_files:
        prefix = f.stem
        df, sig_df = analyze_experiment_file(str(f), args.output, args.n_seeds, prefix)
        
        if len(df) > 0:
            plot_ci_comparison(df, 'M',
                              f"{args.output}/figures/{prefix}_ci.png",
                              xlabel='Budget $M$')
        
        if len(sig_df) > 0:
            sig_df['experiment'] = 'M_sweep'
            all_significance.append(sig_df)
    
    # 分析 Fig 3: p_s Sweep
    fig3_files = list(Path(args.input).glob('fig3_ps_sweep*.csv'))
    for f in fig3_files:
        prefix = f.stem
        df, sig_df = analyze_experiment_file(str(f), args.output, args.n_seeds, prefix)
        
        if len(df) > 0:
            plot_ci_comparison(df, 'p_s',
                              f"{args.output}/figures/{prefix}_ci.png",
                              xlabel='Sync Success Rate $p_s$')
        
        if len(sig_df) > 0:
            sig_df['experiment'] = 'ps_sweep'
            all_significance.append(sig_df)
    
    # 汇总所有显著性结果
    if all_significance:
        combined_sig = pd.concat(all_significance, ignore_index=True)
        combined_sig.to_csv(f"{args.output}/data/all_significance_summary.csv", index=False)
        print(f"\n✅ Combined significance summary saved")
        
        # 生成 LaTeX 表格
        generate_latex_table(combined_sig, f"{args.output}/data/significance_table.tex")
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("📊 SIGNIFICANCE SUMMARY")
        print("=" * 60)
        
        n_total = len(combined_sig)
        n_sig_05 = combined_sig['significant_0.05'].sum()
        n_sig_01 = combined_sig['significant_0.01'].sum()
        
        print(f"Total comparisons: {n_total}")
        print(f"Significant (p<0.05): {n_sig_05} ({n_sig_05/n_total*100:.1f}%)")
        print(f"Highly significant (p<0.01): {n_sig_01} ({n_sig_01/n_total*100:.1f}%)")
        
        print("\nBest improvements:")
        best = combined_sig.nlargest(5, 'improvement_pct')
        for _, row in best.iterrows():
            exp = row.get('experiment', '')
            imp = row['improvement_pct']
            sig = row['significance_level']
            print(f"  {exp}: +{imp:.1f}% {sig}")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  📁 {args.output}/data/")
    print(f"  📁 {args.output}/figures/")


if __name__ == "__main__":
    main()
