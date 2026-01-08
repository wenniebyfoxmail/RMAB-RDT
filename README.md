# Road Digital Twin AoII-ARD RMAB Simulation

**Parallel-Enabled Edition** - 自动检测CPU核心数并行执行

## 项目概述 test etst test2

本项目实现了基于 **Age of Incorrect Information (AoII)** 与 **Age-Rate-Distortion (ARD)** 理论的 **Restless Multi-Armed Bandit (RMAB)** 调度仿真系统。

### 核心特性
- ⚡ **自动并行计算**: 检测CPU核心数，多seed并行执行
- 📊 **完整论文图表**: Fig1-5 + Table1 一键生成
- ✅ **导师决策已落实**: Q1-Q6全部实现

---

## 快速开始

### 1. 环境配置

```bash
# 解压
unzip rmab_road_dt_parallel.zip
cd clean_package

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 一键运行

```bash
# 快速测试 (~15-20分钟，验证代码正确性)
python run_all.py --quick

# 完整实验 (~1-1.5小时，论文级结果)
python run_all.py --full

# 指定核心数（默认自动检测）
python run_all.py --full --workers 4
```

### 3. 单独运行各脚本

```bash
# 按执行顺序运行

# [01] 主实验 (Fig1-3, Table1) - P0必须
python 01_main_experiments.py --full --output results

# [02] Regime Map (Fig4) - P1顶刊防守
python 02_regime_map.py --output results

# [03] Time-Varying (Fig5) - P1 Novelty证据
python 03_time_varying.py --output results

# [04] Indexability - P1附录验证
python 04_indexability.py --output results/indexability

# [05] Noise Sensitivity - P2附录
python 05_noise_sensitivity.py --output results

# [06] LP Comparison (Fig6) - 回应审稿人 ⭐新增
python lp_comparison.py --output results
```

---

## 📁 文件结构

```
clean_package/
│
├── 核心模块 (Core Modules)
│   ├── config.py             # 配置与参数（含工程语义）
│   ├── nhgp_builder.py       # NHGP转移矩阵构建器
│   ├── environment.py        # RMAB环境
│   ├── policies.py           # 调度策略
│   ├── whittle_solver.py     # Whittle Index求解器
│   └── parallel_utils.py     # 并行计算工具 ⭐
│
├── 实验脚本 (Experiment Scripts) - 按顺序命名
│   ├── 01_main_experiments.py   # P0: Fig1-3, Table1
│   ├── 02_regime_map.py         # P1: Fig4 策略边界
│   ├── 03_time_varying.py       # P1: Fig5 季节性变化
│   ├── 04_indexability.py       # P1: 可索引性验证
│   └── 05_noise_sensitivity.py  # P2: Q_R噪声敏感性
│
├── 运行脚本 (Runner)
│   └── run_all.py            # 一键运行所有实验
│
├── 配置文件
│   ├── requirements.txt      # Python依赖
│   └── README.md             # 本文档
│
└── results/                  # 输出目录
    ├── data/                 # CSV数据
    ├── figures/              # PDF/PNG图表
    └── indexability/         # 验证图
```

---

## ⚡ 并行计算说明

### 自动检测
```python
from parallel_utils import get_cpu_count, get_optimal_workers

print(f"CPU cores: {get_cpu_count()}")      # 检测核心数
print(f"Workers: {get_optimal_workers()}")   # 推荐worker数
```

### 性能提升（估计）

| 环境 | 核心数 | 预计时间 (full) |
|------|--------|-----------------|
| Colab (免费) | 2 | ~2小时 |
| Colab Pro | 4 | ~1.5小时 |
| 本地 (8核) | 7 | ~45分钟 |
| 本地 (16核) | 15 | ~30分钟 |

---

## 📊 输出文件

### P0: 主实验（论文主图）
| 文件 | 说明 |
|------|------|
| `fig1_n_sweep.csv/pdf` | N sweep: AoII vs 臂数量 |
| `fig2_m_sweep.csv/pdf` | M sweep: AoII vs 预算 |
| `fig3_ps_sweep.csv/pdf` | p_s sweep: AoII vs 信道可靠性 |
| `table1_summary.csv` | 统计摘要表 |

### P1: 顶刊防守件
| 文件 | 说明 |
|------|------|
| `fig4_regime_map.csv/pdf` | Regime Map: Whittle vs Myopic边界 ⭐ |
| `fig5_time_varying.csv/pdf` | Time-Varying: 季节性验证 ⭐ |
| `fig6_lp_comparison.csv/pdf` | LP Bound vs Whittle性能 + Wall-clock ⭐ |
| `indexability_*.png` | 可索引性验证 |

### P2: 附录 & 校准
| 文件 | 说明 |
|------|------|
| `ltpp_calibration.csv/pdf` | LTPP参数校准验证 ⭐新增 |
| `noise_sensitivity.csv/pdf` | Q_R噪声敏感性分析 |

---

## 🔧 导师决策落实

| 决策 | 实现位置 |
|------|----------|
| Q1: 季节性c(t) + 窗口化P̄ | `03_time_varying.py` |
| Q2: 主线D=0，附录Q_R | `05_noise_sensitivity.py` |
| Q3: 1 epoch = 1月 | `config.py` 注释 |
| Q4: PCI五档映射 | `config.py` 注释 |
| Q5: 交通荷载差异 | `config.py` 注释 |
| Q6: LTPP量级说明 | `config.py` 注释 |

---

## ✅ 结果验证清单

运行完成后，验证以下趋势：

- [ ] **Fig1**: N↑ → AoII↑
- [ ] **Fig2**: M↑ → AoII↓
- [ ] **Fig3**: p_s↑ → AoII↓
- [ ] **Fig4**: 低p_s/低M区域Whittle优势明显
- [ ] **Fig5**: Windowed接近Oracle，优于Fixed
- [ ] **Table1**: Random显著最差（>200% gap）
- [ ] **Indexability**: passive set单调递增

---

## 📝 Colab使用说明

```python
# 在Colab中运行

# 1. 上传zip文件后解压
!unzip rmab_road_dt_parallel.zip
%cd clean_package

# 2. 安装依赖
!pip install -r requirements.txt

# 3. 查看系统信息
!python -c "from parallel_utils import print_system_info; print_system_info()"

# 4. 运行实验
!python run_all.py --quick  # 先跑quick验证

# 5. 完整实验
!python run_all.py --full
```

---

## 技术规范

- **DR-06A**: ARD建模与规范
- **DR-06B**: ARD极限与最优更新律
- **DR-06C**: RMAB调度Whittle Index
- **DR-07**: 仿真图表规格
- **Advisor Q1-Q6**: 导师决策
