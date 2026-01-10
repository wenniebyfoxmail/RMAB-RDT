# 🛣️ RMAB-RDT 实验指南

## 📊 参数对比一览表

### 1. `--quick` vs 完整实验

| 参数 | `--quick` | 完整实验 | 说明 |
|------|-----------|----------|------|
| 时间步 T | 500 | 2000 | 每次实验的模拟长度 |
| 随机种子 | 3 | 10 | 统计可靠性 |
| N 值 | [20, 50] | [20, 50, 100, 200] | 路段数量范围 |
| M 值 | [2, 5, 10] | [1, 2, 3, 5, 8, 10] | 预算范围 |
| p_s 值 | [0.90, 0.996] | [0.70~0.996] | 同步成功率 |
| **运行时间** | **~10 分钟** | **~2-4 小时** | |
| **用途** | 调试验证 | 论文结果 | |

---

### 2. `--heterogeneous` vs `--use-ontario`

| 维度 | `--heterogeneous` | `--use-ontario` |
|------|-------------------|-----------------|
| **P̄ (转移矩阵)** | NHGP 模拟生成 | Ontario 真实数据 |
| **p_s 值** | [0.3, 0.7] | [0.7, 0.5, 0.3] |
| **臂类型** | slow, fast | responsive, moderate, unresponsive |
| **论文用途** | Section V-A 理论验证 | Section V-B 案例研究 |
| **数据依赖** | 无（纯模拟） | 需要 data/ontario/*.csv |

---

## 🧪 推荐实验流程

### 阶段 1: Quick 验证（~30 分钟）
```bash
# 1.1 NHGP 异质性（推荐）
python 01_main_experiments.py --quick --heterogeneous --output results/quick_het

# 1.2 Ontario 真实数据
python 01_main_experiments.py --quick --use-ontario --output results/quick_ontario

# 1.3 NHGP 同质性（对照组）
python 01_main_experiments.py --quick --output results/quick_homo
```

### 阶段 2: 完整实验（~6-8 小时）
```bash
# 2.1 NHGP 异质性
python 01_main_experiments.py --heterogeneous --output results/full_het

# 2.2 Ontario 真实数据
python 01_main_experiments.py --use-ontario --output results/full_ontario

# 2.3 NHGP 同质性（对照组）
python 01_main_experiments.py --output results/full_homo
```

---

## 📁 输出文件结构

```
results/
├── quick_het/                    # Quick + NHGP + 异质
│   ├── data/                     # 📊 CSV 数据（诊断用）
│   │   ├── fig1_n_sweep.csv
│   │   ├── fig2_m_sweep.csv
│   │   ├── fig3_ps_sweep.csv
│   │   ├── table1_aoii.csv
│   │   └── p1_optimal_benchmark.csv
│   │
│   └── figures/                  # 📈 图表（论文用）
│       ├── fig1_n_sweep.png      # PNG (300 DPI)
│       ├── fig1_n_sweep.pdf      # PDF (矢量)
│       ├── fig2_m_sweep.png/pdf
│       └── fig3_ps_sweep.png/pdf
│
├── quick_ontario/                # Quick + Ontario
├── quick_homo/                   # Quick + NHGP 同质
├── full_het/                     # 完整 + NHGP 异质
├── full_ontario/                 # 完整 + Ontario
└── full_homo/                    # 完整 + NHGP 同质
```

---

## 📊 CSV 文件格式说明

### fig1_n_sweep.csv
| N | M | policy | mean_aoii | std_aoii | mean_delta |
|---|---|--------|-----------|----------|------------|
| 20 | 1 | Whittle | 29.24 | 2.15 | 24.1 |
| 20 | 1 | Myopic | 31.15 | 2.43 | 25.1 |
| ... | ... | ... | ... | ... | ... |

### 诊断公式
```python
gap = (myopic_aoii - whittle_aoii) / whittle_aoii * 100  # 百分比差距
```

---

## 🎨 Automation in Construction 风格要点

- **图片格式**: PDF（矢量，缩放不失真）
- **DPI**: 300（位图时）
- **字体**: Times New Roman / Computer Modern
- **线宽**: 1.5pt
- **图例位置**: 图内右上角或图外
- **颜色**: 区分度高、色盲友好

---

## ⚠️ 常见问题

### Q: Ontario 数据找不到？
```bash
# 确保文件存在
ls data/ontario/ontario_2022.csv
ls data/ontario/ontario_2023.csv
```

### Q: 内存不足？
```bash
# 使用 --quick 或减少 N 值
python 01_main_experiments.py --quick --output results
```

### Q: 想只跑某个实验？
```bash
# 只跑 P1 benchmark
python 01_main_experiments.py --p1-only --output results
```
