# Huawei DPD：轻量级 AI 数字预失真

这是第三届华为无线“声光电磁”硬件创新竞赛 E3 赛题的项目记录。我们尝试用一个计算量受限的神经网络学习数字预失真（DPD）：给定输入信号 `x`，模型预测数据集提供的目标信号 `target_y`，希望在控制硬件计算量的同时，减小预测误差和邻道频谱泄漏。

仓库里保留了模型、训练与评估脚本，以及不同计算量下的实验记录。如果只是想了解项目，可以先看下面的结果；需要复现时，再看运行说明和评估口径。

## 项目做了什么

- 使用时序卷积网络（TCN）处理 I/Q 信号，并结合包络特征、多项式旁路和输出 FIR 滤波。
- 比较约 2,000、1,000 和 800 个实乘法器量级的模型，观察模型误差与计算量之间的取舍。这里的实乘法器数量是项目中的复杂度估计，不等同于已经完成的硬件资源实测。
- 用同一份数据的前 80% 训练、后 20% 验证，记录 NMSE 和频谱相关结果。

## 已保存的实验结果

下表摘自仓库中已有的 `eval_outputs_direct/unified_eval/summary.csv`；它反映的是这份历史结果文件，不代表本次重新运行的结果。

| 模型 | 实乘法器数 | 验证 NMSE | 已保存的 worst ACLR |
| --- | ---: | ---: | ---: |
| H15/L6，FIR 43 | 2,284 | -40.15 dB | -42.98 dBc |
| H15/L6，FIR 39 | 2,268 | -40.03 dB | -42.80 dBc |
| H10/L4，FIR 39 | 998 | -37.37 dB | -40.09 dBc |
| H10/L4，低秩，FIR 41 | 786 | -35.46 dB | -38.51 dBc |
| H10/L3，FIR 41 | 796 | -35.20 dB | -38.03 dBc |

历史记录中，前两个模型的 NMSE 达到 `≤ -40 dB`，但表中的 ACLR 均未达到 `≤ -45 dBc`。这些数值不能脱离所用频段和计算方法比较。

**评估口径需要注意：**旧 README 把 ACLR 写成频带平均值之差，而当前 `steps/final_evaluate_unified.py` 使用各频带 PSD 最低点计算。保存的 CSV 与当前脚本的输出字段也不同。表中数值应作为历史记录；如果要按当前脚本报告或比较结果，请重新评估并注明算法、频段与权重版本，不要直接把两种口径混用。

## 如何运行

以下命令在项目根目录执行。脚本使用 Python、PyTorch、NumPy、SciPy 和 Matplotlib；仓库未提供锁定版本的环境文件，因此需要先按自己的 Python 与计算设备配置依赖。

```bash
python -m pip install numpy scipy matplotlib torch
```

确认 `data/DPD_data.mat` 可用后，可以从头训练主模型：

```bash
python -B steps/final_train_main_2284.py --no-fallback-scan
```

默认最多训练 500 个 epoch，运行时间取决于设备和数据；重新训练不保证得到表中的完全相同结果。

如果已有主模型权重，可以单独运行评估：

```bash
python -B steps/final_train_main_2284.py --evaluate-only
```

**权重文件不会随此本地 Git 快照提交。**仓库的 `.gitignore` 排除了 `checkpoints/` 和 `*.pth`；上述评估命令需要先训练得到权重，或自行提供与脚本匹配的权重文件。默认路径可通过 `--final-checkpoint` 修改。

要使用统一评估脚本，可显式指定已有权重和输出目录，例如：

```bash
python -B steps/final_evaluate_unified.py --checkpoint checkpoints/AI-DPD/final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth --out-dir eval_outputs_direct/my_eval
```

脚本会在输出目录写入 `summary.csv` 和 PSD 图；重复运行可能覆盖同名输出。权重文件名中带有旧实验指标，不能将文件名里的 ACLR 数字直接当作当前评估结果。

## 指标与频段

- **NMSE**：预测信号相对目标信号的归一化均方误差；这里以 dB 表示。
- **worst ACLR**：分别计算上下邻道后取较差的一侧；数值的含义取决于所采用的频带统计方法。
- 当前统一评估脚本默认采样率为 `491.52 MHz`，主频带为 `[20, 80] MHz`，下邻道为 `[-180, -120] MHz`，上邻道为 `[120, 180] MHz`。

## 主要文件

| 路径 | 内容 |
| --- | --- |
| `data/DPD_data.mat` | 项目使用的数据文件 |
| `steps/AI-DPD.py` | 模型、数据加载及评估工具 |
| `steps/final_train_main_2284.py` | 主模型训练与评估入口 |
| `steps/final_train_lightweight_999_distill.py` | 轻量模型实验入口 |
| `steps/step800_scratch_innovative.py` | 800 实乘法器量级的低秩模型实验 |
| `steps/final_evaluate_unified.py` | 指定权重的统一评估与绘图 |
| `eval_outputs_direct/` | 已保存的评估结果和图 |
| `reports/` | 阶段性实验总结 |
| `other/final_cleanup_20260629/` | 历史实验与中间材料归档 |

本项目是竞赛实验记录；结果受数据、模型权重和评估口径影响，不应视为已完成实机部署或硬件性能验证。
