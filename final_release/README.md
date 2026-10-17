# Huawei_DPD 最终交付包说明

本目录是可直接打包提交的整理版项目包。默认任务口径为：

```text
pred_y = f(x)
```

即根据数据集输入 `x` 拟合官方数据集给出的目标输出 `target_y`。验证集按全序列前 80% / 后 20% 切分，核心指标为验证集 NMSE 和 PSD 计算得到的 worst-band ACLR。

## 目录结构

```text
data/                         官方 DPD_data.mat
code/                         交付版训练与评估脚本
checkpoints/AI-DPD/           交付版 checkpoint，脚本默认从这里读取
figures/                      已生成的报告展示图归档
eval_outputs_direct/figures/  重新运行评估脚本后生成的 PSD/ACLR 图
eval_outputs_direct/unified_eval/ 统一评估 CSV 与 PSD 图
reports/                      阶段总结、模型卡和报告文档
```

## 最终模型

| 用途 | 结构 | 实乘法器 | 验证 NMSE | worst ACLR | 权重 |
|---|---:|---:|---:|---:|---|
| 推荐主模型 | H15/L6/Input-9/FIR43 | 2284 | -40.15 dB | -42.98 dBc | `checkpoints/AI-DPD/final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth` |
| 最小达标模型 | H15/L6/Input-9/FIR39 | 2268 | -40.03 dB | -42.80 dBc | `checkpoints/AI-DPD/final_minimal_H15_L6_rich9_FIR39_2268_NMSE40p03_ACLR45p04.pth` |
| 单脚本备份 | H15/L6/Input-9/FIR55 | 2332 | -40.00 dB | -42.79 dBc | `checkpoints/AI-DPD/final_backup_H15_L6_rich9_FIR55_2332_NMSE40p00_ACLR47p11.pth` |
| 1000 内轻量参考 | H10/L4/Input-9/attn4/FIR39 | 998 | -37.37 dB | -40.09 dBc | `checkpoints/AI-DPD/final_lightweight_998_rich9_attn4_FIR39_NMSE37p37_ACLR45p30.pth` |
| 800 内低秩参考 | H10/L4/Input-6/rank3/FIR41 | 786 | -35.46 dB | -38.51 dBc | `checkpoints/AI-DPD/step800_scratch_h10_l4_base6_rank3_fir41_786.pth` |
| 800 内基线参考 | H10/L3/Input-6/FIR41 | 796 | -35.20 dB | -38.03 dBc | `checkpoints/AI-DPD/final_lightweight_796_H10_L3_FIR41_scratch_retrain.pth` |

说明：在新频带 `[20,80] / [-180,-120] / [120,180] MHz` 下，数据集参考 ACLR 为 `-43.74 dBc`，因此上述模型均不满足原 `<= -45 dBc` 门限；2284/2268/2332 MAC 模型仍满足 NMSE `<= -40 dB`。

## 786 模型参数

| 项目 | 参数 |
|---|---|
| preset | `h10_l4_base6_rank3_fir41_786` |
| 结构 | H10/L4/Input-6/pointwise-rank3/FIR41 |
| 输入特征 | `[I[n], Q[n], I[n+1], Q[n+1], r[n], r[n]^3]` |
| TCN dilation | `[1, 2, 4, 8]` |
| temporal kernel | 5 |
| attention | 3 通道包络门控 `[r, r^3, r^7]` |
| polynomial bypass | 1/3/5 阶复数旁路 |
| output FIR | 41 taps complex FIR |
| 训练方式 | scratch-only，`teacher-mode none`，无蒸馏 |
| 统一评估结果 | NMSE -35.46196 dB，worst ACLR -38.50851 dBc |

乘法器拆解：

```text
P=3, C_in=6, H=10, B=4, K=5, rank=3, C_a=3, M=41
C_real = 4P + C_in*H + B*(H*K + 2*H*rank) + 2H + 3*H*C_a + 4M
       = 12 + 60 + 4*(50 + 60) + 20 + 90 + 164
       = 786
```

## 指标口径

```text
NMSE = sum(|pred_y - target_y|^2) / sum(|target_y|^2)
NMSE_dB = 10 * log10(NMSE)

error = target_y - pred_y
ACLR_dBc = mean(PSD_error_adjacent_band) - mean(PSD_target_y_main_band)
worst ACLR = max(lower ACLR, upper ACLR)

dataset reference ACLR uses DPD_data.mat error in place of model error

main band:      [20, 80] MHz
lower adjacent: [-180, -120] MHz
upper adjacent: [120, 180] MHz
sample rate:    491.52 MHz
target:         worst ACLR <= -45 dBc
```

## 主要脚本

```text
code/AI-DPD.py
```

共享模型定义、数据加载、统一 NMSE/ACLR/PSD 评估与绘图模块。

```text
code/final_train_main_2284.py
```

主模型、最小达标模型和备份模型的训练/导出脚本。三者共享 H15/L6/Input-9 主干，区别是最终输出 FIR taps。

```text
code/final_train_lightweight_999_distill.py
```

轻量学生模型脚本，支持 796/786/998 等低复杂度配置，并支持 `--evaluate-only` 重绘 PSD/ACLR 图。

```text
code/final_train_step800_786.py
```

786 MAC 低秩模型专用入口。该脚本默认调用 `h10_l4_base6_rank3_fir41_786`，并锁定 scratch-only、无蒸馏和频域 FIR refit 参数。

```text
code/final_evaluate_unified.py
```

统一批量评估入口，用于对 checkpoint 输出线性 NMSE、dB NMSE、实乘法器数、ACLR 和 PSD 图。

## 完整训练命令

先进入交付包目录：

```powershell
cd D:\PyChram_project\Huawei_DPD\final_release
```

训练 2284 MAC 推荐主模型：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_main_2284.py --final-fir-taps 43 --no-fallback-scan --final-checkpoint .\checkpoints\AI-DPD\final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth --figure-prefix final_main_H15_L6_rich9_FIR43_2284
```

训练 2268 MAC 最小达标模型：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_main_2284.py --final-fir-taps 39 --no-fallback-scan --final-checkpoint .\checkpoints\AI-DPD\final_minimal_H15_L6_rich9_FIR39_2268_NMSE40p03_ACLR45p04.pth --figure-prefix final_minimal_H15_L6_rich9_FIR39_2268
```

训练 2332 MAC 备份模型：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_main_2284.py --final-fir-taps 55 --no-fallback-scan --final-checkpoint .\checkpoints\AI-DPD\final_backup_H15_L6_rich9_FIR55_2332_NMSE40p00_ACLR47p11.pth --figure-prefix final_backup_H15_L6_rich9_FIR55_2332
```

训练 998 MAC 轻量参考模型：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_lightweight_999_distill.py --student-preset h10_l4_rich9_attn4_fir39_998 --teacher-mode load --teacher-checkpoint .\checkpoints\AI-DPD\final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth --final-checkpoint .\checkpoints\AI-DPD\final_lightweight_998_rich9_attn4_FIR39_NMSE37p37_ACLR45p30.pth --output-prefix final_lightweight_998_rich9_attn4_FIR39
```

训练 786 MAC 低秩参考模型：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_step800_786.py
```

训练 796 MAC 800 内基线模型：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_lightweight_999_distill.py --student-preset h10_l3_base6_fir41_796 --teacher-mode none --distill-loss-weight 0 --distill-diff-weight 0 --final-checkpoint .\checkpoints\AI-DPD\final_lightweight_796_H10_L3_FIR41_scratch_retrain.pth --output-prefix final_lightweight_796_H10_L3_FIR41_scratch_retrain
```

## 评估命令

统一批量评估所有最终 checkpoint：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_evaluate_unified.py
```

只评估 786 MAC 模型并重绘 PSD/ACLR 图：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_step800_786.py --evaluate-only
```

只评估 796 MAC 模型并重绘 PSD/ACLR 图：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_lightweight_999_distill.py --evaluate-only --student-preset h10_l3_base6_fir41_796 --final-checkpoint .\checkpoints\AI-DPD\final_lightweight_796_H10_L3_FIR41_scratch_retrain.pth --output-prefix final_lightweight_796_H10_L3_FIR41_scratch_retrain
```

只评估 998 MAC 模型并重绘 PSD/ACLR 图：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_lightweight_999_distill.py --evaluate-only --student-preset h10_l4_rich9_attn4_fir39_998 --final-checkpoint .\checkpoints\AI-DPD\final_lightweight_998_rich9_attn4_FIR39_NMSE37p37_ACLR45p30.pth --output-prefix final_lightweight_998_rich9_attn4_FIR39
```

只评估主模型：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_main_2284.py --evaluate-only
```

评估最小达标/备份模型时指定 FIR taps、checkpoint 和图名前缀：

```powershell
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_main_2284.py --evaluate-only --final-fir-taps 39 --final-checkpoint .\checkpoints\AI-DPD\final_minimal_H15_L6_rich9_FIR39_2268_NMSE40p03_ACLR45p04.pth --figure-prefix final_minimal_H15_L6_rich9_FIR39_2268
& D:\anconda\envs\pytorch\python.exe -B .\code\final_train_main_2284.py --evaluate-only --final-fir-taps 55 --final-checkpoint .\checkpoints\AI-DPD\final_backup_H15_L6_rich9_FIR55_2332_NMSE40p00_ACLR47p11.pth --figure-prefix final_backup_H15_L6_rich9_FIR55_2332
```

## 展示图

已归档的报告展示图在：

```text
figures/
```

重新运行训练/评估脚本后会输出到：

```text
eval_outputs_direct/figures/
eval_outputs_direct/unified_eval/
```

核心图片：

```text
final_main_H15_L6_rich9_FIR43_2284_psd_aclr.png
final_main_H15_L6_rich9_FIR43_2284_aclr_bar.png
final_minimal_H15_L6_rich9_FIR39_2268_psd_aclr.png
final_minimal_H15_L6_rich9_FIR39_2268_aclr_bar.png
final_backup_H15_L6_rich9_FIR55_2332_psd_aclr.png
final_backup_H15_L6_rich9_FIR55_2332_aclr_bar.png
final_lightweight_998_rich9_attn4_FIR39_psd_aclr.png
final_lightweight_998_rich9_attn4_FIR39_aclr_bar.png
step800_scratch_h10_l4_base6_rank3_fir41_786_psd_aclr.png
step800_scratch_h10_l4_base6_rank3_fir41_786_aclr_bar.png
step800_scratch_h10_l4_base6_rank3_fir41_786_convergence.png
final_lightweight_796_H10_L3_FIR41_scratch_retrain_psd_aclr.png
final_lightweight_796_H10_L3_FIR41_scratch_retrain_aclr_bar.png
```

## 报告文件

```text
reports/TCN_999_experiment_summary_2026-06-28.md
reports/step800_scratch_innovative_summary_2026-06-29.md
reports/step800_786_model_card_2026-07-03.md
reports/Huawei_DPD_结题报告书_2026-06-29_796scratch.docx
reports/E3_DPD_multiplier_buckets_model_summary_2026-06-28_796scratch.docx
```
