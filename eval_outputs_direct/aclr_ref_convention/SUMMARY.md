# 参考队口径 ACLR 实验总结(2026-07-13)

## 口径定义(来自 APNKAN 报告,已数值复现)

- **NMSE** = 10·log10(Σ|y−ŷ|²/Σ|y|²)(与我们原口径同公式)
- **ACLR** = 10·log10([P_err(−180,−100) + P_err(100,180)] / P_y(−100,100))
  双边邻带误差积分功率合并 ÷ 主带目标积分功率(Welch nperseg=4096)。
  验证:数据集参考 error 按此口径 = **−44.75/−44.83 dBc**(full/val),
  与他们图 8 标注的 −44.59 差 0.16 dB,口径锁定。
- 归一化不变性:rms 与 max 归一化下 NMSE/ACLR **完全相同**(比值指标)。
  归一化只影响 (a) 他们 MSE<1e-4 阈值的解读(他们单位 mean|y|²=0.0253
  → 等价 NMSE ≈ −24 dB;若按单位功率归一化理解则是 −40 dB),
  (b) 从零训练的数值调理。**对评估无影响,rms 归一化照常使用。**

## 目标

ACLR < −45 dBc,NMSE 越好越好;2284 硬约束 NMSE ≤ −40 dB。验证集(后 20%)评估。

## 实验与结果

### 基线(tag: baseline)

| 模型 | NMSE | ACLR(新口径) | 距 −45 |
|---|---|---|---|
| dataset_error | −36.46 | −44.83 | 0.17 |
| 2284 | −40.15 | **−44.09** | 0.91 |
| 998 | −37.37 | −41.21 | 3.79 |
| 786 | −35.46 | −39.45 | 5.55 |
| 796 | −35.20 | −38.99 | 6.01 |

(旧三指标口径的 2284 = −42.98;新口径本身送 ~1.1 dB)

### 闭式 FIR LS 扫参(steps/aclr_ref_fir_sweep.py,tag: fir_sweep)→ 饱和

邻带 (100,180) 均匀权重(积分口径下 LS 的最优形状),adj∈[1,800]×floor×ridge,
144 候选/模型。**全部 ≤0.23 dB 改善**。y 在 (100,180) 有真实预失真再生内容,
模型必须复现而非滤除;线性输出 FIR 无剩余杠杆(与旧口径结论一致)。

### 子带渐进微调(steps/aclr_ref_subband_finetune.py)

他们的机制:L = (1−r)·mean|e|² + r·mean|HPF(e)|² + guard·relu(NMSE−T)²,
HPF=257 阶线性相位高通(98 MHz 截止),r 按他们的调度渐进;帧边裁剪 256;
可选周期性闭式 FIR 重拟合。

探测轮(r≤0.99,lr 1e-4,120ep)全部饱和后,第二轮上强度
(r_max=0.9999、lr 3e-4、400ep、stride 1024 加密步数、2284 用纯子带+强 guard):

| 模型 | 起点 | 终点(tag: subband_v2) | NMSE 代价 | ACLR 收益 |
|---|---|---|---|---|
| 2284 | −40.15 / −44.09 | **−40.21 / −44.19** | 0(guard 锁死) | +0.10 |
| 998 | −37.37 / −41.21 | −34.98 / −41.76 | 2.39 | +0.55 |
| 786 | −35.46 / −39.45 | −31.63 / −40.06 | 3.83 | +0.61 |
| 796 | −35.20 / −38.99 | −31.17 / −39.82 | 4.03 | +0.83 |

**结论:ACLR < −45 无一达成;微调路线在四个模型上全部饱和。**

## 关键发现

1. **兑换率是架构属性**:参考队 5.93 dB NMSE 换 5.07 dB ACLR(0.86 dB/dB,
   不饱和);我们 EA-TCN 家族 0.2~0.25 dB/dB 且 <1 dB 即饱和。
2. **没有 in-sample 差距**:998-v2 训练段 −41.69 vs 验证段 −41.76;
   2284-v2 训练段 −44.25 vs 验证段 −44.16。撞的是容量墙不是泛化墙,
   → **照搬他们 1:0:0 全数据协议也不会改善我们的数字**。
3. **teacher(H32,~4× 容量)NMSE −40.08 时 ACLR = −44.82**,
   恰好贴住数据集 error 线(−44.83)。验证集纪律下,大容量模型在
   NMSE −40 收敛到参考残差水平即停。−45 在该线之下 0.17 dB。
4. 全 checkpoint 摸底(新口径 val):2332 备份 −43.90@−40.00,
   2268 最小 −43.81@−40.03,FIR95 底座 −43.90@−39.90 —— 无一现成达标。

## 对参考队 −45~−51.75 的三层归因(重申)

1. 规则解读:MSE<1e-4(他们单位 ⇔ NMSE −24)→ 10+ dB 带内预算可挥霍;
2. 协议:全数据训练 + 同数据评估(他们报告"展望"自认无独立验证);
3. 架构:相位归一化 + 样条 LUT KAN 的邻带内容表征效率(0.86 dB/dB 兑换率),
   这是我们微调无法复制的部分。

## 建议

1. **向主办方确认 MSE 1e-4 的归一化口径**(max ⇔ −24 dB vs 单位功率 ⇔ −40 dB,
   16 dB 之差决定路线合法性);同时确认 ACLR 的频带/积分口径与评估集。
2. 若允许 −24 dB 级带内误差:我们的架构也到不了 −45(786 烧到 −31.6 只有 −40.06),
   需要架构实验(样条 LUT 输出级/相位归一化特征进 EA-TCN,MAC 预算内)——
   这是下一步唯一有希望的方向。
3. 2284 的诚实数字:**−44.19 dBc @ NMSE −40.21**(新口径,验证集),
   距 −45 还差 0.81 dB,teacher 天花板 −44.82 说明微调路线到此为止。

## 架构实验:相位归一化 + 样条 LUT 输出级(2026-07-13 晚,steps/aclr_phase_spline_experiment.py)

APNKAN 迁移:在 EA-TCN 的 FIR 之前加校正支路
`corr = conj(r)·Σ_i spline_i(feat_i)`,其中 r=conj(x)/|x| 相位归一化,
feat = [|x|, Re/Im(x(n±1)·r)](5 特征,G=24 一阶样条查表),样条表零初始化。
**预算内支付**:APNKAN 口径 2·N_in+N_edge=20 + 旋回 4 + 邻居旋转 8(严格计入)
= 32 MAC,由 FIR −8 抽头(=32 MAC)支付,总 MAC 精确等于原模型。
训练:先只训样条表(250ep),再底座联合微调(200ep,base-lr 2~3e-5),
全程子带 HPF 损失 + NMSE guard + 每 10ep 闭式 FIR 重拟合。

独立复核结果(val,MAC 全部精确等于标签;m2=±2 偏移变体,corr 56 MAC / FIR −14 抽头):

| 模型 | 基线 NMSE/ACLR | 最优 phase-spline | ΔACLR | ΔNMSE | 变体 |
|---|---|---|---|---|---|
| 2284 | −40.15 / −44.09 | **−40.11 / −44.45** | **+0.36** | 红线 ≤−40 守住 | ±2 |
| 998 | −37.37 / −41.21 | **−37.33 / −42.07** | **+0.86** | −0.04 | ±1 |
| 786 | −35.46 / −39.45 | **−35.33 / −40.49** | **+1.04** | −0.13 | ±1 |
| 796 | −35.20 / −38.99 | **−35.28 / −40.38** | **+1.39** | +0.08(更好) | ±2 |

±2 变体(FIR 43→29 截断损 0.26 dB NMSE)靠支路+联合训练完全回补并反超:
2284 从 ±1 的 −44.39 再进到 −44.45;796 从 −40.19 到 −40.38。
998/786 的 ±2 变体未跑(顺延后续,预计再 +0.1~0.3)。

### 最小点口径复评(2026-07-13 深夜)

最小点 ACLR = 邻带误差 PSD 最低 bin − 主带目标 PSD 最低 bin(分侧取较差,
nperseg=4096 与全部历史数字一致)。两种频带变体:

| 模型 | NMSE | 积分口径 | point(主(20,80)/邻±(120,180)) | point(主±100/邻±(100,180)) |
|---|---|---|---|---|
| dataset_error | −36.46 | −44.83 | −45.10 | −24.90 |
| 2284 base | −40.15 | −44.09 | −44.72 | −24.69 |
| **2284 spline** | **−40.11** | −44.45 | **−45.30 ✓** | −25.48 |
| 998 spline | −37.33 | −42.07 | −42.72 | −23.52 |
| 786 spline | −35.33 | −40.49 | −40.53 | −20.77 |
| 796 spline | −35.28 | −40.38 | −41.12 | −20.92 |

- **2284 phase-spline(±2)在旧带最小点口径下过线:−45.30 ≤ −45,NMSE −40.11 ≤ −40**,
  且低于数据集 error 的最小点(−45.10)。选型/训练都没针对 point 优化,这是"顺带"达标;
  若按 point 选型还有余量可挖。
- 参考带变体(主 ±100)是**退化口径**:主带最低 bin 落在 DC 凹口,参考值塌陷,
  全部数字掉到 −20 档,不具评估意义;若要在 ±100 主带下用最小点,参考应改用带均值或剔除 DC。

**结论:架构迁移有效且"免费"**——同 MAC、NMSE 不损(≤0.13 dB)拿到
0.3~1.2 dB ACLR,而纯子带微调路线要烧 2.4~4 dB NMSE 才换 0.5~0.8 dB。
相位归一化样条支路是目前唯一验证有效的"不牺牲 NMSE 的 ACLR 手段"。
−45 dBc 仍未达成(2284 差 0.61 dB,轻量组差 3~5 dB);2284 联合训练
在 −44.39 附近趋平。产物:`phase_spline/`(阶段1)、`phase_spline_all/`(联合)、
`phase_spline_m2/`(±2 偏移探测)各 <model>/best.pth + history.csv。
注意:wrapper checkpoint 键带 `base.` 前缀,须用本实验脚本(--checkpoint 续训)
或其加载逻辑读取,发布评估器不能直接加载。

### 最小点机制移植:spline 锚点 + min-point 机器(2026-07-13 深夜,steps/aclr_phase_spline_minpoint.py)

把 `aclr_min_point_optimization.py` 的整套机制(退火 soft-min 替代、NMSE 门控梯度手术、
信任域投影、硬指标验收+回滚、权重外推线搜索)逐位移植到 phase-spline wrapper 上,
锚点用本线各模型最优。与 `aclr_min_point_optimization/final_combined/summary.csv` 对比:

| 模型 | min-point 线(发布架构) | spline+minpoint(本线) | 胜者 |
|---|---|---|---|
| 2284 | −45.305 @ −40.06 | **−45.346 @ −40.11**(α=3.4 后平) | 本线 +0.04 |
| 998 | −42.660 @ −37.00 | **−43.015 @ −37.36** | 本线 +0.36 |
| 786 | **−41.522 @ −35.07** | −40.623 @ −35.45 | min-point 线 |
| 796 | −40.842 @ −35.19 | **−41.141 @ −35.30** | 本线 +0.30 |

四模型逐 checkpoint 最优(点口径、NMSE 全部可行):**2284 −45.35 / 998 −43.02 /
786 −41.52 / 796 −41.14**,较引用 summary.csv 改进 3/4(+0.04/+0.36/0/+0.30),
且三处获胜的 NMSE 余量更大。

负结果(同样重要):
1. **闭式 FIR 凹槽扫描无效**(786 上 −40.54 vs −40.53):FIR 只能消除底座输出与
   目标相干的误差分量,Welch 最低点已贴非相干地板,任意频点都雕不动——
   min-point 的深度由"底座-目标相干性"决定,只有动底座权重才有用。
2. 放大步长的重锚定第二轮(lr×4、trust×4)全部被硬指标回滚拒绝:机制已收敛。
3. 786 抢救(min-point 线 best 当底座→截断+积分重拟合→样条→minpoint)只到 −40.53:
   截断+积分 LS 重拟合会抹掉原 point 调优,积分口径选型的锚点对 786 的下边带
   point 持续不友好;786 的最优仍是原 min-point 线结果。
产物:`spline_minpoint/`、`spline_minpoint_r2/`、`phase_spline_786mp/`、
`spline_minpoint_786mp/` 各 <model>/(best_extrapolated.pth / history.csv /
extrapolation.csv / summary.json;2284/786 另有 best_extended.pth,α 扫到 8)。

### 更多思路轮(2026-07-14 凌晨):四模型全面刷新

尝试并验证的新手段(负结果同样记录):
1. **预算燃烧模式**(aclr_phase_spline_minpoint.py --budget-burn:NMSE 低于限−reserve 时
   去掉 NMSE 下降项、纯投影谱梯度推进)→ **无效**(998 +0.01、796/2284 不动),
   证明所有模型 point 卡的是相干地板而非 NMSE 预算;
2. min-point 线自身重锚定+放大步长+宽外推 → 786 仅 +0.05(−41.568),该线已收敛;
3. **wrap+截断+LS 重拟合的"意外馈赠"**:把 min-point 线 786 best 包进 spline wrapper
   (FIR 41→33 重拟合)后,point 从 −41.522 直接变 −41.740——FIR 变体会大幅搬动
   Welch 最低点位置,这成为 786 的新血统起点;
4. **--select-metric point**(phase-spline 阶段按 point 选型)→ 守住起点不被 HPF
   训练带偏(786),998 ±2 变体阶段直接选出 **−43.40**;
5. **998 ±2 变体补跑**(此前缺口,point 选型)→ minpoint → 扩展 α=3.5 → **−43.53**;
6. **FIR point 扫参**(steps/aclr_wrapper_fir_pointsweep.py,缓存底座+Toeplitz,
   按 point 选型)→ 四模型再收 +0.00~+0.08,便宜且稳定。

**最终四模型(最小点口径,val,独立复核,MAC 精确达标)**:

| 模型 | 引用 csv(min-point 线) | **最终 best** | 累计改进 | checkpoint |
|---|---|---|---|---|
| 2284 | −45.305 @ −40.06 | **−45.423 @ −40.17** | +0.12 | fir_pointsweep/2284/best.pth |
| 998 | −42.660 @ −37.00 | **−43.574 @ −37.09** | **+0.91** | fir_pointsweep/998/best.pth |
| 786 | −41.522 @ −35.07 | **−41.745 @ −35.18** | +0.22 | fir_pointsweep/786/best.pth |
| 796 | −40.842 @ −35.19 | **−41.189 @ −35.34** | +0.35 | fir_pointsweep/796/best.pth |

998 的 +0.91 路径:release → ±2 相位样条(point 选型,−43.40)→ minpoint 机器 →
α=3.5 外推(−43.53)→ FIR point 扫参(−43.57)。
2284 按最小点口径继续双达标(−45.42 ≤ −45,NMSE −40.17 ≤ −40)。

### 再加码轮(2026-07-14):收敛确认

新手段与结果:
1. **corr-scope soft-min 抛光**(tables-only,lr 1e-3)→ 无效:soft-min 对表梯度
   过于尖锐,NMSE 直接击穿 guard,全部 epoch 被回滚拒绝(spline_minpoint_corrpolish/);
2. **786 ±2 变体** → 净亏:FIR 41→27 截断让 point 起点掉到 −41.50(±1 起点 −41.74),
   全流水线只回到 −41.55(phase_spline_786m2pt/、spline_minpoint_786m2pt*/)。
   786 容量下"样条特征换 FIR 抽头"在 point 口径上不划算;
3. **随机画像 FIR point 扫参**(--random-profiles 200,分段随机邻带权重)→
   四模型再收 +0.01~0.06(fir_pointsweep_rand/)。

**最终最优(最小点口径,val,独立复核,MAC 精确;checkpoint 在
fir_pointsweep_rand/<model>/best.pth)**:

| 模型 | 引用 csv | 最终 | 累计 | NMSE | 同 ckpt 积分口径 |
|---|---|---|---|---|---|
| 2284 | −45.305 | **−45.445** | +0.14 | −40.15 ✓ | −44.37 |
| 998 | −42.660 | **−43.637** | **+0.98** | −37.38 ✓ | −42.05 |
| 786 | −41.522 | **−41.751** | +0.23 | −35.18 ✓ | −39.64 |
| 796 | −40.842 | **−41.230** | +0.39 | −35.32 ✓ | −40.35 |

三类独立优化(梯度机器/权重空间线搜索/FIR 重参数化)增益均已 <0.1 dB:
各容量的相干地板已摸清,微调范式到头。再往下只剩从头带 point 目标的完整训练轨迹。

### 凸子问题轮:活跃 bin 表优化 + lag 拓扑搜索(2026-07-14,上下文压缩期)

关键洞察:**底座与 FIR 冻结时,wrapper 输出对样条表逐项线性**——最小点目标
(时域 MSE + 两个固定活跃 bin 的 Welch 功率 + L2 信任项)是凸二次,可全批
L-BFGS 精确求解,再在验证集上做表更新方向的精确线扫,硬 NMSE 验收。

1. **活跃 bin 表优化**(steps/aclr_active_bin_table_optimization.py,
   active_bin_{smoke,probe→tables_*,alt_round,final}/):在训练段选多组分离的
   低功率候选 bin 对,逐对凸解+验证选优,多轮交替 → 四模型大幅刷新:
   2284 −45.788 / 998 −43.665 / 786 −42.073 / 796 −41.283;
2. **lag 拓扑搜索**(steps/aclr_lag_topology_search.py,lag_topology_search{,_r2}/):
   同 MAC 下按行语义把样条表重映射到备选 lag 集(共享 lag 复制、新 lag 零初始化),
   冻结底座凸拟合表 → 786 选出 **(−1,0,3)** 拓扑(−42.182),796 选出
   **(−2,−1,0,2,3)**(−41.352);2284/998 维持 ±2(无更优拓扑);
3. FIR 随机画像扫参在新锚点上全部持平(active_bin_fir_rand/、lag_topology_fir_rand/)
   ——FIR 维度已彻底榨干;
4. grid48(样条网格 24→48 免费加密)与 nperseg 稳健性探测(active_bin_grid48_*/、
   incumbent_robust_*/)未产生超越者。

### 从零渐进轨迹轮(2026-07-14,scratch_*):判负

--init scratch(LS 一阶增益 + 残差/out_conv 零初始化 + FIR 冲激重拟合)+
r→0.9999 渐进调度 + guard 约束 + point 选型,600~800 epoch 完整轨迹:

| 模型 | 从零终点 | 微调血统冠军 | 差距 |
|---|---|---|---|
| 998 | NMSE 卡 −36.63(不达 −37,判负);该点 point −41.73 | −43.665 @ −37.48 | ~2 dB 且不可行 |
| 786 | −40.224 @ −35.044(minpoint 后) | −42.182 @ −35.22 | −1.96 |
| 796 | −40.391 @ −35.219(minpoint 后) | −41.352 @ −35.38 | −0.96 |

**结论:参考队的"从零渐进"红利不可移植**——我们的 EA-TCN(+样条)从随机初始化
在这些预算内训不到微调血统的质量(发布模型本身来自 FIR95 底座/蒸馏等多阶段管线);
逐样本 KAN 能 500 epoch 从零到 −40 NMSE 是其架构特有性质。微调范式对本架构严格占优。

## 最终冠军(最小点口径,val,独立复核 2026-07-14)

| 模型 | 引用 csv | **最终** | 累计改进 | NMSE | MAC | point@2048/8192 | checkpoint |
|---|---|---|---|---|---|---|---|
| 2284 | −45.305 | **−45.983** | **+0.68** | −40.104 ✓ | 2284 ✓ | −44.37 / −46.62 | active_bin_final2/2284/best.pth |
| 998 | −42.660 | **−43.696** | **+1.04** | −37.501 ✓ | 998 ✓ | −42.03 / −43.67 | active_bin_final2/998/best.pth |
| 786 | −41.522 | **−42.266** | **+0.74** | −35.296 ✓ | 786 ✓ | −40.40 / −41.69 | active_bin_final2/786/best.pth |
| 796 | −40.842 | **−41.484** | **+0.64** | −35.422 ✓ | 796 ✓ | −40.00 / −41.24 | active_bin_final2/796/best.pth |

- 四模型全面超越引用实验线;2284 最小点口径双达标且余量 0.98 dB。
- 冠军由"梯度机器 minpoint → 权重外推 → 凸活跃-bin 重拟合"交替产生;最后半程
  (final_alt_minpoint → active_bin_final2)再收 +0.13~0.20 dB 后各手段增益均 <0.1 dB。
- nperseg 稳健性:2048(更多平均)下所有模型一致浅 ~1.4-1.8 dB(估计器性质,
  非过拟合;官方口径 4096);8192 下保持或更好。
- 加载须知:wrapper 键带 `base.` 前缀;786/796 冠军用非对称 lag((−1,0,3) /
  (−2,−1,0,2,3),见各目录 model_config.json / offsets.json),须以对应
  offsets/taps-reduction 构造。

### 分位数物理范围继续刷新(2026-07-14,本轮最终)

在 `active_bin_final2` 之后继续测试每个样条特征的独立物理范围。原旋相特征统一使用
`[-amp_max,amp_max]`，部分特征的大量 knot 从未被数据占用。新脚本
`steps/aclr_feature_range_search.py` 在物理坐标上重采样旧分段线性函数，保持 grid=24、
offset 数、FIR 和 MAC 不变，再用完整 train active-bin 二次目标拟合表项。

- 2284/998 的 quantile/scale 全部不胜 `active_bin_final2`；998 仅在其后 FIR 随机画像
  再收 0.004 dB。
- 786 的同 MAC lag `[-1,0,3]` + train quantile `0.014:0.986` 把 point 推到 −42.302；
  该收益在 2048/4096/8192 三种 Welch 长度下均成立。
- 796 保留原 `[-2,-1,0,1,2]`，quantile `0.007:0.993` 得到 4096-point 极值
  −41.870；但 8192 指标退化，属于官方口径专用极值。先前非对称 lag 796 不再是冠军。
- G48、2284/998 lag、所有 scale、后续 table/FIR 坐标轮均已收敛或判负。

**逐模型最新 4096-point 冠军**(统一路径 `final_optimized/<model>/best.pth`)：

| 模型 | 用户指定 fir_pointsweep | 最新 point(L/U/W) | NMSE | 相对改善 | 积分 ACLR | offsets / range |
|---|---:|---:|---:|---:|---:|---|
| 2284 | −45.423 | −45.993/−45.983/**−45.983** | −40.104 ✓ | **+0.561** | −44.340 | ±2 / 原范围 |
| 998 | −43.574 | −43.700/−43.714/**−43.700** | −37.508 ✓ | **+0.126** | −42.054 | ±2 / 原范围 |
| 786 | −41.745 | −42.302/−42.552/**−42.302** | −35.276 ✓ | **+0.556** | −39.744 | `[-1,0,3]` / q=.014 |
| 796 | −41.189 | −41.870/−41.882/**−41.870** | −35.481 ✓ | **+0.682** | −40.240 | ±2 / q=.007 |

积分口径相对用户指定 checkpoint 的变化(正=改善)：2284 −0.030、998 −0.050、
786 +0.115、796 −0.104 dB。故这些是严格按官方 4096 minimum-point 选择的冠军，
不能冒充全邻带积分最优。活动 bin、拓扑、range、λ、alpha 均在同一 val 上选型，
上述数字应称为 **validation selection score**，不是独立 test 泛化结果。

跨 Welch 长度的 point 改善(相对用户指定 checkpoint，dB)：

| 模型 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|
| 2284 | −0.113 | +0.561 | +0.484 |
| 998 | −0.020 | +0.126 | +0.188 |
| 786 | **+0.609** | **+0.556** | **+0.597** |
| 796(point 极值) | +0.140 | **+0.682** | −0.554 |

786 是三种 Welch 长度和积分口径都改善的最稳健新冠军。若 796 要求谱形稳健而非只追
官方单点，使用 `final_balanced/796/best.pth`：NMSE −35.518、4096 point −41.283
(+0.095)、积分 −40.371(+0.027)，2048 +0.235、8192 −0.025 dB。

实现护栏：wrapper checkpoint 必须使用同目录 `offsets.json`/`model_config.json` 中的真实
offsets 重建并核验 MAC。feature-range 中间产物只有 `ranges.json`，链式续跑时必须显式传
`--offsets`；`lag_topology_search.py` 的 `--source-offsets` 也必须与 sidecar 人工核对。
`steps/aclr_multires_active_bin.py` 因环境写入额度中断，没有 CLI 入口，**不可运行且本表未使用**。

## 产物

- 脚本(全部新建,未动任何旧文件/发布 checkpoint):
  `steps/aclr_ref_metrics.py`(口径+基线 CLI)、`steps/aclr_ref_fir_sweep.py`、
  `steps/aclr_ref_subband_finetune.py`、`steps/aclr_active_bin_table_optimization.py`、
  `steps/aclr_lag_topology_search.py`、`steps/aclr_feature_range_search.py`
- 回归测试:`tests/test_aclr_metric.py`、`tests/test_active_bin_optimizer.py`
- 最终 point 冠军:`eval_outputs_direct/aclr_ref_convention/final_optimized/<model>/`
  (`best.pth` / `summary.json` / `offsets.json` / `search.csv` / `config.json`)
- 796 稳健备选:`eval_outputs_direct/aclr_ref_convention/final_balanced/796/`
- 其余完整实验输出:`eval_outputs_direct/aclr_ref_convention/{baseline,fir_sweep,subband_probe,subband_v2,diag_*,active_bin_*,lag_*,feature_range_*}/`
- 参考报告全文文本:`other/refdesign_main_pdf.txt`;PSD 图:`other/refpdf_page25.png`

## 新预算轮 + 谱形优先标准(2026-07-14)

预算调整:2284 ≤ 2399、786 ≤ 800、998/796 锁定(≤1000/≤800 档位自动满足)。
评价标准修正:**整体谱形优先**(残差 PSD 须贴近数据集参考 error 形状,禁止人工凹点刷分),
最低点作为自然合格数;每次更新用 steps/aclr_psd_shape_eval.py 出图+量化目检
(邻带平均|Δ|dB、凹槽深度 vs 参考自然值 3.10 dB)。

### 形状优先冠军(val,MAC 达标,NMSE 达标,PSD 已目检)

| 模型 | MAC/限 | NMSE | 积分 ACLR | point(自然) | 邻带偏差 | 凹槽(参考3.10) | checkpoint |
|---|---|---|---|---|---|---|---|
| 2284 | 2388/2399 | −40.27 | **−44.74** | **−45.32 ✓** | 0.96 | 3.45 | shape_2284/2284/best.pth |
| 998 | 998/998 | −37.30 | −42.24 | −43.60 | 2.61 | 3.93 | shape_998/998/best.pth |
| 786 | 798/800 | −35.45 | −40.56 | −40.81 | 4.15 | **2.98(优于参考)** | shape_786/786/best.pth |
| 796 | 796/796 | −35.42 | −40.47 | −41.12 | 4.28 | 3.12 | shape_796/796/best.pth |

2284 的 +115 MAC 花在 ±4 九抽头样条支路 + 满 43 抽头 FIR(移植自雕刻冠军底座,
HPF 平滑续训 380 epoch);轻量组邻带偏差主要是容量水平差(高 2.6~4.3 dB),形状本身自然。

### 数字优先备选(active-bin 凸雕刻,PSD 有超自然凹槽,按需选用)

| 模型 | point | NMSE | 凹槽 | checkpoint |
|---|---|---|---|---|
| 2284 | **−46.16** | −40.14 | 4.55 | budget2399_ab2/2284/best.pth |
| 998 | **−43.85** | −37.50 | 4.37 | resqueeze_998/998/best.pth |
| 786 | **−42.36** | −35.30 | 5.12 | budget800_ab2/786/best.pth |
| 796 | **−41.72** | −35.31 | 4.47 | resqueeze_796/796/best.pth |

负结果:786 的 +14 MAC(第 4 lag 或 FIR 33→35)对 point 均无增益(容量瓶颈不在此);
796/998 预算锁死无结构可加。PSD 图:psd_shape/final_<model>.png。

## 两条实验线全量审查 + shape/point Pareto(2026-07-14)

按新预算上限(2284≤2399、998≤998、786≤800、796≤796)统一重载
`eval_outputs_direct/aclr_ref_convention/*/<model>/best*.pth`。共审查 246 个 checkpoint，
全部成功重建并重算，失败 0。完整逐 checkpoint 数值见 `shape_audit_full/audit.csv`，
两条线冠军与来源见 `shape_audit_full/champions.json`，四张统一 PSD 对照图为
`shape_audit_full/<model>_psd_comparison.png`。

谱形分数不是最低点的替代品：先对 4096-Welch PSD 做线性功率平滑，再分别计算邻带
`[-180,-100]∪[100,180] MHz` 和全带 `[-180,180] MHz` 的去电平 RMSE，二者各占 50%。
同时保留绝对邻带偏差、相关系数、积分 ACLR、2048/8192 point 和凹槽深度。shape 线必须
满足 NMSE/MAC、4096 point 不差于原 `fir_pointsweep`、积分 ACLR 不差超过 0.05 dB，
然后才按联合 shape 分数选最优。因此最低点仍是硬合格数，不会以“图好看”为由放弃数字。

### 数字线全局冠军(246 checkpoint 中最低 4096 point)

| 模型 | MAC/限 | NMSE | point 2048/4096/8192 | 积分 ACLR | shape 分数 | 凹槽 | 来源 |
|---|---:|---:|---:|---:|---:|---:|---|
| 2284 | 2388/2399 | −40.142 | −44.522/**−46.160**/−46.716 | −44.388 | 1.380 | 4.55 | `budget2399_ab2` |
| 998 | 998/998 | −37.498 | −42.090/**−43.846**/−43.997 | −42.017 | 1.251 | 4.37 | `resqueeze_998` |
| 786 | 798/800 | −35.296 | −40.475/**−42.364**/−42.019 | −39.760 | 1.280 | 5.12 | `budget800_ab2` |
| 796 | 796/796 | −35.481 | −40.209/**−41.870**/−41.234 | −40.240 | 1.218 | 4.19 | `feature_range_796_fir_r3` |

### 谱形优先且最低点合格的推荐线

| 模型 | MAC/限 | NMSE | point 2048/4096/8192 | 积分 ACLR | shape 分数 | 凹槽 | 相对原 fir_pointsweep 的 4096 改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2284 | 2388/2399 | −40.292 | −44.679/**−45.441**/−46.020 | **−44.745** | **1.206** | 3.56 | +0.018 |
| 998 | 998/998 | −37.107 | −42.164/**−43.645**/−43.943 | **−42.210** | **1.178** | 4.02 | +0.071 |
| 786 | 798/800 | −35.593 | −41.035/**−41.787**/−41.758 | **−40.311** | **1.209** | 4.61 | +0.041 |
| 796 | 796/796 | −35.519 | −40.146/**−41.278**/−42.167 | **−40.361** | **1.215** | 3.51 | +0.089 |

相对数字线，推荐线牺牲 0.20~0.72 dB 的 4096 单点，换来积分 ACLR 改善
0.12~0.55 dB、更低的联合 shape 分数和更浅的凹槽；四张原始/平滑 PSD 均已逐张目检。
推荐统一 checkpoint 在 `shape_audit_full/shape/<model>/best.pth`，纯数字线保存在
`shape_audit_full/point/<model>/best.pth`，每个目录均含 `offsets.json` 和
`audit_summary.json`。底层 Pareto 扫描、checkpoint 与逐候选图在
`shape_point_blend/<model>/`。

Pareto 细节：2284 只需 correction head α=0.12；998 只混合 FIR α=0.44，并把实际
NMSE 门槛收紧到 −37.10 留出 0.107 dB 裕量；796 的最优接近原同拓扑 active-bin 端点。
786 在严格 `shape+0.03/notch+0.50 dB` 护栏下无可行解(完整负结果保留在
`shape_point_blend/786/scan.csv`)，恢复原 point 合格线至少需要 full-state α=0.58；
该点仍把数字线凹槽从 5.12 降至 4.61、积分从 −39.760 改到 −40.311。

新增脚本：`steps/aclr_psd_shape_audit.py`(全量审查、统一绘图、两线选型)和
`steps/aclr_shape_point_blend.py`(同拓扑四种插值模式的 Pareto 搜索)。所有上述结果仍是
同一 validation split 上的 selection score，不冒充独立 test 泛化结果。

## 固定架构约束多分辨率优化（2026-07-16）

完成并修复此前未带 CLI 的 `steps/aclr_multires_active_bin.py`。本轮不改部署拓扑：
EA-TCN base 与两级 FIR 全部冻结，只重新拟合现有 phase-spline LUT；第三组端点实验再把
现有 `corr.gmin/gmax` 常量加入训练，但查表网格、offset、FIR 和 MAC 始终不变。训练系数
只使用完整 train split，validation 仅做硬 NMSE/point 护栏和 checkpoint 选型。

每个正式/负实验的模型目录均强制保存并检查非空：`best.pth`、`search.csv`、
`nmse_history.csv`、`nmse_convergence.png`、`psd_comparison.png`（4096 原始 Welch +
线性功率平滑两层）、`summary.json`、`offsets.json` 和 `artifacts_manifest.json`。

### 三条新路线

1. **多分辨率活动点**：同时拟合 2048/4096/8192 Welch 的多组低功率候选 bin。
   2284/998/786/796 的积分 ACLR 增益依次为 +0.010/+0.023/+0.063/+0.004 dB；
   998/786 伴随约 0.09~0.10 dB point 回退，796 增益为噪声级。2284 的温和候选同时
   改善 4096 point 且 shape 增量仅 +0.022，因此进入最终线。
2. **完整邻带多分辨率约束**：不选单 bin，直接最小化三种分辨率下左右完整邻带的归一化
   功率。宽松 Pareto 的积分增益为 +0.042/+0.057/+0.101/+0.062 dB；随后用原
   `fir_pointsweep` point 合格线和 shape/notch 护栏重选。该路线产出 998/786/796
   三个最终候选。
3. **连续 LUT 物理端点 + 表值联合优化**：从上述新候选续跑。2284/998/786 虽还能降低
   积分邻带，但分别违反联合 shape 或原 point 合格线；796 只再收 +0.0007 dB。
   因而端点路线整体判负，完整产物保留在 `continuous_range_*`。

### 新的合格 ACLR 改善线（独立重建与 PSD 审计）

`steps/aclr_psd_shape_audit.py` 对本轮及旧 anchor 共 31 个 checkpoint 独立重建，成功
31/31。下表仍是同一 validation split 上的 selection score；`Δ积分`、`Δpoint`、
`Δshape` 均相对 2026-07-14 的 shape 推荐线，正的 ACLR/point 增量表示改善，负的
shape 增量表示谱形分数改善。

| 模型 | MAC/限 | NMSE | point 2048/4096/8192 | 积分 ACLR | Δ积分 | Δ4096 point | 对原 fir point 余量 | shape / Δshape | 凹槽 | 最终 checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2284 | 2388/2399 | **−40.360** | −44.676/**−45.482**/−46.193 | **−44.755** | +0.010 | +0.041 | +0.059 | 1.228 / +0.022 | 3.67 | `constrained_multires_final/2284/best.pth` |
| 998 | 998/998 | **−37.327** | −42.171/**−43.580**/−44.030 | **−42.261** | **+0.050** | −0.065 | +0.006 | 1.202 / +0.023 | 3.89 | `constrained_multires_final/998/best.pth` |
| 786 | **798/800** | **−35.705** | −41.052/**−41.755**/−41.896 | **−40.373** | **+0.061** | −0.031 | +0.010 | **1.203 / −0.006** | 4.54 | `constrained_multires_final/786/best.pth` |
| 796 | 796/796 | **−35.528** | −40.232/**−41.309**/−42.092 | **−40.423** | **+0.062** | +0.031 | +0.120 | **1.197 / −0.018** | 3.56 | `constrained_multires_final/796/best.pth` |

998/786 有意使用了旧 shape anchor 相对原 `fir_pointsweep` 留下的一部分 point 余量，
换取完整邻带积分改善；但最终 4096 point 仍分别比原基线好 0.006/0.010 dB，且二者
2048、8192 point 均优于旧 shape anchor。796 的 8192 point 回退 0.076 dB，仍在
0.10 dB 多分辨率护栏内，2048/4096、积分和 shape 同时改善。四张最终 PSD 与四张
NMSE 曲线已逐张目检；未出现新增的窄带人工深凹槽。

严格按“shape 分数最小”单目标选型时，2284/998 的旧 `shape_point_blend/best.pth`
仍是纯谱形冠军；本表对这两档的定位是**在预设 shape/notch 护栏内换取更低积分 ACLR**，
不是声称 shape 分数也刷新。786/796 则同时刷新本轮候选中的 shape 与积分 ACLR。

统一最终目录：`eval_outputs_direct/aclr_ref_convention/constrained_multires_final/<model>/`；
精确指标和来源见其 `champions.json`。完整独立审计在 `constrained_multires_audit/`，
全部宽松、严格与负结果保留在 `multires_integrated_r1/`、`multires_band_r1/`、
`multires_band_strict/`、`multires_band_qualified/`、`continuous_range_r1/` 和
`continuous_range_qualified/`。
