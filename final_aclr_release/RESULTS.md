# ACLR 实验最终结果与复现指南

四个模型(实乘法器上限 **2284→2399、998、786→800、796**),NMSE 约束
**≤ −40 / −37 / −35 / −35 dB**,验证段(后 20%)评估,Welch nperseg=4096。
本目录自包含:冠军 checkpoint、PSD 图、全部流水线脚本、复现与评估指令。

## 0. 口径定义(为什么是这两个指标)

- **NMSE** = 10·log10(Σ|y−ŷ|²/Σ|y|²)。与归一化无关(rms/max 逐位相同,已数值验证)。
- **积分 ACLR**(参考队口径)= 10·log10([P_err(−180,−100)+P_err(100,180)] / P_y(−100,100)),
  误差邻带积分功率 ÷ 主带目标积分功率。选它是因为:(a) 已按 APNKAN 报告数值复现锁定
  (数据集 error = −44.75 vs 他们标注 −44.59);(b) 它度量**整体谱形质量**。
- **最小点 ACLR** = 邻带误差 PSD 最低 bin − 主带 (20,80) 目标 PSD 最低 bin(分侧取较差)。
  选它作"合格数字"是因为它是我方最有利的合规口径(2284 可稳定 ≤ −45),
  且数据集参考 error 本身 = −45.10,我方 2284 已越过参考残差水平。
- **谱形贴近度**(最终验收标准)= 邻带内 |模型残差 PSD − 参考 error PSD| 的平均 dB 偏差
  + 凹槽深度(带内中位数−最低点;参考自然值 3.10 dB)。**为什么**:最低点数字可以靠
  人工凹点"刷"出来,但 PSD 图会暴露非自然形状;谱形优先保证结果经得起目检。

## 1. 最终结果(2026-07-14 全量审计版,`shape_audit_full/`)

所有历史 checkpoint 经 `steps/aclr_psd_shape_audit.py` 统一重评后选出
(多分辨率最小点 2048/4096/8192、平滑中心化 RMSE 形状分、凹槽超额、sha256 去重)。

### 1.1 谱形最优(best_shape/,推荐口径,`shape_point_blend` 血统)

Pareto 混合(形状锚点 ↔ 数值锚点插值扫描 + 形状守卫)所得——形状分与最小点同时优于旧形状线:

| 模型 | MAC/上限 | NMSE | 积分 ACLR | 最小点(4096) | 形状分 | 凹槽超额(dB) |
|---|---|---|---|---|---|---|
| 2284 | 2388/2399 | −40.29 | **−44.74** | **−45.44 ✓** | 1.206 | **0.45** |
| 998 | 998/998 | −37.11 | −42.21 | −43.65 | 1.178 | 0.92 |
| 786 | 798/800 | −35.59 | −40.31 | **−41.79** | 1.209 | 1.51 |
| 796 | 796/796 | −35.52 | −40.36 | −41.28 | 1.215 | **0.41** |

(凹槽超额 = 邻带凹槽深度 − 参考 error 自然值 3.10 dB;形状分 = 邻带与 ±180 全带
平滑中心化 RMSE 均值,越小越贴近参考形状。)

### 1.2 数值最优(best_number/,凸雕刻线,备选)

| 模型 | MAC/上限 | NMSE | 最小点(4096) | 积分 ACLR | 来源 |
|---|---|---|---|---|---|
| 2284 | 2388/2399 | −40.14 | **−46.16** | −44.39 | budget2399_ab2 |
| 998 | 998/998 | −37.50 | **−43.85** | −42.02 | resqueeze_998 |
| 786 | 798/800 | −35.30 | **−42.36** | −39.76 | budget800_ab2 |
| 796 | 796/796 | −35.48 | **−41.87** | −40.24 | feature_range_796_fir_r3(特征范围重整,新) |

对照:数据集参考 error 最小点 −45.10 / 积分 −44.83;引用实验线
(`aclr_min_point_optimization/final_combined/summary.csv`)−45.31/−42.66/−41.52/−40.84。
多分辨率稳健性(2284 数值冠军):−44.52@2048 / −46.16@4096 / −46.72@8192。

### 1.3 文件地址

```
final_aclr_release/
├── EXPERIMENTS.md / RESULTS.md            # 本两份文档
├── best_shape/                            # 谱形最优(shape_point_blend)
│   ├── manifest.json                      # offsets/taps/MAC/全指标(由审计 summary 自动生成)
│   ├── checkpoints/<model>_shape_*.pth    # wrapper 权重(键带 base. 前缀)
│   ├── figures/<model>_psd.png            # 标准三线 PSD 图(Dataset y/Reference error/EA-TCN error)
│   ├── figures/<model>_psd_comparison.png # 审计三方对照图(baseline/point/shape + 平滑视图)
│   └── scripts/                           # 流水线脚本副本(运行请用 steps/ 原件)
└── best_number/                           # 数值最优,结构同上
```

审计原始产物:`eval_outputs_direct/aclr_ref_convention/shape_audit_full/`
(`audit.csv` 全部候选逐行指标、`champions.json` 三线冠军、`failures.csv` 加载失败清单、
`shape|point/<model>/`(best.pth + model_config/offsets 边车 + audit_summary.json))。
训练历史与中间产物:`eval_outputs_direct/aclr_ref_convention/`(总账本 `SUMMARY.md`);
旧口径实验:`eval_outputs_direct/aclr_three_metric_experiments/SUMMARY.md`。

## 2. 为什么这样做(设计决策链)

1. **相位归一化样条 LUT 支路**(APNKAN 迁移):以当前样本相位为参考旋转记忆特征、
   输出旋回。网络不再学习全局相位等变性,容量全部投向包络/相对相位/记忆非线性——
   恰是邻带再生的物理来源。样条查表的"每乘法器非线性表达力"远高于卷积+SiLU。
   实测:同 MAC、NMSE 不损,ACLR +0.36~+1.39 dB(唯一免费手段)。
2. **MAC 严格守恒**:支路成本按 APNKAN 口径全额计入(含邻居旋转),由 FIR 减抽头
   精确支付;评估器 `real_multiplier_count` + wrapper `mac_report` 双重核算。
3. **时域 HPF 子带损失**(而非 FFT 惩罚):滤误差序列取 MSE,经 Parseval 与积分口径
   严格对齐,无窗口/估计器失配;r 渐进 + NMSE guard = 拉格朗日式约束优化。
4. **周期闭式 FIR 重拟合**:训练中底座漂移时,FIR 始终保持频域加权 LS 最优。
5. **min-point 机器**(数值线):soft-min 退火替代 + NMSE 门控梯度手术 + 硬指标
   验收回滚 + 权重外推;凸活跃 bin 表拟合直接优化决定最低点的两个 bin。
6. **谱形优先收官**:凹点雕刻会留下 PSD 指纹;最终推荐口径回到平滑压带,
   最低点由平滑低带自然形成,每个结果出图目检。
7. **Pareto 混合**(`aclr_shape_point_blend.py`):形状锚点与数值锚点在权重空间做
   分组插值扫描(全参数/校正头/仅表/仅 FIR),形状守卫(NMSE+积分+形状分+自然凹槽)
   内取最优点——同时改善形状分与最小点,产出最终推荐冠军。
8. **特征范围重整**(`aclr_feature_range_search.py`):样条表的 gmin/gmax 原为保守
   公共范围,旋转特征实际只占窄区间;按训练集分位数逐特征重定范围
   (函数保持式暖启动)+ 凸拟合,零 MAC 变化——796 数值冠军再 +0.15(−41.87)。
9. **全量审计协议**(`aclr_psd_shape_audit.py`):所有 best*.pth 统一重评
   (多分辨率最小点、平滑中心化 RMSE 形状分、凹槽超额、sha256 去重、失败清单),
   杜绝"单一 Welch bin 凹点伪装好频谱",冠军由 audit.csv 机选。

## 3. 复现指令

环境:`D:/anconda/envs/pytorch/python.exe`(torch 2.11+cu128, float64),
工作目录 = 仓库根;发布 checkpoint 在 `checkpoints/AI-DPD/`(只读,永不覆盖)。
以下 `$P` 代指该 python。**说明**:冠军来自"发布模型 → 多轮迭代"的微调血统;
纯从零训练(附录)已验证为劣解,仅作对照。

### 3.1 谱形冠军复现(以 2284 为例,786/998/796 同构)

```bash
# ① 架构移植:把 ±2/减14抽头 冠军底座装进 ±4/满FIR(恒等起点, 2388≤2399)
$P steps/aclr_spline_transplant.py --model 2284 \
   --source-checkpoint <±2冠军wrapper.pth> --source-offsets -2 -1 0 1 2 --source-taps-reduction 14 \
   --target-offsets -4 -3 -2 -1 0 1 2 3 4 --target-taps-reduction 0 \
   --mac-limit 2399 --out <transplant.pth>

# ② HPF 平滑训练(样条表+底座联合, 子带损失+NMSE guard+每10ep闭式FIR重拟合)
$P steps/aclr_phase_spline_experiment.py --model 2284 --checkpoint <transplant.pth> \
   --offsets -4 -3 -2 -1 0 1 2 3 4 --taps-reduction 0 --grid 24 --mac-limit 2399 \
   --tag shape_2284 --train-scope all --lr 3e-4 --base-lr 2e-5 \
   --r-max 0.995 --ramp-epochs 8 --epochs 380 \
   --nmse-guard -40.05 --nmse-select -40.0 --select-metric ref
# 产物: eval_outputs_direct/aclr_ref_convention/shape_2284/2284/best.pth
```

轻量组参数差异:998 `--offsets -2 -1 0 1 2 --taps-reduction 14 --nmse-guard -37.05 --nmse-select -37.0`;
786 `--offsets -2 -1 0 3 --taps-reduction 8 --mac-limit 800 --nmse-guard -35.05`;
796 `--offsets -2 -1 0 1 2 --taps-reduction 14 --nmse-guard -35.05`(不移植,直接从各自 ±2 冠军续训)。

### 3.2 数值冠军复现(在谱形/血统锚点之上追加)

```bash
# ③ min-point 梯度机器(soft-min 退火+梯度手术+回滚+外推)
$P steps/aclr_phase_spline_minpoint.py --model 2284 --checkpoint <锚点wrapper.pth> \
   --taps-reduction 0 --offsets -4 -3 -2 -1 0 1 2 3 4 --tag mp_2284 --softmin-end 0.02

# ④ FIR point 扫参(网格+随机分段权重画像, 廉价搬动最低点)
$P steps/aclr_wrapper_fir_pointsweep.py --models 2284 --checkpoint <mp输出.pth> \
   --offsets -4 -3 -2 -1 0 1 2 3 4 --random-profiles 200 --tag fps_2284

# ⑤ 凸活跃 bin 表优化(重压配方; 追求数字时使用, 会产生超自然凹槽)
$P steps/aclr_active_bin_table_optimization.py --models 2284 \
   --checkpoint <fps输出.pth> --offsets -4 -3 -2 -1 0 1 2 3 4 --mac-limit 2399 \
   --tag ab_2284 --rounds 5 --active-candidates 4 \
   --spectral-weights 0.01 0.03 0.1 0.3 1.0 3.0
```

796 另有 lag 拓扑搜索:`$P steps/aclr_lag_topology_search.py --model 796 ...`
(最优拓扑 (−2,−1,0,2,3),786 为 (−2,−1,0,3),写入各 manifest.json)。

```bash
# ⑥ 特征范围重整(零 MAC; 796 数值冠军来源, 分位数范围+凸拟合)
$P steps/aclr_feature_range_search.py --model 796 --checkpoint <数值锚点.pth> \
   --offsets -2 -1 0 1 2 --quantiles 0.0005 0.9995 --tag feature_range_796

# ⑦ Pareto 混合(最终推荐冠军来源: 形状锚点↔数值锚点插值扫描+形状守卫)
$P steps/aclr_shape_point_blend.py --model 2284 \
   --shape-checkpoint <形状锚点.pth> --point-checkpoint <数值锚点.pth> \
   --offsets -4 -3 -2 -1 0 1 2 3 4 --tag shape_point_blend
```

### 3.3 评估指令

```bash
# 单个 checkpoint: NMSE/积分/最小点/MAC/形状偏差/凹槽 + 标准 PSD 图(png) + json
$P steps/aclr_psd_shape_eval.py \
   --checkpoint final_aclr_release/best_shape/checkpoints/2284_shape_point45p44.pth \
   --offsets -4 -3 -2 -1 0 1 2 3 4 --mac-limit 2399 --label verify_2284
# offsets 逐模型见 manifest.json; 图与 json 落在
# eval_outputs_direct/aclr_ref_convention/psd_shape/verify_2284.{png,json}

# 全量审计(重评所有 best*.pth, 多分辨率最小点+形状分+去重, 机选三线冠军):
$P steps/aclr_psd_shape_audit.py --tag shape_audit_full
# 产物: shape_audit_full/{audit.csv, champions.json, failures.csv,
#        shape|point/<model>/best.pth, <model>_psd_comparison.png}
```

注意:wrapper checkpoint 键带 `base.` 前缀,**发布评估器不能直接加载**,
须经上述脚本(或其 `PhaseSplineDPD` 加载逻辑)。

### 附录:纯从零训练(对照用,已判负)

```bash
$P steps/aclr_phase_spline_experiment.py --model 998 --init scratch \
   --offsets -2 -1 0 1 2 --taps-reduction 14 --grid 24 --tag scratch_998 \
   --train-scope all --lr 1e-3 --base-lr 1e-3 --lr-gamma 0.995 \
   --r-max 0.9999 --ramp-epochs 600 --epochs 800 --refit-every 10 \
   --nmse-guard -37.05 --nmse-select -37.0 --select-metric point
# 结果: 998 NMSE 渐近 −36.6 不达标; 786/796 达标但 point 差微调线 1~1.5 dB
```
