# XF 新风转轮 KAN 重训与双转轮稳态仿真实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use spml:ml-subagent-dev to implement this plan task-by-task.

**Goal:** 使用 `data/XF.xls` 建立新风转轮专用 KAN 集成模型，固定接入“XF 新风转轮 + 当前主转轮”的双转轮拓扑，并完成严格域内的稳态系统仿真与物理验收。

**Experiment directory:** `experiments/xf_rotor1_retrain/`

**Hypothesis:** 在无数据泄漏的未见几何组合测试下，10 输入/7 输出的 XF KAN 逐输出集成能够达到既定精度；配合统一名义流量、气液分相和三变量稳态求解后，双转轮系统可得到唯一、域内且质量/水量自洽的稳态结果。

**Validation scope:** 仅训练流水线集成任务运行一次 L0 + L1；完整实验另执行 5 候选 × 5 折、外层 `300×1070 mm` 测试和整机物理验收。模型指标不替代输入域、湿空气状态和水量守恒检查。

**Evaluation design:** 每 2,000 个优化步骤做一次完整验证，训练中内存模型和 checkpoint 重载共用同一评估核心；保存原始与部署后处理指标、独立进度和失败原因。外层测试在路由锁定前完全不可见。

**Architecture:** XF 使用独立的 10 输入/7 输出 profile；当前主转轮模型保持参数和预测不变。系统按既定级联拓扑计算，内部以 `R1 再生入口实际温度、R1 再生入口含湿量、R2 再生入口实际温度` 为三个未知量做有界多起点稳态求解。正式运行不提供 legacy、新旧模型切换、自动降级或回滚入口。

---

## 1. 已锁定契约

### 1.1 数据边界

- 数据源：`data/XF.xls`；工作表：`CollectRecordXF`；第一行为技术字段名，第二行为说明文字，不得作为样本。
- 当前只读共享方式计算的 SHA-256：`116ea18458bf831225765472440a21da8045cb8e9bfadd898f677b004ce026cb`。
- 文件当前被其他进程占用。执行前必须关闭占用进程，以普通独占读重新计算哈希；仍被占用或哈希变化时停止，不训练、不自动接受新哈希。
- 有效原始样本 57,186 条；只删除 ID `1407`，清洗后 57,185 条；不插值、不做 IQR 删除、不改写原始 XLS。
- 外层测试完整留出 `300×1070 mm` 的 4,750 条；开发集 52,435 条。该测试验证“未见厚度×直径组合”，不是全新厚度水平或全新直径水平外推。
- 固定包络：转速 8 rpm、再生角 90°、冷吹角 15°。

### 1.2 数据字段到模型标签

KAN 仅使用以下 10 个原始输入，不生成派生输入特征：

| XF 技术字段 | 规范模型输入标签 | 运行时字段 |
|---|---|---|
| `TextBox_Rotor_depth` | `转轮厚度(mm)` | `rotor_thickness` |
| `TextBox_Rotor_diameter` | `转轮直径(mm)` | `rotor_diameter` |
| `TextBox_Process_inlet_Temp`，并校验等于 `TextBox_Purge_Inlet_Temp` | `处理/冷吹共享入口温度(℃)` | `process_inlet_T` |
| `TextBox_Purge_Inlet_AbsHum` | `处理/冷吹共享入口含湿量(g/kg)` | `process_inlet_W` |
| `Label_Process_FV` | `处理风速(m/s)` | `airspeed` |
| `Label_Purge_FV` | `冷吹风速(m/s)` | `cool_blow_airspeed` |
| `TextBox_Rotation_Speed` | `转轮转速(rpm)` | `rpm` |
| `TextBox_React_Heater_Temp` | `再生入口温度(℃)` | `regen_T` |
| `Label_React_CMH` | `再生入口风量(m³/h)` | `regen_flow_m3h` |
| `Label_React_Inlet_AbsHum` | `再生入口含湿量(g/kg)` | `regen_inlet_W` |

KAN 直接输出固定为 7 项：

| XF 技术字段 | 规范模型输出标签 | 运行时字段 |
|---|---|---|
| `Label_Process_Outlet_Temp` | `处理出口温度` | `process_outlet_T` |
| `Label_Process_Outlet_AbsHum` | `处理出口含湿量` | `process_outlet_W` |
| `Label_Purge_Outlet_Temp` | `冷吹出口温度` | `cool_blow_outlet_T` |
| `Label_Purge_Outlet_AbsHum` | `冷吹出口含湿量` | `cool_blow_outlet_W` |
| `Label_React_Outlet_Temp` | `再生出口温度` | `regen_outlet_T` |
| `Label_React_Outlet_AbsHum` | `再生出口总含水量` | `regen_outlet_total_W` |
| `Label_PowerConsumption` | `转轮平均功率(kW)` | `power_kW` |

- 正式标签使用“冷吹出口含湿量”，不得继续把旧拼写“冷处出口含湿量”写入 XF 新产物。
- `Label_Process_Outlet_DP` 不训练；水面露点由处理出口含湿量和系统压力派生。
- `power` 是转轮本体稳态平均功率 kW，不是 kWh，也不含风机、加热器或盘管。
- `TextBox_Process_CMH`、`TextBox_Purge_CMH` 用于系统流量与守恒审计，不进入 KAN。
- `TextBox_OA_CMH`、`TextBox_OA_AbsHum` 是新风转轮上游再生子流审计字段；权威 KAN 输入仍为 `Label_React_CMH` 和 `Label_React_Inlet_AbsHum`。

### 1.3 流量、压力与拓扑

- 每份配置中的所有 `m³/h` 均解释为在该配置 `outdoor_air` 温度、含湿量以及 `psychrometrics.pressure_Pa` 下标定的体积流量。每次加载配置只计算一个室外新风基准干空气密度，并把它用于新风、回风、再生新风以及两级冷吹共五股独立体积流量；不得按各支路实际温湿度分别换算。
- 干空气质量流量统一为：
  `mdot_kg_da_s = Q_m3h × rho_da,OA / 3600`，其中
  `rho_da,OA = P_da,OA / (287.05 × (T_OA + 273.15))`，
  `P_da,OA = P_system × 0.622 / (0.622 + W_OA/1000)`。
- KAN 的 `regen_flow_m3h` 直接使用名义流量代数和；混风、加热负荷、冷负荷和凝结水率使用上述统一换算后的 kg dry-air/s。
- 配置只保留五个独立流量：总新风、回风、再生新风、R1 冷吹和 R2 冷吹。其余流量由拓扑唯一派生：`R1_process = OA_total - R1_cold_blow`，`R2_regen = R2_cold_blow`，`R2_process = R1_process + return_air - R2_cold_blow`，`R1_regen = R2_cold_blow + regen_outdoor_air + R1_cold_blow`。
- 新风转轮（R1）再生入口是三股混合：`R2 再生出口 + 再生新风 + R1 冷吹出口`。
- 主转轮（R2）冷吹出口经过 `Heater_Low` 后进入同一主转轮再生区；R2 再生流量严格等于 R2 冷吹流量。
- `config` 只保存独立工况、设备参数、设定值和数值求解参数，不保存求得的三个稳态状态。

### 1.4 为什么只联立三个量

- R1 再生入口实际温度和含湿量由三股混合及 `Heater_OA` 实际出口决定；R1 输出又参与该混合，因此二者回到自身输入。
- R2 再生入口实际温度由 `Heater_Low` 实际出口决定；加热器入口是 R2 自身冷吹出口，因此温度回到自身输入。
- 两台加热器只能升温：`T_out = max(T_in, T_set)`；入口高于设定值时旁通，不能用设定值替代实际出口温度。
- 当前主转轮 KAN 没有“再生入口含湿量”输入，所以 R2 再生入口含湿量由冷吹出口经纯显热加热直接传播，仅是迭代中的派生量，不是第四个独立未知量。若未来主转轮模型新增该输入，必须把求解维度升级为四维并重新审查本计划。
- 流量、几何、转速、外气/回风和设定值是外部边界；转轮出口、混合焓、负荷、露点和凝结水在三个未知量给定后均可单向计算，因此只需在每次残差评估中重算，不另列未知量。

---

## 2. Shared Scaffold

### Existing infra（复用，发现问题先报告）

- 数据读取与训练入口：`training/src/data_loader.py`、`training/train_kan.py`。
- KAN 构造：`training/src/kan_model.py` 与 `simulation/src/kan_model.py`；若修改必须保持字节一致。
- 逐输出集成基础：`training/ensemble_predict.py`。
- 转轮运行时与系统拓扑：`simulation/src/system/components/rotor.py`、`simulation/src/system/system.py`。
- CLI 与当前配置：`simulation/dual_rotor_sim/cli.py`、`simulation/configs/default_case.yaml`。
- 当前主转轮产物：`artifacts/kan_main_ensemble_best`；不重训、不改变模型权重或路由。

### Needs setup

- 实验代码、配置、测试和报告全部放在 `experiments/xf_rotor1_retrain/`。
- XF 候选的中间 checkpoint、OOF 预测和未选模型不得散落到正式 `artifacts/`。
- 正式 XF 产物唯一目录为 `artifacts/kan_xf_ensemble_best`，仅包含路由实际引用并用 57,185 条全数据重训的候选。
- `requirements.txt` 与 `simulation/pyproject.toml` 显式加入 `scipy>=1.10`，用于唯一的有界多变量求解实现；不得依赖 scikit-learn 的传递依赖。

---

## 3. 原配置审查结论与迁移目标

当前 `simulation/configs/default_case.yaml` 不可直接用于 XF 正式系统：

1. `outdoor_air.mdot_m3h=5040` 与 `system_params.oa_mdot_m3h=4800` 重复且矛盾；运行时实际覆盖前者。
2. 多处 kg/s 近似注释与数值不符；例如 8,000 m³/h 回风不是约 0.5 kg dry-air/s。
3. 未配置 R1 厚度/直径，现代码会隐藏地使用 `200×1220 mm`，超出 XF 直径范围；正式值必须为 `300×1070 mm`。
4. R1/R2 共用 `process_airspeed=3.5` 和 `regen_airspeed=3.3`；R1 的数据支持仅为处理 `1.2–1.6`、冷吹 `1.0–1.3 m/s`。
5. 现有名义流量令 R1 再生总流量为 `800+560+240=1600 m³/h`；1070 mm XF 联合支持只允许 `1111 m³/h`。
6. 当前系统曾得到约 `17.03℃` 的 R2 冷吹入口，而主转轮训练值固定为 `27℃`，不能作为工程有效输入。
7. CLI 按每股空气自己的实际 T/W 换算 kg/s，违反“每份配置统一按该配置室外新风状态标定”的流量口径。
8. 缺少显式系统压力、流量参考状态、稳态求解设置和严格域策略；`visualization.artifacts_dir` 又不被 `simulate` CLI 使用。
9. `rpm_range=[2,10]` 与 R1 固定 8 rpm、R2 固定 6 rpm 冲突；正式验收不得用该扫描。
10. `target_Tdp=-60℃` 只能作为结果差值目标，不得替代模型/系统 promotion 门槛。

迁移后的配置采用 `schema_version: 2`，拒绝未知键和隐藏默认值。核心结构固定为：

```yaml
schema_version: 2
psychrometrics:
  pressure_Pa: 101325
  dew_point_basis: water
flow_reference:
  basis: outdoor_air
outdoor_air: {T_degC: 35.0, W_g_per_kg: 27.0, flow_m3h: 5040}
return_air: {T_degC: 22.0, W_g_per_kg: 0.49, flow_m3h: 8000}
regen_outdoor_air: {T_degC: 35.0, W_g_per_kg: 27.0, flow_m3h: 560}
system_params:
  fan1_P_kW: 4.0
  fan2_P_kW: 18.5
  rotor1:
    thickness_mm: 300.0
    diameter_mm: 1070.0
    process_airspeed_m_s: 1.6
    cool_blow_airspeed_m_s: 1.3
    cool_blow_flow_m3h: 240
    regen_angle_deg: 90.0
    cool_blow_angle_deg: 15.0
  rotor2:
    thickness_mm: 200.0
    diameter_mm: 1220.0
    process_airspeed_m_s: 3.5
    cool_blow_airspeed_m_s: 3.3
    cool_blow_flow_m3h: 800
    regen_angle_deg: 22.5
    cool_blow_angle_deg: 22.5
control_setpoints:
  cc1_T_outlet_degC: 11.0
  cc2_T_outlet_degC: 15.0
  heater_low_T_degC: 140.0
  heater_oa_T_degC: 120.0
  rotor1_rpm: 8.0
  rotor2_rpm: 6.0
solver:
  grid_points_per_axis: 3
  max_nfev_per_start: 500
  temperature_residual_tolerance_degC: 1.0e-4
  humidity_residual_tolerance_g_per_kg: 1.0e-6
  temperature_root_merge_tolerance_degC: 1.0e-3
  humidity_root_merge_tolerance_g_per_kg: 1.0e-5
run_options:
  device: cpu
  require_promoted_artifacts: true
  domain_policy: strict
reporting:
  target_supply_dew_point_degC: -60.0
```

- `default_case.yaml` 是真实/当前工况，唯一总新风量采用用户确认的 `5040 m³/h`；派生得到 R1 处理 `4800`、R2 处理 `12000`、R2 再生 `800`、R1 再生 `1600 m³/h`。该工况即使被严格模型域检查拒绝，也不得暗改为域内数值或绕过检查继续预测。
- 另建 `simulation/configs/xf_validation_case.yaml` 作为刻意域内的验收工况：总新风 `3499`、回风 `3333`、再生新风 `145`、R1 冷吹 `166`、R2 冷吹 `800 m³/h`，从而派生 R1 处理 `3333`、R2 再生 `800`、R2 处理 `5866`、R1 再生 `1111 m³/h`。
- `return_air.T_degC=25.0` 只属于域内验收工况的校准初值。正式 XF 模型产生后，按任务 9 的唯一根校准规则求出使 `R2_CB_in=27℃` 的值，并把完整精度结果写入 `xf_validation_case.yaml`；不得在每次系统仿真中偷偷调节回风温度。
- 两份正式配置均不保留 `profile`、`allow_unpromoted`、`fallback`、`warn-only` 或自定义模型路径；不存在 legacy 加载或回滚入口。

---

## 4. 实施子任务

## Subtask 1: 可审计 XF 数据适配器

**Role:** 固定源文件、字段语义、清洗和分组边界，为所有候选提供唯一数据入口。

**Implementation:** 新建 `experiments/xf_rotor1_retrain/data.py` 和配置；复用现有数据读取能力，但显式支持技术表头后跳过一行说明文字、ID 删除和字段别名。

**Unit Tests:** `experiments/xf_rotor1_retrain/tests/test_data.py` 覆盖表头、57,186/57,185 计数、ID 1407、入口温度相等、数值/有限值、重复 ID、固定角度/转速、外层与内层组不交叉。

**Expected Conclusion:** `Implemented + data contract tests passing`。

### Steps

1. 先写测试，并运行 `\.venv\Scripts\python.exe -m pytest experiments/xf_rotor1_retrain/tests/test_data.py -v`，确认因适配器不存在而失败。
2. 实现第一行技术字段读取、第二行跳过、17 个模型字段强制数值转换；任何 NaN/Inf、缺列或重复 ID 直接失败。
3. 仅删除 ID 1407，并记录其 `TextBox_Purge_CMH=3395` 等异常证据；不得修正后保留。
4. 验证删除后 `Label_React_CMH = TextBox_OA_CMH + TextBox_Purge_CMH`；子流混湿偏差只报告，不改写权威再生入口字段。
5. 输出 `reports/dataset_contract.json`、`cleaning_report.json`、`input_domain.json`、`split_assignments.csv` 和源哈希；再次运行测试至通过。
6. 提交：`experiment: add audited XF dataset contract`。

## Subtask 2: 分组划分、共享评估器与部署后处理

**Role:** 防止数据泄漏，并让训练选择、外层验收和运行时采用相同物理语义。

**Implementation:** 新建 `experiments/xf_rotor1_retrain/evaluator.py`；与运行时物理函数通过固定金标向量做数值一致性测试。

**Unit Tests:** 覆盖 fold scaler 只拟合训练组、OOF ID 覆盖、checkpoint/内存双入口、原始/后处理指标、非有限值失败、露点范围和气液分相。

**Expected Conclusion:** `Implemented + evaluator tests passing`。

### Steps

1. 固定开发集 5 折验证组：
   - Fold 1：`200×1070`、`400×1070`、`400×965`
   - Fold 2：`300×550`、`300×700`、`400×650`
   - Fold 3：`200×965`、`400×550`
   - Fold 4：`200×650`、`300×965`
   - Fold 5：`200×550`、`200×700`、`300×650`
2. 写失败测试：每折 scaler 只能 `fit` 本折训练 ID；外层 4,750 ID 不得进入 scaler、早停、NRMSE 分母、路由或训练步数决定。
3. 每折 NRMSE 分母使用该折训练分区相应目标的物理尺度跨度；跨度为零直接失败。OOF 以 ID 连接，每候选必须恰好覆盖 52,435 个唯一 ID，无缺失、无重复。
4. 物理后处理顺序固定为：反目标变换 → 反缩放 → 非负投影 → 再生出口气液分相 → 派生水面露点 → 指标。
5. 再生出口保持原始总含水量 `W_total_raw`；气相 `W_vapor=min(W_total_nonnegative,W_sat(T,P))`；凝结水 `W_liquid=max(W_total_nonnegative-W_sat,0)`。混风只使用气相，凝结水单独排出并进入总水守恒。
6. 标签也按同一 T/P 做气液分相。再生含湿量路由指标采用相态向量误差：
   `RMSE_phase = sqrt(mean((Wv_pred-Wv_true)^2 + (Wl_pred-Wl_true)^2))`，再除以本折训练 `W_total` 跨度。
7. 路由顺序固定：先选择三个温度和功率，再将已锁定的对应温度与每个含湿量候选组合，按部署后处理指标选择三项含湿量。全部原始指标同时保存，不参与首要排序。
8. 水面饱和压：低于 0℃ 使用 Murphy-Koop 过冷水面公式，0℃以上使用 ASHRAE 液态水公式；覆盖至少 `-75–60℃`。当饱和压不低于系统总压时令 `W_sat=+inf`，不得沿用当前 Magnus 裁剪伪值。
9. 运行评估器测试至通过；提交：`experiment: add leak-free evaluator and phase split`。

## Subtask 3: 五候选 KAN 训练组件

**Role:** 在完全相同的数据契约下比较容量、损失和目标变换，不改变输入特征集合。

**Implementation:** 新建实验配置与训练组件，复用现有 KAN builder；核心代码不得导入测试/验证工具。

**Unit Tests:** 10→7 形状、MinMax `[-1,1]`、`log1p` 可逆、加权 MSE/Huber、固定种子、checkpoint 保存恢复。

**Expected Conclusion:** `Implemented + model/trainer unit tests passing`。

### Candidate matrix

| 候选 | hidden widths | grid | loss / transform | lr / weight decay |
|---|---|---:|---|---|
| `xf_v1_baseline` | `[16]` | 5 | MSE；无 log | `1e-3 / 0` |
| `xf_v2_balanced` | `[24]` | 6 | MSE；无 log | `5e-4 / 1e-4` |
| `xf_v3_capacity` | `[32]` | 7 | MSE；无 log | `5e-4 / 1e-4` |
| `xf_v4_weighted` | `[24]` | 6 | 加权 MSE；仅处理出口含湿量 `log1p` | `5e-4 / 1e-4` |
| `xf_v5_robust` | `[32,16]` | 6 | 加权 Huber，delta=1.0（缩放空间）；仅处理出口含湿量 `log1p` | `3e-4 / 5e-4` |

共同参数：spline order 3、batch 64、最大 300 epochs、grad clip 1.0、seed 42；三个温度权重 1.0、三个含湿量权重 2.0、功率权重 1.0。

### Steps

1. 写并运行候选配置与训练组件失败测试。
2. 实现五候选；禁止添加相对湿度、焓、温差、面积、停留时间或任何交叉派生输入。
3. 训练中每 2,000 steps 做完整验证；`patience_evaluations=10`，ReduceLROnPlateau 也只在完整验证后更新。
4. 保存 `best`/`last` checkpoint、optimizer/scheduler/scaler/step/seed；恢复后下一步的批次和学习率必须与连续训练一致。
5. 日志包含 loss、grad norm、lr、step time、samples/s、内存和验证进度；CPU 环境记录 `MFU=not_applicable`，不伪造数值。
6. 运行测试至通过；提交：`experiment: add XF KAN candidate family`。

## Subtask 4: 逐输出 OOF 路由核心

**Role:** 只使用开发集折外预测锁定 7 个输出的候选来源。

**Implementation:** 新建 `experiments/xf_rotor1_retrain/routing.py`，消费 Subtask 2 的共享评估器输出。

**Unit Tests:** fold 聚合、相态路由顺序、tie-break、ID 完整性、外层 ID 禁用、路由序列化/重载一致。

**Expected Conclusion:** `Implemented + routing tests passing`。

### Steps

1. 写失败测试，特别断言 52,435 OOF ID 覆盖和外层 ID 零访问。
2. 每项输出首要分数固定为：`mean(5-fold deployed NRMSE) + 0.25 × std(5-fold deployed NRMSE)`。
3. 分数差 `<1e-4` 时依次选 mean NRMSE 更小、std 更小、参数量更小、候选名词典序更小者，保证确定性。
4. 保存每候选每折原始/后处理 MAE、RMSE、NRMSE、R²、相态误差、投影和凝结统计。
5. 序列化 `routing.json` 后重载，对同一 OOF 输入逐位一致；运行测试至通过。
6. 提交：`experiment: add deterministic OOF output routing`。

## Subtask 5: XF 最终训练流水线 [INTEGRATION]

**Hypothesis:** 组装后的流水线能在真实 XF 子集上无泄漏地完成数据读取、训练、定期评估、checkpoint 恢复、物理后处理和路由输入生成。

**Components consumed:** Subtask 1–4 的 `data.py`、`evaluator.py`、训练组件和 `routing.py`。

**Implementation:** 新建 `experiments/xf_rotor1_retrain/train.py`，支持 `--smoke-test`、单候选/单折和完整矩阵模式。

**Integration Tests:** 真实 XF 小子集完成 data → scale → forward → loss → backward → step → save → reload → evaluate；恢复前后预测一致。

**Validation Pyramid:** L0 + L1，仅本任务运行一次。L1 限时约 5 分钟，不执行完整 25 次训练。

**Evaluation contract:** `validation_interval_steps=2000`、完整验证、内存/checkpoint 双入口、独立进度条和阶段开始/结束摘要；checkpoint 缺失/损坏、空验证集、聚合失败、非有限指标或长时间无进度均失败。

**Expected Conclusion:** L0 无静态阻断；L1 全链路 6 阶段通过、loss 有下降、无 NaN/Inf、checkpoint 可恢复，并记录实际 samples/s 与内存。

### Steps

1. 写端到端失败测试并确认失败原因是流水线尚未组装。
2. 组装命令行、固定路径、日志、checkpoint、恢复和 evaluator；核心训练代码不得导入 tests。
3. 运行集成测试至通过。
4. 运行 L0 静态检查；修正数据泄漏、路径、种子、checkpoint 和评估可观测性问题。
5. 运行 L1：`\.venv\Scripts\python.exe experiments\xf_rotor1_retrain\train.py --smoke-test --max-runtime-seconds 300`。
6. 把 L1 实际指标、吞吐、内存和六阶段状态写入 `reports/validation_pyramid.json`。
7. 提交：`experiment: integrate XF training pipeline`。

## Subtask 6: 完整 5×5 实验、外层验收与正式产物

**Role:** 运行实验并决定是否允许正式系统加载 XF；不得因耗时减少候选或折数。

**Implementation:** 使用已验证流水线；预计当前 CPU 环境约 5–7 小时。

**Tests:** 产物完整性、OOF 覆盖、outer 单次访问、路由重载、全数据重训和 manifest 哈希。

**Expected Conclusion:** `promoted=true + artifacts/kan_xf_ensemble_best complete`，或明确失败且正式系统不可启动。

### Steps

1. 执行 25 次候选×折训练，保存日志、best step、OOF 预测和所有 fold scaler；不得读取外层标签。
2. 按 Subtask 4 锁定路由。每候选最终步数固定为：
   `min(max_steps, max(2000, ceil_to_2000(1.10 × median(fold_best_step))))`。
3. 在 52,435 条开发集上按固定步数各训练一次候选；仅此后第一次读取 4,750 条外层数据并组装路由集成预测。
4. 外层必须同时满足：
   - 三个温度输出各 `MAE ≤ 1.5℃`；
   - 三个含湿量相态 NRMSE 与功率 NRMSE 各 `≤10%`；
   - 七个直接原始标签各 `R²≥0.90`；
   - 处理出口水面露点 `MAE≤2℃`；
   - 非负投影（不含预期的再生气液分相）每输出比例 `≤1%`，最大修正 `≤开发集跨度1%`；
   - 预测值全部有限；温度不超出开发集输出范围两端各扩展 5% 的边界；
   - 含凝结水的总水量残差 P95 `≤5%`、最大 `≤10%`。
5. 任一门槛失败，写 `promotion_status.json: promoted=false` 并停止；不得降低门槛、不得加载 legacy、不得自动降级或继续整机数值仿真。
6. 全部通过后，将外层并回 57,185 条；只对路由实际引用的候选按固定步数全数据重训，并写入 `artifacts/kan_xf_ensemble_best`。
7. 未选候选、fold 模型和开发集最终模型留在实验目录。正式 manifest 记录源/配置/代码/模型/scaler/路由哈希及原始/后处理外层指标。
8. 提交：`experiment: train and promote XF ensemble`（仅提交代码/配置/小型报告；大模型按仓库现有产物策略处理）。

## Subtask 7: XF 运行时契约、湿空气公式与气液状态

**Role:** 让正式系统正确加载 7 输出 XF，同时保持当前主转轮预测回归不变。

**Implementation:** 修改 `simulation/src/system/components/rotor.py`、`simulation/src/system/psychrometrics.py` 及相应测试。

**Unit Tests:** XF 10 输入/7 输出、main 现有契约、正式标签、露点金标、饱和分相、凝结水、产物缺失/未 promotion 失败。

**Expected Conclusion:** `XF runtime + main regression tests passing`。

### Steps

1. 先写失败测试；保留一组当前主转轮固定输入及切换前输出作为回归金标。
2. `RotorInputs` 增加必填 `regen_flow_m3h`、`regen_inlet_W`；系统对 R1 传真实值，对 main profile 传值但主模型元数据不读取它们。
3. XF profile 只要求 7 个直接输出；`process_outlet_Tdp` 在运行时由 `process_outlet_W` 与 `pressure_Pa` 派生，不伪造第八个模型输出。
4. `RotorOutputs` 同时保留再生 `total_W_raw`、气相 `W`、`condensate_g_per_kg_da`；系统结合 dry-air mdot 计算 `condensate_kg_s`。
5. 处理/冷吹预测若出现超饱和，正式运行直接物理失败；再生出口按已确认规则分相。温度保持模型预测值，报告说明当前只闭合水量，不宣称转轮内部相变能量守恒。
6. 删除 `legacy` profile、默认路由、兼容回退与 `allow_unpromoted`；固定只识别 `xf` 和 `main`，并要求两个正式 manifest 都为 promoted。
7. 运行 XF 测试及当前 main 金标回归；提交：`simulation: add strict XF rotor contract`。

## Subtask 8: 三变量有界稳态求解与系统拓扑重排

**Role:** 使用加热器实际出口调用两个 KAN，覆盖主动加热和旁通两种状态。

**Implementation:** 修改 `simulation/src/system/system.py`，新增独立稳态求解模块；加入 SciPy 直接依赖。

**Unit Tests:** 唯一根、零根、多根、边界根、加热/旁通组合、越域、非有限值、三股质量/水量守恒和迭代 trace。

**Expected Conclusion:** `unique in-domain steady state or explicit failure`。

### Residual contract

未知量 `x=[T_R1_regen, W_R1_regen, T_R2_regen]`。每次评估完整前向重算两个转轮、三股混合和两台加热器，残差为：

```text
r1 = T_R1_regen - max(T_Mix3, heater_oa_setpoint)
r2 = W_R1_regen - W_Mix3_gas
r3 = T_R2_regen - max(T_R2_cold_blow_out, heater_low_setpoint)
```

温度残差除以 10℃、含湿量残差除以 1 g/kg 后交给求解器。边界来自两个已加载模型的实际输入域交集，不在代码中另抄范围。

### Deterministic solve

1. 使用 `scipy.optimize.least_squares(method="trf", bounds=...)`；唯一算法，不设固定点或无界求根后备路径。
2. 从每个维度 low/mid/high 的笛卡尔积共 27 个确定性起点求解；另对 9×9×9 粗网格的局部残差极小点补充起点，去重后逐一求解。
3. 每起点 `max_nfev=500`。候选根必须有限、全输入严格域内，并同时满足温度残差绝对值 `≤1e-4℃`、含湿量残差 `≤1e-6 g/kg`。
4. 按温度 `1e-3℃`、含湿量 `1e-5 g/kg` 聚类。恰好一个根才成功；零根、不收敛或两个及以上不同根均失败。
5. 结果记录全部起点、收敛状态、nfev、根、残差、两台加热器 active/bypass 状态和每个模型输入距训练域边界的距离。
6. 求解完成后再执行一次无缓存完整前向计算；结果必须与根状态在上述容差内一致，防止返回最后一次迭代的混合状态。

### Topology order

处理侧保持：`OA → Fan1 → CC1 → R1处理/冷吹分流 → R1处理出+RA → Fan2 → R2冷吹/CC2分流 → R2处理出Supply`。

再生侧保持：`R2冷吹出 → Heater_Low → R2再生区 → 与RegenOA及R1冷吹出三股混合 → Heater_OA → R1再生区 → 排放`。

凝结水在 R1/R2 再生出口形成后从气相移除，不进入 Mixer；其质量流率单列并计入总水守恒。

### Steps

1. 写测试替身构造可控唯一/多根函数，确认失败测试覆盖所有分支。
2. 实现残差评估、严格域预检、多起点 least-squares、根聚类和 trace。
3. 将现有单向“先调用 KAN、后算加热/混合”重排为求解器内部完整拓扑评估。
4. 运行系统与求解测试；提交：`simulation: solve coupled rotor steady state`。

## Subtask 9: 配置 schema、CLI 与固定工况迁移

**Role:** 消除 YAML 隐藏默认值、重复流量和域外扫描，使配置与新拓扑一一对应。

**Implementation:** 修改 `simulation/dual_rotor_sim/cli.py`、`simulation/configs/default_case.yaml`，新增 `simulation/configs/xf_validation_case.yaml` 和配置解析测试。

**Unit Tests:** schema 必填/未知键、统一名义换算、流量恒等式、R1 联合支持、严格域失败、禁止 legacy/allow-unpromoted/profile/model-path。

**Expected Conclusion:** `one truthful current-condition case + one intentionally in-domain validation case`。

### Steps

1. 新增严格 schema 解析；正式配置缺键或多余键均报中文错误，不使用 `.get(..., hidden_default)`。
2. 删除重复 `system_params.oa_mdot_m3h`，以 `outdoor_air.flow_m3h` 为唯一 R1 总新风来源；总新风固定采用用户确认的 `5040 m³/h`。五个独立流量全部按该配置 `outdoor_air` 状态和系统压力计算出的同一干空气密度换算。
3. R1/R2 厚度、直径、处理风速和冷吹风速均独立必填；正式 R1 为 `300×1070 mm`。
4. 严格检查 XF 直径联合支持：

   | 直径 mm | 处理/冷吹流量 m³/h | 处理/冷吹风速 m/s | 后级/总再生流量 m³/h |
   |---:|---:|---:|---:|
   | 550 | 667 / 33 | 1.2 / 1.0 | 189 / 222 |
   | 650 | 1000 / 50 | 1.3 / 1.1 | 283 / 333 |
   | 700 | 1333 / 66 | 1.5 / 1.3 | 378 / 444 |
   | 965 | 2667 / 133 | 1.5 / 1.3 | 756 / 889 |
   | 1070 | 3333 / 166 | 1.6 / 1.3 | 945 / 1111 |

5. 正式模型生成后，以 `return_air.T_degC=25℃` 为起点，在 `[15,45]℃` 扫描 201 点；对 `R2_CB_in(T_RA)-27℃` 的每个变号区间用 Brent 求根。恰好一个根且残差 `≤0.01℃` 时，将完整精度值写入 `xf_validation_case.yaml`；无根/多根失败。真实 `default_case.yaml` 不做这项反向校准。
6. 正式仿真固定值容差：R2 冷吹/处理入口温度各 `±0.05℃`，R1/R2 rpm `±0.01`，固定角度 `±0.05°`；连续范围仍按模型元数据 min/max，R1 联合表必须精确匹配名义离散组合。
7. `default_case.yaml` 删除 `2–10 rpm` 和混合 heater 扫描；`-60℃` 只保留在 reporting。
8. `default_case.yaml` 保存真实当前数值；严格模式下若模型输入域不合格，应在 KAN 推理前明确失败且不得生成预测结果。`xf_validation_case.yaml` 用于域内成功验收，二者不得互相替代。
9. CLI 固定加载 `artifacts/kan_xf_ensemble_best` 与 `artifacts/kan_main_ensemble_best`；删除模型 profile、任意 artifact 覆盖和未 promotion 绕过参数。
10. 运行配置/CLI 测试；提交：`simulation: migrate strict v2 configuration`。

## Subtask 10: 整机 promotion、系统仿真与交付

**Role:** 将模型、求解器、CLI 和固定工况作为一个不可拆分的正式系统验收。

**Implementation:** 扩展 promotion manifest 和中文报告；运行全量 simulation tests 与两个固定配置检查。

**Tests:** 正式成功工况、域外预检失败工况、产物篡改、主转轮回归、能耗分类、凝结水和全局守恒。

**Expected Conclusion:** `formal system promoted and reproducible`，否则正式入口明确失败。

### Steps

1. 运行 `\.venv\Scripts\python.exe -m pytest simulation\tests\ -v`，要求全部通过且无意外 skip。
2. 重算完整性清单，绑定 XF 模型/scaler/routing、当前主转轮、求解器、psychrometrics、CLI、schema 与两个配置的哈希。
3. 运行域内验收工况：`\.venv\Scripts\dual-rotor-sim.exe simulate --config simulation\configs\xf_validation_case.yaml --json --quiet`；必须唯一收敛、全部输入域内且输出有限。
4. 运行真实当前工况 `default_case.yaml`；若其输入域外，必须在任何 KAN 推理前失败，并列出字段、值、允许域、绝对距离和相对跨度，不得自动切换到验收工况。
5. 正式结果报告包含：送风 T/W/水面露点及距 `-60℃` 的差值；三个稳态未知量；两台加热器模式；两个转轮完整输入/输出和域位置；转轮平均功率、风机电功率、冷负荷、热负荷；各节点质量/水量残差；再生凝结水率。
6. 能耗继续分为 `electrical_kW`、`cooling_kW`、`heating_kW`。没有 COP/加热效率时不得把冷/热负荷冒充整机电功率。
7. 全局干空气质量守恒达到数值容差 `1e-9 kg/s`；含凝结水总水量残差 P95/最大值沿用 `5%/10%` promotion 门槛。
8. promotion 仅在数据、外层模型、运行时、稳态求解和正式域内工况全部通过后为 true。无回滚选项；任何缺失、篡改、未通过或不收敛均使正式 CLI 报错停止。
9. 保存中文总结、JSON、固定图表和 manifest；回读所有交付文件并记录最终命令、退出码和哈希。
10. 提交：`simulation: promote XF dual-rotor system`。

---

## 5. 最终交付物与执行边界

- 本计划、实验配置、数据契约和清洗/分组报告。
- 25 次交叉验证日志、OOF 预测、路由选择及原始/部署后处理指标。
- 外层 `300×1070 mm` 报告、气液分相与水量守恒报告。
- `artifacts/kan_xf_ensemble_best`、现有主转轮绑定清单和最终 `promotion_status.json`。
- 真实 `default_case.yaml`、域内 `xf_validation_case.yaml`、稳态求解 trace、整机 JSON/PNG/中文总结。
- 不执行动态控制、参数优化、能耗最优搜索、主转轮重训或真实尺寸水平外推。
- 不删除旧 artifact 文件，但正式代码和配置中不存在任何 legacy 加载、回滚、自动降级或运行时切换入口。
- 完成后在新对话中使用 `spml:ml-subagent-dev` 按 Subtask 1–10 顺序执行；只有 Subtask 5 运行一次 Validation Pyramid。

---

## 6. 执行记录

### Subtask 1 Conclusion (code)

**Role:** 固定 `data/XF.xls` 的唯一数据入口、10 输入/7 输出契约、仅删除 ID 1407 的清洗规则、外层留出与五折分组。

**Result:** implemented

**Evidence:**

- TDD RED：适配器缺失时测试收集因 `ModuleNotFoundError` 失败；质量修复 RED：快照验证 API 缺失时测试收集失败。
- TDD GREEN：`20 passed`；规格复审 `compliant`；质量复审 `approved`。
- 源 SHA-256：`116ea18458bf831225765472440a21da8045cb8e9bfadd898f677b004ce026cb`。
- 样本数：57,186 → 57,185；outer/dev = 4,750 / 52,435。
- 报告快照 ID：`13e497845c3f411b99fa22382778590792d0ae25ceb8f14b8506d94facaf9fa9`；四份报告通过哈希、字节数、行数和 commit marker 回读。
- Files: `experiments/xf_rotor1_retrain/{config.py,data.py,tests/test_data.py,reports/*}`。

### Subtask 2 Conclusion (code)

**Role:** 建立无外层泄漏的固定五折数据边界、内存/checkpoint 共享评估核心、水面露点与再生出口气液分相。

**Result:** implemented

**Evidence:**

- TDD RED：初始缺少 `evaluator.py`；后续攻击测试暴露 outer/dev ID 交换、错折 OOF、runtime 水面公式不一致、失败事件缺失和零含湿量露点指标中断。
- TDD GREEN：`test_evaluator.py` 37 项通过；Subtask 1 + evaluator + runtime psychrometrics 组合 `74 passed`。
- 完整 simulation：`140 passed`；仅保留旧 artifact 的 sklearn 1.6.1→1.7.2 版本警告。
- 规格复审 `compliant`；质量复审 `approved`。
- evaluator 与 runtime 共用 Murphy–Koop/ASHRAE water-basis 公式；`Psat >= P` 时 `W_sat=+inf`。
- `0 g/kg` 的热力学露点保留 `-inf`；仅指标报告副本显式、可审计地使用 `-75℃` 下限。
- Files: `experiments/xf_rotor1_retrain/{evaluator.py,tests/test_evaluator.py}`、`simulation/src/system/psychrometrics.py`及相应回归测试。

### Subtask 3 Conclusion (code)

**Role:** 在锁定的 10 输入/7 输出契约下实现五个固定 KAN 候选、目标变换、损失、确定性训练与严格可续训 checkpoint。

**Result:** implemented

**Evidence:**

- TDD RED：初始因 `candidates` 模块缺失失败；后续攻击测试复现了非有限 checkpoint、trainer/scheduler 微小改进语义不一致、同 ID 训练数值漂移、早停状态丢失及 step/epoch/cursor/seed/cadence 矛盾。
- TDD GREEN：`test_trainer.py` 44 项通过；Subtask 1–3 聚焦回归 `101 passed`，无 warning/skip。
- 五候选矩阵精确锁定；统一复用 `build_kan`，不添加派生输入；train-only MinMax `[-1,1]`，仅处理出口含湿量可用 `log1p`。
- Checkpoint v2 包含 optimizer/scheduler/scaler/RNG/批次游标/早停及控制量，并用 SHA-256 绑定字段契约、ID 顺序和训练输入/目标数值；旧版、损坏、篡改或不一致状态明确拒绝。
- 规格复审 `compliant`；独立质量复审 `approved`；未运行 L0/L1。
- Files: `experiments/xf_rotor1_retrain/{candidates.py,trainer.py,tests/test_trainer.py}`。

### Subtask 4 Conclusion (code)

**Role:** 只使用权威开发集五折 OOF 预测，按固定顺序为 7 个输出选择候选模型并生成可回读的确定性路由。

**Result:** implemented

**Evidence:**

- TDD RED：初始因 `routing` 模块缺失失败；攻击测试后续复现了 OOF 行序依赖、`<1e-4` 死区、伪造指标/相态/投影、伪造权威 ID 与 FoldData 数值/scaler、旧 artifact 应用到改写预测、ID hash 未回算以及 build/apply ULP 误拒。
- TDD GREEN：`test_routing.py` 43 项通过；Subtask 1–4 聚焦回归 `144 passed`。
- 主分数固定为五折 deployed NRMSE 的 `mean + 0.25×population std`；分差严格 `<1e-4` 时才依次用 mean/std/实际参数量/名称确定 tie-break。
- 先锁定三个温度和功率，再锁定三个含湿量；再生含湿量必须使用已锁定再生温度经共享 evaluator 重算相态 NRMSE。
- 每次建路由实时核对固定 `XF.xls` bytes/hash 与已提交 report snapshot；用权威 `build_fold_data` 逐 ID 深比对物理/缩放数组、训练跨度和 scaler 状态，禁止伪造 NRMSE 分母或外层 ID。
- Artifact 绑定经 ID 稳定排序的 OOF 内容 hash，完整保存 raw/deployed/相态/投影/凝结/hybrid 审计，严格 JSON schema/self-hash/权威 ID 回算与路由交叉验证；无 pickle 或任意 artifact 入口。
- 全面规格复审 `compliant`；独立质量复审 `approved`；未运行 L0/L1。
- Files: `experiments/xf_rotor1_retrain/{routing.py,tests/test_routing.py}`。

### Subtask 5 Conclusion (code)

**Role:** 组装真实 XF 数据读取、训练、定期评估、checkpoint 恢复、物理后处理、OOF 路由输入生成的完整训练流水线，并运行一次 L0 + L1。

**Result:** implemented and validated

**Evidence:**

- 当前环境为 Python 3.14 / NumPy 2.3.5 / pandas 3.0.5；已兼容修复：
  - `test_data.py` 在 pandas 3 StringDtype 下先转 object 再注入坏值；
  - `trainer.py` 的 `_object_graph_equal` 忽略 PyTorch deepcopy 不保留的 `_warned_capturable_if_run_uncaptured` 运行时标记；
  - `_ExternalValidationArray.restore_if_changed` 使用 `ndarray.__setstate__` 原位恢复 NumPy 2.x 无法通过 strides 属性恢复的 dtype/shape/strides 攻击。
- 实验侧测试：`180 passed`。
- 仿真侧回归：`140 passed, 1 warning`。
- L1 smoke：`train.py --smoke-test --max-runtime-seconds 240` 成功，约 9s 完成；loss `0.3919 → 0.00026`；六阶段全部 `end`；checkpoint 内存/重载双入口预测一致。
- 产物：`reports/validation_pyramid.json`、`work/xf_smoke/`。
- Files: `experiments/xf_rotor1_retrain/{train.py,reports/validation_pyramid.json,work/xf_smoke/*}`，并修改 `tests/test_data.py`、`trainer.py`。

### Subtask 6 Status (not yet completed)

**Role:** 完整 5×5 训练、外层 300×1070 验收、全数据重训与正式 `artifacts/kan_xf_ensemble_best`。

**Status:** pending — 流水线已通过 smoke 与集成测试，但完整 25 次生产训练预计数小时，尚未在当前会话中执行完成。

### Subtask 6 Conclusion (failed)

**Result:** failed at outer 300×1070 validation

**Evidence:**

- 完整 5×5 训练成功：25/25 runs，耗时约 75 分钟，`work/xf_full_matrix/routing.json` 已生成。
- 选中候选：`xf_v1_baseline`、`xf_v2_balanced`、`xf_v5_robust`。
- 在 52,435 条 development 上重训选中候选，并首次读取 4,750 条 outer 数据评估。
- 多数指标优秀，但外层门槛未通过：
  - 处理出口水面露点 MAE = 3.82 ℃ > 2 ℃；
  - 处理出口含湿量非负投影比例最高 8.95% > 1%；
  - 最大修正相对开发集跨度最高 1.57% > 1%。
- 按计划不降低门槛、不生成正式 XF 产物、不进入整机仿真。
- 已写入 `experiments/xf_rotor1_retrain/reports/xf_promotion_status.json` 和 `artifacts/kan_xf_ensemble_best/reports/promotion_status.json`，均为 `promoted=false`。
