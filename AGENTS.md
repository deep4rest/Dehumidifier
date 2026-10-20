# AGENTS.md

本文件是所有 Agent 的项目维护规则唯一来源；操作步骤见 `README.md`。用户的最新明确指令优先。项目中文优先，新增用户可见提示、日志和文档使用中文。

## 目录与职责

```text
training/
  single_kan/          当前唯一训练入口、数据契约、模型、评估器、测试与设计记录
  src/                 保留的 KAN/预处理/特征工程兼容模块
simulation/
  dual_rotor_sim/      可安装包：CLI、配置、运行记录、dashboard
    core/system/      元件、拓扑、求解器、湿空气公式
  configs/            当前 schema v2 工况 YAML
  tests/              仿真与兼容性回归测试
  visualize.py        绘图入口，同 CLI 一样按次留档
artifacts/            仅当前 kan_xf_single/、kan_main_single/ 运行时包
results/
  training/           每次训练的模型副本、checkpoint、日志、配置和审计报告
  simulation/         每次仿真的 result.json 与 fig1_dashboard.png
archive/
  training/           历史通用训练工具、配置和数据处理工具
  xf_rotor1_retrain/   历史 XF 候选/路由训练源码、测试、计划
  artifacts/          历史模型、集成结果与离线报告
data/                两份权威 XLS；templates/ 为企业表格；reference/ 为参考资料
docs/                reference/ 参考文档；maintenance/ 维护记录与迁移清单
```

所有 Python 代码共用根目录 `.venv/`。`需求.md` 是需求依据；当前 v2 拓扑以 `simulation/dual_rotor_sim/core/system/system.py` 模块文档和实现为准。`docs/reference/系统建模v1方案.md` 仅作历史参考。

## 纯源码 GitHub 移交

- 接手先读 `docs/handoff/README.md`，其中链接训练、拓扑和 ONNX/DLL 规范。首次建立环境先 `python3 -m venv .venv`。
- GitHub 仅发布源码、配置、必要说明和测试，不发布 XLS/XLSX、权重、scaler、checkpoint、训练/仿真结果、ONNX/DLL/NuGet/ZIP 成品或本机 Agent 设置。`scripts/prepare_source_upload.py` 使用白名单产生干净源码目录；旧 Git 历史已有模型/结果，禁止直接推送该历史。首次上传以新仓库 README 提交为父提交建立源码快照，不覆盖本地旧历史或删除本机产物。
- `.gitignore` 不能取消已经跟踪的历史文件；不能只修改 ignore 后执行 `git push --all`。`deployment/dotnet/src/Dehumidifier.Simulation/build/*.targets` 是 NuGet 构建源码，必须保留。
- `data/README.md` 保存数据路径、sheet 和 SHA-256；外部资料缺失时 `requires_data` / `requires_models` 测试明确跳过。恢复后用 `REQUIRE_EXTERNAL_FIXTURES=1` 强制执行外部资料测试，缺文件应失败。
- 本工作树当前输入为风速版；`deployment/dotnet/` 是从本机活跃部署源码恢复的历史风量版迁移基线，C# 与当前 Python 在输入、混风、湿空气公式和求解默认值上不一致。必须完成文档列出的同步并重新验证，才能宣称匹配 DLL；不得拿历史包验证结果作当前证据。
- `results/source_upload/` 只存放本次本机源码快照、上传清单和验证记录，不纳入 GitHub；训练/仿真继续遵守原目录规则。

## 固定工作流

1. **接手检查**：读取本文件和 README；查看 `git status --short`，保留已有未提交修改；检查工况、模型元数据和相关运行报告。不能把历史记录中的路径、模型状态或结果当作当前事实。
2. **修改与小样本验证**：当前训练只修改 `training/single_kan/` 及明确依赖。先运行相关单元测试，再按 README 用 `--smoke-test --max-steps 4` 进行有界验证。原始 XLS 只读，每次读取校验 SHA-256；不得改工况迎合模型。
3. **全量训练**：需要用户明确授权，入口必须带 `--confirm-full`。整理目录、修代码或“继续联调”不等于授权全量训练。默认全量训练完成后更新对应运行时包；失败或 smoke 不更新。
4. **仿真**：默认只读 `artifacts/kan_xf_single/` 与 `artifacts/kan_main_single/`，使用 `simulation/configs/` 的真实工况。缺少模型或元数据不合契约就报错，不换用历史模型。
5. **留档与复核**：每次仿真 JSON 和 PNG 必须同目录；绘图调整优先调用 `save_dashboard_from_record` 从已有 JSON 重绘。训练精度、数值可运行与工程物理验收分别报告；测试通过不代表工程验收。
6. **交付**：报告改动、测试、实际产物路径及尚未解决的限制。入口、路径、输出结构或操作步骤变化时，同步更新 README 和本文件。

## 命令

从仓库根目录运行：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e simulation/

# 当前回归；pytest.ini 默认只收集当前训练与仿真测试
.venv/bin/python -m pytest -q

# 有界 smoke：不会安装到运行时 artifacts
.venv/bin/python -m training.single_kan.train --profile all --smoke-test --max-steps 4 --validation-interval-steps 2 --checkpoint-interval-steps 2 --no-progress

# 仅用户明确授权全量训练后执行
.venv/bin/python -m training.single_kan.train --profile all --confirm-full

# 仿真 / 可视化，均写入 results/simulation/<运行编号>/
.venv/bin/dual-rotor-sim simulate --config simulation/configs/default_case.yaml
.venv/bin/python simulation/visualize.py --config simulation/configs/qy1070_case.yaml

# 仅修改历史兼容代码时补充此回归
.venv/bin/python -m pytest archive/xf_rotor1_retrain/tests -q
```

也支持直接执行 `training/single_kan/train.py`；旧 `experiments/...` 入口已迁移，不保留重复入口。

## 输出约定：不得再次分散存放

- **训练默认路径**：`results/training/<时间>_<smoke或full>_<profile>_<唯一后缀>/`。`pipeline_report.json` 记录 running/completed/failed、每台转轮结果及失败原因；`checkpoints/<profile>/` 保存续训状态；`kan_<profile>_single/` 保存该次模型、日志、配置和 `reports/`。
- **运行时路径**：`artifacts/` 仅放 `kan_xf_single/` 与 `kan_main_single/`（允许说明文件）。模型文件位于各包 `model/`；包根有 `model_metadata.json`、`config_snapshot.yaml`、`training_run.json`。后者指向完整训练记录，仿真不依赖它作为门禁。
- **默认全量运行**：全部请求 profile 完成后才开始更新运行时包，旧包保存在本次运行的 `previous_artifacts/`。这只是保存历史，不提供运行时自动回退。更新写入失败需如实检查两个包状态，不能宣称整组更新成功。
- **显式 `--output-root`**：使用调用方指定的空目录，不自动安装模型。已有内容则拒绝覆盖。长期保留的手工运行放 `results/training/`，单元测试用临时目录。
- **恢复训练**：`--resume` 指向旧运行目录、checkpoint 根目录或文件；结果写入新的运行目录，先复制 checkpoint，不能回写旧记录。不要让 `--output-root` 指向原运行。
- **续训身份**：仅恢复带数据指纹的 schema v2 checkpoint；源哈希、profile、smoke/full、抽样与分区行 ID、实际训练/验证数组顺序、种子、batch size、验证/checkpoint 间隔及模型契约必须一致。不得重新引入注入占位数组后静默续训的路径；缺少数据身份的旧 checkpoint 明确拒绝。允许增加目标总步数。
- **仿真路径**：`results/simulation/<时间>_<工况名>/`，JSON 内保存工况快照及完整结果。CLI 和 `visualize.py` 共用 `save_simulation_run`，不得重新创建 `simulation/outputs` 或源码下的 `results`。
- **历史位置**：XF 旧训练记录在 `results/training/legacy_xf/`；无完整工况来源的旧图片在 `results/simulation/legacy_figures/`。历史报告保留原始字段与内嵌路径，不伪装成当前单 KAN 输出。迁移映射见 `docs/maintenance/structure-migration.json`。
- 不在仓库根、源码目录或系统临时目录散落长期产物；新生成结果和二进制模型默认 Git 忽略，代码/配置/说明保留跟踪。已有跟踪的历史文件不会因 ignore 自动取消跟踪。
- 删除有数据的历史目录前必须有明确删除授权；本次仅迁移保留。`.venv`、`.agents`、`.codex`、`.claude` 和字体缓存不是无用实验目录，不随意清理。

## 必须保留的技术契约

**训练→仿真**：新风 10 个原始输入、主转轮 9 个原始输入，各一个固定六输出 KAN。运行时核验 profile、原始输入列、六个 `direct_outputs`、架构和 scaler 列；不读 `routing.json`、候选 registry 或 fallback。当前为调试阶段，不增加 promotion/完整性清单硬门禁，除非用户明确恢复工程验收阶段。

**复用与保存**：`dual_rotor_sim/scalers.py` 是输入/输出 scaler 列顺序、维度校验与目标变换适配器的共用实现；新 pickle 只保存原生 scaler 和普通变换状态，不能依赖 `training.*` 自定义对象。`fit` 与有界训练共用循环，按 step 间隔及结束时保存；周期验证共用一次前向计算损失与指标。未验证的最佳损失序列化为 `null`，不能伪造为 0。

**Pickle 与重复模块**：历史 scaler 使用 `src.preprocessing.OutputTransformer`，必须保留 `rotor.py` 的 `_ensure_scaler_pickle_compat` 及仿真包内副本。`kan_model.py`、`preprocessing.py`、`feature_engineering.py` 在 `training/src/` 与 `simulation/dual_rotor_sim/core/` 必须逐字节相同，修改须双侧同步并用 `md5sum` 核验；此次没有合并它们。旧数据处理工具移至 `archive/training/src/`，当前训练不依赖这些历史工具。

**拓扑**：主转轮冷吹流量等于其再生流量；新风转轮再生入口由主转轮再生出口、再生新风、新风转轮冷吹出口三股混风得到。两台转轮各自预测，不进行逐输出路由。

**能量与露点**：权威 XLS 的 `power` 是加热器功率记录，只作离线审计，不进入输入、目标、scaler、loss 或运行时输出。六个输出仅为空气状态；处理出口露点始终用 `dew_point_water_C(process_outlet_W, pressure_Pa)` 派生，Excel 露点列独立审计。

`energy_breakdown` 仅含 `CC1_kW`、`CC2_kW`、`Heater_Low_kW`、`Heater_OA_kW`。表冷器分项为负，`cooling_kW = abs(CC1) + abs(CC2)`；`heating_kW = Heater_OA + Heater_Low`。它们是热负荷（冷量隐含 COP=1），不代表整机直接电耗；公开结果无电机/风机/`electrical_kW`。风机配置兼容保留，但不进入公开能耗。

**凝结与自然满足**：表冷器在出口温度低于入口露点时有意模拟凝结，虽与早期需求的“只降温”表述不同，不得改回。加热器/表冷器在自然满足设定点时 `Q_kW=0` 属正常行为。

**压力与干燥极限**：`AirState.pressure_Pa` 随系统压力贯穿分流、混风、加热、表冷及全部转轮出口；露点、RH 和饱和含湿量使用该压力。不同压力的独立状态不得直接混合。零湿度的 `-inf` 露点是合法干燥极限，JSON 用 `null`，重绘结合 W=0 还原；不能用 epsilon 湿度掩盖。露点审计记录干燥极限数量与有限可比样本数，无可比样本的 MAE 为 `null`；NaN/+inf 或非零湿度对应 -inf 仍报错。

**收敛留档**：联立求解结果必须记录收敛状态、迭代次数、末次实际计算输入、残差与容差。未收敛可保存 JSON/PNG 调试结果，但 CLI/可视化退出码为 4，终端和仪表盘必须标识未收敛。历史记录缺字段只能标识未评估，不能推断成功；此数值状态不等于工程验收或 promotion。

**单位**：YAML/CLI 使用 m³/h，内部使用干空气质量流量 kg/s。五个边界流量经 `build_runtime_inputs`，统一调用 `SimulationConfig.m3h_to_kg_da_s` 按室外新风参考状态换算，不复制其他换算实现。
