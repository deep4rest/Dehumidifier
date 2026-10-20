# 低露点双转轮除湿机建模平台

平台使用新风转轮与主转轮各一个固定多输出 KAN，完成稳态整机前向仿真，输出空气状态和冷/热负荷。当前处于模型联调阶段；数值可运行与工程验收分别判断。

所有 Agent 先读 [AGENTS.md](AGENTS.md)，再按下列工作流操作。需求依据为 [需求.md](需求.md)。

## GitHub 源码与另一台电脑接手

从 [交接入口](docs/handoff/README.md) 开始，依次阅读 [训练流程](docs/handoff/训练流程.md)、[系统拓扑与物理契约](docs/handoff/系统拓扑与物理契约.md)、[ONNX 与 DLL 交付](docs/handoff/ONNX与DLL交付.md)。源码仓库不携带本机旧 Git 历史、原始数据、模型/scaler/checkpoint、运行结果或 ONNX/DLL/NuGet/ZIP 成品；这些文件本机保留，另行恢复。

当前工作树训练契约使用处理/冷吹**风速**，导入的 `deployment/dotnet/` 是历史**风量**版迁移基线，其拓扑、物性与当前 Python 存在明确差异，尚不能当作当前系统的匹配 DLL。逐项差异和继续适配、验证、封装的步骤已写入交接文档。本次移交不执行全量训练。

## 目录导航

| 位置 | 用途 |
| --- | --- |
| [training/single_kan](training/single_kan/) | 当前唯一训练流水线、数据契约、评估、checkpoint 与测试 |
| [training/src](training/src/) | 三个保留的 KAN/预处理/特征工程兼容模块 |
| [simulation/dual_rotor_sim](simulation/dual_rotor_sim/) | 仿真包、元件、求解器、CLI、仪表盘与记录器 |
| [simulation/configs](simulation/configs/) | schema v2 真实工况 YAML |
| [data](data/) | 两份权威 XLS；企业表格在 templates/，其他参考表在 reference/ |
| [artifacts](artifacts/) | 当前运行时两个单 KAN 模型包 |
| `results/training/` | 每次训练生成完整模型副本、日志、checkpoint、审计与失败记录，GitHub 不附带 |
| `results/simulation/` | 每次仿真生成配套 JSON、PNG；GitHub 不附带历史记录 |
| [archive](archive/) | 旧训练代码、XF 候选实验、历史模型与离线报告 |
| [docs](docs/) | 参考文档、维护说明和迁移清单 |

纯源码 clone 后 `artifacts/` 只有说明文件，`data/` 只有带外恢复清单。直接训练/仿真会如实报缺数据/模型；历史结果不代表当前模型已经训练完成。后续以实际模型元数据和训练记录为准。

## 固定工作流

### 1. 接手检查与环境

先查看已有修改、工况、模型元数据与最近的运行记录；不要覆盖其他任务留下的修改。

```bash
git status --short
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e simulation/
```

下列命令均从仓库根目录执行。训练与仿真共用 `.venv/`，无需另建实验环境。

### 2. 代码回归与小样本验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m training.single_kan.train \
  --profile all --smoke-test --max-steps 4 \
  --validation-interval-steps 2 --checkpoint-interval-steps 2 --no-progress
```

smoke 使用每份权威 XLS 约 500 行代表性副本，在此命令中每台模型最多训练 4 steps；它检查数据适配、训练、评估和保存链路，不能用于判断收敛或工程精度。默认保存到 `results/training/<时间>_smoke_all_<唯一后缀>/`，不更新运行时模型。

没有带外数据/历史模型时，默认 pytest 会明确跳过标记为 `requires_data` / `requires_models` 的外部资料测试，合成数据和临时模型测试照常执行。恢复资料后，使用 `REQUIRE_EXTERNAL_FIXTURES=1 .venv/bin/python -m pytest -q -m 'requires_data or requires_models'` 强制检查；缺资料将失败，不能把源码回归通过当作完整外部资料验证。

检查该目录 `pipeline_report.json` 的 `status`、`failures` 和各 profile 的评估报告。失败记录也会保留；进程意外终止时可能停留在 `running`，不能仅凭目录存在认定成功。测试覆盖 checkpoint 重载、推理和整机接口。

### 3. 明确授权后全量训练

只有用户明确授权全量训练后执行：

```bash
.venv/bin/python -m training.single_kan.train --profile all --confirm-full
```

默认保存到新的 `results/training/<时间>_full_all_<唯一后缀>/`。全部请求 profile 成功后，才从记录中提取对应模型更新 `artifacts/kan_xf_single/`、`artifacts/kan_main_single/`。历史运行完整保留，原有运行时包移入该次运行的 `previous_artifacts/`。这一步不代表精度验收，不创建 promotion 门禁。

每次训练的结构：

```text
results/training/<运行编号>/
├── pipeline_report.json
├── checkpoints/{xf,main}/{checkpoint.pt,best_model.pt}
├── kan_xf_single/
│   ├── model/{best_model.pt,scalers.pkl,scaler_meta.json,metadata.json}
│   ├── model_metadata.json
│   ├── config_snapshot.yaml
│   ├── train.log.jsonl
│   ├── run_report.json
│   └── reports/
│       ├── cleaning_report.json
│       ├── split_report.json
│       ├── train_history.json
│       ├── evaluation.json
│       └── power_comparison.json
└── kan_main_single/    同上
```

运行时包仅包含 `model/`、`model_metadata.json`、`config_snapshot.yaml`、指回训练记录的 `training_run.json`；训练日志和 checkpoint 不混入 `artifacts/`。

按需使用的参数：

| 参数 | 规则 |
| --- | --- |
| `--profile xf\|main\|all` | 默认 all；单 profile 只处理对应转轮 |
| `--output-root PATH` | 必须是新建或空目录；不自动更新 artifacts，保留记录建议放 results/training 下 |
| `--resume PATH` | 指向旧运行目录、checkpoint 根目录或 checkpoint 文件，恢复写入新记录，不覆盖原运行 |
| `--max-steps N` | 有界验证的总步数上限；续训时表示目标总 step，不是新增 step |
| `--validation-interval-steps N` / `--checkpoint-interval-steps N` | 按 optimizer step 设置评估/存储间隔 |
| `--seed N --batch-size N --device cpu\|cuda` | 复现与执行设置 |

恢复时保持原 profile、smoke/full 模式、种子、样本范围、batch size，以及验证和 checkpoint 间隔；全量恢复仍须 `--confirm-full`。checkpoint schema v2 绑定源文件哈希、分区行 ID、实际训练/验证数组及顺序、模型契约与关键训练控制，任一不一致就拒绝续训。缺少数据身份的旧 checkpoint 不能直接恢复；须启动新训练记录。`--max-steps` 可增加为新的目标总步数。示例中将 `旧运行编号` 替换为实际已核验记录，并按旧运行的 `config_snapshot.yaml` 补齐非默认参数：

```bash
.venv/bin/python -m training.single_kan.train \
  --profile all --confirm-full --resume results/training/旧运行编号
```

训练循环按配置间隔保存 checkpoint，并在结束时保存最终状态；最佳模型在验证改善时单独保存。每次周期验证只做一次模型前向，共用预测计算缩放损失和物理指标。未到首个验证间隔的短运行仍可完成末尾评估和保存，`train_history.json` 中 `best_validation_loss: null` 表示尚无周期验证结果。

### 4. 使用真实工况仿真

先确认两个运行时包存在并满足列与架构契约，再运行：

```bash
.venv/bin/dual-rotor-sim simulate --config simulation/configs/default_case.yaml
.venv/bin/dual-rotor-sim simulate --config simulation/configs/qy1070_case.yaml --json
.venv/bin/python simulation/visualize.py --config simulation/configs/qy1070_case.yaml
```

三个命令均按次写入 `results/simulation/<时间>_<工况名>/`，每次目录有：

- `result.json`：原工况内容快照、时间、完整状态（含压力）、热负荷及 `convergence`。
- `fig1_dashboard.png`：同一次计算的仪表盘。

不会覆盖旧图，也不会再创建 `simulation/results/` 或 `simulation/outputs/`。缺少模型直接报错；不从 `archive/` 自动挑选替代模型，不修改真实工况来绕过失败。

检查 `result.convergence` 的 `converged`、`iterations`、`residuals` 和 `tolerances`。未收敛时仍保存末次计算的 JSON/PNG 调试记录，但 CLI 和 `visualize.py` 以退出码 **4** 结束，仪表盘明确标记“未收敛”。缺少收敛字段的历史记录重绘为“未评估收敛”，不能据此判断稳态成立。

### 5. 重绘、审计与交付

只改图片时，读取已留档 JSON，不重跑仿真。可在 Python 中调用：

```python
from pathlib import Path
from dual_rotor_sim.dashboard import save_dashboard_from_record

run_dir = Path("results/simulation/实际运行编号")
save_dashboard_from_record(
    record_path=run_dir / "result.json",
    target=run_dir / "fig1_dashboard_redraw.png",
)
```

保留原图，重绘文件另命名并与源 JSON 同目录。历史 JSON 可能含旧版电耗字段；重绘不会使其变成新模型的仿真证据。

交付需写明：本次改动、实际执行的测试、训练/仿真记录路径、模型精度及物理问题。不要把 smoke、接口回归或旧结果当作工程验收。

## 模型与物理契约

- 权威源固定为 `data/新风转轮数据集.xls`（CollectRecordXF）和 `data/主转轮数据集.xls`（CollectRecord）；只读并校验 SHA-256。
- 新风：10 个原始输入，KAN hidden `[24]`、grid 6、order 3；主转轮：9 个原始输入，hidden `[16]`、grid 5、order 3，不加派生特征。
- 每台转轮直接输出处理、冷吹、再生出口的温度和含湿量，共六个。处理出口露点由预测含湿量和压力用 `dew_point_water_C` 统一派生。
- 零含湿量对应干燥极限露点 `-inf`；JSON 保存为 `null`，重绘结合含湿量恢复其含义。评估报告单列 `dew_point_dry_limit_prediction_count`、`dew_point_dry_limit_target_count`；Excel 露点 MAE 仅统计有限可比预测，并报告 `dew_point_excel_comparable_count`，无可比样本时 MAE 为 `null`。六个直接输出仍参与完整指标计算。
- 新产物的 scaler pickle 保存原生 sklearn scaler 与普通变换状态，不包含训练模块自定义对象。训练和仿真共用 `dual_rotor_sim/scalers.py`，在保存/加载时核对输入、输出列顺序、拟合列名（如有）和维度。
- `power` 是源表加热器功率记录，仅离线审计；Excel 露点列同样不作目标。仿真公开结果仅为空气状态和热负荷，不含风机/电机/整机直接电耗。
- `cooling_kW = abs(CC1_kW) + abs(CC2_kW)`，`heating_kW = Heater_OA_kW + Heater_Low_kW`。表冷器分项为负；冷负荷隐含 COP=1，不等于设备电耗。
- YAML/CLI 用 m³/h，内部用 kg dry-air/s；五个边界流量由 `build_runtime_inputs` 统一换算。主转轮冷吹量等于再生量，新风再生入口为三股混风。
- 表冷器低于入口露点时模拟凝结；自然满足设定点时加热器/表冷器热负荷为零。
- `psychrometrics.pressure_Pa` 贯穿边界与中间空气状态、转轮、表冷器饱和含湿量、露点及相对湿度。独立元件默认标准大气压；混合不同压力的空气状态会报错。

## 历史资料与维护

旧 XF 训练工作目录与报告在 `results/training/legacy_xf/`，源码在 `archive/xf_rotor1_retrain/`。旧候选/ensemble 模型在 `archive/artifacts/`，仅供离线比较。历史配置可能指向已停用数据文件，不能当作当前训练入口；旧报告内嵌路径按原始证据保留。

当前默认测试入口由 `pytest.ini` 固定。修改历史兼容工具时另运行：

```bash
.venv/bin/python -m pytest archive/xf_rotor1_retrain/tests -q
```

完整整理说明和逐文件迁移清单见 [结构维护记录](docs/maintenance/2026-09-05-structure.md)。新增入口、变更默认路径或调整工作流时，必须同时维护本 README 与 AGENTS.md；CLAUDE.md 只引用它们。
