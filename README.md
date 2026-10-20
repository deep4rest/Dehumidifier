# 低露点双转轮除湿机建模平台

当前最终实现采用两台固定六输出**风量 KAN**、企业再生补/漏风规则及DptAH露点表。Python和C#采用同一套业务契约，NuGet业务版本1.0.1。所有Agent先读 [AGENTS.md](AGENTS.md)，再读 [另一台电脑接手入口](docs/handoff/README.md)。

本次源码已包含主项目尚未提交的最终修改；最初GitHub提交里的旧风速Python已纠正。模型、数据、仿真结果、ONNX/DLL/NuGet/ZIP和旧本机Git历史不上传。实际模型与数据身份见 [清单](docs/handoff/模型与数据身份.json)。

## 目录与入口

| 路径 | 职责 |
| --- | --- |
| training/single_kan/train.py | 统一训练CLI；**传 --dataset-dir 才走最终风量CSV训练** |
| training/single_kan/candidate_*.py | 尺寸隔离CSV读取、MLP/KAN、训练、验证与保存 |
| training/src/ | 共用KAN和保留的兼容模块 |
| simulation/dual_rotor_sim/ | 最终风量仿真、物性、求解器、记录与JSONL bridge |
| simulation/configs/ | 真实schema v2工况，不能为迎合模型修改 |
| deployment/dotnet/ | 最终C#整机、ONNX导出、Windows构建、严格比较和封装源码 |
| deployment/winforms/ | 较早的Python JSONL桥接开发路线，区别于纯C# DLL路线 |
| docs/handoff/ | 训练、拓扑、导出/封装和恢复说明 |
| artifacts/ | 本机两台正式模型包；GitHub仅附README |
| data/ | 带外数据；GitHub仅附恢复说明 |
| results/ | 本机训练/仿真/部署/源码上传验证记录，不上传 |
| archive/ | 历史兼容源码，不作为当前模型自动回退 |

## 环境与源码回归

Linux/WSL从仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e simulation/
.venv/bin/python -m pytest -q
```

没有外部资料时，相应用例明确跳过，合成数据与临时模型测试照常运行。恢复后使用 `REQUIRE_EXTERNAL_FIXTURES=1 .venv/bin/python -m pytest -q -m 'requires_data or requires_models'` 强制外部资料检查。运行环境版本没有全部锁定，不能仅凭相同源码保证跨机逐位一致。

## 最终风量模型的训练

先按 [数据恢复清单](data/README.md) 带外恢复六份尺寸隔离CSV，执行有界CPU smoke：

```bash
.venv/bin/python -m training.single_kan.train \
  --profile all --model-type kan \
  --dataset-dir data/转轮数据集_尺寸隔离_8比1比1 \
  --device cpu --smoke-test --max-steps 4 --batch-size 256 --seed 20260908
```

**仅明确获得全量训练授权后**，移除 `--smoke-test --max-steps 4` 并加 `--confirm-full`；CUDA可见时才显式选择 `--device cuda`。`--model-type all`可比较MLP与KAN，正式运行时只接受KAN。每个候选独立留档，候选流程不自动安装artifacts。不要省略 `--dataset-dir`：不传会进入保留的旧XLS风速流程，产物不符合最终风量运行时。

详细分区、scaler、优化器、早停、checkpoint、产物和安装步骤见 [训练流程](docs/handoff/训练流程.md)。默认不重新训练，恢复现有完整模型包才能复现既有预测。

## 仿真与部署

带外恢复两个完整包 `artifacts/kan_xf_single/`、`artifacts/kan_main_single/` 并按身份清单校验后：

```bash
.venv/bin/dual-rotor-sim simulate --config simulation/configs/qy1070_case.yaml
.venv/bin/python simulation/visualize.py --config simulation/configs/default_case.yaml
```

每次JSON/PNG同目录留档；未收敛仍留调试结果，退出4。改图使用已有JSON重绘，不重算。模型缺失/契约错误直接报错，不自动换历史模型。拓扑、流量、露点与负荷见 [系统契约](docs/handoff/系统拓扑与物理契约.md)。

ONNX和C#构建见 [部署交接](docs/handoff/ONNX与DLL交付.md)。源码契约一致、ONNX图误差、Windows部署、严格数值迁移、工程验收分别举证。历史QY1070有Python12次/C#13次迭代差异，不能推断严格迁移全部通过。

## 后续源码上传

使用 `scripts/prepare_source_upload.py` 白名单生成空目录快照及独立manifest；排除全部数据、模型、结果、构建元数据与缓存。旧本机历史已跟踪模型，禁止直接push --all/--mirror。新GitHub clone可正常提交源码，仍应检查暂存文件。`deployment/dotnet/src/Dehumidifier.Simulation/build/*.targets`是必须保留的源码。
