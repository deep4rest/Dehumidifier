# 低露点双转轮除湿机建模平台

当前最终实现采用两台固定六输出**风量 KAN**、企业再生补/漏风规则及DptAH露点表。Python和C#采用同一套业务契约，NuGet业务版本1.0.1。所有Agent先读 [AGENTS.md](AGENTS.md)，再读 [另一台电脑接手入口](docs/handoff/README.md)。

源码包含最终风量仿真实现及2026-11-09新风基线升级；最初GitHub提交里的旧风速Python已纠正。模型、数据、仿真结果、ONNX/DLL/NuGet/ZIP和旧本机Git历史不上传。实际模型与数据身份见 [清单](docs/handoff/模型与数据身份.json)。

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

使用 Miniconda，在仓库根目录执行（Windows、Linux/WSL 通用）：

```bash
conda env create -f environment.yml
conda activate dehumidifier
python -m pytest -q
```

环境使用 Python 3.12，`environment.yml` 安装根依赖并以 editable 方式安装仿真包；已有环境可用 `conda env update -n dehumidifier -f environment.yml` 更新。非交互执行可用 `conda run -n dehumidifier python ...`，不再创建项目内 `.venv`。

需要 NVIDIA GPU 时，在该环境内安装 CUDA 13.0 版 PyTorch，并验证实际 GPU 计算；驱动兼容性以目标机检查结果为准，其他平台按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择安装包：

```bash
python -m pip install --upgrade "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(2, device='cuda').sum().item())"
python -m pip check
```

`environment.yml` 保留通用依赖，重建环境后需再执行 GPU 安装命令。安装或验证 CUDA 环境不代表已授权全量训练。

没有外部资料时，相应用例明确跳过，合成数据与临时模型测试照常运行。恢复后设置环境变量 `REQUIRE_EXTERNAL_FIXTURES=1`，执行 `python -m pytest -q -m 'requires_data or requires_models'` 强制外部资料检查（PowerShell 用 `$env:REQUIRE_EXTERNAL_FIXTURES='1'`）。运行环境版本没有全部锁定，不能仅凭相同源码保证跨机逐位一致。

## 当前新风基线与训练

2026-11-09用户已选定**9输入、三个湿度非负输出的KAN**为新风正式基线。主体9→24→6、grid6/order3、3600参数，处理出口含湿量保留log1p/expm1；三个湿度采用参数量为0的Softplus平滑物理下界(beta=50)，温度输出保持原形式。主转轮模型与整机物理契约不变。

本机已安装到`artifacts/kan_xf_single`。GitHub仅提供源码、模型身份和说明，模型/scaler、CSV、训练报告及ONNX必须带外恢复，完整SHA见[身份清单](docs/handoff/模型与数据身份.json)。新风三份CSV固定放在`data/新风转轮基线_9输入`，605,379条及485,041/57,000/63,338的尺寸隔离分区不变。原始转速列仍在审计数据中。

```bash
python -m training.single_kan.train \
  --profile xf --model-type kan \
  --dataset-dir data/新风转轮基线_9输入 \
  --device cpu --smoke-test --max-steps 4 --batch-size 256 --seed 20260908
```

默认`--humidity-output-constraint auto`会为新风log1p候选启用新基线头；主转轮仍无头。`none`显式重现旧无头对照；`--process-humidity-transform log`仅用于纯log实验。仅明确授权全量训练时才移除smoke参数并增加`--confirm-full --device cuda`。新候选不自动替换正式包，不省略dataset-dir进入旧XLS入口。

本次选定权重为最佳117轮、137轮早停。测试集处理出口负预测282→0，W<0.01 MAE降低62.63%，整体W MAE增加0.77%；再生出口W MAE增加28.20%的取舍如实保留。[本次修改、结果与验证边界](docs/handoff/2026-11-09-新风基线升级.md)，详细流程见[训练流程](docs/handoff/训练流程.md)。

## 仿真与部署

带外恢复两个完整包 `artifacts/kan_xf_single/`、`artifacts/kan_main_single/` 并按身份清单校验后：

```bash
dual-rotor-sim simulate --config simulation/configs/qy1070_case.yaml
python simulation/visualize.py --config simulation/configs/default_case.yaml
```

每次JSON/PNG同目录留档；未收敛仍留调试结果，退出4。改图使用已有JSON重绘，不重算。模型缺失/契约错误直接报错，不自动换历史模型。拓扑、流量、露点与负荷见 [系统契约](docs/handoff/系统拓扑与物理契约.md)。

ONNX和C#构建见 [部署交接](docs/handoff/ONNX与DLL交付.md)。源码契约一致、ONNX图误差、Windows部署、严格数值迁移、工程验收分别举证。历史QY1070有Python12次/C#13次迭代差异，不能推断严格迁移全部通过。

## 后续源码上传

每次需求完成并通过相关验证后，检查差异和暂存清单，提交本次需求相关源码及文档，再推送到当前工作分支的远程上游。无上游时使用 `git push -u origin <当前分支>`。不强推；推送失败需明确报告。提交者姓名和邮箱通过本仓库的 Git 配置设置。

远程操作优先使用已连接的 GitHub MCP：以当前远程提交为父提交创建新提交，更新分支时检查原 head 且不强推，再 fetch 并快进同步本地。MCP 使用连接账号的提交身份；本机 Git 命令行认证独立配置。

使用 `scripts/prepare_source_upload.py` 白名单生成空目录快照及独立manifest；排除全部数据、模型、结果、构建元数据与缓存。旧本机历史已跟踪模型，禁止直接push --all/--mirror。新GitHub clone可正常提交源码，仍应检查暂存文件。`deployment/dotnet/src/Dehumidifier.Simulation/build/*.targets`是必须保留的源码。
