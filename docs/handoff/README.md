# 另一台电脑的 Agent 接手入口

本次交付是可继续开发的纯源码：训练、仿真、测试、工况配置、ONNX 导出及 .NET DLL/验证器/WinForms 源码与说明。模型权重、scaler、原始表格、训练/仿真结果、封装成品和本机旧 Git 历史均不上传。不要将“clone 成功”理解为已有可用模型。

## 先读什么

1. 根 [AGENTS.md](../../AGENTS.md) 与 [README.md](../../README.md)：唯一维护规则、当前入口、训练授权和输出目录。
2. [训练流程](训练流程.md)：两份 XLS 的哈希与技术列、10/9 输入、六输出、数据清洗及分区、变换、架构、训练与续训、运行时安装。
3. [系统拓扑与物理契约](系统拓扑与物理契约.md)：v2 完整气路、三股混风、流量关系、迭代约束、压力、露点、凝结和热负荷。
4. [ONNX 与 DLL 交付](ONNX与DLL交付.md)：导出边界、DLL API、构建/封装、产物目录、迁移检查和当前未完成的同步项。

## 必须先知道的当前状态

截至本次 2026-10-20 移交核对，本工作树根没有两台正式运行时模型。GitHub 版也不带两份 XLS。数据恢复清单见 [data/README.md](../../data/README.md)；模型须带外恢复完整包或另获授权后全量训练。不要只复制 `.pt`、混用不同训练运行的 scaler/元数据、改源表哈希或用历史 ensemble 回退。

本工作树训练和 Python 仿真的处理/冷吹输入为**风速（m/s）**。从本机活跃部署目录恢复的历史 .NET 基线用**风量（m³/h）**，并具有不同混风规则、露点公式和求解默认值。导出器按本工作树数据契约导出，不代表当前 C# 已适配。接手者应先阅读部署文档中的差异表，完成同步后重新生成整机参考记录、ONNX 和 DLL；本次未宣称数值迁移通过或工程验收通过。

`archive/training/` 和 `archive/xf_rotor1_retrain/` 保留兼容源码，但历史数据/权重/报告不在 GitHub 中。它们不是当前训练入口。`docs/maintenance/structure-migration.json` 是原项目本机整理的历史映射，其中的产物路径不保证在纯源码 clone 中存在。

## 在 Linux / WSL 建立环境

```bash
git clone https://github.com/deep4rest/Dehumidifier.git
cd Dehumidifier
git status --short
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e simulation/
.venv/bin/python -m pytest -q
```

默认源码测试覆盖合成数据、临时模型、scaler/checkpoint 和仿真组件；缺少外部资料的测试以中文原因跳过。数据/历史权重恢复后，运行严格外部资料检查，缺失会失败：

```bash
REQUIRE_EXTERNAL_FIXTURES=1 .venv/bin/python -m pytest -q -m 'requires_data or requires_models'
```

两份当前数据到位后执行有界 smoke，检查实际生成的 `pipeline_report.json`：

```bash
.venv/bin/python -m training.single_kan.train \
  --profile all --smoke-test --max-steps 4 \
  --validation-interval-steps 2 --checkpoint-interval-steps 2 --no-progress
```

仅用户明确授权后才能执行 `--confirm-full`。默认全量成功将安装运行时包；显式 `--output-root` 不安装。正式模型到位后再按根 README 使用真实 YAML 仿真。原生 Windows 可使用 `.venv\Scripts\python.exe` 对应 Python 命令；本文主流程保持 Linux/WSL，Windows 的 .NET 构建步骤见部署文档。

## 后续如何继续上传

旧本机仓库已有跟踪的历史产物，不能直接推送它的历史。上传脚本使用明确源码目录、文本扩展名和内容检查，创建没有 `.git` 的新目录及独立 SHA-256 清单：

```bash
.venv/bin/python scripts/prepare_source_upload.py \
  --output results/source_upload/新的编号/source \
  --manifest results/source_upload/新的编号/manifest.json
```

输出目录须为空，脚本不删除本机数据。核对清单、在该目录测试后，仅从该快照建立 Git 提交；不要对原仓库使用 `push --all`、`--mirror` 或先提交再删模型。新 GitHub clone 的历史已经是纯源码，可以正常提交后续源码修改，但提交前仍需检查暂存文件及 ignore。

验证结论要分别报告：源码回归、有界 smoke、单转轮精度、系统数值收敛、ONNX 单轮误差、跨语言整机严格迁移、解压后部署验证、工程物理验收。没有执行的层级保留“未验证”。
