# 最终版本：另一台电脑的Agent接手入口

当前源码以用户确认的最终Python/C#企业版本为准：两台风量KAN、受控再生补/漏风、DptAH露点、零补风和relax=0.8。原先上传的旧worktree Python已纠正。主目录此前有尚未提交修改，所以只比较Git提交号无法识别最终状态；此次按实际文件核对同步。

依次阅读根 [AGENTS.md](../../AGENTS.md)、[README](../../README.md)、[训练流程](训练流程.md)、[系统拓扑与物理契约](系统拓扑与物理契约.md)、[ONNX与DLL交付](ONNX与DLL交付.md)。

## 恢复与启动

```bash
git clone https://github.com/deep4rest/Dehumidifier.git
cd Dehumidifier
conda env create -f environment.yml
conda activate dehumidifier
python -m pytest -q
```

GitHub不含数据、权重、scaler、结果和封装成品。按 [模型与数据身份](模型与数据身份.json) 带外恢复两个完整模型包，就可运行仿真和ONNX导出；无需重训。按 [data/README](../../data/README.md) 恢复六份尺寸隔离CSV，才能用最终训练流程。每个文件重新计算SHA-256，不能改列名替换风速/风量。

```bash
dual-rotor-sim simulate --config simulation/configs/qy1070_case.yaml
python -m pip install -r deployment/dotnet/requirements-export.txt
python deployment/dotnet/export_onnx.py --output results/deployment/新的编号/onnx
```

只有用户明确授权才可全量训练。默认无dataset-dir的旧XLS入口仍为历史风速流程，不能用于最终运行时。最终CPU smoke见训练文档，candidate不自动安装模型。

## 证据边界

Python和C#的输入/拓扑/物性/求解源码契约已同步；ONNX仍有float32误差，C#预处理的运算顺序也需严格数值比较。历史QY1070出现Python12次/C#13次迭代差异，不代表代码仍缺最终企业规则。重新构建后应保留真实比较报告，不能宣称工程验收通过。

Windows使用.NET 8及ONNX Runtime 1.20.1，实际包版本和工具链应锁定。相同源文件不保证重训权重、DLL/ZIP逐字节相同。现有权重身份、依赖版本、构建参数及ZIP时间戳都影响复现。

## 纯源码更新

```bash
python scripts/prepare_source_upload.py \
  --output results/source_upload/新的编号/source \
  --manifest results/source_upload/新的编号/manifest.json
```

脚本不复制Git历史、数据、模型、结果、egg-info和构建缓存，保留必要targets。旧本机仓库不得push --all/--mirror；GitHub新仓库仅有纯源码历史。运行资料缺失时明确跳过相关测试，恢复后REQUIRE_EXTERNAL_FIXTURES=1使缺资料失败。历史测试和文档仅供追溯，不作为当前模型替代或本次验证证据。
