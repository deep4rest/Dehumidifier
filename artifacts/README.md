# 正式风量KAN运行时包

当前新风`kan_xf_single`为2026-11-09选定的9输入、六输出、Softplus湿度非负基线（schema_version=4，beta=50，最佳117轮）。主转轮`kan_main_single`保持原9输入模型。旧10输入新风包不能替代当前包。

模型、scaler、训练记录与ONNX不上传GitHub，必须带外恢复完整包并按[身份清单](../docs/handoff/模型与数据身份.json)逐文件核验。新风包包含model/{best_model.pt,scalers.pkl,scaler_meta.json,metadata.json}、model_metadata.json、config_snapshot.yaml与training_run.json。推理头已集成到Python与ONNX，不能重复执行Softplus或仅替换权重。

本次用户明确授权提升后已安装；未来candidate训练仍不自动安装。主转轮包本机尚未恢复时整机测试不可用，不能静默回退历史模型。流程见[升级说明](../docs/handoff/2026-11-09-新风基线升级.md)。
