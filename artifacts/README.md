# 最终风量KAN运行时包

只放kan_xf_single与kan_main_single，分别10/9输入、固定六输出。模型、scaler和元数据不上传；带外恢复完整包，并按docs/handoff/模型与数据身份.json校验。主项目中已存在这两个最终包，纯源码clone不会附带它们。

candidate风量训练不自动安装模型。旧XLS风速包不符合正式运行时，不得拿来替换。精确复现既有包应恢复原权重，不重新训练。安装与结构见docs/handoff/训练流程.md。
