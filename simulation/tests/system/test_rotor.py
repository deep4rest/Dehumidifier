"""⑥  转轮 KAN 集成模型测试。

验证：
- Rotor.process(inputs) 与历史 RotorEnsemble.predict(inputs) 结果完全一致（零容差）
- 预测输出的物理合理性：出口含湿量 > 0；温度在合理范围内
- 多工况下预测一致性（相同输入多次调用返回相同结果）；公开结果不含功率

注意：
  本测试依赖 artifacts/ 目录下的真实 KAN 权重文件，加载较慢。
  conftest.py 通过 session 级 fixture rotor_ensemble 只加载一次。
"""

from __future__ import annotations

import pytest

from dual_rotor_sim.core.system.components.rotor import Rotor, RotorInputs, RotorEnsemble


# ---------------------------------------------------------------------------
# ⑥-A  Rotor.process 与 RotorEnsemble.predict 结果完全一致
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inputs", [
    RotorInputs(
        process_inlet_T=35.0,
        process_inlet_W=18.0,
        regen_T=120.0,
        cool_blow_inlet_T=25.0,
        regen_angle=60.0,
        cool_blow_angle=30.0,
        rpm=5.0,
        airspeed=2.5,
    ),
    RotorInputs(
        process_inlet_T=28.0,
        process_inlet_W=12.0,
        regen_T=100.0,
        cool_blow_inlet_T=20.0,
        regen_angle=60.0,
        cool_blow_angle=30.0,
        rpm=5.0,
        airspeed=2.5,
    ),
    RotorInputs(
        process_inlet_T=20.0,
        process_inlet_W=8.0,
        regen_T=80.0,
        cool_blow_inlet_T=18.0,
        regen_angle=60.0,
        cool_blow_angle=30.0,
        rpm=5.0,
        airspeed=2.5,
    ),
])
def test_rotor_process_equals_ensemble_predict(rotor_ensemble, inputs):
    """Rotor.process 应与 RotorEnsemble.predict 逐字段完全一致（容差 0）。"""
    rotor = Rotor(ensemble=rotor_ensemble)
    out_rotor = rotor.process(inputs)
    out_ensemble = rotor_ensemble.predict(inputs)

    assert out_rotor.process_outlet_T    == out_ensemble.process_outlet_T,    "process_outlet_T 不一致"
    assert out_rotor.process_outlet_W    == out_ensemble.process_outlet_W,    "process_outlet_W 不一致"
    assert out_rotor.process_outlet_Tdp  == out_ensemble.process_outlet_Tdp,  "process_outlet_Tdp 不一致"
    assert out_rotor.regen_outlet_T      == out_ensemble.regen_outlet_T,      "regen_outlet_T 不一致"
    assert out_rotor.regen_outlet_W      == out_ensemble.regen_outlet_W,      "regen_outlet_W 不一致"
    assert out_rotor.cool_blow_outlet_T  == out_ensemble.cool_blow_outlet_T,  "cool_blow_outlet_T 不一致"
    assert out_rotor.cool_blow_outlet_W  == out_ensemble.cool_blow_outlet_W,  "cool_blow_outlet_W 不一致"
    assert not hasattr(out_rotor, "power_kW")


# ---------------------------------------------------------------------------
# ⑥-B  物理合理性：处理侧除湿（出口含湿量 < 入口）
# ---------------------------------------------------------------------------

def test_rotor_process_dehumidification(rotor_ensemble):
    """转轮处理侧：出口含湿量应低于入口（基本除湿效果）。"""
    inputs = RotorInputs(
        process_inlet_T=35.0,
        process_inlet_W=18.0,
        regen_T=120.0,
        cool_blow_inlet_T=25.0,
    )
    rotor = Rotor(ensemble=rotor_ensemble)
    out = rotor.process(inputs)
    assert out.process_outlet_W < inputs.process_inlet_W, (
        f"处理侧出口 W={out.process_outlet_W:.4f} 应 < 入口 W={inputs.process_inlet_W}"
    )


# ---------------------------------------------------------------------------
# ⑥-C  物理合理性：含湿量为正值
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inputs", [
    RotorInputs(process_inlet_T=35.0, process_inlet_W=18.0,
                regen_T=120.0, cool_blow_inlet_T=25.0),
    RotorInputs(process_inlet_T=25.0, process_inlet_W=10.0,
                regen_T=90.0,  cool_blow_inlet_T=20.0),
])
def test_rotor_W_positive(rotor_ensemble, inputs):
    """所有侧的出口含湿量预测值应为正值。"""
    rotor = Rotor(ensemble=rotor_ensemble)
    out = rotor.process(inputs)

    assert out.process_outlet_W > 0,   f"process_outlet_W={out.process_outlet_W:.4f} 应 > 0"
    assert out.regen_outlet_W   > 0,   f"regen_outlet_W={out.regen_outlet_W:.4f} 应 > 0"
    assert out.cool_blow_outlet_W > 0, f"cool_blow_outlet_W={out.cool_blow_outlet_W:.4f} 应 > 0"


# ---------------------------------------------------------------------------
# ⑥-D  公开契约：转轮不输出电机功率
# ---------------------------------------------------------------------------

def test_rotor_has_no_power_output(rotor_ensemble):
    """转轮输出只包含空气状态和运行时派生露点。"""
    inputs = RotorInputs(
        process_inlet_T=35.0, process_inlet_W=18.0,
        regen_T=120.0, cool_blow_inlet_T=25.0,
    )
    rotor = Rotor(ensemble=rotor_ensemble)
    out = rotor.process(inputs)
    assert not hasattr(out, "power_kW")


# ---------------------------------------------------------------------------
# ⑥-E  确定性（相同输入多次调用返回相同结果）
# ---------------------------------------------------------------------------

def test_rotor_deterministic(rotor_ensemble):
    """相同输入两次调用应返回完全相同的预测结果。"""
    inputs = RotorInputs(
        process_inlet_T=30.0, process_inlet_W=14.0,
        regen_T=110.0, cool_blow_inlet_T=22.0,
    )
    rotor = Rotor(ensemble=rotor_ensemble)
    out1 = rotor.process(inputs)
    out2 = rotor.process(inputs)

    assert out1.process_outlet_T   == out2.process_outlet_T
    assert out1.process_outlet_W   == out2.process_outlet_W
    assert out1.process_outlet_Tdp == out2.process_outlet_Tdp
    assert out1.regen_outlet_T     == out2.regen_outlet_T
    assert out1.regen_outlet_W     == out2.regen_outlet_W
    assert out1.cool_blow_outlet_T == out2.cool_blow_outlet_T
    assert out1.cool_blow_outlet_W == out2.cool_blow_outlet_W
    assert not hasattr(out1, "power_kW")


pytestmark = pytest.mark.requires_models
