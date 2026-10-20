"""pytest 共享 fixtures。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
# dual_rotor_sim 包所在目录（未 pip install 时也可运行测试）
_SIMULATION_ROOT = _PROJECT_ROOT / "simulation"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def rotor_ensemble():
    """加载一次 RotorEnsemble 供所有转轮测试复用（session 级避免重复加载 KAN 权重）。"""
    import sys
    for path in (_PROJECT_ROOT, _SIMULATION_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from dual_rotor_sim.core.system.components.rotor import (
        RotorEnsemble, _RUN_REGISTRY, _load_saved_routing,
    )
    artifacts_dir = _PROJECT_ROOT / "archive" / "artifacts"
    # 只检查实际路由使用的外部文件；损坏的路由、权重或元数据继续报错，
    # 不把模型回归失败误判为缺少资料。临时生成的模型测试不使用此 fixture。
    routing = _load_saved_routing(artifacts_dir, "legacy")
    required = [
        artifacts_dir / _RUN_REGISTRY[alias]["dir"] / "model" / name
        for alias in sorted(set(routing.values()))
        for name in ("scaler_meta.json", "scalers.pkl", "best_model.pt")
    ]
    missing = [str(path.relative_to(_PROJECT_ROOT)) for path in required
               if not path.is_file()]
    if missing:
        reason = "缺少单独提供的历史转轮模型资料：" + "、".join(missing)
        if os.environ.get("REQUIRE_EXTERNAL_FIXTURES") == "1":
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)
    return RotorEnsemble.from_artifacts(artifacts_dir=artifacts_dir, device="cpu")
