"""训练与仿真共用的缩放器列契约及目标反变换。"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np


def validate_scaler_columns(scaler: Any, columns: Sequence[str], *, role: str) -> None:
    """检查顺序与维度，禁止同维度但语义不同的 scaler 混用。"""
    attribute = 'input_cols' if role == '输入' else 'output_cols'
    actual = getattr(scaler, attribute, None)
    if actual is None or list(actual) != list(columns):
        raise ValueError(f'{role} scaler 列顺序不符: {actual!r}，预期 {list(columns)!r}')
    if getattr(scaler, 'n_features_in_', None) != len(columns):
        raise ValueError(f'{role} scaler n_features_in_ 与列数不符')
    fitted_names = getattr(scaler, 'feature_names_in_', None)
    if fitted_names is not None and list(fitted_names) != list(columns):
        raise ValueError(f'{role} scaler 拟合列顺序不符')


class TargetScalerAdapter:
    """将目标缩放与 log1p 反变换合为同一个运行时接口。"""
    def __init__(self, scaler: Any, output_cols: Sequence[str], log_columns: Sequence[str] = ()):
        self.scaler = scaler
        self.output_cols = list(output_cols)
        self.log_columns = list(log_columns)
        if not set(self.log_columns) <= set(self.output_cols):
            raise ValueError('目标 log1p 列不属于输出契约')
        self._log_indices = [self.output_cols.index(c) for c in self.log_columns]
        validate_scaler_columns(scaler, self.output_cols, role='输出')

    @property
    def n_features_in_(self):
        return self.scaler.n_features_in_

    @property
    def feature_range(self):
        return self.scaler.feature_range

    def _matrix(self, values):
        result = np.asarray(values, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] != len(self.output_cols) or not np.isfinite(result).all():
            raise ValueError('目标 scaler 数组维度错误或含非有限值')
        return result

    def transform(self, values):
        matrix = self._matrix(values).copy()
        for index in self._log_indices:
            if np.any(matrix[:, index] < 0):
                raise ValueError('目标含湿量不能进行 log1p')
            matrix[:, index] = np.log1p(matrix[:, index])
        return self.scaler.transform(matrix)

    def inverse_transform(self, values):
        restored = np.asarray(self.scaler.inverse_transform(self._matrix(values)), dtype=np.float64)
        for index in self._log_indices:
            restored[:, index] = np.expm1(restored[:, index])
        if not np.isfinite(restored).all():
            raise ValueError('目标反变换产生非有限值')
        return restored
