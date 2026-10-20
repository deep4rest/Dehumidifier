"""尺寸隔离风量 CSV 的严格读取与共用预处理。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DIRECT_OUTPUT_COLUMNS,
    ID_COLUMN,
    VOLUME_INPUT_COLUMNS_BY_PROFILE,
)
from .model import Preprocessor, fit_preprocessor

SPLITS = ("train", "validation", "test")
_SPLIT_LABELS = {"train": "训练集", "validation": "验证集", "test": "测试集"}
_PROFILE_LABELS = {"xf": "新风转轮", "main": "主转轮"}


@dataclass(frozen=True)
class CandidateDataset:
    profile: str
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    frames: dict[str, pd.DataFrame]
    raw: dict[str, tuple[np.ndarray, np.ndarray]]
    scaled: dict[str, tuple[np.ndarray, np.ndarray]]
    preprocessor: Preprocessor
    file_hashes: dict[str, str]
    geometry_groups: dict[str, tuple[tuple[int, int], ...]]
    data_fingerprint: str

    @property
    def train_rows(self) -> int:
        return len(self.frames["train"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_for(root: Path, profile: str, split: str) -> Path:
    try:
        label = _PROFILE_LABELS[profile]
        split_label = _SPLIT_LABELS[split]
    except KeyError as exc:
        raise ValueError(f"未知 profile/split: {profile!r}/{split!r}") from exc
    return root / f"{label}_{split_label}.csv"


def _expected_columns(profile: str) -> tuple[str, ...]:
    try:
        inputs = tuple(VOLUME_INPUT_COLUMNS_BY_PROFILE[profile])
    except KeyError as exc:
        raise ValueError(f"未知 profile: {profile!r}") from exc
    return (ID_COLUMN, *inputs, *DIRECT_OUTPUT_COLUMNS)


def _read_frame(path: Path, profile: str, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"缺少 {profile} {split} CSV: {path}")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"读取 CSV 失败: {path}: {exc}") from exc
    expected = _expected_columns(profile)
    actual = tuple(str(column).strip() for column in frame.columns)
    if actual != expected:
        raise ValueError(
            f"{path.name} 列顺序/字段不符合契约: expected={expected}, actual={actual}"
        )
    if frame.empty:
        raise ValueError(f"{path.name} 不得为空")
    converted = frame.copy()
    for column in expected:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    values = converted.loc[:, expected].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} 含缺失、非数字或非有限值")
    ids = converted[ID_COLUMN].to_numpy(dtype=np.float64)
    if not np.equal(ids, np.floor(ids)).all():
        raise ValueError(f"{path.name} ID 必须为整数")
    converted[ID_COLUMN] = ids.astype(np.int64)
    if converted[ID_COLUMN].duplicated().any():
        raise ValueError(f"{path.name} ID 重复")
    for column in ("转轮厚度", "转轮直径"):
        geometry = converted[column].to_numpy(dtype=np.float64)
        if not np.equal(geometry, np.floor(geometry)).all():
            raise ValueError(f"{path.name} {column} 必须为整数")
        converted[column] = geometry.astype(np.int64)
    humidity_columns = [
        column for column in expected
        if "含湿量" in column
    ]
    if (converted[humidity_columns] < 0).any().any():
        raise ValueError(f"{path.name} 含湿量不得为负")
    converted.attrs.update({"profile": profile, "split": split, "source_path": str(path)})
    return converted


def _fingerprint(
    profile: str,
    input_columns: tuple[str, ...],
    frames: dict[str, pd.DataFrame],
    file_hashes: dict[str, str],
) -> str:
    digest = hashlib.sha256()
    digest.update(profile.encode("utf-8"))
    digest.update("\x00".join(input_columns).encode("utf-8"))
    for split in SPLITS:
        digest.update(split.encode("utf-8"))
        digest.update(file_hashes[split].encode("ascii"))
        ids = np.ascontiguousarray(frames[split][ID_COLUMN].to_numpy(dtype=np.int64))
        digest.update(ids.tobytes())
    return digest.hexdigest()


def load_candidate_dataset(dataset_dir: str | Path, profile: str) -> CandidateDataset:
    """加载一个转轮的三份 CSV，并只用训练分区拟合 scaler。"""

    profile = str(profile).strip().lower()
    root = Path(dataset_dir).resolve()
    input_columns = tuple(VOLUME_INPUT_COLUMNS_BY_PROFILE.get(profile, ()))
    if not input_columns:
        raise ValueError(f"未知 profile: {profile!r}")
    frames: dict[str, pd.DataFrame] = {}
    file_hashes: dict[str, str] = {}
    for split in SPLITS:
        path = _path_for(root, profile, split)
        before = _sha256(path)
        frames[split] = _read_frame(path, profile, split)
        after = _sha256(path)
        if before != after:
            raise ValueError(f"读取期间 CSV 发生变化: {path}")
        file_hashes[split] = before

    id_sets = {split: set(frames[split][ID_COLUMN].tolist()) for split in SPLITS}
    for first_index, first in enumerate(SPLITS):
        for second in SPLITS[first_index + 1:]:
            overlap = id_sets[first] & id_sets[second]
            if overlap:
                raise ValueError(f"分区 ID 重叠 {first}/{second}: {sorted(overlap)[:5]}")
            first_groups = set(zip(frames[first]["转轮厚度"], frames[first]["转轮直径"]))
            second_groups = set(zip(frames[second]["转轮厚度"], frames[second]["转轮直径"]))
            if first_groups & second_groups:
                raise ValueError(f"分区几何组重叠 {first}/{second}: {sorted(first_groups & second_groups)[:5]}")
            first_inputs = set(map(tuple, frames[first].loc[:, input_columns].to_numpy(dtype=np.float64)))
            second_inputs = set(map(tuple, frames[second].loc[:, input_columns].to_numpy(dtype=np.float64)))
            if first_inputs & second_inputs:
                raise ValueError(f"分区输入向量重叠 {first}/{second}")

    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        frame = frames[split]
        raw[split] = (
            frame.loc[:, input_columns].to_numpy(dtype=np.float32),
            frame.loc[:, DIRECT_OUTPUT_COLUMNS].to_numpy(dtype=np.float32),
        )
    train_x, train_y = raw["train"]
    preprocessor = fit_preprocessor(
        train_x,
        train_y,
        input_columns,
        DIRECT_OUTPUT_COLUMNS,
        log1p_targets=("处理出口含湿量",),
        fit_ids=frames["train"][ID_COLUMN].tolist(),
    )
    scaled = {
        split: (
            preprocessor.transform_features(values[0]),
            preprocessor.transform_targets(values[1]),
        )
        for split, values in raw.items()
    }
    geometry_groups = {
        split: tuple(sorted({
            (int(depth), int(diameter))
            for depth, diameter in zip(frame["转轮厚度"], frame["转轮直径"])
        }))
        for split, frame in frames.items()
    }
    return CandidateDataset(
        profile=profile,
        input_columns=input_columns,
        output_columns=tuple(DIRECT_OUTPUT_COLUMNS),
        frames=frames,
        raw=raw,
        scaled=scaled,
        preprocessor=preprocessor,
        file_hashes=file_hashes,
        geometry_groups=geometry_groups,
        data_fingerprint=_fingerprint(profile, input_columns, frames, file_hashes),
    )
