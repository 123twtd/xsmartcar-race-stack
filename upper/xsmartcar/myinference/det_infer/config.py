# Copyright (c) 2026 清影/123twtd
"""检测模型配置加载与严格校验。

配置文件格式（{model}.rknn.json）：
    {
      "model_file": "yoloe_fp_3cls.rknn",
      "input_mode": "image",
      "output_format": "raw_dfl",
      "label_count": 3,
      "label_list": ["gold", "car", "human"],
      "input_size": [640, 640]   // 可选，[w, h]，由探针自动写入
    }

校验规则
--------
    找不到 .rknn.json          → 拒绝启动
    label_list 为空             → 拒绝启动（标签未确认）
    label_count != len(labels)  → 拒绝启动
    model_file 不匹配           → 拒绝启动
    input_mode/output_format 非法 → 拒绝启动
    input_size 非法（非 2 元正整数列表） → 拒绝启动

不再使用 status 字段——label_list 填齐且数量对得上就是 ready。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

# 合法枚举值（类型声明，供 IDE 提示用）
InputMode = Literal["image", "image_scale_factor"]
OutputFormat = Literal["raw_dfl", "boxes_scores", "decoded_boxes", "final_dets"]

# 合法枚举值集合（运行时校验用）
_VALID_INPUT_MODES = {"image", "image_scale_factor"}
_VALID_OUTPUT_FORMATS = {"raw_dfl", "boxes_scores", "decoded_boxes", "final_dets"}


class ModelConfigError(Exception):
    """配置加载或校验失败。

    抛出此异常意味着模型不能启动，调用方应中止初始化。
    """


def load_model_config(model_path: str) -> dict:
    """加载并严格校验检测模型配置。

    从 ``{model_path}.json`` 读取配置（例如 ``model.rknn`` → ``model.rknn.json``），
    校验通过后返回标准化的 dict。

    参数
    ----
    model_path : str
        .rknn 模型文件路径。配置文件路径为 ``{model_path}.json``。

    返回
    ----
    dict，包含以下键：
        - model_file   : str        模型文件名
        - input_mode   : str        预处理方式（image / image_scale_factor）
        - output_format: str        后处理分支（raw_dfl / boxes_scores / decoded_boxes / final_dets）
        - label_count  : int        类别数量
        - label_list   : list[str]  类别名称列表
        - input_size   : tuple[int, int] | None  模型输入 (w, h)，未配置时为 None

    异常
    ----
    ModelConfigError
        以下任一情况都会抛出：
        - 配置文件不存在
        - 缺少必须字段
        - input_mode / output_format 值非法
        - model_file 与实际模型名不一致
        - label_list 为空或包含非字符串
        - label_count 与 len(label_list) 不一致
        - input_size 非法（非 2 元正整数列表）
    """
    config_path = Path(f"{model_path}.json")

    # 规则 1：配置文件必须存在
    if not config_path.is_file():
        raise ModelConfigError(f"检测模型缺少配置文件: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))

    # 规则 2：必须存在的字段
    for key in ("model_file", "input_mode", "output_format", "label_count", "label_list"):
        if key not in data:
            raise ModelConfigError(f"配置缺少字段 '{key}': {config_path}")

    # 规则 3：枚举字段值必须合法
    if data["input_mode"] not in _VALID_INPUT_MODES:
        raise ModelConfigError(
            f"非法 input_mode '{data['input_mode']}': {config_path}"
        )
    if data["output_format"] not in _VALID_OUTPUT_FORMATS:
        raise ModelConfigError(
            f"非法 output_format '{data['output_format']}': {config_path}"
        )

    # 规则 4：model_file 必须与实际加载的模型文件名一致
    # 防止配置被复制到其他模型目录后误用
    actual_name = Path(model_path).name
    if data["model_file"] != actual_name:
        raise ModelConfigError(
            f"配置 model_file='{data['model_file']}' 与实际模型名 '{actual_name}' 不一致: {config_path}"
        )

    # 规则 5：label_list 必须填齐且与 label_count 一致
    label_count = int(data["label_count"])
    label_list = data["label_list"]

    if not isinstance(label_list, list) or not label_list:
        raise ModelConfigError(
            f"label_list 为空，标签未确认，禁止启动: {config_path}"
        )
    if not all(isinstance(label, str) and label.strip() for label in label_list):
        raise ModelConfigError(
            f"label_list 包含非字符串或空值: {config_path}"
        )
    if len(label_list) != label_count:
        raise ModelConfigError(
            f"label_count={label_count} 与 len(label_list)={len(label_list)} 不一致: {config_path}"
        )

    # 规则 6：input_size 可选，配置时必须是 [w, h] 两个正整数
    input_size = data.get("input_size")
    if input_size is not None:
        if (
            not isinstance(input_size, (list, tuple))
            or len(input_size) != 2
            or not all(isinstance(v, int) and v > 0 for v in input_size)
        ):
            raise ModelConfigError(
                f"非法 input_size {input_size!r}，须为 [w, h] 两个正整数: {config_path}"
            )
        input_size = (int(input_size[0]), int(input_size[1]))

    # 返回标准化的 dict（去掉了 status 字段，不再需要）
    result = {
        "model_file": data["model_file"],
        "input_mode": data["input_mode"],
        "output_format": data["output_format"],
        "label_count": label_count,
        "label_list": [label.strip() for label in label_list],
        "input_size": input_size,
    }
    return result
