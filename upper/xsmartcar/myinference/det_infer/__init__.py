# Copyright (c) 2026 清影/123twtd
"""目标检测推理包。

只保留两个正式文件：
- ``config.py``    读取 {model}.rknn.json 并严格校验
- ``detector.py``  检测器、Detection 类型、两个输出分支、分核多实例池

导出
----
- ``Detection``           单个检测结果
- ``LetterboxMeta``        letterbox 预处理元信息
- ``draw_detections``     调试用画框
- ``letterbox_bgr``        letterbox 预处理
- ``decode_outputs_fast``  DFL 解码（YOLOv8 6 输出 / YOLOE 9 输出）
- ``RKNNDetector``         单实例检测器（一个模型 + 一个 NPU core）
- ``MultiCoreDetector``    多实例池（多个 RKNN 实例分配到不同 core）
- ``load_model_config``    加载 {model}.rknn.json 配置（含严格校验）
- ``ModelConfigError``     配置加载或校验失败异常
"""

from .config import load_model_config, ModelConfigError
from .detector import (
    Detection,
    LetterboxMeta,
    MultiCoreDetector,
    RKNNDetector,
    decode_outputs_fast,
    draw_detections,
    letterbox_bgr,
)

__all__ = [
    "Detection",
    "LetterboxMeta",
    "ModelConfigError",
    "MultiCoreDetector",
    "RKNNDetector",
    "decode_outputs_fast",
    "draw_detections",
    "letterbox_bgr",
    "load_model_config",
]
