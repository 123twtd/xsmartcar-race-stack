# Copyright (c) 2026 清影/123twtd
"""myinference — 感知推理与运行时编排包。

子模块
------
- ``rknnpool``            RKNN 多 runtime 线程池（异步分割底座）
- ``runtime_pipeline``   比赛运行时流水线编排（感知调度 + 几何 + 控制 + UART）
- ``seg_infer``           语义分割推理（同步 / 异步）
- ``det_infer``           目标检测推理 + 后处理
"""
