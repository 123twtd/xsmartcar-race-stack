# Copyright (c) 2026 清影/123twtd
"""RKNN 目标检测推理模块。

一个模型实例固定绑定一个 NPU core，配置由 {model}.rknn.json 驱动。
多实例并行使用 ``MultiCoreDetector``，每个实例绑定不同 core。

结构
----
- ``Detection``           单个检测结果（类别 + 分数 + 坐标）
- ``LetterboxMeta``        letterbox 预处理元信息
- ``letterbox_bgr``        等比缩放加边填充
- ``draw_detections``      调试用画框
- ``decode_outputs_fast``  DFL 解码（YOLOv8 6 输出 / YOLOE 9 输出）
- ``RKNNDetector``         单实例检测器（一个模型 + 一个 NPU core）
- ``MultiCoreDetector``    多实例池（多个 RKNN 实例分配到不同 core）

用法
----
    # 单实例：检测绑定 core 1
    det = RKNNDetector("model.rknn", core_id=1)
    detections = det.infer(frame_bgr)
    det.release()

    # 多实例：core 1 + core 2 并行
    pool = MultiCoreDetector("model.rknn", core_ids=(1, 2))
    future = pool.submit(frame_bgr)   # 异步
    detections = future.result()
    # 或同步便利接口
    detections = pool.infer(frame_bgr)
    pool.release()
"""



from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from queue import Empty, Queue
from typing import Iterable

import cv2
import numpy as np

from .config import load_model_config


# ─── 默认配置 ───────────────────────────────────────────────

# 模型输入分辨率（须与 .rknn 导出时一致）
DEFAULT_INPUT_SIZE = (640, 640)
# 置信度阈值：低于此分数的检测框直接丢弃
DEFAULT_CONF_THR = 0.5
# NMS IoU 阈值：同类别框重叠超过此值时抑制低分框
DEFAULT_IOU_THR = 0.45


# ─── 数据结构 ───────────────────────────────────────────────

@dataclass(slots=True)
class Detection:
    """单个检测结果。

    Attributes
    ----------
    class_id : int
        类别 ID（从 0 开始）。
    class_name : str
        类别名称（从 config.label_list 查表得到）。
    score : float
        置信度分数（0~1）。
    box_xyxy : list[float]
        边界框 [x1, y1, x2, y2]，原图坐标（已从 letterbox 空间还原）。
    """

    class_id: int
    class_name: str
    score: float
    box_xyxy: list[float]

    @property
    def center_xy(self) -> list[float]:
        """检测框中心点 [cx, cy]。"""
        x1, y1, x2, y2 = self.box_xyxy
        return [0.5 * (x1 + x2), 0.5 * (y1 + y2)]

    @property
    def size_wh(self) -> list[float]:
        """检测框宽高 [w, h]。"""
        x1, y1, x2, y2 = self.box_xyxy
        return [x2 - x1, y2 - y1]

    def to_dict(self) -> dict:
        """转换为 dict（含 center_xy 和 size_wh），用于 JSON 序列化。"""
        data = asdict(self)
        data["center_xy"] = self.center_xy
        data["size_wh"] = self.size_wh
        return data


@dataclass(slots=True)
class LetterboxMeta:
    """letterbox 预处理元信息，用于把检测框还原回原图坐标。

    letterbox 会把原图等比缩放后放在画布中央，四周补灰边。
    后处理时需要用这些信息把检测框从模型输入空间还原回原图空间。

    Attributes
    ----------
    scale : float
        缩放比例（原图 → 模型输入）。
    pad_x, pad_y : int
        水平/垂直 padding 像素数（画布左/上侧的灰边宽度）。
    input_w, input_h : int
        模型输入尺寸（通常 640×640）。
    orig_w, orig_h : int
        原图尺寸。
    """

    scale: float
    pad_x: int
    pad_y: int
    input_w: int
    input_h: int
    orig_w: int
    orig_h: int


# ─── 预处理 ───────────────────────────────────────────────

def letterbox_bgr(
    image_bgr: np.ndarray,
    input_size: tuple[int, int],
    pad_value: int = 114,
) -> tuple[np.ndarray, LetterboxMeta]:
    """等比例缩放加边填充，保持目标宽高比。

    为什么需要 letterbox：
        模型输入是固定的 640×640，但相机帧的宽高比通常不是 1:1。
        直接 resize 会拉伸目标形状，导致检测框不准。
        letterbox 等比缩放后把图放在画布中央，四周补灰边（pad_value=114）。

    返回
    ----
    (canvas, meta)
        canvas : (input_h, input_w, 3) 的 letterbox 后图像
        meta   : LetterboxMeta，后处理时用于坐标还原
    """
    input_w, input_h = input_size
    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(input_w / max(orig_w, 1), input_h / max(orig_h, 1))
    resized_w = max(1, int(round(orig_w * scale)))
    resized_h = max(1, int(round(orig_h * scale)))
    resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((input_h, input_w, 3), pad_value, dtype=np.uint8)
    pad_x = (input_w - resized_w) // 2
    pad_y = (input_h - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized

    meta = LetterboxMeta(
        scale=float(scale),
        pad_x=int(pad_x),
        pad_y=int(pad_y),
        input_w=int(input_w),
        input_h=int(input_h),
        orig_w=int(orig_w),
        orig_h=int(orig_h),
    )
    return canvas, meta


def _preprocess(
    frame_bgr: np.ndarray,
    input_size: tuple[int, int],
    input_mode: str,
) -> tuple[list[np.ndarray], LetterboxMeta]:
    """BGR 原图 → letterbox → NPU 输入张量列表。

    步骤：
        1. letterbox 等比例缩放加边填充到 input_size
        2. BGR → RGB（RKNN 模型训练时用 RGB）
        3. 按 input_mode 决定数据类型和输入数量：
           - image:             uint8 (1, H, W, 3)，单输入
           - image_scale_factor: float32 (1, H, W, 3) + scale_factor (1, 2)，双输入
             PaddleDetection PP-YOLOE 模型需要 image + scale_factor 两个输入。
             传 scale_factor=[1,1] 让模型不做反向缩放，坐标还原由 unletterbox 处理。
    """
    letterboxed, meta = letterbox_bgr(frame_bgr, input_size)
    img_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

    if input_mode == "image_scale_factor":
        # PP-YOLOE: float32 image（不除 255，模型内部做归一化）+ scale_factor [1,1]
        img_input = np.expand_dims(img_rgb.astype(np.float32), axis=0)
        scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)
        return [img_input, scale_factor], meta
    else:
        img_input = np.expand_dims(img_rgb, axis=0)
        return [img_input], meta


# ─── 坐标还原 ─────────────────────────────────────────────

def unletterbox_boxes(boxes_xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """把 letterbox 后的框坐标还原回原图坐标。

    还原步骤（与 letterbox_bgr 相反）：
        1. 减去 padding（pad_x, pad_y）
        2. 除以缩放比例（scale）
        3. 裁剪到原图范围内
    """
    boxes = boxes_xyxy.astype(np.float32).copy()
    boxes[:, [0, 2]] -= float(meta.pad_x)
    boxes[:, [1, 3]] -= float(meta.pad_y)
    boxes /= max(meta.scale, 1e-6)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(meta.orig_w - 1))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(meta.orig_h - 1))
    return boxes


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h] → [x1, y1, x2, y2]。"""
    boxes = boxes_xywh.astype(np.float32).copy()
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    out[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    out[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    out[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    return out


# ─── NMS（非极大值抑制）───────────────────────────────────

def nms_xyxy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """标准 NMS，返回保留的索引数组。

    原理：按分数从高到低排序，每次取最高分的框，抑制与它 IoU 超过阈值的框。
    """
    if boxes_xyxy.size == 0:
        return np.zeros((0,), dtype=np.int64)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[np.where(iou <= iou_thr)[0] + 1]

    return np.asarray(keep, dtype=np.int64)


def per_class_nms(
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    iou_thr: float,
) -> np.ndarray:
    """按类别分别做 NMS，返回保留的索引数组。

    不同类别的框即使重叠也不互相抑制。
    """
    keep_all: list[np.ndarray] = []
    for class_id in np.unique(class_ids):
        idx = np.where(class_ids == class_id)[0]
        keep_local = nms_xyxy(boxes_xyxy[idx], scores[idx], iou_thr)
        if keep_local.size > 0:
            keep_all.append(idx[keep_local])
    if not keep_all:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(keep_all)


# ─── 构造结果 ─────────────────────────────────────────────

def detections_from_arrays(
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
) -> list[Detection]:
    """从 numpy 数组构造 Detection 列表。"""
    detections: list[Detection] = []
    for box, class_id, score in zip(boxes_xyxy, class_ids, scores):
        cid = int(class_id)
        detections.append(
            Detection(
                class_id=cid,
                class_name=class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}",
                score=float(score),
                box_xyxy=[float(v) for v in box.tolist()],
            )
        )
    return detections


# ─── 可视化 ───────────────────────────────────────────────

def draw_detections(image_bgr: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """在原图上绘制检测框和标签，用于调试预览。"""
    out = image_bgr.copy()
    det_list = list(detections)
    if not det_list:
        cv2.putText(out, "NO DETECTIONS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return out

    for det in det_list:
        x1, y1, x2, y2 = [int(round(v)) for v in det.box_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.score:.2f}"
        cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


# ─── DFL 解码（YOLOv8 6 输出 / YOLOE 9 输出）──────────────

def _softmax_last(values: np.ndarray) -> np.ndarray:
    """对最后一维做 softmax（数值稳定版）。"""
    values = values.astype(np.float32, copy=False)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.maximum(np.sum(exp_values, axis=-1, keepdims=True), 1e-6)


def _decode_selected_boxes(
    position: np.ndarray,
    selected: np.ndarray,
    *,
    input_hw: tuple[int, int],
) -> np.ndarray:
    """只对 selected 列出的 grid 位置做 DFL 解码，返回 (N, 4) 的 xyxy 框。

    DFL 解码步骤：
        1. 把 position 重整为 (4, bins) 的分布
        2. softmax 得到概率分布
        3. 加权求和得到到网格中心的 4 个距离（left, top, right, bottom）
        4. 网格中心坐标 ± 距离 × stride = 检测框坐标
    """
    position = np.asarray(position)
    if position.ndim != 4 or position.shape[0] != 1 or position.shape[1] % 4:
        raise ValueError(f"invalid DFL tensor shape: {position.shape}")

    _, channels, grid_h, grid_w = position.shape
    bins = channels // 4

    distributions = (
        position.reshape(1, 4, bins, grid_h, grid_w)[0]
        .transpose(2, 3, 0, 1)
        .reshape(-1, 4, bins)[selected]
    )
    probabilities = _softmax_last(distributions)

    bin_values = np.arange(bins, dtype=np.float32)
    distances = np.sum(probabilities * bin_values, axis=-1)

    rows = selected // grid_w
    cols = selected % grid_w
    centers = np.stack((cols, rows), axis=1).astype(np.float32) + 0.5

    input_h, input_w = input_hw
    stride = np.array((input_w / grid_w, input_h / grid_h), dtype=np.float32)

    boxes = np.empty((selected.size, 4), dtype=np.float32)
    boxes[:, :2] = (centers - distances[:, :2]) * stride
    boxes[:, 2:] = (centers + distances[:, 2:]) * stride
    return boxes


def _branch_candidates(
    position: np.ndarray,
    class_scores: np.ndarray,
    objectness: np.ndarray | None,
    *,
    conf_thr: float,
    input_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """处理单个分支：置信度筛选 + DFL 解码。

    流程：
        1. 把 class_scores 重整为 (grid_h*grid_w, num_classes)
        2. 如果有 objectness，乘上去
        3. 对每个 grid cell 取最大类别的分数
        4. 筛掉低于 conf_thr 的候选
        5. 只对幸存者做 DFL 解码
    """
    class_scores = np.asarray(class_scores)
    if class_scores.ndim != 4 or class_scores.shape[0] != 1:
        raise ValueError(f"invalid class tensor shape: {class_scores.shape}")

    scores = class_scores[0].transpose(1, 2, 0).reshape(-1, class_scores.shape[1])

    if objectness is not None:
        obj = np.asarray(objectness)
        if obj.ndim != 4 or obj.shape[0] != 1:
            raise ValueError(f"invalid objectness tensor shape: {obj.shape}")
        scores = scores * obj[0].transpose(1, 2, 0).reshape(-1, 1)

    class_ids = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(scores.shape[0]), class_ids]

    selected = np.flatnonzero(best_scores >= conf_thr)
    if selected.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    boxes = _decode_selected_boxes(position, selected, input_hw=input_hw)
    return boxes, class_ids[selected], best_scores[selected].astype(np.float32, copy=False)


def decode_outputs_fast(
    outputs: Sequence[np.ndarray],
    *,
    conf_thr: float,
    iou_thr: float,
    class_names: Sequence[str],
    letterbox_meta: LetterboxMeta,
    input_hw: tuple[int, int],
) -> list[Detection]:
    """解码 YOLOv8（6 输出）或 YOLOE（9 输出）的 RKNN 检测头。

    整体流程：
        1. 按输出数量判断是 6 输出（每分支 2 张量）还是 9 输出（每分支 3 张量）
        2. 对 3 个分支分别做候选提取
        3. 合并所有分支的候选框
        4. 把坐标从 letterbox 空间还原回原图
        5. 按类别做 NMS
        6. 构造 Detection 列表
    """
    num_outputs = len(outputs)
    if num_outputs not in (6, 9):
        shapes = [tuple(np.asarray(output).shape) for output in outputs]
        raise RuntimeError(f"Unsupported RKNN output count={num_outputs}, shapes={shapes}")

    per_branch = num_outputs // 3
    boxes_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []

    for branch in range(3):
        base = branch * per_branch
        boxes, class_ids, scores = _branch_candidates(
            outputs[base],
            outputs[base + 1],
            outputs[base + 2] if per_branch == 3 else None,
            conf_thr=conf_thr,
            input_hw=input_hw,
        )
        if boxes.size:
            boxes_parts.append(boxes)
            class_parts.append(class_ids)
            score_parts.append(scores)

    if not boxes_parts:
        return []

    boxes_xyxy = unletterbox_boxes(np.concatenate(boxes_parts), letterbox_meta)
    class_ids = np.concatenate(class_parts)
    best_scores = np.concatenate(score_parts)

    keep = per_class_nms(boxes_xyxy, class_ids, best_scores, iou_thr)
    if keep.size == 0:
        return []

    return detections_from_arrays(
        boxes_xyxy[keep], class_ids[keep], best_scores[keep], list(class_names)
    )


# ─── 后处理分发 ───────────────────────────────────────────

def _decode_boxes_scores(
    outputs: Sequence[np.ndarray],
    *,
    class_names: list[str],
    letterbox_meta: LetterboxMeta,
    conf_thr: float,
    iou_thr: float,
) -> list[Detection]:
    """已解码格式 → 置信度筛选 + NMS + 坐标还原。

    适用于模型已经内置了 DFL 解码、直接输出检测框的情况。

    支持的输出形状：
    - [1, N, 6] 或 [N, 6]    列含义：x1,y1,x2,y2,score,class_id
    - [1, N, 5+cls] 或 [N, 5+cls]  列含义：x1,y1,x2,y2,score_0,...

    注意：当前实现不支持 boxes 和 class_scores 分开两个张量的情况
    （如 (1,18400,4) + (1,10,8400)），那需要单独的分支处理。
    """
    arr = np.asarray(outputs[0])
    arr = arr.reshape(-1, arr.shape[-1])

    if arr.shape[-1] == 6:
        boxes = arr[:, :4]
        scores = arr[:, 4]
        class_ids = arr[:, 5].astype(np.int64)
    elif arr.shape[-1] >= 6:
        boxes = arr[:, :4]
        class_scores = arr[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    else:
        raise ValueError(f"boxes_scores 输出列数不足: {arr.shape[-1]}，需要 >= 6")

    mask = scores >= conf_thr
    if not mask.any():
        return []

    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    boxes = unletterbox_boxes(boxes, letterbox_meta)
    keep = per_class_nms(boxes, class_ids, scores, iou_thr)
    if keep.size == 0:
        return []

    return detections_from_arrays(boxes[keep], class_ids[keep], scores[keep], class_names)


def _decode_decoded_boxes(
    outputs: Sequence[np.ndarray],
    *,
    class_names: list[str],
    letterbox_meta: LetterboxMeta,
    conf_thr: float,
    iou_thr: float,
) -> list[Detection]:
    """框和分数分开两个张量输出 → 置信度筛选 + NMS + 坐标还原。

    适用于模型已经输出解码后的检测框，但 boxes 和 class_scores 是分开的情况。

    支持的输出组合：
    - output[0]: [N, 4]          框坐标 (xyxy)
    - output[1]: [N, C]          class_scores（每类分数）
    - output[1]: [N]             class_ids（已 argmax）
    """
    boxes = np.asarray(outputs[0]).reshape(-1, 4)
    scores_data = np.asarray(outputs[1])

    # PP-YOLOE 输出 class_scores 为 (1, C, N) 格式，需转置为 (N, C)
    if scores_data.ndim == 3 and scores_data.shape[1] < scores_data.shape[2]:
        scores_data = scores_data.transpose(0, 2, 1)
    if scores_data.ndim >= 2:
        scores_data = scores_data.reshape(-1, scores_data.shape[-1])

    if scores_data.ndim == 1:
        # class_ids 已确定
        class_ids = scores_data.astype(np.int64)
        scores = np.ones_like(class_ids, dtype=np.float32)
    elif scores_data.ndim >= 2:
        # class_scores [N, C]
        class_ids = np.argmax(scores_data, axis=1)
        scores = scores_data[np.arange(scores_data.shape[0]), class_ids]
    else:
        raise ValueError(f"unexpected scores shape: {scores_data.shape}")

    mask = scores >= conf_thr
    if not mask.any():
        return []

    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    boxes = unletterbox_boxes(boxes, letterbox_meta)
    keep = per_class_nms(boxes, class_ids, scores, iou_thr)
    if keep.size == 0:
        return []

    return detections_from_arrays(boxes[keep], class_ids[keep], scores[keep], class_names)


def _postprocess(
    outputs: Sequence[np.ndarray],
    config: dict,
    letterbox_meta: LetterboxMeta,
    *,
    conf_thr: float,
    iou_thr: float,
    input_hw: tuple[int, int],
) -> list[Detection]:
    """按 config["output_format"] 选择后处理分支。

    - raw_dfl        → decode_outputs_fast（DFL 解码 + NMS）
    - boxes_scores   → _decode_boxes_scores（单输出已解码格式）
    - decoded_boxes  → _decode_decoded_boxes（多输出已解码格式）
    """
    fmt = config["output_format"]

    if fmt == "raw_dfl":
        return decode_outputs_fast(
            outputs,
            conf_thr=conf_thr,
            iou_thr=iou_thr,
            class_names=config["label_list"],
            letterbox_meta=letterbox_meta,
            input_hw=input_hw,
        )

    if fmt == "boxes_scores":
        return _decode_boxes_scores(
            outputs,
            class_names=config["label_list"],
            letterbox_meta=letterbox_meta,
            conf_thr=conf_thr,
            iou_thr=iou_thr,
        )

    if fmt == "decoded_boxes":
        return _decode_decoded_boxes(
            outputs,
            class_names=config["label_list"],
            letterbox_meta=letterbox_meta,
            conf_thr=conf_thr,
            iou_thr=iou_thr,
        )

    raise ValueError(f"unsupported output_format: {fmt}")


# ─── 单实例检测器 ─────────────────────────────────────────

class RKNNDetector:
    """一个模型实例，固定绑定一个 NPU core。

    生命周期
    --------
        __init__ : 读 {model}.rknn.json → 加载 .rknn → init_runtime(core_id)
        infer    : 每帧调用，返回 list[Detection]
        release   : 退出前释放 NPU runtime

    配置驱动
    --------
    所有模型相关的行为都由 {model}.rknn.json 决定：
    - label_list     : 类别列表（必须填齐，否则 load_model_config 拒绝启动）
    - input_mode     : 预处理方式（image / image_scale_factor）
    - input_size     : 模型输入 (w, h)（可选，由探针自动写入；未配置时用构造参数兜底）
    - output_format  : 后处理分支（raw_dfl / boxes_scores）
    """

    def __init__(
        self,
        model_path: str,
        core_id: int,
        *,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        conf_thr: float = DEFAULT_CONF_THR,
        iou_thr: float = DEFAULT_IOU_THR,
    ) -> None:
        """
        参数
        ----
        model_path : str
            .rknn 模型文件路径。同目录下必须有 {model}.rknn.json 配置文件。
        core_id : int
            NPU core 编号（0/1/2）。
        input_size : tuple[int, int]
            模型输入 (w, h) 兜底值。若 {model}.rknn.json 配置了
            ``input_size``，则以配置为准（由探针自动写入）。
        conf_thr : float
            置信度阈值，低于此分数的框丢弃。
        iou_thr : float
            NMS IoU 阈值。
        """
        self.model_path = model_path

        # 从 {model}.rknn.json 加载配置
        self.config = load_model_config(model_path)
        self.class_names = self.config["label_list"]
        self.input_mode = self.config["input_mode"]
        self.output_format = self.config["output_format"]

        # input_size 优先取配置（由探针写入），其次取构造参数兜底
        cfg_input_size = self.config.get("input_size")
        if cfg_input_size is not None:
            self.input_size = tuple(cfg_input_size)
        else:
            self.input_size = input_size
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self._released = False

        # core_id → core_mask
        from rknnlite.api import RKNNLite
        core_masks = {0: RKNNLite.NPU_CORE_0, 1: RKNNLite.NPU_CORE_1, 2: RKNNLite.NPU_CORE_2}
        if core_id not in core_masks:
            raise ValueError(f"NPU core must be 0, 1 or 2; got {core_id}")
        core_mask = core_masks[core_id]

        # 加载 RKNN 模型并初始化 NPU runtime
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: ret={ret}, path={model_path}")
        ret = self.rknn_lite.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: ret={ret}")
        print(
            f"[RKNNDetector] loaded: {model_path} "
            f"(core={core_id}, format={self.output_format}, "
            f"input_size={self.input_size}, classes={self.class_names})"
        )

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        """对单帧做目标检测推理，返回 Detection 列表。

        失败时抛异常，由调用方决定是否沿用上一帧结果。
        """
        if self._released:
            raise RuntimeError("RKNNDetector has been released")
        if frame_bgr is None:
            return []
        inputs, meta = _preprocess(frame_bgr, self.input_size, self.input_mode)
        # RKNN inference 默认按 uint8 解析；float32 输入须显式指定 data_type。
        # 双输入模型（image_scale_factor）需要为每个输入指定对应的 data_type。
        data_type = ['float32' if a.dtype == np.float32 else None for a in inputs]
        if len(data_type) == 1:
            data_type = data_type[0]
        outputs = self.rknn_lite.inference(inputs=inputs, data_type=data_type)
        return _postprocess(
            outputs,
            self.config,
            meta,
            conf_thr=self.conf_thr,
            iou_thr=self.iou_thr,
            input_hw=self.input_size[::-1],  # (w, h) → (h, w)
        )

    def release(self) -> None:
        """释放 NPU runtime。可重复调用。"""
        if self._released:
            return
        self.rknn_lite.release()
        self._released = True


# ─── 多实例池 ─────────────────────────────────────────────

class MultiCoreDetector:
    """同一模型的多个 RKNN 实例，每实例固定占用一个 NPU core。

    用于检测吞吐不足时并行推理：同一帧同时提交给多个 core，
    谁先空闲谁处理。无空闲实例时 submit() 会阻塞等待。

    用法
    ----
        pool = MultiCoreDetector("model.rknn", core_ids=(1, 2))
        future = pool.submit(frame)       # 异步
        detections = future.result()     # 取结果
        # 或同步便利接口
        detections = pool.infer(frame)
        pool.release()
    """

    def __init__(
        self,
        model_path: str,
        core_ids: tuple[int, ...],
        *,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        conf_thr: float = DEFAULT_CONF_THR,
        iou_thr: float = DEFAULT_IOU_THR,
    ) -> None:
        if not core_ids:
            raise ValueError("core_ids must not be empty")
        self._detectors = [
            RKNNDetector(model_path, core_id, input_size=input_size, conf_thr=conf_thr, iou_thr=iou_thr)
            for core_id in core_ids
        ]
        self._idle: Queue[RKNNDetector] = Queue()
        for detector in self._detectors:
            self._idle.put(detector)
        self._executor = ThreadPoolExecutor(max_workers=len(self._detectors))

    def submit(self, frame_bgr: np.ndarray, timeout: float = 5.0) -> Future[list[Detection]]:
        """异步提交一帧，返回 Future。

        无空闲 RKNN 实例时阻塞等待最多 timeout 秒；
        超时则抛 TimeoutError，避免永久死锁。
        """
        try:
            detector = self._idle.get(timeout=timeout)
        except Empty:
            raise TimeoutError(f"No idle RKNN detector within {timeout}s")
        frame_copy = frame_bgr.copy()  # 各线程独立副本，避免竞争

        def _run() -> list[Detection]:
            try:
                return detector.infer(frame_copy)
            finally:
                self._idle.put(detector)  # 归还实例

        return self._executor.submit(_run)

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        """同步便利接口：提交并等待结果。

        超时（所有 core 繁忙超过 submit timeout）时返回空列表而非抛异常，
        便于主循环继续运行。
        """
        try:
            return self.submit(frame_bgr).result()
        except TimeoutError:
            print("[MultiCoreDetector] submit timeout, all cores busy")
            return []

    def release(self) -> None:
        """关闭线程池并释放所有 RKNN 实例。"""
        self._executor.shutdown(wait=True)
        for detector in self._detectors:
            detector.release()


"""
frame_bgr (原图, e.g. 640x480)
│
▼ _preprocess()
letterbox(640x640) → RGB → tensor(s)
│
▼ rknn_lite.inference()
│
outputs (RKNN 原始输出，模型不同输出不同)
│
▼ _postprocess() ──── 按 config["output_format"] 分发 ─────────────┐
│                                                                    │
├──── "raw_dfl" ──────────────────────────────────────────────────┐  │
│  6个张量(YOLOv8) 或 9个张量(YOLOE)                              │  │
│  每个分支 2 个(position + class_scores) 或 3 个(+objectness)     │  │
│                                                                  │  │
│  decode_outputs_fast():                                          │  │
│    _branch_candidates() × 3分支 → DFL解码 + conf筛选             │  │
│    合并 → unletterbox → per_class_nms → Detection               │  │
│                                                                  │  │
│  适用于：模型输出原始分布，需要自己做 DFL 解码                     │  │
├──── "boxes_scores" ────────────────────────────────────────────┐  │
│  1个张量，形状 [1, N, 6] 或 [N, 6] 或 [1, N, 5+C]             │  │
│  列: x1 y1 x2 y2 score class_id                                │  │
│                                                                  │  │
│  _decode_boxes_scores():                                         │  │
│    直接取 boxes/scores → conf筛选                                │  │
│    → unletterbox → per_class_nms → Detection                    │  │
│                                                                  │  │
│  适用于：模型已内置解码，直接输出检测框                            │  │
├──── "decoded_boxes" ───────────────────────────────────────────┐  │
│  2+个张量：                                                     │  │
│    output[0]: [N, 4]  框坐标 (xyxy)                             │  │
│    output[1]: [N, C]  class_scores  或  [N]  class_ids          │  │
│                                                                  │  │
│  _decode_decoded_boxes():                                        │  │
│    直接取 boxes → argmax(scores) → class_id                      │  │
│    → conf筛选 → unletterbox → per_class_nms → Detection         │  │
│                                                                  │  │
│  适用于：模型输出框和分数分开，但已经解码好了                      │  │
└──────────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              list[Detection] (原图坐标)
"""