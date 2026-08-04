# Copyright (c) 2026 清影/123twtd
"""RKNN 检测模型输出探针（增强版）。

对指定 .rknn 模型做推理，自动检测输入预处理模式、输出格式、类别数，
并将结果写入 {model}.rknn.json。

检测方式
--------
1. 通过 RKNN Runtime API (get_in_out_num / get_tensor_attr) 查询模型的
   输入数量、名称和形状，**不依赖模型名硬编码**。
2. 根据查询结果自动构建正确的输入列表：
   - 单输入 (image)          → 传入 uint8 图像
   - 双输入 (image+scale_factor) → 传入 uint8 图像 + scale_factor 张量
3. 对输出张量的形状和值分布分析，推断 output_format 和 label_count。

用法
----
    python dev_tools/probes/rknn_det_model_probe.py --model xsmartcar/mymodel/det_model/model.rknn

可选参数
    --image PATH      测试图片路径（默认 probes/10000 (95).png）
    --input-w W       模型输入宽度（省略时自动从模型 dims 探测）
    --input-h H       模型输入高度（省略时自动从模型 dims 探测）
    --core 0|1|2      NPU 核心（默认 1）
    --metadata        在 JSON 中写入 _metadata 元数据字段（默认不写入）
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


# ─── 常量 ───────────────────────────────────────────────────

DEFAULT_IMAGE = Path(__file__).resolve().parent / "10000 (95).png"

# 输出值"合理"的 absmean 范围下限/上限
_REASONABLE_ABSMEAN_MIN = 1e-6
_REASONABLE_ABSMEAN_MAX = 1e6


# ─── 预处理 ────────────────────────────────────────────────

def letterbox(img_bgr: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    """等比例缩放加边填充，返回 (1, H, W, 3) uint8 RGB。"""
    input_w, input_h = input_size
    orig_h, orig_w = img_bgr.shape[:2]
    scale = min(input_w / max(orig_w, 1), input_h / max(orig_h, 1))
    resized_w = max(1, int(round(orig_w * scale)))
    resized_h = max(1, int(round(orig_h * scale)))
    resized = cv2.resize(img_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    pad_x = (input_w - resized_w) // 2
    pad_y = (input_h - resized_h) // 2
    canvas[pad_y: pad_y + resized_h, pad_x: pad_x + resized_w] = resized

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img_rgb, axis=0)


# ─── 模型信息查询 ──────────────────────────────────────────

def _input_size_from_dims(dims: list[int]) -> tuple[int, int] | None:
    """从输入张量 dims 推断模型输入 (w, h)。

    支持 NHWC [1,H,W,3] / NCHW [1,3,H,W] / HWC [H,W,3]。
    通道维取值为 3 的那个；无法确定时按 NHWC 兜底。
    """
    if len(dims) == 4:
        if dims[1] == 3:        # NCHW
            h, w = dims[2], dims[3]
        elif dims[3] == 3:      # NHWC
            h, w = dims[1], dims[2]
        else:                   # 兜底按 NHWC
            h, w = dims[1], dims[2]
        return (int(w), int(h))
    if len(dims) == 3:          # HWC
        return (int(dims[1]), int(dims[0]))
    return None


def _query_model_info(
    model_path: str,
    core: int,
) -> dict:
    """加载模型并查询输入/输出信息，返回后释放。

    返回
    ----
    dict:
        n_inputs      : int          输入数量
        n_outputs     : int          输出数量
        inputs        : list[dict]   每个输入的属性
        output_shapes : list[list]   每个输出的 shape
    """
    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn 失败: ret={ret}, path={model_path}")

    core_masks = {0: RKNNLite.NPU_CORE_0, 1: RKNNLite.NPU_CORE_1, 2: RKNNLite.NPU_CORE_2}
    ret = rknn.init_runtime(core_mask=core_masks[core])
    if ret != 0:
        raise RuntimeError(f"init_runtime 失败: ret={ret}")

    rt = rknn.rknn_runtime

    # 查询输入/输出数量
    n_inputs, n_outputs = rt.get_in_out_num()

    # 查询每个输入的属性
    inputs_info = []
    for i in range(n_inputs):
        attr = rt.get_tensor_attr(i)
        dims = [attr.dims[j] for j in range(attr.n_dims)]
        name = attr.name.decode("utf-8") if isinstance(attr.name, bytes) else str(attr.name)
        inputs_info.append({
            "index": attr.index,
            "name": name,
            "dims": dims,
            "n_elems": attr.n_elems,
            "fmt": attr.fmt,
            "qnt_type": attr.qnt_type,
            "type": attr.type,
            "zp": attr.zp,
            "scale": attr.scale,
            "pass_through": attr.pass_through,
        })

    # 查询每个输出的 shape
    output_shapes = []
    for i in range(n_outputs):
        shape = rt.get_output_shape(i)
        output_shapes.append(shape)

    rknn.release()

    return {
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
        "inputs": inputs_info,
        "output_shapes": output_shapes,
    }


# ─── 输出分析 ──────────────────────────────────────────────

def _analyze_outputs(outputs: list[np.ndarray]) -> list[dict]:
    """返回每个输出的统计字典列表。"""
    info = []
    for out in outputs:
        arr = np.asarray(out)
        s = {
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(arr.min()) if arr.size else float("nan"),
            "max": float(arr.max()) if arr.size else float("nan"),
            "mean": float(np.mean(arr)) if arr.size else float("nan"),
            "std": float(np.std(arr)) if arr.size else float("nan"),
            "absmean": float(np.mean(np.abs(arr))) if arr.size else float("nan"),
        }
        info.append(s)
    return info


def _outputs_reasonable(stats_list: list[dict]) -> bool:
    """判断输出统计是否像有效的检测模型结果。"""
    for st in stats_list:
        if not (np.isfinite(st["min"]) and np.isfinite(st["max"])):
            return False
        if st["absmean"] < _REASONABLE_ABSMEAN_MIN or st["absmean"] > _REASONABLE_ABSMEAN_MAX:
            return False
        if st["std"] < 1e-12 and st["absmean"] > 1e-6:
            return False
    return True


# ─── 格式推断 ──────────────────────────────────────────────

def _guess_output_format(
    outputs: list[np.ndarray],
) -> tuple[str, str]:
    """推断 (output_format, subtype)。

    output_format 取值（与 config.py 中的校验枚举一致）：
        raw_dfl       — DFL 检测头，需要进一步解码
        boxes_scores  — 已解码格式，单输出含框+分数
        decoded_boxes — 框和分数分开输出

    subtype 为更详细的描述，仅供探针输出参考。
    """
    num = len(outputs)

    # ── DFL 格式 ──────────────────────────────────────────
    if num == 6:
        return "raw_dfl", "yolov8_6out"
    if num == 9:
        return "raw_dfl", "yoloe_9out"

    # 检查是否有 4D 张量且通道为 4 的倍数 → DFL 特征
    for o in outputs:
        arr = np.asarray(o)
        if arr.ndim == 4 and arr.shape[1] >= 16 and arr.shape[1] % 4 == 0:
            return "raw_dfl", f"{num}out_dfl_like"

    # ── 单输出格式 ───────────────────────────────────────
    if num == 1:
        arr = np.asarray(outputs[0])
        last_dim = arr.shape[-1] if arr.ndim >= 2 else 0
        if last_dim == 6:
            return "boxes_scores", "xyxy_score_cid"
        if last_dim > 6:
            return "boxes_scores", f"xyxy_{last_dim - 4}cls"
        if last_dim == 4:
            return "decoded_boxes", "xyxy"
        return "boxes_scores", f"dim{last_dim}"

    # ── 双输出格式 ───────────────────────────────────────
    if num == 2:
        arr0 = np.asarray(outputs[0])
        arr1 = np.asarray(outputs[1])
        if arr0.ndim >= 2 and arr0.shape[-1] == 4:
            return "decoded_boxes", "xyxy_scores"
        if arr1.ndim >= 2 and arr1.shape[-1] == 4:
            return "decoded_boxes", "scores_xyxy"
        return "boxes_scores", f"{num}out_other"

    # ── 回退 ─────────────────────────────────────────────
    return "boxes_scores", f"{num}out_unknown"


def _guess_label_count(
    outputs: list[np.ndarray],
    output_format: str,
    subtype: str,
) -> int:
    """从输出张量推断类别数。"""
    if output_format == "raw_dfl":
        if len(outputs) >= 2:
            arr = np.asarray(outputs[1])
            if arr.ndim >= 3:
                return int(arr.shape[-3]) if arr.ndim == 4 else int(arr.shape[1])
        return 0

    if output_format == "boxes_scores":
        arr = np.asarray(outputs[0])
        if arr.ndim >= 2:
            cols = arr.shape[-1]
            if subtype.startswith("xyxy_"):
                if "cid" in subtype:
                    return 1
                return max(1, cols - 4)
            if cols >= 6:
                return cols - 5
        return 0

    if output_format == "decoded_boxes" and len(outputs) >= 2:
        # 判断哪个输出是 boxes、哪个是 scores
        arr0 = np.asarray(outputs[0])
        arr1 = np.asarray(outputs[1])

        if arr0.shape[-1] == 4:
            # arr0 = boxes, arr1 = scores
            scores = arr1
        elif arr1.shape[-1] == 4:
            # arr1 = boxes, arr0 = scores
            scores = arr0
        else:
            return 0

        # scores 形状可能是 (1, C, N) 或 (1, N, C) 或 (N, C)
        # 类别数 = 维度中明显较小的那个（C << N）
        if scores.ndim >= 2:
            # 取最后两个维度中较小的作为类别数
            d1, d2 = scores.shape[-2], scores.shape[-1]
            if d1 < d2:
                return int(d1)
            return int(d2)
    return 0


# ─── 输入构建与推理 ───────────────────────────────────────

def _build_inputs(
    model_info: dict,
    img: np.ndarray,
    input_size: tuple[int, int],
) -> list[np.ndarray]:
    """根据模型输入信息构建推理输入列表。

    策略
    ----
    - 单输入（name 含 "image" 或仅有 1 个输入）→ uint8 图像
    - 双输入（input[0]=image, input[1]=scale_factor）→ 图像 + scale_factor
    """
    inputs_info = model_info["inputs"]

    # 构建主图像输入
    img_input = letterbox(img, input_size)  # (1, H, W, 3) uint8

    if len(inputs_info) == 1:
        return [img_input]

    # 双输入：查找 scale_factor 输入
    input_list = [img_input]
    for info in inputs_info[1:]:
        name = info["name"].lower()
        if "scale" in name or "factor" in name:
            # scale_factor: [1, 2]，值为 [x_scale, y_scale]
            # 量化为 uint8：scale = value / 256
            h, w = img.shape[:2]
            input_w, input_h = input_size
            x_scale = w / input_w
            y_scale = h / input_h
            # RKNN 量化方式: real_value = (uint8_value - zp) * scale
            # zp=0, scale=1.0 → uint8 值 = real_value
            # 但 scale_factor 通常是 [1.0, 1.0]（letterbox 不需要额外缩放）
            sf = np.array([[x_scale, y_scale]], dtype=np.float32)
            # 量化为 uint8：如果 qnt_type=2 且 zp=0, scale=1.0，则直接 cast
            if info["qnt_type"] == 2 and info["zp"] == 0 and info["scale"] == 1.0:
                sf_uint8 = np.clip(sf * 256, 0, 255).astype(np.uint8)
                input_list.append(sf_uint8)
            else:
                input_list.append(sf)
        else:
            # 未知第二输入，尝试传入零张量
            dims = info["dims"]
            zero_tensor = np.zeros(dims, dtype=np.uint8)
            input_list.append(zero_tensor)

    return input_list


def _infer_with_inputs(
    model_path: str,
    inputs: list[np.ndarray],
    core: int,
) -> list[np.ndarray]:
    """加载模型 → 传入指定输入列表 → 推理 → 释放 → 返回输出列表。"""
    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn 失败: ret={ret}, path={model_path}")

    core_masks = {0: RKNNLite.NPU_CORE_0, 1: RKNNLite.NPU_CORE_1, 2: RKNNLite.NPU_CORE_2}
    ret = rknn.init_runtime(core_mask=core_masks[core])
    if ret != 0:
        raise RuntimeError(f"init_runtime 失败: ret={ret}")

    try:
        outputs = rknn.inference(inputs=inputs)
    except Exception as e:
        rknn.release()
        raise RuntimeError(f"推理异常: {e}")

    if outputs is None:
        rknn.release()
        raise RuntimeError("推理返回 None")

    rknn.release()
    return outputs


# ─── input_mode 检测 ──────────────────────────────────────

def _detect_input_mode(
    model_info: dict,
    model_path: str,
    img: np.ndarray,
    input_size: tuple[int, int],
    core: int,
    *,
    verbose: bool = True,
) -> tuple[str, list[np.ndarray], dict]:
    """自动检测 input_mode。

    优先根据模型查询到的输入信息判断：
    - 单输入           → input_mode = "image"
    - 双输入含 scale_factor → input_mode = "image_scale_factor"

    如果查询信息不足以判断，则通过实际推理验证。
    """
    inputs_info = model_info["inputs"]
    n_inputs = model_info["n_inputs"]

    # ── 根据查询结果直接判断 ────────────────────────────────
    if n_inputs == 1:
        input_mode = "image"
    elif n_inputs == 2:
        # 检查是否有 scale_factor 输入
        has_scale = any(
            "scale" in info["name"].lower() or "factor" in info["name"].lower()
            for info in inputs_info
        )
        input_mode = "image_scale_factor" if has_scale else "image"
    else:
        input_mode = "image"

    if verbose:
        print(f"  [probe] 模型输入数量: {n_inputs}")
        for i, info in enumerate(inputs_info):
            print(f"    input[{i}]: name={info['name']}, dims={info['dims']}, qnt_type={info['qnt_type']}")
        print(f"  [probe] 推断 input_mode: {input_mode}")

    # ── 构建输入并推理 ──────────────────────────────────────
    inputs = _build_inputs(model_info, img, input_size)

    if verbose:
        for i, inp in enumerate(inputs):
            print(f"  [probe] 推理输入[{i}]: shape={inp.shape}, dtype={inp.dtype}")

    try:
        outputs = _infer_with_inputs(model_path, inputs, core)
        stats = _analyze_outputs(outputs)
        ok = _outputs_reasonable(stats)

        if verbose:
            if ok:
                print(f"  [probe] → 推理成功, 首输出 absmean={stats[0]['absmean']:.6g}")
            else:
                print(f"  [probe] → 推理成功但输出值分布异常")
                for i, s in enumerate(stats):
                    print(f"    output[{i}]: absmean={s['absmean']:.6g}, std={s['std']:.6g}")

        if not ok and input_mode == "image":
            # 尝试 image_scale_factor 模式（float32/255）
            if verbose:
                print("  [probe] 尝试 float32/255 模式 ...")
            img_input_f32 = letterbox(img, input_size).astype(np.float32) / 255.0
            inputs_f32 = [img_input_f32] + inputs[1:]  # 保留可能的 scale_factor
            try:
                outputs_f32 = _infer_with_inputs(model_path, inputs_f32, core)
                stats_f32 = _analyze_outputs(outputs_f32)
                ok_f32 = _outputs_reasonable(stats_f32)
                if ok_f32:
                    if verbose:
                        print(f"  [probe] → float32/255 输出合理, 切换为 image_scale_factor")
                    return "image_scale_factor", outputs_f32, {
                        "input_mode_trials": [
                            {"mode": "image", "reasonable": False},
                            {"mode": "image_scale_factor", "reasonable": True},
                        ],
                    }
            except RuntimeError:
                pass

        return input_mode, outputs, {
            "input_mode_trials": [{"mode": input_mode, "reasonable": ok}],
        }

    except RuntimeError as e:
        if verbose:
            print(f"  [probe] ✗ 推理失败: {e}")

        # 回退：如果是双输入模型但用了单输入构建，可能构建方式不对
        # 尝试最基础的 uint8 单输入
        if n_inputs > 1:
            if verbose:
                print("  [probe] 尝试回退：仅传入 uint8 单输入 ...")
            try:
                img_input = letterbox(img, input_size)
                outputs = _infer_with_inputs(model_path, [img_input], core)
                stats = _analyze_outputs(outputs)
                return "image", outputs, {
                    "input_mode_trials": [
                        {"mode": input_mode, "reasonable": False, "error": str(e)},
                        {"mode": "image_fallback", "reasonable": _outputs_reasonable(stats)},
                    ],
                }
            except RuntimeError as e2:
                if verbose:
                    print(f"  [probe] ✗ 回退也失败: {e2}")
                return input_mode, None, {
                    "input_mode_trials": [{"mode": "image_fallback", "reasonable": False, "error": str(e2)}],
                    "mode_ambiguous": True,
                }

        return input_mode, None, {
            "input_mode_trials": [{"mode": input_mode, "reasonable": False, "error": str(e)}],
            "mode_ambiguous": True,
        }


# ─── JSON 配置写入 ─────────────────────────────────────────

def update_model_config(
    config_path: Path,
    model_file: str,
    input_mode: str,
    output_format: str,
    label_count: int,
    input_size: tuple[int, int] | None = None,
    output_shapes: list[list[int]] | None = None,
    probe_details: dict | None = None,
    write_metadata: bool = False,
) -> dict:
    """生成或更新 {model}.rknn.json。

    保留已有 label_list 不覆盖；只更新结构字段。
    探针推断元数据仅在 ``write_metadata=True`` 时写入 ``_metadata`` 字段
    （不参与 config 校验），默认不写入以保持配置文件简洁。
    """
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}

    config["model_file"] = model_file
    config["input_mode"] = input_mode
    config["output_format"] = output_format
    config["label_count"] = label_count

    # 输入尺寸 [w, h]：用于运行时预处理，须与模型导出尺寸一致
    if input_size is not None:
        config["input_size"] = [int(input_size[0]), int(input_size[1])]
    else:
        config.pop("input_size", None)

    # 输出张量形状：仅供调试参考，不参与运行时校验
    if output_shapes is not None:
        config["output_shapes"] = [list(s) for s in output_shapes]
    else:
        config.pop("output_shapes", None)

    label_list = config.get("label_list")
    if not isinstance(label_list, list) or not label_list:
        config["label_list"] = None
    else:
        config["label_list"] = label_list

    # 移除旧版 _probe 字段（向后兼容）
    config.pop("_probe", None)

    if probe_details and write_metadata:
        config["_metadata"] = probe_details
    else:
        config.pop("_metadata", None)

    config.pop("status", None)

    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config


# ─── 主流程 ────────────────────────────────────────────────

def run_probe(
    model_path: str,
    image_path: str,
    input_size: tuple[int, int] | None,
    core: int,
    write_metadata: bool = False,
) -> None:
    """执行探针：查询模型 → 检测 input_mode → 推理 → 分析格式 → 写入 JSON。

    ``input_size`` 为 None 时自动从模型输入 dims 探测 (w, h)。
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    model_dir = Path(model_path).resolve().parent
    model_file = os.path.basename(model_path)
    config_path = model_dir / f"{model_file}.json"

    # ── 1. 查询模型输入输出信息 ─────────────────────────────
    print("━" * 50)
    print(f"模型: {model_path}")
    print(f"图片: {image_path}")
    print("━" * 50)

    print("\n[1/3] 查询模型信息 ...")
    model_info = _query_model_info(model_path, core)
    print(f"  输入数量: {model_info['n_inputs']}")
    for i, info in enumerate(model_info["inputs"]):
        print(f"    input[{i}]: name={info['name']}, dims={info['dims']}, qnt_type={info['qnt_type']}")
    print(f"  输出数量: {model_info['n_outputs']}")
    for i, shape in enumerate(model_info["output_shapes"]):
        print(f"    output[{i}]: shape={shape}")

    # ── 1b. 自动探测输入尺寸 ───────────────────────────────
    auto_size = None
    if model_info["inputs"]:
        auto_size = _input_size_from_dims(model_info["inputs"][0]["dims"])
    if input_size is None:
        input_size = auto_size
        if input_size is None:
            raise RuntimeError(
                "无法从模型输入 dims 推断 input_size，请用 --input-w/--input-h 显式指定"
            )
        print(f"\n  [probe] 自动探测 input_size: {input_size} (来自 input[0] dims)")
    else:
        print(f"\n  [probe] 使用指定 input_size: {input_size}")
        if auto_size is not None and auto_size != input_size:
            print(
                f"  [probe] ⚠ 指定尺寸 {input_size} 与模型 dims 推断 {auto_size} 不一致"
            )
    print(f"  输入尺寸: {input_size}")

    # ── 2. 检测 input_mode 并推理 ──────────────────────────
    print("\n[2/3] 检测输入模式并推理 ...")
    input_mode, outputs, probe_info = _detect_input_mode(
        model_info, model_path, img, input_size, core, verbose=True,
    )

    # ── 3. 分析输出格式 ────────────────────────────────────
    if outputs is None:
        print("\n[3/3] ✗ 推理失败，无法分析输出格式")
        print("━" * 50)
        print("⚠ 推理完全失败，请检查模型文件和输入是否匹配")
        return

    output_format, subtype = _guess_output_format(outputs)
    label_count = _guess_label_count(outputs, output_format, subtype)
    stats = _analyze_outputs(outputs)

    print(f"\n[3/3] 分析输出格式 ...")
    print(f"  output_format : {output_format}  ({subtype})")
    print(f"  input_mode    : {input_mode}")
    print(f"  label_count   : {label_count}")
    print()
    print(f"  输出张量 ({len(outputs)} 个):")

    for i, s in enumerate(stats):
        print(
            f"    [{i}] shape={list(s['shape'])}  dtype={s['dtype']}  "
            f"min={s['min']:.6g}  max={s['max']:.6g}  "
            f"mean={s['mean']:.6g}  std={s['std']:.6g}"
        )

    # ── 4. 写入 JSON ───────────────────────────────────────
    probe_details = {
        "_description": "由 rknn_det_model_probe 自动生成的模型元数据，仅供调试参考，不参与运行时配置",
        "_generated_by": "rknn_det_model_probe",
        "model_inputs": [
            {"name": info["name"], "dims": info["dims"], "qnt_type": info["qnt_type"]}
            for info in model_info["inputs"]
        ],
        "output_shapes": [list(s["shape"]) for s in stats],
        "output_dtype": stats[0]["dtype"],
        "output_subtype": subtype,
        "probe_input_shape": list(input_size),
    }
    if "input_mode_trials" in probe_info:
        probe_details["input_mode_trials"] = probe_info["input_mode_trials"]
    if probe_info.get("mode_ambiguous"):
        probe_details["input_mode_ambiguous"] = True

    config = update_model_config(
        config_path,
        model_file=model_file,
        input_mode=input_mode,
        output_format=output_format,
        label_count=label_count,
        input_size=input_size,
        output_shapes=[list(s["shape"]) for s in stats],
        probe_details=probe_details,
        write_metadata=write_metadata,
    )

    print(f"\n配置已写入: {config_path}")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("━" * 50)

    if config.get("label_list") is None:
        print(f"⚠ label_list 为空，请编辑 {config_path} 补全类别名称后再启动检测。")


def main() -> int:
    parser = argparse.ArgumentParser(description="RKNN 检测模型输出探针（增强版）")
    parser.add_argument("--model", required=True, help=".rknn 模型文件路径")
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help=f"测试图片路径（默认: {DEFAULT_IMAGE}）",
    )
    parser.add_argument(
        "--input-w",
        type=int,
        default=None,
        help="模型输入宽度（省略时自动从模型 dims 探测）",
    )
    parser.add_argument(
        "--input-h",
        type=int,
        default=None,
        help="模型输入高度（省略时自动从模型 dims 探测）",
    )
    parser.add_argument("--core", type=int, choices=(0, 1, 2), default=1, help="NPU 核心编号")
    parser.add_argument(
        "--metadata",
        action="store_true",
        default=False,
        help="在 JSON 中写入 _metadata 元数据字段（默认不写入）",
    )
    args = parser.parse_args()

    # --input-w / --input-h 必须同时指定或同时省略（省略则自动探测）
    if args.input_w is not None or args.input_h is not None:
        if args.input_w is None or args.input_h is None:
            parser.error("--input-w 和 --input-h 必须同时指定，或同时省略以自动探测")
        input_size = (args.input_w, args.input_h)
    else:
        input_size = None  # 自动从模型 dims 探测

    run_probe(
        model_path=args.model,
        image_path=args.image,
        input_size=input_size,
        core=args.core,
        write_metadata=args.metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
