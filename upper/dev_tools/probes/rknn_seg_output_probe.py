# Copyright (c) 2026 清影/123twtd
"""单张图像 RKNN 语义分割原始输出探针。

本脚本有意与实时流水线分离。它绕过仓库正常的分割后处理，
以便查看 RKNN 模型输出的是类别 logits/概率，还是已经过 argmax 的类别图。

在香橙派上的典型用法：
    python dev_tools/probes/rknn_seg_output_probe.py --image_path frame.png --output_dir debug_probe

有用输出：
    summary.txt              原始输出形状、数据类型、最小/最大值及诊断信息。
    raw_output_0.npy         第一个 RKNN 输出张量，供后续检查。
    argmax_mask.png          由 argmax 生成的掩码，若存在类别通道。
    class1_prob.png          类别 1 的概率/分数可视化，若可用。
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMG_SIZE = (384, 384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="探针 RKNN 语义分割原始输出")
    parser.add_argument("--model_dir", type=str, default="mymodel/seg_model", help="模型目录，相对于 xsmartcar/")
    parser.add_argument("--image_path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output_dir", type=str, default="debug_probe", help="探针输出目录")
    parser.add_argument("--num_classes", type=int, default=2, help="分割类别数")
    parser.add_argument("--class_index", type=int, default=1, help="前景/车道类别索引")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.40, 0.50, 0.60, 0.70],
        help="存在类别通道时使用的概率/分数阈值",
    )
    parser.add_argument(
        "--core",
        choices=["0", "1", "2", "all", "default"],
        default="0",
        help="init_runtime 的 NPU 核心掩码",
    )
    return parser.parse_args()


def resolve_model_path(model_dir: str) -> Path:
    """按项目推理模块的方式解析模型目录。"""
    raw = Path(model_dir)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(REPO_ROOT / "xsmartcar" / model_dir)

    for candidate in candidates:
        if candidate.is_file() and candidate.suffix == ".rknn":
            return candidate
        if candidate.is_dir():
            files = sorted(glob.glob(str(candidate / "*.rknn")))
            if files:
                return Path(files[0])
    raise FileNotFoundError(f"No .rknn seg model found from model_dir={model_dir!r}")


def core_mask_from_arg(core: str):
    if core == "0":
        return RKNNLite.NPU_CORE_0
    if core == "1":
        return RKNNLite.NPU_CORE_1
    if core == "2":
        return RKNNLite.NPU_CORE_2
    if core == "all":
        return RKNNLite.NPU_CORE_0_1_2
    return None


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """图像预处理：BGR -> RGB，resize 到 384x384，增加 batch 维度。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(img_resized, axis=0)


def stable_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(np.sum(exp_x, axis=axis, keepdims=True), 1e-12)


def resize_mask(mask: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    """将掩码 resize 回原始尺寸。"""
    h, w = original_hw
    if mask.shape[:2] == (h, w):
        return mask.astype(np.uint8)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def resize_float_map(score: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    """将浮点分数图 resize 回原始尺寸。"""
    h, w = original_hw
    if score.shape[:2] == (h, w):
        return score.astype(np.float32)
    return cv2.resize(score.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def save_gray(path: Path, image: np.ndarray) -> None:
    """保存灰度图，先裁剪到 0-255 范围。"""
    cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))


def save_prob(path: Path, prob: np.ndarray) -> None:
    """保存概率图，若数值范围异常则先做 min-max 归一化。"""
    prob = prob.astype(np.float32)
    if prob.size == 0:
        return
    if float(np.nanmax(prob)) > 1.5 or float(np.nanmin(prob)) < -0.1:
        lo = float(np.nanmin(prob))
        hi = float(np.nanmax(prob))
        prob = (prob - lo) / max(hi - lo, 1e-6)
    save_gray(path, prob * 255.0)


def small_unique_report(arr: np.ndarray) -> str:
    """生成数组唯一值的精简报告，用于诊断输出类型。"""
    flat = arr.reshape(-1)
    if flat.size > 200000:
        flat = flat[:: max(1, flat.size // 200000)]
    unique = np.unique(flat)
    if unique.size <= 20:
        return np.array2string(unique, separator=", ")
    head = np.array2string(unique[:10], separator=", ")
    tail = np.array2string(unique[-10:], separator=", ")
    return f"{unique.size} unique values, head={head}, tail={tail}"


def detect_class_axis(arr: np.ndarray, num_classes: int) -> int | None:
    """检测类别通道所在的轴（0 或 -1），若无法检测则返回 None。"""
    if arr.ndim != 3:
        return None
    if arr.shape[0] == num_classes:
        return 0
    if arr.shape[-1] == num_classes:
        return -1
    if arr.shape[0] <= 32:
        return 0
    if arr.shape[-1] <= 32:
        return -1
    return None


def analyze_first_output(
    raw: np.ndarray,
    *,
    original_hw: tuple[int, int],
    num_classes: int,
    class_index: int,
    thresholds: list[float],
    output_dir: Path,
) -> list[str]:
    """分析第一个 RKNN 输出张量，生成诊断信息和可视化图。"""
    lines: list[str] = []
    arr = np.squeeze(np.asarray(raw))
    lines.append(f"压缩后形状: {arr.shape}")
    lines.append(f"压缩后数据类型: {arr.dtype}")

    class_axis = detect_class_axis(arr, num_classes)
    if class_axis is not None:
        lines.append(f"诊断: 输出在轴 {class_axis} 上有类别通道; argmax 未内置于模型输出中。")
        class_map = np.argmax(arr, axis=class_axis).astype(np.uint8)
        class_map = resize_mask(class_map, original_hw)
        cv2.imwrite(str(output_dir / "argmax_class_map.png"), class_map)
        cv2.imwrite(str(output_dir / "argmax_mask.png"), (class_map == class_index).astype(np.uint8) * 255)

        prob = stable_softmax(arr, axis=class_axis)
        if class_axis == 0:
            class_score = prob[class_index]
            raw_score = arr[class_index].astype(np.float32)
        else:
            class_score = prob[..., class_index]
            raw_score = arr[..., class_index].astype(np.float32)

        class_score = resize_float_map(class_score, original_hw)
        raw_score = resize_float_map(raw_score, original_hw)
        save_prob(output_dir / f"class{class_index}_prob.png", class_score)
        save_prob(output_dir / f"class{class_index}_raw_score.png", raw_score)

        for threshold in thresholds:
            mask = (class_score >= threshold).astype(np.uint8) * 255
            cv2.imwrite(str(output_dir / f"class{class_index}_thr_{threshold:.2f}.png"), mask)

        lines.append(f"类别{class_index}_概率最小值: {float(np.min(class_score)):.6f}")
        lines.append(f"类别{class_index}_概率最大值: {float(np.max(class_score)):.6f}")
        lines.append(f"类别{class_index}_概率平均值: {float(np.mean(class_score)):.6f}")
        return lines

    if arr.ndim == 2:
        lines.append("诊断: 压缩后输出为 2 维; 它可能是类别图、单通道概率图或单通道分数图。")
        lines.append(f"唯一值报告: {small_unique_report(arr)}")
        arr_f = arr.astype(np.float32)
        looks_like_class_map = np.all(np.isclose(arr_f, np.round(arr_f))) and np.nanmin(arr_f) >= 0 and np.nanmax(arr_f) < num_classes
        if looks_like_class_map:
            lines.append("类别图猜测: 是，数值看起来像已经过 argmax 的类别 id。")
            class_map = resize_mask(arr.astype(np.uint8), original_hw)
            cv2.imwrite(str(output_dir / "class_map.png"), class_map)
            cv2.imwrite(str(output_dir / "class_map_mask.png"), (class_map == class_index).astype(np.uint8) * 255)
        else:
            lines.append("类别图猜测: 否，将其作为单通道分数/概率图进行可视化。")
            score = resize_float_map(arr_f, original_hw)
            save_prob(output_dir / "single_channel_score.png", score)
            score_for_thr = score
            if float(np.nanmax(score_for_thr)) > 1.5 or float(np.nanmin(score_for_thr)) < -0.1:
                lo = float(np.nanmin(score_for_thr))
                hi = float(np.nanmax(score_for_thr))
                score_for_thr = (score_for_thr - lo) / max(hi - lo, 1e-6)
                lines.append("阈值说明: 分数在生成阈值掩码前已做 min-max 归一化。")
            for threshold in thresholds:
                mask = (score_for_thr >= threshold).astype(np.uint8) * 255
                cv2.imwrite(str(output_dir / f"single_channel_thr_{threshold:.2f}.png"), mask)
        return lines

    lines.append("诊断: 压缩后的维度不支持自动可视化; 请直接检查 raw_output_0.npy。")
    return lines


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image_path)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    model_path = resolve_model_path(args.model_dir)
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: ret={ret}")

    core_mask = core_mask_from_arg(args.core)
    if core_mask is None:
        ret = rknn.init_runtime()
    else:
        ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: ret={ret}")

    try:
        input_data = preprocess_image(img_bgr)
        outputs = rknn.inference(inputs=[input_data])
    finally:
        rknn.release()

    if outputs is None:
        raise RuntimeError("RKNN myinference returned None")
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    cv2.imwrite(str(output_dir / "input_original.png"), img_bgr)
    summary: list[str] = [
        f"模型路径: {model_path}",
        f"图像路径: {image_path}",
        f"图像形状_高宽通道: {img_bgr.shape}",
        f"输入张量形状: {input_data.shape}",
        f"输出数量: {len(outputs)}",
    ]

    for idx, output in enumerate(outputs):
        arr = np.asarray(output)
        np.save(output_dir / f"raw_output_{idx}.npy", arr)
        summary.extend(
            [
                "",
                f"输出[{idx}]_形状: {arr.shape}",
                f"输出[{idx}]_数据类型: {arr.dtype}",
                f"输出[{idx}]_最小值: {float(np.nanmin(arr)):.6f}",
                f"输出[{idx}]_最大值: {float(np.nanmax(arr)):.6f}",
                f"输出[{idx}]_平均值: {float(np.nanmean(arr)):.6f}",
            ]
        )
        if idx == 0:
            summary.extend(
                analyze_first_output(
                    arr,
                    original_hw=img_bgr.shape[:2],
                    num_classes=args.num_classes,
                    class_index=args.class_index,
                    thresholds=list(args.thresholds),
                    output_dir=output_dir,
                )
            )

    summary_text = "\n".join(summary) + "\n"
    (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"探针输出已保存到: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
