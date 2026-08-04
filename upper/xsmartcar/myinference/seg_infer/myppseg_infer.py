# Copyright (c) 2026 清影/123twtd
"""PP-Seg 语义分割「异步多 runtime」推理模块（香橙派 RKNN）。

本文件与 myppseg_sync_infer.py 功能等价，但底层走 rknnpool 线程池，可用多个
runtime 并发提高吞吐。对外仍提供 myinference(frame_bgr) -> (mask, ok) 的统一接口。

未对接说明（重要）
------------------
- 当前比赛主链 run_pipe.py 默认选用 **同步单 runtime** 的
  myppseg_sync_infer.PPSegSyncInfer，并不导入本模块。
- 本模块是为「需要更高吞吐」保留的备选实现，目前不接入主循环。
- TPEs>1 时为异步流水线：myinference 返回的是较早提交帧的结果；若要严格对齐
  「结果 ↔ 原帧」，请用 infer_with_frame()，它借助内部 frame 队列做配对。

数据流
------
    frame_bgr -> preprocess(512x512 RGB) -> rknnPoolExecutor(put/get)
        -> postprocess(最近邻缩放回原尺寸) -> 0/1 mask

与同步版的差异
--------------
- 同步版：每帧阻塞等 NPU，调用简单、行为可预期（当前默认）。
- 本异步版：吞吐更高，但存在帧延迟与对齐复杂度，联调时更难定位问题。
"""

from __future__ import annotations

import argparse
from collections import deque
import glob
import os
import time
from functools import partial
from typing import TYPE_CHECKING

import cv2
import numpy as np

try:
    from xsmartcar.myinference.rknnpool import rknnPoolExecutor
except ImportError:
    from rknnpool import rknnPoolExecutor

if TYPE_CHECKING:
    from xsmartcar.race_pipeline import FramePacket

# 可视化调色板：class 0 背景黑，class 1 车道白（RGB，叠图时再转 BGR）
SEG_COLORS = np.array(
    [
        [0, 0, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)

# 模型输入分辨率，须与 .rknn 导出时一致
# IMG_SIZE = (384, 384)  # 须与 .rknn 导出时的 input size 一致（当前模型 ppseg_384_*）
IMG_SIZE = (384,384)


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """BGR 原图 -> NPU 输入张量 (1, H, W, 3) uint8 RGB。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(img_resized, axis=0)


def postprocess_segmentation(output, original_size: tuple[int, int]) -> np.ndarray:
    """RKNN 原始输出 -> 与原图同尺寸的类别图（最近邻放大，保持 0/1 边界）。"""
    seg_output = output[0] if isinstance(output, list) else output
    seg_map = seg_output[0]

    orig_h, orig_w = original_size
    seg_map_resized = cv2.resize(
        seg_map.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)
    return seg_map_resized


def _prepare_seg_map(seg_map: np.ndarray) -> np.ndarray:
    """把各种形状的 RKNN 输出统一为 2-D 二值 mask：0=背景，1=前景。"""
    seg_map = np.asarray(seg_map)
    seg_map = np.squeeze(seg_map)

    if seg_map.ndim == 3:
        # 选最可能存放类别 logits 的轴做 argmax（通道在前或在后）
        if seg_map.shape[0] <= 32:
            seg_map = np.argmax(seg_map, axis=0)
        elif seg_map.shape[-1] <= 32:
            seg_map = np.argmax(seg_map, axis=-1)
        else:
            seg_map = seg_map[..., 0]

    if seg_map.ndim != 2:
        raise ValueError(f"Expected a 2-D seg_map after normalization, got shape={seg_map.shape}")

    # 当前 pipeline 只需要二值赛道 mask
    seg_map = (seg_map > 0).astype(np.uint8)
    return seg_map


def colorize_segmentation(seg_map: np.ndarray) -> np.ndarray:
    """二值 mask -> RGB 伪彩色图，仅用于可视化。"""
    seg_map = _prepare_seg_map(seg_map)
    h, w = seg_map.shape
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(SEG_COLORS):
        colored_seg[seg_map == class_idx] = color
    return colored_seg


def blend_images(original_img: np.ndarray, colored_seg: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """原图与伪彩色分割图按 alpha 混合，便于调试对齐。"""
    if original_img.shape[:2] != colored_seg.shape[:2]:
        colored_seg = cv2.resize(
            colored_seg,
            (original_img.shape[1], original_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    colored_seg_bgr = cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(original_img, 1 - alpha, colored_seg_bgr, alpha, 0)


def seg_visualization(seg_map: np.ndarray, img_bgr: np.ndarray | None = None, blend_alpha: float | None = None) -> np.ndarray:
    """生成可 imwrite/imshow 的 BGR 预览图。"""
    colored_seg = colorize_segmentation(seg_map)
    if blend_alpha is not None and 0 < blend_alpha < 1 and img_bgr is not None:
        return blend_images(img_bgr, colored_seg, alpha=blend_alpha)
    return cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR)


def myFunc(rknn_lite, img_bgr: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """线程池中的单帧 worker：在某个 runtime 上跑一帧推理（供 rknnPoolExecutor 调度）。"""
    try:
        original_size = img_bgr.shape[:2]
        input_data = preprocess_image(img_bgr)
        outputs = rknn_lite.inference(inputs=[input_data])
        seg_map = postprocess_segmentation(outputs, original_size)
        seg_map = _prepare_seg_map(seg_map)
        return seg_map, True
    except Exception as e:
        print(f"Error in myFunc: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def get_current_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


class PPSegInfer:
    """异步多 runtime 的 PP-Seg 封装（备选实现，当前不接入 run_pipe 主链）。

    与 PPSegSyncInfer 接口一致（myinference/visualize/normalize_mask/release），
    但底层用 rknnPoolExecutor 做流水线。TPEs=1 时退化为近似同步。
    """

    def __init__(self, model_dir: str = "../../mymodel/seg_model", TPEs: int = 1, blend_alpha: float | None = None) -> None:
        # model_dir 相对 xsmartcar/ 模块目录，而非当前工作目录
        model_dir = os.path.join(get_current_dir(), model_dir)
        model_path = self.get_model_path(model_dir)

        self.TPEs = TPEs
        self.blend_alpha = blend_alpha
        self._pool_initialized = False
        self._released = False
        self._frame_queue = deque()
        self._packet_queue = deque()

        infer_func = partial(myFunc)
        self.rknn_pool = rknnPoolExecutor(
            rknnModel=model_path,
            TPEs=self.TPEs,
            func=infer_func,
        )
        print(f"PP-Seg model loaded: {model_path}")

    def get_model_path(self, model_dir: str) -> str:
        matches = sorted(glob.glob(os.path.join(model_dir, "*.rknn")))
        if not matches:
            raise FileNotFoundError(f"No .rknn found in {model_dir}")
        return matches[0]

    def _pool_init(self, img_bgr: np.ndarray, packet: FramePacket | None = None) -> None:
        # 预灌 TPEs+1 帧，填满流水线，使后续每次 put 都能立即 get 到结果
        for _ in range(self.TPEs + 1):
            self.rknn_pool.put(img_bgr)
            self._frame_queue.append(img_bgr.copy())
            self._packet_queue.append(packet)
        self._pool_initialized = True

    def infer(self, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """标准分割接口：传入 BGR 帧，返回 (mask, ok)。"""
        _, seg_map, ok = self.infer_with_frame(frame_bgr)
        return seg_map, ok

    def infer_with_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
        """返回与本次结果配对的「原帧」及其 mask。

        TPEs>1 时线程池是异步流水线：get() 拿到的是更早提交那一帧的结果，
        不一定是本次传入的 frame。因此用内部 _frame_queue 做「结果 ↔ 原帧」对齐。
        """
        if self._released:
            raise RuntimeError("PPSegInfer has been released")

        if frame_bgr is None:
            return None, None, False

        if not self._pool_initialized:
            self._pool_init(frame_bgr)

        self.rknn_pool.put(frame_bgr)
        self._frame_queue.append(frame_bgr.copy())
        func_result, pool_success = self.rknn_pool.get()
        if not pool_success or func_result is None:
            return None, None, False

        result_frame = self._frame_queue.popleft() if self._frame_queue else None
        if self._packet_queue:
            self._packet_queue.popleft()

        if isinstance(func_result, tuple) and len(func_result) == 2:
            seg_map, ok = func_result
            return result_frame, seg_map, bool(ok)
        return result_frame, func_result, True

    def infer_with_packet(
        self, packet: FramePacket | None
    ) -> tuple[FramePacket | None, np.ndarray | None, bool]:
        """带 FramePacket 的推理接口，返回 (packet, seg_map, ok)。

        内部调用 infer_with_frame，packet 中的 frame_bgr 和返回的 seg_map
        是严格配对的（通过 _frame_queue 机制）。
        用于与目标检测模块按 fid 融合。
        """
        if packet is None or packet.frame_bgr is None:
            return packet, None, False

        if not self._pool_initialized:
            self._pool_init(packet.frame_bgr, packet)

        self.rknn_pool.put(packet.frame_bgr)
        self._frame_queue.append(packet.frame_bgr.copy())
        self._packet_queue.append(packet)

        func_result, pool_success = self.rknn_pool.get()
        if not pool_success or func_result is None:
            return None, None, False

        result_packet = self._packet_queue.popleft() if self._packet_queue else packet
        if self._frame_queue:
            self._frame_queue.popleft()

        if isinstance(func_result, tuple) and len(func_result) == 2:
            seg_map, ok = func_result
            return result_packet, seg_map, bool(ok)
        return result_packet, func_result, True

    def visualize(self, seg_map: np.ndarray, frame_bgr: np.ndarray | None = None) -> np.ndarray:
        try:
            return seg_visualization(seg_map, frame_bgr, blend_alpha=self.blend_alpha)
        except Exception as exc:
            arr = np.asarray(seg_map)
            print(
                "visualize failed:",
                f"shape={arr.shape}",
                f"dtype={arr.dtype}",
                f"min={arr.min() if arr.size else 'n/a'}",
                f"max={arr.max() if arr.size else 'n/a'}",
                f"error={exc}",
            )
            raise

    def normalize_mask(self, seg_map: np.ndarray) -> np.ndarray:
        return _prepare_seg_map(seg_map)

    def __call__(self, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """语法糖：seg(frame) 等价 seg.myinference(frame)。"""
        return self.infer(frame_bgr)

    def release(self) -> None:
        """释放线程池与所有 NPU runtime；进程退出前调用，可重复调用。"""
        if self._released:
            return
        self.rknn_pool.release()
        self._released = True
        print("PP-Seg resources released")


def main() -> int:
    parser = argparse.ArgumentParser(description="PP-Seg Inference Demo")
    parser.add_argument("--model_dir", type=str, default="../../mymodel/seg_model", help="Directory containing the .rknn seg model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, default="./result.png", help="Path to save result image")
    parser.add_argument("--TPEs", type=int, default=1, help="Number of thread pool executors")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark test")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations for benchmark")
    parser.add_argument("--blend", type=float, default=None, help="Blend alpha (0-1) with original image")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return -1

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Failed to read image: {args.image_path}")
        return -1

    print(f"Input image size: {img.shape[1]}x{img.shape[0]}", img.shape)

    try:
        infer = PPSegInfer(model_dir=args.model_dir, TPEs=args.TPEs, blend_alpha=args.blend)
    except Exception as e:
        print(f"Error initializing PPSegInfer: {e}")
        return -1

    try:
        if args.benchmark:
            print(f"\nRunning benchmark with {args.num_iterations} iterations...")
            for _ in range(10):
                _, ok = infer.infer(img)
                if not ok:
                    print("Warmup failed")
                    return -1

            start_time = time.time()
            seg_map = None
            for i in range(args.num_iterations):
                seg_map, ok = infer.infer(img)
                if not ok or seg_map is None:
                    print(f"Iteration {i + 1} failed!")
                    return -1
            end_time = time.time()

            result_img = infer.visualize(seg_map, img)
            elapsed_time = end_time - start_time
            fps = args.num_iterations / elapsed_time
            avg_time = (elapsed_time / args.num_iterations) * 1000

            print("\nBenchmark Results:")
            print(f"  Total time: {elapsed_time:.2f}s")
            print(f"  Average time: {avg_time:.2f}ms per frame")
            print(f"  FPS: {fps:.2f}")
        else:
            seg_map, ok = infer.infer(img)
            if not ok or seg_map is None:
                print("Inference failed")
                return -1
            result_img = infer.visualize(seg_map, img)

        cv2.imwrite(args.output_path, result_img)
        print(f"Result saved to: {args.output_path}")
        return 0
    finally:
        infer.release()


if __name__ == "__main__":
    raise SystemExit(main())
