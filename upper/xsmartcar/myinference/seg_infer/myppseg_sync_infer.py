# Copyright (c) 2026 清影/123twtd
"""PP-Seg 语义分割同步推理模块（香橙派 RKNN）。

本文件是「感知层」的第二环：接收 BGR 相机帧，在 NPU 上跑 PP-LiteSeg 等分割模型，
输出与原图同尺寸的 0/1 mask，供 IPM、扫线、车道跟踪等几何层使用。

整体数据流
----------
    SharedMemoryFrameSource.read()  --BGR 帧-->
        PPSegSyncInfer.infer()  --预处理 512x512 RGB-->
            RKNNLite.inference()  --NPU-->
        postprocess  --最近邻缩放回 (H,W)-->  seg_map (uint8, 0=背景 1=车道)
            |
            v
    run_pipe / trk_preview -> FastIPM -> LaneTracker ...

输入 / 输出约定
--------------
    infer(frame_bgr)
        入参：OpenCV BGR 图像，shape=(H, W, 3)，uint8；None 时返回 (None, False)
        出参：(seg_map, ok)
            seg_map：与原图同高宽的灰度 mask，像素值 0 或 1（经 normalize_mask 二值化）
            ok：本次推理是否成功

    visualize(seg_map, frame_bgr) 仅用于调试预览，不参与控制闭环。

模型与路径
----------
    model_dir 默认 "model"，相对本文件所在目录（xsmartcar/model/）。
    目录下第一个 *.rknn 会被自动加载；部署时把 pp_liteseg*.rknn 拷入即可。

使用方式
--------
    seg = PPSegSyncInfer(model_dir="model")
    mask, ok = seg.infer(frame_bgr)
    if not ok or mask is None:
        ...  # 保留上一帧控制或跳过
    seg.release()  # 进程退出前释放 NPU runtime

注意
----
- 同步单 runtime：每帧阻塞等 NPU 返回，联调最稳；多 runtime 吞吐更高但易卡顿。
- infer 失败时打印异常并返回 (None, False)，不抛错，便于主循环继续跑。
- 预处理固定缩放到 512×512；后处理用 INTER_NEAREST 拉回原始分辨率，保持类别边界。
- release() 后不可再 infer；重复 release 安全。

预处理 pipeline 说明
--------------------
    BGR -> RGB -> resize(512,512) LINEAR -> expand_dims -> (1,512,512,3)
    与训练/导出 RKNN 时的输入布局一致；改 IMG_SIZE 需同步改模型。

后处理与 normalize_mask
-----------------------
    RKNN 输出可能是 list、多通道 logits 或已是单通道；normalize_mask 统一成 2-D 二值图：
    - 3-D 且通道数 <=32：沿通道 argmax 取类别
    - 否则取首通道或 squeeze
    - 最终 (seg_map > 0) 得到 0/1 mask（0=背景，非 0=前景/车道）

SEG_COLORS 与可视化
-------------------
    仅用于 colorize / blend 调试叠图：0=黑，1=白。不影响 infer 返回的 mask 数值。
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# 可视化调色板：class 0 背景黑，class 1 车道白（RGB 顺序，叠图时再转 BGR）
SEG_COLORS = np.array(
    [
        [0, 0, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)

# 模型输入分辨率，须与 .rknn 导出时的 input size 一致
IMG_SIZE = (384, 384)  # 当前模型 ppseg_384_*
# IMG_SIZE = (512, 512)  # 备用：512×512 模型

def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """BGR 原图 -> NPU 输入张量 (1, H, W, 3) uint8 RGB。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    # RKNN 期望 batch 维；uint8 由 runtime 内部按模型 config 做归一化
    return np.expand_dims(img_resized, axis=0)


def postprocess_segmentation(output, original_size: tuple[int, int]) -> np.ndarray:
    """RKNN 原始输出 -> 与原图同尺寸的类别图（仍为模型输出 dtype，未二值化）。

    output 可能是 [tensor] 或单个 ndarray；取第 0 个 batch、第 0 个输出头。
    用最近邻放大，避免线性插值把 0/1 边界插成中间灰度。
    """
    seg_output = output[0] if isinstance(output, list) else output
    seg_map = seg_output[0]

    orig_h, orig_w = original_size
    return cv2.resize(
        seg_map.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)


def normalize_mask(seg_map: np.ndarray) -> np.ndarray:
    """把各种形状的 RKNN 输出统一为 2-D 二值 mask：0=背景，1=前景。

    不同导出方式可能给出 (C,H,W) logits、(H,W,C) 或 (H,W)；此处做兼容。
    """
    seg_map = np.asarray(seg_map)
    seg_map = np.squeeze(seg_map)

    if seg_map.ndim == 3:
        # 通道在前的 logits：(C,H,W)，C 为类别数
        if seg_map.shape[0] <= 32:
            seg_map = np.argmax(seg_map, axis=0)
        # 通道在后的 logits：(H,W,C)
        elif seg_map.shape[-1] <= 32:
            seg_map = np.argmax(seg_map, axis=-1)
        else:
            seg_map = seg_map[..., 0]

    if seg_map.ndim != 2:
        raise ValueError(f"Expected a 2-D seg_map after normalization, got shape={seg_map.shape}")

    # 二值化：模型 class 0 为背景，其余视为车道/前景
    return (seg_map > 0).astype(np.uint8)


def colorize_segmentation(seg_map: np.ndarray) -> np.ndarray:
    """二值 mask -> RGB 伪彩色图，仅用于可视化。"""
    seg_map = normalize_mask(seg_map)
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


def get_current_dir() -> str:
    """本模块所在目录，用于解析相对路径 model_dir。"""
    return os.path.dirname(os.path.abspath(__file__))


class PPSegSyncInfer:
    """同步单 RKNN runtime 的 PP-Seg 封装，面向「每帧取最新图」的实时赛道场景。

    生命周期：__init__ 加载 .rknn 并 init_runtime -> 循环 infer -> release。
    与多线程/多 runtime 版本相比，调用简单、行为可预期，是当前 pipeline 默认选型。
    """

    def __init__(self, model_dir: str = "../../mymodel/seg_model", blend_alpha: float | None = None, core_mask=None) -> None:
        # model_dir 相对 xsmartcar/ 模块目录，而非当前工作目录
        model_dir = os.path.join(get_current_dir(), model_dir)
        self.model_path = self.get_model_path(model_dir)
        # 非 None 时 visualize 会与原图半透明叠加
        self.blend_alpha = blend_alpha
        self._released = False
        self.rknn_lite = RKNNLite()

        ret = self.rknn_lite.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: ret={ret}")

        # 默认占用 NPU core 0；多模型时可传 RKNNLite.NPU_CORE_0_1 等
        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_0
        ret = self.rknn_lite.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: ret={ret}")

        print(f"PP-Seg sync model loaded: {self.model_path}")

    def get_model_path(self, model_dir: str) -> str:
        """在目录中查找第一个 .rknn 文件。"""
        model_files = glob.glob(os.path.join(model_dir, "*.rknn"))
        if not model_files:
            raise FileNotFoundError(f"No .rknn model found in {model_dir}")
        return model_files[0]

    def infer(self, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """对单帧 BGR 图像做分割推理。

        返回
        ----
        (seg_map, ok)
            seg_map：与原图同尺寸的 uint8 mask（0/1），失败时为 None
            ok：是否成功；失败时主循环可沿用上一帧控制量
        """
        if self._released:
            raise RuntimeError("PPSegSyncInfer has been released")
        if frame_bgr is None:
            return None, False

        try:
            input_data = preprocess_image(frame_bgr)
            outputs = self.rknn_lite.inference(inputs=[input_data])
            seg_map = postprocess_segmentation(outputs, frame_bgr.shape[:2])
            seg_map = normalize_mask(seg_map)
            return seg_map, True
        except Exception as exc:
            # 不向上抛，避免整圈 pipeline 因单帧 NPU 抖动退出
            print(f"sync infer failed: {exc}")
            return None, False

    def visualize(self, seg_map: np.ndarray, frame_bgr: np.ndarray | None = None) -> np.ndarray:
        """调试预览：伪彩色或与原图混合。"""
        return seg_visualization(seg_map, frame_bgr, blend_alpha=self.blend_alpha)

    def normalize_mask(self, seg_map: np.ndarray) -> np.ndarray:
        """对外暴露与工具函数相同的 mask 规范化，供 shm_ppseg_latest_frame 等脚本使用。"""
        return normalize_mask(seg_map)

    def release(self) -> None:
        """释放 NPU runtime 与模型句柄；进程退出或切换模型前调用。"""
        if self._released:
            return
        self.rknn_lite.release()
        self._released = True
        print("PP-Seg sync resources released")


def main() -> int:
    """单张图片离线推理 demo：测耗时并保存可视化结果。"""
    parser = argparse.ArgumentParser(description="PP-Seg Sync Inference Demo")
    parser.add_argument("--model_dir", type=str, default="../../mymodel/seg_model")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./result_sync.png")
    parser.add_argument("--blend", type=float, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return -1

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Failed to read image: {args.image_path}")
        return -1

    infer = PPSegSyncInfer(model_dir=args.model_dir, blend_alpha=args.blend)
    try:
        start = time.time()
        seg_map, ok = infer.infer(img)
        elapsed_ms = (time.time() - start) * 1000.0
        if not ok or seg_map is None:
            print("Inference failed")
            return -1
        result_img = infer.visualize(seg_map, img)
        cv2.imwrite(args.output_path, result_img)
        print(f"Result saved to: {args.output_path}")
        print(f"sync_infer_ms: {elapsed_ms:.2f}")
        return 0
    finally:
        infer.release()


if __name__ == "__main__":
    raise SystemExit(main())
