# Copyright (c) 2026 清影/123twtd
# 模块化重构贡献：高志禹

"""比赛版 IPM 四点标定、矩阵保存与本地预览工具。"""

import argparse
from pathlib import Path

import cv2
import numpy as np

UPPER_ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = Path(__file__).resolve().parent

CALIB_IMG = TOOL_DIR / "assets" / "calibration_track.png"
MASK_PATH = TOOL_DIR / "assets" / "mask_input.png"
OUTPUT_DIR = TOOL_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "ipm_preview.png"
NPZ_PATH = UPPER_ROOT / "xsmartcar" / "data_npz" / "race_data.npz"

# 默认直接使用比赛时标定的四个角点；传入 --interactive 可重新手动点选四点。
SRC_POINTS = np.float32(
    [
        [264, 327],
        [382, 331],
        [227, 392],
        [416, 404],
    ]
)

# 这里是 IPM 的物理和输出尺寸定义。
# W_CM / D_CM 用来把赛道物理尺寸映射到输出平面，
# NEAR_CM 是近端保留距离，IPM_W / IPM_H 是输出图像尺寸。
W_CM, D_CM, NEAR_CM = 60.0, 90.0, 5.0
IPM_W, IPM_H = 160, 120
# 与最终完赛 race_data.npz 保存的 dst_points 一致。
BOARD = (-7.5, 7.5, 5.0, 20.0)


def make_dst() -> np.ndarray:
    sx, sy = IPM_W / W_CM, IPM_H / D_CM

    def p(x: float, y: float) -> list[float]:
        return [(x + W_CM / 2) * sx, IPM_H - (y - NEAR_CM) * sy]

    l, r, n, f = BOARD
    return np.float32([p(l, f), p(r, f), p(l, n), p(r, n)])


def get_src(*, interactive: bool = False) -> np.ndarray:
    if not interactive:
        return SRC_POINTS

    img = cv2.imread(str(CALIB_IMG))
    if img is None:
        raise FileNotFoundError(f"cannot read calibration image: {CALIB_IMG}")

    pts: list[list[int]] = []
    labels = ["left_far", "right_far", "left_near", "right_near"]
    win = "Pick 4 board corners"

    def cb(evt, x, y, *_):
        if evt == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])
            print(f"{len(pts) - 1} {labels[len(pts) - 1]}: ({x},{y})")
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(img, str(len(pts) - 1), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(win, img)
            if len(pts) == 4:
                cv2.destroyAllWindows()

    cv2.imshow(win, img)
    cv2.setMouseCallback(win, cb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(pts) != 4:
        raise RuntimeError(f"expected 4 calibration points, got {len(pts)}")
    src = np.float32(pts)
    print("Copy the following values into SRC_POINTS after checking the preview:")
    print(f"SRC_POINTS = np.float32({src.tolist()})")
    return src


def run_ipm(M: np.ndarray, ipm_w: int | None = None, ipm_h: int | None = None, *, show: bool = True) -> None:
    w = ipm_w or IPM_W
    h = ipm_h or IPM_H
    img = cv2.imread(str(MASK_PATH))
    if img is None:
        raise FileNotFoundError(f"cannot read mask image: {MASK_PATH}")
    if img.shape[:2] != (480, 640):
        img = cv2.resize(img, (640, 480))

    ipm = cv2.warpPerspective(img, M, (w, h))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), ipm)
    print(f"saved: {OUTPUT_PATH}")
    # 同时保存输入 mask 和输出 IPM 图，方便你对照标定结果是否正确。
    cv2.imwrite(str(OUTPUT_DIR / "mask_input.png"), img)
    cv2.imwrite(str(OUTPUT_DIR / f"ipm_{w}x{h}.png"), ipm)
    if show:
        cv2.imshow(f"IPM {w}x{h}", ipm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比赛版 IPM 四点标定与预览")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="打开标定图，依次点击左远、右远、左近、右近四个角点。",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=NPZ_PATH,
        help="NPZ 输出路径；默认更新上位机运行时 race_data.npz。",
    )
    parser.add_argument("--no-show", action="store_true", help="Skip the OpenCV preview window.")
    args = parser.parse_args()

    src = get_src(interactive=args.interactive)
    dst = make_dst()
    M = cv2.getPerspectiveTransform(src, dst)
    output_npz = args.output_npz
    saved = dict(np.load(str(output_npz))) if output_npz.exists() else {}
    saved["M"] = M
    saved["src_points"] = src
    saved["dst_points"] = dst
    saved["ipm_size"] = np.array([IPM_W, IPM_H])
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_npz), **saved)
    print(f"saved: {output_npz} ipm_size={IPM_W}x{IPM_H}")
    run_ipm(M, IPM_W, IPM_H, show=not args.no_show)
