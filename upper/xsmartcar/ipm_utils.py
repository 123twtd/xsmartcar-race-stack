# Copyright (c) 2026 清影/123twtd
# 模块化重构贡献：高志禹

"""
ipm_utils.py — IPM 逆透视变换模块（比赛调用接口）
==================================================

唯一类：FastIPM — 比赛版单档位处理器

    初始化时一次性加载 M 矩阵，之后逐帧调用 process() 即可，
    不再访问磁盘，适合嵌入式实车主循环。

用法::

    from ipm_utils import FastIPM

    # 初始化（程序启动时执行一次）
    ipm = FastIPM("data_npz/race_data.npz", out_w=160, out_h=120)

    # 主循环逐帧调用
    bird_view = ipm.process(frame)              # 最近邻插值，最快
    bird_view = ipm.process(frame, nearest=False)  # 双线性插值，更平滑

注意
----
- M 矩阵在 640×480 原图上标定，process() 自动将输入缩放到 640×480
- 不要先做去畸变再套 M（除非标定时也先去了畸变）
"""

import cv2
import numpy as np
from pathlib import Path


class FastIPM:
    """
    比赛版 IPM 处理器（单档位，极速推理）

    适用于 race_ipm.py 标定输出的 race_data.npz。

    Parameters
    ----------
    npz_path : str or Path
        race_data.npz 文件路径
    out_w : int
        输出宽度（像素），默认 160
    out_h : int
        输出高度（像素），默认 120
    """

    def __init__(self, npz_path, out_w=160, out_h=120):
        self.out_size = (out_w, out_h)

        npz_file = Path(npz_path)
        if not npz_file.exists():
            raise FileNotFoundError(f"找不到标定文件: {npz_file}")

        with np.load(str(npz_file)) as data:
            if 'M' not in data:
                raise KeyError(f"文件 {npz_file} 中没有找到矩阵 'M'")
            self.M = data['M']

    def process(self, img, nearest=True):
        """
        对单帧图像执行 IPM 逆透视变换。

        Parameters
        ----------
        img : np.ndarray
            输入图像（BGR），任意尺寸，会自动缩放到 640×480
        nearest : bool
            True  → INTER_NEAREST 最近邻插值，速度最快，适合掩膜/二值图
            False → INTER_LINEAR 双线性插值，更平滑，适合彩色图预览

        Returns
        -------
        np.ndarray
            IPM 鸟瞰图，尺寸为 (out_h, out_w, C)
        """
        if img.shape[0] != 480 or img.shape[1] != 640:
            img = cv2.resize(img, (640, 480))

        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.warpPerspective(img, self.M, self.out_size, flags=flags)
