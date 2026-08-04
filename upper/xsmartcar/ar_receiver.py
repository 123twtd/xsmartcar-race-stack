"""AR 相机共享内存读帧模块（客户端侧）。

本文件是「感知层」的入口：从另一进程（通常是 AR/相机服务）写入的 POSIX 共享内存
中读取最新视频帧，并转换成 OpenCV 常用的 BGR 格式，供后续分割、跟踪等模块使用。

整体数据流
----------
    [相机/AR 服务进程]  --写入-->  shm_ar_video  --读取-->  SharedMemoryFrameSource.read()
                                                                    |
                                                                    v
                                                          run_pipe / shm_ppseg_* 等

共享内存布局（与生产者约定，见 code_ref_test/ref_prog/img_collect_2.py）
--------------------------------------------------------------------------
    偏移 0~7   : fid   (uint64, Q)  帧序号，每写一帧 +1，用于判断是否有新帧
    偏移 8~11  : width (uint32, I)  图像宽（像素）
    偏移 12~15 : height(uint32, I)  图像高（像素）
    偏移 16~   : 像素数据，行主序 RGB888，长度 = width * height * 3

使用方式
--------
    src = SharedMemoryFrameSource()
    frame = src.read()          # 无新帧时返回 None，不要忙等
    if frame is None:
        time.sleep(0.01)
        continue
    # frame 为 numpy.ndarray，shape=(H,W,3)，dtype=uint8，BGR

注意
----
- 本模块只「附着」共享内存，不负责创建；生产者未启动时 connect() 会失败。
- fid 可能跳号（中间帧被跳过），只要递增即表示有新帧。
- 读帧后会做垂直翻转 + RGB→BGR，与 OpenCV 显示/推理习惯一致。

struct.unpack('QII') 是什么？
----------------------------
    struct 按 C 语言内存布局解析二进制。格式串每个字母对应一个字段：

        Q  -> unsigned long long，8 字节，小端序，即帧号 fid
        I  -> unsigned int，4 字节，即 width
        I  -> unsigned int，4 字节，即 height

    8 + 4 + 4 = 16 字节，与 SHM_HEADER_SIZE 一致。
    小端序：低字节在前（x86/ARM 常见），与生产者 struct.pack('QII', ...) 必须匹配。

为什么要「先视图、再 copy」？
----------------------------
    np.ndarray(..., buffer=shm.buf) 不分配新内存，数组直接指向共享内存。
    若直接对这个视图做 cv2.flip / cvtColor，或在推理里异步使用，生产者可能
    同时覆写同一块 shm，导致花屏或崩溃。

    因此流程是：视图读 shm -> .copy() 到本进程堆内存 -> 再翻转/转色。
    copy 之后本帧与 shm 解耦，后续处理安全。

RGB 翻转与 BGR 转换
-------------------
    生产者通常按相机原始朝向写 RGB；OpenCV 默认 BGR，且 imshow 的 y 轴向下。
    flip_vertical：沿水平轴翻转（cv2.flip(frame, 0)），纠正上下颠倒。
    rgb_to_bgr：交换 R/B 通道，否则颜色会偏蓝/偏红。
"""
# Copyright (c) 2026 清影/123twtd

import struct
import time
from multiprocessing import resource_tracker, shared_memory

import cv2
import numpy as np

# 与 AR/相机服务约定的共享内存名称，双方必须一致
SHM_NAME = "shm_ar_video"
# 帧头固定 16 字节：8(fid) + 4(w) + 4(h)
SHM_HEADER_SIZE = 16


def remove_shm_from_resource_tracker(name: str = SHM_NAME) -> None:
    """从本进程的 resource_tracker 注销共享内存，避免退出时误删生产者创建的块。

    Python multiprocessing 在 attach 共享内存时会登记到 resource_tracker；
    客户端进程退出时 tracker 可能尝试 unlink，导致服务端内存被删掉。
    客户端只读不写，因此连接成功后应立即 unregister。
    """
    try:
        resource_tracker.unregister('/' + name, 'shared_memory')
    except Exception:
        pass


class SharedMemoryFrameSource:
    """从 shm_ar_video 读取帧，对外提供与 FrameSource.read() 一致的接口。

    典型调用链：connect() 在首次 read() 时自动执行；重复 read() 直到返回非 None
  即得到最新帧。last_fid / last_error 用于调试与心跳诊断。
    """

    def __init__(
        self,
        name: str = SHM_NAME,
        header_size: int = SHM_HEADER_SIZE,
        *,
        flip_vertical: bool = True,
        rgb_to_bgr: bool = True,
    ) -> None:
        self.name = name
        self.header_size = header_size
        # 生产者写入的 RGB 常与显示坐标系上下颠倒，默认翻转以匹配 OpenCV
        self.flip_vertical = flip_vertical
        self.rgb_to_bgr = rgb_to_bgr
        self._shm: shared_memory.SharedMemory | None = None
        self._last_fid = 0  # 上次成功读到的帧号，用于去重
        self._last_error: str | None = None

    @property
    def last_fid(self) -> int:
        """最近一次成功读帧的 fid，可供日志或 UI 显示。"""
        return self._last_fid

    @property
    def last_error(self) -> str | None:
        """最近一次失败原因；成功读帧后会被清空。"""
        return self._last_error

    def connect(self) -> bool:
        """附着到已存在的共享内存块。已连接则直接返回 True。"""
        if self._shm is not None:
            return True
        try:
            # create=False（默认）：只 attach 已有块，不创建；创建是生产者的事
            self._shm = shared_memory.SharedMemory(name=self.name)
            remove_shm_from_resource_tracker(self.name)
            self._last_error = None
            return True
        except FileNotFoundError:
            # 生产者尚未创建 shm_ar_video
            self._last_error = f"shared memory '{self.name}' not found"
            return False
        except Exception as exc:
            self._last_error = f"connect failed: {exc}"
            self._reset_shm()
            return False

    def read(self) -> np.ndarray | None:
        """读取一帧；若无新帧或出错则返回 None。

        返回
        ----
        np.ndarray | None
            成功：BGR 图像 (H, W, 3)，uint8。
            失败/无新帧：None，可查看 last_error。
        """
        if not self.connect() or self._shm is None:
            return None

        try:
            # 1. 解析 16 字节帧头
            #    bytes(...) 把 memoryview 拷成独立 bytes，避免 unpack 时 buf 被并发改写
            header = bytes(self._shm.buf[: self.header_size])
            #    'QII' => fid(uint64), width(uint32), height(uint32)，见模块文档
            fid, width, height = struct.unpack('QII', header)

            # 2. fid 未变说明生产者还没写新帧（或我们读得太快）
            #    返回 None 让上层 sleep 一小会儿，不要 while True 空转占满 CPU
            if fid == self._last_fid:
                return None
            if width <= 0 or height <= 0:
                self._last_error = f"invalid frame size: {width}x{height}"
                return None

            # 3. 按协议计算 RGB 数据区并校验缓冲区长度
            size = width * height * 3
            start = self.header_size
            stop = start + size
            if stop > len(self._shm.buf):
                self._last_error = (
                    f"buffer too small: need {stop}, have {len(self._shm.buf)}"
                )
                self._reset_shm()
                return None

            # 4. 从 shm 像素区构造 RGB 数组（仍是生产者内存，未 copy）
            #    shape (H,W,3)：行主序，buf[i] 对应第 i 个像素的 R,G,B
            frame_rgb_view = np.ndarray(
                (height, width, 3),
                dtype=np.uint8,
                buffer=self._shm.buf[start:stop],
            )
            #    深拷贝到本进程；之后 flip/cvtColor 不再依赖 shm，生产者可写下一帧
            frame = frame_rgb_view.copy()
            del frame_rgb_view  # 尽快释放对 shm 的引用

            self._last_fid = fid  # 仅成功拷贝后才更新，避免丢帧后永远跳过
            self._last_error = None

            # 5. 与 OpenCV / 本仓库其它模块对齐（可在构造时关掉 flip_vertical/rgb_to_bgr）
            if self.flip_vertical:
                frame = cv2.flip(frame, 0)  # 0 = 沿 x 轴翻转，即上下颠倒
            if self.rgb_to_bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # R<->B
            return frame
        except (BufferError, ValueError, struct.error) as exc:
            self._last_error = f"read failed: {exc}"
            self._reset_shm()
            return None
        except Exception as exc:
            self._last_error = f"unexpected read error: {exc}"
            self._reset_shm()
            return None

    def close(self) -> None:
        """关闭本进程对共享内存的附着（不删除共享内存本身）。"""
        self._reset_shm()

    def _reset_shm(self) -> None:
        """断开 shm 连接，出错或 close 时调用。

        close() 只解除本进程映射，不 unlink 共享内存（unlink 会删掉整块内存）。
        出错后置 _shm=None，下次 read() 会重新 connect()。
        """
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None


def main() -> None:
    """本地冒烟测试：循环读帧、算 FPS、OpenCV 窗口预览。按 ESC 退出。"""
    source = SharedMemoryFrameSource()
    print('SharedMemoryFrameSource ready. Waiting for frames...')

    fps_t = time.time()
    fps_n = 0
    cur_fps = 0.0

    try:
        while True:
            frame = source.read()
            if frame is None:
                # 无新帧时短暂休眠，避免 CPU 空转
                time.sleep(0.01)
                if cv2.waitKey(1) == 27:
                    raise KeyboardInterrupt
                continue

            # 统计「成功 read 到的新帧」频率，不是相机原始 FPS（中间帧会被 fid 去重跳过）
            fps_n += 1
            now = time.time()
            if now - fps_t >= 1.0:
                cur_fps = fps_n / (now - fps_t)
                fps_n = 0
                fps_t = now

            vis = frame.copy()  # 叠字不污染 read() 返回的 frame
            cv2.putText(
                vis,
                f'Client FPS: {cur_fps:.1f} fid={source.last_fid}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow('shared_memory_preview', vis)

            if cv2.waitKey(1) == 27:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        print('\nShared memory preview stopped.')
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
