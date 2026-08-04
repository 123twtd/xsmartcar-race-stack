# Copyright (c) 2026 清影/123twtd
"""共享内存相机读取吞吐探针，不执行分割、检测或 UART 发送。"""
import os
import sys
import time

# 把仓库根目录加入 sys.path，保证直接运行本脚本时也能 import xsmartcar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from xsmartcar.ar_receiver import SharedMemoryFrameSource


def test_max_read_fps():
    # flip_vertical=False 和 rgb_to_bgr=False 彻底关闭 CPU 矩阵操作，测最纯粹的读取 IO
    source = SharedMemoryFrameSource(flip_vertical=False, rgb_to_bgr=False)
    print("Testing RAW Shared Memory Read FPS... Press Ctrl+C to stop.")

    if not source.connect():
        print("Failed to connect to shared memory. Is the camera producer running?")
        return

    fps_t = time.time()
    fps_n = 0

    try:
        while True:
            frame = source.read()
            if frame is None:
                # 极短休眠(1毫秒)，防止当前 while 循环把 CPU 核心吃满，导致写入端抢不到 CPU 资源
                time.sleep(0.001)
                continue

            fps_n += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps = fps_n / (now - fps_t)
                print(f"[Raw Read Limit] FPS: {fps:.2f} | current_fid: {source.last_fid}")
                fps_n = 0
                fps_t = now

    except KeyboardInterrupt:
        print("\nTest stopped.")
    finally:
        source.close()


if __name__ == "__main__":
    test_max_read_fps()
