# Copyright (c) 2026 清影/123twtd
"""RKNN 多 runtime 线程池（异步分割推理底座）。

本文件是「感知层」的可选加速底座：把多个 RKNNLite runtime 绑定到不同 NPU core，
用线程池并发跑推理，从而把多帧推理重叠起来提高吞吐。仅 myppseg_infer.PPSegInfer
使用它。

整体关系
--------
    PPSegInfer(TPEs>1)
        -> rknnPoolExecutor.put(frame)   # 提交一帧，立即返回（异步）
        -> rknnPoolExecutor.get()        # 取出「较早提交」的一帧结果

注意（重要）
------------
- 当前比赛主链 run_pipe.py 默认走 **同步单 runtime** 的 myppseg_sync_infer，
  并不使用本文件。这里是为「需要更高吞吐」时保留的异步备选路径。
- 异步语义：put/get 是流水线，get() 返回的是之前某帧的结果，不保证与最近 put 的帧
  一一对应。调用方（PPSegInfer）用一个 frame 队列来对齐「结果 ↔ 原帧」。
- TPEs（Thread Pool Executors）= 同时存在的 runtime 数量，按 i%3 轮流绑到 NPU
  core 0/1/2。core 数量有限，TPEs 过大不会线性加速。

cloudpickle 替换 pickle 的原因
------------------------------
    线程池本身不需要序列化，但部分 RKNN/驱动栈在初始化时会触发 pickle；
    标准 pickle 无法序列化某些闭包/对象，这里用 cloudpickle 顶替以避免初始化报错。
"""

from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import cloudpickle
import sys


def initRKNN(rknnModel="yolov.rknn", id=0):
    """加载一个 .rknn 并在指定 NPU core 上初始化 runtime，返回 RKNNLite 句柄。

    id 与 NPU core 的映射：
        0/1/2 -> 单独占用 core 0/1/2
        -1    -> 同时使用 core 0+1+2（大模型单实例）
        其它   -> 由驱动自动选择
    任一步失败直接 exit(ret)，因为没有可用 runtime 时整条感知链无法继续。
    """
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, "\t\tdone")
    return rknn_lite


def initRKNNs(rknnModel="yolo.rknn", TPEs=1):
    """创建 TPEs 个 runtime，按 i%3 轮流绑定到 core 0/1/2。"""
    # 见模块文档：用 cloudpickle 顶替 pickle，规避初始化期的序列化问题
    sys.modules['pickle'] = cloudpickle
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 3))
    return rknn_list


class rknnPoolExecutor():
    """RKNN 推理线程池：put 提交帧，get 取较早一帧的结果。

    func 形如 func(rknn_lite, frame) -> 任意结果；每个任务从池中轮流取一个
    runtime 执行，实现多帧并发。
    """

    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()  # 保存 future，按提交顺序 FIFO 取结果
        self.rknnPool = initRKNNs(rknnModel, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        # self.pool = ProcessPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0  # 已提交帧计数，用于轮询选择 runtime

    def put(self, frame):
        """提交一帧到线程池（异步，立即返回）。"""
        self.queue.put(self.pool.submit(
            self.func, self.rknnPool[self.num % self.TPEs], frame))
        self.num += 1

    def get(self):
        """取出最早提交那一帧的结果；队列为空返回 (None, False)。"""
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True  # fut.result() 会阻塞直到该帧推理完成

    def release(self):
        """关闭线程池并释放所有 runtime（进程退出前必须调用）。"""
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
