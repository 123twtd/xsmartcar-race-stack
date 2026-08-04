# Copyright (c) 2026 清影/123twtd
# 赛后实车记录：车辆避让实际完赛，50 cm/s，2:18；原始运行参数见 指令.md。
"""循迹主控 — 七线程异步流水线（无 GUI）。

架构：
    Thread 1  capture_worker   读 shm → cap_q
    Thread 2  seg_worker       分割推理 → seg_q
    Thread 2b  det_worker        cap_q → 目标检测 → det_q + det_q2
    Thread 3  compute_worker   seg_q → IPM→扫线→前瞻 → line_q (seg 速度)
    Thread 3b  decision_worker   det_q → 连续帧确认+API+决策 → command+fork_bias
    Thread 3c  avoid_worker      det_q2 → IPM 投影 → 固定偏置+3帧确认+冲突感知衰减 → ar_offset_cm
    Thread 4  uart_worker      line_q → 30fps 固定下发 L + C 帧

UART 独立线程，便于后续加避障 API：
  - 避障 → command.stop_request / run_enable（C 帧控制）
  - 避障偏移 → det_offset_cm（L 帧第三变量，由 compute_worker 填入）

无可视化，用 --debug 开启终端日志。

【避障方案：固定偏置 + 3帧确认 + 冲突感知衰减】
  - 检测到车（70cm 内）→ 按车宽算固定偏置 = 15cm，方向由方向锁决定
  - 车消失 → 先 3 帧确认（防漏检抖动），再开始衰减
  - 衰减时读 track_offset_cm（循迹线横向偏移）：
      循迹线与避障偏置方向冲突 → 衰减减速（0.95），让循迹线先发力（防弯道冲出）
      无冲突 → 正常衰减（0.85），快速回正
  - 状态简洁：direction_lock + miss_count + ar_offset_cm[0]

【参数说明】
速度50cm/s时
kp1=3.3 kp2=0.1 kp3=0 ki=0.03 kd=0.02 非线性偏置增益-0.02【50cm/s的速度可以，回正能力增强，但是针对增大】
kp1=2.6 kp2=0.0 kp3=0 ki=0.00 kd=0.01非线性偏置增益-0.05 global平滑 -几乎不抖动，仅在弯道有略微波动
速度改为60cm/s，串口5抖动变大，修改窗口后默认5->7，则抖动变小，效果为 -几乎不抖动，仅在弯道有略微波动

推荐参数组合（速度 60cm/s，几乎不抖动，弯道微抖）：
  kp1=2.6 kp2=0.0 kp3=0 ki=0.00 kd=0.01
  --offset-alpha 0.05（非线性偏置增益）
  --smooth global --global-smooth-window 7（默认5→7改善，>7反而恶化）

关键结论：
  1. KD=0.02 有抖动，KD=0 无法回正，KD=0.01 是甜点
  2. global 平滑是"治本"，local 平滑在噪声阈值边缘不稳定
  3. 窗口5→7改善，7→9恶化（相位滞后导致振荡）
  4. 速度↑需同步↑平滑窗口（60cm/s时5→7刚好）
"""

import argparse
import sys
import time
import threading
import queue
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xsmartcar.ar_receiver import SharedMemoryFrameSource
from xsmartcar.ipm_utils import FastIPM
from xsmartcar.myinference.seg_infer.myppseg_infer import PPSegInfer
from xsmartcar.trk_preview import (
    GuidanceCommand,
    PreviewConfig,
    PreviewExtractor,
    compute_line_target,
    control_values,
)
from xsmartcar.trk_tracker_module import LaneTracker, TrackerConfig
from xsmartcar.myinference.det_infer.detector import RKNNDetector
from xsmartcar.qianfan_api.qianfan_model_api_post import call_roadsign_api, load_config as load_api_config
## ==避障==
from xsmartcar.ipm_M import FastDetectionIPM
from xsmartcar.obt import detections_to_ar_objects

# ============================================================
# 默认路径
# ============================================================
DEFAULT_SEG_DIR = REPO_ROOT / "xsmartcar" / "mymodel" / "seg_model"
DEFAULT_DET_MODEL = REPO_ROOT / "xsmartcar" / "mymodel" / "det_model" / "mbjc_384_int8.rknn"
DEFAULT_NPZ = REPO_ROOT / "xsmartcar" / "data_npz" / "race_data.npz"

class ForkBiasState:
    """fork 引导偏置，叠加到 near_yaw（航向角）。

    API 返回 L/R 后设偏转角度值，正值=偏左转，负值=偏右转。
    发车后可选衰减，偏置归零后还原默认偏好。经过拱门也会重置。
    """
    def __init__(self, initial_deg=25.0, decay_rate=0.3):
        self.value = 0.0       # 当前偏置（deg），正=偏左转，负=偏右转
        self.initial = initial_deg  # fork 确定时的初值（deg）
        self.decay_rate = decay_rate  # 每帧衰减量（deg）

    def set_fork(self, direction):
        """fork 确定时调，设初值。正值=偏左转，负值=偏右转。"""
        if direction == "L":
            self.value = self.initial       # L 也用满幅，近距离分叉后才能拉回
        elif direction == "R":
            self.value = -self.initial      # 急弯时偏右转

    def tick(self):
        """每帧衰减偏置，向 0 逼近。"""
        if self.value > 0:
            self.value = max(0.0, self.value - self.decay_rate)
        elif self.value < 0:
            self.value = min(0.0, self.value + self.decay_rate)

    def clear(self):
        """清除偏置。"""
        self.value = 0.0

    def get(self):
        """读当前偏置（deg）。"""
        return self.value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="循迹主控 (六线程, 无GUI)")
    # --- 分割 ---
    p.add_argument("--model_dir", type=str, default=str(DEFAULT_SEG_DIR))  ## 默认不修改【使用默认model目录即可】
    p.add_argument("--TPEs", type=int, default=1, help="RKNN pool worker count (必须 1)") ## 【分核心设计，为后续接入目标检测模型做准备，所以使用默认值 1核心】
    p.add_argument("--det-core", type=int, default=2, help="检测模型 NPU core（seg=0, det=2, 各独占一核并行）") ## 【seg core 0 + det core 2】
    # --- IPM ---
    p.add_argument("--npz", type=str, default=str(DEFAULT_NPZ)) ## 默认不修改【使用默认 npz即可 】
    p.add_argument("--ipm-w", type=int, default=160)  ## 约定目标检测和语义分割最终输出图像为 160x120，便于算法计算
    p.add_argument("--ipm-h", type=int, default=120)  ## 约定目标检测和语义分割最终输出图像为 160x120，便于算法计算
    # --- 物理标定 ---
    p.add_argument("--track-width-cm", type=float, default=60.0)  ## 约定ipm输出图像对应轨道宽度，单位：厘米【便于算法计算】
    p.add_argument("--track-depth-cm", type=float, default=90.0)  ## 约定ipm输出图像对应轨道深度，单位：厘米【便于算法计算】
    # --- 解析 ---
    p.add_argument("--smooth", choices=("none", "local", "global"), default="global")  ## 【平滑模式】local
    p.add_argument("--global-smooth-window", type=int, default=5, help="global 平滑窗口；越大相位滞后越大→越抖，5 是推荐起点，不要加大")  ## 【实测：5→7→9 越来越抖，相位滞后导致振荡】
    p.add_argument("--offset-alpha", type=float, default=0.05, help="非线性偏置增益 α；0=线性，推荐 0.02 起步，上限 ~0.10")  ## 大偏置增强纠正力度
    # --- 控制 ---【可以更改！】
    p.add_argument("--base-speed", type=int, default=50)
    p.add_argument("--speed-ref", type=float, default=50.0)
    p.add_argument("--fork-prefer", choices=("L", "R", "none"), default="L")  ## 默认寻找大圈，仅在遇到岔路时如果有路牌API时才据此选择左转或右转
    # --- UART ---【默认开启，与下位机通讯，不可更改开启状态】
    p.add_argument("--uart-port", default="/dev/ttyUSB0")  ## 默认使用 /dev/ttyUSB0 串口【可选S3/USB1】
    p.add_argument("--uart-baud", type=int, default=115200)  ## 与下位机约定波特率为 115200
    p.add_argument("--uart-fps", type=float, default=30.0, help="Thread 4 固定下发频率为 30fps") ## 可以更改
    # --- 帧源 ---【可以更改！】
    p.add_argument("--flip_vertical", action="store_true", default=True)  ## 根据摄像头安装正反决定是否翻转垂直方向【当前3D结构下，默认翻转】
    p.add_argument("--idle-sleep", type=float, default=0.005) ## 空闲时 sleep 时间，单位：秒【默认 0.005s】
    # --- 日志 ---
    p.add_argument("--debug", action="store_true", default=False, help="开启终端心跳日志")  ## 默认不开启节省资源，调试时开启查看数据流
    p.add_argument("--heartbeat-s", type=float, default=1.0, help="心跳间隔，单位：秒")  ## 默认 1s
    p.add_argument("--uart-fail-max", type=int, default=90, help="UART 连续失败最大次数（30fps×3秒=90），超此判定断连")  ## 默认 90
    return p.parse_args()


def init_components(args):
    """初始化组件，返回 (source, segmenter, ipm, tracker, preview_extractor,
    command, uart, detector, api_config, fork_bias)。"""
    # ① 帧源
    source = SharedMemoryFrameSource(flip_vertical=args.flip_vertical, rgb_to_bgr=True)

    # ② 分割
    segmenter = PPSegInfer(model_dir=args.model_dir, TPEs=args.TPEs)

    # ③ IPM
    ipm = FastIPM(args.npz, out_w=args.ipm_w, out_h=args.ipm_h)

    # ④ 扫线
    tracker = LaneTracker(TrackerConfig(height=args.ipm_h, width=args.ipm_w))
    tracker.config.fork_prefer = args.fork_prefer if args.fork_prefer in {"L", "R"} else "L"

    # ⑤ 前瞻
    preview_cfg = PreviewConfig(
        track_width_cm=args.track_width_cm,
        track_depth_cm=args.track_depth_cm,
        smoothing_mode=args.smooth,
        global_smooth_window=args.global_smooth_window,  ## global 平滑窗口（弯道波动时加大）【消除弯道波动方案】
        dynamic_lookahead=False,  ## 【默认关闭动态前瞻，先固定行调参】
        offset_nonlinear_alpha=args.offset_alpha,  ## 非线性偏置映射（0=线性，>0 大偏置放大）
    )
    preview_extractor = PreviewExtractor(preview_cfg, image_width=args.ipm_w)

    # ⑥ 行为命令
    command = GuidanceCommand(
        fork_prefer=args.fork_prefer if args.fork_prefer in {"L", "R"} else "L",
        speed_ref=args.speed_ref,
        speed_limit=args.base_speed,
        run_enable=True,
        min_moving_speed_limit=50,
    )

    # ⑦ UART（在这里初始化，uart_worker 只负责发送）
    from xsmartcar.myuart.myuart import MyUART
    uart = MyUART(port=args.uart_port, baudrate=args.uart_baud)
    uart.send_C(1, args.base_speed)  ## 启动信号
    uart.send_L(0, 0, 0)             ## 舵机初始化

    # ⑧ 目标检测器（NPU core 1，和 seg 的 core 0 分开并行）
    detector = RKNNDetector(str(DEFAULT_DET_MODEL), core_id=args.det_core, conf_thr=0.7)

    # ⑨ 千帆 API 配置
    api_config = load_api_config()

    # ⑩ fork 状态（偏置叠加到 near_yaw，单位 deg）
    fork_bias = ForkBiasState(initial_deg=25.0)

    # ⑪ 车避障组件：检测目标 IPM 投影器（复用图像 IPM 的标定矩阵）
    det_ipm = FastDetectionIPM.from_ipm(ipm)

    return source, segmenter, ipm, tracker, preview_extractor, command, uart, \
           detector, api_config, fork_bias, det_ipm

def capture_worker(source, cap_q, det_cap_q, global_frame, stop_event, debug=False, heartbeat_s=1.0):
    """Thread 1: 读共享内存 → cap_q + det_cap_q（单帧源，双消费队列）。

        只负责喂帧，唯一调用 source.read() 的线程。
        同一帧对象放入两个 queue：cap_q 给 seg_worker，det_cap_q 给 det_worker。
        满时丢旧不丢新：get_nowait 取出最旧的丢掉，再 put 新的。
    """
    heartbeat_at = time.monotonic()
    frame_cnt = 0
    prev_cnt = 0  # 用于 debug 心跳算 FPS
    while not stop_event.is_set():
        frame = source.read()
        if frame is None:
            time.sleep(0.005)  # 无新帧，短睡
            continue
        frame_cnt += 1
        global_frame[0] += 1
        # → seg_worker
        if cap_q.full():
            try:
                cap_q.get_nowait()  # 丢旧帧
            except queue.Empty:
                pass
        cap_q.put(frame)  ## 满时丢旧不丢新

        # → det_worker（说明：同一帧对象，infer 内部 letterbox 会 copy，不会原地修改）
        if det_cap_q.full():
            try:
                det_cap_q.get_nowait()
            except queue.Empty:
                pass
        det_cap_q.put(frame)

        # debug 心跳
        now = time.monotonic()
        if debug and now - heartbeat_at >= heartbeat_s:
            fps = (frame_cnt - prev_cnt) / heartbeat_s
            print("─" * 50)
            print(f"[cap] fps={fps:.1f} frames={frame_cnt} cap_q={cap_q.qsize()} det_cap_q={det_cap_q.qsize()}")
            heartbeat_at = now
            prev_cnt = frame_cnt


def seg_worker(segmenter, cap_q, seg_q, stop_event, debug=False, heartbeat_s=1.0):
    """Thread 2: cap_q 取帧 → 分割推理 → seg_q。

    只做 NPU 推理，不碰 IPM/扫线/前瞻。
    NPU 速度由模型决定（约 20-30fps），不受 30fps 节拍约束。
    满时丢旧保证下游拿到最新分割。
    """
    heartbeat_at = time.monotonic()
    frame_cnt = 0
    prev_cnt = 0  # 用于 debug 心跳算 FPS
    while not stop_event.is_set():
        try:
            frame = cap_q.get(timeout=0.05)   # 等 50ms 新帧
            ## 定期检查 stop_event ，让线程能在 Ctrl+C 时退出【后面同理】
        except queue.Empty:
            continue

        t0 = time.monotonic()
        _, seg_map, ok = segmenter.infer_with_frame(frame)
        infer_ms = (time.monotonic() - t0) * 1000.0
        if not ok or seg_map is None:
            continue

        frame_cnt += 1
        if seg_q.full():
            try:
                seg_q.get_nowait()            # 丢旧分割
            except queue.Empty:
                pass
        seg_q.put(seg_map)

        # debug 心跳
        now = time.monotonic()
        if debug and now - heartbeat_at >= heartbeat_s:
            fps = (frame_cnt - prev_cnt) / heartbeat_s
            print(f"[seg] fps={fps:.1f} frames={frame_cnt} infer_ms={infer_ms:.1f} "
                  f"cap_q={cap_q.qsize()} seg_q={seg_q.qsize()}")
            heartbeat_at = now
            prev_cnt = frame_cnt

def det_worker(detector, det_cap_q, det_q, det_q2, stop_event, debug=False, heartbeat_s=1.0):
    """Thread 2b: det_cap_q 取帧 → 目标检测推理 → det_q + det_q2。

    和 seg_worker 并行，各占不同 NPU core。
    det_q 存 (frame, detections) 元组——decision_worker 调 API 时需要原图裁路牌小图。
    det_q2 存同一份 (frame, detections) 元组——avoid_worker 做车避障 IPM 投影用。
    满时丢旧保证下游拿到最新检测。
    """
    heartbeat_at = time.monotonic()
    frame_cnt = 0
    prev_cnt = 0
    while not stop_event.is_set():
        try:
            frame = det_cap_q.get(timeout=0.05)  # 等 50ms 新帧
        except queue.Empty:
            continue

        t0 = time.monotonic()
        detections = detector.infer(frame)
        infer_ms = (time.monotonic() - t0) * 1000.0

        frame_cnt += 1
        # → decision_worker
        if det_q.full():
            try:
                det_q.get_nowait()
            except queue.Empty:
                pass
        det_q.put((frame, detections))
        # → avoid_worker（同一份检测结果，avoid 只读 car 类）
        if det_q2.full():
            try:
                det_q2.get_nowait()
            except queue.Empty:
                pass
        det_q2.put((frame, detections))

        # debug 心跳
        now = time.monotonic()
        if debug and now - heartbeat_at >= heartbeat_s:
            fps = (frame_cnt - prev_cnt) / heartbeat_s
            det_summary = ",".join(f"{d.class_name}:{d.score:.2f}" for d in detections) or "-"
            print(f"[det]      fps={fps:4.1f} frm={frame_cnt:5d} ms={infer_ms:5.1f} "
                  f"capq={det_cap_q.qsize()} detq={det_q.qsize()} detq2={det_q2.qsize()} det={det_summary}")
            heartbeat_at = now
            prev_cnt = frame_cnt


def compute_worker(ipm, tracker, preview_extractor, command, fork_bias,
                   fork_active_flag, ar_offset_cm, track_offset_cm, roadsign_seen_flag,
                   seg_q, line_q, stop_event,
                   debug=False, heartbeat_s=1.0):
    """Thread 3: seg_q → IPM → 扫线 → 前瞻 → LineTarget → line_q.

    不发 UART！只产出 LineTarget 放入 line_q。
    运行速度由 seg_worker 喂帧速度决定（约 20-30fps）。
    满时丢旧保证下游拿到最新 LineTarget。

    避障接入（共享变量）：
      - 读 ar_offset_cm[0] → 写入 line_target.det_offset_cm（L 帧第三变量，由 avoid_worker 算）
      - 写 track_offset_cm[0] = line_target.near_offset（供 avoid_worker 读循迹线横向偏移做冲突感知）

    说明：FPS 计算： fps = (frame_cnt - prev_cnt) / heartbeat_s ，每次心跳打印当前帧率和 LineTarget 值
    """
    heartbeat_at = time.monotonic()
    frame_cnt = 0
    prev_cnt = 0  # 用于 debug 心跳算 FPS

    # ── 赛道丢失回正（救车）状态 ──
    # preview 无效（近端行左右边界都没找到）时，按最近一帧有效历史帧记录的边界丢失侧
    # 叠加固定回正偏置到 near_offset，尝试把车拉回赛道
    # 右边界丢(车偏左)→正偏置(下位机右转)往右回正；左边界丢(车偏右)→负偏置(下位机左转)往左回正
    # preview 再次有效 → 偏置立即消失；fork_bias 生效中不叠加（分叉由 fork_bias 负责）
    RECOVERY_OFFSET_CM = 10.0          # 固定回正偏置幅值（cm），叠加到 near_offset
    RECOVERY_OFF_THRESHOLD_CM = 1.0    # 双边界都在时按 offset 符号兜底的阈值（cm）
    last_valid_recovery_off = 0.0      # 最近有效帧算出的回正方向（cm），0=无方向
    while not stop_event.is_set():
        # 阻塞等新分割结果，最多等 100ms
        # timeout 是为了定期检查 stop_event，不是节拍控制
        try:
            seg_map = seg_q.get(timeout=0.1)
        except queue.Empty:
            continue

        # seg_map (二值 mask) → IPM 逆透视 → 鸟瞰图 ipm_mask (160×120)
        ipm_mask = ipm.process(seg_map, nearest=True)

        # 分叉倾向写入 tracker 配置（每帧都设，开销可忽略）
        if command.fork_prefer in {"L", "R"}:
            tracker.config.fork_prefer = command.fork_prefer

        # 鸟瞰图 → 扫线 → TrackResult（左右边界 + 中心线 + steer 点）
        track = tracker.process(ipm_mask, copy_result=False)

        # 写 fork_active 状态供 decision_worker 读取
        fork_active_flag[0] = track.fork.active

        # TrackResult → 前瞻观测 PreviewState（near_offset + near_yaw）
        # lateral_bias_cm=0.0 因为 GuidanceCommand 已删除该字段，直接传 0[默认就是0，不用传]
        preview = preview_extractor.compute(track)

        # PreviewState → LineTarget（纯几何映射，det_offset_cm 默认 0.0）
        line_target = compute_line_target(preview, preview_extractor.config)

        # ── 赛道丢失回正（救车）──
        # 右边界丢 → 车偏左 → +偏置(下位机右转)往右回正；左边界丢 → 车偏右 → -偏置(下位机左转)往左回正
        # T=强检测到边界；F/H=未检测到/贴边(视为丢)；W=中点猜测(不确定，不刷新方向)
        near_y = preview.near_y
        if 0 <= near_y < track.is_left_find.shape[0]:
            ls = str(track.is_left_find[near_y])
            rs = str(track.is_right_find[near_y])
            left_strong = (ls == "T")
            right_strong = (rs == "T")
            left_lost = ls in ("F", "H")
            right_lost = rs in ("F", "H")
        else:
            left_strong = right_strong = False
            left_lost = right_lost = False
        if preview.valid:
            # 赛道可用：刷新最近有效帧的回正方向，本帧不叠加
            if left_strong and right_lost:
                last_valid_recovery_off = RECOVERY_OFFSET_CM        # 右边界丢 → 往右回正
            elif right_strong and left_lost:
                last_valid_recovery_off = -RECOVERY_OFFSET_CM      # 左边界丢 → 往左回正
            elif left_strong and right_strong:
                # 双边界都在：按 offset 符号兜底（正=偏左→往右，负=偏右→往左）
                if line_target.near_offset > RECOVERY_OFF_THRESHOLD_CM:
                    last_valid_recovery_off = RECOVERY_OFFSET_CM
                elif line_target.near_offset < -RECOVERY_OFF_THRESHOLD_CM:
                    last_valid_recovery_off = -RECOVERY_OFFSET_CM
                else:
                    last_valid_recovery_off = 0.0                   # 居中，无需回正
            # 单边为 W（非 T 也非 F/H）但行仍有效：沿用上次方向，不刷新
            recovery_off_to_apply = 0.0                             # 赛道在 → 偏置消失
        else:
            # 赛道丢失：叠加最近有效帧记录的固定回正偏置救车
            # fork_bias 生效中不叠加（分叉由 fork_bias 负责，避免方向冲突）
            recovery_off_to_apply = last_valid_recovery_off if abs(fork_bias.get()) < 0.01 else 0.0

        if not command.stop_request and not command.finish_request:  # 发车后才叠加 fork 偏置（停车/完赛期间不拉偏）

            # fork 偏置叠加到 near_yaw（航向角）
            # near_yaw 正值=左转，负值=右转；fork_bias 正=偏左转，负=偏右转
            #

            # 动态放大：融合 offset（位置偏离）+ yaw（航向偏离）判断放大强度【融合yaw
            # 】
            #   R 转弯 fork_bias=-25deg：
            #     车偏左(off>0)+车头朝左(yaw>0) → 双重偏离 → 大力放大右转
            #     车偏左(off>0)+车头已朝右(yaw<0) → 正在纠正 → 适度放大
            #     车偏右(off<0)+车头朝右(yaw<0)  → 赛道在帮忙 → 不放大
            bias_val = fork_bias.get()
            if abs(bias_val) > 0.01:
                off_ratio = abs(line_target.near_offset) / preview_extractor.config.clamp_near_offset_cm
                yaw_ratio = abs(line_target.near_yaw) / preview_extractor.config.clamp_near_yaw_deg
                k = 1.0  # 放大增益
                # 偏置方向与偏离方向是否一致（需要纠正）
                off_against = (bias_val < 0 and line_target.near_offset > 0) or \
                              (bias_val > 0 and line_target.near_offset < 0)
                yaw_against = (bias_val < 0 and line_target.near_yaw > 0) or \
                              (bias_val > 0 and line_target.near_yaw < 0)
                if off_against and yaw_against:
                    # 双重偏离：位置+航向都偏了 → 大力纠正
                    scale = 1.0 + k * max(off_ratio, yaw_ratio)
                elif off_against or yaw_against:
                    # 单一偏离：只位置或只航向偏了 → 适度纠正
                    scale = 1.0 + k * 0.5 * (off_ratio + yaw_ratio)
                else:
                    # 无偏离或赛道已在帮忙 → 不放大
                    scale = 1.0
                bias_val *= scale
            raw = line_target.near_yaw + bias_val

            # ── 旧版：仅 offset 放大（无 yaw 融合），需要时注释掉上面新版即可切回 ──
            # bias_val = fork_bias.get()
            # if abs(bias_val) > 0.01:
            #     off_ratio = abs(line_target.near_offset) / preview_extractor.config.clamp_near_offset_cm
            #     k = 1.0
            #     same_sign = (bias_val < 0 and line_target.near_offset > 0) or \
            #                 (bias_val > 0 and line_target.near_offset < 0)
            #     scale = 1.0 + k * off_ratio if same_sign else 1.0
            #     bias_val *= scale
            # raw = line_target.near_yaw + bias_val

            # # fork 偏置叠加到 near_yaw（航向角），而非 near_offset【仅注释，时刻可恢复】
            # # near_yaw 正值=左转，负值=右转；fork_bias 正=偏左转，负=偏右转，符号一致
            # raw = line_target.near_yaw + fork_bias.get()

            line_target.near_yaw = max(-preview_extractor.config.clamp_near_yaw_deg,
                                       min(preview_extractor.config.clamp_near_yaw_deg, raw))

            # 叠加赛道丢失回正偏置到 near_offset（preview 无效时救车，有效时为 0 不影响）
            line_target.near_offset = max(-preview_extractor.config.clamp_near_offset_cm,
                                          min(preview_extractor.config.clamp_near_offset_cm,
                                              line_target.near_offset + recovery_off_to_apply))

        ## 读取避障偏移（cm，由 avoid_worker 写入）【避障偏置量写入】
        line_target.det_offset_cm = ar_offset_cm[0]
        ## 保存原始 near_offset 供 avoid_worker 冲突感知使用
        raw_near_offset = line_target.near_offset
        ## 避障时降低循迹线力度，让避障偏置在下位机加权叠加中占主导
        ## 冲突感知降权：循迹线与避障方向冲突时大幅降，同向时也降防叠加过冲
        ## 分叉时额外降更多（fork_active_flag 由 tracker 写入，分叉路段激活）
        if abs(ar_offset_cm[0]) > 1.0:
            near_off = line_target.near_offset
            det_off = ar_offset_cm[0]
            # 冲突：near_off>0(推右) + det_off<0(推左) 或 反之
            conflict = (near_off > 0 and det_off < 0) or (near_off < 0 and det_off > 0)
            if fork_active_flag[0]:
                line_ratio = 0.1                                 # 分叉：强降
            elif conflict:
                line_ratio = 0.2                                 # 冲突：降循迹线让避障主导
            else:
                line_ratio = 0.3                                 # 同向：适度降防叠加过冲【！！！】
            line_target.near_offset *= line_ratio
        ## 写入当前赛道偏移供 avoid_worker 读取（用于冲突感知衰减）
        ## 用原始 near_offset，避免影响 avoid_worker 冲突感知判断
        track_offset_cm[0] = raw_near_offset

        # 满时丢旧：line_q maxsize=1，compute 速度 > uart 消费速度时
        # 旧的 LineTarget 已经过时，丢掉只留最新的
        if line_q.full():
            try:
                line_q.get_nowait()
            except queue.Empty:
                pass
        line_q.put(line_target)

        frame_cnt += 1

        # debug 心跳：打印 FPS + 当前 LineTarget 值 + 队列深度
        now = time.monotonic()
        if debug and now - heartbeat_at >= heartbeat_s:
            fps = (frame_cnt - prev_cnt) / heartbeat_s
            print(f"[compute] fps={fps:.1f} off={line_target.near_offset:+.2f} "
                  f"yaw={line_target.near_yaw:+.2f} det={line_target.det_offset_cm:+.2f} "
                  f"v={int(preview.valid)} rec={recovery_off_to_apply:+.2f} line_q={line_q.qsize()}")
            heartbeat_at = now
            prev_cnt = frame_cnt


def avoid_worker(det_q2, ar_offset_cm, track_offset_cm, det_ipm,
                 track_width_cm, track_depth_cm,
                 stop_event, debug=False, heartbeat_s=1.0):
    """Thread 3c: 车避障 — det_q2 → IPM 投影 → 固定偏置 + 3帧确认 + 冲突感知衰减 → ar_offset_cm。

    方案说明：
      1. 检测到车（AVOID_TRIGGER_CM 内）→ 方向锁首次锁定，目标偏置 = 车半宽（CAR_HALF_WIDTH_CM）
         距离加权：远处 0.5 倍力度，近处满力度，线性插值
      2. 车消失 → 先 MISS_CONFIRM_FRAMES 帧确认（防漏检抖动），保持当前偏置不变
      3. 确认消失后开始衰减：每帧乘 DECAY
         - 冲突感知：读 track_offset_cm（循迹线横向偏移）
           循迹线与避障偏置方向冲突（一个推左一个推右）→ 衰减减速为 DECAY_SLOW
           让循迹线先把 ego 拉回中线，避免避障偏置抵消循迹力导致弯道冲出
           无冲突 → 正常 DECAY_FAST 快速回正
      4. 偏置归零（< DEAD_ZONE_CM）→ 清方向锁，等下次检测到车重新锁定

    共享变量契约：
      - 写 ar_offset_cm[0]（cm，正=推左，负=推右）→ compute_worker 读
      - 读 track_offset_cm[0]（cm，循迹线 near_offset）→ compute_worker 写
    """
    AVOID_TRIGGER_CM = 70.0       # 车横向距离触发阈值（cm）
    CAR_HALF_WIDTH_CM =1       # 障碍车半宽 + 安全余量（固定偏置量）【！！！！！】
    STEP_CM = 15.0                 # 有车时每帧最大偏置变化（防阶跃）
    MISS_CONFIRM_FRAMES = 3       # 车消失确认帧数（防漏检抖动）
    DECAY_FAST = 0.85              # 无冲突时衰减系数（每帧乘）
    DECAY_SLOW = 0.95              # 冲突时衰减系数（让循迹线先发力）
    DEAD_ZONE_CM = 0.5             # 偏置归零死区
    CONFLICT_THRESHOLD_CM = 1.0    # 循迹线偏移判定阈值（小于此值视为无偏移，不冲突）

    direction_lock = None           # 避障方向锁（+1.0=车在右→推左, -1.0=车在左→推右, None=未锁定）
    miss_count = 0                 # 车消失连续帧计数
    heartbeat_at = time.monotonic()

    while not stop_event.is_set():
        try:
            frame, detections = det_q2.get(timeout=0.1)
        except queue.Empty:
            continue

        car_dets = [d for d in detections if d.class_name == "car"]
        target = None  # None 表示不主动设目标（保持/衰减）

        if car_dets:
            # ── 有车：算目标偏置 ──
            miss_count = 0  # 重置漏检计数
            ar_objs, ok = detections_to_ar_objects(
                car_dets, det_ipm,
                frame_shape=frame.shape,
                track_width_cm=track_width_cm,
                track_depth_cm=track_depth_cm,
            )
            if ok and ar_objs:
                car_obj = min(ar_objs, key=lambda o: o.y_cm)
                if car_obj.y_cm <= AVOID_TRIGGER_CM:
                    # 方向锁：基于赛道视野判断，补偿 ego 横向偏移
                    # car_obj.x_cm = 车相对相机(ego)的位置（正=车在ego右）
                    # track_offset_cm[0] = ego相对赛道偏移（正=ego偏左）
                    # car_x_track = 车相对赛道的位置 = 车相对ego - ego偏移量
                    #   ego偏左(near_off>0)时减去，避免 ego 避让移动后方向锁误判
                    car_x_track = car_obj.x_cm - track_offset_cm[0]
                    if direction_lock is None:
                        direction_lock = 1.0 if car_x_track >= 0 else -1.0
                    # 距离加权：远处 0.5，近处 1.0
                    proximity = max(0.0, 1.0 - car_obj.y_cm / AVOID_TRIGGER_CM)
                    scale = 0.7 + 1 * proximity ##【！！！！】
                    # 固定偏置：方向锁决定符号，负 = 推左（车在右避左）
                    target = -direction_lock * CAR_HALF_WIDTH_CM * scale
            else:
                # IPM 失败兜底：用原始检测框中心判方向，固定 0.6 力度
                # 注：此处无 IPM 投影，无法精确补偿 ego 偏移，为粗略兜底
                det = car_dets[0]
                cx = 0.5 * (det.box_xyxy[0] + det.box_xyxy[2])
                if direction_lock is None:
                    direction_lock = 1.0 if cx >= frame.shape[1] / 2 else -1.0
                target = -direction_lock * CAR_HALF_WIDTH_CM * 0.6

            if target is not None:
                # 赛道边界限幅：det 是位移量，ego新位置 = near_off - target
                # 保证新位置 ∈ [-half_w, +half_w]，留 3cm 安全余量
                # det>0推右 → 限制 det <= near_off + half_w - 3（防冲出右边界）
                # det<0推左 → 限制 det >= near_off - half_w + 3（防冲出左边界）
                half_w = track_width_cm / 2.0
                near_off = track_offset_cm[0]
                target = max(near_off - half_w + 3.0,
                             min(near_off + half_w - 3.0, target))
                # 步长限制，防阶跃
                delta = max(-STEP_CM, min(STEP_CM, target - ar_offset_cm[0]))
                ar_offset_cm[0] += delta
        else:
            # ── 无车：先确认再衰减 ──
            miss_count += 1
            if miss_count <= MISS_CONFIRM_FRAMES:
                # 确认期：保持当前偏置不变（防 1-2 帧漏检抖动）
                pass
            else:
                # 确认消失：冲突感知衰减
                det_off = ar_offset_cm[0]
                near_off = track_offset_cm[0]
                # 判断循迹线与避障偏置是否方向冲突
                # near_off > 0 表示循迹线在右拉（ego 偏左），det_off < 0 表示避障推左
                # 两者方向相反 → 互相抵消 → 冲突
                conflict = ((near_off > CONFLICT_THRESHOLD_CM and det_off < 0) or
                            (near_off < -CONFLICT_THRESHOLD_CM and det_off > 0))
                decay = DECAY_SLOW if conflict else DECAY_FAST
                ar_offset_cm[0] *= decay
                # 死区归零 + 清方向锁
                if abs(ar_offset_cm[0]) < DEAD_ZONE_CM:
                    ar_offset_cm[0] = 0.0
                    direction_lock = None

        # debug 心跳
        now = time.monotonic()
        if debug and now - heartbeat_at >= heartbeat_s:
            near_off = track_offset_cm[0]
            det_off = ar_offset_cm[0]
            conflict = ((near_off > CONFLICT_THRESHOLD_CM and det_off < 0) or
                        (near_off < -CONFLICT_THRESHOLD_CM and det_off > 0))
            cars_str = "-"
            if car_dets:
                # 简单打印车数 + 方向锁
                cars_str = f"n={len(car_dets)} lock={'R' if direction_lock and direction_lock > 0 else 'L' if direction_lock else '-'}"
            print(f"[avoid] {cars_str} offset={det_off:+.2f}cm miss={miss_count} "
                  f"near_off={near_off:+.1f} {'C' if conflict else '-'}")
            heartbeat_at = now


def decision_worker(det_q, command, fork_bias, api_config, fork_active_flag, roadsign_seen_flag, stop_event,
                    confirm_frames=3, debug=False, heartbeat_s=0.1):
    """Thread 3b: 路牌 → API → fork_bias → 分叉口偏置。

    规则：
      1. 连续 confirm_frames 帧检测到路牌 → 停车 + 后台调 API
      2. 有行人在路牌前 → 发车需同时满足 API 已返回 + 无障碍
      3. 仅路牌无行人 → API 返回后正常发车
      4. 经过拱门 → 清 api_result，清除分叉偏好，允许下一圈重新调 API
      5. fork_bias 每帧衰减，约 0 时还原默认
    """
    from concurrent.futures import ThreadPoolExecutor

    api_executor = ThreadPoolExecutor(max_workers=1)

    default_fork = command.fork_prefer
    api_future = None
    api_submit_time = 0.0
    API_TIMEOUT_S = 15.0

    roadsign_consecutive = 0
    api_pending = False
    api_result = None
    fork_bias_applied = False
    roadsign_handled = False       # 本圈路牌已处理过，防止同一路牌重复触发 API
    roadsign_armed = True          # 路牌触发允许标志；触发后置 False，必须先离开视野再出现才重新置 True

    fork_not_active_count = 0   # fork_active 消失帧计数（⑥ 用）
    FORK_CLEAR_FRAMES = 25      # 连续 N 帧无 fork → 清偏好【低速】
    fork_seen = False           # fork_active 是否曾出现过（⑥ 的前提，修复路牌前被误清）

    # ── 完赛停车 ──
    # 已移除：完赛停车逻辑已迁移到 xsmartcar/unused/finish_detector.py
    # 如需恢复：import + 实例化 FinishDetector + 每帧调 update(detections, image_height)

    # 偏好持续时间统计【仅用于调试统计】
    fork_prefer_set_at = None   # 偏好设定时刻（monotonic），None=未设
    fork_prefer_direction = None  # 当前偏好方向 "L"/"R"，None=未设

    heartbeat_at = time.monotonic()

    while not stop_event.is_set():
        try:
            frame, detections = det_q.get(timeout=0.1)
        except queue.Empty:
            continue

        roadsign_dets = [d for d in detections if d.class_name == "roadsign"]
        has_roadsign = len(roadsign_dets) > 0
        roadsign_consecutive = roadsign_consecutive + 1 if has_roadsign else 0

        # 本圈见过路牌 → 置 True，标记为 API 场景【过完分叉口/拱门才重置】
        if has_roadsign:
            roadsign_seen_flag[0] = True

        # roadsign_armed 边沿防抖：路牌离开视野后才允许下次触发
        if not has_roadsign and not roadsign_armed:
            roadsign_armed = True

        # 路牌停车逻辑：需满足 roadsign_handled=False + roadsign_armed=True 防重复触发
        #    新增：必须同时检测到分叉口 fork_active，确保路牌在分叉区域才停车调 API
        #    避免远处误识别路牌导致无故停车
        #    roadsign_armed：触发后置 False，路牌离开视野（has_roadsign=False 至少 1 帧）才重新置 True
        #    防止：⑥/⑦ 解锁 roadsign_handled 后路牌仍在视野 → 立刻再停车
        if (roadsign_consecutive >= confirm_frames and not api_pending
                and api_result is None and not roadsign_handled and roadsign_armed
                and fork_active_flag[0]):  # ← 关键：分叉激活时才停车调 API
            api_pending = True
            roadsign_handled = True   # 锁住，直到拱门或 fork_active 消失清偏好后才解锁
            roadsign_armed = False    # 锁住，路牌必须先离开再出现才允许下次触发
            command.stop_request = True
            # 取第一个路牌：检测器按 score 降序排列，[0] 即最高置信度的路牌，
            # 用于裁剪送 API；若多路牌场景需改取面积最大或距车最近的
            best_roadsign = roadsign_dets[0]
            api_future = api_executor.submit(call_roadsign_api, frame, best_roadsign.box_xyxy, api_config)
            api_submit_time = time.monotonic()

        # 行人+路牌：底边 y > 路牌底边 y → 行人在路牌前方（更近）
        has_pedestrian_front = False
        if roadsign_dets:
            roadsign_bottom_y = max(d.box_xyxy[3] for d in roadsign_dets)
            has_pedestrian_front = any(
                d.class_name == "human" and d.box_xyxy[3] > roadsign_bottom_y
                for d in detections
            )

        # API 结果轮询：非阻塞检查 future 状态
        #   done() → API 正常返回（L/R/none），异常时 fallback "none"
        #   超时 >8s → 强制终止等待，fallback "none"
        #   两路都会：存 api_result + 清 api_pending，后续 ③ 据此设偏置
        if api_future is not None:
            if api_future.done():
                try:
                    api_result = api_future.result()
                except Exception as e:
                    if debug:
                        print(f"[decision] API error: {e}")
                    api_result = "none"
                api_future = None
                api_pending = False
            elif time.monotonic() - api_submit_time > API_TIMEOUT_S:
                if debug:
                    print(f"[decision] API timeout ({API_TIMEOUT_S}s)")
                api_result = "none"
                api_future = None
                api_pending = False

        # ③ 分叉偏好：API 返回 L/R 才设偏置 + 写 fork_prefer 驱动 tracker
        #    超时/错误 → api_result="none" → 不进此分支 → 沿用 default_fork (L)
        #    即超时默认走左（大圈），合理
        if api_result in ("L", "R") and not fork_bias_applied:
            fork_bias.set_fork(api_result)
            command.fork_prefer = api_result
            fork_bias_applied = True

            fork_prefer_set_at = time.monotonic()  # 记录偏好设定时刻
            fork_prefer_direction = api_result

        # fork_active 首次出现 → 标记 fork_seen
        #    修复：路牌前停车阶段 fork_active 一直 False，⑥ 不应在此阶段清偏好
        if fork_bias_applied and not fork_seen and fork_active_flag[0]:
            fork_seen = True

        # ④ 发车判定：
        #    有路牌前行人 → 必须 can_start（避障放行）才能发车
        #    无行人 → API 返回即可发车（stop_request=False 解除停车）
        #    注意：发车 ≠ 舵机立刻有偏置输出，compute_worker 在 stop_request
        #    期间不叠加 fork_bias，所以发车瞬间偏置才生效，可能出现短暂空转向
        if command.stop_request and api_result is not None:
            if has_pedestrian_front:
                if command.can_start:
                    command.stop_request = False
            else:
                command.stop_request = False

        # ⑤ 偏置衰减（暂时禁用，保留代码随时可恢复）
        # if not command.stop_request and fork_bias_applied:
        #     fork_bias.tick()
        #     if abs(fork_bias.get()) < 0.1:
        #         fork_bias_applied = False
        #         fork_bias.value = 0.0

        # ⑥ 偏置清除：fork_seen 后，fork_active 消失 N 帧 → 说明已过完分叉口 → 清偏好
        #    兼容小圈（无拱门）：不依赖 arch，靠 fork_active 信号消失来清
        #    大圈额外有 arch 兜底（⑦）
        #    关键：fork_seen=False 时（路牌前停车阶段）不会误清偏好
        #    关键2：拱门存在期间不清偏置（拱门附近 fork_active 可能闪烁，误清导致 R→L 失败）
        has_arch_now = any(d.class_name == "arch" for d in detections)
        if fork_bias_applied and fork_seen and not has_arch_now:
            if fork_active_flag[0]:
                fork_not_active_count = 0
            else:
                fork_not_active_count += 1
                if fork_not_active_count >= FORK_CLEAR_FRAMES:
                    fork_bias.clear()
                    command.fork_prefer = default_fork
                    fork_bias_applied = False
                    api_result = None
                    roadsign_handled = False  # 解锁，允许下一圈重新调 API
                    fork_seen = False
                    fork_not_active_count = 0
                    fork_prefer_set_at = None  # 重置持续时间统计
                    fork_prefer_direction = None
                    roadsign_seen_flag[0] = False  # 过完分叉口，重置 API 场景标志

        # ⑦ 拱门兜底：经过拱门时的清除策略
        #    关键：拱门可能在分叉口附近与 fork_active 同时出现，
        #    此时 fork 偏置还在生效，不能清偏好（否则 R→L 转向失败）
        #
        #    分两种情况：
        #    A. fork 偏置已生效（fork_bias_applied=True）→ 只清 api_result，
        #       允许下一圈重新调 API；偏置和偏好保留，等 ⑥（fork_active 消失）来清
        #    B. fork 偏置未生效（fork_bias_applied=False）→ 全部清零，
        #       回到初始状态（上一圈已过完分叉口，或 API 超时走了默认方向）
        if any(d.class_name == "arch" for d in detections):
            if fork_bias_applied:
                # A: 偏置生效中，拱门只清 api_result（解锁下一圈 API 调用）
                #    偏置 + 偏好 + fork_seen 保留，等 ⑥ 来清
                api_result = None
                roadsign_handled = False  # 解锁，允许下一圈重新调 API
            else:
                # B: 偏置未生效，全清回到初始状态
                fork_bias.clear()
                command.fork_prefer = default_fork
                fork_bias_applied = False
                api_result = None
                roadsign_handled = False
                fork_seen = False
                fork_not_active_count = 0
                fork_prefer_set_at = None
                fork_prefer_direction = None
                roadsign_seen_flag[0] = False  # 过拱门，重置 API 场景标志

        # debug 心跳
        now = time.monotonic()
        if debug and now - heartbeat_at >= heartbeat_s:
            api_state = "pending" if api_pending else "idle"
            api_result_str = api_result or "-"
            # 偏好持续时间
            if fork_prefer_set_at is not None:
                prefer_dur = now - fork_prefer_set_at
                prefer_info = f" {fork_prefer_direction}@{prefer_dur:.1f}s"
            else:
                prefer_info = " -"
            print(f"[decision] api={api_state} result={api_result_str} "
                  f"prefer={command.fork_prefer} default={default_fork} "
                  f"roadsign={roadsign_consecutive}/{confirm_frames} "
                  f"ped={int(has_pedestrian_front)} "
                  f"bias={fork_bias.get():+4.1f}deg applied={int(fork_bias_applied)}"
                  f" 检测到分叉口 fork_seen={int(fork_seen)} 非分叉no_fork={fork_not_active_count}"
                  f" dur={prefer_info}")
            heartbeat_at = now
    api_executor.shutdown(wait=False)


def uart_worker(uart, args, line_q, command, stop_event):
    """Thread 4: 30fps 固定节拍 → 取最新 LineTarget → 发 L + C。

    UART 在 init_components 里初始化，这里只负责发送。
    无新 LineTarget 时重发缓存，保证 30fps 下发。
    send_L/send_C 返回 None = myuart 内部重连也失败了，连续失败超 fail_max 判定断连。

    ── C 帧决策逻辑（control_values 优先级：finish > stop > run_enable）──
    command 状态            返回值       发的帧       含义
    finish_request=True     (0, 0)      C,0,0       完赛：下位机停电机
    stop_request=True       (1, 0)      C,1,0       紧急停：保持 run 但速度 0
    run_enable=True         (1, speed)  C,1,100     正常运行
    run_enable=False        (0, 0)      C,0,0       停车但不是完赛
    全部 None               None        不发 C      —

    ── run_enable 字段说明 ──
    run_enable 是 GuidanceCommand 的「运行允许」标志：
    - True  → control_values 返回 (1, speed_limit)，下位机正常行驶
    - False → control_values 返回 (0, 0)，下位机停电机
    当前初始化为 True（init_components 里写死）。
    后续接入避障 / API 时，外部线程可修改 command.run_enable 来紧急停车，
    uart_worker 下一个 tick 就会发 C,0,0。

    ── 后续避障 / API 接入点 ──
    1. 避障线程（二选一，不要同时设）：
       - 检测到障碍需暂停 → command.stop_request=True（C,1,0 保持run但速度0，可恢复）
       - 紧急停机 → command.run_enable=False（C,0,0 停电机，不可恢复）
       注意：若同时设，stop 优先级高于 run_enable，stop_request 会覆盖 run_enable
    2. 赛程 API：收到完赛信号 → command.finish_request=True（最高优先级 C,0,0）
    3. 避障偏移：写入 line_target.det_offset_cm（由 compute_worker 填，当前固定 0.0）
       避障线程若要覆盖，需通过共享变量传给 compute_worker，不能直接改 line_q 里的对象
    """
    period = 1.0 / args.uart_fps
    latest_line = None
    latest_c = None  # 最近一次 C 帧值 (run_is, speed_limit)
    fail_count = 0  # 连续失败计数（成功就清零）
    heartbeat_at = time.monotonic()

    next_tick = time.monotonic() + period
    while not stop_event.is_set():
        # ① 非阻塞排空 line_q，只保留最新（丢弃过时的 LineTarget）
        while True:
            try:
                latest_line = line_q.get_nowait()
            except queue.Empty:
                break

        # ② 发 L（有缓存就发，没有就跳过等下一 tick）
        if latest_line is not None:
            ret = uart.send_L(latest_line.near_offset,
                               latest_line.near_yaw,
                               latest_line.det_offset_cm)
            if ret is None:
                fail_count += 1
                if fail_count >= args.uart_fail_max:
                    uart.stop()  # 紧急停车
                    raise RuntimeError(
                        f"UART 连续失败 {fail_count} 次 >= {args.uart_fail_max}，判定断连"
                    )
            else:
                fail_count = 0  # 成功就清零

        # ③ C 帧（control_values 返回 None 表示不需要发）
        values = control_values(command)
        if values is not None:
            latest_c = values
            uart.send_C(*values)  # C 帧失败不计数，L 帧才是主链

        # ④ debug 心跳 + tick 对齐
        now = time.monotonic()
        if args.debug and now - heartbeat_at >= args.heartbeat_s:
            line = latest_line
            preview_str = (f"off={line.near_offset:+.2f} yaw={line.near_yaw:+.2f} "
                           f"det={line.det_offset_cm:+.2f}") if line else "no_line"
            c_str = f"C,{latest_c[0]},{latest_c[1]}" if latest_c else "no_C"
            print(f"[uart] {preview_str} {c_str} line_q={line_q.qsize()} fails={fail_count}")
            heartbeat_at = now

        # tick 对齐（让出 CPU，保证固定 30fps）
        sleep_s = next_tick - now
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = now
        next_tick += period

    # 退出时停车：重试 3 次，确保 C,1,0 发出去
    for attempt in range(3):
        ret = uart.send_C(1, 0)
        if ret is not None:
            break
        time.sleep(0.05)
    try:
        uart.close()
    except Exception:
        pass



def _signal_handler(signum, frame, stop_event=None):
    """Ctrl+C 信号处理：设 stop_event 让六线程退出。"""
    if stop_event is not None:
        stop_event.set()
    print("\n[main] 收到退出信号，正在停止...")


def safe_cleanup(source, segmenter, uart, detector, threads, stop_event):
    """安全退出：设 stop_event → join 线程 → 释放资源。

    无论正常退出还是异常退出都会被调用（放在 main 的 finally 块里）。
    每步都 try/except 保证一个失败不阻塞后续清理。

    资源释放审计：
        source     (SharedMemoryFrameSource) → close()   释放共享内存映射  ✓
        segmenter  (PPSegInfer)              → release() 释放 NPU runtime  ✓
        detector   (RKNNDetector)            → release() 释放 NPU runtime  ✓
        uart       (MyUART)                  → close()   先发 C,1,0 再关串口 ✓
        ipm        (FastIPM)                 → 无需释放（纯 numpy 矩阵）
        tracker    (LaneTracker)            → 无需释放（纯 numpy 计算）
        preview_extractor                     → 无需释放（纯计算）
    释放顺序：shm → NPU(seg+det) → UART（先放上游再放下游）
    uart.close() 内部调 self.stop() 发 C,1,0 停车帧，即使 uart_worker 异常退出也能兜底。
    """
    # ① 设停止信号，让所有 worker 的 while not stop_event.is_set() 退出循环
    stop_event.set()

    # ② join 线程（2s 超时，避免卡死）
    for t in threads:
        t.join(timeout=2.0)
        if t.is_alive():
            print(f"[main] {t.name} join 超时，强制继续")

    # ③ 释放资源（顺序：shm → NPU(seg+det) → UART，每步 try/except 互不阻塞）
    for obj, method, name in [
        (source, "close", "source"),         # 释放共享内存映射
        (segmenter, "release", "segmenter"), # 释放 NPU runtime (seg core 0)
        (detector, "release", "detector"),   # 释放 NPU runtime (det core 1)
        (uart, "close", "uart"),            # 先发 C,1,0 停车再关串口（uart_worker 退出时已调过，这里兜底）
    ]:
        try:
            getattr(obj, method)()
        except Exception as e:
            print(f"[main] {name} 释放失败: {e}")

    print("[main] 已安全退出")


def main():
    args = parse_args()
    source, segmenter, ipm, tracker, preview_extractor, command, uart, \
        detector, api_config, fork_bias, det_ipm = init_components(args)

    # 队列：cap_q(3) → seg_q(1) → line_q(1)
    #        det_cap_q(1) → det_q(1)（目标检测独立管道）
    #        det_cap_q(1) → det_q2(1)（avoid_worker 独立管道）
    cap_q = queue.Queue(maxsize=3)
    seg_q = queue.Queue(maxsize=1)
    line_q = queue.Queue(maxsize=1)
    det_cap_q = queue.Queue(maxsize=1)  # capture → det_worker
    det_q = queue.Queue(maxsize=1)      # det_worker → decision_worker
    det_q2 = queue.Queue(maxsize=1)     # det_worker → avoid_worker

    # 共享变量：global_frame（全局帧计数）+ fork_active_flag（tracker 分叉状态）
    global_frame = [0]
    fork_active_flag = [False]
    ar_offset_cm = [0.0]      ## 避障偏移量（cm）【avoid_worker 写，compute_worker 读】
    track_offset_cm = [0.0]   ## 赛道偏移（near_offset）【compute_worker 写，avoid_worker 读】
    roadsign_seen_flag = [False]  ## 本圈是否见过路牌【decision_worker 写，compute_worker 读】
                                   ## True=API场景(有路牌的分叉)，False=普通分叉(无路牌)

    # ──────────────────────────────────────────────────────────────
    # stop_event：六线程共享的「停止信号」（不是完赛停车标志！）
    #   - stop_event.set()     → 内部标志设为 True
    #   - stop_event.is_set()  → 检查标志是否为 True
    #   - 所有 worker 循环都是 while not stop_event.is_set():
    #
    # 触发路径（5 种，殊途同归 → stop_event.set()）：
    #   ① Ctrl+C              → signal handler → stop_event.set()
    #   ② kill 命令 (SIGTERM)  → signal handler → stop_event.set()
    #   ③ worker 线程异常退出  → main while 循环检测 not t.is_alive() → stop_event.set()
    #   ④ main 主循环异常      → except Exception → stop_event.set()
    #   ⑤ safe_cleanup 兜底    → stop_event.set()（防止 ①~④ 漏设）
    #
    # 完赛停车走的是另一条路：
    #   command.finish_request = True → control_values() 返回 (0,0) → uart_worker 发 C,0,0
    # ──────────────────────────────────────────────────────────────
    stop_event = threading.Event()

    # 注册 Ctrl+C / kill 信号 → 优雅退出入口（没有这段 Ctrl+C 直接杀进程，下位机不会收到 C,1,0 停车帧）
    import signal
    signal.signal(signal.SIGINT, lambda s, f: _signal_handler(s, f, stop_event))
    signal.signal(signal.SIGTERM, lambda s, f: _signal_handler(s, f, stop_event))

    dbg = args.debug
    hbs = args.heartbeat_s

    # 启动七线程
    ## 完赛停车 走的是 command.finish_request = True → control_values() 返回 (0, 0)
    ##  → uart_worker 发 C,0,0 给下位机停车。
    threads = [
        threading.Thread(target=capture_worker,
                         args=(source, cap_q, det_cap_q, global_frame, stop_event, dbg, hbs), daemon=True),
        threading.Thread(target=seg_worker,
                         args=(segmenter, cap_q, seg_q, stop_event, dbg, hbs), daemon=True),
        threading.Thread(target=det_worker,
                         args=(detector, det_cap_q, det_q, det_q2, stop_event, dbg, hbs), daemon=True),
        threading.Thread(target=compute_worker,
                         args=(ipm, tracker, preview_extractor, command,
                               fork_bias, fork_active_flag, ar_offset_cm, track_offset_cm, roadsign_seen_flag,
                               seg_q, line_q, stop_event, dbg, hbs), daemon=True),
        threading.Thread(target=decision_worker,
                         args=(det_q, command, fork_bias, api_config,
                               fork_active_flag, roadsign_seen_flag, stop_event, 2, dbg, hbs), daemon=True),
        threading.Thread(target=avoid_worker,
                         args=(det_q2, ar_offset_cm, track_offset_cm, det_ipm,
                               args.track_width_cm, args.track_depth_cm,
                               stop_event, dbg, hbs), daemon=True),
        threading.Thread(target=uart_worker,
                         args=(uart, args, line_q, command, stop_event), daemon=True),
    ]

    for t in threads:
        t.start()
    print(f"[main] 七线程已启动。uart={args.uart_port} fps={args.uart_fps} "
          f"ipm={args.ipm_w}x{args.ipm_h} fork={args.fork_prefer} debug={dbg}")

    # 主线程等待：stop_event 被设 或 任意 worker 线程异常退出
    try:
        while not stop_event.is_set():
            for t in threads:
                if not t.is_alive() and not stop_event.is_set():
                    print(f"[main] 线程 {t.name} 异常退出")
                    stop_event.set()
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
    except Exception as exc:
        print(f"[main] 主线程异常: {exc}")
        stop_event.set()
    finally:
        safe_cleanup(source, segmenter, uart, detector, threads, stop_event)

    return 0


if __name__ == "__main__":
    main()

"""
=== 后续避障接入方式（参考） ===

# 避障线程往 line_q 里的 LineTarget 写 det_offset_cm
line_target.det_offset_cm = det_offset  # 来自 det 线程，无就 0.0

# uart_worker 自动合并最新「寻线+避障」后下发
uart.send_L(line_target.near_offset, line_target.near_yaw, line_target.det_offset_cm)

# C 帧由 uart_worker 根据 command 状态自动发
values = control_values(command)
if values is not None:
    uart.send_C(*values)
"""