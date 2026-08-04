# Copyright (c) 2026 清影/123twtd
"""Race 前瞻提取和目标计算工具。

== 职责 ==
1. 从 TrackResult 提取几何观测（PreviewExtractor.compute）
2. 把观测值转换为下发目标（compute_line_target）
3. 计算可选的 C 帧值（control_values / build_control_frame）

不负责：扫线（LaneTracker）、分割推理、IPM、帧发送、线程管理、避障合并。
这些都在 run_track.py 中。

== 四个数据类 ==
1. PreviewConfig  — 调参旋钮      — 物理标定、前瞻行位置、平滑、限幅
2. PreviewState   — compute() 输出  — near_offset(cm)、near_yaw(度)、near_y、valid
3. LineTarget     — 最终 L 帧包     — near_offset(cm)、near_yaw(度)、det_offset_cm(cm)、source
4. GuidanceCommand — C 帧控制状态   — fork_prefer、speed_ref、speed_limit、run_enable

== 数据流（不开启动态前瞻）==
    TrackResult (来自 LaneTracker, 160x120 鸟瞰图)
        │
        ├─ near_y = (h-1) × near_y_ratio           # 固定行，不随速度变
        ├─ base_y = (h-1) × yaw_base_y_ratio       # 固定基准行
        │
        ├─ near_center = 第 near_y 行的中线 x 坐标（带平滑）
        ├─ base_center = 第 base_y 行的中线 x 坐标（带平滑）
        │
        ├─ near_offset = -(near_center - 图像中心x) × cm_per_px_x  # 取负：正=偏左（需右转）
        ├─ near_yaw = angle_deg(near_center_cm - base_center_cm, 行高差 × cm_per_px_y)  # 不取负：原始计算值
        │
        └─ PreviewState(near_offset, near_yaw, near_y, valid)
            └─ compute_line_target() → LineTarget(det_offset_cm=0.0)
                └─ [run_track.py 填入 det_offset_cm]
                    └─ uart.send_L(off, yaw, det)

不开动态前瞻只需调：near_y_ratio、yaw_base_y_ratio（上位机几何参数，和下位机无关）。

== 数据流（开启动态前瞻）==
    和上面一样，唯一区别在 near_y 的计算：
        near_y = (h-1) × near_y_ratio - (speed_ref - default_speed_ref) × speed_to_near_y_px
        速度 > 基准 → near_y 变小 → 往上看更远
        速度 < 基准 → near_y 变大 → 往下看更近
    base_y 不受速度影响，始终固定。

开启动态前瞻需调：speed_to_near_y_px（灵敏度）、default_speed_ref（基准速度）。

== GuidanceCommand 字段调用关系 ==
    fork_prefer:  run_track.py → tracker.config.fork_prefer    写：初始化/API/路牌
    speed_ref:    run_track.py → compute(speed_ref=...)         写：初始化/动态调
    speed_limit:  control_values() 读                           写：初始化/动态调
    min_moving_speed_limit: control_values() 读                 写：初始化
    run_enable:   control_values() 读                           写：初始化
    stop_request: control_values() 读                           写：异常处理/API
    finish_request: control_values() 读                         写：完赛信号/API

== 外部完整调用示例 ==
    # 初始化
    cfg = PreviewConfig(track_width_cm=60, track_depth_cm=90)
    extractor = PreviewExtractor(cfg, image_width=160)
    cmd = GuidanceCommand(speed_ref=75, speed_limit=100, run_enable=True)

    # 每帧
    track = tracker.process(ipm_mask)
    state = extractor.compute(track, speed_ref=cmd.speed_ref)
    line_target = compute_line_target(state, cfg)
    line_target.det_offset_cm = det_offset  # run_track.py 填避障结果

    # 下发
    uart.send_L(line_target.near_offset, line_target.near_yaw, line_target.det_offset_cm)
    values = control_values(cmd)
    if values is not None:
        uart.send_C(*values)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, copysign, degrees
from typing import Literal

import numpy as np

from .trk_tracker_module import ForkPrefer, TrackResult

SmoothMode = Literal["none", "local", "global"]


@dataclass(slots=True)
class GuidanceCommand:
    """C 帧控制状态。

    只影响 C 帧（run/stop/speed），不影响 L 帧的 offset/yaw。
    避障偏移不走这里，直接写入 LineTarget.det_offset_cm。
    """
    fork_prefer: ForkPrefer | None = "L"        # 分叉倾向，传给 LaneTracker
    speed_ref: float | None = None              # 动态前瞻参考速度
    speed_limit: int | None = None              # C 帧速度上限
    min_moving_speed_limit: int = 40            # 最低行驶速度（挡住下位机降速到 0）
    run_enable: bool | None = None              # C 帧 run/stop
    stop_request: bool = False                  # 紧急停车
    finish_request: bool = False                # 完赛停车
    can_start: bool = True                      # 发车许可（避障接口，默认允许）
    lateral_bias_cm: float = 0.0                # 相机/车体横向标定偏置
    control_yaw_bias_deg: float = 0.0           # 避障叠加到 L 帧 yaw 的偏置


@dataclass(slots=True)
class PreviewConfig:
    """前瞻提取配置。

    调参分组：
    1. 物理标定：track_width_cm, track_depth_cm
    2. 前瞻行位置：near_y_ratio, yaw_base_y_ratio
    3. 动态前瞻：dynamic_lookahead, speed_to_near_y_px, default_speed_ref
    4. 平滑和有效行规则：smoothing_mode, local_window, offline_margin
    5. 输出限幅：clamp_near_offset_cm, clamp_near_yaw_deg
    """

    default_speed_ref: float = 50.0                 # 基准速度，等于这个时 near 行位置不变
    ## 物理标定
    track_width_cm: float = 60.0                    # IPM 宽方向对应的真实赛道宽度
    track_depth_cm: float = 90.0                    # IPM 高方向对应的真实赛道深度

    ## 前瞻行位置
    near_y_ratio: float = 0.82                      # 近端行比例。1.0=最底（最近），0=最顶（最远）
    yaw_base_y_ratio: float = 0.90                  # 偏航角基准行，比 near 更靠下

    ## 动态前瞻
    dynamic_lookahead: bool = False  # 开启后速度越快 near 行越往上（看更远）
    speed_to_near_y_px: float = 0.20 # 速度每比基准多 1，near 行往上移多少像素

    ## 平滑和有效行
    smoothing_mode: SmoothMode = "local"  # none=不平滑, local=局部窗口平均, global=全局平滑
    local_window: int = 5                # local 平滑窗口大小
    global_smooth_window: int = 7        # global 平滑窗口大小
    offline_margin: int = 1              # 有效行与 offline 的最小间距（像素）

    ## 输出限幅（协议安全）
    clamp_near_offset_cm: float = 30.0  # offset 限幅 30【校内可以】，45
    clamp_near_yaw_deg: float = 45.0    # yaw 限幅

    ## 非线性偏置映射（大偏置增强纠正力度，缓解单边/大偏差纠正不足）
    # 公式：offset_out = offset × (1 + α × |offset|)
    # α=0 时线性（默认，不放大）；α>0 时大偏置放大，小偏置几乎不变
    # 推荐 α=0.02 起步，上限 ~0.10；过大 → 小偏置也过敏 → 直道抖动
    offset_nonlinear_alpha: float = 0.05


@dataclass(slots=True)
class PreviewState:
    """前瞻观测结果。"""

    near_offset: float    # 车偏离赛道中心多少 cm，正=偏左（需右转），负=偏右（需左转）
    near_yaw: float       # 近端行偏航角（度），原始计算值（未取负）
    near_y: int           # 近端行在 IPM 图中的行号
    valid: bool           # 这帧观测是否有效
    valid_reason: str     # 无效时的原因


@dataclass(slots=True)
class LineTarget:
    """LineTarget 是最终 L 帧包"""

    near_offset: float
    near_yaw: float
    det_offset_cm: float = 0.0   # 避障横向偏移，0.0 = 无避障介入
    source: str = "preview"       # 标记目标来源，用于外部区分


def angle_deg(dx_cm: float, dy_cm: float) -> float:
    """ 算偏航角
        计算两段 cm 距离对应的角度（度）。
        compute() 内部用
    """
    if dy_cm <= 1e-6:
        return 0.0
    return degrees(atan2(dx_cm, dy_cm))


def clamp_value(value: float, limit: float) -> float:
    """ 限幅到 ±limit
        把 value 限制在 [-limit, +limit] 范围内。
        compute() 和 compute_line_target() 内部用
    """
    return max(-limit, min(limit, float(value)))


class PreviewExtractor:
    """从 TrackResult 提取前瞻观测值。

    只产出观测量：near_offset, near_yaw
    不决定最终下发目标，由 compute_line_target 和 run_track.py 完成。
    """

    def __init__(self, config: PreviewConfig | None = None, *, image_width: int | None = None) -> None:
        self.config = config or PreviewConfig()
        self.image_width = image_width   ## 必须传 ，等于 IPM 输出宽度（160）。否则 compute() 报错。

    def compute(self, track: TrackResult, *, speed_ref: float | None = None, lateral_bias_cm: float = 0.0) -> PreviewState:
        """【模块核心】从扫线结果算出 near_offset 和 near_yaw

        Parameters
        ----------
        track : TrackResult,即来自 LaneTracker 的扫线结果【不传报错】
        speed_ref : float | None , 传一个速度值 = 动态调整行位置
            None 表示使用默认行。【传 None = 不启用动态前瞻，用 near_y_ratio 默认行】
        lateral_bias_cm : float , 相机标定偏置
            相机标定偏置，默认 0.0【当前不用 ，后续标定时用】
        """
        h = int(track.center.shape[0])
        w = self._image_width()
        center_x = (w - 1) * 0.5
        cm_per_px_x = self.config.track_width_cm / max(w, 1)
        cm_per_px_y = self.config.track_depth_cm / max(h, 1)

        # near_y = 近端控制行，base_y = 偏航角基准行
        near_y = self._dynamic_row(
            default_y=int((h - 1) * self.config.near_y_ratio),
            speed_ref=speed_ref,
            factor=self.config.speed_to_near_y_px,
            h=h,
            offline=track.offline,
        )
        base_y = self._valid_clamped_row(int((h - 1) * self.config.yaw_base_y_ratio), h, track.offline)

        smoothed = self._global_smooth_centers(track) if self.config.smoothing_mode == "global" else None
        near_center = self._sample_center(track, near_y, smoothed)
        base_center = self._sample_center(track, base_y, smoothed)

        if near_center is None:
            return PreviewState(0.0, 0.0, near_y, False, "near row invalid")

        # 基准行无效时回退到近端行，保持输出稳定
        if base_center is None:
            base_center = near_center

        near_center_cm = (near_center - center_x) * cm_per_px_x + lateral_bias_cm
        base_center_cm = (base_center - center_x) * cm_per_px_x + lateral_bias_cm

        near_yaw = angle_deg(near_center_cm - base_center_cm, abs(base_y - near_y) * cm_per_px_y)

        # ── 非线性偏置放大（大偏置增强纠正力度）──
        # 公式：x_out = x × (1 + α × |x|)，在取负和 clamp 之前做
        # 作用：小偏置几乎不变（防直道过敏），大偏置明显放大（单边/急弯能纠正回来）
        # α=0 时此块跳过（等同线性）；α>0 时生效，clamp 仍兜底防过大
        if self.config.offset_nonlinear_alpha > 0:
            a = self.config.offset_nonlinear_alpha
            mag = abs(near_center_cm)
            near_center_cm = copysign(mag * (1.0 + a * mag), near_center_cm)

        return PreviewState(
            # 取负：让 near_offset 正=偏左（需右转），与下位机舵机「正左负右」一致
            near_offset=clamp_value(-near_center_cm, self.config.clamp_near_offset_cm),
            # 不取负：使用原始计算值（实测匹配下位机方向）
            near_yaw=clamp_value(near_yaw, self.config.clamp_near_yaw_deg),
            near_y=near_y,
            valid=True,
            valid_reason="ok",
        )

    def _image_width(self) -> int:
        if self.image_width is None:
            raise ValueError("PreviewExtractor needs image_width; do not myinference it from tracked borders.")
        return self.image_width

    def _dynamic_row(self, default_y: int, speed_ref: float | None, factor: float, h: int, offline: int) -> int:
        """
            根据速度算实际行号【最后用 _valid_clamped_row 夹到有效行】
        """
        y = default_y
        if self.config.dynamic_lookahead and speed_ref is not None:
            y = int(round(default_y - (speed_ref - self.config.default_speed_ref) * factor))
        return self._valid_clamped_row(y, h, offline)

    def _valid_clamped_row(self, y: int, h: int, offline: int) -> int:
        """确保行号不落在 invalid 线区"""
        lo = min(h - 1, max(0, offline + self.config.offline_margin))
        return max(lo, min(h - 1, y))

    def _sample_center(self, track: TrackResult, y: int, smoothed: np.ndarray | None) -> float | None:
        """取第 y 行的中线 x 坐标，带平滑"""
        if smoothed is not None and self._row_valid(track, y):
            return float(smoothed[y])
        if self.config.smoothing_mode == "none":
            return float(track.center[y]) if self._row_valid(track, y) else None

        h = int(track.center.shape[0])
        # 统一逻辑：从目标行往下取 local_window 行（往下 = 离车更近 = 更可靠）
        if y <= track.offline:
            start = min(h - 1, track.offline + self.config.offline_margin)
        else:
            start = max(track.offline + self.config.offline_margin, y)
        stop = min(h, start + self.config.local_window)

        values = [float(track.center[row]) for row in range(start, stop) if self._row_valid(track, row)]
        if not values:
            return None
        return float(sum(values) / len(values))

    def _global_smooth_centers(self, track: TrackResult) -> np.ndarray:
        """对整列中线做全局滑动平均。 你不需要直接调 ， smoothing_mode="global" 时 compute() 自动调。"""
        centers = track.center.astype(np.float32).copy()
        radius = max(0, self.config.global_smooth_window // 2)
        if radius == 0:
            return centers

        out = centers.copy()
        h = int(centers.shape[0])
        for y in range(max(0, track.offline), h):
            start = max(track.offline + self.config.offline_margin, y - radius)
            stop = min(h, y + radius + 1)
            values = [float(centers[row]) for row in range(start, stop) if self._row_valid(track, row)]
            if values:
                out[y] = sum(values) / len(values)
        return out

    def _row_valid(self, track: TrackResult, y: int) -> bool:
        """判断第 y 行中线是否可用。
           1.y < 0 或 y >= h → 无效
           2. y < offline（在有效行起点之下） → 无效
           3. 左右线都未检测到 → 无效
           说明：只有一边丢失 → 有效 （单边也用）
        """
        if y < 0 or y >= track.center.shape[0]:
            return False
        if y < track.offline:
            return False
        if track.is_left_find[y] == "F" and track.is_right_find[y] == "F":
            return False
        return True


def compute_line_target(
    preview: PreviewState,
    command: GuidanceCommand | PreviewConfig | None = None,
    config: PreviewConfig | None = None,
) -> LineTarget:
    """ 把前瞻观测转换为下位机下发目标。

        纯几何映射，【避障偏移由调用方写入 LineTarget.det_offset_cm。】
    """
    if isinstance(command, PreviewConfig):
        config = command
        command = None
    cfg = config or PreviewConfig()
    # 限幅统一使用 clamp_near_*（观测值限幅和下发值限幅一致）
    near_offset = clamp_value(preview.near_offset, cfg.clamp_near_offset_cm)
    yaw_bias = command.control_yaw_bias_deg if command is not None else 0.0
    near_yaw = clamp_value(preview.near_yaw + yaw_bias, cfg.clamp_near_yaw_deg)
    return LineTarget(near_offset=near_offset, near_yaw=near_yaw, det_offset_cm=0.0, source="preview")


def build_line_frame(target: LineTarget) -> str:
    """构建 L 帧字符串。"""
    return f"L,{target.near_offset:.3f},{target.near_yaw:.3f},{target.det_offset_cm:.3f}\r\n"


def control_values(command: GuidanceCommand) -> tuple[int, int] | None:
    """从 GuidanceCommand 算 C,run_is,speed_limit 帧值。

    返回 None 表示不需要发 C 帧。
    返回 (run_is, speed_limit) 表示需要发 C 帧。

    优先级：finish > stop > run_enable。
    """
    # 四个字段都没设 → 不发 C 帧
    if (
        command.run_enable is None
        and command.speed_limit is None
        and not command.stop_request
        and not command.finish_request
    ):
        return None

    # 完赛：C,0,0（下位机停电机）
    if command.finish_request:
        return 0, 0
    # 可恢复软暂停：C,1,0（保持本地发车锁存，仅把目标速度降为 0）
    if command.stop_request:
        return 1, 0

    # 正常运行
    run_is = int(bool(command.run_enable)) if command.run_enable is not None else 1
    # run=0：停车但不是完赛
    if run_is == 0:
        return run_is, 0

    # 速度兜底：speed_limit 不能低于 min_moving_speed_limit
    speed_limit = int(command.speed_limit or 0)
    if command.min_moving_speed_limit > 0:
        speed_limit = max(speed_limit, int(command.min_moving_speed_limit))
    return run_is, speed_limit


def build_control_frame(command: GuidanceCommand) -> str | None:
    """构建 C 帧字符串。
        调 control_values ，是 None 就返回 None，否则拼 "C,run_is,speed_limit\r\n"
        un_track.py 里直接用 control_values 取值再 uart.send_C(*values) 【更直接】"""
    values = control_values(command)
    if values is None:
        return None
    run_is, speed_limit = values
    return f"C,{run_is},{speed_limit}\r\n"


def send_uart_frames(uart, target: LineTarget, command: GuidanceCommand) -> tuple[str | None, str | None]:
    """发送 L 帧和可选 C 帧，返回实际发送成功的文本帧。"""
    line_msg = uart.send_L(target.near_offset, target.near_yaw, target.det_offset_cm)
    control_msg = None
    values = control_values(command)
    if values is not None:
        control_msg = uart.send_C(*values)
    return line_msg, control_msg



"""
# 初始化
preview_cfg = PreviewConfig(track_width_cm=60, track_depth_cm=90)
preview_extractor = PreviewExtractor(preview_cfg, image_width=160)
command = GuidanceCommand(speed_ref=75, speed_limit=100, run_enable=True, min_moving_speed_limit=0)

# 每帧
preview = preview_extractor.compute(track, speed_ref=command.speed_ref)
line_target = compute_line_target(preview, preview_cfg)
line_target.det_offset_cm = det_offset  # run_track.py 里填避障结果

# 下发
uart.send_L(line_target.near_offset, line_target.near_yaw, line_target.det_offset_cm)
values = control_values(command)
if values is not None:
    uart.send_C(*values)

"""