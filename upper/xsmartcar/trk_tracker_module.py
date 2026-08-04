# Copyright (c) 2026 清影/123twtd
"""
比赛版 TRK 扫线模块。

这个文件只负责赛道线提取：
输入一帧已经过前置处理的图像，输出左右边线、中线、宽度、offline 和 fork 信息。

典型输入通常是：
1. seg 输出的二值图
2. 经过 IPM 变换后的图
3. 尺寸已经固定好的比赛输入，例如 160x120

这个模块不负责：
1. 读共享区图像
2. seg 推理
3. IPM 标定交互
4. 画图、弹窗、保存调试图
5. near_offset / near_yaw / far_yaw 计算
6. UART 打包与发送

因此它在比赛主流程中的位置是：

    ipm_image -> LaneTracker.process(ipm_image) -> TrackResult

如果后面还要继续做前瞻、偏置、UART，对接层应放在 trk_preview.py 中，
而不是继续塞回这个文件。

当前内部流程固定分成三段：
1. Stage 1：基础行初始化
2. Stage 2：非基础行向上生长
3. Stage 3：元素处理，目前只处理 fork

对外推荐入口：
1. LaneTracker(config).process(image)
   适合比赛实时循环
2. process_track(image, config)
   适合测试脚本或一次性调用
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

Status = Literal["F", "T", "W", "H"]
ForkPrefer = Literal["L", "R", "none"]
InnerFindMode = Literal["first_black", "jump"]


@dataclass(slots=True)
class TrackerConfig:
    """
    扫线稳定参数集合。

    这里放的是同一套比赛逻辑下相对稳定的参数，
    例如图像尺寸、窗口大小、offline 边界和 fork 阈值。
    """

    height: int
    width: int
    base_rows: int = 5     ## 320x240 下默认 5 行; 160x120 下也是 5 行
    grow_window: int = 2   ## 320x240 下默认 5 px；160x120 下默认 2px
    left_offline: int = 3
    right_offline: int | None = None
    offwidth: int | None = None
    fork_black_min: int = 2  ## 320x240 下默认 3 px；160x120 下默认 2px
    fork_min_width_factor: int = 3  # 4 or 5, 5与主版一致，但是会导致部分分支行未被识别到
    fork_prefer: ForkPrefer = "L"
    # 这里默认使用 jump,可显式设为 "first_black"。
    inner_find_mode: InnerFindMode = "jump" ## 内棱查找模式【默认不修改，防止内棱查找错位】
    debug_fork: bool = False  ## 默认不开启，调试时开启查看数据流

    def __post_init__(self) -> None:
        if self.right_offline is None:
            self.right_offline = (self.width - 1) - self.left_offline
        if self.offwidth is None:
            self.offwidth = self.width // 10

    @property
    def img_mid(self) -> int:
        return (self.width - 1) // 2

    @property
    def fork_min_width(self) -> int:
        return self.offwidth * self.fork_min_width_factor


@dataclass(slots=True)
class TrackerState:
    """
    单帧运行工作区。

    这里保存本帧扫线过程中的中间状态。
    外部一般不直接修改它，而是复用同一个 LaneTracker，
    让 state 在每帧开始时 reset。
    """

    leftborder: np.ndarray
    rightborder: np.ndarray
    center: np.ndarray
    wide: np.ndarray
    is_left_find: np.ndarray
    is_right_find: np.ndarray
    fork_score: np.ndarray

    offline: int = 0
    fork_y_top: int = -1
    fork_y_bot: int = -1
    fork_hit_count: int = 0
    fork_active: bool = False
    fork_choice: ForkPrefer = "none"
    fork_inner0: int = -1
    left_lost_count: int = 0
    right_lost_count: int = 0
    white_lost_count: int = 0

    @classmethod
    def create(cls, config: TrackerConfig) -> "TrackerState":
        h = config.height
        return cls(
            leftborder=np.zeros(h, dtype=np.int16),
            rightborder=np.full(h, config.width - 1, dtype=np.int16),
            center=np.full(h, config.img_mid, dtype=np.int16),
            wide=np.full(h, config.width - 1, dtype=np.int16),
            is_left_find=np.full(h, "F", dtype="U1"),
            is_right_find=np.full(h, "F", dtype="U1"),
            fork_score=np.zeros(h, dtype=np.uint16),
        )

    def reset(self, config: TrackerConfig) -> None:
        self.leftborder.fill(0)
        self.rightborder.fill(config.width - 1)
        self.center.fill(config.img_mid)
        self.wide.fill(config.width - 1)
        self.is_left_find.fill("F")
        self.is_right_find.fill("F")
        self.fork_score.fill(0)
        self.offline = 0
        self.fork_y_top = -1
        self.fork_y_bot = -1
        self.fork_hit_count = 0
        self.fork_active = False
        self.fork_choice = "none"
        self.fork_inner0 = -1
        self.left_lost_count = 0
        self.right_lost_count = 0
        self.white_lost_count = 0


@dataclass(slots=True)
class ForkInfo:
    """fork 摘要信息，供后续前瞻层、行为层或调试代码读取。"""

    active: bool
    choice: ForkPrefer
    y_top: int
    y_bot: int
    hit_count: int
    inner0: int


@dataclass(slots=True)
class TrackResult:
    """
    TRK 对外输出结果。

    这就是扫线模块与外部逻辑的边界。
    后续 preview、yaw、UART 都应基于这个结果继续处理。
    """

    leftborder: np.ndarray
    rightborder: np.ndarray
    center: np.ndarray
    wide: np.ndarray
    is_left_find: np.ndarray
    is_right_find: np.ndarray
    fork_score: np.ndarray
    offline: int
    steer_row: int
    steer_x: int
    left_lost_count: int
    right_lost_count: int
    white_lost_count: int
    fork: ForkInfo


@dataclass(slots=True)
class RowEdgeHit:
    """单行扫边结果：边界点 + F/T/W/H 状态位。"""

    point: int
    status: Status


def get_binary_mask(image: np.ndarray) -> np.ndarray:
    """把 BGR 或灰度输入统一转换成 0/1 mask。"""

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (image > 0).astype(np.uint8)


class LaneTracker:
    """
    比赛主流程中应复用的扫线器。

    这个类绑定两部分内容：
    1. config：稳定参数
    2. state：可复用的单帧工作区

    它只负责赛道线提取，不负责更上层的前瞻、协议和行为策略。
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.state = TrackerState.create(config)

    def process(self, image: np.ndarray, *, copy_result: bool = False) -> TrackResult:
        """
        扫描一帧图像并返回结构化结果。

        内部固定顺序为：
        1. 预处理为二值 mask
        2. reset 本帧状态
        3. 做基础行初始化
        4. 做非基础行向上生长
        5. 做元素处理
        6. 打包 TrackResult
        """

        mask = self._prepare_mask(image)
        self.state.reset(self.config)
        self.init_base_rows(mask)
        self.grow_non_base_rows(mask)
        self.handle_elements(mask)
        return self.build_result(copy_result=copy_result)

    def _prepare_mask(self, image: np.ndarray) -> np.ndarray:
        mask = get_binary_mask(image)
        expected_shape = (self.config.height, self.config.width)
        if mask.shape != expected_shape:
            raise ValueError(f"expected {expected_shape}, got {mask.shape}")
        return mask

    # ---------------------------------------------------------------------
    # Stage 1: 基础行初始化
    # 对应原始脚本底部几行的初始扫线。
    # 这一步的目标是给后续向上生长提供可靠起点。
    # ---------------------------------------------------------------------
    def init_base_rows(self, mask: np.ndarray) -> None:
        state = self.state
        cfg = self.config
        y = cfg.height - 1
        row = mask[y]

        if row[cfg.img_mid] == 1:
            state.rightborder[y] = self._scan_base_right(row, cfg.img_mid)
            state.leftborder[y] = self._scan_base_left(row, cfg.img_mid)
        else:
            xs = 0
            for x in range(cfg.img_mid):
                if row[cfg.img_mid - x] != 0:
                    xs = x
                    break
                if row[cfg.img_mid + x] != 0:
                    xs = x
                    break
            if row[cfg.img_mid + xs] != 0:
                state.leftborder[y] = cfg.img_mid + xs - 1
                state.rightborder[y] = self._scan_base_right(row, int(state.leftborder[y]))
            elif row[cfg.img_mid - xs] != 0:
                state.rightborder[y] = cfg.img_mid - xs + 1
                state.leftborder[y] = self._scan_base_left(row, int(state.rightborder[y]))

        self._finish_row(y, "T", "T")

        for y in range(cfg.height - 2, cfg.height - 1 - cfg.base_rows, -1):
            row = mask[y]
            prev_center = int(state.center[y + 1])
            state.rightborder[y] = self._scan_base_right(row, prev_center)
            state.leftborder[y] = self._scan_base_left(row, prev_center)
            self._finish_row(y, "T", "T")

    def _scan_base_right(self, row: np.ndarray, start: int) -> int:
        cfg = self.config
        for x in range(start, cfg.right_offline):
            if row[x] == 0 and row[x + 1] == 0 and row[x + 2] == 0:
                return x
        return cfg.width - 1

    def _scan_base_left(self, row: np.ndarray, start: int) -> int:
        cfg = self.config
        for x in range(start, cfg.left_offline, -1):
            if row[x] == 0 and row[x - 1] == 0 and row[x - 2] == 0:
                return x
        return 0

    # ---------------------------------------------------------------------
    # Stage 2: 非基础行常规生长
    # 这里是主循环：逐行扫边、修 H、补 W、更新 center/wide、检测 fork。
    # ---------------------------------------------------------------------
    def grow_non_base_rows(self, mask: np.ndarray) -> None:
        state = self.state
        cfg = self.config
        left_found_t = False
        right_found_t = False
        get_left_line = False
        get_right_line = False
        d_left = 0.0
        d_right = 0.0
        ytemp_w_l = 0
        ytemp_w_r = 0

        for y in range(cfg.height - 1 - cfg.base_rows, state.offline - 1, -1):
            row = mask[y]
            left_hit, right_hit = self._scan_row_basic(row, y)

            state.leftborder[y] = left_hit.point
            state.is_left_find[y] = left_hit.status
            state.rightborder[y] = right_hit.point
            state.is_right_find[y] = right_hit.status

            self._fix_h_border(row, y, "L")
            if self._should_cutoff(y, stage=1):
                state.offline = y + 1
                break
            self._fix_h_border(row, y, "R")

            if state.is_left_find[y] == "W" and y >= state.offline and y < cfg.height - 20:
                if not get_left_line:
                    get_left_line = True
                    ytemp_w_l = y + 2
                    left_count = self._count_following_true_rows(state.is_left_find, y + 1, y + 15)
                    if left_count > 8:
                        d_left = (state.leftborder[y + 3] - state.leftborder[y + left_count]) / (left_count - 3)
                        left_found_t = True
                if left_found_t:
                    pred = state.leftborder[ytemp_w_l] + d_left * (ytemp_w_l - y)
                    state.leftborder[y] = self._clamp_x(int(round(pred)))

            if state.is_right_find[y] == "W" and y >= state.offline and y < cfg.height - 20:
                if not get_right_line:
                    get_right_line = True
                    ytemp_w_r = y + 2
                    right_count = self._count_following_true_rows(state.is_right_find, y + 1, y + 15)
                    if right_count > 8:
                        d_right = (state.rightborder[y + 3] - state.rightborder[y + right_count]) / (right_count - 3)
                        right_found_t = True
                if right_found_t:
                    pred = state.rightborder[ytemp_w_r] + d_right * (ytemp_w_r - y)
                    state.rightborder[y] = self._clamp_x(int(round(pred)))

            if state.is_left_find[y] == "W" and state.is_right_find[y] == "W":
                state.white_lost_count += 1
            if state.is_left_find[y] == "W" and y < cfg.height - cfg.base_rows:
                state.left_lost_count += 1
            if state.is_right_find[y] == "W" and y < cfg.height - cfg.base_rows:
                state.right_lost_count += 1

            self._clamp_row_borders(y)

            fork_score = self._detect_fork(row, y)
            state.fork_score[y] = fork_score
            if fork_score > 0:
                state.fork_hit_count += 1
                if state.fork_y_top < 0 or y < state.fork_y_top:
                    state.fork_y_top = y
                if state.fork_y_bot < 0 or y > state.fork_y_bot:
                    state.fork_y_bot = y

            state.wide[y] = state.rightborder[y] - state.leftborder[y]
            state.center[y] = (state.leftborder[y] + state.rightborder[y]) // 2

            if self._should_cutoff(y, stage=2):
                state.offline = y + 1
                break

    def _scan_row_basic(self, row: np.ndarray, y: int) -> tuple[RowEdgeHit, RowEdgeHit]:
        state = self.state
        cfg = self.config
        rr = min(cfg.right_offline, int(state.rightborder[y + 1]) + cfg.grow_window)
        rl = max(cfg.left_offline, int(state.rightborder[y + 1]) - cfg.grow_window)
        right_hit = self._scan_row_window_edge(row, "R", rl, rr)

        lr = min(cfg.right_offline, int(state.leftborder[y + 1]) + cfg.grow_window)
        ll = max(cfg.left_offline, int(state.leftborder[y + 1]) - cfg.grow_window)
        left_hit = self._scan_row_window_edge(row, "L", ll, lr)

        if right_hit.status == "W":
            right_hit.point = int(state.rightborder[y + 1])
        if left_hit.status == "W":
            left_hit.point = int(state.leftborder[y + 1])
        return left_hit, right_hit

    def _scan_row_window_edge(self, row: np.ndarray, side: Literal["L", "R"], col_lo: int, col_hi: int) -> RowEdgeHit:
        mid = (col_lo + col_hi) // 2
        if side == "L":
            for x in range(col_hi, col_lo, -1):
                if row[x] == 1 and row[x - 1] == 0:
                    return RowEdgeHit(x - 1, "T")
            if row[mid] != 0:
                return RowEdgeHit(mid, "W")
            return RowEdgeHit(col_hi, "H")

        for x in range(col_lo, col_hi):
            if row[x] == 1 and row[x + 1] == 0:
                return RowEdgeHit(x + 1, "T")
        if row[mid] != 0:
            return RowEdgeHit(mid, "W")
        return RowEdgeHit(col_lo, "H")

    def _fix_h_border(self, row: np.ndarray, y: int, side: Literal["L", "R"]) -> None:
        state = self.state
        if side == "L" and state.is_left_find[y] == "H":
            for x in range(int(state.leftborder[y]) + 1, int(state.rightborder[y]) - 1):
                if row[x] == 0 and row[x + 1] == 1:
                    state.leftborder[y] = x
                    state.is_left_find[y] = "T"
                    break
                if row[x] == 1:
                    state.is_left_find[y] = "T"
                    break

        if side == "R" and state.is_right_find[y] == "H":
            for x in range(int(state.rightborder[y]) - 1, int(state.leftborder[y]) + 1, -1):
                if row[x] == 0 and row[x - 1] == 1:
                    state.rightborder[y] = x
                    state.is_right_find[y] = "T"
                    break
                if row[x] == 1:
                    state.is_right_find[y] = "T"
                    break

    def _should_cutoff(self, y: int, stage: int) -> bool:
        state = self.state
        cfg = self.config
        if stage == 1:
            return (state.rightborder[y] - state.leftborder[y]) < cfg.offwidth
        if state.wide[y] < cfg.offwidth:
            return True
        if state.rightborder[y] > cfg.right_offline or state.leftborder[y] < cfg.left_offline:
            return True
        return False

    def _detect_fork(self, row: np.ndarray, y: int) -> int:
        state = self.state
        cfg = self.config
        lo = int(state.leftborder[y])
        hi = int(state.rightborder[y])
        if hi - lo < cfg.fork_min_width:
            return 0
        max_black = 0
        cur_black = 0
        for x in range(lo + 1, hi):
            if row[x] == 0:
                cur_black += 1
                if cur_black > max_black:
                    max_black = cur_black
            else:
                cur_black = 0
        return max_black if max_black >= cfg.fork_black_min else 0

    # ---------------------------------------------------------------------
    # Stage 3: 元素处理
    # 当前只处理 fork，并且故意与主扫线循环分开，便于后续继续扩展。
    # ---------------------------------------------------------------------
    def handle_elements(self, mask: np.ndarray) -> None:
        state = self.state
        cfg = self.config
        state.fork_active = state.fork_hit_count >= 3
        state.fork_choice = cfg.fork_prefer if state.fork_active else "none"

        if state.fork_choice not in ("L", "R"):
            return

        inner0 = self._find_inner_at_row(
            mask[state.fork_y_bot],
            int(state.leftborder[state.fork_y_bot]),
            int(state.rightborder[state.fork_y_bot]),
        )
        if inner0 < 0:
            return

        state.fork_inner0 = inner0
        if state.fork_choice == "L":
            state.rightborder[state.fork_y_bot] = inner0
        else:
            state.leftborder[state.fork_y_bot] = inner0

        self._grow_inner_edge_up(mask)
        self._extrap_inner_below()

        for y in range(state.fork_y_top, cfg.height):
            self._clamp_row_borders(y)
            if state.rightborder[y] > state.leftborder[y]:
                state.center[y] = (state.leftborder[y] + state.rightborder[y]) // 2
                state.wide[y] = state.rightborder[y] - state.leftborder[y]

        self._clear_non_branch_above()
        self._update_row_center_after_fork()

    def _find_inner_at_row(self, row: np.ndarray, lo: int, hi: int) -> int:
        """在 fork_y_bot 行按当前策略寻找所选分支的内拐。"""

        if hi <= lo + 2:
            return -1

        if self.config.inner_find_mode == "jump":
            if self.state.fork_choice == "L":
                for x in range(lo + 1, hi - 1):
                    if row[x] == 1 and row[x + 1] == 0:
                        return x + 1
                return -1
            inner = -1
            for x in range(lo + 1, hi - 1):
                if row[x] == 0 and row[x + 1] == 1:
                    inner = x
            return inner

        if self.state.fork_choice == "L":
            for x in range(lo + 1, hi):
                if row[x] == 0:
                    return x
            return -1

        for x in range(hi - 1, lo, -1):
            if row[x] == 0:
                return x
        return -1

    def _grow_inner_edge_up(self, mask: np.ndarray) -> None:
        """在 fork 区间内向上跟踪所选分支的内侧边。"""

        state = self.state
        cfg = self.config
        side = "R" if state.fork_choice == "L" else "L"
        edge = state.rightborder if state.fork_choice == "L" else state.leftborder

        for y in range(state.fork_y_bot - 1, state.fork_y_top - 1, -1):
            rl = max(cfg.left_offline, int(edge[y + 1]) - cfg.grow_window)
            rr = min(cfg.right_offline, int(edge[y + 1]) + cfg.grow_window)
            hit = self._scan_row_window_edge(mask[y], side, rl, rr)
            edge[y] = edge[y + 1] if hit.status == "W" else hit.point

    def _extrap_inner_below(self) -> None:
        """在 fork_y_bot 下方向近端外推所选分支的内侧边。"""

        state = self.state
        cfg = self.config
        inner_arr = state.rightborder if state.fork_choice == "L" else state.leftborder
        outer_arr = state.leftborder if state.fork_choice == "L" else state.rightborder
        outer_flag = state.is_left_find if state.fork_choice == "L" else state.is_right_find
        y0 = state.fork_y_bot

        if y0 >= cfg.height - 1:
            return

        # 这里采用斜率外推，而不是 trk_line_merged 主版的 polyfit 外侧拟合。
        # 这样算力更低，实车上也更可控，但它不是原主版的逐行等价搬运。
        dy = min(5, y0 - state.fork_y_top)
        k_inner = (inner_arr[y0] - inner_arr[y0 - dy]) / dy if dy > 0 else 0.0

        for y in range(y0 + 1, cfg.height):
            if state.fork_choice == "L":
                outer_healthy = outer_flag[y] == "T" and outer_arr[y] > cfg.left_offline + 3
            else:
                outer_healthy = outer_flag[y] == "T" and outer_arr[y] < cfg.right_offline - 3

            if outer_healthy:
                k_outer = outer_arr[y] - outer_arr[y - 1]
                inner_predict = inner_arr[y - 1] + k_outer
            else:
                inner_predict = inner_arr[y - 1] + k_inner
            inner_arr[y] = self._clamp_x(int(round(inner_predict)))

    def _clear_non_branch_above(self) -> None:
        """fork 补线后，清理未选中分支上方区域。"""

        state = self.state
        cfg = self.config
        branch_top = 0
        side_arr = state.rightborder if state.fork_choice == "R" else state.leftborder

        for y in range(state.fork_y_top - 1, 0, -1):
            if abs(int(side_arr[y]) - int(side_arr[y + 1])) > cfg.offwidth:
                branch_top = y + 1
                break

        # 这是从 trk_cal_line 实验思路保留下来的行为：
        # 清掉未选分支后同步更新 offline，后续前瞻层不会再采到被清理区域。
        state.offline = branch_top
        for y in range(branch_top):
            state.is_left_find[y] = "F"
            state.is_right_find[y] = "F"

    def _update_row_center_after_fork(self) -> None:
        state = self.state
        for y in range(self.config.height - 1, state.offline, -1):
            # 原版里这段 W 行 center 外推是实验逻辑且默认注释。
            # 模块版启用它，并补了 y+2 边界保护，避免最底部行越界。
            if state.is_left_find[y] == "W" and state.fork_choice == "L" and y + 2 < self.config.height:
                state.center[y] = state.center[y + 1] + (state.center[y + 1] - state.center[y + 2])
            elif state.is_right_find[y] == "W" and state.fork_choice == "R" and y + 2 < self.config.height:
                state.center[y] = state.center[y + 1] + (state.center[y + 1] - state.center[y + 2])

    # ---------------------------------------------------------------------
    # 结果构造 / 通用工具
    # ---------------------------------------------------------------------
    def build_result(self, *, copy_result: bool = False) -> TrackResult:
        """
        构造对外结果。

        copy_result=False 时直接返回内部数组视图，适合实时循环。
        copy_result=True 时复制数组，适合测试或需要长期保留本帧结果。
        """

        state = self.state
        steer_row = self.config.height - 1 - self.config.base_rows

        def maybe_copy(arr: np.ndarray) -> np.ndarray:
            return arr.copy() if copy_result else arr

        return TrackResult(
            leftborder=maybe_copy(state.leftborder),
            rightborder=maybe_copy(state.rightborder),
            center=maybe_copy(state.center),
            wide=maybe_copy(state.wide),
            is_left_find=maybe_copy(state.is_left_find),
            is_right_find=maybe_copy(state.is_right_find),
            fork_score=maybe_copy(state.fork_score),
            offline=state.offline,
            steer_row=steer_row,
            steer_x=int(state.center[steer_row]),
            left_lost_count=state.left_lost_count,
            right_lost_count=state.right_lost_count,
            white_lost_count=state.white_lost_count,
            fork=ForkInfo(
                active=state.fork_active,
                choice=state.fork_choice,
                y_top=state.fork_y_top,
                y_bot=state.fork_y_bot,
                hit_count=state.fork_hit_count,
                inner0=state.fork_inner0,
            ),
        )

    def _finish_row(self, y: int, left_status: Status, right_status: Status) -> None:
        state = self.state
        self._clamp_row_borders(y)
        state.center[y] = (state.leftborder[y] + state.rightborder[y]) // 2
        state.wide[y] = state.rightborder[y] - state.leftborder[y]
        state.is_left_find[y] = left_status
        state.is_right_find[y] = right_status

    def _clamp_row_borders(self, y: int) -> None:
        state = self.state
        state.leftborder[y] = self._clamp_x(int(state.leftborder[y]))
        state.rightborder[y] = self._clamp_x(int(state.rightborder[y]))

    def _clamp_x(self, x: int) -> int:
        return max(self.config.left_offline, min(self.config.right_offline, x))

    def _count_following_true_rows(self, flags: np.ndarray, start: int, stop: int) -> int:
        cfg = self.config
        count = 0
        for y in range(start, min(stop, cfg.height)):
            if flags[y] == "T":
                count += 1
        return count


def process_track(image: np.ndarray, config: TrackerConfig, *, copy_result: bool = False) -> TrackResult:
    """
    一次性调用入口，适合测试脚本。

    如果是比赛实时循环，更推荐先创建 LaneTracker，再逐帧调用 process()，
    这样可以复用内部 state 数组。
    """

    tracker = LaneTracker(config)
    return tracker.process(image, copy_result=copy_result)
