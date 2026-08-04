#!/usr/bin/env python3
# Copyright (c) 2026 清影/123twtd
# 仅用于分割/检测异步效果、fid 对齐和新鲜度验证；不发送 UART 控制帧。

"""共享内存感知链统一 smoke。

这是工程唯一推荐的实时 smoke 入口。通过 --mode 选择要验证的链路：

    seg     原图 -> 语义分割
    track   原图 -> 分割 -> IPM -> TRK
    det     原图 -> 异步目标检测
    fusion  同步分割(core 0) + 异步检测(core 1) + 同帧融合
    full    展示上述全部阶段

fusion/full 不会把旧检测框直接画到当前帧上。脚本会按 fid 保存短期分割历史，
检测结果返回后只与同 fid 的原图和分割结果配对，同时显示当前帧与检测帧的 lag、
结果 age 和是否仍满足实时融合门限。
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xsmartcar.ar_receiver import SharedMemoryFrameSource


SEG_MODES = {"seg", "track", "fusion", "full"}
DET_MODES = {"det", "fusion", "full"}
TRACK_MODES = {"track", "full"}
WINDOW_NAME = "trkdetection pipeline smoke"


@dataclass(slots=True)
class FramePacket:
    """统一 smoke 的本地帧载体，避免依赖已失效的旧 race_pipeline。"""

    fid: int
    captured_at: float
    frame_bgr: np.ndarray

    def age_ms(self, now: float | None = None) -> float:
        current = now if now is not None else time.time()
        return (current - self.captured_at) * 1000.0


@dataclass(slots=True)
class SegRecord:
    """某个 fid 对应的分割与几何阶段结果。"""

    packet: FramePacket
    mask: np.ndarray
    blend: np.ndarray
    ipm_mask: np.ndarray | None = None
    track_vis: np.ndarray | None = None


@dataclass(slots=True)
class DetRecord:
    """异步检测线程最近一次完成的结果。"""

    packet: FramePacket
    detections: list
    ok: bool
    vis: np.ndarray
    fusion_vis: np.ndarray | None
    matched: bool
    fresh: bool
    lag: int
    age_ms: float


@dataclass(slots=True)
class Counters:
    read: int = 0
    none_read: int = 0
    seg_ok: int = 0
    seg_fail: int = 0
    det_submit: int = 0
    det_result: int = 0
    det_ok: int = 0
    det_fail: int = 0
    fusion_ok: int = 0
    fusion_stale: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="trkdetection unified smoke: shm -> selected perception stages"
    )
    parser.add_argument(
        "--mode",
        choices=("seg", "track", "det", "fusion", "full"),
        default="fusion",
        help="要验证和展示的链路，默认 fusion",
    )
    parser.add_argument("--seg-det_model-dir", default="det_model", help="相对 xsmartcar/ 的分割模型目录")
    parser.add_argument("--det-det_model", default=None, help="检测 RKNN 绝对路径；默认使用标准模型名")
    parser.add_argument("--npz", default="xsmartcar/data_npz/race_data.npz", help="IPM 标定文件")
    parser.add_argument("--ipm-w", type=int, default=160)
    parser.add_argument("--ipm-h", type=int, default=120)
    parser.add_argument("--det-interval", type=int, default=3, help="每隔多少个源 fid 提交一次检测")
    parser.add_argument("--max-det-lag", type=int, default=10, help="允许当前帧领先检测帧的最大 fid")
    parser.add_argument("--max-det-age-ms", type=float, default=400.0, help="检测结果最大可融合年龄")
    parser.add_argument("--history-size", type=int, default=32, help="按 fid 保存的分割历史长度")
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--flip-vertical", action="store_true", default=True)
    parser.add_argument("--no-flip-vertical", action="store_false", dest="flip_vertical")
    parser.add_argument("--show", action="store_true", dest="show")
    parser.add_argument("--no-show", action="store_false", dest="show")
    parser.set_defaults(show=True)
    parser.add_argument("--window-scale", type=float, default=0.65)
    parser.add_argument("--save-dir", default="dev_tools/output/smoke")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 表示运行到 ESC/Ctrl+C")
    parser.add_argument("--heartbeat-s", type=float, default=1.0)
    parser.add_argument("--idle-sleep", type=float, default=0.005)
    return parser.parse_args()


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def fit_tile(image: np.ndarray, width: int, height: int, *, nearest: bool = False) -> np.ndarray:
    if image.ndim == 2:
        image = mask_to_bgr(image)
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return image.copy()


def label_tile(image: np.ndarray, label: str) -> np.ndarray:
    vis = image.copy()
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 250), 34), (0, 0, 0), -1)
    cv2.putText(
        vis,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def placeholder(width: int, height: int, label: str, detail: str = "waiting") -> np.ndarray:
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(vis, label, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, detail, (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    return vis


def draw_track(ipm_mask: np.ndarray, track) -> np.ndarray:
    """在 IPM 图上画左右边线、中线和控制点。"""

    vis = mask_to_bgr(ipm_mask)
    h, w = vis.shape[:2]
    valid = (track.is_left_find != "F") & (track.is_right_find != "F")

    def draw_line(values: np.ndarray, color: tuple[int, int, int]) -> None:
        points = [
            (int(values[y]), int(y))
            for y in range(min(h, len(values)))
            if bool(valid[y]) and 0 <= int(values[y]) < w
        ]
        if len(points) >= 2:
            cv2.polylines(vis, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)

    draw_line(track.leftborder, (80, 220, 80))
    draw_line(track.rightborder, (80, 80, 255))
    draw_line(track.center, (0, 255, 255))
    if 0 <= int(track.steer_row) < h:
        cv2.circle(vis, (int(track.steer_x), int(track.steer_row)), 3, (255, 0, 255), -1)
    return vis


def grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def status_band(panel: np.ndarray, lines: list[str], healthy: bool) -> np.ndarray:
    band_h = 92
    band = np.zeros((band_h, panel.shape[1], 3), dtype=np.uint8)
    primary = (80, 220, 80) if healthy else (0, 80, 255)
    split = (len(lines) + 1) // 2
    col_w = max(1, panel.shape[1] // 2)
    for index, text in enumerate(lines):
        col = 0 if index < split else 1
        row = index if col == 0 else index - split
        cv2.putText(
            band,
            text,
            (12 + col * col_w, 26 + row * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            primary if index == 0 else (230, 230, 230),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    return np.vstack([band, panel])


def build_panel(
    mode: str,
    current_packet: FramePacket,
    seg_record: SegRecord | None,
    det_record: DetRecord | None,
    counters: Counters,
    fps: float,
    seg_ms: float,
    source_error: str | None,
) -> np.ndarray:
    frame = current_packet.frame_bgr
    height, width = frame.shape[:2]
    raw = label_tile(fit_tile(frame, width, height), f"RAW fid={current_packet.fid}")

    seg_blend = placeholder(width, height, "SEG", "not enabled")
    seg_mask = placeholder(width, height, "SEG MASK", "not enabled")
    ipm = placeholder(width, height, "IPM", "not enabled")
    track = placeholder(width, height, "TRK", "not enabled")
    if seg_record is not None:
        seg_blend = label_tile(fit_tile(seg_record.blend, width, height), f"SEG fid={seg_record.packet.fid}")
        seg_mask = label_tile(fit_tile(seg_record.mask, width, height, nearest=True), "SEG MASK")
        if seg_record.ipm_mask is not None:
            ipm = label_tile(fit_tile(seg_record.ipm_mask, width, height, nearest=True), "IPM")
        if seg_record.track_vis is not None:
            track = label_tile(fit_tile(seg_record.track_vis, width, height, nearest=True), "TRK")

    det = placeholder(width, height, "DET", "waiting for async result")
    fusion = placeholder(width, height, "FUSION", "waiting for matching fid")
    det_age = -1.0
    det_lag = -1
    det_state = "WAIT"
    if det_record is not None:
        det_age = det_record.age_ms
        det_lag = det_record.lag
        det_state = "FRESH" if det_record.fresh else "STALE"
        det = label_tile(
            fit_tile(det_record.vis, width, height),
            f"DET fid={det_record.packet.fid} {det_state}",
        )
        if det_record.fusion_vis is not None:
            fusion = label_tile(
                fit_tile(det_record.fusion_vis, width, height),
                f"FUSION fid={det_record.packet.fid} {det_state}",
            )

    if mode == "seg":
        body = grid([raw, seg_blend, seg_mask], columns=3)
    elif mode == "track":
        body = grid([raw, seg_mask, ipm, track], columns=2)
    elif mode == "det":
        body = grid([raw, det], columns=2)
    elif mode == "fusion":
        body = grid([raw, seg_blend, det, fusion], columns=2)
    else:
        body = grid([raw, seg_blend, ipm, track, det, fusion], columns=3)

    det_required = mode in DET_MODES
    det_healthy = not det_required or (
        det_record is not None and det_record.ok and det_record.fresh
    )
    seg_required = mode in SEG_MODES
    seg_healthy = not seg_required or seg_record is not None
    healthy = seg_healthy and det_healthy and not source_error
    lines = [
        f"mode={mode} status={'OK' if healthy else 'CHECK'} fps={fps:.1f} seg_ms={seg_ms:.1f}",
        f"read={counters.read} none={counters.none_read} seg_ok={counters.seg_ok} seg_fail={counters.seg_fail}",
        f"det_submit={counters.det_submit} det_result={counters.det_result} det_ok={counters.det_ok}",
        f"det_state={det_state} lag={det_lag} age_ms={det_age:.0f}",
        f"fusion_ok={counters.fusion_ok} fusion_stale={counters.fusion_stale}",
        f"source_error={source_error or 'none'}  keys: ESC quit / S save",
    ]
    return status_band(body, lines, healthy)


def save_panel(panel: np.ndarray, save_dir: Path, mode: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = save_dir / f"{mode}_{stamp}.png"
    if not cv2.imwrite(str(path), panel):
        raise RuntimeError(f"Failed to save panel: {path}")
    return path


def main() -> int:
    args = parse_args()
    if args.det_interval < 1:
        raise ValueError("--det-interval must be >= 1")
    if args.history_size < 2:
        raise ValueError("--history-size must be >= 2")

    need_seg = args.mode in SEG_MODES
    need_det = args.mode in DET_MODES
    need_track = args.mode in TRACK_MODES

    source = SharedMemoryFrameSource(flip_vertical=args.flip_vertical)
    segmenter = None
    detector = None
    ipm = None
    tracker = None

    if need_seg:
        from xsmartcar.myinference.seg_infer.myppseg_sync_infer import PPSegSyncInfer

        # 同步分割固定 core 0，保证比赛控制链的帧身份最直接。
        segmenter = PPSegSyncInfer(model_dir=args.seg_model_dir, blend_alpha=args.blend)

    if need_det:
        from xsmartcar.myinference.det_infer import RKNNDetectInfer

        # 异步检测固定 core 1，只保留最新待处理帧，不阻塞分割主链。
        detector = RKNNDetectInfer(model_path=args.det_model)

    if need_track:
        from xsmartcar.ipm_utils import FastIPM
        from xsmartcar.trk_tracker_module import LaneTracker, TrackerConfig

        ipm = FastIPM(args.npz, out_w=args.ipm_w, out_h=args.ipm_h)
        tracker = LaneTracker(TrackerConfig(height=args.ipm_h, width=args.ipm_w))

    print(
        f"smoke ready: mode={args.mode} show={args.show} "
        f"det_interval={args.det_interval} max_lag={args.max_det_lag} "
        f"max_age_ms={args.max_det_age_ms}"
    )

    counters = Counters()
    history: OrderedDict[int, SegRecord] = OrderedDict()
    latest_det: DetRecord | None = None
    last_submit_fid = -1
    last_result_fid = -1
    last_panel: np.ndarray | None = None
    seg_ms = 0.0
    fps = 0.0
    fps_frames = 0
    fps_started = time.time()
    started = time.time()
    heartbeat_at = started
    last_frame_at = started

    try:
        while args.duration_s <= 0 or time.time() - started < args.duration_s:
            frame = source.read()
            now = time.time()
            if frame is None:
                counters.none_read += 1
                if now - heartbeat_at >= args.heartbeat_s:
                    print(
                        f"heartbeat mode={args.mode} waiting_for_frame "
                        f"fid={source.last_fid} none={counters.none_read} "
                        f"stale_ms={(now - last_frame_at) * 1000.0:.0f} "
                        f"error={source.last_error}"
                    )
                    heartbeat_at = now
                if args.show and last_panel is not None:
                    waiting = last_panel.copy()
                    cv2.rectangle(waiting, (0, 0), (waiting.shape[1], 42), (0, 0, 150), -1)
                    cv2.putText(
                        waiting,
                        f"NO NEW SHM FRAME  stale_ms={(now - last_frame_at) * 1000.0:.0f}",
                        (12, 29),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(WINDOW_NAME, waiting)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                time.sleep(args.idle_sleep)
                continue

            counters.read += 1
            last_frame_at = now
            fps_frames += 1
            packet = FramePacket(fid=int(source.last_fid), captured_at=now, frame_bgr=frame)

            seg_record = None
            if segmenter is not None:
                seg_t0 = time.time()
                seg_packet, seg_map, seg_ok = segmenter.infer_with_packet(packet)
                seg_ms = (time.time() - seg_t0) * 1000.0
                if seg_ok and seg_map is not None and seg_packet is not None:
                    counters.seg_ok += 1
                    blend = segmenter.visualize(seg_map, seg_packet.frame_bgr)
                    ipm_mask = None
                    track_vis = None
                    if ipm is not None and tracker is not None:
                        ipm_mask = (ipm.process(seg_map, nearest=True) > 0).astype(np.uint8)
                        track_result = tracker.process(ipm_mask, copy_result=False)
                        track_vis = draw_track(ipm_mask, track_result)
                    seg_record = SegRecord(seg_packet, seg_map, blend, ipm_mask, track_vis)
                    history[seg_packet.fid] = seg_record
                    while len(history) > args.history_size:
                        history.popitem(last=False)
                else:
                    counters.seg_fail += 1

            if detector is not None and detector.should_submit(
                packet.fid, last_submit_fid, interval=args.det_interval
            ):
                detector.submit(packet)
                last_submit_fid = packet.fid
                counters.det_submit += 1

            if detector is not None:
                result = detector.get_latest()
                if result is not None and result[0].fid != last_result_fid:
                    det_packet, detections, det_ok = result
                    last_result_fid = det_packet.fid
                    counters.det_result += 1
                    counters.det_ok += int(det_ok)
                    counters.det_fail += int(not det_ok)

                    lag = packet.fid - det_packet.fid
                    age_ms = det_packet.age_ms(now)
                    matched_seg = history.get(det_packet.fid)
                    matched = matched_seg is not None
                    fresh = (
                        det_ok
                        and 0 <= lag <= args.max_det_lag
                        and age_ms <= args.max_det_age_ms
                        and (args.mode == "det" or matched)
                    )
                    det_vis = detector.visualize(det_packet.frame_bgr, detections) if det_ok else det_packet.frame_bgr.copy()
                    fusion_vis = None
                    if matched_seg is not None and det_ok:
                        fusion_vis = detector.visualize(matched_seg.blend, detections)
                    if args.mode in {"fusion", "full"}:
                        counters.fusion_ok += int(fresh)
                        counters.fusion_stale += int(det_ok and not fresh)
                    latest_det = DetRecord(
                        packet=det_packet,
                        detections=detections,
                        ok=det_ok,
                        vis=det_vis,
                        fusion_vis=fusion_vis,
                        matched=matched,
                        fresh=fresh,
                        lag=lag,
                        age_ms=age_ms,
                    )

            # latest result 会被多帧复用；展示前按当前 fid/时间重新计算新鲜度。
            if latest_det is not None:
                latest_det.lag = packet.fid - latest_det.packet.fid
                latest_det.age_ms = latest_det.packet.age_ms(now)
                latest_det.fresh = (
                    latest_det.ok
                    and 0 <= latest_det.lag <= args.max_det_lag
                    and latest_det.age_ms <= args.max_det_age_ms
                    and (args.mode == "det" or latest_det.matched)
                )

            if now - fps_started >= 1.0:
                fps = fps_frames / (now - fps_started)
                fps_frames = 0
                fps_started = now

            last_panel = build_panel(
                args.mode,
                packet,
                seg_record,
                latest_det,
                counters,
                fps,
                seg_ms,
                source.last_error,
            )

            if now - heartbeat_at >= args.heartbeat_s:
                det_lag = latest_det.lag if latest_det is not None else -1
                det_age = latest_det.age_ms if latest_det is not None else -1.0
                print(
                    f"heartbeat mode={args.mode} fid={packet.fid} fps={fps:.1f} "
                    f"seg_ok={counters.seg_ok} det_result={counters.det_result} "
                    f"fusion_ok={counters.fusion_ok} det_lag={det_lag} "
                    f"det_age_ms={det_age:.0f} error={source.last_error}"
                )
                heartbeat_at = now

            if args.show:
                shown = last_panel
                if abs(args.window_scale - 1.0) > 1e-6:
                    shown = cv2.resize(
                        shown,
                        None,
                        fx=args.window_scale,
                        fy=args.window_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow(WINDOW_NAME, shown)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key in (ord("s"), ord("S")):
                    print(f"saved: {save_panel(last_panel, Path(args.save_dir), args.mode)}")
    except KeyboardInterrupt:
        print("\nsmoke interrupted")
    finally:
        source.close()
        if segmenter is not None:
            segmenter.release()
        if detector is not None:
            detector.release()
        cv2.destroyAllWindows()

    elapsed = max(time.time() - started, 1e-6)
    print(
        "summary "
        f"mode={args.mode} elapsed_s={elapsed:.1f} read={counters.read} "
        f"seg_ok={counters.seg_ok} seg_fail={counters.seg_fail} "
        f"det_submit={counters.det_submit} det_result={counters.det_result} "
        f"det_ok={counters.det_ok} det_fail={counters.det_fail} "
        f"fusion_ok={counters.fusion_ok} fusion_stale={counters.fusion_stale}"
    )

    passed = counters.read > 0
    if need_seg:
        passed = passed and counters.seg_ok > 0 and counters.seg_fail == 0
    if need_det:
        passed = passed and counters.det_ok > 0 and counters.det_fail == 0
    if args.mode in {"fusion", "full"}:
        passed = passed and counters.fusion_ok > 0
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
