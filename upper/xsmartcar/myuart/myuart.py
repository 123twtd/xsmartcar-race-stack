# Copyright (c) 2026 清影/123twtd
"""UART helpers for the race pipeline.

Current protocol:
- Line frame:    L,near_offset,near_yaw,det_offset_cm\r\n
- Control frame: C,run_is,speed_limit\r\n
Notes:
- The lower computer keeps the PID loop.
- The upper computer only sends the line target and optional speed command.
- `send_text()` is retained as the most robust escape hatch for debugging.
"""
import serial
import time


class MyUART:
    """Simple serial sender used by the race runtime pipeline."""

    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, timeout=0.1, write_timeout=0.5, encoding='utf-8'):
        self.port = port            # 新增：重连时要用
        self.baudrate = baudrate    # 新增
        self.timeout = timeout      # 新增
        self.write_timeout = write_timeout  # 新增
        self._closed = False

        self.ser = serial.Serial(port, baudrate, timeout=timeout, write_timeout=write_timeout)
        self.encoding = encoding

        time.sleep(1)               # sleep 1s 等待串口初始化完成

    def send_text(self, text: str):
        """Send an already-formatted ASCII/UTF-8 frame."""
        if self._closed:
            return None
        try:
            self.ser.write(text.encode(self.encoding))
            return text
        except serial.SerialException:
            # return None --v1
            if self._reconnect():  ## --v4
                # 重连成功：重发一次
                try:
                    self.ser.write(text.encode(self.encoding))
                    return text
                except serial.SerialException:
                    return None  # 重连失败：返回 None 让外层处理
            return None  # 重连失败 = 返回 None 给外层计数处理

    def send_L(self, near_offset=0.0, near_yaw=0.0, det_offset_cm=0.0):
        """发送行目标帧：L,near_offset,near_yaw,det_offset_cm"""

        return self.send_text(f"L,{near_offset:.3f},{near_yaw:.3f},{det_offset_cm:.3f}\r\n")

    def send_C(self, run_is=0, speed_lim=0):
        """发送控制帧：C,run_is,speed_limit"""

        return self.send_text(f"C,{int(run_is)},{int(speed_lim)}\r\n")

    def stop(self):
        """发送 C,1,0 可恢复软暂停；不会清除下位机运行锁存。"""
        try:
            return self.send_C(1, 0)
        except Exception:
            return None

    def _reconnect(self) -> bool:  ## --v4
        """尝试重连一次，成功返回 True，失败返回 False。"""
        if self._closed:
            return False
        try:
            self.ser.close()
        except Exception:
            pass

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout, write_timeout=self.write_timeout)
            return True
        except Exception:
            return False

    # def _reconnect_and_stop(self):  ## --v3
    #     """自动重连方法"""
    #     try:
    #         self.ser.close()
    #     except Exception:
    #         pass

    #     self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout, write_timeout=self.write_timeout)

    #     try:
    #         self.send_C(1, 0)
    #     except Exception:
    #         pass  # 重连后发停车也失败就没办法了，让外层抛异常

    def close(self):
        if self._closed:
            return
        try:
            self.stop()  ## 当前 close 发送 C,1,0 软暂停；完整安全停车需另行改为 C,0,0。
        finally:
            self._closed = True
            if self.ser.is_open:
                self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


"""外层逻辑
### 完整逻辑【两套条件失败次数&&持续时间】
# 放在 compute_worker 函数体开头，while 循环外的初始化里：
uart_fail_count = 0
uart_fail_start = None
UART_FAIL_TOLERANCE_S = 3.0   # 3 秒内算"间歇抖动"
UART_FAIL_MAX = 10            # 3 秒内最多容忍 10 次失败

# 放在每帧 UART 下发那里：
if uart is not None:
    result = uart.send_L(line_target.near_offset, line_target.near_yaw)
    if result is None:
        now = time.monotonic()
        if uart_fail_start is None:
            uart_fail_start = now
        uart_fail_count += 1
        # 检查容忍窗口
        if uart_fail_count > UART_FAIL_MAX or (now - uart_fail_start) > UART_FAIL_TOLERANCE_S:
            # 超容忍：停车 + 抛异常
            uart.send_C(1, 0)
            raise RuntimeError(f"UART failed {uart_fail_count} times in {now-uart_fail_start:.1f}s")
    else:
        # 成功：清零计数
        uart_fail_count = 0
        uart_fail_start = None

###简化
##说明，run设计为固定 30fps 下发，所以次数本身就等于时间
uart_fail_count = 0
uart_fail_max = int(args.uart_fps * 3.0)

result = uart.send_L(line_target.near_offset, line_target.near_yaw)
if result is None:
    uart_fail_count += 1
    if uart_fail_count >= uart_fail_max:
        uart.stop()
        raise RuntimeError(f"UART failed for {uart_fail_count} consecutive frames")
else:
    uart_fail_count = 0
"""