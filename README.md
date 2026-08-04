# XSmartCar 视觉循迹与控制栈

第 21 届全国大学生智能汽车竞赛相关的赛后开源整理版本，包含香橙派视觉上位机和 Infineon TC264 下位机。

本仓库只保留赛后确认真正完成比赛的 3 个上位机入口、它们的运行依赖，以及可重新编译烧录的下位机源码。比赛期间的备份、半成品、人员避让探索、watchdog、旧模型和构建产物均未收入。

## 实车记录

| 入口 | 功能 | 实车记录 | 备注 |
|---|---|---|---|
| `upper/run_track_include_fork.py` | 循迹、分叉 | 50 cm/s，1:52 | API 结果会被后续逻辑覆盖，但已实际完赛 |
| `upper/run_track_include_fork_gai.py` | 循迹、分叉 | 实际状态 60 cm/s，1:33 | API 结果不被覆盖；赛时命令仍使用 `50/50` 参数 |
| `upper/run_track_include_fork_avoid_v3.py` | 循迹、分叉、车辆避让 | 50 cm/s，2:18 | 实际车辆避让完赛 |

以上速度和耗时来自赛后操作者确认。它们不是仿真结果，也不表示换车、换赛道或换标定后仍能复现相同成绩。

## 目录

```text
upper/                     OrangePi 上位机
  run_track_*.py           三个实际完赛入口
  xsmartcar/               感知、IPM、循迹、UART 与 API 模块
  dev_tools/smoke_tests/   唯一保留的实时双模型观察脚本
  tests/                   不依赖 NPU 的检测解码单元测试
lower/                     TC264 下位机完整工程
docs/                      协议、验证边界和发行范围
scripts/                   发布前自动检查
```

上下位机刻意放在不同目录，不能混用工作目录或构建方式。

## 快速开始

### 1. 启动相机服务

相机 WebUI 不在本仓库中，按比赛设备上的安装位置启动：

```bash
cd /home/orangepi/Desktop/setupUI1.0.8K/dist
./setup_webui
```

### 2. 进入上位机目录

```bash
cd /home/orangepi/Desktop/xsmartcar-race-stack/upper
```

### 3. 运行实际完赛入口

```bash
# 50 cm/s，1:52
python run_track_include_fork.py --global-smooth-window 5 --offset-alpha 0.05 --smooth global --debug --base-speed 50 --speed-ref 50

# 实际状态 60 cm/s，1:33；命令参数按赛时手记原样保留
python run_track_include_fork_gai.py --global-smooth-window 5 --offset-alpha 0.05 --smooth global --debug --base-speed 50 --speed-ref 50

# 车辆避让，50 cm/s，2:18
python run_track_include_fork_avoid_v3.py --global-smooth-window 5 --offset-alpha 0.05 --smooth global --debug --base-speed 50 --speed-ref 50
```

更完整的依赖、模型、API 配置和 smoke 说明见 [`upper/README.md`](upper/README.md)。

## 下位机

使用 AURIX Development Studio 导入 `lower/` 下已有工程。新拉取仓库不包含 `Debug/`、ELF 或 HEX；必须先 Build，再烧录新生成的文件。详见 [`lower/README.md`](lower/README.md)。

## 串口协议与停车语义

```text
L,near_offset,near_yaw,det_offset_cm\r\n
C,run_is,speed_limit\r\n
```

- `C,1,speed`：保持本地发车锁存并设置目标速度；
- `C,1,0`：可恢复软暂停，停车但保留发车锁存；
- `C,0,0`：完赛/安全停车并清除锁存，之后必须重新按下位机本地发车键。

下位机只有在本地发车键已经授权后才解析控制帧。这是比赛安全门控，串口不能远程绕过。完整契约见 [`docs/PROTOCOL.md`](docs/PROTOCOL.md)。

## 安全提醒

上位机会访问真实相机、NPU 和 UART。首次运行或改动参数时，应按以下顺序验证：

1. PC 静态编译和单元测试；
2. OrangePi 感知 smoke，不发送 UART；
3. 下位机串口台架；
4. 车辆架空轮验证；
5. 低速空旷场地验证。

本仓库的静态检查不能替代 OrangePi、TC264 和实车测试。

## 版权与许可证

项目自有代码以 **GPL-3.0-or-later** 开源，完整条款见 [`LICENSE`](LICENSE)。逐飞库原有 GPLv3-or-later 声明、Infineon 原有 Boost Software License 声明和所有第三方版权头均保持不变。

版权和贡献边界见 [`COPYRIGHT.md`](COPYRIGHT.md)，第三方内容见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 凭据

仓库只包含 `<apikey>` / `<token>` 占位符。真实凭据应使用环境变量或被忽略的本地配置，禁止提交到 Git。

## 已知边界

- 相机 WebUI、Rockchip RKNN runtime/toolkit wheel、AURIX Development Studio 不随仓库分发；
- RKNN 模型只适用于对应的 Rockchip 运行环境和当前输入配置；
- 本发行不包含人员避让 v4、高志禹相关实验代码；
- 不包含比赛 Git 历史，避免携带旧凭据和无关构建产物。

验证状态见 [`docs/VALIDATION.md`](docs/VALIDATION.md)。
