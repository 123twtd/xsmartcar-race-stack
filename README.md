# XSmartCar 视觉循迹与控制栈

第 21 届全国大学生智能汽车竞赛相关的赛后开源整理版本，包含香橙派视觉上位机和 Infineon TC264 下位机。

本项目整体成果归西北师范大学瞬之队共有，开源内容主要为瞬之队后续成员和其他智能车初学者提供参考。

代码以比赛期间快速迭代和完成任务为优先，结构、命名与部分实现仍较粗糙，仅供学习和复现思路参考，不应视为可直接用于其他车辆的成熟控制系统。

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
  dev_tools/               IPM 标定、RKNN/相机探针与实时 smoke
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

## 使用前说明

三个入口均为实际完赛版本。更换相机、OrangePi 系统、RKNN runtime、TC264 工程、模型或 IPM 标定后，建议先低速确认感知结果、转向方向和串口通信，再调整速度参数。仓库 CI 只覆盖静态检查和不依赖 NPU 的单元测试。

## 版权与许可证

项目自有代码以 **GPL-3.0-or-later** 开源，完整条款见 [`LICENSE`](LICENSE)。逐飞库原有 GPLv3-or-later 声明、Infineon 原有 Boost Software License 声明和所有第三方版权头均保持不变。

版权和贡献边界见 [`COPYRIGHT.md`](COPYRIGHT.md)，第三方内容见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 凭据

仓库只包含 `<apikey>` / `<token>` 占位符。真实凭据应使用环境变量或被忽略的本地配置，禁止提交到 Git。

## 模型与运行环境

- 相机 WebUI、Rockchip RKNN runtime/toolkit wheel、AURIX Development Studio 不随仓库分发；
- RKNN 模型只适用于对应的 Rockchip 运行环境和当前输入配置；
- 仓库包含完赛部署模型、RKNN 输出探针和 IPM race 标定工具；
- 训练数据、训练脚本和完整模型训练流程正在单独整理，当前暂未公开。

验证状态见 [`docs/VALIDATION.md`](docs/VALIDATION.md)。
