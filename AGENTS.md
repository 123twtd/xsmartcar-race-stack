# AGENTS 指南

本文件面向在本仓库中执行任务的自动化/智能体，目标是在**不破坏实车可复现性**的前提下完成最小改动。

## 1. 仓库结构与边界

- `upper/`：OrangePi 上位机（Python、RKNN、共享内存相机、UART）。
- `lower/`：TC264 下位机工程（AURIX Development Studio）。
- `docs/`：协议、验证边界、发布说明。
- `scripts/`：发布前检查脚本。

关键边界：

- 上下位机目录独立，**不要混用构建方式或运行目录**。
- 仅在任务明确要求时修改 `lower/`；默认优先处理 `upper/` 和文档。

## 2. 修改原则

- 只做与任务直接相关的最小修改，不顺手重构无关代码。
- 保持既有协议语义，尤其是控制帧：
  - `C,1,speed`：运行；
  - `C,1,0`：软暂停（保留发车锁存）；
  - `C,0,0`：清除锁存并停车。
- 不要在仓库中加入真实凭据（API Key、Token、设备口令）。

## 3. 本地最小验证（提交前）

在仓库根目录执行：

```bash
python -m py_compile upper/run_track_include_fork.py upper/run_track_include_fork_gai.py upper/run_track_include_fork_avoid_v3.py
python -m unittest discover -s upper/tests -p "test_*.py" -v
python scripts/release_check.py
```

补充说明：

- 若仅改文档且不影响代码逻辑，可说明未运行硬件相关验证。
- PC 环境无法替代 OrangePi+NPU+实车验证，避免夸大验证结论。

## 4. 硬件与安全约束

- 下位机需要本地发车按键授权；串口不能绕过本地门控。
- 涉及速度、转向、限幅、锁存/复位逻辑的改动必须谨慎，且在变更说明中明确影响。

## 5. 提交说明建议

提交或 PR 描述至少覆盖：

- 影响范围（上位机感知/IPM/TRK/UART/下位机/文档）；
- 运行或测试命令；
- 未验证项与原因（尤其硬件实测缺失时）。
