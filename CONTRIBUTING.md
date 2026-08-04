# 贡献指南

提交改动前请先说明影响范围：上位机感知、IPM/TRK、UART 协议、下位机控制或文档。

最低检查要求：

```bash
python -m py_compile upper/run_track_include_fork.py upper/run_track_include_fork_gai.py upper/run_track_include_fork_avoid_v3.py
python -m unittest discover -s upper/tests -p "test_*.py" -v
python scripts/release_check.py
```

控制策略改动还必须记录：

- 使用的入口和完整命令；
- 模型、标定和硬件版本；
- PC、OrangePi、串口台架、架空轮及实车分别验证到哪一级；
- L/C 字段数量、单位、符号、限幅、锁存和复位语义是否变化。

禁止提交真实 API Key、token、设备口令、Debug/Release 构建产物或无法说明来源的大文件。
