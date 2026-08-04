# OrangePi 上位机

本目录是赛后确认的最小上位机发行，只包含三个实际完赛入口。

## 环境

- Python 3.10 或更高版本；
- OrangePi/Rockchip NPU 及与系统匹配的 `rknn-toolkit-lite2` wheel；
- 共享内存相机服务；
- `/dev/ttyUSB0` 或通过参数指定的下位机串口。

```bash
python -m pip install -r requirements.txt
# 再安装与板端 Python、librknnrt 匹配的 Rockchip wheel
```

## 模型和标定

开源发行只保留实际入口需要的文件：

```text
xsmartcar/mymodel/seg_model/inference_model_384_final_int8.rknn
xsmartcar/mymodel/det_model/mbjc_384_int8.rknn
xsmartcar/mymodel/det_model/mbjc_384_int8.rknn.json
xsmartcar/data_npz/race_data.npz
```

分割加载器会选择 `seg_model/` 中第一个 `*.rknn`，因此不要把备份模型直接堆进该目录。

模型大小和 SHA-256 校验值见 [`MODELS.md`](MODELS.md)。

## 实际运行命令

```bash
python run_track_include_fork.py --global-smooth-window 5 --offset-alpha 0.05 --smooth global --debug --base-speed 50 --speed-ref 50

python run_track_include_fork_gai.py --global-smooth-window 5 --offset-alpha 0.05 --smooth global --debug --base-speed 50 --speed-ref 50

python run_track_include_fork_avoid_v3.py --global-smooth-window 5 --offset-alpha 0.05 --smooth global --debug --base-speed 50 --speed-ref 50
```

`run_track_include_fork_gai.py` 的实车记录是 60 cm/s、1:33，但上述 `50/50` 是赛时保存的原始命令参数，两者不应擅自相互替换。

## 千帆 API 配置

仓库配置只含占位符。加载优先级为：

1. `QIANFAN_API_KEY` / `QIANFAN_ACCESS_TOKEN` 环境变量；
2. 被 Git 忽略的 `xsmartcar/qianfan_api/qianfan_api_config.local.yaml`；
3. 仓库中的占位配置。

```bash
cp xsmartcar/qianfan_api/qianfan_api_config.yaml xsmartcar/qianfan_api/qianfan_api_config.local.yaml
# 只在 local.yaml 中填写真实凭据
```

没有有效凭据时不要把 API 失败误判为 NPU、分叉或串口故障。

## 感知 smoke

该脚本只观察相机、分割、检测、IPM/TRK 和 fid 新鲜度，不发送 UART 控制帧：

```bash
python dev_tools/smoke_tests/shm_pipeline_smoke_test.py --mode seg --show
python dev_tools/smoke_tests/shm_pipeline_smoke_test.py --mode track --show
python dev_tools/smoke_tests/shm_pipeline_smoke_test.py --mode det --show
python dev_tools/smoke_tests/shm_pipeline_smoke_test.py --mode fusion --show
python dev_tools/smoke_tests/shm_pipeline_smoke_test.py --mode full --show
```

## PC 检查

```bash
python -m py_compile run_track_include_fork.py run_track_include_fork_gai.py run_track_include_fork_avoid_v3.py
python -m unittest discover -s tests -p "test_*.py" -v
```

PC 没有 RKNNLite 时不能运行真实 NPU 入口；静态编译成功不代表板端、串口或实车通过。

## 未收入内容

- `run_watchdog.py`、停车辅助脚本；
- `gai_recover` 等 AI 半成品；
- 人员避让 v4/v5 和高志禹相关实验；
- 旧 run_pipe、备份入口、旧模型、错误模型和离线探索工具。
