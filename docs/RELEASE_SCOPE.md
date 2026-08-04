# 开源发行范围

## 必须保留

- 三个实际完赛上位机入口；
- 它们直接依赖的 `xsmartcar` 模块；
- `inference_model_384_final_int8.rknn`；
- `mbjc_384_int8.rknn` 及对应 JSON；
- `race_data.npz`；
- IPM race 四点标定脚本及必要输入图；
- RKNN 检测/分割输出探针和相机共享内存探针；
- 统一实时感知 smoke；
- 当前检测解码单元测试；
- 下位机 `code/`、`user/`、`libraries/` 和 ADS/Eclipse 工程配置；
- 协议、版权、第三方、构建和验证文档。

## 明确排除

- 未完成的人员避让 v4/v5 实验；
- 尚在整理中的训练数据、训练脚本和完整模型训练流程；
- `gai_recover`、watchdog、停车辅助和其他半成品；
- 旧 run_pipe、缺失依赖的重构链、备份和日期副本；
- 分割/检测备份模型、错误模型和未使用模型；
- Debug/Release、ELF、HEX、MAP、对象文件和 Python 缓存；
- API Key、token、本地配置与原比赛 Git 历史；
- 相机 WebUI、RKNN 厂商 wheel、ADS、winIDEA 和烧录工具。

## 发行原则

本仓库以“能够说明实际完赛程序并可重新构建”为目标，不以保存比赛期间所有探索过程为目标。历史研究材料继续保留在赛后归档仓库，不进入公开发行。
