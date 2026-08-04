# 模型与标定文件

本发行只保留三个实际完赛入口会加载的模型。

| 文件 | 大小（字节） | SHA-256 |
|---|---:|---|
| `xsmartcar/mymodel/seg_model/inference_model_384_final_int8.rknn` | 13,790,677 | `1456432cff629599009566758540b8ed41ac17eeaf66042eaf2b32722f0940ac` |
| `xsmartcar/mymodel/det_model/mbjc_384_int8.rknn` | 25,241,827 | `2177bf76cc9b552750c9827c64d788a304bfb2e314dde77c4179d1bb02b29569` |
| `xsmartcar/data_npz/race_data.npz` | 1,160 | `fe64dd810315d867bac3af392dad220ac9b7b5914a4932ad390567c5f5ce5a53` |

检测模型还必须和同目录的 `mbjc_384_int8.rknn.json` 一起使用；加载器会校验模型文件名、类别和输入配置。

`.rknn` 是部署产物，不包含 Rockchip runtime。使用者仍需安装与 OrangePi 系统、Python 和 `librknnrt` 匹配的厂商 wheel。

这些文件只针对比赛设备和当前预处理流程整理。修改输入尺寸、颜色顺序、量化配置、类别或后处理后，必须重新导出并重新验证。
