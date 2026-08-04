# RKNN 与相机探针

这些脚本用于定位部署模型和相机共享内存问题，不属于模型训练代码，也不参与三个完赛入口的正常运行。

## 检测模型输出与配置探针

```bash
cd upper
python dev_tools/probes/rknn_det_model_probe.py --model xsmartcar/mymodel/det_model/mbjc_384_int8.rknn
```

脚本读取 RKNN 输入输出元数据、尝试匹配预处理方式，并更新模型同目录 JSON。执行前应备份人工确认过的 JSON；自动结果仍需结合实际类别和输出张量核对。

## 分割模型原始输出探针

```bash
cd upper
python dev_tools/probes/rknn_seg_output_probe.py --image_path path/to/frame.png --output_dir debug_probe
```

输出原始张量、统计摘要、argmax 结果和可用的类别分数图，用于判断模型输出是 logits、概率还是类别图。

## 相机共享内存吞吐探针

```bash
cd upper
python dev_tools/probes/cap_arvideo_press_test.py
```

该脚本只统计共享内存原始读帧速度，不运行 NPU，也不发送 UART。

这些探针需要在具备相机服务或匹配 RKNN runtime 的 OrangePi 上执行，PC 静态编译不能替代板端结果。
