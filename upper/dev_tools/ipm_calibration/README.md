# IPM race 标定工具

`race_ipm.py` 保留比赛使用的四点标定、物理尺度和坐标映射规则，并生成运行时读取的 `upper/xsmartcar/data_npz/race_data.npz`。

版权归清影/123twtd所有；高志禹参与过比赛调用接口的模块化重构。

## 直接复现当前比赛标定

```bash
cd upper
python dev_tools/ipm_calibration/race_ipm.py
```

默认使用代码中的比赛四点坐标，重新计算矩阵、更新 `race_data.npz`，并在 `dev_tools/ipm_calibration/output/` 保存预览结果。

如只想检查生成链、不覆盖运行时标定，可把 NPZ 写入工具的忽略目录：

```bash
cd upper
python dev_tools/ipm_calibration/race_ipm.py --no-show --output-npz dev_tools/ipm_calibration/output/race_data.test.npz
```

## 重新手动选四点

```bash
cd upper
python dev_tools/ipm_calibration/race_ipm.py --interactive
```

按“左远、右远、左近、右近”的顺序点击。确认控制台打印的点坐标和预览结果后，再决定是否把新生成的 `race_data.npz` 用于车辆。

物理尺度由脚本中的 `W_CM`、`D_CM`、`NEAR_CM` 和 `BOARD` 定义。修改任一值都会改变图像像素与物理世界坐标之间的关系，必须重新核对循迹、检测偏置和障碍物距离语义。

当前 `BOARD=(-7.5, 7.5, 5.0, 20.0)` 与完赛 `race_data.npz` 保存的 `dst_points` 一致；使用默认四点重算出的矩阵与仓库内完赛矩阵完全一致。

`assets/calibration_track.png` 是四点选取原图，`assets/mask_input.png` 用于本地透视变换预览。工具只负责标定和检查，不发送 UART。
