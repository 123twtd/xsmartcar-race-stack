# 第三方内容说明

本文件用于帮助定位第三方内容，不替代对应文件内的完整声明。

## Infineon AURIX iLLD

- 位置：`lower/libraries/infineon_libraries/` 及相关工程文件；
- 权利人：Infineon Technologies AG；
- 许可证：文件头所列 Boost Software License 1.0 或相应原始条款；
- 处理：原版权头和许可证文本未修改。

## SEEKFREE 逐飞 TC264 开源库

- 位置：`lower/libraries/zf_*`；
- 权利人：逐飞科技及文件中列明的贡献者；
- 许可证：文件头声明的 GPL version 3 or later；
- 处理：原版权头和许可证声明未修改。

## Rockchip RKNN

- 仓库包含项目部署使用的 `.rknn` 模型；
- `rknn-toolkit-lite2`、`librknnrt` 和厂商 wheel 不在仓库中；
- 使用者必须自行从合法来源获取与板卡、Python 和 runtime 版本匹配的组件，并遵守其许可条款。

## OpenCV、NumPy、PySerial、Requests、PyYAML、Cloudpickle

这些依赖不以源码形式复制进本仓库，由使用者通过 Python 包管理器安装，并分别遵守各自许可证。

## 百度千帆 API

仓库只包含 HTTP 客户端集成和占位配置，不包含平台 SDK、账户或凭据。API 使用受服务提供方条款约束。

## 相机 WebUI 与开发工具

相机 WebUI、AURIX Development Studio、winIDEA 和相关编译/烧录工具不在本仓库中，其使用受各自供应方条款约束。
