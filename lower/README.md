# TC264 下位机控制工程

赛后冻结分支：`race-embeded`  
赛后冻结提交：`94dfc6cb2222b2e190e2a3978c1c62e444e2f404`（2026-07-21）

工程基于 Infineon AURIX/TC264 和 SEEKFREE 逐飞 TC264 开源库，完成上位机 UART 指令接收、编码器测速、电机速度 PID、舵机控制、按键发车/调参、TFT 显示、蜂鸣器和电池低压报警。

## 目录

- `code/`：比赛业务模块；
- `user/`：CPU0/CPU1 入口与中断集成；
- `libraries/`：Infineon 与 SEEKFREE 第三方库；
- `.project`、`.cproject`：AURIX Development Studio/Eclipse 工程配置；
- `race_win_v1_3 Debug.launch`：调试/烧录启动配置。

## 构建与烧录

1. 在 AURIX Development Studio 中导入本目录已有工程；
2. 选择 `Debug` 配置并执行 Build Project；
3. 构建系统会重新生成 `Debug/` 及 `race_win_v1_3.elf/.hex`；
4. 构建成功后再使用调试/烧录配置下载到 TC264。

`Debug/`、`Release/`、`.o/.d/.src/.map/.elf/.hex` 等是可再生构建产物，不再纳入 Git。删除这些产物不会移除任何源码或库，但新拉取后必须先构建，才能烧录新生成的 ELF/HEX。

## UART 协议

```text
L,near_offset,near_yaw,det_offset_cm\r\n
C,run_is,speed_limit\r\n
```

- L 帧三个字段分别参与横向误差、航向误差和高级行为偏置；
- `C,1,speed`：保持发车锁存并设置目标速度；
- `C,1,0`：保持发车锁存的可恢复软暂停；
- `C,0,0`：完赛/安全停车并清除运行锁存。

赛后确认的三个实际完赛程序均按比赛后期人工停车方式整理：`C,1,0` 停车但保留锁存，重新启动程序后可及时恢复。该选择是比赛运行策略，不应擅自改成 `C,0,0`。

## 本地发车安全门控

下位机只有在本地 `start_flag == 1` 时才解析上位机控制帧。本地按键承担首次发车授权；`C,1,0` 保留该锁存，`C,0,0` 清除锁存。锁存被清除后必须再次按本地按键，串口不能远程越过该门控发车。该行为是安全保护，不是协议缺陷。

## 版权和第三方许可证

下位机包含清影/123twtd保留的原始模块和参数，以及高志禹重新编写的当前实现；限幅、按键和中断逻辑由清影/123twtd提出并审查，由高志禹实现。详见 [`COPYRIGHT.md`](COPYRIGHT.md)。Infineon 与 SEEKFREE 文件保留原版权头和许可证，未作删改。

## 验证边界

本次赛后整理完成静态工程结构和配置检查；未在本机执行 AURIX 专用工具链编译、TC264 烧录或实车测试。首次使用整理版时应依次做：工程构建、串口台架、架空轮、电机/舵机低速验证。