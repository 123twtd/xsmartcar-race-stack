/* Copyright (c) 2026 清影/123twtd */
#ifndef _KEY_H_
#define _KEY_H_

#include "zf_common_headfile.h"

#define KEY_START_PIN           (P20_7)  // 按键 1：选择/切换键
#define KEY_SPEED_ADD_PIN       (P20_6)  // 按键 2：增加 (+) 键
#define KEY_SPEED_SUB_PIN       (P33_12) // 按键 3：减少 (-) 键
#define KEY_RESET_PIN           (P33_11) // 按键 4：紧急重置键

#define KP1_DEFAULT             (2.46f)
#define KP2_DEFAULT             (0.0f)
#define KP3_DEFAULT             (1.5f)
#define SERVO_KI_DEFAULT        (0.0f)
#define SERVO_KD_DEFAULT        (0.01f)

extern int ui_select_index; // 暴露给外部获取当前选择的菜单项

// 暴露三个新系数给按键逻辑
extern volatile float kp1;
extern volatile float kp2;
extern volatile float kp3;
extern volatile float servo_ki;
extern volatile float servo_kd;

void Key_Init(void);
void Key_Service(void);

#endif
