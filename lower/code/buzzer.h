/* Copyright (c) 2026 清影/123twtd */
/* Copyright (c) 2026 高志禹 */
/*
 * buzzer.h
 *
 *  Created on: 2026年6月8日
 *      Author: A
 */

#ifndef _BUZZER_H_
#define _BUZZER_H_

#include "zf_common_headfile.h"

// ==========================================
// 硬件引脚配置 (根据你的实际接线修改)
// ==========================================
#define BUZZER_PIN      (P33_10)

// ==========================================
// 发声模式宏定义
// ==========================================
#define BEEP_OFF        0   // 关闭
#define BEEP_LONG       1   // 长鸣一声 (发车/停车警告)
#define BEEP_DOUBLE     2   // 短促双声 (滴滴，确认收到)
#define BEEP_TRIPLE     3   // 连续三声 (紧急重置报警)
#define BEEP_LOW_VOLT   4   // 低压报警，持续 10 秒

// ==========================================
// 对外提供的函数接口
// ==========================================
void Buzzer_Init(void);
void Buzzer_Service(void);
void Beep_Play(int pattern);

#endif
